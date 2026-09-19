import { WebGPUEngine } from './webgpu-engine';
import { CPUEngine } from './cpu-engine';
import { WebGPUContextManager } from './context-manager';
import { packUnicodeToGPUBuffer, checkMemoryBudget } from './buffer';
import { normalizeText } from './unicode-preprocess';
import {
  searchMultiFieldCpuReference,
  type FieldScoreDefinition
} from './cpu-reference';
import {
  clampLimit,
  throwIfAborted,
  nowMs
} from './runtime-guards';
import {
  DOC_FORMAT_VERSION,
  QUERY_TOKENS_MAX,
  SCORING_VERSION,
  UNICODE_VERSION,
  DuplicateIdError,
  IncompatibleOptionError,
  ProfileMismatchError,
  QueryTooLongError,
  type TextProfileId
} from './text-profile';
import { alignHighlights, renderHighlightedText } from './highlight';
import type {
  AddOptions,
  DocumentId,
  DocumentIndexOptions,
  DocumentIndexStats,
  DocumentSearchOptions,
  DocumentSearchResponse,
  DocumentSearchResultItem,
  EngineType,
  FallbackReason,
  HighlightRange,
  MutationBatch,
  MutationResult,
  SearchMode,
  SearchTimings
} from './types';

export interface InternalField<TDoc> extends FieldScoreDefinition {
  name: string;
  weight: number;
  getter: (doc: TDoc) => string | string[] | undefined | null;
  originalIndex: number;
}

/**
 * High-level multi-field document search index with dynamic mutations and highlighting.
 * Full implementation lands across M2 (record engine), M3 (highlighting), and M4 (mutations).
 */
export class DocumentIndex<TDoc = Record<string, unknown>> {
  readonly options: DocumentIndexOptions<TDoc>;
  private isDestroyed: boolean = false;
  private engineType: EngineType = 'cpu';
  private gpuEngine: WebGPUEngine | null = null;
  private vramAllocatedBytes: number = 0;
  private unsubscribeDeviceLost?: () => void;
  private readonly profileId: TextProfileId = 'unicode-default';
  private readonly folded: boolean;
  private readonly preferGpu: boolean;
  private fallbackReason?: FallbackReason;

  // Documents and IDs
  private records: TDoc[] = [];
  private docIds: DocumentId[] = [];
  private idToDocIndex: Map<DocumentId, number> = new Map();
  private readonly getId: (doc: TDoc) => DocumentId;

  // Fields and weights (stratified: sorted by weight desc)
  private readonly sortedFields: InternalField<TDoc>[];
  private readonly fieldNameToIndex: Map<string, number> = new Map();

  // Tokens and row mapping
  private rowTokens: Uint32Array[] = [];
  private rowToDocIndex: number[] = [];
  private rowToFieldIndex: number[] = [];
  private rawFieldStrings: string[][] = []; // [docIndex][fieldIndex]
  private totalTokens: number = 0;
  private candidateCapacity: number = 8192;
  private buildTimeMs: number = 0;
  private mutationEpoch: number = 0;
  private tombstones: Set<number> = new Set(); // M4 tombstone candidate filtering

  constructor(options: DocumentIndexOptions<TDoc>) {
    if (!options || !Array.isArray(options.fields) || options.fields.length === 0) {
      throw new TypeError('[webgpu-search] DocumentIndex expects options.fields to be a non-empty array.');
    }
    this.options = options;
    this.folded = !(options.caseSensitive ?? false);
    this.preferGpu = options.preferGpu ?? false;

    if (typeof options.idField === 'function') {
      this.getId = options.idField;
    } else if (typeof options.idField === 'string') {
      const prop = options.idField;
      this.getId = (doc: any) => doc[prop];
    } else if (options.idField === undefined) {
      this.getId = (doc: any) => doc.id;
    } else {
      throw new TypeError('[webgpu-search] idField must be a string property name or function.');
    }

    const seenNames = new Set<string>();
    const normalized: InternalField<TDoc>[] = options.fields.map((f, originalIndex) => {
      let name: string;
      let weight = 1.0;
      let getter: (doc: TDoc) => string | string[] | undefined | null;

      if (typeof f === 'string') {
        if (f.trim().length === 0) {
          throw new TypeError('[webgpu-search] Field name string must not be empty.');
        }
        name = f;
        getter = (doc: any) => doc[name];
      } else if (f && typeof f === 'object') {
        if (typeof f.name !== 'string' || f.name.length === 0) {
          throw new TypeError('[webgpu-search] Field definition requires a non-empty string name.');
        }
        name = f.name;
        if (f.weight !== undefined) {
          if (typeof f.weight !== 'number' || !Number.isFinite(f.weight) || f.weight <= 0) {
            throw new RangeError(
              `[webgpu-search] Field "${name}" weight must be a positive finite number, got ${f.weight}.`
            );
          }
          weight = f.weight;
        }
        if (f.getter !== undefined) {
          if (typeof f.getter !== 'function') {
            throw new TypeError(`[webgpu-search] Field "${name}" getter must be a function.`);
          }
          getter = f.getter;
        } else {
          getter = (doc: any) => doc[name];
        }
      } else {
        throw new TypeError('[webgpu-search] Field must be a string or FieldDefinition object.');
      }

      if (seenNames.has(name)) {
        throw new Error(`[webgpu-search] Duplicate field name: "${name}".`);
      }
      seenNames.add(name);

      return { name, weight, getter, originalIndex };
    });

    // Field-stratified packing: sort fields descending by weight
    this.sortedFields = [...normalized].sort((a, b) => {
      if (b.weight !== a.weight) return b.weight - a.weight;
      return a.originalIndex - b.originalIndex;
    });

    for (let i = 0; i < this.sortedFields.length; i++) {
      this.fieldNameToIndex.set(this.sortedFields[i].name, i);
    }
  }

  static async create<TDoc = Record<string, unknown>>(
    records: TDoc[],
    options: DocumentIndexOptions<TDoc>
  ): Promise<DocumentIndex<TDoc>> {
    if (!Array.isArray(records)) {
      throw new TypeError('[webgpu-search] DocumentIndex.create expects records: TDoc[].');
    }
    if (options.slotBytes !== undefined) {
      throw new IncompatibleOptionError(
        'slotBytes',
        '[webgpu-search] slotBytes throw-on-use in v0.2 (fixed slots removed; removal in v0.3).'
      );
    }
    if (options.textProfile !== undefined && options.textProfile !== 'unicode-default') {
      throw new ProfileMismatchError('unicode-default', options.textProfile, 'textProfile');
    }

    const index = new DocumentIndex<TDoc>(options);
    await index.initialize(records);
    return index;
  }

  private async initialize(records: TDoc[]): Promise<void> {
    const t0 = nowMs();
    const docCount = records.length;
    const fieldCount = this.sortedFields.length;
    const totalRows = docCount * fieldCount;

    this.records = records.slice();
    this.docIds = new Array(docCount);
    this.idToDocIndex = new Map<DocumentId, number>();

    for (let i = 0; i < docCount; i++) {
      const doc = this.records[i];
      if (doc === null || typeof doc !== 'object') {
        throw new TypeError(`[webgpu-search] Document at index ${i} must be an object.`);
      }
      const id = this.getId(doc);
      if (
        id === null ||
        id === undefined ||
        (typeof id !== 'string' && typeof id !== 'number') ||
        (typeof id === 'number' && !Number.isFinite(id))
      ) {
        throw new TypeError(
          `[webgpu-search] Document at index ${i} has invalid id: ${String(id)}. DocumentId must be a non-empty string or finite number.`
        );
      }
      if (typeof id === 'string' && id.length === 0) {
        throw new TypeError(
          `[webgpu-search] Document at index ${i} has an empty string id. DocumentId must not be empty.`
        );
      }
      if (this.idToDocIndex.has(id)) {
        throw new DuplicateIdError(id);
      }
      this.docIds[i] = id;
      this.idToDocIndex.set(id, i);
    }

    // Dynamic Candidate Pool Scaling (Section 3.1):
    // candidateCapacity = min(32768, max(8192, docCount * fieldCount * 0.1))
    const rawCap = this.options.candidateCapacity;
    const validUserCap = typeof rawCap === 'number' && Number.isFinite(rawCap) && rawCap > 0
      ? Math.floor(rawCap)
      : Math.floor(docCount * fieldCount * 0.1);
    this.candidateCapacity = Math.min(32768, Math.max(8192, validUserCap));

    // Field-Stratified Row Packing:
    // Rows 0..N-1: Primary fields (highest weight).
    // Rows N..2N-1: Secondary fields.
    // Rows 2N..3N-1: Lower-weight fields.
    this.rowTokens = new Array(totalRows);
    this.rowToDocIndex = new Array(totalRows);
    this.rowToFieldIndex = new Array(totalRows);
    this.rawFieldStrings = Array.from({ length: docCount }, () => new Array(fieldCount));

    let corpusTokens = 0;
    let rowIdx = 0;

    for (let f = 0; f < fieldCount; f++) {
      const field = this.sortedFields[f];
      for (let d = 0; d < docCount; d++) {
        const doc = this.records[d];
        const rawVal = field.getter(doc);
        let rawStr = '';
        if (rawVal !== null && rawVal !== undefined) {
          if (Array.isArray(rawVal)) {
            rawStr = rawVal.filter((x) => x !== null && x !== undefined).join(' ');
          } else {
            rawStr = typeof rawVal === 'string' ? rawVal : String(rawVal);
          }
        }
        this.rawFieldStrings[d][f] = rawStr;
        const norm = normalizeText(rawStr, this.folded);
        this.rowTokens[rowIdx] = norm.tokens;
        this.rowToDocIndex[rowIdx] = d;
        this.rowToFieldIndex[rowIdx] = f;
        corpusTokens += norm.tokenCount;
        rowIdx++;
      }
    }

    this.totalTokens = corpusTokens;

    if (docCount === 0) {
      this.engineType = 'cpu';
      this.buildTimeMs = nowMs() - t0;
      return;
    }

    const threshold = this.options.threshold ?? 30_000;
    const shouldAttemptGpu = this.options.preferGpu === false
      ? false
      : this.preferGpu || totalRows >= threshold;

    if (this.options.preferGpu === false) {
      this.fallbackReason = 'prefer-cpu';
    } else if (!this.preferGpu && totalRows < threshold) {
      this.fallbackReason = 'below-threshold';
    }

    if (shouldAttemptGpu) {
      const measuredAvgBytes = totalRows > 0 ? (corpusTokens * 4) / totalRows : 0;
      const budget = checkMemoryBudget(totalRows, measuredAvgBytes, this.options.device, this.candidateCapacity);
      if (!budget.allowed) {
        console.warn(`[webgpu-search] ${budget.reason} Falling back to CPU.`);
        this.engineType = 'cpu';
        this.fallbackReason = 'memory-budget-exceeded';
      } else {
        let gpu: WebGPUEngine | null = null;
        try {
          gpu = new WebGPUEngine();
          const initialized = await gpu.init(this.options.device);

          if (initialized && gpu.isReady) {
            gpu.ensureCandidateCapacity(this.candidateCapacity);
            const packed = packUnicodeToGPUBuffer(this.rowTokens, {
              folded: this.folded,
              totalTokens: corpusTokens
            });
            await gpu.loadDataset(packed);

            this.gpuEngine = gpu;
            this.engineType = 'webgpu';
            this.fallbackReason = undefined;
            this.vramAllocatedBytes = packed.recordsByteLength + packed.offsetsByteLength;
            gpu = null; // Ownership transferred

            this.unsubscribeDeviceLost = WebGPUContextManager.onDeviceLost(() => {
              console.warn('[webgpu-search] GPU device lost, falling back to CPU.');
              if (this.gpuEngine) {
                this.gpuEngine.destroy();
                this.gpuEngine = null;
              }
              this.engineType = 'cpu';
              this.fallbackReason = 'device-lost';
            });
          } else {
            this.engineType = 'cpu';
            this.fallbackReason = 'device-request-failed';
          }
        } catch (gpuErr) {
          console.warn('[webgpu-search] WebGPU initialization failed, falling back to CPU:', gpuErr);
          this.engineType = 'cpu';
          this.fallbackReason = 'webgpu-unsupported';
        } finally {
          if (gpu) {
            gpu.destroy();
          }
        }
      }
    } else {
      this.engineType = 'cpu';
    }

    this.buildTimeMs = nowMs() - t0;
  }

  async search(
    query: string,
    options: DocumentSearchOptions<TDoc> = {}
  ): Promise<DocumentSearchResponse<TDoc>> {
    if (this.isDestroyed) {
      throw new Error('[webgpu-search] DocumentIndex has been destroyed.');
    }
    throwIfAborted(options.signal);
    if (typeof query !== 'string') {
      throw new TypeError(`[webgpu-search] search expects query: string, got ${typeof query}.`);
    }

    const {
      mode = 'fuzzy',
      caseSensitive = false,
      signal,
      cpuAlgorithm = 'parity',
      onQueryTooLong = 'throw',
      fields: searchFields,
      filter
    } = options;

    if (caseSensitive === this.folded) {
      throw new ProfileMismatchError(!this.folded, caseSensitive);
    }
    if (this.preferGpu && cpuAlgorithm === 'ufuzzy') {
      throw new IncompatibleOptionError(
        'cpuAlgorithm',
        "cpuAlgorithm:'ufuzzy' is CPU-only; use preferGpu:false or cpuAlgorithm:'parity'."
      );
    }

    const rawTrimmed = query.trim();
    let forceCpu = false;
    if (rawTrimmed.length > QUERY_TOKENS_MAX * 4) {
      if (rawTrimmed.length > 1_000_000) {
        let cpCount = 0;
        for (const _ch of rawTrimmed) cpCount++;
        const est = cpCount * 3;
        if (est > QUERY_TOKENS_MAX) {
          if (onQueryTooLong !== 'cpu-fallback') {
            throw new QueryTooLongError(QUERY_TOKENS_MAX, est, this.profileId);
          }
          forceCpu = true;
        }
      }
    }

    const normalizedQuery = normalizeText(query, this.folded);
    const queryTokenCount = normalizedQuery.tokenCount;
    if (queryTokenCount > QUERY_TOKENS_MAX) {
      if (onQueryTooLong === 'cpu-fallback') {
        forceCpu = true;
      } else {
        throw new QueryTooLongError(QUERY_TOKENS_MAX, queryTokenCount, this.profileId);
      }
    }

    const clampedLimit = clampLimit(options.limit ?? options.maxResults ?? 50);
    throwIfAborted(signal);

    const noHits = (q: string): DocumentSearchResponse<TDoc> => ({
      query: q,
      mode,
      engine: 'cpu',
      totalMatches: 0,
      candidateCount: 0,
      hasOverflow: false,
      results: [],
      timings: {
        queryUploadMs: 0,
        encodeSubmitMs: 0,
        gpuExecutionMs: null,
        readbackMs: 0,
        totalMs: 0,
        gpuDispatchMs: 0
      },
      profileId: this.profileId,
      scoringVersion: SCORING_VERSION,
      cpuAlgorithm,
      fallbackReason: forceCpu ? 'query-too-long' : this.fallbackReason
    });

    if (normalizedQuery.isEmpty) {
      return noHits('');
    }
    if (this.records.length === 0) {
      return noHits(query);
    }

    if (filter !== undefined && typeof filter !== 'function') {
      throw new TypeError('[webgpu-search] options.filter must be a function.');
    }

    let allowedFieldIndices: Set<number> | undefined = undefined;
    if (searchFields !== undefined) {
      if (!Array.isArray(searchFields)) {
        throw new TypeError('[webgpu-search] options.fields must be an array of string field names.');
      }
      if (searchFields.length === 0) {
        return noHits(query);
      }
      allowedFieldIndices = new Set<number>();
      for (const fName of searchFields) {
        if (typeof fName !== 'string') {
          throw new TypeError('[webgpu-search] options.fields elements must be strings.');
        }
        const idx = this.fieldNameToIndex.get(fName);
        if (idx === undefined) {
          throw new Error(`[webgpu-search] Unknown search field: "${fName}".`);
        }
        allowedFieldIndices.add(idx);
      }
    }

    // When field-restricted, route to CPU to prevent unselected high-priority
    // fields from saturating GPU candidate buffer before low-priority allowed fields.
    const isFieldRestricted = allowedFieldIndices !== undefined && allowedFieldIndices.size < this.sortedFields.length;

    // 1. WebGPU execution path
    const gpuHandle = this.gpuEngine;
    const useGpu = !forceCpu &&
      !isFieldRestricted &&
      cpuAlgorithm !== 'ufuzzy' &&
      this.engineType === 'webgpu' &&
      gpuHandle !== null &&
      gpuHandle.isReady;

    if (useGpu && gpuHandle !== null) {
      try {
        const gpuResult = await gpuHandle.search(query, {
          ...options,
          mode,
          limit: this.candidateCapacity,
          caseSensitive,
          signal
        });

        if (this.isDestroyed) {
          throw new Error('[webgpu-search] DocumentIndex has been destroyed.');
        }
        throwIfAborted(signal);

        // WebGPU candidate readback & fixed-point score enrichment
        const docMatches = new Map<number, {
          bestScore: number;
          bestFieldIdx: number;
          fieldScores: Map<number, number>;
        }>();

        for (let i = 0; i < gpuResult.results.length; i++) {
          const item = gpuResult.results[i];
          const r = item.index;
          if (this.tombstones.has(r)) continue;
          const fIdx = this.rowToFieldIndex[r];
          if (allowedFieldIndices && !allowedFieldIndices.has(fIdx)) continue;
          const dIdx = this.rowToDocIndex[r];
          if (filter && !filter(this.records[dIdx])) continue;

          const rawScore = item.score;
          const fDef = this.sortedFields[fIdx];
          const weightedScore = Math.round(rawScore * fDef.weight);

          let entry = docMatches.get(dIdx);
          if (!entry) {
            entry = {
              bestScore: weightedScore,
              bestFieldIdx: fIdx,
              fieldScores: new Map<number, number>()
            };
            entry.fieldScores.set(fIdx, weightedScore);
            docMatches.set(dIdx, entry);
          } else {
            entry.fieldScores.set(fIdx, weightedScore);
            if (
              weightedScore > entry.bestScore ||
              (weightedScore === entry.bestScore && fIdx < entry.bestFieldIdx)
            ) {
              entry.bestScore = weightedScore;
              entry.bestFieldIdx = fIdx;
            }
          }
        }

        interface RankedCandidate<TDoc> {
          dIdx: number;
          item: DocumentSearchResultItem<TDoc>;
        }

        const hits: RankedCandidate<TDoc>[] = [];
        for (const [dIdx, entry] of docMatches.entries()) {
          const primaryField = this.sortedFields[entry.bestFieldIdx];
          const auxMatches: Array<{ field: string; score: number }> = [];
          for (const [fIdx, score] of entry.fieldScores.entries()) {
            if (fIdx !== entry.bestFieldIdx) {
              auxMatches.push({ field: this.sortedFields[fIdx].name, score });
            }
          }
          if (auxMatches.length > 1) {
            auxMatches.sort((a, b) => {
              if (b.score !== a.score) return b.score - a.score;
              return a.field < b.field ? -1 : (a.field > b.field ? 1 : 0);
            });
          }

          hits.push({
            dIdx,
            item: {
              id: this.docIds[dIdx],
              doc: this.records[dIdx],
              score: entry.bestScore,
              matchedField: primaryField.name,
              matches: auxMatches.length > 0 ? auxMatches : undefined
            }
          });
        }

        // Two-key sort: score descending, docIndex ascending
        hits.sort((a, b) => {
          if (b.item.score !== a.item.score) return b.item.score > a.item.score ? 1 : -1;
          if (a.dIdx !== b.dIdx) return a.dIdx > b.dIdx ? 1 : -1;
          return 0;
        });

        const totalMatches = hits.length;
        const candidateCount = Math.min(totalMatches, this.candidateCapacity);
        const results = hits.slice(0, clampedLimit).map((h) => h.item);
        this.enrichHighlights(results, query, mode, options);

        return {
          query: gpuResult.query,
          mode: gpuResult.mode,
          engine: 'webgpu',
          totalMatches,
          candidateCount,
          hasOverflow: gpuResult.hasOverflow,
          results,
          timings: gpuResult.timings,
          profileId: this.profileId,
          scoringVersion: SCORING_VERSION,
          cpuAlgorithm
        };
      } catch (err: any) {
        if (err.name === 'AbortError') {
          throw err;
        }
        console.warn('[webgpu-search] GPU document search failed, falling back to CPU:', err);
      }
    }

    // 2. CPU execution path
    throwIfAborted(signal);
    const t0 = nowMs();

    if (cpuAlgorithm === 'ufuzzy') {
      const cpuEngine = new CPUEngine();
      const docMatches = new Map<number, {
        bestScore: number;
        bestFieldIdx: number;
        fieldScores: Map<number, number>;
      }>();

      for (let fIdx = 0; fIdx < this.sortedFields.length; fIdx++) {
        if (allowedFieldIndices && !allowedFieldIndices.has(fIdx)) continue;
        const fDef = this.sortedFields[fIdx];
        const fieldStrings = this.records.map((_, dIdx) => this.rawFieldStrings[dIdx][fIdx]);
        const legacyResult = mode === 'fuzzy'
          ? cpuEngine.searchUFuzzy(fieldStrings, query, this.records.length, caseSensitive)
          : cpuEngine.searchNative(fieldStrings, query, this.records.length, caseSensitive);

        for (const hit of legacyResult.results) {
          const dIdx = hit.index;
          if (filter && !filter(this.records[dIdx])) continue;
          const weightedScore = Math.round(hit.score * fDef.weight);
          let entry = docMatches.get(dIdx);
          if (!entry) {
            entry = {
              bestScore: weightedScore,
              bestFieldIdx: fIdx,
              fieldScores: new Map<number, number>()
            };
            entry.fieldScores.set(fIdx, weightedScore);
            docMatches.set(dIdx, entry);
          } else {
            entry.fieldScores.set(fIdx, weightedScore);
            if (
              weightedScore > entry.bestScore ||
              (weightedScore === entry.bestScore && fIdx < entry.bestFieldIdx)
            ) {
              entry.bestScore = weightedScore;
              entry.bestFieldIdx = fIdx;
            }
          }
        }
      }

      interface RankedLegacyCandidate<TDoc> {
        dIdx: number;
        item: DocumentSearchResultItem<TDoc>;
      }

      const hits: RankedLegacyCandidate<TDoc>[] = [];
      for (const [dIdx, entry] of docMatches.entries()) {
        const primaryField = this.sortedFields[entry.bestFieldIdx];
        const auxMatches: Array<{ field: string; score: number }> = [];
        for (const [fIdx, score] of entry.fieldScores.entries()) {
          if (fIdx !== entry.bestFieldIdx) {
            auxMatches.push({ field: this.sortedFields[fIdx].name, score });
          }
        }
        if (auxMatches.length > 1) {
          auxMatches.sort((a, b) => {
            if (b.score !== a.score) return b.score - a.score;
            return a.field < b.field ? -1 : (a.field > b.field ? 1 : 0);
          });
        }
        hits.push({
          dIdx,
          item: {
            id: this.docIds[dIdx],
            doc: this.records[dIdx],
            score: entry.bestScore,
            matchedField: primaryField.name,
            matches: auxMatches.length > 0 ? auxMatches : undefined
          }
        });
      }

      hits.sort((a, b) => {
        if (b.item.score !== a.item.score) return b.item.score > a.item.score ? 1 : -1;
        if (a.dIdx !== b.dIdx) return a.dIdx > b.dIdx ? 1 : -1;
        return 0;
      });

      const totalMatches = hits.length;
      const candidateCount = Math.min(totalMatches, this.candidateCapacity);
      const results = hits.slice(0, clampedLimit).map((h) => h.item);
      this.enrichHighlights(results, query, mode, options);
      const durationMs = nowMs() - t0;

      return {
        query,
        mode,
        engine: 'cpu',
        totalMatches,
        candidateCount,
        hasOverflow: totalMatches > this.candidateCapacity,
        results,
        timings: {
          queryUploadMs: 0,
          encodeSubmitMs: 0,
          gpuExecutionMs: null,
          readbackMs: 0,
          totalMs: durationMs,
          gpuDispatchMs: 0
        },
        profileId: this.profileId,
        scoringVersion: SCORING_VERSION,
        cpuAlgorithm,
        fallbackReason: 'cpu-algorithm-requested'
      };
    }

    // Default parity CPU algorithm
    const parityResult = searchMultiFieldCpuReference(
      this.records.length,
      this.sortedFields,
      this.rowTokens,
      this.rowToDocIndex,
      this.rowToFieldIndex,
      normalizedQuery.tokens,
      mode,
      clampedLimit,
      this.candidateCapacity,
      allowedFieldIndices,
      this.tombstones,
      filter ? (dIdx) => filter(this.records[dIdx]) : undefined
    );

    throwIfAborted(signal);

    const enrichedResults: DocumentSearchResultItem<TDoc>[] = parityResult.results.map((hit) => ({
      id: this.docIds[hit.docIndex],
      doc: this.records[hit.docIndex],
      score: hit.score,
      matchedField: hit.matchedField,
      matches: hit.matches
    }));
    this.enrichHighlights(enrichedResults, query, mode, options);

    const timings: SearchTimings = {
      queryUploadMs: 0,
      encodeSubmitMs: 0,
      gpuExecutionMs: null,
      readbackMs: 0,
      totalMs: parityResult.durationMs,
      gpuDispatchMs: 0
    };

    let effectiveFallbackReason = this.fallbackReason;
    if (forceCpu) {
      effectiveFallbackReason = 'query-too-long';
    } else if (useGpu && gpuHandle !== null) {
      effectiveFallbackReason = 'gpu-execution-error';
    }

    return {
      query,
      mode,
      engine: 'cpu',
      totalMatches: parityResult.totalMatches,
      candidateCount: parityResult.candidateCount,
      hasOverflow: parityResult.hasOverflow,
      results: enrichedResults,
      timings,
      profileId: this.profileId,
      scoringVersion: SCORING_VERSION,
      cpuAlgorithm,
      fallbackReason: effectiveFallbackReason
    };
  }

  private enrichHighlights(
    results: DocumentSearchResultItem<TDoc>[],
    query: string,
    mode: SearchMode,
    options: DocumentSearchOptions<TDoc>
  ): void {
    const shouldHighlight = options.highlightOptions?.highlight ?? options.highlight ?? true;
    if (!shouldHighlight) return;

    const tag = options.highlightOptions?.tag ?? options.tag;
    const highlightFieldsRule = options.highlightOptions?.fields ?? options.highlightFields ?? 'all-matched';

    for (let i = 0; i < results.length; i++) {
      const item = results[i];
      const dIdx = this.idToDocIndex.get(item.id);
      if (dIdx === undefined) continue;

      const highlights: Record<string, HighlightRange[]> = {};
      const highlightedText: Record<string, string> = {};

      if (highlightFieldsRule === 'all-fields') {
        for (let f = 0; f < this.sortedFields.length; f++) {
          const fieldDef = this.sortedFields[f];
          const rawStr = this.rawFieldStrings[dIdx][f] ?? '';
          const ranges = alignHighlights(rawStr, query, { mode, folded: this.folded });
          if (ranges.length > 0) {
            highlights[fieldDef.name] = ranges;
            if (tag) {
              highlightedText[fieldDef.name] = renderHighlightedText(rawStr, ranges, tag);
            }
          }
        }
      } else if (Array.isArray(highlightFieldsRule)) {
        for (let k = 0; k < highlightFieldsRule.length; k++) {
          const fName = highlightFieldsRule[k];
          const fIdx = this.fieldNameToIndex.get(fName);
          if (fIdx !== undefined) {
            const rawStr = this.rawFieldStrings[dIdx][fIdx] ?? '';
            const ranges = alignHighlights(rawStr, query, { mode, folded: this.folded });
            if (ranges.length > 0) {
              highlights[fName] = ranges;
              if (tag) {
                highlightedText[fName] = renderHighlightedText(rawStr, ranges, tag);
              }
            }
          }
        }
      } else {
        // 'matched-field' or 'all-matched' (default)
        const primaryFIdx = this.fieldNameToIndex.get(item.matchedField);
        if (primaryFIdx !== undefined) {
          const primaryRaw = this.rawFieldStrings[dIdx][primaryFIdx] ?? '';
          const ranges = alignHighlights(primaryRaw, query, { mode, folded: this.folded });
          highlights[item.matchedField] = ranges;
          if (tag) {
            highlightedText[item.matchedField] = renderHighlightedText(primaryRaw, ranges, tag);
          }
        }

        if (highlightFieldsRule === 'all-matched' && item.matches) {
          for (let m = 0; m < item.matches.length; m++) {
            const aux = item.matches[m];
            const auxFIdx = this.fieldNameToIndex.get(aux.field);
            if (auxFIdx !== undefined) {
              const auxRaw = this.rawFieldStrings[dIdx][auxFIdx] ?? '';
              const ranges = alignHighlights(auxRaw, query, { mode, folded: this.folded });
              aux.highlights = ranges;
              highlights[aux.field] = ranges;
              if (tag) {
                highlightedText[aux.field] = renderHighlightedText(auxRaw, ranges, tag);
              }
            }
          }
        }
      }

      item.highlights = highlights;
      if (tag) {
        item.highlightedText = highlightedText;
      }
    }
  }

  async add(_docs: TDoc | TDoc[], _options?: AddOptions): Promise<MutationResult> {
    throw new Error('DocumentIndex.add is scheduled for M4 implementation.');
  }

  async update(_docs: TDoc | TDoc[]): Promise<MutationResult> {
    throw new Error('DocumentIndex.update is scheduled for M4 implementation.');
  }

  async remove(_ids: DocumentId | DocumentId[]): Promise<MutationResult> {
    throw new Error('DocumentIndex.remove is scheduled for M4 implementation.');
  }

  async applyBatch(_batch: MutationBatch<TDoc>, _options?: AddOptions): Promise<MutationResult> {
    throw new Error('DocumentIndex.applyBatch is scheduled for M4 implementation.');
  }

  serialize(): ArrayBuffer {
    throw new Error('DocumentIndex.serialize is scheduled for M6 implementation.');
  }

  restore(_buffer: ArrayBuffer, _options?: { transfer?: boolean }): void {
    throw new Error('DocumentIndex.restore is scheduled for M6 implementation.');
  }

  getStats(): DocumentIndexStats {
    const adapter = this.gpuEngine?.adapterInfo;
    const docCount = this.records.length;
    const rowCount = this.rowTokens.length;
    const tombstoneCount = this.tombstones.size;
    const tombstoneRatio = rowCount > 0 ? tombstoneCount / rowCount : 0;
    const ramBytes = this.totalTokens * 4;

    return {
      size: docCount,
      engine: this.engineType,
      vramAllocatedBytes: this.vramAllocatedBytes,
      adapterVendor: adapter?.vendor,
      adapterRenderer: adapter?.renderer,
      profileId: this.profileId,
      unicodeVersion: UNICODE_VERSION,
      scoringVersion: SCORING_VERSION,
      tokenCount: this.totalTokens,
      folded: this.folded,
      formatVersion: DOC_FORMAT_VERSION,
      docCount,
      rowCount,
      tombstoneCount,
      tombstoneRatio,
      buildTimeMs: this.buildTimeMs,
      mutationEpoch: this.mutationEpoch,
      memory: {
        vramBytes: this.vramAllocatedBytes,
        ramBytes,
        totalBytes: this.vramAllocatedBytes + ramBytes
      },
      fallbackReason: this.fallbackReason
    };
  }

  destroy(): void {
    this.isDestroyed = true;
    if (this.unsubscribeDeviceLost) {
      this.unsubscribeDeviceLost();
      this.unsubscribeDeviceLost = undefined;
    }
    if (this.gpuEngine) {
      this.gpuEngine.destroy();
      this.gpuEngine = null;
    }
    this.records = [];
    this.docIds = [];
    this.idToDocIndex.clear();
    this.rowTokens = [];
    this.rowToDocIndex = [];
    this.rowToFieldIndex = [];
    this.rawFieldStrings = [];
    this.totalTokens = 0;
    this.tombstones.clear();
    this.engineType = 'cpu';
    this.vramAllocatedBytes = 0;
  }

  [Symbol.dispose](): void {
    this.destroy();
  }
}
