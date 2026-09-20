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
  abortError,
  nowMs
} from './runtime-guards';
import {
  DOC_FORMAT_VERSION,
  QUERY_TOKENS_MAX,
  SCORING_VERSION,
  UNICODE_VERSION,
  DuplicateIdError,
  DocumentNotFoundError,
  IncompatibleOptionError,
  ProfileMismatchError,
  QueryTooLongError,
  type TextProfileId
} from './text-profile';
import { alignHighlights, renderHighlightedText } from './highlight';
import {
  deserializeDocumentSnapshot,
  restoreDocumentIndex,
  serializeDocumentIndex,
  type RestoredDocumentSnapshot
} from './persistence';
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
  RestoreDocumentIndexOptions,
  SearchMode,
  SearchTimings,
  SerializeDocumentIndexOptions,
  DocumentIndexSchema
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
  private readonly folded: boolean;
  private readonly preferGpu: boolean;
  private gpuEngine: WebGPUEngine | null = null;
  private engineType: EngineType = 'cpu';
  private fallbackReason?: FallbackReason;
  private vramAllocatedBytes: number = 0;
  private unsubscribeDeviceLost?: () => void;
  private isDestroyed: boolean = false;
  private readonly profileId: TextProfileId = 'unicode-default';

  // Active documents state
  private records: TDoc[] = [];
  private docIds: DocumentId[] = [];
  private idToDocIndex: Map<DocumentId, number> = new Map();
  private getId: (doc: TDoc) => DocumentId;

  // Fields and weights (stratified: sorted by weight desc)
  private sortedFields: InternalField<TDoc>[] = [];
  private readonly fieldNameToIndex: Map<string, number> = new Map();

  // Tokens and row mapping
  private rowTokens: Uint32Array[] = [];
  private rowToDocIndex: number[] = [];
  private rowToFieldIndex: number[] = [];
  private docToRowIndices: number[][] = []; // [docIndex][fieldIndex] -> row index
  private rawFieldStrings: string[][] = []; // [docIndex][fieldIndex]
  private totalTokens: number = 0;
  private candidateCapacity: number = 8192;
  private buildTimeMs: number = 0;
  private restoreTimeMs?: number;
  private lastMutationTimeMs?: number;
  private mutationEpoch: number = 0;
  private tombstones: Set<number> = new Set(); // M4 tombstone candidate filtering
  private searchMutex: Promise<any> = Promise.resolve();
  private generation: number = 0;
  private readonly initialCapacity: number;
  private readonly growthFactor: number;

  constructor(options: DocumentIndexOptions<TDoc>) {
    if (!options || !Array.isArray(options.fields) || options.fields.length === 0) {
      throw new TypeError('[webgpu-search] DocumentIndex expects options.fields to be a non-empty array.');
    }
    this.options = options;
    this.folded = !(options.caseSensitive ?? false);
    this.preferGpu = options.preferGpu ?? false;
    this.initialCapacity = typeof options.initialCapacity === 'number' && Number.isFinite(options.initialCapacity) && options.initialCapacity > 0
      ? Math.floor(options.initialCapacity)
      : 0;
    this.growthFactor = typeof options.growthFactor === 'number' && Number.isFinite(options.growthFactor) && options.growthFactor >= 1.0
      ? options.growthFactor
      : 1.5;

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
    this.docToRowIndices = Array.from({ length: docCount }, () => new Array(fieldCount));
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
        this.docToRowIndices[d][f] = rowIdx;
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
            const effectiveCap = this.initialCapacity > 0 ? this.initialCapacity : Math.max(16, docCount);
            const rowCap = effectiveCap * fieldCount;
            const avgTokens = totalRows > 0 ? corpusTokens / totalRows : 16;
            const tokenCap = Math.max(Math.floor(rowCap * avgTokens), 64);
            await gpu.loadDataset(packed, {
              rowCapacity: rowCap,
              tokenCapacity: tokenCap,
              growthFactor: this.initialCapacity > 0 ? 1.0 : this.growthFactor
            });

            this.gpuEngine = gpu;
            this.engineType = 'webgpu';
            this.fallbackReason = undefined;
            this.vramAllocatedBytes = gpu.vramAllocatedBytes;
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
    await this.searchMutex;
    if (this.isDestroyed) {
      throw new Error('[webgpu-search] DocumentIndex has been destroyed.');
    }
    throwIfAborted(options.signal);
    const gen = this.generation;

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
    if (this.idToDocIndex.size === 0) {
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

        if (this.isDestroyed || this.generation !== gen) {
          throw abortError();
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
          const doc = this.records[dIdx];
          if (!doc) continue;
          if (filter && !filter(doc)) continue;

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
    if (this.isDestroyed || this.generation !== gen) {
      throw abortError();
    }
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
        const fieldStrings = this.records.map((doc, dIdx) => (doc && this.rawFieldStrings[dIdx] ? this.rawFieldStrings[dIdx][fIdx] : ''));
        const legacyResult = mode === 'fuzzy'
          ? cpuEngine.searchUFuzzy(fieldStrings, query, this.records.length, caseSensitive)
          : cpuEngine.searchNative(fieldStrings, query, this.records.length, caseSensitive);

        for (const hit of legacyResult.results) {
          const dIdx = hit.index;
          const doc = this.records[dIdx];
          if (!doc) continue;
          if (filter && !filter(doc)) continue;
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
      filter ? (dIdx) => {
        const doc = this.records[dIdx];
        return doc !== null && doc !== undefined && filter(doc);
      } : undefined
    );

    if (this.isDestroyed || this.generation !== gen) {
      throw abortError();
    }
    throwIfAborted(signal);

    const idProp = typeof this.options.idField === 'string' ? this.options.idField : 'id';
    const enrichedResults: DocumentSearchResultItem<TDoc>[] = [];
    for (let i = 0; i < parityResult.results.length; i++) {
      const hit = parityResult.results[i];
      const id = this.docIds[hit.docIndex];
      if (id === null || id === undefined) continue;
      const doc = this.records[hit.docIndex] ?? ({ [idProp]: id } as any);
      enrichedResults.push({
        id,
        doc,
        score: hit.score,
        matchedField: hit.matchedField,
        matches: hit.matches
      });
    }
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
    const escapeHtml = options.highlightOptions?.escapeHtml ?? options.escapeHtml ?? false;

    if (
      typeof highlightFieldsRule === 'string' &&
      highlightFieldsRule !== 'all-matched' &&
      highlightFieldsRule !== 'matched-field' &&
      highlightFieldsRule !== 'all-fields'
    ) {
      throw new TypeError(
        `[webgpu-search] Invalid highlightFields option: "${highlightFieldsRule}". Must be 'all-matched', 'matched-field', 'all-fields', or an array of field names.`
      );
    }

    const queryTokens = normalizeText(query, this.folded).tokens;
    const alignOpts = { mode, folded: this.folded, queryTokens };

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
          const ranges = alignHighlights(rawStr, query, alignOpts);
          if (ranges.length > 0) {
            highlights[fieldDef.name] = ranges;
            if (tag) {
              highlightedText[fieldDef.name] = renderHighlightedText(rawStr, ranges, tag, escapeHtml);
            }
          }
        }
      } else if (Array.isArray(highlightFieldsRule)) {
        for (let k = 0; k < highlightFieldsRule.length; k++) {
          const fName = highlightFieldsRule[k];
          const fIdx = this.fieldNameToIndex.get(fName);
          if (fIdx !== undefined) {
            const rawStr = this.rawFieldStrings[dIdx][fIdx] ?? '';
            const ranges = alignHighlights(rawStr, query, alignOpts);
            if (ranges.length > 0) {
              highlights[fName] = ranges;
              if (tag) {
                highlightedText[fName] = renderHighlightedText(rawStr, ranges, tag, escapeHtml);
              }
            }
          }
        }
      } else {
        // 'matched-field' or 'all-matched' (default)
        const primaryFIdx = this.fieldNameToIndex.get(item.matchedField);
        if (primaryFIdx !== undefined) {
          const primaryRaw = this.rawFieldStrings[dIdx][primaryFIdx] ?? '';
          const ranges = alignHighlights(primaryRaw, query, alignOpts);
          highlights[item.matchedField] = ranges;
          if (tag) {
            highlightedText[item.matchedField] = renderHighlightedText(primaryRaw, ranges, tag, escapeHtml);
          }
        }

        if (highlightFieldsRule === 'all-matched' && item.matches) {
          for (let m = 0; m < item.matches.length; m++) {
            const aux = item.matches[m];
            const auxFIdx = this.fieldNameToIndex.get(aux.field);
            if (auxFIdx !== undefined) {
              const auxRaw = this.rawFieldStrings[dIdx][auxFIdx] ?? '';
              const ranges = alignHighlights(auxRaw, query, alignOpts);
              aux.highlights = ranges;
              highlights[aux.field] = ranges;
              if (tag) {
                highlightedText[aux.field] = renderHighlightedText(auxRaw, ranges, tag, escapeHtml);
              }
            }
          }
        }
      }

      // Synchronize auxiliary match highlights across all modes if highlights exist
      if (item.matches) {
        for (let m = 0; m < item.matches.length; m++) {
          const aux = item.matches[m];
          if (highlights[aux.field] !== undefined) {
            aux.highlights = highlights[aux.field];
          }
        }
      }

      item.highlights = highlights;
      if (tag) {
        item.highlightedText = highlightedText;
      }
    }
  }

  private validateDocumentId(id: unknown, context: string): void {
    if (
      id === null ||
      id === undefined ||
      (typeof id !== 'string' && typeof id !== 'number') ||
      (typeof id === 'number' && !Number.isFinite(id))
    ) {
      throw new TypeError(
        `[webgpu-search] Invalid document ID in ${context}: ${String(id)}. DocumentId must be a non-empty string or finite number.`
      );
    }
    if (typeof id === 'string' && id.length === 0) {
      throw new TypeError(
        `[webgpu-search] Document ID in ${context} must not be an empty string.`
      );
    }
  }

  private prepareDocFields(doc: TDoc, id: DocumentId): {
    id: DocumentId;
    doc: TDoc;
    rawStrings: string[];
    tokens: Uint32Array[];
    tokenCount: number;
  } {
    const fieldCount = this.sortedFields.length;
    const rawStrings = new Array<string>(fieldCount);
    const tokens = new Array<Uint32Array>(fieldCount);
    let tokenCount = 0;

    for (let f = 0; f < fieldCount; f++) {
      const field = this.sortedFields[f];
      const rawVal = field.getter(doc);
      let rawStr = '';
      if (rawVal !== null && rawVal !== undefined) {
        if (Array.isArray(rawVal)) {
          rawStr = rawVal.filter((x) => x !== null && x !== undefined).join(' ');
        } else {
          rawStr = typeof rawVal === 'string' ? rawVal : String(rawVal);
        }
      }
      rawStrings[f] = rawStr;
      const norm = normalizeText(rawStr, this.folded);
      tokens[f] = norm.tokens;
      tokenCount += norm.tokenCount;
    }

    return { id, doc, rawStrings, tokens, tokenCount };
  }

  private queued<T>(fn: () => Promise<T>): Promise<T> {
    const p = this.searchMutex.then(fn, fn);
    this.searchMutex = p.catch(() => {});
    return p;
  }

  async add(docs: TDoc | TDoc[], options?: AddOptions): Promise<MutationResult> {
    const list = Array.isArray(docs) ? docs : [docs];
    return this.applyBatch({ add: list }, options);
  }

  async update(docs: TDoc | TDoc[]): Promise<MutationResult> {
    const list = Array.isArray(docs) ? docs : [docs];
    return this.applyBatch({ update: list });
  }

  async remove(ids: DocumentId | DocumentId[]): Promise<MutationResult> {
    const list = Array.isArray(ids) ? ids : [ids];
    return this.applyBatch({ remove: list });
  }

  async applyBatch(batch: MutationBatch<TDoc>, options?: AddOptions): Promise<MutationResult> {
    if (this.isDestroyed) {
      throw new Error('[webgpu-search] DocumentIndex has been destroyed.');
    }
    if (!batch || typeof batch !== 'object') {
      throw new TypeError('[webgpu-search] applyBatch expects batch object.');
    }
    return this.queued(async () => {
      if (this.isDestroyed) {
        throw new Error('[webgpu-search] DocumentIndex has been destroyed.');
      }
      const tStart = nowMs();
      const upsert = options?.upsert ?? false;

      // PHASE 1: ATOMIC TWO-PHASE VALIDATION (State remains untouched on error)
      const removeIds: DocumentId[] = [];
      const willRemoveSet = new Set<DocumentId>();
      if (batch.remove !== undefined) {
        if (!Array.isArray(batch.remove)) {
          throw new TypeError('[webgpu-search] batch.remove must be an array of document IDs.');
        }
        for (let i = 0; i < batch.remove.length; i++) {
          const id = batch.remove[i];
          this.validateDocumentId(id, `remove at index ${i}`);
          removeIds.push(id);
          if (this.idToDocIndex.has(id)) {
            willRemoveSet.add(id);
          }
        }
      }

      interface PreparedDocUpdate {
        id: DocumentId;
        doc: TDoc;
        rawStrings: string[];
        tokens: Uint32Array[];
        tokenCount: number;
      }
      const preparedUpdates: PreparedDocUpdate[] = [];
      const updatedIdsInBatch = new Set<DocumentId>();
      if (batch.update !== undefined) {
        if (!Array.isArray(batch.update)) {
          throw new TypeError('[webgpu-search] batch.update must be an array of documents.');
        }
        for (let i = 0; i < batch.update.length; i++) {
          const doc = batch.update[i];
          if (doc === null || typeof doc !== 'object') {
            throw new TypeError(`[webgpu-search] batch.update document at index ${i} must be an object.`);
          }
          const id = this.getId(doc);
          this.validateDocumentId(id, `update at index ${i}`);
          if (!this.idToDocIndex.has(id) || willRemoveSet.has(id)) {
            throw new DocumentNotFoundError(id);
          }
          if (updatedIdsInBatch.has(id)) {
            throw new DuplicateIdError(id);
          }
          updatedIdsInBatch.add(id);
          const prepared = this.prepareDocFields(doc, id);
          preparedUpdates.push(prepared);
        }
      }

      interface PreparedDocAdd {
        id: DocumentId;
        doc: TDoc;
        rawStrings: string[];
        tokens: Uint32Array[];
        tokenCount: number;
      }
      const preparedAdds: PreparedDocAdd[] = [];
      const addedIdsInBatch = new Set<DocumentId>();

      if (batch.add !== undefined) {
        if (!Array.isArray(batch.add)) {
          throw new TypeError('[webgpu-search] batch.add must be an array of documents.');
        }
        for (let i = 0; i < batch.add.length; i++) {
          const doc = batch.add[i];
          if (doc === null || typeof doc !== 'object') {
            throw new TypeError(`[webgpu-search] batch.add document at index ${i} must be an object.`);
          }
          const id = this.getId(doc);
          this.validateDocumentId(id, `add at index ${i}`);

          const existsInIndex = this.idToDocIndex.has(id) && !willRemoveSet.has(id);
          const existsInBatch = addedIdsInBatch.has(id);

          if (existsInIndex || existsInBatch) {
            if (!upsert) {
              throw new DuplicateIdError(id);
            }
          }
          addedIdsInBatch.add(id);

          const prepared = this.prepareDocFields(doc, id);
          preparedAdds.push(prepared);
        }
      }

      // PHASE 2: DETERMINISTIC EXECUTION (remove -> update -> add)
      this.generation++;
      let removedCount = 0;
      let updatedCount = 0;
      let addedCount = 0;

      // A. Remove
      for (let i = 0; i < removeIds.length; i++) {
        const id = removeIds[i];
        const d = this.idToDocIndex.get(id);
        if (d !== undefined) {
          const rows = this.docToRowIndices[d];
          if (rows) {
            for (let f = 0; f < rows.length; f++) {
              this.tombstones.add(rows[f]);
            }
          }
          this.idToDocIndex.delete(id);
          this.records[d] = null as any;
          this.docIds[d] = null as any;
          this.rawFieldStrings[d] = null as any;
          this.docToRowIndices[d] = null as any;
          removedCount++;
        }
      }

      // B. Update
      const newRowTokensToAppend: Uint32Array[] = [];
      const newRowDocIndicesToAppend: number[] = [];
      const newRowFieldIndicesToAppend: number[] = [];
      let newTokensToAppendCount = 0;

      for (let i = 0; i < preparedUpdates.length; i++) {
        const upd = preparedUpdates[i];
        const oldD = this.idToDocIndex.get(upd.id)!;
        const oldRows = this.docToRowIndices[oldD];
        if (oldRows) {
          for (let f = 0; f < oldRows.length; f++) {
            this.tombstones.add(oldRows[f]);
          }
        }
        this.records[oldD] = null as any;
        this.docIds[oldD] = null as any;
        this.rawFieldStrings[oldD] = null as any;
        this.docToRowIndices[oldD] = null as any;

        const newD = this.records.length;
        this.records.push(upd.doc);
        this.docIds.push(upd.id);
        this.idToDocIndex.set(upd.id, newD);
        this.rawFieldStrings.push(upd.rawStrings);

        const docRows: number[] = new Array(this.sortedFields.length);
        for (let f = 0; f < this.sortedFields.length; f++) {
          const rowIdx = this.rowTokens.length + newRowTokensToAppend.length;
          newRowTokensToAppend.push(upd.tokens[f]);
          newRowDocIndicesToAppend.push(newD);
          newRowFieldIndicesToAppend.push(f);
          docRows[f] = rowIdx;
          newTokensToAppendCount += upd.tokens[f].length;
        }
        this.docToRowIndices.push(docRows);
        updatedCount++;
      }

      // C. Add
      for (let i = 0; i < preparedAdds.length; i++) {
        const ad = preparedAdds[i];
        if (this.idToDocIndex.has(ad.id)) {
          const oldD = this.idToDocIndex.get(ad.id)!;
          const oldRows = this.docToRowIndices[oldD];
          if (oldRows) {
            for (let f = 0; f < oldRows.length; f++) {
              this.tombstones.add(oldRows[f]);
            }
          }
          this.records[oldD] = null as any;
          this.docIds[oldD] = null as any;
          this.rawFieldStrings[oldD] = null as any;
          this.docToRowIndices[oldD] = null as any;
          updatedCount++;
        } else {
          addedCount++;
        }

        const newD = this.records.length;
        this.records.push(ad.doc);
        this.docIds.push(ad.id);
        this.idToDocIndex.set(ad.id, newD);
        this.rawFieldStrings.push(ad.rawStrings);

        const docRows: number[] = new Array(this.sortedFields.length);
        for (let f = 0; f < this.sortedFields.length; f++) {
          const rowIdx = this.rowTokens.length + newRowTokensToAppend.length;
          newRowTokensToAppend.push(ad.tokens[f]);
          newRowDocIndicesToAppend.push(newD);
          newRowFieldIndicesToAppend.push(f);
          docRows[f] = rowIdx;
          newTokensToAppendCount += ad.tokens[f].length;
        }
        this.docToRowIndices.push(docRows);
      }

      // Append newly prepared rows to rowTokens, rowToDocIndex, rowToFieldIndex
      if (newRowTokensToAppend.length > 0) {
        for (let r = 0; r < newRowTokensToAppend.length; r++) {
          this.rowTokens.push(newRowTokensToAppend[r]);
          this.rowToDocIndex.push(newRowDocIndicesToAppend[r]);
          this.rowToFieldIndex.push(newRowFieldIndicesToAppend[r]);
        }
        this.totalTokens += newTokensToAppendCount;
      }

      // D. Check Compaction / GPU Headroom
      const totalRows = this.rowTokens.length;
      const tombstoneCount = this.tombstones.size;
      const tombstoneRatio = totalRows > 0 ? tombstoneCount / totalRows : 0;

      let compacted = false;
      const isGpu = this.engineType === 'webgpu' && this.gpuEngine !== null && this.gpuEngine.isReady;
      const gpuHeadroomExhausted = isGpu && !this.gpuEngine!.canFitHeadroom(totalRows, this.totalTokens);

      const activeDocCount = this.idToDocIndex.size;
      const newCandidateCap = Math.min(32768, Math.max(8192, Math.floor(activeDocCount * this.sortedFields.length * 0.1)));
      if (newCandidateCap > this.candidateCapacity) {
        this.candidateCapacity = newCandidateCap;
        this.gpuEngine?.ensureCandidateCapacity(newCandidateCap);
      }

      try {
        if (tombstoneRatio >= 0.25 || gpuHeadroomExhausted) {
          await this.compact();
          compacted = true;
        } else if (isGpu && newRowTokensToAppend.length > 0) {
          await this.syncAppendedRowsToGpu(newRowTokensToAppend, totalRows, newTokensToAppendCount);
        } else if (!isGpu && (this.preferGpu || totalRows >= (this.options.threshold ?? 30_000)) && this.options.preferGpu !== false && activeDocCount > 0) {
          await this.tryInitializeGpuEngine();
        }
      } catch (err) {
        console.warn('[webgpu-search] GPU sync failed during mutation; falling back to CPU:', err);
        if (this.gpuEngine) {
          try { this.gpuEngine.destroy(); } catch {}
          this.gpuEngine = null;
        }
        this.engineType = 'cpu';
        this.fallbackReason = 'gpu-execution-error';
        this.vramAllocatedBytes = 0;
      }

      this.mutationEpoch++;
      const durationMs = nowMs() - tStart;
      this.lastMutationTimeMs = durationMs;

      return {
        added: addedCount,
        updated: updatedCount,
        removed: removedCount,
        mutationEpoch: this.mutationEpoch,
        compacted,
        durationMs
      };
    });
  }

  private async syncAppendedRowsToGpu(
    newRows: Uint32Array[],
    totalRows: number,
    newTokensCount: number
  ): Promise<void> {
    if (!this.gpuEngine || !this.gpuEngine.isReady) return;
    const numNewRows = newRows.length;
    const oldTotalTokens = this.totalTokens - newTokensCount;

    const newOffsets = new Uint32Array(numNewRows);
    let runningOffset = oldTotalTokens;
    for (let i = 0; i < numNewRows; i++) {
      runningOffset += newRows[i].length;
      newOffsets[i] = runningOffset;
    }

    const newTokens = new Uint32Array(newTokensCount);
    let pos = 0;
    for (let i = 0; i < numNewRows; i++) {
      newTokens.set(newRows[i], pos);
      pos += newRows[i].length;
    }

    await this.gpuEngine.appendRows(newTokens, newOffsets, totalRows);
    this.vramAllocatedBytes = this.gpuEngine.vramAllocatedBytes;
  }

  compactCpu(): void {
    const activeDocCount = this.idToDocIndex.size;
    const fieldCount = this.sortedFields.length;
    const totalRows = activeDocCount * fieldCount;

    const activeOldDocIndices: number[] = [];
    for (let d = 0; d < this.records.length; d++) {
      const id = this.docIds[d];
      if (id !== null && id !== undefined && this.idToDocIndex.get(id) === d) {
        activeOldDocIndices.push(d);
      }
    }

    const newRecords: TDoc[] = new Array(activeDocCount);
    const newDocIds: DocumentId[] = new Array(activeDocCount);
    const newIdToDocIndex = new Map<DocumentId, number>();
    const newRawFieldStrings: string[][] = Array.from({ length: activeDocCount }, () => new Array(fieldCount));

    for (let newD = 0; newD < activeDocCount; newD++) {
      const oldD = activeOldDocIndices[newD];
      const doc = this.records[oldD];
      const id = this.docIds[oldD];
      newRecords[newD] = doc;
      newDocIds[newD] = id;
      newIdToDocIndex.set(id, newD);
      for (let f = 0; f < fieldCount; f++) {
        newRawFieldStrings[newD][f] = this.rawFieldStrings[oldD][f];
      }
    }

    const newRowTokens: Uint32Array[] = new Array(totalRows);
    const newRowToDocIndex: number[] = new Array(totalRows);
    const newRowToFieldIndex: number[] = new Array(totalRows);
    const newDocToRowIndices: number[][] = Array.from({ length: activeDocCount }, () => new Array(fieldCount));

    let corpusTokens = 0;
    let rowIdx = 0;

    for (let f = 0; f < fieldCount; f++) {
      for (let newD = 0; newD < activeDocCount; newD++) {
        const oldD = activeOldDocIndices[newD];
        const oldRowIdx = this.docToRowIndices[oldD][f];
        const tokens = this.rowTokens[oldRowIdx];
        newRowTokens[rowIdx] = tokens;
        newRowToDocIndex[rowIdx] = newD;
        newRowToFieldIndex[rowIdx] = f;
        newDocToRowIndices[newD][f] = rowIdx;
        corpusTokens += tokens.length;
        rowIdx++;
      }
    }

    this.records = newRecords;
    this.docIds = newDocIds;
    this.idToDocIndex = newIdToDocIndex;
    this.rawFieldStrings = newRawFieldStrings;
    this.rowTokens = newRowTokens;
    this.rowToDocIndex = newRowToDocIndex;
    this.rowToFieldIndex = newRowToFieldIndex;
    this.docToRowIndices = newDocToRowIndices;
    this.totalTokens = corpusTokens;
    this.tombstones.clear();
  }

  private isStratified(): boolean {
    const docCount = this.records.length;
    if (docCount === 0) return true;
    for (let r = 0; r < this.rowTokens.length; r++) {
      if (
        this.rowToFieldIndex[r] !== Math.floor(r / docCount) ||
        this.rowToDocIndex[r] !== (r % docCount)
      ) {
        return false;
      }
    }
    return true;
  }

  compactSync(): void {
    if (this.tombstones.size > 0 || !this.isStratified()) {
      this.compactCpu();
      if (this.gpuEngine) {
        this.gpuEngine.destroy();
        this.gpuEngine = null;
        this.engineType = 'cpu';
        this.fallbackReason = 'prefer-cpu';
      }
    }
  }

  private async compact(): Promise<void> {
    this.compactCpu();

    if (this.engineType === 'webgpu' && this.gpuEngine && this.gpuEngine.isReady) {
      const activeDocCount = this.idToDocIndex.size;
      const totalRows = this.rowTokens.length;
      if (activeDocCount === 0) {
        const packed = packUnicodeToGPUBuffer([], { folded: this.folded });
        await this.gpuEngine.loadDataset(packed);
        this.vramAllocatedBytes = this.gpuEngine.vramAllocatedBytes;
      } else {
        const packed = packUnicodeToGPUBuffer(this.rowTokens, {
          folded: this.folded,
          totalTokens: this.totalTokens
        });
        const rowCap = Math.max(totalRows, 16);
        const tokenCap = Math.max(this.totalTokens, 64);
        await this.gpuEngine.loadDataset(packed, {
          rowCapacity: rowCap,
          tokenCapacity: tokenCap,
          growthFactor: this.growthFactor
        });
        this.vramAllocatedBytes = this.gpuEngine.vramAllocatedBytes;
      }
    }
  }

  getRecords(): TDoc[] {
    return this.records.slice();
  }

  getDocIds(): DocumentId[] {
    return this.docIds.slice();
  }

  getSortedFields(): InternalField<TDoc>[] {
    return this.sortedFields;
  }

  getRowTokens(): Uint32Array[] {
    return this.rowTokens;
  }

  getTotalTokens(): number {
    return this.totalTokens;
  }

  getTombstoneCount(): number {
    return this.tombstones.size;
  }

  isFolded(): boolean {
    return this.folded;
  }

  getProfileId(): TextProfileId {
    return this.profileId;
  }

  getUnicodeVersion(): string {
    return UNICODE_VERSION;
  }

  getScoringVersion(): string {
    return SCORING_VERSION;
  }

  serialize(options?: SerializeDocumentIndexOptions): ArrayBuffer {
    if (this.isDestroyed) {
      throw new Error('[webgpu-search] DocumentIndex has been destroyed.');
    }
    return serializeDocumentIndex(this, options);
  }

  async restore(buffer: ArrayBuffer, options?: RestoreDocumentIndexOptions<TDoc>): Promise<void> {
    if (this.isDestroyed) {
      throw new Error('[webgpu-search] DocumentIndex has been destroyed.');
    }
    const t0 = nowMs();
    await this.queued(async () => {
      const snapshot = deserializeDocumentSnapshot<TDoc>(buffer, options);
      await this.applySnapshotData(snapshot, options);
      this.restoreTimeMs = nowMs() - t0;
    });
  }

  static async fromSnapshot<TDoc = Record<string, unknown>>(
    buffer: ArrayBuffer,
    options?: RestoreDocumentIndexOptions<TDoc>
  ): Promise<DocumentIndex<TDoc>> {
    return restoreDocumentIndex<TDoc>(buffer, options);
  }

  static async fromSnapshotData<TDoc = Record<string, unknown>>(
    snapshot: RestoredDocumentSnapshot<TDoc>,
    options?: RestoreDocumentIndexOptions<TDoc>
  ): Promise<DocumentIndex<TDoc>> {
    const t0 = nowMs();
    const userFields = options?.options?.fields;
    const userFieldMap = new Map<string, any>();
    if (Array.isArray(userFields)) {
      for (const uf of userFields) {
        if (typeof uf === 'string') {
          userFieldMap.set(uf, uf);
        } else if (uf && typeof uf === 'object' && typeof uf.name === 'string') {
          userFieldMap.set(uf.name, uf);
        }
      }
    }

    const resolvedFields = snapshot.schema.fields.map((sf) => {
      const uf = userFieldMap.get(sf.name);
      const getter = typeof uf === 'object' && uf !== null && typeof uf.getter === 'function'
        ? uf.getter
        : undefined;
      return {
        name: sf.name,
        weight: sf.weight,
        getter
      };
    });

    const mergedOptions: DocumentIndexOptions<TDoc> = {
      ...(options?.options as any),
      fields: resolvedFields,
      idField: options?.options?.idField ?? snapshot.schema.idField ?? 'id',
      caseSensitive: options?.options?.caseSensitive ?? snapshot.schema.caseSensitive ?? !snapshot.header.folded,
      device: (options?.device ?? options?.options?.device) as GPUDevice | undefined,
      preferGpu: options?.options?.preferGpu ?? snapshot.schema.preferGpu,
      threshold: options?.options?.threshold ?? snapshot.schema.threshold,
      candidateCapacity: options?.options?.candidateCapacity ?? snapshot.schema.candidateCapacity,
      initialCapacity: options?.options?.initialCapacity ?? snapshot.schema.initialCapacity,
      growthFactor: options?.options?.growthFactor ?? snapshot.schema.growthFactor
    };

    const index = new DocumentIndex<TDoc>(mergedOptions);
    await index.applySnapshotData(snapshot, options);
    index.restoreTimeMs = nowMs() - t0;
    return index;
  }

  private configureFromSchema(
    schema: DocumentIndexSchema,
    options?: RestoreDocumentIndexOptions<TDoc>
  ): void {
    const userFields = options?.options?.fields;
    const userFieldMap = new Map<string, any>();
    if (Array.isArray(userFields)) {
      for (const uf of userFields) {
        if (typeof uf === 'string') {
          userFieldMap.set(uf, uf);
        } else if (uf && typeof uf === 'object' && typeof uf.name === 'string') {
          userFieldMap.set(uf.name, uf);
        }
      }
    }

    const normalized: InternalField<TDoc>[] = schema.fields.map(
      (f: { name: string; weight: number }, originalIndex: number) => {
      const name = f.name;
      const weight = f.weight ?? 1.0;
      const uf = userFieldMap.get(name);
      let getter: (doc: TDoc) => string | string[] | undefined | null;
      if (uf && typeof uf === 'object' && typeof uf.getter === 'function') {
        getter = uf.getter;
      } else {
        getter = (doc: any) => doc[name];
      }
      return { name, weight, getter, originalIndex };
    });

    this.sortedFields = [...normalized].sort((a, b) => {
      if (b.weight !== a.weight) return b.weight - a.weight;
      return a.originalIndex - b.originalIndex;
    });

    this.fieldNameToIndex.clear();
    for (let i = 0; i < this.sortedFields.length; i++) {
      this.fieldNameToIndex.set(this.sortedFields[i].name, i);
    }

    if (typeof options?.options?.idField === 'function') {
      this.getId = options.options.idField;
    } else if (typeof options?.options?.idField === 'string') {
      const prop = options.options.idField;
      this.getId = (doc: any) => doc[prop];
    } else if (typeof schema.idField === 'string') {
      const prop = schema.idField;
      this.getId = (doc: any) => doc[prop];
    } else if (options?.options?.idField === undefined && schema.idField === undefined) {
      this.getId = (doc: any) => doc.id;
    }
  }

  private async applySnapshotData(
    snapshot: RestoredDocumentSnapshot<TDoc>,
    options?: RestoreDocumentIndexOptions<TDoc>
  ): Promise<void> {
    this.configureFromSchema(snapshot.schema, options);

    if (this.unsubscribeDeviceLost) {
      this.unsubscribeDeviceLost();
      this.unsubscribeDeviceLost = undefined;
    }
    if (this.gpuEngine) {
      this.gpuEngine.destroy();
      this.gpuEngine = null;
    }
    this.tombstones.clear();

    const docCount = snapshot.header.docCount;
    const fieldCount = this.sortedFields.length;
    const totalRows = snapshot.header.rowCount;

    this.records = snapshot.records.slice();
    this.docIds = snapshot.docIds.slice();
    this.idToDocIndex.clear();

    if (this.records.length > 0 && this.docIds.length === 0) {
      this.docIds = new Array(docCount);
      for (let i = 0; i < docCount; i++) {
        const id = this.getId(this.records[i]);
        this.docIds[i] = id;
        this.idToDocIndex.set(id, i);
      }
    } else {
      for (let i = 0; i < this.docIds.length; i++) {
        this.idToDocIndex.set(this.docIds[i], i);
      }
    }

    // If both records and docIds are present, ensure exact ID correspondence alignment
    if (this.records.length === docCount && this.docIds.length === docCount) {
      const recMap = new Map<DocumentId, TDoc>();
      for (const rec of this.records) {
        recMap.set(this.getId(rec), rec);
      }
      if (recMap.size === docCount) {
        const aligned: TDoc[] = new Array(docCount);
        let allFound = true;
        for (let i = 0; i < docCount; i++) {
          const rec = recMap.get(this.docIds[i]);
          if (rec === undefined) {
            allFound = false;
            break;
          }
          aligned[i] = rec;
        }
        if (allFound) {
          this.records = aligned;
        }
      }
    }

    if (this.records.length === 0 && docCount > 0) {
      const idProp = typeof this.options.idField === 'string' ? this.options.idField : 'id';
      this.records = new Array(docCount);
      for (let i = 0; i < docCount; i++) {
        this.records[i] = { [idProp]: this.docIds[i] } as any;
      }
    }

    const rawCap = this.options.candidateCapacity;
    const validUserCap = typeof rawCap === 'number' && Number.isFinite(rawCap) && rawCap > 0
      ? Math.floor(rawCap)
      : Math.floor(docCount * fieldCount * 0.1);
    this.candidateCapacity = Math.min(32768, Math.max(8192, validUserCap));

    this.rowTokens = new Array(totalRows);
    this.rowToDocIndex = new Array(totalRows);
    this.rowToFieldIndex = new Array(totalRows);
    this.docToRowIndices = Array.from({ length: docCount }, () => new Array(fieldCount));
    this.rawFieldStrings = Array.from({ length: docCount }, () => new Array(fieldCount));

    for (let r = 0; r < totalRows; r++) {
      const start = snapshot.offsets[r];
      const end = snapshot.offsets[r + 1];
      this.rowTokens[r] = snapshot.tokens.subarray(start, end);

      const f = fieldCount > 0 && docCount > 0 ? Math.floor(r / docCount) : 0;
      const d = docCount > 0 ? r % docCount : 0;
      this.rowToDocIndex[r] = d;
      this.rowToFieldIndex[r] = f;
      if (this.docToRowIndices[d]) {
        this.docToRowIndices[d][f] = r;
      }
    }

    this.totalTokens = snapshot.header.tokenCount;

    if (this.records.length === docCount && docCount > 0) {
      for (let d = 0; d < docCount; d++) {
        const doc = this.records[d];
        for (let f = 0; f < fieldCount; f++) {
          const field = this.sortedFields[f];
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
        }
      }
    }

    this.engineType = 'cpu';
    this.vramAllocatedBytes = 0;

    if (docCount > 0) {
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
        const measuredAvgBytes = totalRows > 0 ? (this.totalTokens * 4) / totalRows : 0;
        const deviceToUse = options?.device ?? this.options.device;
        const budget = checkMemoryBudget(totalRows, measuredAvgBytes, deviceToUse, this.candidateCapacity);
        if (!budget.allowed) {
          this.engineType = 'cpu';
          this.fallbackReason = 'memory-budget-exceeded';
        } else {
          try {
            const gpu = new WebGPUEngine();
            const initialized = await gpu.init(deviceToUse);
            if (initialized && gpu.isReady) {
              gpu.ensureCandidateCapacity(this.candidateCapacity);
              const packed = packUnicodeToGPUBuffer(this.rowTokens, {
                folded: this.folded,
                totalTokens: this.totalTokens
              });
              const effectiveCap = this.initialCapacity > 0 ? this.initialCapacity : Math.max(16, docCount);
              const rowCap = effectiveCap * fieldCount;
              const avgTokens = totalRows > 0 ? this.totalTokens / totalRows : 16;
              const tokenCap = Math.max(Math.floor(rowCap * avgTokens), 64);
              await gpu.loadDataset(packed, {
                rowCapacity: rowCap,
                tokenCapacity: tokenCap,
                growthFactor: this.initialCapacity > 0 ? 1.0 : this.growthFactor
              });

              this.gpuEngine = gpu;
              this.engineType = 'webgpu';
              this.fallbackReason = undefined;
              this.vramAllocatedBytes = gpu.vramAllocatedBytes;

              this.unsubscribeDeviceLost = WebGPUContextManager.onDeviceLost(() => {
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
          } catch {
            this.engineType = 'cpu';
            this.fallbackReason = 'webgpu-unsupported';
          }
        }
      }
    }

    this.generation++;
  }

  private async tryInitializeGpuEngine(): Promise<void> {
    const totalRows = this.rowTokens.length;
    const threshold = this.options.threshold ?? 30_000;
    const shouldAttempt = this.options.preferGpu === false
      ? false
      : this.preferGpu || totalRows >= threshold;
    if (!shouldAttempt || this.gpuEngine) return;

    const measuredAvgBytes = totalRows > 0 ? (this.totalTokens * 4) / totalRows : 0;
    const budget = checkMemoryBudget(totalRows, measuredAvgBytes, this.options.device, this.candidateCapacity);
    if (!budget.allowed) {
      this.fallbackReason = 'memory-budget-exceeded';
      return;
    }

    try {
      const gpu = new WebGPUEngine();
      const initialized = await gpu.init(this.options.device);
      if (initialized && gpu.isReady) {
        gpu.ensureCandidateCapacity(this.candidateCapacity);
        const packed = packUnicodeToGPUBuffer(this.rowTokens, {
          folded: this.folded,
          totalTokens: this.totalTokens
        });
        const rowCap = Math.max(totalRows, 16);
        const tokenCap = Math.max(this.totalTokens, 64);
        await gpu.loadDataset(packed, {
          rowCapacity: rowCap,
          tokenCapacity: tokenCap,
          growthFactor: this.growthFactor
        });
        this.gpuEngine = gpu;
        this.engineType = 'webgpu';
        this.fallbackReason = undefined;
        this.vramAllocatedBytes = gpu.vramAllocatedBytes;
        this.unsubscribeDeviceLost = WebGPUContextManager.onDeviceLost(() => {
          if (this.gpuEngine) {
            this.gpuEngine.destroy();
            this.gpuEngine = null;
          }
          this.engineType = 'cpu';
          this.fallbackReason = 'device-lost';
        });
      }
    } catch {
      this.engineType = 'cpu';
      this.fallbackReason = 'webgpu-unsupported';
    }
  }

  getStats(): DocumentIndexStats {
    const adapter = this.gpuEngine?.adapterInfo;
    const docCount = this.idToDocIndex.size;
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
      restoreTimeMs: this.restoreTimeMs,
      lastMutationTimeMs: this.lastMutationTimeMs,
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
    this.generation++;
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
    this.docToRowIndices = [];
    this.rawFieldStrings = [];
    this.totalTokens = 0;
    this.tombstones.clear();
    this.engineType = 'cpu';
    this.vramAllocatedBytes = 0;
    const p = this.searchMutex.then(() => {}, () => {});
    this.searchMutex = p.catch(() => {});
  }

  [Symbol.dispose](): void {
    this.destroy();
  }
}
