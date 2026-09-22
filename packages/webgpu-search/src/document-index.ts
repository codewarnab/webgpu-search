import { WebGPUEngine } from './webgpu-engine';
import { CPUEngine } from './cpu-engine';
import { GpuDevicePool } from './gpu-device-pool';
import { packDataset, checkMemoryBudget } from './dataset-packing';
import { normalizeText } from './text-normalization';
import {
  scoreExactMatchesMultiField,
  scoreFuzzyTokens,
  scoreSubstringTypoTokens,
  type FieldScoreDefinition,
  type CpuModeOptions
} from './exact-scorer';
import {
  normalizeTypoTolerance,
  type NormalizedTypoOptions
} from './search/typo-tolerance';
import {
  normalizeTokenMatchOptions,
  type NormalizedTokenMatchOptions
} from './search/token-search';
import {
  normalizePrefixOptions,
  assertPrefixLengthForQuery,
  scorePrefixTokens,
  type NormalizedPrefixOptions
} from './search/prefix-search';
import {
  compareRanked,
  isExactTokenMatch,
  normalizeTieBreakers,
  type RankableCandidate
} from './ranking';
import {
  normalizeAutocompleteOptions,
  type NormalizedAutocompleteOptions
} from './autocomplete';
import {
  normalizeSearchHooks,
  resolveEffectiveHooks,
  getTokenTermsForQuery,
  applyScoringHook,
  applyPostProcess,
  assertHooksSatisfied
} from './hooks';
import {
  clampLimit,
  throwIfAborted,
  abortError,
  nowMs
} from './guard';
import {
  normalizeCostBudgetOptions,
  throwIfBudgetAborted,
  assertTimeBudget,
  assertCandidateBudget,
  computeFilterSelectivity,
  isBroadQueryHeuristic,
  isBroadSelectivity,
  broadQueryRouteWarning,
  broadSelectivityWarning,
  candidateOverflowWarning
} from './diagnostics';
import {
  QUERY_TOKENS_MAX,
  SCORING_VERSION,
  SNAPSHOT_FORMAT_VERSION,
  UNICODE_VERSION,
  normalizeCpuScorer,
  DuplicateIdError,
  DocumentNotFoundError,
  IncompatibleIndexError,
  IncompatibleOptionError,
  ProfileMismatchError,
  QueryTooLongError,
  type TextProfileId
} from './text-profile';
import { DocumentBitset } from './filtering/doc-bitset';
import { ColumnarStore } from './filtering/columnar-store';
import { compileFilter } from './filtering/compile-filter';
import {
  FacetEngine,
  excludeFieldFromFilter,
  normalizeFacetRequests,
  type NormalizedFacet
} from './faceting/facet-engine';
import { alignHighlights, renderHighlightedText } from './highlight';
import {
  decodeSnapshot,
  restoreSnapshot,
  encodeSnapshot,
  type RestoredDocumentSnapshot
} from './snapshot-codec';
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
  SearchHooks,
  SearchMode,
  SearchTimings,
  SerializeDocumentIndexOptions,
  AutocompleteOptions,
  SuggestionItem,
  SuggestResponse,
  TieBreakerCriterion,
  DocumentIndexSchema,
  FacetResult,
  FilterExpression,
  FilterFieldDefinition,
  FilterFieldType,
  QueryDiagnostics,
  QueryDiagnosticsTimings
} from './types';

export interface InternalField<TDoc> extends FieldScoreDefinition {
  name: string;
  weight: number;
  getter: (doc: TDoc) => string | string[] | undefined | null;
  originalIndex: number;
}

/**
 * fail-closed custom filter-getter restore guard (mirrors the
 * hookIds idiom). Snapshots recording `hasGetter: true` for a filter field
 * require the caller to supply a matching getter override via
 * `options.options.filterFields`; otherwise columnar rebuild via default
 * `doc[name]` would silently change filter semantics.
 */
export function assertSnapshotFilterGettersSatisfied(
  snapshotFilterFields: Array<{ name: string; hasGetter?: boolean }> | undefined,
  userFilterFields: Array<string | { name: string; getter?: unknown }> | undefined
): void {
  if (!snapshotFilterFields || snapshotFilterFields.length === 0) return;
  const needed = snapshotFilterFields.filter((ff) => ff && ff.hasGetter === true);
  if (needed.length === 0) return;
  const userGetterNames = new Set<string>();
  if (Array.isArray(userFilterFields)) {
    for (const uf of userFilterFields) {
      if (uf && typeof uf === 'object' && typeof (uf as { name?: unknown }).name === 'string') {
        const u = uf as { name: string; getter?: unknown };
        if (typeof u.getter === 'function') userGetterNames.add(u.name);
      }
    }
  }
  for (const ff of needed) {
    if (!userGetterNames.has(ff.name)) {
      throw new IncompatibleIndexError(
        `filter-getter:${ff.name}`,
        'missing custom getter override for snapshot filter field (supply options.options.filterFields with getter)'
      );
    }
  }
}

/**
 * High-level multi-field document search index with dynamic mutations and highlighting.
 * Full document-index implementation.
 */
export class DocumentIndex<TDoc = Record<string, unknown>> {
  readonly options: DocumentIndexOptions<TDoc>;
  private readonly normalized: boolean;
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
  private tombstones: Set<number> = new Set(); // tombstone candidate filtering
  private searchMutex: Promise<any> = Promise.resolve();
  private generation: number = 0;
  private readonly initialCapacity: number;
  private readonly growthFactor: number;
  private columnarStore: ColumnarStore<TDoc>;
  private filterFieldDefinitions: FilterFieldDefinition<TDoc>[] = [];
  private indexHooks: SearchHooks<TDoc> | undefined = undefined;

  constructor(options: DocumentIndexOptions<TDoc>) {
    if (!options || !Array.isArray(options.fields) || options.fields.length === 0) {
      throw new TypeError('[webgpu-search] DocumentIndex expects options.fields to be a non-empty array.');
    }
    this.options = options;
    this.normalized = !(options.caseSensitive ?? false);
    this.preferGpu = options.preferGpu ?? false;
    // fail-closed extension hook validation at construction.
    this.indexHooks = normalizeSearchHooks(options.hooks ?? options.extensions);
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
      if (b.weight !== a.weight) return b.weight > a.weight ? 1 : -1;
      if (a.originalIndex !== b.originalIndex) return a.originalIndex > b.originalIndex ? 1 : -1;
      return 0;
    });

    for (let i = 0; i < this.sortedFields.length; i++) {
      this.fieldNameToIndex.set(this.sortedFields[i].name, i);
    }

    const seenFilterNames = new Set<string>();
    const normalizedFilterFields: FilterFieldDefinition<TDoc>[] = [];
    if (options.filterFields !== undefined) {
      if (!Array.isArray(options.filterFields)) {
        throw new TypeError('[webgpu-search] DocumentIndex expects options.filterFields to be an array.');
      }
      for (let i = 0; i < options.filterFields.length; i++) {
        const ff = options.filterFields[i];
        let name: string;
        let type: FilterFieldType | undefined = undefined;
        let getter: ((doc: TDoc) => any) | undefined = undefined;
        let hasGetter = false;

        if (typeof ff === 'string') {
          if (ff.trim().length === 0) {
            throw new TypeError('[webgpu-search] Filter field name string must not be empty.');
          }
          name = ff;
          getter = (doc: any) => doc[name];
          hasGetter = false;
        } else if (ff && typeof ff === 'object') {
          if (typeof ff.name !== 'string' || ff.name.trim().length === 0) {
            throw new TypeError('[webgpu-search] Filter field definition requires a non-empty string name.');
          }
          name = ff.name;
          type = ff.type;
          if (ff.getter !== undefined) {
            if (typeof ff.getter !== 'function') {
              throw new TypeError(`[webgpu-search] Filter field "${name}" getter must be a function.`);
            }
            getter = ff.getter;
            hasGetter = true;
          } else {
            getter = (doc: any) => doc[name];
            hasGetter = false;
          }
        } else {
          throw new TypeError('[webgpu-search] Filter field must be a string or FilterFieldDefinition object.');
        }

        if (seenFilterNames.has(name)) {
          throw new Error(`[webgpu-search] Duplicate filter field name: "${name}".`);
        }
        seenFilterNames.add(name);
        normalizedFilterFields.push(hasGetter ? { name, type, getter, hasGetter: true } : { name, type, getter });
      }
    }
    this.filterFieldDefinitions = normalizedFilterFields;
    this.columnarStore = new ColumnarStore<TDoc>(normalizedFilterFields, {
      initialCapacity: this.initialCapacity,
      growthFactor: this.growthFactor
    });
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
        '[webgpu-search] slotBytes is not supported (fixed slots removed).'
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
        const norm = normalizeText(rawStr, this.normalized);
        this.rowTokens[rowIdx] = norm.tokens;
        this.rowToDocIndex[rowIdx] = d;
        this.rowToFieldIndex[rowIdx] = f;
        this.docToRowIndices[d][f] = rowIdx;
        corpusTokens += norm.tokenCount;
        rowIdx++;
      }
    }

    this.totalTokens = corpusTokens;
    this.columnarStore.init(this.records);

    if (docCount === 0) {
      this.engineType = 'cpu';
      this.fallbackReason = this.options.preferGpu === false ? 'prefer-cpu' : 'below-threshold';
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
            const packed = packDataset(this.rowTokens, {
              normalized: this.normalized,
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

            this.unsubscribeDeviceLost = GpuDevicePool.onDeviceLost(() => {
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
            this.fallbackReason =
              typeof navigator === 'undefined' || !('gpu' in navigator) || !navigator.gpu
                ? 'webgpu-unsupported'
                : 'device-request-failed';
          }
        } catch (gpuErr) {
          console.warn('[webgpu-search] WebGPU initialization failed, falling back to CPU:', gpuErr);
          this.engineType = 'cpu';
          this.fallbackReason =
            typeof navigator === 'undefined' || !('gpu' in navigator) || !navigator.gpu
              ? 'webgpu-unsupported'
              : 'device-request-failed';
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

  /**
   * Validate search options fail-closed (extracted from `search`).
   * Covers query shape, mode/scorer/profile checks, token/prefix/typo
   * normalization, hooks, budget/diagnostics, query-length gates, facet,
   * ranking/autocomplete, field scoping, and tokenizer expansion.
   * No behavior change: code moved verbatim from `search`.
   */
  private validateSearchOptions(query: string, options: DocumentSearchOptions<TDoc>) {
    if (typeof query !== 'string') {
      throw new TypeError(`[webgpu-search] search expects query: string, got ${typeof query}.`);
    }

    const {
      mode = 'fuzzy',
      caseSensitive = false,
      signal,
      cpuAlgorithm: cpuAlgorithmOpt,
      cpuScorer: cpuScorerOpt,
      onQueryTooLong = 'throw',
      fields: searchFields,
      facets,
      faceting
    } = options;

    const requestedScorer = cpuScorerOpt ?? cpuAlgorithmOpt ?? 'exact';
    const cpuScorer = normalizeCpuScorer(requestedScorer as 'exact' | 'ufuzzy' | 'parity') ?? 'exact';
    const cpuAlgorithm = requestedScorer;
    if (caseSensitive === this.normalized) {
      throw new ProfileMismatchError(!this.normalized, caseSensitive);
    }
    if (this.preferGpu && cpuScorer === 'ufuzzy') {
      throw new IncompatibleOptionError(
        'cpuScorer',
        "cpuScorer:'ufuzzy' is CPU-only; use preferGpu:false or cpuScorer:'exact'."
      );
    }
    if (mode !== 'fuzzy' && mode !== 'substring' && mode !== 'token' && mode !== 'prefix') {
      throw new IncompatibleOptionError(
        'mode',
        `Unknown search mode '${String(mode)}'. Expected 'fuzzy', 'substring', 'token', or 'prefix'.`
      );
    }
    // fail-closed option validation up front so malformed
    // token/prefix/typo shapes throw identically on GPU and CPU paths.
    // 'fuzzy' validates typo shape but ignores it (subsequence matching is
    // inherently typo-tolerant); 'substring'/'token'/'prefix' honor it.
    const tokenOpts: NormalizedTokenMatchOptions = normalizeTokenMatchOptions(options.tokenMatch);
    const prefixOpts: NormalizedPrefixOptions = normalizePrefixOptions(options.prefixMatch);
    const typo: NormalizedTypoOptions = normalizeTypoTolerance(options.typoTolerance);
    if (mode === 'prefix' && options.prefixMatch?.exactCase !== undefined && prefixOpts.exactCase !== caseSensitive) {
      throw new ProfileMismatchError(caseSensitive, prefixOpts.exactCase, 'prefixMatch.exactCase');
    }
    // Legacy ufuzzy/native scorers only implement fuzzy/substring-exact.
    if (cpuScorer === 'ufuzzy' && (mode === 'token' || mode === 'prefix' || typo.enabled)) {
      throw new IncompatibleOptionError(
        'cpuScorer',
        `cpuScorer:'ufuzzy' supports only exact 'fuzzy'/'substring' modes without typo tolerance (got mode '${mode}'${typo.enabled ? ' with typoTolerance' : ''}). Use cpuScorer:'exact'.`
      );
    }
    // fail-closed extension hook validation up front so malformed
    // hook shapes throw identically on GPU and CPU paths (before early exits).
    const effectiveHooks = resolveEffectiveHooks(this.indexHooks, options.hooks ?? options.extensions);
    // Threaded into every exact CPU call below (single normalization).
    const cpuModeOptions: CpuModeOptions = {
      tokenMatch: { operator: tokenOpts.operator, minMatchCount: tokenOpts.minMatchCount },
      prefixMatch: prefixOpts.prefixLength !== undefined
        ? { prefixLength: prefixOpts.prefixLength, exactCase: prefixOpts.exactCase }
        : { exactCase: prefixOpts.exactCase },
      typoTolerance: {
        enabled: typo.enabled,
        maxDistance: typo.maxDistance,
        minWordLengthForOneTypo: typo.minWordLengthForOneTypo,
        minWordLengthForTwoTypos: typo.minWordLengthForTwoTypos,
        prefixExactLength: typo.prefixExactLength
      }
    };
    // fail-closed cost-budget + diagnostics validation up front so
    // malformed budgets throw identically on GPU and CPU paths and on empty
    // corpora/queries (before early exits). Enforcement is independent of the
    // diagnostics flag: budgets constrain even when telemetry is off.
    const budget = normalizeCostBudgetOptions(options.budget);
    if (options.diagnostics !== undefined && typeof options.diagnostics !== 'boolean') {
      throw new TypeError('[webgpu-search] options.diagnostics must be a boolean.');
    }
    const wantsDiagnostics = options.diagnostics === true;
    const needsClock = wantsDiagnostics || budget?.maxExecutionTimeMs !== undefined;
    const queryStartMs = needsClock ? nowMs() : 0;
    const diagWarnings: string[] = [];

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

    const normalizedQuery = normalizeText(query, this.normalized);
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
    throwIfBudgetAborted(budget);

    // Hoisted prefixLength check: fail-closed even on empty queries/corpora.
    // Skipped for empty queries to match the scorer's early noMatch.
    if (mode === 'prefix' && normalizedQuery.tokens.length > 0) {
      assertPrefixLengthForQuery(prefixOpts, normalizedQuery.tokens.length);
    }

    // custom tokenizer terms for 'token' mode (CPU-only).
    // Other modes ignore the tokenizer. Empty queries skip hook invocation
    // (no-match by design); non-empty token queries expand via the hook so
    // scoring and highlights share one term set (score-highlight symmetry).
    let customTokenTerms: Uint32Array[] | undefined = undefined;
    if (effectiveHooks?.tokenizer !== undefined && mode === 'token' && !normalizedQuery.isEmpty) {
      customTokenTerms = getTokenTermsForQuery(
        query,
        normalizedQuery.tokens,
        this.normalized,
        effectiveHooks.tokenizer
      );
      cpuModeOptions.tokenTermsOverride = customTokenTerms;
    }

    // (): validate facet contracts fail-fast, before early exits.
    // An empty facet list is equivalent to not requesting facets (absent key).
    // `faceting` is ignored unless facets are requested (no throw on absent).
    const facetSpec = normalizeFacetRequests(facets);
    const wantsFacets = facetSpec !== undefined && facetSpec.length > 0;
    let facetingMode: 'auto' | 'force-exact' = 'auto';
    if (wantsFacets && faceting !== undefined) {
      if (faceting !== 'auto' && faceting !== 'force-exact') {
        throw new TypeError(`[webgpu-search] Invalid faceting option: "${String(faceting)}". Must be 'auto' or 'force-exact'.`);
      }
      facetingMode = faceting;
    }
    const facetEngine = wantsFacets ? new FacetEngine<TDoc>(this.columnarStore) : null;
    if (wantsFacets && facetEngine && facetSpec) {
      // Fail-closed field validation up front: unknown facet fields and
      // range-on-non-number must throw even when an empty match set would
      // otherwise take an early noHits exit below.
      facetEngine.validateRequests(facetSpec);
    }

    // Inline autocomplete config: deterministic ranking hierarchy.
    // All scope validation (ranking, autocomplete, fields) is fail-closed before
    // early exits so malformed shapes throw identically on empty and
    // non-empty corpora/queries.
    if (options.ranking !== undefined && (typeof options.ranking !== 'object' || options.ranking === null || Array.isArray(options.ranking))) {
      throw new TypeError('[webgpu-search] options.ranking must be an object.');
    }
    const tieBreakers: TieBreakerCriterion[] = normalizeTieBreakers(
      options.ranking?.tieBreakers
    );
    let suggestSpec: NormalizedAutocompleteOptions | undefined = undefined;
    const suggestRaw = options.autocomplete ?? options.suggest;
    if (suggestRaw !== undefined && suggestRaw !== false) {
      suggestSpec = normalizeAutocompleteOptions(suggestRaw as AutocompleteOptions | boolean);
      // Inline autocomplete inherits the search ranking hierarchy unless the
      // autocomplete object carries its own explicit tieBreakers.
      const rawSuggest = suggestRaw as AutocompleteOptions;
      if (
        typeof rawSuggest === 'object' &&
        rawSuggest !== null &&
        rawSuggest.tieBreakers === undefined &&
        options.ranking?.tieBreakers !== undefined
      ) {
        suggestSpec.tieBreakers = [...tieBreakers];
      }
      if (suggestSpec.field !== undefined && !this.fieldNameToIndex.has(suggestSpec.field)) {
        throw new Error(`[webgpu-search] Unknown autocomplete field: "${suggestSpec.field}".`);
      }
    }
    // Hoisted search-field validation (fail-closed before empty-query,
    // empty-corpus, and empty-filter early exits below).
    let allowedFieldIndices: Set<number> | undefined = undefined;
    if (searchFields !== undefined) {
      if (!Array.isArray(searchFields)) {
        throw new TypeError('[webgpu-search] options.fields must be an array of string field names.');
      }
      if (searchFields.length !== 0) {
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
    }
    const hasEmptyFieldList = searchFields !== undefined && searchFields.length === 0;

    return {
      mode,
      caseSensitive,
      signal,
      cpuScorer,
      cpuAlgorithm,
      tokenOpts,
      prefixOpts,
      typo,
      effectiveHooks,
      cpuModeOptions,
      budget,
      wantsDiagnostics,
      queryStartMs,
      diagWarnings,
      normalizedQuery,
      clampedLimit,
      forceCpu,
      facetSpec,
      wantsFacets,
      facetingMode,
      facetEngine,
      tieBreakers,
      suggestSpec,
      allowedFieldIndices,
      hasEmptyFieldList,
      customTokenTerms
    };
  }

  /**
   * Compile structured/predicate filters (extracted from `search`).
   * Compiles `FilterExpression` to a bitset, measures `filteringMs`,
   * enforces abort/budget gates, and composes extension predicates.
   * Empty bitsets return early with `filteringMs: 0` so callers can take
   * the `noHits` path with identical telemetry. No behavior change.
   */
  private compileFilterPhase(
    filter: DocumentSearchOptions<TDoc>['filter'],
    effectiveHooks: SearchHooks<TDoc> | undefined,
    wantsDiagnostics: boolean,
    signal: AbortSignal | undefined,
    budget: ReturnType<typeof normalizeCostBudgetOptions>,
    queryStartMs: number
  ) {
    let filterBitset: DocumentBitset | undefined = undefined;
    let filterPredicate: ((doc: TDoc) => boolean) | undefined = undefined;
    let structuredFilter: FilterExpression | undefined = undefined;
    let filteringMs = 0;

    const tFilter0 = wantsDiagnostics ? nowMs() : 0;
    if (filter !== undefined) {
      if (typeof filter === 'function') {
        filterPredicate = filter;
      } else if (typeof filter === 'object' && filter !== null) {
        structuredFilter = filter as FilterExpression;
        filterBitset = compileFilter(structuredFilter, this.columnarStore);
        if (filterBitset.isEmpty()) {
          return { filterBitset, filterPredicate, structuredFilter, filteringMs };
        }
      } else {
        throw new TypeError('[webgpu-search] options.filter must be a function or FilterExpression.');
      }
    }
    if (wantsDiagnostics) filteringMs = nowMs() - tFilter0;
    throwIfAborted(signal);
    throwIfBudgetAborted(budget);
    assertTimeBudget(queryStartMs, budget);

    // extension filter predicates compose conjunctively (AND) with
    // `options.filter` functions. Index-level vs per-query hooks already
    // merged via `resolveEffectiveHooks` (per-query wins per key).
    const extensionPredicate = effectiveHooks?.filterPredicate;
    if (extensionPredicate !== undefined) {
      const basePredicate = filterPredicate;
      if (basePredicate === undefined) {
        filterPredicate = extensionPredicate;
      } else {
        const optFn = basePredicate;
        const extFn = extensionPredicate;
        filterPredicate = (doc: TDoc) => {
          const d = doc as TDoc;
          return optFn(d) && extFn(d);
        };
      }
    }

    return { filterBitset, filterPredicate, structuredFilter, filteringMs };
  }

  /**
   * Route to WebGPU vs CPU (extracted from `search`).
   * Applies candidate-budget enforcement, broad-query CPU routing, and
   * field-restriction routing. Mutates `diagWarnings` for the broad-query
   * warning exactly as before. No behavior change.
   */
  private routeEngine(params: {
    mode: SearchMode;
    typoEnabled: boolean;
    cpuScorer: string;
    forceCpu: boolean;
    allowedFieldIndices: Set<number> | undefined;
    normalizedQueryTokenCount: number;
    filterBitset: DocumentBitset | undefined;
    budget: ReturnType<typeof normalizeCostBudgetOptions>;
    queryStartMs: number;
    wantsDiagnostics: boolean;
    diagWarnings: string[];
  }) {
    const {
      mode,
      typoEnabled,
      cpuScorer,
      forceCpu,
      allowedFieldIndices,
      normalizedQueryTokenCount,
      filterBitset,
      budget,
      queryStartMs,
      wantsDiagnostics,
      diagWarnings
    } = params;
    // token/prefix modes and typo-tolerant queries are CPU-only
    // (exact-only WGSL kernels) and skip GPU dispatch with fallbackReason
    // 'unsupported-mode' (parity boundary).
    const isGpuSupportedMode: boolean =
      (mode === 'fuzzy' || mode === 'substring') && !typoEnabled;

    // broad-query pre-dispatch guard + candidate ceiling.
    // `filteredCandidateCount` is the exact post-filter population to score;
    // exceeding `maxCandidates` throws before any scoring work. Short queries
    // over massive corpora route to the CPU streaming scan to avoid GPU
    // buffer saturation and driver timeouts (TDR). With function-predicate
    // filters the pre-predicate population is checked (conservative — the
    // true post-predicate count is unknowable pre-scan).
    const activeDocCount = this.idToDocIndex.size;
    const filteredCandidateCount = filterBitset !== undefined
      ? filterBitset.popcount()
      : activeDocCount;
    assertCandidateBudget(filteredCandidateCount, budget);
    let broadQueryCpuRoute = false;
    if (isBroadQueryHeuristic(normalizedQueryTokenCount, activeDocCount)) {
      broadQueryCpuRoute = true;
      // Suppress the GPU-avoidance warning when already CPU-by-design
      // (token/prefix/typo modes, explicit ufuzzy, query-too-long fallback):
      // routing outcome is correct but the stated cause would misattribute.
      const cpuByDesign = !isGpuSupportedMode || cpuScorer === 'ufuzzy' || forceCpu;
      if (wantsDiagnostics && !cpuByDesign) {
        diagWarnings.push(broadQueryRouteWarning(activeDocCount, normalizedQueryTokenCount));
      }
    }
    assertTimeBudget(queryStartMs, budget);

    // When field-restricted, route to CPU to prevent unselected high-priority
    // fields from saturating GPU candidate buffer before low-priority allowed fields.
    const isFieldRestricted = allowedFieldIndices !== undefined && allowedFieldIndices.size < this.sortedFields.length;

    // 1. WebGPU execution path (exact fuzzy/substring only; token/prefix
    // and typo queries skip dispatch via isGpuSupportedMode above).
    const gpuHandle = this.gpuEngine;
    const useGpu = !forceCpu &&
      !isFieldRestricted &&
      !broadQueryCpuRoute &&
      isGpuSupportedMode &&
      cpuScorer !== 'ufuzzy' &&
      this.engineType === 'webgpu' &&
      gpuHandle !== null &&
      gpuHandle.isReady;

    return {
      isGpuSupportedMode,
      broadQueryCpuRoute,
      isFieldRestricted,
      filteredCandidateCount,
      useGpu,
      gpuHandle
    };
  }

  /**
   * Aggregate per-row matches into per-document best-field hits (extracted).
   * Shared by the WebGPU readback and legacy ufuzzy/native CPU paths which
   * previously duplicated this ranking block. Deterministic sort via
   * `compareRanked` + `tieBreakers`. No behavior change.
   */
  private aggregateDocMatches(
    docMatches: Map<number, { bestScore: number; bestFieldIdx: number; fieldScores: Map<number, number> }>,
    normalizedQueryTokens: Uint32Array,
    tieBreakers: TieBreakerCriterion[]
  ) {
    interface RankedDocCandidate {
      dIdx: number;
      item: DocumentSearchResultItem<TDoc>;
      rank: RankableCandidate;
    }
    const hits: RankedDocCandidate[] = [];
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
          if (b.score !== a.score) return b.score > a.score ? 1 : -1;
          return a.field < b.field ? -1 : (a.field > b.field ? 1 : 0);
        });
      }

      const bestRowIdx = this.docToRowIndices[dIdx]?.[entry.bestFieldIdx];
      const bestRowTokens = bestRowIdx !== undefined ? this.rowTokens[bestRowIdx] : undefined;
      hits.push({
        dIdx,
        item: {
          id: this.docIds[dIdx],
          doc: this.records[dIdx],
          score: entry.bestScore,
          matchedField: primaryField.name,
          matches: auxMatches.length > 0 ? auxMatches : undefined
        },
        rank: {
          score: entry.bestScore,
          fieldWeight: primaryField.weight,
          isExactMatch: bestRowTokens !== undefined
            ? isExactTokenMatch(bestRowTokens, normalizedQueryTokens)
            : false,
          matchedLength: bestRowTokens !== undefined ? bestRowTokens.length : 0,
          id: this.docIds[dIdx],
          docIndex: dIdx
        }
      });
    }

    // deterministic ranking: score DESC, weight DESC, exact DESC,
    // length ASC, id ASC (docIndex ASC implicit fallback).
    hits.sort((a, b) => compareRanked(a.rank, b.rank, tieBreakers));
    return hits;
  }

  /**
   * Enrich Top-K results with highlights + post-match hooks (extracted).
   * Runs highlight enrichment, scoring-hook boost + deterministic re-sort,
   * and postProcess transform. Returns updated results with phase timings.
   * No behavior change.
   */
  private enrichResults(
    results: DocumentSearchResultItem<TDoc>[],
    query: string,
    mode: SearchMode,
    options: DocumentSearchOptions<TDoc>,
    customTokenTerms: Uint32Array[] | undefined,
    normalizedQueryTokens: Uint32Array,
    effectiveHooks: SearchHooks<TDoc> | undefined,
    tieBreakers: TieBreakerCriterion[],
    wantsDiagnostics: boolean
  ) {
    const tHl0 = wantsDiagnostics ? nowMs() : 0;
    this.enrichHighlights(results, query, mode, options, customTokenTerms, normalizedQueryTokens);
    const highlightMs = wantsDiagnostics ? nowMs() - tHl0 : 0;

    // post-match extension pipeline (Top-K only). Scoring boosts
    // apply identically on GPU and CPU paths (parity preserved), followed
    // by a deterministic re-sort; postProcess is the final transform.
    // hook time joins the scoring bucket on all paths for comparability.
    const tHook0 = wantsDiagnostics ? nowMs() : 0;
    if (effectiveHooks?.scoringHook !== undefined) {
      applyScoringHook(results, effectiveHooks.scoringHook, query);
      this.resortResultsAfterScoring(results, normalizedQueryTokens, tieBreakers);
    }
    let out = results;
    if (effectiveHooks?.postProcess !== undefined) {
      out = applyPostProcess(results, effectiveHooks.postProcess);
    }
    const hookMs = wantsDiagnostics ? nowMs() - tHook0 : 0;
    return { results: out, highlightMs, hookMs };
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

    const validated = this.validateSearchOptions(query, options);
    const {
      mode,
      caseSensitive,
      signal,
      cpuScorer,
      cpuAlgorithm,
      typo,
      effectiveHooks,
      cpuModeOptions,
      budget,
      wantsDiagnostics,
      queryStartMs,
      diagWarnings,
      normalizedQuery,
      clampedLimit,
      forceCpu,
      facetSpec,
      wantsFacets,
      facetingMode,
      facetEngine,
      tieBreakers,
      suggestSpec,
      allowedFieldIndices,
      hasEmptyFieldList,
      customTokenTerms
    } = validated;

    const filterState = this.compileFilterPhase(
      options.filter,
      effectiveHooks,
      wantsDiagnostics,
      signal,
      budget,
      queryStartMs
    );
    const { filterBitset, filterPredicate, structuredFilter, filteringMs } = filterState;
    /**
     * assemble response diagnostics (undefined unless requested).
     * `filteringMs` is read at call time so early exits after compilation
     * report measured filter cost; `totalMs` is wall-clock from query entry
     * (end-to-end including inline autocomplete; see `autocompleteMs` for the slice).
     * Scoring includes post-match scoring-hook time on all paths.
     */
    const buildDiagnostics = (
      routedEngine: EngineType,
      scoringMs: number,
      highlightMs: number,
      facetingMs: number | undefined,
      hasOverflow: boolean,
      suggestMs?: number
    ): QueryDiagnostics | undefined => {
      if (!wantsDiagnostics) return undefined;
      const scanned = this.idToDocIndex.size;
      const selectivity = filterBitset !== undefined
        ? computeFilterSelectivity(filterBitset.popcount(), scanned)
        : 1.0;
      const totalMs = nowMs() - queryStartMs;
      const timings: QueryDiagnosticsTimings = {
        filteringMs,
        scoringMs,
        highlightMs,
        ...(facetingMs !== undefined ? { facetingMs } : {}),
        ...(suggestMs !== undefined ? { autocompleteMs: suggestMs, suggestMs } : {}),
        totalMs
      };
      const diag: QueryDiagnostics = {
        scannedCandidates: scanned,
        filterSelectivity: selectivity,
        routedEngine,
        hasOverflow,
        timings
      };
      if (diagWarnings.length > 0) diag.warnings = [...diagWarnings];
      return diag;
    };
    /**
     * post-hoc broad-query + overflow warnings (non-fatal, gated on
     * `diagnostics:true` — routing decisions stay ungated for safety).
     * `rawTotalMatches` carries the pre-filter pool count on the GPU path so
     * the overflow sentence stays accurate; `facetsExact` suppresses the
     * stale "approximate" claim after a force-exact rescan.
     */
    const pushPostHocWarnings = (
      totalMatches: number,
      hasOverflow: boolean,
      opts?: { rawTotalMatches?: number; facetsRequested?: boolean; facetsExact?: boolean }
    ): void => {
      if (!wantsDiagnostics) return;
      const scanned = this.idToDocIndex.size;
      const selectivity = scanned > 0 ? totalMatches / scanned : 0;
      if (isBroadSelectivity(selectivity, scanned)) {
        diagWarnings.push(broadSelectivityWarning(selectivity, scanned));
      }
      if (hasOverflow) {
        diagWarnings.push(
          candidateOverflowWarning(totalMatches, this.candidateCapacity, {
            facetsRequested: opts?.facetsRequested ?? false,
            facetsExact: opts?.facetsExact ?? false,
            rawTotalMatches: opts?.rawTotalMatches,
          })
        );
      }
    };

    const noHits = (q: string): DocumentSearchResponse<TDoc> => {
      // even trivial exits honor caller aborts and time budgets.
      throwIfAborted(signal);
      throwIfBudgetAborted(budget);
      assertTimeBudget(queryStartMs, budget);
      const diag = buildDiagnostics('cpu', 0, 0, wantsFacets ? 0 : undefined, false);
      // Public totalMs is scorer wall-clock (0 on no-hit: nothing scored);
      // diagnostics.totalMs carries the wall-clock including validation.
      return {
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
        cpuScorer,
        cpuAlgorithm,
        fallbackReason: forceCpu ? 'query-too-long' : this.fallbackReason,
        ...(wantsFacets && facetEngine && facetSpec
          ? { facets: facetEngine.emptyResults(facetSpec) }
          : {}),
        ...(diag ? { diagnostics: diag } : {})
      };
    };

    if (filterBitset !== undefined && filterBitset.isEmpty()) {
      return noHits(query);
    }

    if (normalizedQuery.isEmpty) {
      return noHits('');
    }
    if (this.idToDocIndex.size === 0) {
      return noHits(query);
    }
    if (hasEmptyFieldList) {
      return noHits(query);
    }

    const route = this.routeEngine({
      mode,
      typoEnabled: typo.enabled,
      cpuScorer,
      forceCpu,
      allowedFieldIndices,
      normalizedQueryTokenCount: normalizedQuery.tokens.length,
      filterBitset,
      budget,
      queryStartMs,
      wantsDiagnostics,
      diagWarnings
    });
    const { broadQueryCpuRoute, isGpuSupportedMode, useGpu, gpuHandle } = route;

    if (useGpu && gpuHandle !== null) {
      // narrow GPU try to dispatch only so extension hook
      // errors (filterPredicate / scoringHook / postProcess) propagate
      // directly without destroying the engine or double-invoking via CPU
      // fallback. Only dispatch failures fall through to CPU.
      let gpuResult: Awaited<ReturnType<WebGPUEngine['search']>> | undefined = undefined;
      try {
        gpuResult = await gpuHandle.search(query, {
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
        // GPU dispatch is async wall-clock work — enforce the caller
        // deadline and budget abort before touching the readback.
        throwIfBudgetAborted(budget);
        assertTimeBudget(queryStartMs, budget);
      } catch (err: any) {
        if (err.name === 'AbortError') {
          throw err;
        }
        // Mirror search-index: mode-routed rejections (token/prefix/typo via
        // direct or raced calls) are not engine failures — keep a healthy GPU
        // up and fall through to exact CPU.
        const modeRouted: boolean =
          err instanceof IncompatibleOptionError && (err.option === 'mode' || err.option === 'typoTolerance');
        if (!modeRouted) {
          console.warn('[webgpu-search] GPU document search failed, falling back to CPU:', err);
          this.engineType = 'cpu';
          this.fallbackReason = 'gpu-execution-error';
          if (this.gpuEngine) {
            try { this.gpuEngine.destroy(); } catch {}
            this.gpuEngine = null;
          }
        }
        gpuResult = undefined;
      }

      if (gpuResult !== undefined) {
        const gpuResultOk = gpuResult;

        // WebGPU candidate readback & fixed-point score enrichment
        const docMatches = new Map<number, {
          bestScore: number;
          bestFieldIdx: number;
          fieldScores: Map<number, number>;
        }>();
        // unfiltered query match set for facet aggregation (filter applied below).
        const queryMatchedAll = wantsFacets ? new Set<number>() : null;

        // hook errors propagate (no fallback / no double-invoke).
        // Only GPU dispatch failures fall back; readback + hook pipeline runs
        // outside the dispatch try above.
        for (let i = 0; i < gpuResultOk.results.length; i++) {
          const item = gpuResultOk.results[i];
          const r = item.index;
          if (this.tombstones.has(r)) continue;
          const fIdx = this.rowToFieldIndex[r];
          if (allowedFieldIndices && !allowedFieldIndices.has(fIdx)) continue;
          const dIdx = this.rowToDocIndex[r];
          if (queryMatchedAll !== null && this.records[dIdx]) {
            queryMatchedAll.add(dIdx);
          }
          if (filterBitset && !filterBitset.has(dIdx)) continue;
          const doc = this.records[dIdx];
          if (!doc) continue;
          if (filterPredicate && !filterPredicate(doc)) continue;

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

        const hits = this.aggregateDocMatches(docMatches, normalizedQuery.tokens, tieBreakers);

        const totalMatches = hits.length;
        const candidateCount = Math.min(totalMatches, this.candidateCapacity);
        let results = hits.slice(0, clampedLimit).map((h) => h.item);
        const gpuEnriched = this.enrichResults(results, query, mode, options, customTokenTerms, normalizedQuery.tokens, effectiveHooks, tieBreakers, wantsDiagnostics);
        results = gpuEnriched.results;
        const gpuHighlightMs = gpuEnriched.highlightMs;
        const gpuHookMs = gpuEnriched.hookMs;

        // facet aggregation. Exact when the GPU pool covered all matches;
        // approximate over the top pool on overflow unless force-exact rescan.
        let gpuFacets: Record<string, FacetResult> | undefined = undefined;
        let gpuFacetMs: number | undefined = undefined;
        let gpuFacetExact = true;
        if (wantsFacets && facetEngine && facetSpec && queryMatchedAll !== null) {
          const facetT0 = nowMs();
          let facetCandidates: Iterable<number> = queryMatchedAll;
          let facetIsApproximate = gpuResultOk.hasOverflow;
          if (gpuResultOk.hasOverflow && facetingMode === 'force-exact') {
            const exact = scoreExactMatchesMultiField(
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
              undefined,
              true,
              cpuModeOptions
            );
            facetCandidates = exact.allMatchedDocIndices ?? exact.results.map((r) => r.docIndex);
            facetIsApproximate = false;
          }
          gpuFacets = this.buildFacetResults(
            facetEngine, facetSpec, facetCandidates, structuredFilter, filterPredicate, facetIsApproximate
          );
          const facetMs = nowMs() - facetT0;
          gpuFacetMs = facetMs;
          gpuFacetExact = !facetIsApproximate;
        }

        // deadline + post-hoc warnings before returning GPU results.
        // Inline autocomplete runs before the diagnostics snapshot so
        // diagnostics.totalMs covers end-to-end latency (autocomplete bucket).
        const tGpuSuggest0 = wantsDiagnostics && suggestSpec ? nowMs() : 0;
        const gpuSuggestions = suggestSpec
          ? this.computeSuggestions(query, normalizedQuery.tokens, suggestSpec, allowedFieldIndices)
          : undefined;
        const gpuSuggestMs = wantsDiagnostics && suggestSpec ? nowMs() - tGpuSuggest0 : undefined;
        throwIfAborted(signal);
        throwIfBudgetAborted(budget);
        assertTimeBudget(queryStartMs, budget);
        pushPostHocWarnings(totalMatches, gpuResultOk.hasOverflow, {
          rawTotalMatches: gpuResultOk.totalMatches,
          facetsRequested: wantsFacets,
          facetsExact: gpuFacetExact,
        });
        // Scoring = engine scan + post-match hooks (facet time excluded —
        // no double-counting, no mutation of the engine timings object).
        const gpuEngineTotal = gpuResultOk.timings.totalMs;
        const gpuDiag = buildDiagnostics(
          'webgpu',
          gpuEngineTotal + gpuHookMs,
          gpuHighlightMs,
          gpuFacetMs,
          gpuResultOk.hasOverflow,
          gpuSuggestMs
        );

        return {
          query: gpuResultOk.query,
          mode: gpuResultOk.mode,
          engine: 'webgpu',
          totalMatches,
          candidateCount,
          hasOverflow: gpuResultOk.hasOverflow,
          results,
          timings: {
            ...gpuResultOk.timings,
            totalMs: gpuEngineTotal + (gpuFacetMs ?? 0) + gpuHighlightMs + gpuHookMs,
          },
          profileId: this.profileId,
          scoringVersion: SCORING_VERSION,
          cpuScorer,
          cpuAlgorithm,
          ...(gpuFacets ? { facets: gpuFacets } : {}),
          ...(gpuDiag ? { diagnostics: gpuDiag } : {}),
          ...(gpuSuggestions ? { suggestions: gpuSuggestions } : {})
        };
      }
    }

    // 2. CPU execution path
    if (this.isDestroyed || this.generation !== gen) {
      throw abortError();
    }
    throwIfAborted(signal);
    throwIfBudgetAborted(budget);
    assertTimeBudget(queryStartMs, budget);
    const t0 = nowMs();

    if (cpuScorer === 'ufuzzy') {
      const cpuEngine = new CPUEngine();
      const docMatches = new Map<number, {
        bestScore: number;
        bestFieldIdx: number;
        fieldScores: Map<number, number>;
      }>();
      // unfiltered query match set for exact facet aggregation.
      const legacyMatchedAll = wantsFacets ? new Set<number>() : null;

      for (let fIdx = 0; fIdx < this.sortedFields.length; fIdx++) {
        if (allowedFieldIndices && !allowedFieldIndices.has(fIdx)) continue;
        const fDef = this.sortedFields[fIdx];
        const fieldStrings = this.records.map((doc, dIdx) => (doc && this.rawFieldStrings[dIdx] ? this.rawFieldStrings[dIdx][fIdx] : ''));
        const legacyResult = mode === 'fuzzy'
          ? cpuEngine.searchUFuzzy(fieldStrings, query, this.records.length, caseSensitive)
          : cpuEngine.searchNative(fieldStrings, query, this.records.length, caseSensitive);

        for (const hit of legacyResult.results) {
          const dIdx = hit.index;
          // Explicit tombstone guard for symmetry with GPU/exact paths:
          // removed docs have null records (columnar presence also cleared).
          if (!this.records[dIdx] || this.docIds[dIdx] === null || this.docIds[dIdx] === undefined) continue;
          if (legacyMatchedAll !== null) {
            legacyMatchedAll.add(dIdx);
          }
          if (filterBitset && !filterBitset.has(dIdx)) continue;
          const doc = this.records[dIdx];
          if (!doc) continue;
          if (filterPredicate && !filterPredicate(doc)) continue;
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

      const hits = this.aggregateDocMatches(docMatches, normalizedQuery.tokens, tieBreakers);

      const totalMatches = hits.length;
      const candidateCount = Math.min(totalMatches, this.candidateCapacity);
      let results = hits.slice(0, clampedLimit).map((h) => h.item);
      const legacyEnriched = this.enrichResults(results, query, mode, options, customTokenTerms, normalizedQuery.tokens, effectiveHooks, tieBreakers, wantsDiagnostics);
      results = legacyEnriched.results;
      const legacyHighlightMs = legacyEnriched.highlightMs;
      const preFacetMs = nowMs() - t0;
      // scoring bucket covers scan + ranking + post-match hooks;
      // highlight enrichment is metered separately above.
      const legacyScoringMs = Math.max(0, preFacetMs - legacyHighlightMs);

      // CPU evaluates the full match set, so legacy facets are exact
      // w.r.t. the serving (ufuzzy/native) match set; see types for caveat.
      let legacyFacets: Record<string, FacetResult> | undefined = undefined;
      let legacyFacetMs = 0;
      if (wantsFacets && facetEngine && facetSpec && legacyMatchedAll !== null) {
        const fT0 = nowMs();
        legacyFacets = this.buildFacetResults(
          facetEngine, facetSpec, legacyMatchedAll, structuredFilter, filterPredicate, false
        );
        legacyFacetMs = nowMs() - fT0;
      }

      // inline autocomplete precedes the diagnostics snapshot so
      // diagnostics.totalMs covers end-to-end latency (autocompleteMs bucket).
      const tLegacySuggest0 = wantsDiagnostics && suggestSpec ? nowMs() : 0;
      const legacySuggestions = suggestSpec
        ? this.computeSuggestions(query, normalizedQuery.tokens, suggestSpec, allowedFieldIndices)
        : undefined;
      const legacySuggestMs = wantsDiagnostics && suggestSpec ? nowMs() - tLegacySuggest0 : undefined;
      throwIfAborted(signal);
      throwIfBudgetAborted(budget);
      assertTimeBudget(queryStartMs, budget);
      pushPostHocWarnings(totalMatches, totalMatches > this.candidateCapacity, {
        facetsRequested: wantsFacets,
        facetsExact: true,
      });
      const legacyDiag = buildDiagnostics(
        'cpu',
        legacyScoringMs,
        legacyHighlightMs,
        wantsFacets ? legacyFacetMs : undefined,
        totalMatches > this.candidateCapacity,
        legacySuggestMs
      );

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
          totalMs: preFacetMs + legacyFacetMs,
          gpuDispatchMs: 0
        },
        profileId: this.profileId,
        scoringVersion: SCORING_VERSION,
        cpuScorer,
        cpuAlgorithm,
        fallbackReason: 'cpu-algorithm-requested',
        ...(legacyFacets ? { facets: legacyFacets } : {}),
        ...(legacyDiag ? { diagnostics: legacyDiag } : {}),
        ...(legacySuggestions ? { suggestions: legacySuggestions } : {})
      };
    }

    // Default exact CPU algorithm (collect full match set only when faceting).
    // passes the deterministic hierarchy + doc IDs so pre-truncation order
    // already reflects (score, weight, exact, length, id).
    const parityResult = scoreExactMatchesMultiField(
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
      (filterBitset || filterPredicate) ? (dIdx) => {
        if (filterBitset && !filterBitset.has(dIdx)) return false;
        if (filterPredicate) {
          const doc = this.records[dIdx];
          return doc !== null && doc !== undefined && filterPredicate(doc);
        }
        return true;
      } : undefined,
      wantsFacets,
      cpuModeOptions,
      { tieBreakers, docIds: this.docIds as unknown as ReadonlyArray<string | number | null | undefined> }
    );

    if (this.isDestroyed || this.generation !== gen) {
      throw abortError();
    }
    throwIfAborted(signal);
    throwIfBudgetAborted(budget);
    assertTimeBudget(queryStartMs, budget);

    const idProp = typeof this.options.idField === 'string' ? this.options.idField : 'id';
    let enrichedResults: DocumentSearchResultItem<TDoc>[] = [];
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
    const parityEnriched = this.enrichResults(enrichedResults, query, mode, options, customTokenTerms, normalizedQuery.tokens, effectiveHooks, tieBreakers, wantsDiagnostics);
    enrichedResults = parityEnriched.results;
    const parityHighlightMs = parityEnriched.highlightMs;
    const parityHookMs = parityEnriched.hookMs;
    const parityScoringMs = parityResult.durationMs + parityHookMs;

    // exact facet aggregation. Disjunctive facets need the unfiltered
    // query match set, so a structured filter triggers one extra unfiltered
    // parity scan (results above stay filtered and correctly ranked; O(F*n)
    // per-facet re-evaluation + F bitset allocs by design for disjunctive
    // semantics). Wall-clock facet work is included in totalMs; future
    // attribute facet work to diagnostics.timings.facetingMs.
    let parityFacets: Record<string, FacetResult> | undefined = undefined;
    const facetT0 = nowMs();
    let facetScanMs = 0;
    if (wantsFacets && facetEngine && facetSpec) {
      let facetCandidates: Iterable<number> =
        parityResult.allMatchedDocIndices ?? parityResult.results.map((r) => r.docIndex);
      if (structuredFilter !== undefined) {
        const unfiltered = scoreExactMatchesMultiField(
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
          undefined,
          true,
          cpuModeOptions
        );
        facetCandidates = unfiltered.allMatchedDocIndices ?? unfiltered.results.map((r) => r.docIndex);
        facetScanMs = unfiltered.durationMs;
      }
      parityFacets = this.buildFacetResults(
        facetEngine, facetSpec, facetCandidates, structuredFilter, filterPredicate, false
      );
    }
    const facetMs = wantsFacets ? Math.max(0, nowMs() - facetT0 - facetScanMs) : 0;
    // the disjunctive unfiltered rescan serves facets, so it joins the
    // faceting bucket (scoring stays scan + post-match hooks).
    const parityFacetingMs = wantsFacets ? facetScanMs + facetMs : undefined;

    // Public totalMs is scorer wall-clock: scan + hooks + highlight + facets
    // (excludes inline autocomplete; diagnostics.totalMs is end-to-end).
    const timings: SearchTimings = {
      queryUploadMs: 0,
      encodeSubmitMs: 0,
      gpuExecutionMs: null,
      readbackMs: 0,
      totalMs: parityScoringMs + parityHighlightMs + facetScanMs + facetMs,
      gpuDispatchMs: 0
    };

    let effectiveFallbackReason = this.fallbackReason;
    if (forceCpu) {
      effectiveFallbackReason = 'query-too-long';
    } else if (!isGpuSupportedMode) {
      // token/prefix/typo queries are CPU-by-design (exact-only
      // WGSL kernels) — recorded per the parity boundary.
      effectiveFallbackReason = 'unsupported-mode';
    } else if (useGpu && gpuHandle !== null) {
      effectiveFallbackReason = 'gpu-execution-error';
    } else if (broadQueryCpuRoute) {
      // broad-query CPU routing is a routing decision (like
      // field-restriction below), not a scorer request — leave the reason
      // as-is and surface the decision in diagnostics.warnings instead.
    }
    // Note: field-restricted routing (isFieldRestricted) intentionally leaves
    // fallbackReason as-is — it is a routing decision, not a scorer request,
    // so reusing 'cpu-algorithm-requested' would mislead telemetry.

    // inline autocomplete precedes the diagnostics snapshot so
    // diagnostics.totalMs covers end-to-end latency (autocompleteMs bucket).
    const tParitySuggest0 = wantsDiagnostics && suggestSpec ? nowMs() : 0;
    const paritySuggestions = suggestSpec
      ? this.computeSuggestions(query, normalizedQuery.tokens, suggestSpec, allowedFieldIndices)
      : undefined;
    const paritySuggestMs = wantsDiagnostics && suggestSpec ? nowMs() - tParitySuggest0 : undefined;
    throwIfAborted(signal);
    throwIfBudgetAborted(budget);
    assertTimeBudget(queryStartMs, budget);
    pushPostHocWarnings(parityResult.totalMatches, parityResult.hasOverflow, {
      facetsRequested: wantsFacets,
      facetsExact: true,
    });
    const parityDiag = buildDiagnostics(
      'cpu',
      parityScoringMs,
      parityHighlightMs,
      parityFacetingMs,
      parityResult.hasOverflow,
      paritySuggestMs
    );

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
      cpuScorer,
      cpuAlgorithm,
      fallbackReason: effectiveFallbackReason,
      ...(parityFacets ? { facets: parityFacets } : {}),
      ...(parityDiag ? { diagnostics: parityDiag } : {}),
      ...(paritySuggestions ? { suggestions: paritySuggestions } : {})
    };
  }

  /**
   * First-party autocomplete / did-you-mean primitive.
   *
   * Scans active documents over the autocomplete field (or all search fields),
   * scores each field row with the prefix scorer (`mode: 'prefix'`) or the
   * fuzzy scorer (`mode: 'fuzzy'`; typo-tolerant substring when
   * `fuzzyDistance > 0`), ranks with the deterministic comparator, and
   * returns the top `limit` completions with Unicode-safe highlight ranges.
   * `prefix` yields `type: 'completion'` (including typo-tolerant prefix);
   * `fuzzy` yields `type: 'did-you-mean'`.
   *
   * Suggestions are index-wide by design: search filters (structured or
   * predicate) narrow search results but never suggestion candidates, so
   * inline `search({ filter, autocomplete })` suggestions match standalone
   * `autocomplete()` output for the same query and options. `search.fields`
   * does scope suggestions unless the autocomplete spec carries an explicit
   * `field`, which wins silently — pass `autocomplete.field` to pin the scan.
   * Suggestion ranking uses `autocomplete.tieBreakers` when provided, else the
   * inline search `ranking.tieBreakers`, else the default; granularity
   * is per (doc, field) row (a document may appear multiple times), unlike
   * per-document best-field-wins search ranking.
   *
   * Autocomplete uses default `prefixMatch` options and only `fuzzyDistance` for
   * typo tolerance — the search `prefixMatch`/`typoTolerance` are ignored.
   *
   * Latency is a single O(docs x fields) scan plus a full sort of the match
   * set (unbounded pre-truncation by design; highlights are bounded to
   * top-K). Measured ~35ms prefix / ~95ms fuzzy+d2 over 50k rows (25k docs
   * x 2 fields); inline `search({ autocomplete })` pays both scans (~2x). There
   * is no `AbortSignal` support on this path; per-keystroke callers should
   * debounce. Queries longer than `QUERY_TOKENS_MAX` throw `QueryTooLongError`.
   */
  async autocomplete(
    query: string,
    options: AutocompleteOptions = {}
  ): Promise<SuggestResponse<TDoc>> {
    if (this.isDestroyed) {
      throw new Error('[webgpu-search] DocumentIndex has been destroyed.');
    }
    if (typeof query !== 'string') {
      throw new TypeError(`[webgpu-search] autocomplete expects query: string, got ${typeof query}.`);
    }
    const spec = normalizeAutocompleteOptions(options);
    if (spec.field !== undefined && !this.fieldNameToIndex.has(spec.field)) {
      throw new Error(`[webgpu-search] Unknown autocomplete field: "${spec.field}".`);
    }
    const t0 = nowMs();
    const normalized = normalizeText(query, this.normalized);
    if (normalized.tokenCount > QUERY_TOKENS_MAX) {
      throw new QueryTooLongError(QUERY_TOKENS_MAX, normalized.tokenCount, this.profileId);
    }
    if (normalized.isEmpty || this.idToDocIndex.size === 0) {
      return { suggestions: [], queryDurationMs: nowMs() - t0 };
    }
    const allowed = spec.field !== undefined
      ? new Set<number>([this.fieldNameToIndex.get(spec.field) as number])
      : undefined;
    const suggestions = this.computeSuggestions(query, normalized.tokens, spec, allowed);
    return { suggestions, queryDurationMs: nowMs() - t0 };
  }

  /** @deprecated Use autocomplete. */
  async suggest(
    query: string,
    options: AutocompleteOptions = {}
  ): Promise<SuggestResponse<TDoc>> {
    return this.autocomplete(query, options);
  }

  /**
   * Synchronous suggestion enumeration shared by `autocomplete()` and inline
   * `search({ autocomplete })`. Callers pass pre-normalized query tokens and a
   * validated spec; `searchFields` scopes the scan when the autocomplete spec
   * carries no explicit field.
   */
  private computeSuggestions(
    query: string,
    queryTokens: Uint32Array,
    spec: NormalizedAutocompleteOptions,
    searchFields?: ReadonlySet<number>
  ): SuggestionItem<TDoc>[] {
    let fieldIndices: number[];
    if (spec.field !== undefined) {
      fieldIndices = [this.fieldNameToIndex.get(spec.field) as number];
    } else if (searchFields !== undefined) {
      fieldIndices = [...searchFields];
    } else {
      fieldIndices = this.sortedFields.map((_, i) => i);
    }

    const typoForSuggest = normalizeTypoTolerance(
      spec.fuzzyDistance > 0
        ? { enabled: true, maxDistance: spec.fuzzyDistance as 1 | 2 }
        : undefined
    );
    const prefixOpts = normalizePrefixOptions(undefined);

    interface SuggestHit {
      keys: RankableCandidate;
      text: string;
      dIdx: number;
      fIdx: number;
      rawStr: string;
    }
    const candidates: SuggestHit[] = [];

    for (let d = 0; d < this.records.length; d++) {
      const doc = this.records[d];
      if (doc === null || doc === undefined) continue;
      const id = this.docIds[d];
      if (id === null || id === undefined) continue;
      const rows = this.docToRowIndices[d];
      if (!rows) continue;
      for (let k = 0; k < fieldIndices.length; k++) {
        const fIdx = fieldIndices[k] as number;
        const rowIdx = rows[fIdx];
        if (rowIdx === undefined) continue;
        if (this.tombstones.has(rowIdx)) continue;
        const rowTok = this.rowTokens[rowIdx];
        if (!rowTok || rowTok.length === 0) continue;
        const rawStr = this.rawFieldStrings[d]?.[fIdx] ?? '';
        if (rawStr.length === 0) continue;

        let matched = false;
        let rawScore = 0;
        if (spec.mode === 'prefix') {
          const r = scorePrefixTokens(rowTok, queryTokens, prefixOpts, typoForSuggest);
          matched = r.matched;
          rawScore = r.score;
        } else if (spec.fuzzyDistance > 0) {
          const r = scoreSubstringTypoTokens(rowTok, queryTokens, typoForSuggest);
          matched = r.matched;
          rawScore = r.score;
        } else {
          const r = scoreFuzzyTokens(rowTok, queryTokens);
          matched = r.matched;
          rawScore = r.score;
        }
        if (!matched) continue;
        const weight = this.sortedFields[fIdx].weight;
        candidates.push({
          keys: {
            score: Math.round(rawScore * weight),
            fieldWeight: weight,
            isExactMatch: isExactTokenMatch(rowTok, queryTokens),
            matchedLength: rowTok.length,
            id,
            docIndex: d
          },
          text: rawStr,
          dIdx: d,
          fIdx,
          rawStr
        });
      }
    }

    candidates.sort((a, b) => compareRanked(a.keys, b.keys, spec.tieBreakers));
    const top = candidates.length > spec.limit ? candidates.slice(0, spec.limit) : candidates;
    const out: SuggestionItem<TDoc>[] = [];
    for (let i = 0; i < top.length; i++) {
      const c = top[i] as SuggestHit;
      const highlightMode = spec.mode === 'prefix' ? 'prefix' : (spec.fuzzyDistance > 0 ? 'substring' : 'fuzzy');
      let matchedRanges: HighlightRange[] = [];
      try {
        matchedRanges = alignHighlights(c.rawStr, query, {
          mode: highlightMode,
          normalized: this.normalized,
          queryTokens,
          typoTolerance: spec.fuzzyDistance > 0
            ? { enabled: true, maxDistance: spec.fuzzyDistance as 1 | 2 }
            : undefined
        });
      } catch (err) {
        // Only tolerate expected highlight-alignment failures for a single
        // candidate (scorer/highlight length gates, polarity mismatch);
        // programmer errors and I/O faults must propagate.
        if (
          err instanceof IncompatibleOptionError ||
          err instanceof ProfileMismatchError ||
          err instanceof RangeError
        ) {
          matchedRanges = [];
        } else {
          throw err;
        }
      }
      out.push({
        text: c.text,
        score: c.keys.score,
        type: spec.mode === 'prefix' ? 'completion' : 'did-you-mean',
        matchedRanges,
        docId: c.keys.id,
        doc: this.records[c.dIdx] as TDoc
      });
    }
    return out;
  }

  /**
   * (): aggregates facet buckets over an explicit query-matched
   * doc set with disjunctive filter exclusion. Each facet ignores structured
   * filter clauses on its own field (`excludeFieldFromFilter`) while keeping
   * all other clauses; function predicates stay conjunctive. `or`/`not`
   * branches referencing the facet field are kept verbatim (conservative,
   * aliased). Cost is O(F*n) with F bitset allocs plus an optional extra
   * unfiltered parity scan when a structured filter is present.
   */
  private buildFacetResults(
    facetEngine: FacetEngine<TDoc>,
    facetSpec: NormalizedFacet[],
    queryMatched: Iterable<number>,
    structuredFilter: FilterExpression | undefined,
    filterPredicate: ((doc: TDoc) => boolean) | undefined,
    isApproximate: boolean
  ): Record<string, FacetResult> {
    const base: number[] = [];
    for (const dIdx of queryMatched) {
      const doc = this.records[dIdx];
      if (doc === null || doc === undefined) continue;
      if (filterPredicate && !filterPredicate(doc)) continue;
      base.push(dIdx);
    }
    const out: Record<string, FacetResult> = {};
    for (const facet of facetSpec) {
      let candidates: number[] = base;
      if (structuredFilter !== undefined) {
        const excluded = excludeFieldFromFilter(structuredFilter, facet.request.field);
        if (excluded !== undefined) {
          const mask = compileFilter(excluded, this.columnarStore);
          candidates = base.filter((d) => mask.has(d));
        }
      }
      const result = facetEngine.aggregateOne(facet, candidates, isApproximate);
      if (facet.name === '__proto__') {
        Object.defineProperty(out, facet.name, { value: result, enumerable: true, configurable: true, writable: true });
      } else {
        out[facet.name] = result;
      }
    }
    return out;
  }

  /**
   * deterministic re-sort after `scoringHook` boosts.
   * Reconstructs rank keys from index state (field weight, exactness via
   * `isExactTokenMatch` on the winning row, matched length, id, docIndex)
   * with updated scores, then sorts via `compareRanked` + `tieBreakers`.
   * Preserves the total-order contract (deterministic across engines).
   */
  private resortResultsAfterScoring(
    results: DocumentSearchResultItem<TDoc>[],
    queryTokens: Uint32Array,
    tieBreakers: TieBreakerCriterion[]
  ): void {
    if (results.length <= 1) return;
    const keys: RankableCandidate[] = new Array(results.length);
    for (let i = 0; i < results.length; i++) {
      const item = results[i] as DocumentSearchResultItem<TDoc>;
      const dIdx = this.idToDocIndex.get(item.id);
      const fIdx = this.fieldNameToIndex.get(item.matchedField);
      if (dIdx !== undefined && fIdx !== undefined) {
        const rowIdx = this.docToRowIndices[dIdx]?.[fIdx];
        const rowTok = rowIdx !== undefined ? this.rowTokens[rowIdx] : undefined;
        const weight = this.sortedFields[fIdx]?.weight ?? 1.0;
        keys[i] = {
          score: item.score,
          fieldWeight: weight,
          isExactMatch: rowTok !== undefined ? isExactTokenMatch(rowTok, queryTokens) : false,
          matchedLength: rowTok !== undefined ? rowTok.length : 0,
          id: item.id,
          docIndex: dIdx
        };
      } else {
        // Defensive fallback (unreachable in normal flow: resort runs before
        // postProcess, so IDs/fields resolve). Positional docIndex preserves
        // the total-order guarantee via unique positions.
        keys[i] = {
          score: item.score,
          fieldWeight: 1.0,
          isExactMatch: false,
          matchedLength: 0,
          id: item.id,
          docIndex: i
        };
      }
    }
    const order = results.map((_, i) => i);
    order.sort((ia, ib) => compareRanked(keys[ia] as RankableCandidate, keys[ib] as RankableCandidate, tieBreakers));
    const sorted = order.map((i) => results[i] as DocumentSearchResultItem<TDoc>);
    for (let i = 0; i < results.length; i++) {
      results[i] = sorted[i] as DocumentSearchResultItem<TDoc>;
    }
  }

  private enrichHighlights(
    results: DocumentSearchResultItem<TDoc>[],
    query: string,
    mode: SearchMode,
    options: DocumentSearchOptions<TDoc>,
    customTokenTerms?: Uint32Array[],
    precomputedTokens?: Uint32Array
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

    const queryTokens = precomputedTokens ?? normalizeText(query, this.normalized).tokens;
    // thread token/prefix/typo options so highlight ranges track
    // the serving scorer (score-highlight symmetry across all modes).
    // custom tokenizer terms replace the default split when supplied.
    const alignOpts = {
      mode,
      normalized: this.normalized,
      folded: this.normalized,
      queryTokens,
      tokenMatch: options.tokenMatch,
      prefixMatch: options.prefixMatch,
      typoTolerance: options.typoTolerance,
      ...(customTokenTerms !== undefined ? { tokenTermsOverride: customTokenTerms } : {})
    };

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
      const norm = normalizeText(rawStr, this.normalized);
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
          this.columnarStore.remove(d);
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
        this.columnarStore.remove(oldD);

        const newD = this.records.length;
        this.records.push(upd.doc);
        this.docIds.push(upd.id);
        this.idToDocIndex.set(upd.id, newD);
        this.rawFieldStrings.push(upd.rawStrings);
        this.columnarStore.add(newD, upd.doc);

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
          this.columnarStore.remove(oldD);
          updatedCount++;
        } else {
          addedCount++;
        }

        const newD = this.records.length;
        this.records.push(ad.doc);
        this.docIds.push(ad.id);
        this.idToDocIndex.set(ad.id, newD);
        this.rawFieldStrings.push(ad.rawStrings);
        this.columnarStore.add(newD, ad.doc);

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

      if (this.engineType === 'cpu' && !this.fallbackReason) {
        this.fallbackReason = this.options.preferGpu === false ? 'prefer-cpu' : 'below-threshold';
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
    const oldToNewDocIndexMap = new Map<number, number>();
    for (let newD = 0; newD < activeDocCount; newD++) {
      oldToNewDocIndexMap.set(activeOldDocIndices[newD], newD);
    }
    this.columnarStore.compact(oldToNewDocIndexMap, activeDocCount);
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
        this.fallbackReason = this.options.preferGpu === false ? 'prefer-cpu' : 'below-threshold';
      }
    }
  }

  private async compact(): Promise<void> {
    this.compactCpu();

    if (this.engineType === 'webgpu' && this.gpuEngine && this.gpuEngine.isReady) {
      const activeDocCount = this.idToDocIndex.size;
      const totalRows = this.rowTokens.length;
      if (activeDocCount === 0) {
        const packed = packDataset([], { normalized: this.normalized });
        await this.gpuEngine.loadDataset(packed);
        this.vramAllocatedBytes = this.gpuEngine.vramAllocatedBytes;
      } else {
        const packed = packDataset(this.rowTokens, {
          normalized: this.normalized,
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

  isNormalized(): boolean {
    return this.normalized;
  }

  /** @deprecated Use isNormalized. */
  isFolded(): boolean {
    return this.normalized;
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

  getColumnarStore(): ColumnarStore<TDoc> {
    return this.columnarStore;
  }

  getFilterFieldDefinitions(): FilterFieldDefinition<TDoc>[] {
    return this.filterFieldDefinitions.slice();
  }

  /**
   * index-level search hooks (shallow copy; function references
   * are shared with the index).
   */
  getHooks(): SearchHooks<TDoc> | undefined {
    return this.indexHooks === undefined ? undefined : { ...this.indexHooks };
  }

  /** @deprecated Use getHooks. */
  getExtensions(): SearchHooks<TDoc> | undefined {
    return this.getHooks();
  }

  serialize(options?: SerializeDocumentIndexOptions): ArrayBuffer {
    if (this.isDestroyed) {
      throw new Error('[webgpu-search] DocumentIndex has been destroyed.');
    }
    return encodeSnapshot(this, options);
  }

  async restore(buffer: ArrayBuffer, options?: RestoreDocumentIndexOptions<TDoc>): Promise<void> {
    if (this.isDestroyed) {
      throw new Error('[webgpu-search] DocumentIndex has been destroyed.');
    }
    const t0 = nowMs();
    await this.queued(async () => {
      const snapshot = decodeSnapshot<TDoc>(buffer, options);
      await this.applySnapshotData(snapshot, options);
      this.restoreTimeMs = nowMs() - t0;
    });
  }

  static async fromSnapshot<TDoc = Record<string, unknown>>(
    buffer: ArrayBuffer,
    options?: RestoreDocumentIndexOptions<TDoc>
  ): Promise<DocumentIndex<TDoc>> {
    return restoreSnapshot<TDoc>(buffer, options);
  }

  static async fromSnapshotData<TDoc = Record<string, unknown>>(
    snapshot: RestoredDocumentSnapshot<TDoc>,
    options?: RestoreDocumentIndexOptions<TDoc>
  ): Promise<DocumentIndex<TDoc>> {
    const t0 = nowMs();
    // fail-closed custom filter-getter guard. Snapshot `hasGetter`
    // entries require a matching getter override; otherwise restore would
    // silently rebuild columnar via default `doc[name]` (presence cleared).
    assertSnapshotFilterGettersSatisfied(
      snapshot.schema.filterFields as Array<{ name: string; hasGetter?: boolean }> | undefined,
      options?.options?.filterFields as Array<string | { name: string; getter?: unknown }> | undefined
    );
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
      caseSensitive: options?.options?.caseSensitive ?? snapshot.schema.caseSensitive ?? !(snapshot.header.normalized ?? snapshot.header.folded),
      device: (options?.device ?? options?.options?.device) as GPUDevice | undefined,
      preferGpu: options?.options?.preferGpu ?? snapshot.schema.preferGpu,
      threshold: options?.options?.threshold ?? snapshot.schema.threshold,
      candidateCapacity: options?.options?.candidateCapacity ?? snapshot.schema.candidateCapacity,
      initialCapacity: options?.options?.initialCapacity ?? snapshot.schema.initialCapacity,
      growthFactor: options?.options?.growthFactor ?? snapshot.schema.growthFactor,
      filterFields: options?.options?.filterFields ?? (snapshot.schema.filterFields as any),
      hooks: options?.options?.hooks ?? options?.options?.extensions
    };

    // fail-closed hook restore guard. Snapshots recording hookIds
    // require matching handlers via `options.options.hooks`; closures
    // are never serialized, only declarative IDs.
    assertHooksSatisfied(
      snapshot.schema.hookIds,
      normalizeSearchHooks(mergedOptions.hooks ?? mergedOptions.extensions)
    );

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
      if (typeof name !== 'string' || name.length === 0) {
        throw new TypeError('[webgpu-search] Restored schema field requires a non-empty string name.');
      }
      const weight = f.weight ?? 1.0;
      if (typeof weight !== 'number' || !Number.isFinite(weight) || weight <= 0) {
        throw new RangeError(
          `[webgpu-search] Restored field "${name}" weight must be a positive finite number, got ${String(weight)}.`
        );
      }
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
      if (b.weight !== a.weight) return b.weight > a.weight ? 1 : -1;
      if (a.originalIndex !== b.originalIndex) return a.originalIndex > b.originalIndex ? 1 : -1;
      return 0;
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
    // fail-closed hook restore guard for instance `restore()`.
    // Restore-supplied handlers (if any) replace the live index hooks;
    // otherwise the live hooks must satisfy the snapshot.
    const restoreHooks = normalizeSearchHooks(options?.options?.hooks ?? options?.options?.extensions);
    if (restoreHooks !== undefined) {
      this.indexHooks = restoreHooks;
      (this.options as { hooks?: SearchHooks<TDoc>; extensions?: SearchHooks<TDoc> }).hooks = restoreHooks;
    }
    assertHooksSatisfied(snapshot.schema.hookIds, this.indexHooks);

    // fail-closed custom filter-getter guard for instance restore.
    // Live filter defs (plus any restore override) must supply getters for
    // snapshot `hasGetter` fields; otherwise columnar rebuild would silently
    // change semantics. When the override supplies the getters, adopt it so
    // `columnarStore.init` uses the correct accessors.
    {
      const overrideFF = options?.options?.filterFields as
        | Array<string | { name: string; type?: FilterFieldType; getter?: (doc: TDoc) => any }>
        | undefined;
      const liveCustom = new Set<string>();
      for (const ff of this.filterFieldDefinitions) {
        if ((ff as { hasGetter?: boolean }).hasGetter === true) liveCustom.add(ff.name as string);
      }
      const overrideGetterNames = new Set<string>();
      if (Array.isArray(overrideFF)) {
        for (const uf of overrideFF) {
          if (uf && typeof uf === 'object' && typeof (uf as { getter?: unknown }).getter === 'function') {
            overrideGetterNames.add((uf as { name: string }).name);
          }
        }
      }
      const needed = ((snapshot.schema.filterFields as Array<{ name: string; hasGetter?: boolean }> | undefined) ?? []).filter(
        (ff) => ff && ff.hasGetter === true
      );
      for (const ff of needed) {
        if (!liveCustom.has(ff.name) && !overrideGetterNames.has(ff.name)) {
          throw new IncompatibleIndexError(
            `filter-getter:${ff.name}`,
            'missing custom getter override for snapshot filter field (supply options.options.filterFields with getter)'
          );
        }
      }
      if (Array.isArray(overrideFF) && overrideFF.length > 0) {
        const normalized: FilterFieldDefinition<TDoc>[] = [];
        const seen = new Set<string>();
        for (const uf of overrideFF) {
          if (typeof uf === 'string') {
            if (seen.has(uf)) continue;
            seen.add(uf);
            normalized.push({ name: uf as never, getter: ((doc: any) => (doc as any)[uf]) as never });
          } else if (uf && typeof uf === 'object' && typeof (uf as { name?: unknown }).name === 'string') {
            const o = uf as { name: string; type?: FilterFieldType; getter?: (doc: TDoc) => any };
            if (seen.has(o.name)) continue;
            seen.add(o.name);
            const hasCustom = typeof o.getter === 'function';
            normalized.push(
              hasCustom
                ? { name: o.name as never, type: o.type, getter: o.getter as never, hasGetter: true }
                : { name: o.name as never, type: o.type, getter: ((doc: any) => (doc as any)[o.name]) as never }
            );
          }
        }
        if (normalized.length > 0) {
          this.filterFieldDefinitions = normalized;
          this.columnarStore = new ColumnarStore<TDoc>(normalized, {
            initialCapacity: this.initialCapacity,
            growthFactor: this.growthFactor
          });
        }
      }
    }

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
    this.mutationEpoch =
      typeof snapshot.schema.mutationEpoch === 'number' &&
      Number.isFinite(snapshot.schema.mutationEpoch) &&
      snapshot.schema.mutationEpoch >= 0
        ? Math.floor(snapshot.schema.mutationEpoch)
        : 0;

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

    this.columnarStore.init(this.records);

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
              const packed = packDataset(this.rowTokens, {
                normalized: this.normalized,
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

              this.unsubscribeDeviceLost = GpuDevicePool.onDeviceLost(() => {
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
            this.fallbackReason =
              typeof navigator === 'undefined' || !('gpu' in navigator) || !navigator.gpu
                ? 'webgpu-unsupported'
                : 'device-request-failed';
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

    if (!shouldAttempt) {
      this.engineType = 'cpu';
      this.fallbackReason = this.options.preferGpu === false ? 'prefer-cpu' : 'below-threshold';
      return;
    }

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
        const packed = packDataset(this.rowTokens, {
          normalized: this.normalized,
          totalTokens: this.totalTokens
        });
        const effectiveCap = this.initialCapacity > 0 ? this.initialCapacity : Math.max(16, this.records.length);
        const rowCap = effectiveCap * this.sortedFields.length;
        const avgTokens = totalRows > 0 ? this.totalTokens / totalRows : 16;
        const tokenCap = Math.max(Math.floor(rowCap * avgTokens), 64);
        await gpu.loadDataset(packed, {
          rowCapacity: rowCap,
          tokenCapacity: tokenCap,
          growthFactor: this.growthFactor
        });
        this.gpuEngine = gpu;
        this.engineType = 'webgpu';
        this.fallbackReason = undefined;
        this.vramAllocatedBytes = gpu.vramAllocatedBytes;
        this.unsubscribeDeviceLost = GpuDevicePool.onDeviceLost(() => {
          if (this.gpuEngine) {
            this.gpuEngine.destroy();
            this.gpuEngine = null;
          }
          this.engineType = 'cpu';
          this.fallbackReason = 'device-lost';
        });
      } else {
        this.engineType = 'cpu';
        this.fallbackReason =
          typeof navigator === 'undefined' || !('gpu' in navigator) || !navigator.gpu
            ? 'webgpu-unsupported'
            : 'device-request-failed';
      }
    } catch {
      this.engineType = 'cpu';
      this.fallbackReason =
        typeof navigator === 'undefined' || !('gpu' in navigator) || !navigator.gpu
          ? 'webgpu-unsupported'
          : 'device-request-failed';
    }
  }

  getStats(): DocumentIndexStats {
    const adapter = this.gpuEngine?.adapterInfo;
    const docCount = this.idToDocIndex.size;
    const rowCount = this.rowTokens.length;
    const tombstoneCount = this.tombstones.size;
    const tombstoneRatio = rowCount > 0 ? tombstoneCount / rowCount : 0;
    const tokenRamBytes = this.totalTokens * 4;
    const offsetRamBytes = (rowCount + 1) * 4;
    const ramBytes = tokenRamBytes + offsetRamBytes;

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
      normalized: this.normalized,
      folded: this.normalized,
      formatVersion: SNAPSHOT_FORMAT_VERSION,
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
        totalBytes: this.vramAllocatedBytes + ramBytes,
        tokenRamBytes,
        offsetRamBytes
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
    this.columnarStore = new ColumnarStore<TDoc>(this.filterFieldDefinitions, {
      initialCapacity: this.initialCapacity,
      growthFactor: this.growthFactor
    });
    this.engineType = 'cpu';
    this.vramAllocatedBytes = 0;
    const p = this.searchMutex.then(() => {}, () => {});
    this.searchMutex = p.catch(() => {});
  }

  [Symbol.dispose](): void {
    this.destroy();
  }
}
