import type {
  CpuAlgorithm,
  DOC_FORMAT_VERSION,
  FORMAT_VERSION,
  OnQueryTooLong,
  SCORING_VERSION,
  TextProfileId,
  U2D4_FORMAT_VERSION,
  UNICODE_VERSION,
} from './text-profile';

export type SearchMode = 'fuzzy' | 'substring' | 'token' | 'prefix';
export type EngineType = 'webgpu' | 'cpu';

export interface SearchOptions {
  mode?: SearchMode;              // Default: 'fuzzy'
  limit?: number;                 // Max results; default 50, clamped to 1..RESULT_LIMIT_MAX
  caseSensitive?: boolean;        // Default: false; must match index packed mode (v0.2 breaking: mismatch throws ProfileMismatchError — build one index per mode instead of varying per query)
  signal?: AbortSignal;           // Cancel stale queries during rapid typing
  maxResults?: number;            // Backwards-compatible alias; limit takes precedence
  /**
   * Default: 'parity' (v0.2 contract default). M2 serves the shared-pipeline
   * parity scorer (`cpu-reference.ts`); 'ufuzzy' = explicit opt-in CPU-only.
   */
  cpuAlgorithm?: CpuAlgorithm;
  /**
   * Default: 'throw'. Enforced in M2 on the exact post-fold token count
   * (`normalizeText(query, folded)` vs QUERY_TOKENS_MAX).
   * 'cpu-fallback' forces the CPU path.
   */
  onQueryTooLong?: OnQueryTooLong;
  /**
   * v0.4 M4: token-mode quorum options (operator/minMatchCount).
   * Only read when mode is 'token'; validated fail-closed otherwise.
   */
  tokenMatch?: TokenMatchOptions;
  /**
   * v0.4 M4: prefix-mode options (prefixLength/exactCase).
   * Only read when mode is 'prefix'; validated fail-closed otherwise.
   */
  prefixMatch?: PrefixSearchOptions;
  /**
   * v0.4 M4: bounded typo tolerance (boolean shorthand or full options).
   * Applies to 'substring', 'token', and 'prefix' modes; 'fuzzy' validates
   * but ignores (inherently typo-tolerant via subsequence matching).
   * Typo queries always route to the CPU reference engine — WGSL shaders
   * are exact-only (see webgpu-engine.ts).
   */
  typoTolerance?: TypoToleranceOptions | boolean;
  /**
   * v0.4 M7: cost budget controls and deadlines. Validated fail-closed via
   * `normalizeCostBudgetOptions`; over-budget execution throws
   * `CostBudgetExceededError`. `abortSignal` aborts with `AbortError`.
   */
  budget?: CostBudgetOptions;
  /**
   * v0.4 M7: whether to populate detailed `diagnostics` on the response.
   * Must be a boolean when provided; defaults to false (no telemetry).
   */
  diagnostics?: boolean;
}

export interface SearchResultItem {
  index: number;                  // Original index in dataset
  score: number;                  // Higher is better (normalized integer)
  text: string;                   // Resolved match string
}

export interface SearchTimings {
  queryUploadMs: number;          // Uniform write latency
  encodeSubmitMs: number;         // Command recording & submit latency
  gpuExecutionMs: number | null;  // Hardware timestamp query (null if unsupported)
  readbackMs: number;             // mapAsync() & CPU candidate slice latency
  totalMs: number;                // Total wall-clock query duration
  gpuDispatchMs?: number;         // Backwards compatibility alias for encodeSubmitMs
}

export interface SearchResponse {
  query: string;
  mode: SearchMode;
  engine: EngineType;             // Which engine serviced this query
  totalMatches: number;           // Total items passing threshold
  candidateCount: number;         // min(totalMatches, pool capacity); NOT results.length (limit truncation does not set overflow)
  hasOverflow: boolean;           // True iff totalMatches > pool capacity (RESULT_LIMIT_MAX); limit truncation alone leaves false
  results: SearchResultItem[];    // Top-K ranked results (length <= clamped limit)
  timings: SearchTimings;
  profileId: TextProfileId;       // v0.2: text profile that served this query
  scoringVersion: typeof SCORING_VERSION; // v0.2: scoring contract version
  cpuAlgorithm: CpuAlgorithm;     // v0.2: requested CPU scorer (M2 serves parity)
  fallbackReason?: FallbackReason; // Reason for CPU execution path if fallback occurred
  /** v0.4 M7: detailed telemetry and diagnostic metrics (when requested via options.diagnostics) */
  diagnostics?: QueryDiagnostics;
}

export interface IndexOptions {
  threshold?: number;             // Item count cutoff for CPU vs GPU (Default: 30,000)
  preferGpu?: boolean;            // Force WebGPU if available regardless of size (conflicts with cpuAlgorithm:'ufuzzy' → IncompatibleOptionError at search())
  device?: GPUDevice;             // Custom injected GPUDevice (for testing/context sharing)
  powerPreference?: GPUPowerPreference; // 'high-performance' | 'low-power' (reserved in M1: accepted, not yet forwarded — tracked for M3)
  slotBytes?: number;             // v0.2: throw-on-use (IncompatibleOptionError; dynamic indexing replaced fixed slots; removal in v0.3)
  textProfile?: TextProfileId;    // v0.2: index-level immutable profile (default 'unicode-default'; unknown values throw ProfileMismatchError at create())
  caseSensitive?: boolean;        // v0.2: pack-time fold control (default false = folded)
}

export interface IndexStats {
  size: number;
  engine: EngineType;
  vramAllocatedBytes: number;
  adapterVendor?: string;
  adapterRenderer?: string;
  profileId: TextProfileId;
  unicodeVersion: typeof UNICODE_VERSION;
  scoringVersion: typeof SCORING_VERSION;
  tokenCount: number;             // M2: exact post-fold code-point total
  folded: boolean;
  formatVersion: typeof FORMAT_VERSION | typeof DOC_FORMAT_VERSION | typeof U2D4_FORMAT_VERSION;
  fallbackReason?: FallbackReason;
  memory?: {
    vramBytes: number;
    ramBytes: number;
    totalBytes: number;
  };
}

export interface AdapterInfo {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
  renderer: string;
  maxBufferSizeMB: number;
  maxStorageBindingSizeMB: number;
  maxComputeWorkgroupsPerDimension: number;
  maxComputeInvocationsPerWorkgroup: number;
  hasTimestampQuery: boolean;
}

// ---------------------------------------------------------------------------
// v0.3 Document, Mutation, Highlighting, Worker, and Telemetry Types
// ---------------------------------------------------------------------------

export type DocumentId = string | number;
export type DocumentRecord = Record<string, unknown>;

export interface FieldDefinition<TDoc = Record<string, unknown>> {
  /** Field identifier / key in TDoc */
  name: (keyof TDoc & string) | string;
  /** Custom property extractor; defaults to (doc) => doc[name] */
  getter?: (doc: TDoc) => string | string[] | undefined | null;
  /** Weight multiplier applied to raw match score (default: 1.0, must be > 0 and finite) */
  weight?: number;
}

export type DocumentField<TDoc> = (keyof TDoc & string) | FieldDefinition<TDoc>;

export interface DocumentIndexOptions<TDoc = Record<string, unknown>> extends IndexOptions {
  /** Property containing unique document identifier (default: 'id') */
  idField?: (keyof TDoc & string) | ((doc: TDoc) => DocumentId);
  /** Configured search fields and weights */
  fields: Array<DocumentField<TDoc>>;
  /** Pre-allocated row capacity for dynamic mutations (default: records.length * 1.5) */
  initialCapacity?: number;
  /** Buffer expansion factor during headroom overflow (default: 1.5) */
  growthFactor?: number;
  /** Candidate pool capacity for multi-field search (default: 8192, up to 32768) */
  candidateCapacity?: number;
  /** Attributes configured for columnar pre-filtering and facet aggregation */
  filterFields?: Array<DocumentFilterField<TDoc>>;
  /** Extension hooks for custom tokenization, scoring boosts, or predicates */
  extensions?: SearchExtensionHooks<TDoc>;
}

export interface HighlightRange {
  /** Start index in UTF-16 code units in the original JS string */
  start: number;
  /** End index (exclusive) in UTF-16 code units in the original JS string */
  end: number;
}

export interface HighlightOptions {
  /** Whether to compute highlight ranges (default: true) */
  highlight?: boolean;
  /** HTML/markdown tag for formatting (e.g. 'mark', 'b') */
  tag?: string;
  /** Fields to highlight ('matched-field' | 'all-matched' | 'all-fields' | string[]) */
  fields?: 'matched-field' | 'all-matched' | 'all-fields' | string[];
  /** Whether to HTML-escape special characters in raw strings (default: false) */
  escapeHtml?: boolean;
}

export interface DocumentSearchOptions<TDoc = any> extends SearchOptions {
  /** Restrict search to specific configured fields */
  fields?: string[];
  /** Whether to compute highlight ranges (default: true) */
  highlight?: boolean;
  /** HTML/formatting tag for snippets (e.g. 'mark', 'b') */
  tag?: string;
  /** Optional highlight configuration options */
  highlightOptions?: HighlightOptions;
  /** Fields to highlight ('matched-field' | 'all-matched' | 'all-fields' | string[]) */
  highlightFields?: 'matched-field' | 'all-matched' | 'all-fields' | string[];
  /** Whether to HTML-escape special characters in snippet rendering (default: false) */
  escapeHtml?: boolean;
  /**
   * Predicate filter applied post-match or structured filter expression
   * evaluated via columnar bitsets.
   */
  filter?: ((doc: TDoc) => boolean) | FilterExpression;
  /** Facet aggregation requests to evaluate over matching candidates.
   * Note: `filter: fn` + `facets` over SearchWorkerClient drops `facets`
   * fail-closed (function closures cannot cross the worker boundary, so
   * worker-computed facets would reflect the unfiltered set). */
  facets?: Record<string, FacetRequest> | FacetRequest[];
  /** Faceting mode: force exact CPU candidate evaluation even if GPU buffer overflowed. Ignored unless `facets` is requested. */
  faceting?: 'auto' | 'force-exact';
  // Note: tokenMatch/prefixMatch/typoTolerance are inherited from
  // SearchOptions (single source of truth — do not redeclare; drift risk).
  /** Deterministic ranking and tie-breaking options */
  ranking?: DeterministicRankingOptions;
  /** Per-query search extension overrides */
  extensions?: SearchExtensionHooks<TDoc>;
  /** Cost budget controls and deadlines */
  budget?: CostBudgetOptions;
  /** Whether to populate detailed diagnostics on the response */
  diagnostics?: boolean;
  /** Autocomplete / did-you-mean suggestion configuration if requested alongside search.
   * `true` uses defaults; an object customizes; `false`/omitted disables.
   * Inline suggestions cost a second O(docs x fields) scan (~2x query cost)
   * and are index-wide by design: `filter` never narrows suggestions, while
   * `fields` scopes them unless `suggest.field` is set (explicit suggest
   * field wins). Suggestions use default `prefixMatch` opts, not the search
   * `prefixMatch`/`typoTolerance` — only `suggest.mode`/`fuzzyDistance` apply.
   * `prefix` yields `type:'completion'` (including typo-tolerant prefix);
   * `fuzzy` yields `type:'did-you-mean'`.
   */
  suggest?: SuggestOptions | boolean;
}

export interface DocumentSearchResultItem<TDoc = any> {
  id: DocumentId;
  doc: TDoc;
  score: number;
  matchedField: string;
  /** Highlight ranges per field in UTF-16 code units of the original string */
  highlights?: Record<string, HighlightRange[]>;
  /** Formatted HTML strings with injected tags (when options.tag is set) */
  highlightedText?: Record<string, string>;
  /** Auxiliary matches across other indexed fields */
  matches?: Array<{ field: string; score: number; highlights?: HighlightRange[] }>;
}

/**
 * Why a query was served by the CPU engine instead of WebGPU.
 * v0.4 M4 adds 'unsupported-mode': 'token'/'prefix' modes and typo-tolerant
 * queries route to the CPU reference engine (WGSL shaders are exact-only
 * for 'fuzzy'/'substring'); recorded per the Issue #10 scoring-parity
 * boundary (unsupported features route to CPU with a recorded reason).
 */
export type FallbackReason =
  | 'webgpu-unsupported'
  | 'device-request-failed'
  | 'memory-budget-exceeded'
  | 'device-lost'
  | 'below-threshold'
  | 'prefer-cpu'
  | 'query-too-long'
  | 'cpu-algorithm-requested'
  | 'unsupported-mode'
  | 'gpu-execution-error';

export interface DocumentSearchResponse<TDoc = any> {
  query: string;
  mode: SearchMode;
  engine: EngineType;
  totalMatches: number;
  candidateCount: number;
  hasOverflow: boolean;
  results: DocumentSearchResultItem<TDoc>[];
  timings: SearchTimings;
  profileId: TextProfileId;
  scoringVersion: typeof SCORING_VERSION;
  cpuAlgorithm: CpuAlgorithm;
  fallbackReason?: FallbackReason;
  /** Facet aggregation results keyed by facet name or field name */
  facets?: Record<string, FacetResult>;
  /** Detailed telemetry and diagnostic metrics (when requested) */
  diagnostics?: QueryDiagnostics;
  /** Autocomplete / did-you-mean suggestions if requested with query */
  suggestions?: SuggestionItem<TDoc>[];
}

export interface AddOptions {
  upsert?: boolean;
}

export interface MutationBatch<TDoc = any> {
  add?: TDoc[];
  update?: TDoc[];
  remove?: DocumentId[];
}

export interface MutationResult {
  added: number;
  updated: number;
  removed: number;
  mutationEpoch: number;
  compacted: boolean;
  durationMs: number;
}

export interface DocumentIndexStats extends IndexStats {
  formatVersion: typeof DOC_FORMAT_VERSION | typeof FORMAT_VERSION | typeof U2D4_FORMAT_VERSION;
  docCount: number;
  rowCount: number;
  tombstoneCount: number;
  tombstoneRatio: number;
  buildTimeMs: number;
  restoreTimeMs?: number;
  lastMutationTimeMs?: number;
  mutationEpoch: number;
  memory: {
    vramBytes: number;
    ramBytes: number;
    totalBytes: number;
    tokenRamBytes?: number;
    offsetRamBytes?: number;
  };
  fallbackReason?: FallbackReason;
}

export interface WorkerClientOptions {
  /** Optional custom worker instance or factory */
  worker?: Worker | (() => Worker);
  /** Whether to strip document text across thread boundary (default: true) */
  stringIsolated?: boolean;
}

export type WorkerMessageType =
  | 'INIT'
  | 'SEARCH'
  | 'MUTATE'
  | 'SERIALIZE'
  | 'RESTORE'
  | 'STATS'
  | 'DESTROY'
  | 'ABORT';

export interface WorkerRequest {
  id: number;
  type: WorkerMessageType;
  payload?: any;
}

export interface WorkerResponse {
  id: number;
  success: boolean;
  result?: any;
  error?: {
    name: string;
    message: string;
    stack?: string;
    details?: Record<string, unknown>;
  };
}

// ---------------------------------------------------------------------------
// v0.3 Snapshot Persistence (U2D3) & IndexedDB Types (M6)
// ---------------------------------------------------------------------------

export interface DocumentIndexSchema {
  fields: Array<{
    name: string;
    weight: number;
  }>;
  idField?: string;
  docIds?: DocumentId[];
  caseSensitive?: boolean;
  preferGpu?: boolean;
  threshold?: number;
  candidateCapacity?: number;
  initialCapacity?: number;
  growthFactor?: number;
  mutationEpoch?: number;
  filterFields?: Array<{
    name: string;
    type?: FilterFieldType;
  }>;
  /**
   * v0.4 M6: declarative extension hook identifiers recorded at snapshot
   * time. Closures are never serialized — only these stable IDs are stored.
   * Restore validates that matching hook handlers are supplied (see
   * `assertHooksSatisfied` in extensions.ts); missing handlers throw
   * `IncompatibleHookError` fail-closed.
   */
  hookIds?: ExtensionHookIds;
}

export interface SerializeDocumentIndexOptions {
  /**
   * If true, document records are not written to the snapshot (docsByteLength = 0).
   * Useful for string-isolated worker or decoupled IndexedDB storage.
   * Default: false.
   */
  decoupled?: boolean;
}

export interface RestoreDocumentIndexOptions<TDoc = Record<string, unknown>> {
  /**
   * Optional custom DocumentIndex options to override or extend schema options
   * (e.g. custom getters, device, preferGpu, threshold).
   *
   * v0.4 M6: `options.extensions` (when supplied to `restore` /
   * `fromSnapshotData` / instance `restore()`) replaces live index hooks
   * wholesale (not per-key merged); the merged set must then satisfy the
   * snapshot `hookIds` via `assertHooksSatisfied` or `IncompatibleHookError`
   * is thrown fail-closed.
   */
  options?: Partial<DocumentIndexOptions<TDoc>>;
  /**
   * Decoupled documents to rehydrate if the snapshot was serialized with decoupled: true.
   */
  documents?: TDoc[];
  /**
   * Optional GPUDevice for WebGPU initialization.
   */
  device?: GPUDevice | null;
  /**
   * Whether to transfer the input buffer.
   */
  transfer?: boolean;
}

export interface DocumentSnapshotHeader {
  magic: number;
  formatVersion: number;
  profileId: TextProfileId;
  unicodeVersion: string;
  scoringVersion: string;
  docCount: number;
  rowCount: number;
  tokenCount: number;
  folded: boolean;
  schemaByteLength: number;
  docsByteLength: number;
  columnarByteLength?: number;
  checksum: number;
}

export interface IDBStorageOptions {
  /** Database name (default: 'webgpu_search_db') */
  dbName?: string;
  /** Object store name for index binary snapshots (default: 'index_snapshots') */
  snapshotStoreName?: string;
  /** Object store name for decoupled documents (default: 'documents') */
  docStoreName?: string;
  /** Key identifying this index in snapshot store (default: 'default_index') */
  key?: string;
  /** Custom IDBFactory instance (defaults to globalThis.indexedDB) */
  indexedDB?: any;
}

export interface SaveIDBOptions<TDoc = Record<string, unknown>> extends IDBStorageOptions {
  /** Whether to store document records decoupled in the document store (default: false) */
  decoupled?: boolean;
  /** Optional decoupled documents to store if not extracted from index */
  documents?: TDoc[];
}

export interface LoadIDBResult<TDoc = Record<string, unknown>> {
  snapshot: ArrayBuffer;
  documents?: TDoc[];
}

export interface LoadIDBOptions extends IDBStorageOptions {
  /** Whether to load decoupled documents from the document store (default: true) */
  loadDocuments?: boolean;
}

// ---------------------------------------------------------------------------
// v0.4 Structured Filters, Facets, Search Modes, Typo-Tolerance,
// Deterministic Ranking, Autocomplete, Extensibility, & Cost Budgets
// ---------------------------------------------------------------------------

// 3.1 Structured Filters & Columnar Metadata
export type FilterValue = string | number | boolean | null;

export interface FieldComparison {
  eq?: FilterValue;
  neq?: FilterValue;
  gt?: number | string;
  gte?: number | string;
  lt?: number | string;
  lte?: number | string;
  in?: FilterValue[];
  nin?: FilterValue[];
  exists?: boolean;
}

export type FieldFilter = {
  [field: string]: FilterValue | FilterValue[] | FieldComparison;
};

/**
 * Structured filter expression AST for pre-match columnar evaluation.
 * - `{ and: [] }` evaluates to true (vacuous truth).
 * - `{ or: [] }` evaluates to false (vacuous falsehood).
 */
export type FilterExpression =
  | FieldFilter
  | { and: FilterExpression[] }
  | { or: FilterExpression[] }
  | { not: FilterExpression };

export type FilterFieldType = 'string' | 'number' | 'boolean' | 'string[]';

export interface FilterFieldDefinition<TDoc = Record<string, unknown>> {
  name: (keyof TDoc & string) | (string & {});
  type?: FilterFieldType;
  getter?: (doc: TDoc) => FilterValue | FilterValue[] | undefined | null;
}

export type DocumentFilterField<TDoc> = (keyof TDoc & string) | FilterFieldDefinition<TDoc>;

// 3.2 Facet Aggregations with Exact vs. Approximate Semantics
export interface TermsFacetRequest {
  type: 'terms';
  field: string;
  /** Max buckets; default 10. Fractional values are floored; must be >= 1. */
  limit?: number; // default: 10
  sortBy?: 'count' | 'value';
}

export interface RangeFacetBucket {
  /** Inclusive lower bound (`value >= from`) */
  from?: number;
  /** Exclusive upper bound (`value < to`) */
  to?: number;
  /** Custom bucket key identifier (defaults to "${from ?? '*'}-${to ?? '*'}") */
  key?: string;
}

export interface RangeFacetRequest {
  type: 'range';
  field: string;
  ranges: RangeFacetBucket[];
}

export type FacetRequest = TermsFacetRequest | RangeFacetRequest;

export interface TermsFacetBucket {
  value: FilterValue;
  count: number;
}

export interface TermsFacetResult {
  type: 'terms';
  field: string;
  /**
   * Exactness is engine-relative: false means exact w.r.t. the serving
   * engine's match set (parity CPU, ufuzzy CPU, or GPU pool), not identical
   * across `cpuAlgorithm: parity | ufuzzy` or GPU vs CPU (scorers may diverge
   * row-for-row). True only on GPU overflow without `force-exact`.
   */
  isApproximate: boolean;
  buckets: TermsFacetBucket[];
}

export interface RangeFacetBucketResult {
  key: string;
  from?: number;
  to?: number;
  count: number;
}

export interface RangeFacetResult {
  type: 'range';
  field: string;
  /**
   * Engine-relative exactness (see TermsFacetResult): false means exact over
   * the serving engine's match set. Overlapping ranges double-count by design
   * (independent half-open buckets).
   */
  isApproximate: boolean;
  buckets: RangeFacetBucketResult[];
}

export type FacetResult = TermsFacetResult | RangeFacetResult;

// 3.3 Expanded Search Modes & Typo Tolerance
export interface TypoToleranceOptions {
  /** Default: false. Note: maxDistance alone does NOT enable — must set enabled:true. */
  enabled?: boolean;                // Default: false
  maxDistance?: 1 | 2;              // Default: 1
  minWordLengthForOneTypo?: number; // Default: 4
  minWordLengthForTwoTypos?: number;// Default: 8
  /** Default: 1 (first N chars must match exactly; leading transpositions never match). */
  prefixExactLength?: number;
}

export interface TokenMatchOptions {
  operator?: 'and' | 'or';          // Default: 'and'
  minMatchCount?: number;
}

export interface PrefixSearchOptions {
  /**
   * Leading query code points used for matching (undefined = full query).
   * Over-length (prefixLength > query.length) throws RangeError fail-closed
   * on every path including empty corpora — autocomplete callers should
   * clamp or catch and treat as no-match.
   */
  prefixLength?: number;
  /**
   * Must agree with the query caseSensitive flag when explicitly set;
   * when omitted the index derives polarity from caseSensitive (default
   * follows the query flag).
   */
  exactCase?: boolean;
}

// 3.4 Deterministic Ranking & Tie-Breaking
export type TieBreakerCriterion = 'score' | 'weight' | 'exact' | 'length' | 'id';

export interface DeterministicRankingOptions {
  /**
   * Tie-breaker order hierarchy evaluated when scores are tied.
   * Default: ['score', 'weight', 'exact', 'length', 'id']
   *
   * M5 behavior change: `DocumentIndex.search()` now applies the full
   * 5-tier order by default on all paths (GPU readback, parity CPU,
   * legacy ufuzzy). Previously results were ordered by
   * `(score DESC, docIndex ASC)` only. To approximate the legacy order,
   * pass `ranking: { tieBreakers: ['score'] }` (remaining ties fall
   * through to `docIndex ASC`; exact legacy order is not bit-reproduced
   * when IDs differ from insertion order).
   */
  tieBreakers?: TieBreakerCriterion[];
}

// 3.5 Suggestions & Autocomplete Primitives
export interface SuggestOptions {
  limit?: number;                   // Default: 5
  mode?: 'prefix' | 'fuzzy';        // Default: 'prefix'
  fuzzyDistance?: number;           // Default: 0 (integer 0..2; explicit 1 enables typo-tolerant suggest)
  field?: string;                   // Restrict to specific field (beats search.fields when both set)
  /**
   * Suggestion tie-breaker hierarchy. Default: the M5 5-tier order.
   * Inline `search({ ranking, suggest })` inherits the search `ranking`
   * hierarchy when the suggest object omits this key.
   */
  tieBreakers?: TieBreakerCriterion[];
}

export interface SuggestionItem<TDoc = any> {
  text: string;
  score: number;
  type: 'completion' | 'did-you-mean';
  matchedRanges: HighlightRange[];
  docId?: DocumentId;
  doc?: TDoc;
}

export interface SuggestResponse<TDoc = any> {
  suggestions: SuggestionItem<TDoc>[];
  queryDurationMs: number;
}

// 3.6 Search Extension Pipeline
export interface MatchInfo {
  query: string;
  matchedField: string;
  rawScore: number;
  normalizedScore: number;
}

/**
 * Declarative extension hook identifiers persisted in snapshots.
 * Each present key records the stable ID of the corresponding hook at
 * serialize time (function `hookId` property when set and non-blank, else
 * `function.name`, else `'anonymous'`). Name-derived IDs can collide across
 * distinct functions sharing an inferred name, and every truly anonymous
 * closure maps to `'anonymous'` (fail-open). Hosts requiring stable restores
 * across builds should assign explicit IDs:
 * `myScorer.hookId = 'recency-v1'` (or use named functions).
 */
export interface ExtensionHookIds {
  tokenizer?: string;
  scoringHook?: string;
  filterPredicate?: string;
  postProcess?: string;
}

/**
 * Type-safe extension hooks for host applications (v0.4 M6).
 *
 * - `tokenizer`: custom query term splitting for `'token'` mode (e.g. code
 *   symbols `_`, `-`, `camelCase`). Only affects `'token'` mode query term
 *   parsing on the CPU path (`'token'` is CPU-by-design with fallbackReason
 *   `'unsupported-mode'`); other modes ignore it. Must be a pure,
 *   deterministic function (runs once per query, never per record).
 * - `scoringHook`: post-match boost over surviving Top-K candidates only
 *   (never per-candidate scanning). Receives `(doc, baseScore, matchInfo)`
 *   and must return a finite integer (unified descending normalized integer
 *   score contract; floats throw `TypeError`). Applied identically on GPU
 *   and CPU paths after deterministic ranking + limit truncation, followed
 *   by a deterministic re-sort. Must be pure and deterministic.
 * - `filterPredicate`: conjunctive post-match predicate composed (AND) with
 *   `options.filter` (function or structured). Applied on every path.
 *   Return values are truthiness-coerced (`!predicate(doc)` excludes).
 *   Must be pure: invocation order is engine-dependent (GPU result order vs
 *   parity row order), so non-deterministic predicates diverge.
 * - `postProcess`: final result transformation after scoring boosts,
 *   deterministic re-sort, and highlight enrichment. Must return an array
 *   (hook throws propagate). Affects only `results`: `totalMatches` /
 *   `candidateCount` / `hasOverflow` are snapshotted pre-pipeline and
 *   `facets` / `suggestions` ignore it. Skipped on empty no-hit paths
 *   (empty query / corpus / empty filter) where there is nothing to transform.
 *
 * Persistence: closures are never serialized. `serialize()` records only
 * `ExtensionHookIds`; `restore` requires matching handlers via
 * `options.options.extensions` or throws `IncompatibleHookError`.
 * Restore-supplied handlers replace (not merge with) live index hooks.
 * Hooks cannot cross the Web Worker boundary — `SearchWorkerClient`
 * `init` / `search` / `restore` reject `extensions` fail-closed with
 * `IncompatibleHookError` (empty `{}` is a no-op and allowed).
 */
export interface SearchExtensionHooks<TDoc = any> {
  tokenizer?: (text: string) => string[];
  scoringHook?: (doc: TDoc, baseScore: number, matchInfo: MatchInfo) => number;
  filterPredicate?: (doc: TDoc) => boolean;
  postProcess?: (results: DocumentSearchResultItem<TDoc>[]) => DocumentSearchResultItem<TDoc>[];
}

// 3.7 Cost Budgets & Query Diagnostics
export interface CostBudgetOptions {
  maxExecutionTimeMs?: number;      // Maximum execution wall-clock time in milliseconds
  maxCandidates?: number;           // Ceiling on candidates scored
  abortSignal?: AbortSignal;        // Caller abort signal
}

export interface QueryDiagnosticsTimings {
  filteringMs: number;            // Time spent evaluating columnar bitsets
  scoringMs: number;              // Time spent in compute kernel or CPU reference
  highlightMs: number;            // Time spent extracting Unicode highlight ranges
  facetingMs?: number;            // Time spent aggregating facet buckets
  totalMs: number;                // End-to-end query latency
}

export interface QueryDiagnostics {
  scannedCandidates: number;        // Total candidate rows evaluated
  filterSelectivity: number;        // Ratio of candidates matching filter (0.0 - 1.0)
  routedEngine: EngineType;         // Engine that processed the query
  hasOverflow: boolean;             // Whether GPU candidate capacity was exceeded
  timings: QueryDiagnosticsTimings;
  warnings?: string[];              // Non-fatal advisory notices (e.g. broad-query fallback)
}
