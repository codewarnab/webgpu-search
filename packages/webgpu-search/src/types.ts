import type {
  CpuAlgorithm,
  DOC_FORMAT_VERSION,
  FORMAT_VERSION,
  OnQueryTooLong,
  SCORING_VERSION,
  TextProfileId,
  UNICODE_VERSION,
} from './text-profile';

export type SearchMode = 'fuzzy' | 'substring';
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
  formatVersion: typeof FORMAT_VERSION | typeof DOC_FORMAT_VERSION;
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
}

export interface DocumentSearchOptions<TDoc = any> extends SearchOptions {
  /** Restrict search to specific configured fields */
  fields?: string[];
  /** Whether to compute highlight ranges (default: true) */
  highlight?: boolean;
  /** HTML/formatting tag for snippets (e.g. 'mark', 'b') */
  tag?: string;
  /** Predicate filter applied post-match */
  filter?: (doc: TDoc) => boolean;
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

export type FallbackReason =
  | 'webgpu-unsupported'
  | 'device-request-failed'
  | 'memory-budget-exceeded'
  | 'device-lost'
  | 'below-threshold'
  | 'prefer-cpu'
  | 'query-too-long'
  | 'cpu-algorithm-requested'
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
  formatVersion: typeof DOC_FORMAT_VERSION | typeof FORMAT_VERSION;
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


