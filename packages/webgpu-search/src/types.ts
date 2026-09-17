import type {
  CpuAlgorithm,
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
   * Default: 'parity' (v0.2 contract default). M1 shape-only note: the parity
   * scorer lands in M2 (`cpu-reference.ts`); the M1 CPU path still serves the
   * legacy uFuzzy (`mode:'fuzzy'`) / native (`mode:'substring'`) scorer while
   * echoing the requested value. 'ufuzzy' = explicit opt-in CPU-only (skips GPU).
   */
  cpuAlgorithm?: CpuAlgorithm;
  /**
   * Default: 'throw'. Enforced in M1 on a pre-fold code-point approximation
   * (`countUnicodeCodePoints(trimmed query)` vs QUERY_TOKENS_MAX); exact
   * post-fold enforcement lands in M2. 'cpu-fallback' forces the CPU path.
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
  candidateCount: number;         // Scored candidates returned from GPU/CPU
  hasOverflow: boolean;           // True if matches > candidate pool capacity (RESULT_LIMIT_MAX)
  results: SearchResultItem[];    // Top-K ranked results
  timings: SearchTimings;
  profileId: TextProfileId;       // v0.2: text profile that served this query
  scoringVersion: typeof SCORING_VERSION; // v0.2: scoring contract version
  cpuAlgorithm: CpuAlgorithm;     // v0.2: requested CPU scorer (M1 echoes request; M2 serves parity)
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
  tokenCount: number;             // M1: pre-fold code-point total (exact post-fold count lands in M2)
  folded: boolean;
  formatVersion: typeof FORMAT_VERSION;
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

