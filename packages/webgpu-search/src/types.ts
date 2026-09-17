import type { CpuAlgorithm, OnQueryTooLong, TextProfileId } from './text-profile';

export type SearchMode = 'fuzzy' | 'substring';
export type EngineType = 'webgpu' | 'cpu';

export interface SearchOptions {
  mode?: SearchMode;              // Default: 'fuzzy'
  limit?: number;                 // Max results; default 50, clamped to 1..8192
  caseSensitive?: boolean;        // Default: false; must match index packed mode (v0.2)
  signal?: AbortSignal;           // Cancel stale queries during rapid typing
  maxResults?: number;            // Backwards-compatible alias; limit takes precedence
  cpuAlgorithm?: CpuAlgorithm;    // Default: 'parity'; 'ufuzzy' = explicit opt-in CPU-only
  onQueryTooLong?: OnQueryTooLong; // Default: 'throw'
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
  hasOverflow: boolean;           // True if matches > candidate pool capacity (8192)
  results: SearchResultItem[];    // Top-K ranked results
  timings: SearchTimings;
  profileId: TextProfileId;       // v0.2: text profile that served this query
  scoringVersion: string;         // v0.2: scoring contract version
  cpuAlgorithm: CpuAlgorithm;     // v0.2: which CPU scorer served / would serve fallback
}

export interface IndexOptions {
  threshold?: number;             // Item count cutoff for CPU vs GPU (Default: 30,000)
  preferGpu?: boolean;            // Force WebGPU if available regardless of size
  device?: GPUDevice;             // Custom injected GPUDevice (for testing/context sharing)
  powerPreference?: GPUPowerPreference; // 'high-performance' | 'low-power'
  slotBytes?: number;             // v0.2: throw-on-use (dynamic indexing replaced fixed slots)
  textProfile?: TextProfileId;    // v0.2: index-level immutable profile (default 'unicode-default')
  caseSensitive?: boolean;        // v0.2: pack-time fold control (default false = folded)
}

export interface IndexStats {
  size: number;
  engine: EngineType;
  vramAllocatedBytes: number;
  adapterVendor?: string;
  adapterRenderer?: string;
  profileId: TextProfileId;
  unicodeVersion: string;
  scoringVersion: string;
  tokenCount: number;
  folded: boolean;
  formatVersion: 2;
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

