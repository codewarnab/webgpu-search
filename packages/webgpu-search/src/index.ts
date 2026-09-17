// High-Level Primary API
export { SearchIndex } from './hybrid-index';

// Low-Level Engines & Hardware Utilities for Power Users & Benchmarks
export { WebGPUEngine, type DatasetLike, type ColdSearchResult, type WebGPUSearchResult, type SearchResult } from './webgpu-engine';
export { CPUEngine, type CPUSearchResult } from './cpu-engine';
export { WebGPUContextManager, type AcquiredDeviceContext } from './context-manager';
export {
  packStringsToGPUBuffer,
  sanitizeStringForSlot,
  checkMemoryBudget,
  type PackedGPUBuffer,
  type MemoryBudgetCheck
} from './buffer';

// Unified Types
export type * from './types';

// v0.2 Unicode text profile (versions, errors, caps)
export {
  FORMAT_VERSION,
  QUERY_TOKENS_MAX,
  RESULT_LIMIT_MAX,
  SCORING_VERSION,
  SERIALIZED_MAGIC,
  UNICODE_VERSION,
  countUnicodeCodePoints,
  IncompatibleIndexError,
  IncompatibleOptionError,
  ProfileMismatchError,
  QueryTooLongError,
  type CpuAlgorithm,
  type OnQueryTooLong,
  type TextProfileId,
} from './text-profile';

// v0.2 shared preprocessing + CPU reference (M2)
export {
  normalizeText,
  toWellFormedSafe,
  tokensEqual,
  LONE_SURROGATE_PATTERN,
  type NormalizedText,
} from './unicode-preprocess';
export {
  compareParityResults,
  scoreFuzzyTokens,
  scoreSubstringTokens,
  searchCpuReference,
  type CpuReferenceOutput,
} from './cpu-reference';
export {
  FOLD_EXPANSIONS,
  FOLD_RANGES,
  FOLD_C_COUNT,
  FOLD_F_COUNT,
  FOLD_UNICODE_VERSION,
  foldCodePoint,
} from './fold-table';
