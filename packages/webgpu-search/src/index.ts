// High-Level Primary API
export { SearchIndex } from './hybrid-index';
export { DocumentIndex } from './document-index';
export { SearchWorkerClient } from './worker/worker-client';


// Low-Level Engines & Hardware Utilities for Power Users & Benchmarks
export { WebGPUEngine, type DatasetLike, type EngineDataset, type ColdSearchResult, type WebGPUSearchResult, type SearchResult } from './webgpu-engine';
export { CPUEngine, type CPUSearchResult } from './cpu-engine';
export { WebGPUContextManager, type AcquiredDeviceContext } from './context-manager';
export {
  packStringsToGPUBuffer,
  sanitizeStringForSlot,
  packUnicodeToGPUBuffer,
  serializeUnicodeDataset,
  deserializeUnicodeDataset,
  validatePackedOffsets,
  checkMemoryBudget,
  computeClampedHeadroomBytes,
  type PackedGPUBuffer,
  type PackedUnicodeBufferV2,
  type UnicodePackOptions,
  type MemoryBudgetCheck,
  type ClampedHeadroomOptions
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
  PROFILE_TO_ENUM,
  ENUM_TO_PROFILE,
  UNICODE_VERSION_TO_ENUM,
  ENUM_TO_UNICODE_VERSION,
  SCORING_TO_ENUM,
  ENUM_TO_SCORING,
  countUnicodeCodePoints,
  IncompatibleIndexError,
  IncompatibleOptionError,
  ProfileMismatchError,
  QueryTooLongError,
  DOC_FORMAT_VERSION,
  SERIALIZED_DOC_MAGIC,
  SERIALIZED_DOC_HEADER_BYTES,
  DuplicateIdError,
  DocumentNotFoundError,
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
  LONE_SURROGATE_SOURCE,
  type NormalizedText,
} from './unicode-preprocess';
export {
  clampLimit,
  DEFAULT_LIMIT,
  abortError,
  nowMs,
  throwIfAborted,
  isAsciiTokens,
  isPrintableAsciiTokens,
} from './runtime-guards';
export {
  compareParityResults,
  scoreFuzzyTokens,
  scoreSubstringTokens,
  searchCpuReference,
  searchMultiFieldCpuReference,
  WORD_BOUNDARY_PREV,
  type CpuReferenceOutput,
  type MultiFieldHit,
  type MultiFieldMatch,
  type MultiFieldCpuReferenceOutput,
  type FieldScoreDefinition,
} from './cpu-reference';
export {
  FOLD_EXPANSIONS,
  FOLD_RANGES,
  FOLD_C_COUNT,
  FOLD_F_COUNT,
  FOLD_UNICODE_VERSION,
  foldCodePoint,
} from './fold-table';

// v0.3 Unicode-safe highlighting engine (M3)
export {
  alignHighlights,
  normalizeWithSourceMap,
  guardClusterBoundary,
  mergeHighlightRanges,
  renderHighlightedText,
  type SourceMappedText,
  type AlignHighlightOptions,
  type RenderHighlightOptions,
} from './highlight';

