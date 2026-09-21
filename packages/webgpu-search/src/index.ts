// High-Level Primary API
export { SearchIndex } from './hybrid-index';
export { DocumentIndex } from './document-index';
export { SearchWorkerClient, INTERNAL_WORKER_ID_KEY } from './worker/worker-client';


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
  crc32Parts,
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
  U2D4_MAGIC,
  U2D4_FORMAT_VERSION,
  FORMAT_VERSION_4,
  DOC_FORMAT_VERSION_4,
  U2D4_HEADER_BYTES,
  type CpuAlgorithm,
  type OnQueryTooLong,
  type TextProfileId,
} from './text-profile';

// v0.4 Error hierarchy
export {
  WebGPUSearchError,
  IncompatibleHookError,
  CostBudgetExceededError,
  InvalidFilterError
} from './errors';

// v0.4 Structured filtering & columnar metadata (M2)
export { DocumentBitset } from './filter/bitset';
export { ColumnarStore, type ColumnarStoreOptions } from './filter/columnar-store';
export { compileFilter } from './filter/filter-evaluator';

// v0.4 Facet aggregation engine (M3)
export {
  FacetEngine,
  normalizeFacetRequests,
  excludeFieldFromFilter,
  filterReferencesField,
  DEFAULT_TERMS_LIMIT,
  MAX_FACET_REQUESTS,
  MAX_RANGE_BUCKETS,
  type NormalizedFacet,
} from './facets/facet-engine';

// v0.4 Deterministic ranking & autocomplete primitives (M5)
export {
  DEFAULT_TIE_BREAKERS,
  compareIdsAsc,
  compareRanked,
  isExactTokenMatch,
  normalizeTieBreakers,
  sortRanked,
  type RankableCandidate,
} from './ranking';
export {
  SUGGEST_DEFAULT_LIMIT,
  SUGGEST_DEFAULT_MODE,
  SUGGEST_MAX_FUZZY_DISTANCE,
  normalizeSuggestOptions,
  type NormalizedSuggestOptions,
  type SuggestCandidateKeys,
} from './suggest';

// v0.4 Extensibility pipeline & safe hook architecture (M6)
export {
  defaultTokenizer,
  codeTokenizer,
  normalizeSearchExtensionHooks,
  resolveEffectiveHooks,
  hasAnyHook,
  getHookId,
  collectHookIds,
  assertHooksSatisfied,
  tokenizeWithHook,
  getTokenTermsForQuery,
  applyScoringHook,
  applyPostProcess,
  type CodeTokenizerOptions,
} from './extensions';

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
// v0.4 Token & prefix search modes with bounded typo tolerance (M4)
export {
  damerauLevenshteinBounded,
  damerauLevenshteinBoundedRange,
  findBestTypoWindow,
  normalizeTypoTolerance,
  allowedDistanceForTerm,
  TYPO_DISTANCE_PENALTY,
  DEFAULT_MAX_DISTANCE,
  DEFAULT_MIN_WORD_LENGTH_FOR_ONE_TYPO,
  DEFAULT_MIN_WORD_LENGTH_FOR_TWO_TYPOS,
  DEFAULT_PREFIX_EXACT_LENGTH,
  type NormalizedTypoOptions,
  type TypoWindowMatch,
} from './modes/typo-distance';
export {
  isTokenDelimiter,
  splitQueryTerms,
  normalizeTokenMatchOptions,
  scoreTokenTokens,
  type NormalizedTokenMatchOptions,
  type TokenMatchResult,
} from './modes/token-search';
export {
  normalizePrefixOptions,
  assertPrefixLengthForQuery,
  scorePrefixTokens,
  type NormalizedPrefixOptions,
  type PrefixMatchResult,
} from './modes/prefix-search';
export {
  compareParityResults,
  scoreFuzzyTokens,
  scoreSubstringTokens,
  scoreSubstringTypoTokens,
  searchCpuReference,
  searchMultiFieldCpuReference,
  WORD_BOUNDARY_PREV,
  type CpuModeOptions,
  type CpuReferenceOutput,
  type MultiFieldHit,
  type MultiFieldMatch,
  type MultiFieldCpuReferenceOutput,
  type MultiFieldRankingOptions,
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

// v0.3 First-party worker client & dedicated worker (M5)
export {
  startSearchWorker,
  isDedicatedWorker,
} from './worker/search-worker';
export {
  serializeError,
  deserializeError,
  type SerializedWorkerError,
  type WorkerInitPayload,
  type WorkerSearchPayload,
  type WorkerMutatePayload,
  type WorkerRestorePayload,
  type WorkerSerializePayload,
  type WorkerAbortPayload,
} from './worker/protocol';

// v0.3 Versioned snapshot persistence & IndexedDB storage (M6)
export {
  serializeDocumentIndex,
  deserializeDocumentSnapshot,
  deserializeDocumentSnapshotHeader,
  restoreDocumentIndex,
  type RestoredDocumentSnapshot,
} from './persistence';
export {
  DEFAULT_IDB_DATABASE_NAME,
  DEFAULT_SNAPSHOT_STORE_NAME,
  DEFAULT_DOCUMENT_STORE_NAME,
  DEFAULT_SNAPSHOT_KEY,
  openSearchDatabase,
  saveIndexToIDB,
  loadIndexFromIDB,
  deleteIndexFromIDB,
  restoreIndexFromIDB,
} from './idb-storage';



