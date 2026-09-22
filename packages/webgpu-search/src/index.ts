// High-Level Primary API
export { SearchIndex } from './search-index';
export { DocumentIndex, assertSnapshotFilterGettersSatisfied } from './document-index';
export { SearchWorkerClient, INTERNAL_WORKER_ID_KEY } from './worker/worker-client';


// Low-Level Engines & Hardware Utilities for Power Users & Benchmarks
export { WebGPUEngine, type DatasetLike, type EngineDataset, type ColdSearchResult, type WebGPUSearchResult, type SearchResult } from './webgpu-engine';
export { CPUEngine, type CPUSearchResult } from './cpu-engine';
export { GpuDevicePool, WebGPUContextManager, type AcquiredDeviceContext } from './gpu-device-pool';
export {
  packStringsToGPUBuffer,
  sanitizeStringForSlot,
  sanitizeString,
  packDataset,
  packUnicodeToGPUBuffer,
  serializeDataset,
  serializeUnicodeDataset,
  deserializeDataset,
  deserializeUnicodeDataset,
  validatePackedOffsets,
  crc32Parts,
  checkMemoryBudget,
  computeClampedHeadroomBytes,
  type PackedGPUBuffer,
  type PackedDataset,
  type PackedUnicodeBufferV2,
  type DatasetPackOptions,
  type UnicodePackOptions,
  type MemoryBudgetCheck,
  type ClampedHeadroomOptions
} from './dataset-packing';

// Unified Types
export type * from './types';

// Unicode text profile (versions, errors, caps)
export {
  DATASET_FORMAT_VERSION,
  DATASET_MAGIC,
  QUERY_TOKENS_MAX,
  RESULT_LIMIT_MAX,
  SCORING_VERSION,
  UNICODE_VERSION,
  PROFILE_TO_ENUM,
  ENUM_TO_PROFILE,
  UNICODE_VERSION_TO_ENUM,
  ENUM_TO_UNICODE_VERSION,
  SCORING_TO_ENUM,
  ENUM_TO_SCORING,
  countUnicodeCodePoints,
  normalizeCpuScorer,
  IncompatibleIndexError,
  IncompatibleOptionError,
  ProfileMismatchError,
  QueryTooLongError,
  DuplicateIdError,
  DocumentNotFoundError,
  SNAPSHOT_FORMAT_VERSION,
  SNAPSHOT_MAGIC,
  SNAPSHOT_HEADER_BYTES,
  LEGACY_SNAPSHOT_VERSION,
  LEGACY_SNAPSHOT_MAGIC,
  LEGACY_SNAPSHOT_HEADER_BYTES,
  // Deprecated aliases (removal in next major)
  FORMAT_VERSION,
  SERIALIZED_MAGIC,
  DOC_FORMAT_VERSION,
  SERIALIZED_DOC_MAGIC,
  SERIALIZED_DOC_HEADER_BYTES,
  U2D4_MAGIC, // deprecated alias for SNAPSHOT_MAGIC
  U2D4_FORMAT_VERSION, // deprecated alias
  FORMAT_VERSION_4, // deprecated alias
  DOC_FORMAT_VERSION_4, // deprecated alias
  U2D4_HEADER_BYTES, // deprecated alias
  type CpuScorer,
  type CpuAlgorithm,
  type OnQueryTooLong,
  type TextProfileId,
} from './text-profile';

// Error hierarchy
export {
  WebGPUSearchError,
  IncompatibleHookError,
  CostBudgetExceededError,
  InvalidFilterError
} from './errors';

// Structured filtering & columnar metadata
export { DocumentBitset } from './filtering/doc-bitset';
export { ColumnarStore, type ColumnarStoreOptions } from './filtering/columnar-store';
export { compileFilter } from './filtering/compile-filter';

// Facet aggregation engine
export {
  FacetEngine,
  normalizeFacetRequests,
  excludeFieldFromFilter,
  filterReferencesField,
  DEFAULT_TERMS_LIMIT,
  MAX_FACET_REQUESTS,
  MAX_RANGE_BUCKETS,
  type NormalizedFacet,
} from './faceting/facet-engine';

// Deterministic ranking & autocomplete primitives
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
  AUTOCOMPLETE_DEFAULT_LIMIT,
  AUTOCOMPLETE_DEFAULT_MODE,
  AUTOCOMPLETE_MAX_FUZZY_DISTANCE,
  normalizeAutocompleteOptions,
  SUGGEST_DEFAULT_LIMIT,
  SUGGEST_DEFAULT_MODE,
  SUGGEST_MAX_FUZZY_DISTANCE,
  normalizeSuggestOptions,
  type NormalizedAutocompleteOptions,
  type NormalizedSuggestOptions,
  type SuggestCandidateKeys,
} from './autocomplete';

// Query diagnostics, cost budgets & broad-search safeguards
export {
  BROAD_SEARCH_SELECTIVITY_THRESHOLD,
  BROAD_SEARCH_MIN_DOCS,
  BROAD_SEARCH_SHORT_QUERY_TOKENS,
  BROAD_QUERY_SELECTIVITY_THRESHOLD,
  BROAD_QUERY_MIN_DOCS,
  BROAD_QUERY_SHORT_QUERY_TOKENS,
  normalizeCostBudgetOptions,
  throwIfBudgetAborted,
  assertTimeBudget,
  assertCandidateBudget,
  computeFilterSelectivity,
  isBroadQueryHeuristic,
  isBroadSelectivity,
  broadQueryRouteWarning,
  broadSelectivityWarning,
  candidateOverflowWarning,
  type NormalizedCostBudget,
  type CandidateOverflowWarningOptions,
} from './diagnostics';

// Extensibility pipeline & safe hook architecture
export {
  defaultTokenizer,
  codeTokenizer,
  normalizeSearchHooks,
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
} from './hooks';

// Shared preprocessing + exact scorer
export {
  normalizeText,
  toWellFormedSafe,
  tokensEqual,
  LONE_SURROGATE_PATTERN,
  LONE_SURROGATE_SOURCE,
  type NormalizedText,
} from './text-normalization';
export {
  clampLimit,
  DEFAULT_LIMIT,
  VALID_SEARCH_MODES,
  assertValidMode,
  abortError,
  nowMs,
  throwIfAborted,
  isAsciiTokens,
  isPrintableAsciiTokens,
} from './guard';
// Token & prefix search with bounded typo tolerance
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
} from './search/typo-tolerance';
export {
  isTokenDelimiter,
  splitQueryTerms,
  normalizeTokenMatchOptions,
  scoreTokenTokens,
  type NormalizedTokenMatchOptions,
  type TokenMatchResult,
} from './search/token-search';
export {
  normalizePrefixOptions,
  assertPrefixLengthForQuery,
  scorePrefixTokens,
  type NormalizedPrefixOptions,
  type PrefixMatchResult,
} from './search/prefix-search';
export {
  compareExactResults,
  compareParityResults,
  scoreFuzzyTokens,
  scoreSubstringTokens,
  scoreSubstringTypoTokens,
  scoreExactMatches,
  searchCpuReference,
  scoreExactMatchesMultiField,
  searchMultiFieldCpuReference,
  WORD_BOUNDARY_PREV,
  type CpuModeOptions,
  type ExactScorerOutput,
  type CpuReferenceOutput,
  type MultiFieldHit,
  type MultiFieldMatch,
  type MultiFieldExactScorerOutput,
  type MultiFieldCpuReferenceOutput,
  type MultiFieldRankingOptions,
  type FieldScoreDefinition,
} from './exact-scorer';
export {
  CASE_FOLD_RANGES,
  CASE_FOLD_EXPANSIONS,
  CASE_FOLD_C_COUNT,
  CASE_FOLD_F_COUNT,
  CASE_FOLD_UNICODE_VERSION,
  foldCaseScalar,
  FOLD_EXPANSIONS,
  FOLD_RANGES,
  FOLD_C_COUNT,
  FOLD_F_COUNT,
  FOLD_UNICODE_VERSION,
  foldCodePoint,
} from './case-fold-table';

// Unicode-safe highlighting engine
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

// First-party worker client & dedicated worker
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

// Versioned snapshot persistence & IndexedDB storage
export {
  encodeSnapshot,
  serializeDocumentIndex,
  decodeSnapshot,
  deserializeDocumentSnapshot,
  decodeSnapshotHeader,
  deserializeDocumentSnapshotHeader,
  restoreSnapshot,
  restoreDocumentIndex,
  encodeColumnarPayload,
  validateColumnarPayload,
  MAX_SNAPSHOT_SCHEMA_BYTES,
  MAX_SNAPSHOT_COLUMNAR_BYTES,
  MAX_SNAPSHOT_DOCS_BYTES,
  MAX_SNAPSHOT_BYTES,
  MAX_SNAPSHOT_DOC_COUNT,
  MAX_SNAPSHOT_TOKEN_COUNT,
  type RestoredDocumentSnapshot,
} from './snapshot-codec';
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
} from './snapshot-idb';
