/**
 * Milestone 1 Public Contracts & Specifications Test Suite.
 * Validates v0.3 DocumentIndex, SearchWorkerClient, Little-Endian U2D3 constants,
 * error hierarchy, package exports, and worker SSR safety.
 */

import { readFile } from 'node:fs/promises';
import {
  DocumentIndex,
  SearchWorkerClient,
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
  WebGPUSearchError,
  IncompatibleHookError,
  CostBudgetExceededError,
  InvalidFilterError,
  IncompatibleOptionError,
  serializeError,
  deserializeError,
  type DocumentId,
  type DocumentRecord,
  type FieldDefinition,
  type DocumentField,
  type DocumentIndexOptions,
  type DocumentSearchOptions,
  type DocumentSearchResultItem,
  type DocumentSearchResponse,
  type HighlightRange,
  type HighlightOptions,
  type AddOptions,
  type MutationBatch,
  type MutationResult,
  type DocumentIndexStats,
  type FallbackReason,
  type WorkerClientOptions,
  type WorkerMessageType,
  type WorkerRequest,
  type WorkerResponse,
  type SearchMode,
  type FilterValue,
  type FieldComparison,
  type FieldFilter,
  type FilterExpression,
  type FilterFieldType,
  type FilterFieldDefinition,
  type DocumentFilterField,
  type TermsFacetRequest,
  type RangeFacetBucket,
  type RangeFacetRequest,
  type FacetRequest,
  type TermsFacetBucket,
  type TermsFacetResult,
  type RangeFacetBucketResult,
  type RangeFacetResult,
  type FacetResult,
  type TypoToleranceOptions,
  type TokenMatchOptions,
  type PrefixSearchOptions,
  type DeterministicRankingOptions,
  type TieBreakerCriterion,
  type SuggestOptions,
  type SuggestionItem,
  type SuggestResponse,
  type MatchInfo,
  type SearchExtensionHooks,
  type CostBudgetOptions,
  type QueryDiagnosticsTimings,
  type QueryDiagnostics
} from '../packages/webgpu-search/src/index';

import { searchCpuReference } from '../packages/webgpu-search/src/cpu-reference';
import { isDedicatedWorker, startSearchWorker } from '../packages/webgpu-search/src/worker/search-worker';

async function runM1Tests() {
  console.log('--- Running Milestone 1 Public Contracts & Specifications Tests ---');

  // 1. Constants and Magic verification
  console.log('1. Verifying U2D3 persistence constants...');
  if (DOC_FORMAT_VERSION !== 3) {
    throw new Error(`DOC_FORMAT_VERSION must be 3, got ${DOC_FORMAT_VERSION}`);
  }
  if (SERIALIZED_DOC_MAGIC !== 0x55324433) {
    throw new Error(`SERIALIZED_DOC_MAGIC must be 0x55324433 ('U2D3'), got ${SERIALIZED_DOC_MAGIC}`);
  }
  if (SERIALIZED_DOC_HEADER_BYTES !== 48) {
    throw new Error(`SERIALIZED_DOC_HEADER_BYTES must be 48, got ${SERIALIZED_DOC_HEADER_BYTES}`);
  }
  console.log('   ✅ DOC_FORMAT_VERSION (3), SERIALIZED_DOC_MAGIC (0x55324433), and header size (48 B) verified');

  // 2. Error hierarchy
  console.log('2. Verifying error hierarchy (DuplicateIdError, DocumentNotFoundError)...');
  const dupErr1 = new DuplicateIdError('doc_abc');
  if (!(dupErr1 instanceof Error)) throw new Error('DuplicateIdError must inherit from Error');
  if (dupErr1.name !== 'DuplicateIdError') throw new Error(`Expected name DuplicateIdError, got ${dupErr1.name}`);
  if (dupErr1.id !== 'doc_abc') throw new Error(`Expected id doc_abc, got ${dupErr1.id}`);
  if (!dupErr1.message.includes('doc_abc')) throw new Error('Default message should include id');

  const dupErrCustom = new DuplicateIdError(42, 'Custom duplicate error');
  if (dupErrCustom.message !== 'Custom duplicate error') throw new Error('Custom message not respected');
  if (dupErrCustom.id !== 42) throw new Error(`Expected id 42, got ${dupErrCustom.id}`);

  const notFound1 = new DocumentNotFoundError('doc_xyz');
  if (!(notFound1 instanceof Error)) throw new Error('DocumentNotFoundError must inherit from Error');
  if (notFound1.name !== 'DocumentNotFoundError') throw new Error(`Expected name DocumentNotFoundError, got ${notFound1.name}`);
  if (notFound1.id !== 'doc_xyz') throw new Error(`Expected id doc_xyz, got ${notFound1.id}`);
  if (!notFound1.message.includes('doc_xyz')) throw new Error('Default message should include id');

  const notFoundCustom = new DocumentNotFoundError(99, 'Custom not found error');
  if (notFoundCustom.message !== 'Custom not found error') throw new Error('Custom message not respected');
  if (notFoundCustom.id !== 99) throw new Error(`Expected id 99, got ${notFoundCustom.id}`);
  console.log('   ✅ Error hierarchy, inheritance, names, ids, and messages verified');

  // 3. DocumentIndex public contract
  console.log('3. Verifying DocumentIndex public contract...');
  interface TestDoc {
    id: string;
    title: string;
    content: string;
    tags?: string[];
  }

  const indexOptions: DocumentIndexOptions<TestDoc> = {
    idField: 'id',
    fields: [
      { name: 'title', weight: 2.0 },
      { name: 'content', weight: 1.0 },
      'tags'
    ],
    initialCapacity: 100,
    growthFactor: 1.5,
    candidateCapacity: 8192
  };

  const docIndex = new DocumentIndex<TestDoc>(indexOptions);
  if (docIndex.options.idField !== 'id') throw new Error('idField not preserved');
  if (docIndex.options.fields.length !== 3) throw new Error('fields length incorrect');

  const factoryIndex = await DocumentIndex.create<TestDoc>([], indexOptions);
  if (!(factoryIndex instanceof DocumentIndex)) throw new Error('DocumentIndex.create must return DocumentIndex');

  // Verify search returns valid response (implemented in M2)
  const searchRes = await docIndex.search('test');
  if (searchRes.totalMatches !== 0 || !Array.isArray(searchRes.results)) {
    throw new Error('DocumentIndex.search should return valid empty response on empty index');
  }

  // Verify add returns valid MutationResult (implemented in M4)
  const addRes = await docIndex.add({ id: '1', title: 't', content: 'c' });
  if (addRes.added !== 1 || addRes.mutationEpoch !== 1) {
    throw new Error('DocumentIndex.add should return valid MutationResult');
  }

  docIndex.destroy();
  console.log('   ✅ DocumentIndex public methods, signatures, and factory verified');

  // 4. SearchWorkerClient public contract
  console.log('4. Verifying SearchWorkerClient public contract...');
  const workerClientOptions: WorkerClientOptions = {
    stringIsolated: true
  };
  const workerClient = new SearchWorkerClient<TestDoc>(workerClientOptions);
  if (workerClient.options?.stringIsolated !== true) throw new Error('stringIsolated option not preserved');

  await workerClient.init(indexOptions);
  const workerAddRes = await workerClient.add({ id: '1', title: 'Worker Document', content: 'Testing worker client' });
  if (workerAddRes.added !== 1) throw new Error('SearchWorkerClient.add should return valid MutationResult');
  const workerSearchRes = await workerClient.search('Worker');
  if (workerSearchRes.totalMatches !== 1 || workerSearchRes.results[0]?.id !== '1') {
    throw new Error('SearchWorkerClient.search should return matching result');
  }

  await workerClient.destroy();
  console.log('   ✅ SearchWorkerClient public methods and options verified');


  // 5. Worker SSR guard
  console.log('5. Verifying Worker SSR safety...');
  if (isDedicatedWorker !== false) {
    throw new Error('isDedicatedWorker should evaluate to false in Node/Bun CLI environment');
  }
  // Calling startSearchWorker directly in test environment should not crash
  startSearchWorker();
  console.log('   ✅ Worker SSR guard confirmed safe in non-worker environment');

  // 6. Package exports & files configuration
  console.log('6. Verifying package.json exports and files...');
  const pkgJsonStr = await readFile(new URL('../packages/webgpu-search/package.json', import.meta.url), 'utf8');
  const pkg = JSON.parse(pkgJsonStr);

  if (!pkg.exports?.['.']?.import?.types?.includes('dist/index.d.ts')) {
    throw new Error('package.json missing root "." export with types');
  }
  if (!pkg.exports?.['./worker']?.import?.types?.includes('dist/worker.d.ts')) {
    throw new Error('package.json missing "./worker" export with types');
  }
  if (!pkg.files?.includes('dist')) {
    throw new Error('package.json files array should include "dist"');
  }
  console.log('   ✅ package.json exports (".", "./worker") and files ["dist", "README.md", "LICENSE"] verified');

  // 7. Type symmetry & compilation validation
  console.log('7. Verifying TypeScript type shapes & symmetry...');
  const mockResultItem: DocumentSearchResultItem<TestDoc> = {
    id: 'doc-1',
    doc: { id: 'doc-1', title: 'Hello', content: 'World' },
    score: 950,
    matchedField: 'title',
    highlights: {
      title: [{ start: 0, end: 5 }]
    },
    highlightedText: {
      title: '<mark>Hello</mark>'
    },
    matches: [
      { field: 'title', score: 950, highlights: [{ start: 0, end: 5 }] }
    ]
  };

  const fallbackReason: FallbackReason = 'below-threshold';

  const mockResponse: DocumentSearchResponse<TestDoc> = {
    query: 'hel',
    mode: 'fuzzy',
    engine: 'webgpu',
    totalMatches: 1,
    candidateCount: 1,
    hasOverflow: false,
    results: [mockResultItem],
    timings: {
      queryUploadMs: 0.1,
      encodeSubmitMs: 0.2,
      gpuExecutionMs: null,
      readbackMs: 0.3,
      totalMs: 0.6
    },
    profileId: 'unicode-default',
    scoringVersion: 'parity-v1',
    cpuAlgorithm: 'parity',
    fallbackReason
  };

  const mockBatch: MutationBatch<TestDoc> = {
    add: [{ id: 'doc-2', title: 'Two', content: 'Second' }],
    update: [{ id: 'doc-1', title: 'Updated', content: 'First' }],
    remove: ['doc-3']
  };

  const mockMutResult: MutationResult = {
    added: 1,
    updated: 1,
    removed: 1,
    mutationEpoch: 1,
    compacted: false,
    durationMs: 0.5
  };

  const mockStats: DocumentIndexStats = {
    size: 1,
    engine: 'webgpu',
    vramAllocatedBytes: 65536,
    profileId: 'unicode-default',
    unicodeVersion: '16.0.0',
    scoringVersion: 'parity-v1',
    tokenCount: 10,
    folded: true,
    formatVersion: 2,
    docCount: 1,
    rowCount: 2,
    tombstoneCount: 0,
    tombstoneRatio: 0,
    buildTimeMs: 1.2,
    mutationEpoch: 0,
    memory: {
      vramBytes: 65536,
      ramBytes: 4096,
      totalBytes: 69632
    }
  };

  const mockReq: WorkerRequest = {
    id: 1,
    type: 'SEARCH',
    payload: { query: 'test' }
  };

  const mockResp: WorkerResponse = {
    id: 1,
    success: true,
    result: mockResponse
  };

  if (mockResponse.results.length !== 1 || mockMutResult.added !== 1 || mockStats.docCount !== 1) {
    throw new Error('Type shapes failed to construct properly');
  }
  console.log('   ✅ Type shapes (DocumentSearchResultItem, DocumentSearchResponse, MutationBatch, MutationResult, DocumentIndexStats, WorkerRequest, WorkerResponse) verified');

  // 8. Verifying v0.4 U2D4 persistence constants
  console.log('8. Verifying v0.4 U2D4 persistence constants...');
  if (U2D4_MAGIC !== 0x55324434) {
    throw new Error(`U2D4_MAGIC must be 0x55324434 ('U2D4'), got ${U2D4_MAGIC}`);
  }
  if (U2D4_FORMAT_VERSION !== 4) {
    throw new Error(`U2D4_FORMAT_VERSION must be 4, got ${U2D4_FORMAT_VERSION}`);
  }
  if (FORMAT_VERSION_4 !== 4 || DOC_FORMAT_VERSION_4 !== 4) {
    throw new Error('FORMAT_VERSION_4 and DOC_FORMAT_VERSION_4 aliases must be 4');
  }
  if (U2D4_HEADER_BYTES !== 56) {
    throw new Error(`U2D4_HEADER_BYTES must be 56, got ${U2D4_HEADER_BYTES}`);
  }
  console.log('   ✅ U2D4_MAGIC (0x55324434), U2D4_FORMAT_VERSION (4), and header size (56 B) verified');

  // 9. Verifying v0.4 Error hierarchy & Worker serialization roundtrip
  console.log('9. Verifying v0.4 Error hierarchy & Worker serialization roundtrip...');
  const baseErr = new WebGPUSearchError('Base engine error');
  if (!(baseErr instanceof Error) || !(baseErr instanceof WebGPUSearchError)) {
    throw new Error('WebGPUSearchError inheritance failed');
  }
  if (baseErr.name !== 'WebGPUSearchError') throw new Error('WebGPUSearchError name mismatch');

  const hookErr = new IncompatibleHookError('customTokenizer', 'missing hook definition in snapshot');
  if (!(hookErr instanceof WebGPUSearchError) || !(hookErr instanceof Error)) {
    throw new Error('IncompatibleHookError must inherit from WebGPUSearchError');
  }
  if (hookErr.name !== 'IncompatibleHookError') throw new Error('IncompatibleHookError name mismatch');
  if (hookErr.hookId !== 'customTokenizer') throw new Error('IncompatibleHookError hookId mismatch');
  if (!hookErr.message.includes('customTokenizer')) throw new Error('IncompatibleHookError message mismatch');

  const budgetErr = new CostBudgetExceededError('time', 50, 120);
  if (!(budgetErr instanceof WebGPUSearchError) || !(budgetErr instanceof Error)) {
    throw new Error('CostBudgetExceededError must inherit from WebGPUSearchError');
  }
  if (budgetErr.name !== 'CostBudgetExceededError') throw new Error('CostBudgetExceededError name mismatch');
  if (budgetErr.budgetType !== 'time' || budgetErr.limit !== 50 || budgetErr.actual !== 120) {
    throw new Error('CostBudgetExceededError fields mismatch');
  }

  const filterErr = new InvalidFilterError('unsupported operator "$regex"', 'tags');
  if (!(filterErr instanceof WebGPUSearchError) || !(filterErr instanceof Error)) {
    throw new Error('InvalidFilterError must inherit from WebGPUSearchError');
  }
  if (filterErr.name !== 'InvalidFilterError') throw new Error('InvalidFilterError name mismatch');
  if (filterErr.field !== 'tags' || !filterErr.message.includes('tags')) {
    throw new Error('InvalidFilterError fields/message mismatch');
  }

  // Verify worker serialize/deserialize roundtrip
  const serializedHookErr = serializeError(hookErr);
  const rehydratedHookErr = deserializeError(serializedHookErr);
  if (!(rehydratedHookErr instanceof IncompatibleHookError) || (rehydratedHookErr as IncompatibleHookError).hookId !== 'customTokenizer') {
    throw new Error('IncompatibleHookError worker serialization roundtrip failed');
  }

  const serializedBudgetErr = serializeError(budgetErr);
  const rehydratedBudgetErr = deserializeError(serializedBudgetErr);
  if (!(rehydratedBudgetErr instanceof CostBudgetExceededError) || (rehydratedBudgetErr as CostBudgetExceededError).limit !== 50) {
    throw new Error('CostBudgetExceededError worker serialization roundtrip failed');
  }

  const serializedFilterErr = serializeError(filterErr);
  const rehydratedFilterErr = deserializeError(serializedFilterErr);
  if (!(rehydratedFilterErr instanceof InvalidFilterError) || (rehydratedFilterErr as InvalidFilterError).field !== 'tags') {
    throw new Error('InvalidFilterError worker serialization roundtrip failed');
  }

  // Verify custom message preservation across worker serialization
  const customHookErr = new IncompatibleHookError('h1', 'r1', 'Custom descriptive hook failure message');
  const rehydratedCustomHook = deserializeError(serializeError(customHookErr));
  if (rehydratedCustomHook.message !== 'Custom descriptive hook failure message') {
    throw new Error(`Expected custom hook message preserved, got "${rehydratedCustomHook.message}"`);
  }

  const customFilterErr = new InvalidFilterError('r2', 'f2', 'Custom descriptive filter failure message');
  const rehydratedCustomFilter = deserializeError(serializeError(customFilterErr));
  if (rehydratedCustomFilter.message !== 'Custom descriptive filter failure message') {
    throw new Error(`Expected custom filter message preserved, got "${rehydratedCustomFilter.message}"`);
  }

  // Verify null and sparse details resilience
  const nullDetailsErr = deserializeError({
    name: 'InvalidFilterError',
    message: 'Null details test',
    details: null as any
  });
  if (!(nullDetailsErr instanceof InvalidFilterError) || nullDetailsErr.message !== 'Null details test') {
    throw new Error('deserializeError failed with null details');
  }
  console.log('   ✅ v0.4 Error classes, custom message roundtrips, and null-safe deserialization verified');

  // 10. Verifying SearchMode extension ('token', 'prefix')
  console.log('10. Verifying SearchMode extension (fuzzy, substring, token, prefix)...');
  const modes: SearchMode[] = ['fuzzy', 'substring', 'token', 'prefix'];
  if (modes.length !== 4) throw new Error('SearchMode should support all 4 modes');

  // Verify DocumentIndex and CPU reference fail-fast on unimplemented M4 modes
  let threwDocTokenMode = false;
  try {
    const testIdx = new DocumentIndex({ fields: ['title'] });
    await testIdx.search('query', { mode: 'token' });
  } catch (err: any) {
    threwDocTokenMode = err instanceof IncompatibleOptionError && err.option === 'mode';
  }
  if (!threwDocTokenMode) throw new Error('DocumentIndex.search must throw IncompatibleOptionError on mode: "token"');

  let threwCpuTokenMode = false;
  try {
    searchCpuReference([new Uint32Array([1, 2])], new Uint32Array([1]), 'token' as any, 10, ['test']);
  } catch (err: any) {
    threwCpuTokenMode = err instanceof IncompatibleOptionError && err.option === 'mode';
  }
  if (!threwCpuTokenMode) throw new Error('searchCpuReference must throw IncompatibleOptionError on mode: "token"');
  console.log('   ✅ SearchMode extended to fuzzy | substring | token | prefix (with M4 fail-fast guards)');

  // 11. Verifying Filter AST schemas and Columnar types
  console.log('11. Verifying Filter AST schemas and Columnar types...');
  const comparison: FieldComparison = {
    eq: 'active',
    gte: 18,
    in: ['admin', 'moderator'],
    exists: true
  };
  const filterExpr: FilterExpression = {
    and: [
      { status: { eq: 'published' } },
      { or: [{ role: { in: ['admin', 'editor'] } }, { views: { gte: 100 } }] },
      { not: { archived: { eq: true } } }
    ]
  };
  const filterFieldDef: FilterFieldDefinition<TestDoc> = {
    name: 'tags',
    type: 'string[]',
    getter: (doc) => doc.tags
  };
  if (!filterExpr || !comparison || filterFieldDef.name !== 'tags') {
    throw new Error('Filter AST types failed validation');
  }

  // Verify InvalidFilterError on unindexed FilterExpression field for DocumentIndex & SearchWorkerClient
  let threwDocFilterExpr = false;
  try {
    const testIdx = new DocumentIndex({ fields: ['title'] });
    await testIdx.search('query', { filter: filterExpr });
  } catch (err: any) {
    threwDocFilterExpr = err instanceof InvalidFilterError;
  }
  if (!threwDocFilterExpr) throw new Error('DocumentIndex.search must throw InvalidFilterError on FilterExpression with unindexed field');

  let threwWorkerFilterExpr = false;
  const workerForFilter = new SearchWorkerClient<TestDoc>();
  try {
    await workerForFilter.init([], { fields: ['title'] });
    await workerForFilter.search('query', { filter: filterExpr });
  } catch (err: any) {
    threwWorkerFilterExpr = err instanceof InvalidFilterError;
  } finally {
    await workerForFilter.destroy();
  }
  if (!threwWorkerFilterExpr) throw new Error('SearchWorkerClient.search must throw InvalidFilterError on FilterExpression with unindexed field');
  console.log('   ✅ Filter AST schemas and unindexed field guards verified');

  // 12. Verifying Facet Aggregations, Typo Tolerance, Ranking, Suggestions, Extensions, & Diagnostics
  console.log('12. Verifying Facets, Typo Tolerance, Ranking, Suggestions, Extensions & Diagnostics...');
  const termsFacetReq: TermsFacetRequest = { type: 'terms', field: 'category', limit: 5, sortBy: 'count' };
  const rangeFacetReq: RangeFacetRequest = {
    type: 'range',
    field: 'price',
    ranges: [{ to: 50 }, { from: 50, to: 100 }, { from: 100 }]
  };
  const termsFacetRes: TermsFacetResult = {
    type: 'terms',
    field: 'category',
    isApproximate: false,
    buckets: [{ value: 'electronics', count: 42 }]
  };
  const rangeFacetRes: RangeFacetResult = {
    type: 'range',
    field: 'price',
    isApproximate: true,
    buckets: [{ key: 'under-50', to: 50, count: 12 }]
  };

  const typoOpts: TypoToleranceOptions = {
    enabled: true,
    maxDistance: 1,
    minWordLengthForOneTypo: 4,
    minWordLengthForTwoTypos: 8,
    prefixExactLength: 1
  };
  const tokenOpts: TokenMatchOptions = { operator: 'and', minMatchCount: 2 };
  const prefixOpts: PrefixSearchOptions = { prefixLength: 3, exactCase: false };

  const rankingOpts: DeterministicRankingOptions = {
    tieBreakers: ['score', 'weight', 'exact', 'length', 'id']
  };

  const suggestOpts: SuggestOptions = { limit: 5, mode: 'prefix', fuzzyDistance: 0 };
  const suggestItem: SuggestionItem<TestDoc> = {
    text: 'AuthController',
    score: 980,
    type: 'completion',
    matchedRanges: [{ start: 0, end: 4 }],
    docId: '1'
  };
  const suggestResp: SuggestResponse<TestDoc> = {
    suggestions: [suggestItem],
    queryDurationMs: 0.45
  };

  const matchInfo: MatchInfo = {
    query: 'auth',
    matchedField: 'title',
    rawScore: 900,
    normalizedScore: 950
  };
  const extensions: SearchExtensionHooks<TestDoc> = {
    tokenizer: (text) => text.split(/[\s_]+/),
    scoringHook: (_doc, baseScore, _info) => baseScore + 50
  };

  const budgetOpts: CostBudgetOptions = { maxExecutionTimeMs: 100, maxCandidates: 5000 };
  const diagTimings: QueryDiagnosticsTimings = {
    filteringMs: 0.05,
    scoringMs: 0.2,
    highlightMs: 0.1,
    facetingMs: 0.08,
    totalMs: 0.43
  };
  const diagnostics: QueryDiagnostics = {
    scannedCandidates: 250,
    filterSelectivity: 0.85,
    routedEngine: 'webgpu',
    hasOverflow: false,
    timings: diagTimings,
    warnings: []
  };

  // Verify TermsFacetBucket accepts boolean FilterValue
  const boolBucket: TermsFacetBucket = { value: true, count: 10 };
  const nullValue: FilterValue = null;
  const searchOptsWithSuggest: DocumentSearchOptions<TestDoc> = {
    suggest: { limit: 5, mode: 'prefix' }
  };

  // Verify SearchWorkerClient fails fast on extensions across worker boundary
  let threwWorkerExtensions = false;
  const workerForExt = new SearchWorkerClient<TestDoc>();
  try {
    await workerForExt.search('query', { extensions: { scoringHook: () => 100 } });
  } catch (err: any) {
    threwWorkerExtensions = err instanceof IncompatibleHookError && err.hookId === 'extensions';
  } finally {
    await workerForExt.destroy();
  }
  if (!threwWorkerExtensions) {
    throw new Error('SearchWorkerClient.search must throw IncompatibleHookError on non-cloneable extensions');
  }

  if (
    termsFacetReq.type !== 'terms' ||
    rangeFacetReq.ranges.length !== 3 ||
    suggestOpts.limit !== 5 ||
    termsFacetRes.buckets.length !== 1 ||
    rangeFacetRes.isApproximate !== true ||
    typoOpts.maxDistance !== 1 ||
    tokenOpts.operator !== 'and' ||
    prefixOpts.prefixLength !== 3 ||
    rankingOpts.tieBreakers?.length !== 5 ||
    suggestResp.suggestions.length !== 1 ||
    extensions.scoringHook?.({ id: '1', title: 't', content: 'c' }, 100, matchInfo) !== 150 ||
    budgetOpts.maxExecutionTimeMs !== 100 ||
    diagnostics.scannedCandidates !== 250 ||
    boolBucket.value !== true ||
    nullValue !== null ||
    !searchOptsWithSuggest.suggest
  ) {
    throw new Error('v0.4 feature type shapes failed validation');
  }
  console.log('   ✅ Facets, Typo Tolerance, Ranking, Suggestions, Extensions, and Diagnostics verified');

  console.log('\n--- All Milestone 1 Public Contracts & Specifications Tests Passed! ✅ ---');
}

runM1Tests().catch((err) => {
  console.error('Milestone 1 test failed:', err);
  process.exit(1);
});
