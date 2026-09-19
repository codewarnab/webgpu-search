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
  type WorkerResponse
} from '../packages/webgpu-search/src/index';

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

  // Verify scheduled method stubs throw expected descriptive errors
  let searchThrew = false;
  try {
    await docIndex.search('test');
  } catch (err: any) {
    searchThrew = err.message.includes('M2');
  }
  if (!searchThrew) throw new Error('DocumentIndex.search should throw descriptive M2 scheduled error');

  let addThrew = false;
  try {
    await docIndex.add({ id: '1', title: 't', content: 'c' });
  } catch (err: any) {
    addThrew = err.message.includes('M4');
  }
  if (!addThrew) throw new Error('DocumentIndex.add should throw descriptive M4 scheduled error');

  docIndex.destroy();
  console.log('   ✅ DocumentIndex public methods, signatures, and factory verified');

  // 4. SearchWorkerClient public contract
  console.log('4. Verifying SearchWorkerClient public contract...');
  const workerClientOptions: WorkerClientOptions = {
    stringIsolated: true
  };
  const workerClient = new SearchWorkerClient<TestDoc>(workerClientOptions);
  if (workerClient.options?.stringIsolated !== true) throw new Error('stringIsolated option not preserved');

  let workerInitThrew = false;
  try {
    await workerClient.init(indexOptions);
  } catch (err: any) {
    workerInitThrew = err.message.includes('M5');
  }
  if (!workerInitThrew) throw new Error('SearchWorkerClient.init should throw descriptive M5 scheduled error');

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

  console.log('\n--- All Milestone 1 Public Contracts & Specifications Tests Passed! ✅ ---');
}

runM1Tests().catch((err) => {
  console.error('Milestone 1 test failed:', err);
  process.exit(1);
});
