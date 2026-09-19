import assert from 'node:assert';
import {
  SearchWorkerClient,
  startSearchWorker,
  isDedicatedWorker,
  QueryTooLongError,
  DuplicateIdError,
  DocumentNotFoundError,
  serializeError,
  deserializeError,
  type WorkerClientOptions,
  type DocumentIndexOptions,
  type DocumentSearchResponse,
  type MutationBatch
} from '../packages/webgpu-search/src/index';

interface BookDoc {
  id: string;
  title: string;
  author: string;
  summary: string;
  complexMetadata?: { [key: string]: unknown };
}

async function runM5Tests() {
  console.log('--- Running Milestone 5: First-Party Worker Client & Protocol Tests ---');

  // =========================================================================
  // 1. SSR & Main-Thread Safety
  // =========================================================================
  console.log('1. Testing SSR and main-thread safety guards...');
  {
    assert.strictEqual(
      isDedicatedWorker,
      false,
      'isDedicatedWorker must evaluate to false in Node/Bun CLI main environment'
    );

    // Calling startSearchWorker in main environment without worker scope must be a safe no-op
    assert.doesNotThrow(() => {
      startSearchWorker();
    }, 'startSearchWorker() without worker scope must not throw or attach listeners to global');

    console.log('   ✅ SSR and main-thread safety confirmed');
  }

  // =========================================================================
  // 2. Error Serialization & Rehydration Symmetry
  // =========================================================================
  console.log('2. Testing error serialization & rehydration symmetry...');
  {
    // QueryTooLongError
    const qErr = new QueryTooLongError(128, 142, 'unicode-default');
    const serQ = serializeError(qErr);
    const reQ = deserializeError(serQ);
    assert(reQ instanceof QueryTooLongError, 'rehydrated error must be instanceof QueryTooLongError');
    assert.strictEqual((reQ as QueryTooLongError).limit, 128);
    assert.strictEqual((reQ as QueryTooLongError).actual, 142);
    assert.strictEqual((reQ as QueryTooLongError).profileId, 'unicode-default');

    // DuplicateIdError
    const dupErr = new DuplicateIdError('doc-42', 'Duplicate document ID: doc-42');
    const serDup = serializeError(dupErr);
    const reDup = deserializeError(serDup);
    assert(reDup instanceof DuplicateIdError, 'rehydrated error must be instanceof DuplicateIdError');
    assert.strictEqual((reDup as DuplicateIdError).id, 'doc-42');
    assert.strictEqual(reDup.message, 'Duplicate document ID: doc-42');

    // DocumentNotFoundError
    const notFoundErr = new DocumentNotFoundError('doc-99');
    const serNF = serializeError(notFoundErr);
    const reNF = deserializeError(serNF);
    assert(reNF instanceof DocumentNotFoundError, 'rehydrated error must be instanceof DocumentNotFoundError');
    assert.strictEqual((reNF as DocumentNotFoundError).id, 'doc-99');

    // AbortError
    const abErr = new Error('Search aborted');
    abErr.name = 'AbortError';
    const serAb = serializeError(abErr);
    const reAb = deserializeError(serAb);
    assert.strictEqual(reAb.name, 'AbortError');

    // Standard TypeError and RangeError
    const typeErr = new TypeError('Type mismatch');
    const reType = deserializeError(serializeError(typeErr));
    assert(reType instanceof TypeError);
    assert.strictEqual(reType.message, 'Type mismatch');

    console.log('   ✅ Error serialization & class rehydration verified 100%');
  }

  // =========================================================================
  // 3. Worker Client Initialization & Search Execution
  // =========================================================================
  console.log('3. Testing SearchWorkerClient initialization and search execution...');
  {
    const initialBooks: BookDoc[] = [
      {
        id: 'book-1',
        title: 'The Rust Programming Language',
        author: 'Steve Klabnik and Carol Nichols',
        summary: 'Systems programming with fearless concurrency and memory safety'
      },
      {
        id: 'book-2',
        title: 'Designing Data-Intensive Applications',
        author: 'Martin Kleppmann',
        summary: 'The big ideas behind reliable, scalable, and maintainable systems'
      },
      {
        id: 'book-3',
        title: 'WebGPU Shading Language Reference',
        author: 'W3C GPU for the Web Community Group',
        summary: 'WGSL compute shaders, bind groups, pipeline states, and buffer layouts'
      }
    ];

    const client = new SearchWorkerClient<BookDoc>({
      stringIsolated: true
    });

    await client.init(initialBooks, {
      fields: [
        { name: 'title', weight: 2.0 },
        { name: 'summary', weight: 1.0 },
        { name: 'author', weight: 0.8 }
      ],
      preferGpu: false
    });

    // Verify search with highlighting
    const res = await client.search('concurrency', {
      tag: 'mark'
    });

    assert.strictEqual(res.totalMatches, 1);
    assert.strictEqual(res.results.length, 1);
    const hit = res.results[0];
    assert.strictEqual(hit.id, 'book-1');
    assert.strictEqual(hit.matchedField, 'summary');
    assert.strictEqual(hit.doc.title, 'The Rust Programming Language');
    assert(hit.highlights?.summary && hit.highlights.summary.length > 0);
    assert(hit.highlightedText?.summary && hit.highlightedText.summary.includes('<mark>'));

    // Search for non-existent term
    const noHitRes = await client.search('nonexistentterm12345');
    assert.strictEqual(noHitRes.totalMatches, 0);
    assert.strictEqual(noHitRes.results.length, 0);

    await client.destroy();
    console.log('   ✅ Worker client initialization and search execution verified');
  }

  // =========================================================================
  // 4. String-Isolated Enrichment & Non-Cloneable Document Safety
  // =========================================================================
  console.log('4. Testing string-isolated enrichment with custom getters and complex objects...');
  {
    interface ComplexDoc {
      id: string;
      title: string;
      tags: string[];
      nested: { value: string };
      nonCloneableFn: () => void;
      symbolProp: symbol;
    }

    const docs: ComplexDoc[] = [
      {
        id: 'c-1',
        title: 'Alpha Stream',
        tags: ['streaming', 'realtime'],
        nested: { value: 'High throughput pipeline' },
        nonCloneableFn: () => { console.log('not cloneable'); },
        symbolProp: Symbol('secret')
      },
      {
        id: 'c-2',
        title: 'Beta Batch',
        tags: ['batch', 'offline'],
        nested: { value: 'MapReduce processing engine' },
        nonCloneableFn: () => { console.log('also not cloneable'); },
        symbolProp: Symbol('beta')
      }
    ];

    const client = new SearchWorkerClient<ComplexDoc>({
      stringIsolated: true
    });

    // Custom getters and function idField
    await client.init(docs, {
      idField: (d) => d.id,
      fields: [
        { name: 'title', weight: 2.0 },
        { name: 'tags', getter: (d) => d.tags },
        { name: 'nestedVal', getter: (d) => d.nested.value, weight: 1.5 }
      ],
      preferGpu: false
    });

    const searchRes = await client.search('throughput');
    assert.strictEqual(searchRes.totalMatches, 1);
    const matchedItem = searchRes.results[0];
    assert.strictEqual(matchedItem.id, 'c-1');
    // Verify that the original complex document was enriched and preserved intact
    assert.strictEqual(matchedItem.doc, docs[0], 'enriched doc must match original object reference');
    assert.strictEqual(typeof matchedItem.doc.nonCloneableFn, 'function');
    assert.strictEqual(typeof matchedItem.doc.symbolProp, 'symbol');

    // Predicate filter on main thread
    const filteredSearch = await client.search('Stream', {
      filter: (d) => d.tags.includes('offline')
    });
    assert.strictEqual(filteredSearch.totalMatches, 0, 'filter should exclude non-matching tags');

    const passFilter = await client.search('Stream', {
      filter: (d) => d.tags.includes('streaming')
    });
    assert.strictEqual(passFilter.totalMatches, 1);
    assert.strictEqual(passFilter.results[0].id, 'c-1');

    await client.destroy();
    console.log('   ✅ String-isolated enrichment and complex object safety verified');
  }

  // =========================================================================
  // 5. Monotonic Query Sequencing & Immediate AbortError Rejection
  // =========================================================================
  console.log('5. Testing monotonic query sequencing and AbortError rejection...');
  {
    const client = new SearchWorkerClient<BookDoc>({
      stringIsolated: true
    });

    const testDocs: BookDoc[] = Array.from({ length: 50 }, (_, i) => ({
      id: `doc-${i}`,
      title: `Document ${i} with searchable terminology and keywords`,
      author: `Author ${i % 5}`,
      summary: `Summary containing indexable text content ${i}`
    }));

    await client.init(testDocs, {
      fields: ['title', 'summary'],
      preferGpu: false
    });

    // Fire 5 rapid queries in sequence without awaiting
    const p1 = client.search('Doc');
    const p2 = client.search('Docu');
    const p3 = client.search('Docum');
    const p4 = client.search('Docume');
    const p5 = client.search('Document');

    let abortCount = 0;
    const errors: any[] = [];

    const handleAbort = (p: Promise<DocumentSearchResponse<BookDoc>>) =>
      p.then(
        () => { /* resolved */ },
        (err) => {
          if (err?.name === 'AbortError') {
            abortCount++;
          }
          errors.push(err);
        }
      );

    await Promise.all([
      handleAbort(p1),
      handleAbort(p2),
      handleAbort(p3),
      handleAbort(p4)
    ]);

    // All 4 prior queries must be aborted immediately
    assert.strictEqual(abortCount, 4, 'All superseded queries must reject with AbortError');

    // The 5th (final) query must resolve successfully
    const finalRes = await p5;
    assert(finalRes.results.length > 0, 'Final query must resolve successfully');
    assert.strictEqual(finalRes.query, 'Document');

    await client.destroy();
    console.log('   ✅ Monotonic query sequencing and immediate AbortError rejection verified');
  }

  // =========================================================================
  // 6. Caller AbortSignal Rejection
  // =========================================================================
  console.log('6. Testing caller AbortSignal cancellation...');
  {
    const client = new SearchWorkerClient<BookDoc>();
    await client.init([
      { id: '1', title: 'Test Document', author: 'Author', summary: 'Summary' }
    ], {
      fields: ['title'],
      preferGpu: false
    });

    // 1. Pre-aborted signal
    let preAbortedThrew = false;
    try {
      await client.search('Test', { signal: AbortSignal.abort() });
    } catch (err: any) {
      preAbortedThrew = err?.name === 'AbortError';
    }
    assert.strictEqual(preAbortedThrew, true, 'Pre-aborted signal must throw AbortError immediately');

    // 2. In-flight abort
    const controller = new AbortController();
    const searchPromise = client.search('Test', { signal: controller.signal });
    controller.abort();

    let inFlightThrew = false;
    try {
      await searchPromise;
    } catch (err: any) {
      inFlightThrew = err?.name === 'AbortError';
    }
    assert.strictEqual(inFlightThrew, true, 'In-flight abort must reject promise with AbortError');

    await client.destroy();
    console.log('   ✅ Caller AbortSignal cancellation verified');
  }

  // =========================================================================
  // 7. Dynamic Batched Mutations across Worker Boundary
  // =========================================================================
  console.log('7. Testing dynamic batched mutations across worker boundary...');
  {
    const client = new SearchWorkerClient<BookDoc>();
    await client.init([
      { id: '1', title: 'Initial Title', author: 'Author A', summary: 'Summary A' }
    ], {
      fields: ['title', 'author', 'summary'],
      preferGpu: false
    });

    // A. Add single and batch
    const addRes = await client.add([
      { id: '2', title: 'Second Title', author: 'Author B', summary: 'Summary B' },
      { id: '3', title: 'Third Title', author: 'Author C', summary: 'Summary C' }
    ]);
    assert.strictEqual(addRes.added, 2);
    assert.strictEqual(addRes.mutationEpoch, 1);

    // Verify added items searchable
    const searchAfterAdd = await client.search('Second');
    assert.strictEqual(searchAfterAdd.totalMatches, 1);
    assert.strictEqual(searchAfterAdd.results[0].id, '2');

    // B. Update
    const updateRes = await client.update({
      id: '2',
      title: 'Updated Second Title',
      author: 'Author B',
      summary: 'Brand new summary'
    });
    assert.strictEqual(updateRes.updated, 1);
    assert.strictEqual(updateRes.mutationEpoch, 2);

    const searchAfterUpdate = await client.search('Updated');
    assert.strictEqual(searchAfterUpdate.totalMatches, 1);
    assert.strictEqual(searchAfterUpdate.results[0].id, '2');
    assert.strictEqual(searchAfterUpdate.results[0].doc.title, 'Updated Second Title');

    // C. Remove
    const removeRes = await client.remove('1');
    assert.strictEqual(removeRes.removed, 1);
    assert.strictEqual(removeRes.mutationEpoch, 3);

    const searchAfterRemove = await client.search('Initial');
    assert.strictEqual(searchAfterRemove.totalMatches, 0);

    // D. Batch with remove, update, add
    const batchRes = await client.applyBatch({
      remove: ['2'],
      update: [{ id: '3', title: 'Renamed Third Title', author: 'Author C', summary: 'Summary C' }],
      add: [{ id: '4', title: 'Fourth Title', author: 'Author D', summary: 'Summary D' }]
    });

    assert.strictEqual(batchRes.removed, 1);
    assert.strictEqual(batchRes.updated, 1);
    assert.strictEqual(batchRes.added, 1);
    assert.strictEqual(batchRes.mutationEpoch, 4);

    const sRenamed = await client.search('Renamed');
    assert.strictEqual(sRenamed.totalMatches, 1);
    assert.strictEqual(sRenamed.results[0].id, '3');

    const sFourth = await client.search('Fourth');
    assert.strictEqual(sFourth.totalMatches, 1);
    assert.strictEqual(sFourth.results[0].id, '4');

    await client.destroy();
    console.log('   ✅ Dynamic batched mutations across worker verified');
  }

  // =========================================================================
  // 8. Two-Phase Validation & Atomic Failure Guarantees
  // =========================================================================
  console.log('8. Testing two-phase validation & atomic failure guarantees...');
  {
    const client = new SearchWorkerClient<BookDoc>();
    await client.init([
      { id: '1', title: 'Existing Item', author: 'Author 1', summary: 'Summary 1' }
    ], {
      fields: ['title'],
      preferGpu: false
    });

    // 1. DuplicateIdError on add without upsert
    let dupThrew = false;
    try {
      await client.add({ id: '1', title: 'Duplicate', author: 'Author', summary: 'Summary' });
    } catch (err: any) {
      dupThrew = err instanceof DuplicateIdError;
      assert.strictEqual((err as DuplicateIdError).id, '1');
    }
    assert.strictEqual(dupThrew, true, 'Adding existing ID without upsert must throw DuplicateIdError');

    // Verify index is still consistent and uncorrupted
    const stats1 = await client.getStats();
    assert.strictEqual(stats1.docCount, 1);
    assert.strictEqual(stats1.mutationEpoch, 0);

    // 2. DocumentNotFoundError on update non-existent ID
    let notFoundThrew = false;
    try {
      await client.update({ id: 'non-existent-99', title: 'Non-existent', author: 'A', summary: 'S' });
    } catch (err: any) {
      notFoundThrew = err instanceof DocumentNotFoundError;
      assert.strictEqual((err as DocumentNotFoundError).id, 'non-existent-99');
    }
    assert.strictEqual(notFoundThrew, true, 'Updating non-existent ID must throw DocumentNotFoundError');

    // 3. Atomic batch failure: if one item fails validation, none are applied
    let batchThrew = false;
    try {
      await client.applyBatch({
        add: [
          { id: 'valid-new-item', title: 'Valid', author: 'A', summary: 'S' },
          { id: '1', title: 'Duplicate Collision', author: 'A', summary: 'S' }
        ]
      });
    } catch (err: any) {
      batchThrew = err instanceof DuplicateIdError;
    }
    assert.strictEqual(batchThrew, true, 'Batch with duplicate ID must fail validation');

    // 'valid-new-item' must NOT have been added to index or docMap
    const searchCheck = await client.search('Valid');
    assert.strictEqual(searchCheck.totalMatches, 0, 'Partially added item must not exist in index');

    await client.destroy();
    console.log('   ✅ Two-phase validation and atomic failure guarantees verified');
  }

  // =========================================================================
  // 9. Query Too Long Error Rehydration across Worker
  // =========================================================================
  console.log('9. Testing QueryTooLongError rehydration across worker boundary...');
  {
    const client = new SearchWorkerClient<BookDoc>();
    await client.init([
      { id: '1', title: 'Short Document', author: 'Author', summary: 'Summary' }
    ], {
      fields: ['title'],
      preferGpu: false
    });

    const longQuery = 'word '.repeat(135);
    let qTooLongThrew = false;
    try {
      await client.search(longQuery, { onQueryTooLong: 'throw' });
    } catch (err: any) {
      qTooLongThrew = err instanceof QueryTooLongError;
      assert.strictEqual(err.limit, 128);
      assert(err.actual > 128);
      assert.strictEqual(err.profileId, 'unicode-default');
    }
    assert.strictEqual(qTooLongThrew, true, 'Exceeding 128 tokens must throw rehydrated QueryTooLongError');

    await client.destroy();
    console.log('   ✅ QueryTooLongError rehydration verified');
  }

  // =========================================================================
  // 10. Transferable Buffer Safety (serialize & restore)
  // =========================================================================
  console.log('10. Testing transferable buffer safety (serialize & restore)...');
  {
    const client = new SearchWorkerClient<BookDoc>();
    await client.init([
      { id: '1', title: 'Doc 1', author: 'A', summary: 'S' }
    ], {
      fields: ['title'],
      preferGpu: false
    });

    // Test serialize (scheduled for M6)
    let serializeThrewM6 = false;
    try {
      await client.serialize();
    } catch (err: any) {
      serializeThrewM6 = err?.message?.includes('M6') ?? false;
    }
    assert.strictEqual(serializeThrewM6, true, 'client.serialize() surfaces M6 scheduled error from worker');

    // Test restore non-transfer buffer preservation
    const dummyBuffer = new ArrayBuffer(64);
    assert.strictEqual(dummyBuffer.byteLength, 64);

    let restoreThrewM6 = false;
    try {
      await client.restore(dummyBuffer); // default: transfer: false (clone)
    } catch (err: any) {
      restoreThrewM6 = err?.message?.includes('M6') ?? false;
    }
    assert.strictEqual(restoreThrewM6, true);
    // Crucial check: original dummyBuffer was NOT neutered/detached!
    assert.strictEqual(dummyBuffer.byteLength, 64, 'Caller buffer must remain intact without { transfer: true }');

    // Test restore with transfer: true
    const transferredBuffer = new ArrayBuffer(64);
    try {
      await client.restore(transferredBuffer, { transfer: true });
    } catch {}
    // In transfer mode, buffer is detached
    assert.strictEqual(transferredBuffer.byteLength, 0, 'Buffer must be neutered when { transfer: true }');

    await client.destroy();
    console.log('   ✅ Transferable buffer safety and options verified');
  }

  // =========================================================================
  // 11. Telemetry & Stats Reporting across Worker
  // =========================================================================
  console.log('11. Testing getStats telemetry reporting across worker...');
  {
    const client = new SearchWorkerClient<BookDoc>();
    await client.init([
      { id: '1', title: 'First Book', author: 'Author', summary: 'Summary' },
      { id: '2', title: 'Second Book', author: 'Author', summary: 'Summary' }
    ], {
      fields: ['title', 'summary'],
      preferGpu: false
    });

    const stats = await client.getStats();
    assert.strictEqual(stats.docCount, 2);
    assert.strictEqual(stats.rowCount, 4); // 2 docs * 2 fields
    assert.strictEqual(stats.tombstoneCount, 0);
    assert.strictEqual(stats.mutationEpoch, 0);
    assert.strictEqual(stats.engine, 'cpu');
    assert(stats.memory.ramBytes > 0);

    await client.destroy();
    console.log('   ✅ getStats telemetry reporting verified');
  }

  // =========================================================================
  // 12. Client Teardown & AsyncDispose Lifecycle
  // =========================================================================
  console.log('12. Testing client teardown and [Symbol.asyncDispose] lifecycle...');
  {
    const client = new SearchWorkerClient<BookDoc>();
    await client.init([
      { id: '1', title: 'Item 1', author: 'A', summary: 'S' }
    ], {
      fields: ['title'],
      preferGpu: false
    });

    // Destroy client
    await client.destroy();

    let searchAfterDestroyThrew = false;
    try {
      await client.search('Item');
    } catch (err: any) {
      searchAfterDestroyThrew = err?.message?.includes('destroyed');
    }
    assert.strictEqual(searchAfterDestroyThrew, true, 'search() on destroyed client must throw');

    // AsyncDispose support
    {
      const disposableClient = new SearchWorkerClient<BookDoc>();
      await disposableClient.init([
        { id: '1', title: 'Disposable', author: 'A', summary: 'S' }
      ], {
        fields: ['title'],
        preferGpu: false
      });

      await disposableClient[Symbol.asyncDispose]();
      let threw = false;
      try {
        await disposableClient.search('Disposable');
      } catch (err: any) {
        threw = err?.message?.includes('destroyed');
      }
      assert.strictEqual(threw, true, 'Symbol.asyncDispose must destroy client');
    }

    console.log('   ✅ Client teardown and Symbol.asyncDispose lifecycle verified');
  }

  console.log('\n--- All Milestone 5 First-Party Worker Client & Protocol Tests Passed! ✅ ---');
}

runM5Tests()
  .then(() => {
    process.exit(0);
  })
  .catch((err) => {
    console.error('Milestone 5 test failed:', err);
    process.exit(1);
  });
