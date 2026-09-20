import assert from 'node:assert';
import {
  DocumentIndex,
  SearchWorkerClient,
  serializeDocumentIndex,
  deserializeDocumentSnapshot,
  deserializeDocumentSnapshotHeader,
  restoreDocumentIndex,
  saveIndexToIDB,
  loadIndexFromIDB,
  deleteIndexFromIDB,
  restoreIndexFromIDB,
  IncompatibleIndexError,
  SERIALIZED_DOC_MAGIC,
  DOC_FORMAT_VERSION,
  SERIALIZED_DOC_HEADER_BYTES,
  DEFAULT_IDB_DATABASE_NAME,
  DEFAULT_SNAPSHOT_STORE_NAME,
  DEFAULT_DOCUMENT_STORE_NAME,
  DEFAULT_SNAPSHOT_KEY,
  startSearchWorker,
  isDedicatedWorker
} from '../packages/webgpu-search/src/index';

interface ArticleDoc {
  id: string;
  title: string;
  content: string;
  tags?: string[];
}

/**
 * In-memory Mock IDB implementation for headless Bun CLI testing.
 * Accurately models database open, object stores, transactions, auto-close in finally,
 * and tracks whether operations were performed before transaction initialization.
 */
class MockIDBRequest {
  result: any = undefined;
  error: any = null;
  onsuccess: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onupgradeneeded: (() => void) | null = null;
  onblocked: (() => void) | null = null;
}

class MockIDBObjectStore {
  readonly name: string;
  private readonly storeMap: Map<any, any>;
  activeTx: MockIDBTransaction | null = null;

  constructor(name: string, storeMap: Map<any, any>) {
    this.name = name;
    this.storeMap = storeMap;
  }

  put(value: any, key?: any): MockIDBRequest {
    const effectiveKey = key !== undefined ? key : (value && typeof value === 'object' && 'id' in value ? value.id : Math.random());
    const req = new MockIDBRequest();
    if (this.activeTx) {
      return this.activeTx.registerRequest(req, () => {
        this.storeMap.set(effectiveKey, value);
        req.result = effectiveKey;
      });
    }
    this.storeMap.set(effectiveKey, value);
    req.result = effectiveKey;
    queueMicrotask(() => req.onsuccess?.());
    return req;
  }

  get(key: any): MockIDBRequest {
    const req = new MockIDBRequest();
    if (this.activeTx) {
      return this.activeTx.registerRequest(req, () => {
        req.result = this.storeMap.get(key);
      });
    }
    req.result = this.storeMap.get(key);
    queueMicrotask(() => req.onsuccess?.());
    return req;
  }

  getAll(): MockIDBRequest {
    const req = new MockIDBRequest();
    if (this.activeTx) {
      return this.activeTx.registerRequest(req, () => {
        req.result = Array.from(this.storeMap.values());
      });
    }
    req.result = Array.from(this.storeMap.values());
    queueMicrotask(() => req.onsuccess?.());
    return req;
  }

  delete(key: any): MockIDBRequest {
    const req = new MockIDBRequest();
    if (this.activeTx) {
      return this.activeTx.registerRequest(req, () => {
        this.storeMap.delete(key);
        req.result = undefined;
      });
    }
    this.storeMap.delete(key);
    req.result = undefined;
    queueMicrotask(() => req.onsuccess?.());
    return req;
  }
}

class MockIDBTransaction {
  readonly mode: 'readonly' | 'readwrite';
  readonly storeNames: string[];
  private readonly db: MockIDBDatabase;
  oncomplete: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onabort: (() => void) | null = null;
  error: any = null;
  private pendingRequests: number = 0;

  constructor(db: MockIDBDatabase, storeNames: string[], mode: 'readonly' | 'readwrite') {
    this.db = db;
    this.storeNames = storeNames;
    this.mode = mode;
    queueMicrotask(() => {
      this.checkCompletion();
    });
  }

  registerRequest(req: MockIDBRequest, execute: () => void): MockIDBRequest {
    this.pendingRequests++;
    queueMicrotask(() => {
      execute();
      req.onsuccess?.();
      this.pendingRequests--;
      this.checkCompletion();
    });
    return req;
  }

  private checkCompletion(): void {
    if (this.pendingRequests === 0) {
      queueMicrotask(() => {
        if (this.pendingRequests === 0) {
          this.oncomplete?.();
        }
      });
    }
  }

  objectStore(name: string): MockIDBObjectStore {
    const store = this.db.getObjectStore(name);
    store.activeTx = this;
    return store;
  }
}

class MockIDBDatabase {
  readonly name: string;
  version: number;
  private stores: Map<string, Map<any, any>> = new Map();
  isClosed: boolean = false;
  closeCallCount: number = 0;
  transactionCount: number = 0;

  constructor(name: string, version: number) {
    this.name = name;
    this.version = version;
  }

  get objectStoreNames(): { contains: (name: string) => boolean } {
    return {
      contains: (name: string) => this.stores.has(name)
    };
  }

  createObjectStore(name: string): MockIDBObjectStore {
    if (!this.stores.has(name)) {
      this.stores.set(name, new Map());
    }
    return new MockIDBObjectStore(name, this.stores.get(name)!);
  }

  getObjectStore(name: string): MockIDBObjectStore {
    if (!this.stores.has(name)) {
      this.stores.set(name, new Map());
    }
    return new MockIDBObjectStore(name, this.stores.get(name)!);
  }

  transaction(storeNames: string | string[], mode: 'readonly' | 'readwrite' = 'readonly'): MockIDBTransaction {
    if (this.isClosed) {
      throw new Error('InvalidStateError: Database is closed');
    }
    this.transactionCount++;
    const names = Array.isArray(storeNames) ? storeNames : [storeNames];
    return new MockIDBTransaction(this, names, mode);
  }

  close(): void {
    this.isClosed = true;
    this.closeCallCount++;
  }
}

class MockIDBFactory {
  private databases: Map<string, MockIDBDatabase> = new Map();
  lastOpenedDatabase: MockIDBDatabase | null = null;

  open(name: string, version?: number): MockIDBRequest {
    const req = new MockIDBRequest();
    let db = this.databases.get(name);
    const isNew = !db;
    if (!db) {
      db = new MockIDBDatabase(name, version ?? 1);
      this.databases.set(name, db);
    }
    db.isClosed = false;
    this.lastOpenedDatabase = db;

    queueMicrotask(() => {
      req.result = db;
      if (isNew) {
        req.onupgradeneeded?.();
      }
      req.onsuccess?.();
    });

    return req;
  }
}

/**
 * Creates a mock Web Worker message port pair for SearchWorkerClient testing.
 */
function createMockWorkerScope(): { clientWorker: any; workerScope: any } {
  const clientListeners: Array<(e: any) => void> = [];
  const workerListeners: Array<(e: any) => void> = [];

  const clientWorker = {
    postMessage(data: any, _transfer?: any[]) {
      queueMicrotask(() => {
        for (const listener of workerListeners) {
          listener({ data });
        }
      });
    },
    addEventListener(event: string, listener: any) {
      if (event === 'message') clientListeners.push(listener);
    },
    removeEventListener(event: string, listener: any) {
      if (event === 'message') {
        const idx = clientListeners.indexOf(listener);
        if (idx >= 0) clientListeners.splice(idx, 1);
      }
    },
    terminate() {
      clientListeners.length = 0;
      workerListeners.length = 0;
    }
  };

  const workerScope = {
    postMessage(data: any, _transfer?: any[]) {
      queueMicrotask(() => {
        for (const listener of clientListeners) {
          listener({ data });
        }
      });
    },
    addEventListener(event: string, listener: any) {
      if (event === 'message') workerListeners.push(listener);
    },
    removeEventListener(event: string, listener: any) {
      if (event === 'message') {
        const idx = workerListeners.indexOf(listener);
        if (idx >= 0) workerListeners.splice(idx, 1);
      }
    }
  };

  startSearchWorker(workerScope);

  return { clientWorker, workerScope };
}

async function runM6Tests() {
  console.log('--- Running Milestone 6: Versioned Snapshot Persistence (U2D3) & IndexedDB Tests ---');

  // Sample multi-field documents with Unicode characters
  const sampleArticles: ArticleDoc[] = [
    {
      id: 'doc-1',
      title: 'High-Performance WebGPU Computing',
      content: 'Massively parallel compute shaders for client-side search and ranking.',
      tags: ['webgpu', 'compute', 'parallel']
    },
    {
      id: 'doc-2',
      title: 'Unicode Stra\u00DFe & Emojis \u{1F525}\u{1F680}',
      content: 'Testing German case folding \u00DF \u2192 ss and astral plane graphemes.',
      tags: ['unicode', 'case-folding', 'emoji']
    },
    {
      id: 'doc-3',
      title: 'T\u00FCrk\u00E7e \u0130stanbul Arama',
      content: 'Dotted capital \u0130 and dotless \u0131 normalization invariants.',
      tags: ['turkish', 'i18n', 'search']
    }
  ];

  // =========================================================================
  // 1. U2D3 Binary Constants & Header Layout
  // =========================================================================
  console.log('1. Testing U2D3 binary specification constants and header layout...');
  {
    assert.strictEqual(SERIALIZED_DOC_MAGIC, 0x55324433, 'SERIALIZED_DOC_MAGIC must be 0x55324433 ("U2D3")');
    assert.strictEqual(DOC_FORMAT_VERSION, 3, 'DOC_FORMAT_VERSION must be 3');
    assert.strictEqual(SERIALIZED_DOC_HEADER_BYTES, 48, 'SERIALIZED_DOC_HEADER_BYTES must be 48 bytes');
    console.log('   ✅ Binary constants confirmed');
  }

  // =========================================================================
  // 2. Full Serialization and Deserialization Parity
  // =========================================================================
  console.log('2. Testing DocumentIndex serialization & restoration parity...');
  {
    const originalIndex = await DocumentIndex.create(sampleArticles, {
      fields: [
        { name: 'title', weight: 2.0 },
        { name: 'content', weight: 1.0 },
        { name: 'tags', weight: 1.5 }
      ]
    });

    const snapshot = originalIndex.serialize();
    assert(snapshot instanceof ArrayBuffer, 'serialize() must return an ArrayBuffer');
    assert(snapshot.byteLength >= SERIALIZED_DOC_HEADER_BYTES, 'snapshot length must exceed 48 bytes');

    // Parse header directly
    const header = deserializeDocumentSnapshotHeader(snapshot);
    assert.strictEqual(header.magic, 0x55324433);
    assert.strictEqual(header.formatVersion, 3);
    assert.strictEqual(header.docCount, 3);
    assert.strictEqual(header.rowCount, 9); // 3 docs * 3 fields
    assert(header.tokenCount > 0);
    assert(header.schemaByteLength > 0);
    assert(header.docsByteLength > 0);
    assert(header.checksum !== 0);

    // Restore into a new index
    const restoredIndex = await DocumentIndex.fromSnapshot(snapshot);

    assert.strictEqual(restoredIndex.getStats().docCount, 3);
    assert.strictEqual(restoredIndex.getStats().rowCount, 9);
    assert.strictEqual(restoredIndex.getStats().tokenCount, originalIndex.getStats().tokenCount);
    assert(typeof restoredIndex.getStats().restoreTimeMs === 'number');

    // Query both and assert identical ranking and highlight parity
    const queries = ['webgpu', 'strasse', 'istanbul', 'emoji', 'parallel'];
    for (const q of queries) {
      const respOriginal = await originalIndex.search(q, { highlight: true });
      const respRestored = await restoredIndex.search(q, { highlight: true });

      assert.strictEqual(respRestored.totalMatches, respOriginal.totalMatches, `Match count mismatch for "${q}"`);
      assert.strictEqual(respRestored.results.length, respOriginal.results.length, `Result length mismatch for "${q}"`);

      for (let i = 0; i < respOriginal.results.length; i++) {
        const orig = respOriginal.results[i];
        const rest = respRestored.results[i];
        assert.strictEqual(rest.id, orig.id, `ID mismatch at rank ${i} for "${q}"`);
        assert.strictEqual(rest.score, orig.score, `Score mismatch at rank ${i} for "${q}"`);
        assert.strictEqual(rest.matchedField, orig.matchedField, `Field mismatch at rank ${i} for "${q}"`);
        assert.deepStrictEqual(rest.doc, orig.doc, `Doc mismatch at rank ${i} for "${q}"`);
        assert.deepStrictEqual(rest.highlights, orig.highlights, `Highlight ranges mismatch for "${q}"`);
      }
    }

    console.log('   ✅ Serialization and restoration parity confirmed across multi-field queries');
  }

  // =========================================================================
  // 3. Compaction Pre-Condition: Tombstones Vacuumed Prior to Snapshot
  // =========================================================================
  console.log('3. Testing compaction pre-condition (zero tombstones in serialized snapshot)...');
  {
    const tenDocs: ArticleDoc[] = Array.from({ length: 10 }, (_, i) => ({
      id: `doc-${i}`,
      title: `Title ${i} ${i === 4 ? 'Temporary Target' : 'Stable'}`,
      content: `Content ${i}`
    }));

    const index = await DocumentIndex.create(tenDocs, {
      fields: ['title', 'content']
    });

    // Remove 1 document out of 10 (tombstoneRatio = 2/20 = 0.1 < 0.25)
    await index.remove(['doc-4']);

    assert(index.getStats().tombstoneCount > 0, 'Index must have tombstones prior to serialize');

    // serialize() must vacuum/compact synchronously
    const snapshot = index.serialize();

    assert.strictEqual(index.getStats().tombstoneCount, 0, 'serialize() must compact all tombstones in live index');

    const header = deserializeDocumentSnapshotHeader(snapshot);
    // Active docs: 9
    assert.strictEqual(header.docCount, 9);
    assert.strictEqual(header.rowCount, 18); // 9 docs * 2 fields

    const restored = await DocumentIndex.fromSnapshot(snapshot);
    assert.strictEqual(restored.getStats().tombstoneCount, 0);
    assert.strictEqual(restored.getStats().docCount, 9);

    // doc-4 must not be in restored index
    const res = await restored.search('Temporary');
    assert.strictEqual(res.totalMatches, 0, 'Tombstoned doc-4 must not exist in restored index');

    console.log('   ✅ Compaction pre-condition validated: tombstones vacuumed cleanly');
  }

  // =========================================================================
  // 4. Decoupled Document Storage (docsByteLength = 0)
  // =========================================================================
  console.log('4. Testing decoupled document storage (docsByteLength = 0)...');
  {
    const index = await DocumentIndex.create(sampleArticles, {
      fields: ['title', 'content']
    });

    // Serialize with decoupled: true
    const decoupledSnapshot = index.serialize({ decoupled: true });
    const header = deserializeDocumentSnapshotHeader(decoupledSnapshot);

    assert.strictEqual(header.docsByteLength, 0, 'Decoupled snapshot must have docsByteLength = 0');

    // Compare with embedded snapshot
    const embeddedSnapshot = index.serialize({ decoupled: false });
    assert(
      decoupledSnapshot.byteLength < embeddedSnapshot.byteLength,
      'Decoupled snapshot must be strictly smaller than embedded snapshot'
    );

    // Restore without documents: index searches by tokens and returns correct IDs
    const restoredDecoupled = await DocumentIndex.fromSnapshot(decoupledSnapshot);
    assert.strictEqual(restoredDecoupled.getStats().docCount, 3);
    const searchRes = await restoredDecoupled.search('WebGPU');
    assert(searchRes.results.length > 0);
    assert.strictEqual(searchRes.results[0].id, 'doc-1');

    // Restore with external documents passed in options
    const restoredWithDocs = await DocumentIndex.fromSnapshot(decoupledSnapshot, {
      documents: sampleArticles
    });
    const searchResWithDocs = await restoredWithDocs.search('WebGPU');
    assert(searchResWithDocs.results.length > 0);
    assert.strictEqual(searchResWithDocs.results[0].id, 'doc-1');
    assert.deepStrictEqual(searchResWithDocs.results[0].doc, sampleArticles[0]);

    console.log('   ✅ Decoupled document storage (docsByteLength = 0) confirmed');
  }

  // =========================================================================
  // 5. Fail-Closed Error Hierarchy & Tamper Resistance
  // =========================================================================
  console.log('5. Testing fail-closed error hierarchy & tamper resistance...');
  {
    const index = await DocumentIndex.create(sampleArticles, { fields: ['title'] });
    const validBuffer = index.serialize();

    // 1. Truncated buffer (< 48 bytes)
    assert.throws(
      () => deserializeDocumentSnapshotHeader(validBuffer.slice(0, 30)),
      (err: any) => err instanceof IncompatibleIndexError
    );

    // 2. Corrupt Magic
    const badMagicBuf = validBuffer.slice(0);
    new DataView(badMagicBuf).setUint32(0, 0x11223344, true);
    assert.throws(
      () => deserializeDocumentSnapshotHeader(badMagicBuf),
      (err: any) => err instanceof IncompatibleIndexError
    );

    // 3. Corrupt Version
    const badVerBuf = validBuffer.slice(0);
    new DataView(badVerBuf).setUint32(4, 999, true);
    assert.throws(
      () => deserializeDocumentSnapshotHeader(badVerBuf),
      (err: any) => err instanceof IncompatibleIndexError
    );

    // 4. Corrupt CRC32 (tampering 1 bit in schema segment)
    const tamperedPayloadBuf = validBuffer.slice(0);
    const u8 = new Uint8Array(tamperedPayloadBuf);
    u8[50] ^= 0xff; // Flip bits in schema segment
    assert.rejects(
      async () => restoreDocumentIndex(tamperedPayloadBuf),
      (err: any) => err instanceof IncompatibleIndexError
    );

    // 5. Tampered Offsets (non-monotonic)
    const rawSnapshot = deserializeDocumentSnapshot(validBuffer);
    const tamperedOffsetsBuf = validBuffer.slice(0);
    const header = deserializeDocumentSnapshotHeader(tamperedOffsetsBuf);
    const offsetsPos = SERIALIZED_DOC_HEADER_BYTES + header.schemaByteLength + header.tokenCount * 4;
    // Set offsets[1] > offsets[2]
    new DataView(tamperedOffsetsBuf).setUint32(offsetsPos + 4, 999999, true);
    // Recompute CRC with the bad offset to bypass CRC check and test offset validation directly
    const tamperedDv = new DataView(tamperedOffsetsBuf);
    const u8Tampered = new Uint8Array(tamperedOffsetsBuf);
    const { crc32Parts } = await import('../packages/webgpu-search/src/index');
    const newCrc = crc32Parts([
      new Uint8Array(tamperedOffsetsBuf, 0, 44),
      new Uint8Array(tamperedOffsetsBuf, SERIALIZED_DOC_HEADER_BYTES)
    ]);
    tamperedDv.setUint32(44, newCrc, true);

    assert.rejects(
      async () => restoreDocumentIndex(tamperedOffsetsBuf),
      (err: any) => err instanceof IncompatibleIndexError
    );

    console.log('   ✅ Fail-closed error hierarchy & tamper checks confirmed');
  }

  // =========================================================================
  // 6. Transaction-Safe IndexedDB Helpers
  // =========================================================================
  console.log('6. Testing transaction-safe IndexedDB persistence...');
  {
    const mockIdb = new MockIDBFactory();
    const index = await DocumentIndex.create(sampleArticles, {
      fields: ['title', 'content']
    });

    // Save embedded snapshot to IDB
    await saveIndexToIDB(index, {
      indexedDB: mockIdb as any,
      key: 'test_key_1'
    });

    // Verify DB closed in finally
    assert(mockIdb.lastOpenedDatabase?.isClosed === true, 'Database connection must be closed in finally block');
    assert(mockIdb.lastOpenedDatabase?.closeCallCount! >= 1, 'db.close() must be called');

    // Load and restore from IDB
    const loaded = await loadIndexFromIDB({
      indexedDB: mockIdb as any,
      key: 'test_key_1'
    });
    assert(loaded !== null, 'Loaded result must not be null');
    assert(loaded!.snapshot instanceof ArrayBuffer, 'Loaded snapshot must be an ArrayBuffer');

    const restoredFromIdb = await restoreIndexFromIDB({
      indexedDB: mockIdb as any,
      key: 'test_key_1'
    });
    assert(restoredFromIdb !== null);
    assert.strictEqual(restoredFromIdb!.getStats().docCount, 3);
    const qRes = await restoredFromIdb!.search('WebGPU');
    assert.strictEqual(qRes.results[0].id, 'doc-1');

    // Test Decoupled Document Persistence in IDB
    await saveIndexToIDB(index, {
      indexedDB: mockIdb as any,
      key: 'test_key_decoupled',
      decoupled: true
    });

    const restoredDecoupledFromIdb = await restoreIndexFromIDB({
      indexedDB: mockIdb as any,
      key: 'test_key_decoupled'
    });
    assert(restoredDecoupledFromIdb !== null);
    assert.strictEqual(restoredDecoupledFromIdb!.getStats().docCount, 3);
    const decoupledQRes = await restoredDecoupledFromIdb!.search('WebGPU');
    assert.strictEqual(decoupledQRes.results[0].id, 'doc-1');
    assert.deepStrictEqual(decoupledQRes.results[0].doc, sampleArticles[0]);

    // Test delete from IDB
    const deleted = await deleteIndexFromIDB({
      indexedDB: mockIdb as any,
      key: 'test_key_1'
    });
    assert.strictEqual(deleted, true);

    const reloaded = await loadIndexFromIDB({
      indexedDB: mockIdb as any,
      key: 'test_key_1'
    });
    assert.strictEqual(reloaded, null, 'Deleted snapshot must return null');

    console.log('   ✅ IndexedDB persistence: transaction safety, decoupled docs & cleanup confirmed');
  }

  // =========================================================================
  // 7. SearchWorkerClient Serialization & Restoration
  // =========================================================================
  console.log('7. Testing SearchWorkerClient snapshot serialization & restoration...');
  {
    const { clientWorker } = createMockWorkerScope();
    const workerClient = new SearchWorkerClient<ArticleDoc>({
      worker: clientWorker
    });

    await workerClient.init(sampleArticles, {
      fields: [
        { name: 'title', weight: 2.0 },
        { name: 'content', weight: 1.0 }
      ]
    });

    // Serialize across worker boundary
    const workerSnapshot = await workerClient.serialize();
    assert(workerSnapshot instanceof ArrayBuffer);
    assert(workerSnapshot.byteLength >= SERIALIZED_DOC_HEADER_BYTES);

    const header = deserializeDocumentSnapshotHeader(workerSnapshot);
    assert.strictEqual(header.docCount, 3);

    // Create a fresh worker client and restore directly from snapshot (without init)
    const { clientWorker: clientWorker2 } = createMockWorkerScope();
    const freshClient = new SearchWorkerClient<ArticleDoc>({
      worker: clientWorker2
    });

    await freshClient.restore(workerSnapshot);

    const stats = await freshClient.getStats();
    assert.strictEqual(stats.docCount, 3);

    const resp = await freshClient.search('WebGPU');
    assert(resp.results.length > 0);
    assert.strictEqual(resp.results[0].id, 'doc-1');
    assert.strictEqual((resp.results[0].doc as any)?.title, sampleArticles[0].title);
    assert.strictEqual((resp.results[0].doc as any)?.content, sampleArticles[0].content);

    // Restore with external documents passed in options to rehydrate unindexed fields
    const { clientWorker: clientWorker3 } = createMockWorkerScope();
    const freshClientWithDocs = new SearchWorkerClient<ArticleDoc>({
      worker: clientWorker3
    });
    await freshClientWithDocs.restore(workerSnapshot, { documents: sampleArticles });
    const respWithDocs = await freshClientWithDocs.search('WebGPU');
    assert(respWithDocs.results.length > 0);
    assert.strictEqual(respWithDocs.results[0].id, 'doc-1');
    assert.deepStrictEqual(respWithDocs.results[0].doc, sampleArticles[0]);

    await workerClient.destroy();
    await freshClient.destroy();
    await freshClientWithDocs.destroy();

    console.log('   ✅ SearchWorkerClient serialize & direct restore confirmed');
  }

  console.log('\n--- All Milestone 6 Persistence Tests Passed Successfully! ✅ ---');
}

runM6Tests().catch((err) => {
  console.error('❌ M6 Tests failed:', err);
  process.exit(1);
});
