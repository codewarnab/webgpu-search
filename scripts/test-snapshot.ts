import assert from 'node:assert';
import {
  DocumentIndex,
  SearchWorkerClient,
  encodeSnapshot,
  decodeSnapshot,
  decodeSnapshotHeader,
  restoreSnapshot,
  saveIndexToIDB,
  loadIndexFromIDB,
  deleteIndexFromIDB,
  restoreIndexFromIDB,
  IncompatibleIndexError,
  IncompatibleHookError,
  ProfileMismatchError,
  LEGACY_SNAPSHOT_MAGIC,
  LEGACY_SNAPSHOT_VERSION,
  LEGACY_SNAPSHOT_HEADER_BYTES,
  SNAPSHOT_MAGIC,
  SNAPSHOT_FORMAT_VERSION,
  SNAPSHOT_HEADER_BYTES,
  MAX_SNAPSHOT_SCHEMA_BYTES,
  MAX_SNAPSHOT_COLUMNAR_BYTES,
  MAX_SNAPSHOT_DOCS_BYTES,
  MAX_SNAPSHOT_BYTES,
  MAX_SNAPSHOT_DOC_COUNT,
  MAX_SNAPSHOT_TOKEN_COUNT,
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

async function runSnapshotTests() {
  console.log('--- Running  Versioned Snapshot Persistence (canonical + legacy read) & IndexedDB Tests ---');

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
  // 1. legacy snapshot Legacy Constants & snapshot Canonical Constants
  // =========================================================================
  console.log('1. Testing binary specification constants and header layout...');
  {
    assert.strictEqual(LEGACY_SNAPSHOT_MAGIC, 0x55324433, 'LEGACY_SNAPSHOT_MAGIC must be 0x55324433 ("legacy snapshot")');
    assert.strictEqual(LEGACY_SNAPSHOT_VERSION, 3, 'LEGACY_SNAPSHOT_VERSION must be 3');
    assert.strictEqual(LEGACY_SNAPSHOT_HEADER_BYTES, 48, 'LEGACY_SNAPSHOT_HEADER_BYTES must be 48 bytes');
    assert.strictEqual(SNAPSHOT_MAGIC, 0x55324434, 'SNAPSHOT_MAGIC must be 0x55324434 ("snapshot")');
    assert.strictEqual(SNAPSHOT_FORMAT_VERSION, 4, 'SNAPSHOT_FORMAT_VERSION must be 4');
    assert.strictEqual(SNAPSHOT_HEADER_BYTES, 56, 'SNAPSHOT_HEADER_BYTES must be 56 bytes (8-byte aligned)');
    assert.strictEqual(SNAPSHOT_HEADER_BYTES % 8, 0, 'snapshot header must be 8-byte aligned');
    console.log('   ✅ Binary constants confirmed (legacy snapshot legacy read path + snapshot canonical write path)');
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
    assert(snapshot.byteLength >= SNAPSHOT_HEADER_BYTES, 'snapshot length must exceed 56-byte snapshot header');

    // Parse header directly (canonical snapshot write path)
    const header = decodeSnapshotHeader(snapshot);
    assert.strictEqual(header.magic, 0x55324434);
    assert.strictEqual(header.formatVersion, 4);
    assert.strictEqual(header.docCount, 3);
    assert.strictEqual(header.rowCount, 9); // 3 docs * 3 fields
    assert(header.tokenCount > 0);
    assert(header.schemaByteLength > 0);
    assert(header.docsByteLength > 0);
    assert(typeof header.columnarByteLength === 'number');
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

    const header = decodeSnapshotHeader(snapshot);
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
    const header = decodeSnapshotHeader(decoupledSnapshot);

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
  // 5. Fail-Closed Error Hierarchy & Corruption Detection
  // =========================================================================
  console.log('5. Testing fail-closed error hierarchy & corruption detection...');
  {
    const index = await DocumentIndex.create(sampleArticles, { fields: ['title'] });
    const validBuffer = index.serialize();

    // 1. Truncated buffer (< 48 bytes)
    assert.throws(
      () => decodeSnapshotHeader(validBuffer.slice(0, 30)),
      (err: any) => err instanceof IncompatibleIndexError
    );

    // 2. Corrupt Magic
    const badMagicBuf = validBuffer.slice(0);
    new DataView(badMagicBuf).setUint32(0, 0x11223344, true);
    assert.throws(
      () => decodeSnapshotHeader(badMagicBuf),
      (err: any) => err instanceof IncompatibleIndexError
    );

    // 3. Corrupt Version
    const badVerBuf = validBuffer.slice(0);
    new DataView(badVerBuf).setUint32(4, 999, true);
    assert.throws(
      () => decodeSnapshotHeader(badVerBuf),
      (err: any) => err instanceof IncompatibleIndexError
    );

    // 4. Corrupt CRC32 (tampering 1 bit in schema segment past the 56-byte header)
    const tamperedPayloadBuf = validBuffer.slice(0);
    const u8 = new Uint8Array(tamperedPayloadBuf);
    u8[60] ^= 0xff; // Flip bits in schema segment
    assert.rejects(
      async () => restoreSnapshot(tamperedPayloadBuf),
      (err: any) => err instanceof IncompatibleIndexError
    );

    // 5. Tampered Offsets (non-monotonic)
    const rawSnapshot = decodeSnapshot(validBuffer);
    const tamperedOffsetsBuf = validBuffer.slice(0);
    const header = decodeSnapshotHeader(tamperedOffsetsBuf);
    const offsetsPos = SNAPSHOT_HEADER_BYTES + header.schemaByteLength + header.tokenCount * 4;
    // Set offsets[1] > offsets[2]
    new DataView(tamperedOffsetsBuf).setUint32(offsetsPos + 4, 999999, true);
    // Recompute CRC with the bad offset to bypass CRC check and test offset validation directly
    const tamperedDv = new DataView(tamperedOffsetsBuf);
    const u8Tampered = new Uint8Array(tamperedOffsetsBuf);
    const { crc32Parts } = await import('../packages/webgpu-search/src/index');
    const newCrc = crc32Parts([
      new Uint8Array(tamperedOffsetsBuf, 0, 52),
      new Uint8Array(tamperedOffsetsBuf, SNAPSHOT_HEADER_BYTES)
    ]);
    tamperedDv.setUint32(52, newCrc, true);

    assert.rejects(
      async () => restoreSnapshot(tamperedOffsetsBuf),
      (err: any) => err instanceof IncompatibleIndexError
    );

    // 6. Duck-typed non-ArrayBuffer ({ byteLength: 50 })
    assert.throws(
      () => decodeSnapshotHeader({ byteLength: 50 } as any),
      (err: any) => err instanceof IncompatibleIndexError
    );

    // 7. Tampered schema.docIds length mismatch
    const badDocIdsBuf = validBuffer.slice(0);
    const badDocIdsHeader = decodeSnapshotHeader(badDocIdsBuf);
    const schemaBytes = new Uint8Array(badDocIdsBuf, SNAPSHOT_HEADER_BYTES, badDocIdsHeader.schemaByteLength);
    const parsedSchema = JSON.parse(new TextDecoder().decode(schemaBytes));
    parsedSchema.docIds = ['doc-1']; // length 1 instead of 3
    const newSchemaBytes = new TextEncoder().encode(JSON.stringify(parsedSchema));
    if (newSchemaBytes.length <= badDocIdsHeader.schemaByteLength) {
      const padded = new Uint8Array(badDocIdsHeader.schemaByteLength);
      padded.set(newSchemaBytes);
      for (let p = newSchemaBytes.length; p < badDocIdsHeader.schemaByteLength; p++) {
        padded[p] = 0x20;
      }
      schemaBytes.set(padded);
      const tamperedDv2 = new DataView(badDocIdsBuf);
      const newCrc2 = crc32Parts([
        new Uint8Array(badDocIdsBuf, 0, 52),
        new Uint8Array(badDocIdsBuf, SNAPSHOT_HEADER_BYTES)
      ]);
      tamperedDv2.setUint32(52, newCrc2, true);
      assert.rejects(
        async () => restoreSnapshot(badDocIdsBuf),
        (err: any) => err instanceof IncompatibleIndexError
      );
    }

    console.log('   ✅ Fail-closed error hierarchy & corruption detection confirmed');
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

    // Test Multi-Index Decoupled Isolation in IDB
    const index2 = await DocumentIndex.create([
      { id: 'item-100', title: 'Rust Systems', content: 'Low level memory safety' }
    ], {
      fields: ['title', 'content']
    });

    await saveIndexToIDB(index2, {
      indexedDB: mockIdb as any,
      key: 'test_key_decoupled_2',
      decoupled: true
    });

    // Ensure index 1 still loads ONLY index 1 docs, and index 2 loads ONLY index 2 docs
    const restoredA = await restoreIndexFromIDB({
      indexedDB: mockIdb as any,
      key: 'test_key_decoupled'
    });
    assert.strictEqual(restoredA!.getStats().docCount, 3);

    const restoredB = await restoreIndexFromIDB({
      indexedDB: mockIdb as any,
      key: 'test_key_decoupled_2'
    });
    assert.strictEqual(restoredB!.getStats().docCount, 1);
    const bRes = await restoredB!.search('Rust');
    assert.strictEqual(bRes.results[0].id, 'item-100');

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

    // Test delete decoupled index from IDB cleans up doc store
    await deleteIndexFromIDB({
      indexedDB: mockIdb as any,
      key: 'test_key_decoupled_2'
    });
    const reloadedDecoupledB = await loadIndexFromIDB({
      indexedDB: mockIdb as any,
      key: 'test_key_decoupled_2'
    });
    assert.strictEqual(reloadedDecoupledB, null);

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
    assert(workerSnapshot.byteLength >= SNAPSHOT_HEADER_BYTES);

    const header = decodeSnapshotHeader(workerSnapshot);
    assert.strictEqual(header.docCount, 3);

    // Create a fresh worker client and restore directly from snapshot (without init)
    const { clientWorker: clientWorker2 } = createMockWorkerScope();
    const freshClient = new SearchWorkerClient<ArticleDoc>({
      worker: clientWorker2
    });

    // Restore with non-cloneable options (function getters, null device) - tests sanitization
    await freshClient.restore(workerSnapshot, {
      options: {
        fields: [
          { name: 'title', weight: 2.0, getter: (d: any) => d.title },
          { name: 'content', weight: 1.0, getter: (d: any) => d.content }
        ]
      }
    });

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

  // =========================================================================
  // 8. Dynamic Mutations (add) Stratification Parity & Custom Getters
  // =========================================================================
  console.log('8. Testing dynamic additions (unstratified rows) serialization & custom getters...');
  {
    interface CustomDoc {
      docKey: string;
      meta: {
        headline: string;
      };
      body: string;
    }

    const initialDocs: CustomDoc[] = [
      { docKey: 'k-1', meta: { headline: 'Apple Fruits' }, body: 'Red fruit with seeds' },
      { docKey: 'k-2', meta: { headline: 'Banana Tropics' }, body: 'Yellow fruit peel' }
    ];

    const index = await DocumentIndex.create(initialDocs, {
      fields: [
        { name: 'headline', getter: (d) => d.meta.headline },
        'body'
      ],
      idField: 'docKey'
    });

    // Dynamically append without tombstones
    await index.add({
      docKey: 'k-3',
      meta: { headline: 'Carrot Veggies' },
      body: 'Orange vegetable root'
    });

    // Serialize must re-stratify rows seamlessly
    const snapshot = index.serialize();
    const restored = await DocumentIndex.fromSnapshot<CustomDoc>(snapshot, {
      options: {
        fields: [
          { name: 'headline', getter: (d) => d.meta.headline },
          'body'
        ],
        idField: 'docKey'
      }
    });

    assert.strictEqual(restored.getStats().docCount, 3);
    assert.strictEqual(restored.getStats().rowCount, 6); // 3 docs * 2 fields

    // Search in specific field (must not be scrambled!)
    const titleRes = await restored.search('Carrot', { fields: ['headline'], highlight: true });
    assert.strictEqual(titleRes.totalMatches, 1, 'Carrot must be matched in headline field');
    assert.strictEqual(titleRes.results[0].id, 'k-3');
    assert.strictEqual(titleRes.results[0].matchedField, 'headline');
    assert(titleRes.results[0].highlights !== undefined);
    assert(titleRes.results[0].highlights!.headline !== undefined && titleRes.results[0].highlights!.headline.length > 0);

    const bodyRes = await restored.search('Orange', { fields: ['body'] });
    assert.strictEqual(bodyRes.totalMatches, 1, 'Orange must be matched in body field');
    assert.strictEqual(bodyRes.results[0].id, 'k-3');
    assert.strictEqual(bodyRes.results[0].matchedField, 'body');

    console.log('   ✅ Dynamic additions stratification parity & custom getters confirmed');
  }

  // =========================================================================
  // 9. snapshot Columnar Segment & Filter/Facet Restore Parity (8)
  // =========================================================================
  console.log('9. Testing snapshot columnar filter metadata roundtrip & facet parity...');
  {
    interface FilterDoc {
      id: string;
      title: string;
      level: string;
      latencyMs: number;
    }
    const filterDocs: FilterDoc[] = [
      { id: 'l-1', title: 'timeout in auth-service fence wait', level: 'ERROR', latencyMs: 1200 },
      { id: 'l-2', title: 'timeout retry on cache-proxy read', level: 'WARN', latencyMs: 320 },
      { id: 'l-3', title: 'healthy heartbeat probe ok', level: 'INFO', latencyMs: 12 },
      { id: 'l-4', title: 'timeout deadlock on orders table', level: 'ERROR', latencyMs: 950 }
    ];

    const index = await DocumentIndex.create(filterDocs, {
      fields: ['title'],
      filterFields: [{ name: 'level' }, { name: 'latencyMs', type: 'number' }]
    });

    const snapshot = index.serialize();
    const header = decodeSnapshotHeader(snapshot);
    assert.strictEqual(header.magic, SNAPSHOT_MAGIC);
    assert.strictEqual(header.formatVersion, SNAPSHOT_FORMAT_VERSION);
    assert(header.columnarByteLength! > 0, 'filter-configured snapshot must carry a columnar segment');

    // Columnar segment is 8-byte aligned by construction (56-byte header).
    assert.strictEqual(SNAPSHOT_HEADER_BYTES % 8, 0);

    const baseline = await index.search('timeout', {
      mode: 'fuzzy',
      filter: { level: 'ERROR' },
      facets: { byLevel: { type: 'terms', field: 'level', limit: 10 } }
    });
    assert.strictEqual(baseline.totalMatches, 2);

    const restored = await DocumentIndex.fromSnapshot<FilterDoc>(snapshot);
    const after = await restored.search('timeout', {
      mode: 'fuzzy',
      filter: { level: 'ERROR' },
      facets: { byLevel: { type: 'terms', field: 'level', limit: 10 } }
    });
    assert.strictEqual(after.totalMatches, baseline.totalMatches);
    assert.deepStrictEqual(
      after.results.map((r) => r.id),
      baseline.results.map((r) => r.id)
    );
    assert.deepStrictEqual(after.facets, baseline.facets);

    // Numeric range filter parity across the restore boundary.
    const rangeAfter = await restored.search('timeout', {
      mode: 'fuzzy',
      filter: { latencyMs: { gte: 900 } }
    });
    assert.strictEqual(rangeAfter.totalMatches, 2);

    // Corrupted columnar payload must fail closed (CRC mismatch / corruption detection).
    const corruptCol = snapshot.slice(0);
    const colStart = SNAPSHOT_HEADER_BYTES + header.schemaByteLength + header.tokenCount * 4 + (header.rowCount + 1) * 4;
    new Uint8Array(corruptCol, colStart, 1)[0] ^= 0xff;
    await assert.rejects(
      async () => restoreSnapshot(corruptCol),
      (err: any) => err instanceof IncompatibleIndexError
    );

    // Canonical byte layout: total length equals the sum of segments.
    // The 56 B header itself is 8-byte aligned (columnar start alignment
    // depends on variable-length schema/tokens/offsets, so only the header
    // width is asserted here).
    {
      const { crc32Parts } = await import('../packages/webgpu-search/src/index');
      const wantTokens = header.tokenCount * 4;
      const wantOffsets = (header.rowCount + 1) * 4;
      const columnarLen = header.columnarByteLength ?? 0;
      assert.strictEqual(
        snapshot.byteLength,
        SNAPSHOT_HEADER_BYTES + header.schemaByteLength + wantTokens + wantOffsets + columnarLen + header.docsByteLength,
        'canonical byte length must equal header+schema+tokens+offsets+columnar+docs'
      );
      assert.strictEqual(SNAPSHOT_HEADER_BYTES % 8, 0);
      const recomputed = crc32Parts([
        new Uint8Array(snapshot, 0, 52),
        new Uint8Array(snapshot, SNAPSHOT_HEADER_BYTES, header.schemaByteLength),
        new Uint8Array(snapshot, SNAPSHOT_HEADER_BYTES + header.schemaByteLength, wantTokens),
        new Uint8Array(snapshot, SNAPSHOT_HEADER_BYTES + header.schemaByteLength + wantTokens, wantOffsets),
        columnarLen > 0
          ? new Uint8Array(snapshot, SNAPSHOT_HEADER_BYTES + header.schemaByteLength + wantTokens + wantOffsets, columnarLen)
          : new Uint8Array(0),
        header.docsByteLength > 0
          ? new Uint8Array(snapshot, SNAPSHOT_HEADER_BYTES + header.schemaByteLength + wantTokens + wantOffsets + columnarLen, header.docsByteLength)
          : new Uint8Array(0)
      ]);
      assert.strictEqual(recomputed, header.checksum, 'independent CRC recompute must match header checksum');
    }

    index.destroy();
    restored.destroy();
    console.log('   ✅ snapshot columnar segment, filter/facet parity, and corruption detection confirmed');
  }

  // =========================================================================
  // 10. legacy snapshot Legacy Migration Read Path (48 B header, no columnar segment)
  // =========================================================================
  console.log('10. Testing legacy snapshot legacy snapshot migration read path...');
  {
    const { crc32Parts } = await import('../packages/webgpu-search/src/index');
    const legacyDocs = [
      { id: 'u2d3-1', title: 'legacy snapshot migration alpha' },
      { id: 'u2d3-2', title: 'legacy snapshot migration beta' }
    ];
    // Build a canonical snapshot with no filter fields (columnarLen 0),
    // then reframe its payloads as a genuine 48 B legacy snapshot.
    const snapshotIndex = await DocumentIndex.create(legacyDocs, { fields: ['title'] });
    const snapshotBuf = snapshotIndex.serialize();
    const snapshotHeader = decodeSnapshotHeader(snapshotBuf);
    assert.strictEqual(snapshotHeader.columnarByteLength ?? 0, 0);
    const snapshotDv = new DataView(snapshotBuf);
    const profileEnum = snapshotDv.getUint32(8, true);
    const unicodeEnum = snapshotDv.getUint32(12, true);
    const scoringEnum = snapshotDv.getUint32(16, true);
    const docCount = snapshotDv.getUint32(20, true);
    const rowCount = snapshotDv.getUint32(24, true);
    const tokenCount = snapshotDv.getUint32(28, true);
    const normalizedVal = snapshotDv.getUint32(32, true);
    const schemaLen = snapshotDv.getUint32(36, true);
    const docsLen = snapshotDv.getUint32(40, true);
    const schemaBytes = new Uint8Array(snapshotBuf, SNAPSHOT_HEADER_BYTES, schemaLen);
    const tokensBytes = new Uint8Array(snapshotBuf, SNAPSHOT_HEADER_BYTES + schemaLen, tokenCount * 4);
    const offsetsBytes = new Uint8Array(
      snapshotBuf,
      SNAPSHOT_HEADER_BYTES + schemaLen + tokenCount * 4,
      (rowCount + 1) * 4
    );
    const docsBytes = new Uint8Array(
      snapshotBuf,
      SNAPSHOT_HEADER_BYTES + schemaLen + tokenCount * 4 + (rowCount + 1) * 4,
      docsLen
    );
    const legacyTotal = LEGACY_SNAPSHOT_HEADER_BYTES + schemaLen + tokenCount * 4 + (rowCount + 1) * 4 + docsLen;
    const legacyBuf = new ArrayBuffer(legacyTotal);
    const legacyDv = new DataView(legacyBuf);
    legacyDv.setUint32(0, LEGACY_SNAPSHOT_MAGIC, true);
    legacyDv.setUint32(4, LEGACY_SNAPSHOT_VERSION, true);
    legacyDv.setUint32(8, profileEnum, true);
    legacyDv.setUint32(12, unicodeEnum, true);
    legacyDv.setUint32(16, scoringEnum, true);
    legacyDv.setUint32(20, docCount, true);
    legacyDv.setUint32(24, rowCount, true);
    legacyDv.setUint32(28, tokenCount, true);
    legacyDv.setUint32(32, normalizedVal, true);
    legacyDv.setUint32(36, schemaLen, true);
    legacyDv.setUint32(40, docsLen, true);
    new Uint8Array(legacyBuf, LEGACY_SNAPSHOT_HEADER_BYTES, schemaLen).set(schemaBytes);
    new Uint8Array(legacyBuf, LEGACY_SNAPSHOT_HEADER_BYTES + schemaLen, tokensBytes.length).set(tokensBytes);
    new Uint8Array(
      legacyBuf,
      LEGACY_SNAPSHOT_HEADER_BYTES + schemaLen + tokensBytes.length,
      offsetsBytes.length
    ).set(offsetsBytes);
    new Uint8Array(
      legacyBuf,
      LEGACY_SNAPSHOT_HEADER_BYTES + schemaLen + tokensBytes.length + offsetsBytes.length,
      docsBytes.length
    ).set(docsBytes);
    const legacyCrc = crc32Parts([
      new Uint8Array(legacyBuf, 0, 44),
      new Uint8Array(legacyBuf, LEGACY_SNAPSHOT_HEADER_BYTES)
    ]);
    legacyDv.setUint32(44, legacyCrc, true);

    const legacyHeader = decodeSnapshotHeader(legacyBuf);
    assert.strictEqual(legacyHeader.magic, LEGACY_SNAPSHOT_MAGIC);
    assert.strictEqual(legacyHeader.formatVersion, LEGACY_SNAPSHOT_VERSION);
    assert.strictEqual(legacyHeader.columnarByteLength ?? 0, 0);
    assert.strictEqual(legacyHeader.docCount, 2);
    const legacyRestored = await restoreSnapshot(legacyBuf);
    assert.strictEqual(legacyRestored.getStats().docCount, 2);
    const legacyRes = await legacyRestored.search('migration');
    assert.strictEqual(legacyRes.totalMatches, 2);
    // Legacy tamper still fails closed.
    const badLegacy = legacyBuf.slice(0);
    new Uint8Array(badLegacy, LEGACY_SNAPSHOT_HEADER_BYTES, 1)[0] ^= 0xff;
    await assert.rejects(async () => restoreSnapshot(badLegacy));
    snapshotIndex.destroy();
    legacyRestored.destroy();
    console.log('   ✅ legacy snapshot legacy migration read path confirmed');
  }

  // =========================================================================
  // 11. Columnar Shape Negative Cases (fail-closed with recomputed CRC)
  // =========================================================================
  console.log('11. Testing columnar shape negatives (fail-closed beyond CRC)...');
  {
    const { crc32Parts, validateColumnarPayload } = await import('../packages/webgpu-search/src/index');
    interface NegDoc {
      id: string;
      title: string;
      level: string;
    }
    const negDocs: NegDoc[] = [
      { id: 'n-1', title: 'timeout alpha', level: 'ERROR' },
      { id: 'n-2', title: 'timeout beta', level: 'INFO' }
    ];
    const negIndex = await DocumentIndex.create(negDocs, {
      fields: ['title'],
      filterFields: [{ name: 'level' }]
    });
    const baseSnap = negIndex.serialize();
    const baseHeader = decodeSnapshotHeader(baseSnap);
    const colStart =
      SNAPSHOT_HEADER_BYTES + baseHeader.schemaByteLength + baseHeader.tokenCount * 4 + (baseHeader.rowCount + 1) * 4;
    const colLen = baseHeader.columnarByteLength ?? 0;
    assert(colLen > 0);
    const recomputeCrc = (buf: ArrayBuffer): number => {
      const h = decodeSnapshotHeader(buf);
      const cLen = h.columnarByteLength ?? 0;
      const tBytes = h.tokenCount * 4;
      const oBytes = (h.rowCount + 1) * 4;
      return crc32Parts([
        new Uint8Array(buf, 0, 52),
        new Uint8Array(buf, SNAPSHOT_HEADER_BYTES, h.schemaByteLength),
        new Uint8Array(buf, SNAPSHOT_HEADER_BYTES + h.schemaByteLength, tBytes),
        new Uint8Array(buf, SNAPSHOT_HEADER_BYTES + h.schemaByteLength + tBytes, oBytes),
        cLen > 0
          ? new Uint8Array(buf, SNAPSHOT_HEADER_BYTES + h.schemaByteLength + tBytes + oBytes, cLen)
          : new Uint8Array(0),
        h.docsByteLength > 0
          ? new Uint8Array(buf, SNAPSHOT_HEADER_BYTES + h.schemaByteLength + tBytes + oBytes + cLen, h.docsByteLength)
          : new Uint8Array(0)
      ]);
    };
    const withColumnar = (mutate: (payload: any) => void): ArrayBuffer => {
      const buf = baseSnap.slice(0);
      const bytes = new Uint8Array(buf, colStart, colLen);
      const payload = JSON.parse(new TextDecoder().decode(bytes));
      mutate(payload);
      const rewritten = new TextEncoder().encode(JSON.stringify(payload));
      // Length-preserving mutations exercise the CRC/shape path (not the
      // length path). Length-changing shape cases are covered via direct
      // `validateColumnarPayload` calls below.
      assert.strictEqual(rewritten.length, colLen, 'shape-negative mutation must preserve columnar length');
      new Uint8Array(buf, colStart, colLen).set(rewritten);
      new DataView(buf).setUint32(52, recomputeCrc(buf), true);
      return buf;
    };
    // Field-name mismatch (same-length alias preserves columnar bytes).
    await assert.rejects(
      async () => restoreSnapshot(withColumnar((p) => { p.fields[0].name = 'LEVEL'; })),
      (err: any) => err instanceof IncompatibleIndexError
    );
    // Type mismatch via direct validator (adding a type changes JSON length,
    // so exercise shape validation without the snapshot length gate).
    assert.throws(
      () => validateColumnarPayload(
        new TextEncoder().encode(JSON.stringify({ v: 1, fields: [{ name: 'level', type: 'number' }], rows: [['ERROR'], ['INFO']] })),
        [{ name: 'level' }],
        2
      ),
      (err: any) => err instanceof IncompatibleIndexError
    );
    // rows.length !== docCount and row-width mismatch via direct shape validation
    // (full-snapshot length equality would mask these, so exercise the validator).
    assert.throws(
      () => validateColumnarPayload(
        new TextEncoder().encode(JSON.stringify({ v: 1, fields: [{ name: 'level' }], rows: [[ 'ERROR' ]] })),
        [{ name: 'level' }],
        2
      ),
      (err: any) => err instanceof IncompatibleIndexError
    );
    assert.throws(
      () => validateColumnarPayload(
        new TextEncoder().encode(JSON.stringify({ v: 1, fields: [{ name: 'level' }], rows: [[ 'ERROR', 'EXTRA' ], [ 'INFO' ]] })),
        [{ name: 'level' }],
        2
      ),
      (err: any) => err instanceof IncompatibleIndexError
    );
    // Stripped empty segment on a filtered non-empty index fails closed.
    assert.throws(
      () => validateColumnarPayload(new Uint8Array(0), [{ name: 'level' }], 2),
      (err: any) => err instanceof IncompatibleIndexError
    );
    // Non-JSON bytes (valid CRC over garbage).
    {
      const buf = baseSnap.slice(0);
      new Uint8Array(buf, colStart, colLen).fill(0x41);
      new DataView(buf).setUint32(52, recomputeCrc(buf), true);
      await assert.rejects(
        async () => restoreSnapshot(buf),
        (err: any) => err instanceof IncompatibleIndexError
      );
    }
    // Envelope version mismatch (v !== 1).
    await assert.rejects(
      async () => restoreSnapshot(withColumnar((p) => { p.v = 2; })),
      (err: any) => err instanceof IncompatibleIndexError
    );
    // Non-zero reserved word (with recomputed CRC to isolate the reserved check).
    // NOTE: recomputeCrc re-parses the header (which rejects reserved!=0), so
    // compute the CRC directly from known segment offsets here.
    {
      const buf = baseSnap.slice(0);
      new DataView(buf).setUint32(48, 1, true);
      const tBytes = baseHeader.tokenCount * 4;
      const oBytes = (baseHeader.rowCount + 1) * 4;
      const cLen = baseHeader.columnarByteLength ?? 0;
      const crcNoHeaderParse = crc32Parts([
        new Uint8Array(buf, 0, 52),
        new Uint8Array(buf, SNAPSHOT_HEADER_BYTES, baseHeader.schemaByteLength),
        new Uint8Array(buf, SNAPSHOT_HEADER_BYTES + baseHeader.schemaByteLength, tBytes),
        new Uint8Array(buf, SNAPSHOT_HEADER_BYTES + baseHeader.schemaByteLength + tBytes, oBytes),
        cLen > 0
          ? new Uint8Array(buf, SNAPSHOT_HEADER_BYTES + baseHeader.schemaByteLength + tBytes + oBytes, cLen)
          : new Uint8Array(0),
        baseHeader.docsByteLength > 0
          ? new Uint8Array(buf, SNAPSHOT_HEADER_BYTES + baseHeader.schemaByteLength + tBytes + oBytes + cLen, baseHeader.docsByteLength)
          : new Uint8Array(0)
      ]);
      new DataView(buf).setUint32(52, crcNoHeaderParse, true);
      await assert.rejects(
        async () => restoreSnapshot(buf),
        (err: any) => err instanceof IncompatibleIndexError
      );
    }
    // Truncated columnar segment (length mismatch).
    {
      const buf = baseSnap.slice(0, baseSnap.byteLength - 1);
      await assert.rejects(
        async () => restoreSnapshot(buf),
        (err: any) => err instanceof IncompatibleIndexError
      );
    }
    // Empty filterFields ⇒ columnarByteLength 0.
    {
      const plain = await DocumentIndex.create(negDocs, { fields: ['title'] });
      const plainSnap = plain.serialize();
      const plainHeader = decodeSnapshotHeader(plainSnap);
      assert.strictEqual(plainHeader.columnarByteLength ?? 0, 0);
      plain.destroy();
    }
    negIndex.destroy();
    console.log('   ✅ Columnar shape negatives fail closed');
  }

  // =========================================================================
  // 12. Custom Filter Getter Guard (fail-closed unless override supplied)
  // =========================================================================
  console.log('12. Testing custom filter getter restore guard...');
  {
    interface GetterDoc {
      id: string;
      title: string;
      perf: { latency: number };
    }
    const getterDocs: GetterDoc[] = [
      { id: 'g-1', title: 'timeout alpha', perf: { latency: 1200 } },
      { id: 'g-2', title: 'timeout beta', perf: { latency: 10 } }
    ];
    const getterIndex = await DocumentIndex.create(getterDocs, {
      fields: ['title'],
      filterFields: [{ name: 'latencyMs', type: 'number', getter: (d: GetterDoc) => d.perf.latency }]
    });
    const getterSnap = getterIndex.serialize();
    const getterHeader = decodeSnapshotHeader(getterSnap);
    assert.strictEqual(getterHeader.columnarByteLength! > 0, true);
    // Restore without override must fail closed (default doc[name] would clear presence).
    await assert.rejects(
      async () => restoreSnapshot<GetterDoc>(getterSnap),
      (err: any) => err instanceof IncompatibleIndexError
    );
    // Restore with matching getter override succeeds with filter parity.
    const restoredGetter = await restoreSnapshot<GetterDoc>(getterSnap, {
      options: {
        filterFields: [{ name: 'latencyMs', type: 'number', getter: (d: GetterDoc) => d.perf.latency }]
      }
    });
    const getterRes = await restoredGetter.search('timeout', {
      mode: 'fuzzy',
      filter: { latencyMs: { gte: 900 } }
    });
    assert.strictEqual(getterRes.totalMatches, 1);
    assert.strictEqual(getterRes.results[0].id, 'g-1');
    getterIndex.destroy();
    restoredGetter.destroy();
    console.log('   ✅ Custom filter getter guard confirmed');
  }

  // =========================================================================
  // 13. Snapshot Size Caps & JSON-Safe Columnar Encoding
  // =========================================================================
  console.log('13. Testing snapshot size caps & JSON-safe columnar encoding...');
  {
    const { encodeColumnarPayload, validateColumnarPayload } = await import('../packages/webgpu-search/src/index');
    // BigInt / function / symbol / undefined normalize without throwing.
    {
      const bytes = encodeColumnarPayload(
        [{ id: 'b-1' } as any],
        [{ name: 'level', getter: () => 10n as any }]
      );
      assert(bytes.length > 0);
      const parsed = JSON.parse(new TextDecoder().decode(bytes));
      assert.strictEqual(parsed.rows[0][0], '10');
      const fnBytes = encodeColumnarPayload([{ id: 'b-1' } as any], [{ name: 'f', getter: () => (() => 1) as any }]);
      const fnParsed = JSON.parse(new TextDecoder().decode(fnBytes));
      assert.strictEqual(fnParsed.rows[0][0], null);
    }
    // Oversize lengths fail closed before decode (no OOM allocation).
    {
      const big = new Uint8Array(8);
      assert.throws(
        () => validateColumnarPayload(new Uint8Array(70 * 1024 * 1024), [{ name: 'level' }], 1),
        (err: any) => err instanceof IncompatibleIndexError
      );
      void big;
      const oversizeHeader = { schemaByteLength: 20 << 20, columnarByteLength: 0, docsByteLength: 0, docCount: 1, tokenCount: 1, rowCount: 1 };
      void oversizeHeader;
      // Full-snapshot oversize path: craft a header claiming >16 MiB schema.
      const small = await DocumentIndex.create([{ id: 's-1', title: 'hello' }], { fields: ['title'] });
      const snap = small.serialize();
      const tampered = snap.slice(0);
      new DataView(tampered).setUint32(36, 20 << 20, true);
      await assert.rejects(
        async () => restoreSnapshot(tampered),
        (err: any) => err instanceof IncompatibleIndexError
      );
      small.destroy();
    }
    console.log('   ✅ Size caps & JSON-safe encoding confirmed');
  }

  // =========================================================================
  // 14. HookIds Restore Guard via Snapshot Path (missing / mismatched)
  // =========================================================================
  console.log('14. Testing hookIds restore guard (missing + mismatched → IncompatibleHookError)...');
  {
    function recencyBoost(doc: any, s: number) { return s + 1; }
    (recencyBoost as any).hookId = 'recency-v1';
    function otherBoost(doc: any, s: number) { return s + 2; }
    (otherBoost as any).hookId = 'other-v1';
    const hookDocs = [
      { id: 'h-1', title: 'timeout alpha' },
      { id: 'h-2', title: 'timeout beta' }
    ];
    const hookIndex = await DocumentIndex.create(hookDocs, {
      fields: ['title'],
      hooks: { scoringHook: recencyBoost as never }
    });
    const hookSnap = hookIndex.serialize();
    const hookDecoded = decodeSnapshot(hookSnap);
    assert.strictEqual(hookDecoded.schema.hookIds?.scoringHook, 'recency-v1');

    // Missing handler → IncompatibleHookError (never IncompatibleIndexError).
    await assert.rejects(
      async () => restoreSnapshot(hookSnap),
      (err: any) => err instanceof IncompatibleHookError && err.hookId === 'recency-v1'
    );
    // Mismatched hookId → IncompatibleHookError.
    await assert.rejects(
      async () => restoreSnapshot(hookSnap, {
        options: { hooks: { scoringHook: otherBoost as never } }
      }),
      (err: any) => err instanceof IncompatibleHookError
    );
    // Matching handler restores with identical matches.
    const hookRestored = await restoreSnapshot<typeof hookDocs[0]>(hookSnap, {
      options: { hooks: { scoringHook: recencyBoost as never } }
    });
    assert.strictEqual(hookRestored.getStats().docCount, 2);
    const hookRes = await hookRestored.search('timeout');
    assert.strictEqual(hookRes.totalMatches, 2);
    // Instance restore() enforces the same guard.
    const hookDst = await DocumentIndex.create(hookDocs, { fields: ['title'] });
    await assert.rejects(
      async () => hookDst.restore(hookSnap),
      (err: any) => err instanceof IncompatibleHookError
    );
    await hookDst.restore(hookSnap, {
      options: { hooks: { scoringHook: recencyBoost as never } }
    });
    assert.strictEqual(hookDst.getStats().docCount, 2);
    hookIndex.destroy();
    hookRestored.destroy();
    hookDst.destroy();
    console.log('   ✅ hookIds restore guard confirmed');
  }

  // =========================================================================
  // 15. Worker Serialize / Restore Version Guard (rehydrated errors)
  // =========================================================================
  console.log('15. Testing worker serialize/restore version guard...');
  {
    const { clientWorker } = createMockWorkerScope();
    const workerClient = new SearchWorkerClient<{ id: string; title: string }>({
      worker: clientWorker
    });
    await workerClient.init(
      [
        { id: 'w-1', title: 'worker version guard alpha' },
        { id: 'w-2', title: 'worker version guard beta' }
      ],
      { fields: ['title'] }
    );
    const workerSnap = await workerClient.serialize();
    assert(workerSnap instanceof ArrayBuffer);
    assert.strictEqual(decodeSnapshotHeader(workerSnap).formatVersion, SNAPSHOT_FORMAT_VERSION);

    // Corrupt magic across the worker boundary → rehydrated IncompatibleIndexError.
    const badMagic = workerSnap.slice(0);
    new DataView(badMagic).setUint32(0, 0x11223344, true);
    const { clientWorker: cwBad } = createMockWorkerScope();
    const badClient = new SearchWorkerClient({ worker: cwBad });
    await assert.rejects(
      async () => badClient.restore(badMagic),
      (err: any) => err instanceof IncompatibleIndexError
    );
    // Corrupt version across the worker boundary → rehydrated IncompatibleIndexError.
    const badVer = workerSnap.slice(0);
    new DataView(badVer).setUint32(4, 999, true);
    await assert.rejects(
      async () => badClient.restore(badVer),
      (err: any) => err instanceof IncompatibleIndexError
    );
    // Corrupt CRC payload across the worker boundary → IncompatibleIndexError.
    const badCrc = workerSnap.slice(0);
    new Uint8Array(badCrc)[60] ^= 0xff;
    await assert.rejects(
      async () => badClient.restore(badCrc),
      (err: any) => err instanceof IncompatibleIndexError
    );
    // Oversize guard on the client (no worker round-trip): >512 MiB rejects.
    assert.strictEqual(MAX_SNAPSHOT_BYTES, 512 << 20);
    await assert.rejects(
      async () => badClient.restore({ byteLength: MAX_SNAPSHOT_BYTES + 1 } as any),
      (err: any) => err instanceof TypeError || err instanceof IncompatibleIndexError
    );
    await workerClient.destroy();
    await badClient.destroy();
    console.log('   ✅ Worker version guard confirmed');
  }

  // =========================================================================
  // 16. Oversize Caps Across All MAX_SNAPSHOT_* (fail-closed, no OOM)
  // =========================================================================
  console.log('16. Testing oversize caps across all MAX_SNAPSHOT_*...');
  {
    assert.strictEqual(MAX_SNAPSHOT_SCHEMA_BYTES, 16 << 20);
    assert.strictEqual(MAX_SNAPSHOT_COLUMNAR_BYTES, 64 << 20);
    assert.strictEqual(MAX_SNAPSHOT_DOCS_BYTES, 256 << 20);
    assert.strictEqual(MAX_SNAPSHOT_BYTES, 512 << 20);
    assert.strictEqual(MAX_SNAPSHOT_DOC_COUNT, 10_000_000);
    assert.strictEqual(MAX_SNAPSHOT_TOKEN_COUNT, 256_000_000);

    const capIndex = await DocumentIndex.create([{ id: 'c-1', title: 'caps' }], { fields: ['title'] });
    const capSnap = capIndex.serialize();
    const DV = (b: ArrayBuffer) => new DataView(b);
    // docsBytes cap (word@40): claim 300 MiB docs.
    {
      const buf = capSnap.slice(0);
      DV(buf).setUint32(40, 300 << 20, true);
      await assert.rejects(
        async () => restoreSnapshot(buf),
        (err: any) => err instanceof IncompatibleIndexError
      );
    }
    // docCount cap (word@20): claim 20M docs.
    {
      const buf = capSnap.slice(0);
      DV(buf).setUint32(20, 20_000_000, true);
      await assert.rejects(
        async () => restoreSnapshot(buf),
        (err: any) => err instanceof IncompatibleIndexError
      );
    }
    // tokenCount cap (word@28): claim 300M tokens.
    {
      const buf = capSnap.slice(0);
      DV(buf).setUint32(28, 300_000_000, true);
      await assert.rejects(
        async () => restoreSnapshot(buf),
        (err: any) => err instanceof IncompatibleIndexError
      );
    }
    // schema cap (word@36): claim 20 MiB schema (re-pins §13 via header word).
    {
      const buf = capSnap.slice(0);
      DV(buf).setUint32(36, 20 << 20, true);
      await assert.rejects(
        async () => restoreSnapshot(buf),
        (err: any) => err instanceof IncompatibleIndexError
      );
    }
    capIndex.destroy();
    console.log('   ✅ All MAX_SNAPSHOT_* caps fail closed');
  }

  // =========================================================================
  // 17. Profile Guards + Live-Format Semantics (never read under wrong profile)
  // =========================================================================
  console.log('17. Testing profile guards + live-format semantics...');
  {
    const { crc32Parts } = await import('../packages/webgpu-search/src/index');
    const profDocs = [
      { id: 'p-1', title: 'Profile guard alpha' },
      { id: 'p-2', title: 'Profile guard beta' }
    ];
    const profIndex = await DocumentIndex.create(profDocs, { fields: ['title'] });
    const profSnap = profIndex.serialize();

    // Canonical restore reports live format 4.
    const profRestored = await DocumentIndex.fromSnapshot(profSnap);
    assert.strictEqual(profRestored.getStats().formatVersion, SNAPSHOT_FORMAT_VERSION);
    assert.strictEqual(profRestored.getStats().formatVersion, 4);

    // Explicit caseSensitive override disagreeing with the snapshot → ProfileMismatchError.
    // Snapshot built case-insensitive (normalized true) → override true mismatches.
    await assert.rejects(
      async () => DocumentIndex.fromSnapshot(profSnap, { options: { caseSensitive: true } as any }),
      (err: any) => err instanceof ProfileMismatchError && (err as any).property === 'caseSensitive'
    );
    // Matching override succeeds.
    const profMatch = await DocumentIndex.fromSnapshot(profSnap, { options: { caseSensitive: false } as any });
    assert.strictEqual(profMatch.getStats().docCount, 2);
    profMatch.destroy();

    // textProfile override mismatch → ProfileMismatchError.
    await assert.rejects(
      async () => DocumentIndex.fromSnapshot(profSnap, { options: { textProfile: 'other-profile' as any } as any }),
      (err: any) => err instanceof ProfileMismatchError && (err as any).property === 'textProfile'
    );

    // Instance restore() cross-polarity → ProfileMismatchError (never reinterpret tokens).
    const sensitiveLive = await DocumentIndex.create(profDocs, { fields: ['title'], caseSensitive: true });
    await assert.rejects(
      async () => sensitiveLive.restore(profSnap),
      (err: any) => err instanceof ProfileMismatchError
    );
    // Instance restore() with agreeing polarity succeeds.
    const insensitiveLive = await DocumentIndex.create(profDocs, { fields: ['title'], caseSensitive: false });
    await insensitiveLive.restore(profSnap);
    assert.strictEqual(insensitiveLive.getStats().docCount, 2);

    // schema.caseSensitive / header.normalized disagreement → IncompatibleIndexError.
    {
      const header = decodeSnapshotHeader(profSnap);
      const schemaLen = header.schemaByteLength;
      const schemaBytes = new Uint8Array(profSnap, SNAPSHOT_HEADER_BYTES, schemaLen);
      const schema = JSON.parse(new TextDecoder().decode(schemaBytes));
      schema.caseSensitive = !schema.caseSensitive;
      let schemaStr = JSON.stringify(schema);
      // Length-preserving rewrite: flipping false<->true changes JSON length
      // by 1, so pad with JSON-whitespace before the final `}` (parser-ignored).
      if (schemaStr.length < schemaLen) {
        const pad = ' '.repeat(schemaLen - schemaStr.length);
        schemaStr = schemaStr.slice(0, -1) + pad + '}';
      }
      const rewritten = new TextEncoder().encode(schemaStr);
      assert.strictEqual(rewritten.length, schemaLen, 'polarity flip must preserve schema length');
      const buf = profSnap.slice(0);
      new Uint8Array(buf, SNAPSHOT_HEADER_BYTES, schemaLen).set(rewritten);
      const tBytes = header.tokenCount * 4;
      const oBytes = (header.rowCount + 1) * 4;
      const cLen = header.columnarByteLength ?? 0;
      const crc = crc32Parts([
        new Uint8Array(buf, 0, 52),
        new Uint8Array(buf, SNAPSHOT_HEADER_BYTES, schemaLen),
        new Uint8Array(buf, SNAPSHOT_HEADER_BYTES + schemaLen, tBytes),
        new Uint8Array(buf, SNAPSHOT_HEADER_BYTES + schemaLen + tBytes, oBytes),
        cLen > 0
          ? new Uint8Array(buf, SNAPSHOT_HEADER_BYTES + schemaLen + tBytes + oBytes, cLen)
          : new Uint8Array(0),
        header.docsByteLength > 0
          ? new Uint8Array(buf, SNAPSHOT_HEADER_BYTES + schemaLen + tBytes + oBytes + cLen, header.docsByteLength)
          : new Uint8Array(0)
      ]);
      new DataView(buf).setUint32(52, crc, true);
      await assert.rejects(
        async () => restoreSnapshot(buf),
        (err: any) => err instanceof IncompatibleIndexError
      );
    }

    // Legacy restore still reports live format 4 (not source version 3).
    {
      const legacyIndex = await DocumentIndex.create(profDocs, { fields: ['title'] });
      const legacySnap = legacyIndex.serialize();
      const snapHeader = decodeSnapshotHeader(legacySnap);
      const snapDv = new DataView(legacySnap);
      const profileEnum = snapDv.getUint32(8, true);
      const unicodeEnum = snapDv.getUint32(12, true);
      const scoringEnum = snapDv.getUint32(16, true);
      const docCount = snapDv.getUint32(20, true);
      const rowCount = snapDv.getUint32(24, true);
      const tokenCount = snapDv.getUint32(28, true);
      const normalizedVal = snapDv.getUint32(32, true);
      const schemaLenL = snapDv.getUint32(36, true);
      const docsLenL = snapDv.getUint32(40, true);
      const schemaBytesL = new Uint8Array(legacySnap, SNAPSHOT_HEADER_BYTES, schemaLenL);
      const tokensBytesL = new Uint8Array(legacySnap, SNAPSHOT_HEADER_BYTES + schemaLenL, tokenCount * 4);
      const offsetsBytesL = new Uint8Array(
        legacySnap,
        SNAPSHOT_HEADER_BYTES + schemaLenL + tokenCount * 4,
        (rowCount + 1) * 4
      );
      const docsBytesL = new Uint8Array(
        legacySnap,
        SNAPSHOT_HEADER_BYTES + schemaLenL + tokenCount * 4 + (rowCount + 1) * 4,
        docsLenL
      );
      const legacyTotal = LEGACY_SNAPSHOT_HEADER_BYTES + schemaLenL + tokenCount * 4 + (rowCount + 1) * 4 + docsLenL;
      const legacyBuf = new ArrayBuffer(legacyTotal);
      const legacyDv = new DataView(legacyBuf);
      legacyDv.setUint32(0, LEGACY_SNAPSHOT_MAGIC, true);
      legacyDv.setUint32(4, LEGACY_SNAPSHOT_VERSION, true);
      legacyDv.setUint32(8, profileEnum, true);
      legacyDv.setUint32(12, unicodeEnum, true);
      legacyDv.setUint32(16, scoringEnum, true);
      legacyDv.setUint32(20, docCount, true);
      legacyDv.setUint32(24, rowCount, true);
      legacyDv.setUint32(28, tokenCount, true);
      legacyDv.setUint32(32, normalizedVal, true);
      legacyDv.setUint32(36, schemaLenL, true);
      legacyDv.setUint32(40, docsLenL, true);
      new Uint8Array(legacyBuf, LEGACY_SNAPSHOT_HEADER_BYTES, schemaLenL).set(schemaBytesL);
      new Uint8Array(legacyBuf, LEGACY_SNAPSHOT_HEADER_BYTES + schemaLenL, tokensBytesL.length).set(tokensBytesL);
      new Uint8Array(
        legacyBuf,
        LEGACY_SNAPSHOT_HEADER_BYTES + schemaLenL + tokensBytesL.length,
        offsetsBytesL.length
      ).set(offsetsBytesL);
      new Uint8Array(
        legacyBuf,
        LEGACY_SNAPSHOT_HEADER_BYTES + schemaLenL + tokensBytesL.length + offsetsBytesL.length,
        docsBytesL.length
      ).set(docsBytesL);
      const legacyCrc = crc32Parts([
        new Uint8Array(legacyBuf, 0, 44),
        new Uint8Array(legacyBuf, LEGACY_SNAPSHOT_HEADER_BYTES)
      ]);
      legacyDv.setUint32(44, legacyCrc, true);
      const legacyRestored = await restoreSnapshot(legacyBuf);
      assert.strictEqual(decodeSnapshotHeader(legacyBuf).formatVersion, LEGACY_SNAPSHOT_VERSION);
      assert.strictEqual(legacyRestored.getStats().formatVersion, SNAPSHOT_FORMAT_VERSION);
      assert.strictEqual(legacyRestored.getStats().formatVersion, 4);
      legacyIndex.destroy();
      legacyRestored.destroy();
    }

    profIndex.destroy();
    profRestored.destroy();
    sensitiveLive.destroy();
    insensitiveLive.destroy();
    console.log('   ✅ Profile guards + live-format semantics confirmed');
  }

  console.log('\n--- All Persistence Tests Passed Successfully! ✅ ---');
}

runSnapshotTests().catch((err) => {
  console.error('❌  Tests failed:', err);
  process.exit(1);
});
