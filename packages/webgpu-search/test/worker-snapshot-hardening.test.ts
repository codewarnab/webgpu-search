/**
 * Regression suite: worker message serialization + build-then-swap, client
 * restore/init commit ordering, snapshot docIds/docs validation,
 * ArrayBufferView snapshot inputs, and IndexedDB custom store provisioning.
 *
 * Run: bun test packages/webgpu-search/test/worker-snapshot-hardening.test.ts
 */
import { describe, test, expect } from 'bun:test';
import { DocumentIndex } from '../src/document-index';
import { SearchWorkerClient } from '../src/worker/worker-client';
import { startSearchWorker } from '../src/worker/search-worker';
import {
  decodeSnapshot,
  decodeSnapshotHeader,
  restoreSnapshot,
  toSnapshotArrayBuffer
} from '../src/snapshot-codec';
import { loadIndexFromIDB, openSearchDatabase, restoreIndexFromIDB, saveIndexToIDB } from '../src/snapshot-idb';
import { crc32Parts } from '../src/dataset-packing';
import { IncompatibleIndexError } from '../src/text-profile';
import { SNAPSHOT_HEADER_BYTES } from '../src/text-profile';

const OPTS = { fields: ['t'], preferGpu: false } as const;

async function makeSnapshot(records: Array<Record<string, unknown>>, decoupled = false): Promise<ArrayBuffer> {
  const idx = await DocumentIndex.create(records, OPTS as any);
  try {
    return idx.serialize({ decoupled });
  } finally {
    idx.destroy();
  }
}

/** Rewrites schema and/or docs JSON of a canonical snapshot with a valid CRC. */
function forgeSnapshot(
  buf: ArrayBuffer,
  mutate: { schema?: (s: any) => any; docs?: (d: any) => any }
): ArrayBuffer {
  const h = decodeSnapshotHeader(buf);
  const u8 = new Uint8Array(buf);
  let cur = SNAPSHOT_HEADER_BYTES;
  const schemaBytes = u8.slice(cur, cur + h.schemaByteLength); cur += h.schemaByteLength;
  const tokens = u8.slice(cur, cur + h.tokenCount * 4); cur += h.tokenCount * 4;
  const offsets = u8.slice(cur, cur + (h.rowCount + 1) * 4); cur += (h.rowCount + 1) * 4;
  const columnar = u8.slice(cur, cur + h.columnarByteLength); cur += h.columnarByteLength;
  const docsBytes = u8.slice(cur, cur + h.docsByteLength);
  const enc = new TextEncoder();
  const dec = new TextDecoder();
  const newSchema = mutate.schema ? enc.encode(JSON.stringify(mutate.schema(JSON.parse(dec.decode(schemaBytes))))) : schemaBytes;
  const newDocs = mutate.docs && docsBytes.length > 0 ? enc.encode(JSON.stringify(mutate.docs(JSON.parse(dec.decode(docsBytes))))) : docsBytes;
  const total = SNAPSHOT_HEADER_BYTES + newSchema.length + tokens.length + offsets.length + columnar.length + newDocs.length;
  const out = new Uint8Array(total);
  out.set(u8.subarray(0, SNAPSHOT_HEADER_BYTES));
  let o = SNAPSHOT_HEADER_BYTES;
  for (const part of [newSchema, tokens, offsets, columnar, newDocs]) { out.set(part, o); o += part.length; }
  const dv = new DataView(out.buffer);
  dv.setUint32(36, newSchema.length, true);
  dv.setUint32(40, newDocs.length, true);
  dv.setUint32(52, crc32Parts([out.subarray(0, 52), newSchema, tokens, offsets, columnar, newDocs]), true);
  return out.buffer;
}

function createWorkerHarness() {
  const out: any[] = [];
  let listener: (e: any) => Promise<void> = async () => {};
  const scope = { postMessage: (m: any) => out.push(m), addEventListener: (_: string, l: any) => (listener = l) };
  startSearchWorker(scope);
  return { out, send: (d: any) => listener({ data: d }), last: (id: number) => out.filter((m) => m.id === id).pop() };
}

function createMockWorkerPair() {
  const clientListeners: Array<(e: any) => void> = [];
  const workerListeners: Array<(e: any) => void> = [];
  const clientWorker: any = {
    postMessage(data: any) { queueMicrotask(() => workerListeners.forEach((l) => l({ data }))); },
    addEventListener(ev: string, l: any) { if (ev === 'message') clientListeners.push(l); },
    removeEventListener(ev: string, l: any) { const i = clientListeners.indexOf(l); if (ev === 'message' && i >= 0) clientListeners.splice(i, 1); },
    terminate() {}
  };
  const workerScope = {
    postMessage(data: any) { queueMicrotask(() => clientListeners.forEach((l) => l({ data }))); },
    addEventListener(ev: string, l: any) { if (ev === 'message') workerListeners.push(l); }
  };
  startSearchWorker(workerScope);
  return clientWorker;
}

describe('worker build-then-swap + serialization', () => {
  test('failed RESTORE / INIT keep the previous index live', async () => {
    const w = createWorkerHarness();
    await w.send({ id: 1, type: 'INIT', payload: { options: OPTS, records: [{ id: 1, t: 'hello' }] } });
    await w.send({ id: 2, type: 'RESTORE', payload: { buffer: new ArrayBuffer(10) } });
    expect(w.last(2).success).toBe(false);
    expect(w.last(2).error.name).toBe('IncompatibleIndexError');
    await w.send({ id: 3, type: 'INIT', payload: { options: OPTS, records: [{ id: 1, t: 'a' }, { id: 1, t: 'b' }] } });
    expect(w.last(3).success).toBe(false);
    await w.send({ id: 4, type: 'SEARCH', payload: { queryId: 1, query: 'hello' } });
    expect(w.last(4).success).toBe(true);
    expect(w.last(4).result.results[0].id).toBe(1);
  });

  test('overlapping INITs + DESTROY end with no index and no leaked instance', async () => {
    const w = createWorkerHarness();
    const created: any[] = [];
    const destroyed = new Set<any>();
    const origCreate = DocumentIndex.create;
    const origDestroy = DocumentIndex.prototype.destroy;
    (DocumentIndex as any).create = async function (...args: any[]) {
      const i = await (origCreate as any).apply(this, args); created.push(i); return i;
    };
    DocumentIndex.prototype.destroy = function () { destroyed.add(this); return origDestroy.call(this); };
    try {
      const pA = w.send({ id: 10, type: 'INIT', payload: { options: OPTS, records: [{ id: 1, t: 'a' }] } });
      const pB = w.send({ id: 11, type: 'INIT', payload: { options: OPTS, records: [{ id: 2, t: 'b' }] } });
      await Promise.resolve(); // let INIT A start building
      await Promise.resolve();
      const pD = w.send({ id: 12, type: 'DESTROY' });
      await Promise.all([pA, pB, pD]);
    } finally {
      (DocumentIndex as any).create = origCreate;
      DocumentIndex.prototype.destroy = origDestroy;
    }
    // Replies superseded by DESTROY are dropped (session epoch); INIT B never starts.
    expect(w.out.filter((m) => m.id >= 10 && m.id <= 12).map((m) => [m.id, m.success])).toEqual([[12, true]]);
    expect(created.length).toBe(1);
    for (const i of created) expect(destroyed.has(i)).toBe(true);
    await w.send({ id: 13, type: 'STATS' });
    expect(w.last(13).success).toBe(false);
  });

  test('SEARCH issued during RESTORE waits for it', async () => {
    const w = createWorkerHarness();
    const good = await makeSnapshot([{ id: 1, t: 'hello' }]);
    await Promise.all([
      w.send({ id: 20, type: 'RESTORE', payload: { buffer: good } }),
      w.send({ id: 21, type: 'SEARCH', payload: { queryId: 5, query: 'hello' } })
    ]);
    expect(w.last(20).success).toBe(true);
    expect(w.last(20).result.fields.map((f: any) => f.name)).toEqual(['t']);
    expect(w.last(21).success).toBe(true);
    expect(w.last(21).result.results.length).toBe(1);
  });
});

describe('worker session epoch', () => {
  test('requests queued before DESTROY never answer after it', async () => {
    const w = createWorkerHarness();
    const pInit = w.send({ id: 1, type: 'INIT', payload: { options: OPTS, records: [{ id: 1, t: 'hello' }] } });
    const pSearch = w.send({ id: 2, type: 'SEARCH', payload: { queryId: 1, query: 'hello' } });
    const pDestroy = w.send({ id: 3, type: 'DESTROY' });
    await Promise.all([pInit, pSearch, pDestroy]);
    expect(w.out.map((m) => m.id)).toEqual([3]);
    // A fresh session on the same worker restarts query ids.
    await w.send({ id: 1, type: 'INIT', payload: { options: OPTS, records: [{ id: 9, t: 'again' }] } });
    await w.send({ id: 2, type: 'SEARCH', payload: { queryId: 1, query: 'again' } });
    expect(w.last(2).success).toBe(true);
    expect(w.last(2).result.results[0].id).toBe(9);
  });
});

describe('client restore/init commit ordering', () => {
  test('restore; add; search (not awaited) preserve call order', async () => {
    const source = new SearchWorkerClient<any>({ worker: createMockWorkerPair() });
    await source.init([{ id: 'x', t: 'other' }], { fields: ['t'], preferGpu: false } as any);
    const snap = await source.serialize();
    await source.destroy();
    const client = new SearchWorkerClient<any>({ worker: createMockWorkerPair() });
    await client.init([{ id: 'a', title: 'alpha' }], { fields: ['title'], preferGpu: false } as any);
    const pRestore = client.restore(snap);
    const pAdd = client.add({ id: 'y', t: 'yonder' });
    const pSearch = client.search('yonder');
    await pRestore;
    await pAdd;
    const res = await pSearch;
    expect(res.results.map((r: any) => r.id)).toEqual(['y']);
    expect(res.results[0].doc).toEqual({ id: 'y', t: 'yonder' });
    await client.destroy();
  });

  test('corrupt restore rejects with IncompatibleIndexError and keeps client state', async () => {
    const client = new SearchWorkerClient<any>({ worker: createMockWorkerPair() });
    await client.init([{ id: 'a', title: 'alpha' }], { fields: ['title'], preferGpu: false } as any);
    const good = await makeSnapshot([{ id: 'x', t: 'other' }]);
    const corrupt = good.slice(0);
    new Uint8Array(corrupt)[corrupt.byteLength - 2] ^= 0xff; // flip a docs byte → CRC mismatch
    await expect(client.restore(corrupt)).rejects.toBeInstanceOf(IncompatibleIndexError);
    expect(client.getRecords()).toEqual([{ id: 'a', title: 'alpha' }]);
    const res = await client.search('alpha');
    expect(res.results[0].doc).toEqual({ id: 'a', title: 'alpha' });
    // Outgoing field defs still match the live worker index.
    await client.add({ id: 'b', title: 'beta' });
    expect((await client.search('beta')).results[0].id).toBe('b');
    await client.destroy();
  });

  test('failed init (worker-side) does not install new field defs', async () => {
    const client = new SearchWorkerClient<any>({ worker: createMockWorkerPair() });
    await client.init([{ id: 'a', title: 'alpha' }], { fields: ['title'], preferGpu: false } as any);
    await expect(
      client.init([{ id: 'd', body: 'x' }, { id: 'd', body: 'y' }], { fields: ['body'], preferGpu: false } as any)
    ).rejects.toThrow();
    await client.add({ id: 'b', title: 'beta' });
    expect((await client.search('beta')).results[0].id).toBe('b');
    await client.destroy();
  });

  test('successful restore commits worker-validated schema and records', async () => {
    const client = new SearchWorkerClient<any>({ worker: createMockWorkerPair() });
    await client.init([{ id: 'a', title: 'alpha' }], { fields: ['title'], preferGpu: false } as any);
    const source = new SearchWorkerClient<any>({ worker: createMockWorkerPair() });
    await source.init([{ id: 'x', t: 'other' }], { fields: ['t'], preferGpu: false } as any);
    const snap = await source.serialize();
    await source.destroy();
    await client.restore(new Uint8Array(snap));
    expect(client.getRecords().map((d: any) => d.id)).toEqual(['x']);
    await client.add({ id: 'y', t: 'yonder' });
    expect((await client.search('yonder')).results[0].id).toBe('y');
    await client.destroy();
  });
});

describe('snapshot codec validation', () => {
  test('rejects duplicate / non-scalar / empty docIds, accepts 1 and "1"', async () => {
    const base = await makeSnapshot([{ id: 'p', t: 'a' }, { id: 'q', t: 'b' }], true);
    for (const ids of [['p', 'p'], [{}, 'q'], [[], 'q'], ['', 'q'], [null, 'q'], [Number.NaN, 'q']]) {
      const forged = forgeSnapshot(base, { schema: (s) => ({ ...s, docIds: ids }) });
      expect(() => decodeSnapshot(forged)).toThrow(IncompatibleIndexError);
    }
    const mixed = forgeSnapshot(base, { schema: (s) => ({ ...s, docIds: [1, '1'] }) });
    expect(decodeSnapshot(mixed).docIds).toEqual([1, '1']);
  });

  test('rejects non-object embedded docs with IncompatibleIndexError', async () => {
    const base = await makeSnapshot([{ id: 'p', t: 'a' }, { id: 'q', t: 'b' }]);
    for (const bad of [null, 5, 'str', []]) {
      const forged = forgeSnapshot(base, { docs: (d) => [d[0], bad] });
      expect(() => decodeSnapshot(forged)).toThrow(IncompatibleIndexError);
    }
  });

  test('accepts ArrayBufferViews, honouring byteOffset', async () => {
    const buf = await makeSnapshot([{ id: 'p', t: 'hello' }]);
    const padded = new Uint8Array(buf.byteLength + 16);
    padded.set(new Uint8Array(buf), 8);
    const view = padded.subarray(8, 8 + buf.byteLength);
    expect(decodeSnapshotHeader(view).docCount).toBe(1);
    expect(decodeSnapshot(view).docIds).toEqual(['p']);
    expect(decodeSnapshot(new DataView(padded.buffer, 8, buf.byteLength)).docIds).toEqual(['p']);
    const restored = await restoreSnapshot(new Uint8Array(buf));
    expect((await restored.search('hello')).results[0].id).toBe('p');
    restored.destroy();
    expect(toSnapshotArrayBuffer(new Uint8Array(buf))).toBe(buf);
    expect(toSnapshotArrayBuffer({ byteLength: 10, slice() {} })).toBeNull();
    expect(() => decodeSnapshot({ byteLength: 100, slice() {} } as any)).toThrow(IncompatibleIndexError);
  });
});

/** Strict versioned IDB mock: stores only created during upgrades. */
function createStrictIDB() {
  const dbs = new Map<string, { version: number; stores: Map<string, Map<any, any>> }>();
  const mkReq = () => ({ result: undefined as any, error: null as any, onsuccess: null as any, onerror: null as any, onupgradeneeded: null as any, onblocked: null as any });
  const connection = (rec: { version: number; stores: Map<string, Map<any, any>> }, upgrading: boolean) => ({
    get version() { return rec.version; },
    objectStoreNames: { contains: (n: string) => rec.stores.has(n) },
    createObjectStore(n: string) {
      if (!upgrading) throw Object.assign(new Error('not in versionchange'), { name: 'InvalidStateError' });
      rec.stores.set(n, new Map());
    },
    onversionchange: null as any,
    close() {},
    transaction(names: string[], _mode: string) {
      for (const n of names) if (!rec.stores.has(n)) throw Object.assign(new Error(`missing store ${n}`), { name: 'NotFoundError' });
      const tx: any = { oncomplete: null, onerror: null, onabort: null, error: null };
      let pending = 0;
      const done = () => queueMicrotask(() => { if (pending === 0) tx.oncomplete?.(); });
      tx.objectStore = (n: string) => {
        const m = rec.stores.get(n)!;
        const op = (fn: (r: any) => void) => { const r = mkReq(); pending++; queueMicrotask(() => { fn(r); r.onsuccess?.(); pending--; done(); }); return r; };
        return { put: (v: any, k: any) => op(() => m.set(k, v)), get: (k: any) => op((r) => { r.result = m.get(k); }), delete: (k: any) => op(() => m.delete(k)) };
      };
      queueMicrotask(done);
      return tx;
    }
  });
  const factory: any = {
    dbs,
    open(name: string, version?: number) {
      const req = mkReq();
      queueMicrotask(() => {
        let rec = dbs.get(name);
        if (!rec) { rec = { version: 0, stores: new Map() }; dbs.set(name, rec); }
        const target = version ?? Math.max(rec.version, 1);
        if (target < rec.version) { req.error = Object.assign(new Error('VersionError'), { name: 'VersionError' }); req.onerror?.(); return; }
        if (target > rec.version) {
          rec.version = target;
          req.result = connection(rec, true);
          req.onupgradeneeded?.();
        }
        req.result = connection(rec, false);
        req.onsuccess?.();
      });
      return req;
    }
  };
  return factory;
}

describe('IndexedDB store provisioning + buffer normalization', () => {
  test('custom store names on an existing DB are provisioned via version bump', async () => {
    const idb = createStrictIDB();
    const buf = await makeSnapshot([{ id: 'p', t: 'hello' }]);
    await saveIndexToIDB(buf, { indexedDB: idb });
    expect(idb.dbs.get('webgpu_search_db').version).toBe(1);
    const custom = { indexedDB: idb, snapshotStoreName: 'snaps2', docStoreName: 'docs2' };
    await saveIndexToIDB(buf, custom);
    expect(idb.dbs.get('webgpu_search_db').version).toBe(2);
    const loaded = await loadIndexFromIDB(custom);
    expect(loaded?.snapshot.byteLength).toBe(buf.byteLength);
    const db = await openSearchDatabase(custom); // no further bump needed
    expect(db.version).toBe(2);
  });

  test('typed-array snapshots are stored and loaded as exact ArrayBuffers', async () => {
    const idb = createStrictIDB();
    const buf = await makeSnapshot([{ id: 'p', t: 'hello' }]);
    const padded = new Uint8Array(buf.byteLength + 4);
    padded.set(new Uint8Array(buf), 4);
    await saveIndexToIDB(padded.subarray(4), { indexedDB: idb });
    const stored = idb.dbs.get('webgpu_search_db').stores.get('index_snapshots').get('default_index');
    expect(stored instanceof ArrayBuffer).toBe(true);
    expect(stored.byteLength).toBe(buf.byteLength);
    // Legacy record saved as a typed array is normalized on load.
    idb.dbs.get('webgpu_search_db').stores.get('index_snapshots').set('default_index', new Uint8Array(buf));
    const restored = await restoreIndexFromIDB({ indexedDB: idb });
    expect((await restored!.search('hello')).results[0].id).toBe('p');
    restored!.destroy();
  });
});
