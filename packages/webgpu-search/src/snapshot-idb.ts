import { restoreSnapshot, toSnapshotArrayBuffer } from './snapshot-codec';
import type { DocumentIndex } from './document-index';
import type { SearchWorkerClient } from './worker/worker-client';
import type {
  IDBStorageOptions,
  LoadIDBOptions,
  LoadIDBResult,
  RestoreDocumentIndexOptions,
  SaveIDBOptions
} from './types';

export const DEFAULT_IDB_DATABASE_NAME = 'webgpu_search_db';
export const DEFAULT_SNAPSHOT_STORE_NAME = 'index_snapshots';
export const DEFAULT_DOCUMENT_STORE_NAME = 'documents';
export const DEFAULT_SNAPSHOT_KEY = 'default_index';

function resolveIDBFactory(options?: IDBStorageOptions): IDBFactory {
  const idb =
    options?.indexedDB ??
    (typeof globalThis !== 'undefined' ? (globalThis as any).indexedDB : undefined);

  if (!idb || typeof idb.open !== 'function') {
    throw new Error('[webgpu-search] IndexedDB is not available in the current environment.');
  }

  return idb as IDBFactory;
}

function openRequest(
  idb: IDBFactory,
  dbName: string,
  version: number | undefined,
  requiredStores: string[]
): Promise<IDBDatabase> {
  return new Promise<IDBDatabase>((resolve, reject) => {
    const req = version === undefined ? idb.open(dbName) : idb.open(dbName, version);

    req.onupgradeneeded = () => {
      const db = req.result;
      for (const name of requiredStores) {
        if (!db.objectStoreNames.contains(name)) {
          db.createObjectStore(name);
        }
      }
    };

    req.onsuccess = () => {
      const db = req.result;
      // Yield to future version upgrades (e.g. another caller provisioning a
      // custom store) instead of blocking them for the connection lifetime.
      db.onversionchange = () => {
        db.close();
      };
      resolve(db);
    };

    req.onerror = () => {
      reject(req.error ?? new Error(`[webgpu-search] Failed to open IndexedDB database "${dbName}".`));
    };

    req.onblocked = () => {
      // The upgrade proceeds once other connections close (ours close on
      // `versionchange`); the request stays pending until then.
      console.warn(`[webgpu-search] IndexedDB database "${dbName}" open request blocked by other open tabs/connections.`);
    };
  });
}

const MAX_STORE_PROVISION_ATTEMPTS = 3;

/**
 * Opens the IndexedDB database, provisioning required object stores cleanly.
 *
 * Opens at the current version first; if any required store (including a
 * custom `snapshotStoreName` / `docStoreName`) is missing on an existing
 * database, the connection is closed and the database is reopened at
 * `version + 1` so `onupgradeneeded` can create the missing stores.
 */
export async function openSearchDatabase(options?: IDBStorageOptions): Promise<IDBDatabase> {
  const idb = resolveIDBFactory(options);
  const dbName = options?.dbName ?? DEFAULT_IDB_DATABASE_NAME;
  const snapshotStoreName = options?.snapshotStoreName ?? DEFAULT_SNAPSHOT_STORE_NAME;
  const docStoreName = options?.docStoreName ?? DEFAULT_DOCUMENT_STORE_NAME;
  const requiredStores = snapshotStoreName === docStoreName ? [snapshotStoreName] : [snapshotStoreName, docStoreName];

  let db = await openRequest(idb, dbName, undefined, requiredStores);
  for (let attempt = 0; attempt < MAX_STORE_PROVISION_ATTEMPTS; attempt++) {
    const missing = requiredStores.some((name) => !db.objectStoreNames.contains(name));
    if (!missing) {
      return db;
    }
    const nextVersion = db.version + 1;
    db.close();
    try {
      db = await openRequest(idb, dbName, nextVersion, requiredStores);
    } catch (err) {
      // A concurrent opener may have upgraded past `nextVersion` first
      // (VersionError); reopen at the current version and re-check.
      if ((err as { name?: string } | null)?.name !== 'VersionError') {
        throw err;
      }
      db = await openRequest(idb, dbName, undefined, requiredStores);
    }
  }
  if (requiredStores.some((name) => !db.objectStoreNames.contains(name))) {
    db.close();
    throw new Error(
      `[webgpu-search] Failed to provision IndexedDB object stores (${requiredStores.join(', ')}) in database "${dbName}".`
    );
  }
  return db;
}

/**
 * Transaction-safe helper to save a DocumentIndex or pre-serialized ArrayBuffer snapshot to IndexedDB.
 *
 * Invariants:
 * 1. Serializes completely BEFORE opening the transaction, preventing auto-commit TransactionInactiveError.
 * 2. Database connections are closed cleanly in finally blocks to allow future version migrations.
 * 3. Supports decoupled document persistence into a secondary document store.
 */
export async function saveIndexToIDB(
  indexOrBuffer: DocumentIndex<any> | SearchWorkerClient<any> | ArrayBuffer | ArrayBufferView,
  options?: SaveIDBOptions
): Promise<void> {
  if (!indexOrBuffer) {
    throw new TypeError('[webgpu-search] saveIndexToIDB requires a DocumentIndex, SearchWorkerClient, or ArrayBuffer.');
  }

  const decoupled = options?.decoupled === true;
  let snapshot: ArrayBuffer;
  let docsToStore: any[] | undefined = options?.documents;

  // 1. Serialization phase (OUTSIDE transaction to avoid auto-commit)
  // ArrayBuffer is stored as-is; ArrayBufferViews (typed arrays, DataView,
  // Buffer) are narrowed to their exact byte window so the stored value is
  // always a plain ArrayBuffer that decodes correctly on load.
  const normalizedBuffer = toSnapshotArrayBuffer(indexOrBuffer);

  if (normalizedBuffer) {
    snapshot = normalizedBuffer;
  } else if (typeof (indexOrBuffer as any)?.serialize === 'function') {
    snapshot = await (indexOrBuffer as any).serialize({ decoupled });
    if (decoupled && !docsToStore && typeof (indexOrBuffer as any)?.getRecords === 'function') {
      docsToStore = (indexOrBuffer as any).getRecords();
    }
  } else {
    throw new TypeError(
      '[webgpu-search] saveIndexToIDB expects DocumentIndex, SearchWorkerClient, or ArrayBuffer.'
    );
  }

  // 2. Transaction phase
  const snapshotStoreName = options?.snapshotStoreName ?? DEFAULT_SNAPSHOT_STORE_NAME;
  const docStoreName = options?.docStoreName ?? DEFAULT_DOCUMENT_STORE_NAME;
  const key = options?.key ?? DEFAULT_SNAPSHOT_KEY;

  let db: IDBDatabase | null = null;
  try {
    db = await openSearchDatabase(options);
    const storeNames =
      decoupled && docsToStore && docsToStore.length > 0
        ? [snapshotStoreName, docStoreName]
        : [snapshotStoreName];

    await new Promise<void>((resolve, reject) => {
      const tx = db!.transaction(storeNames, 'readwrite');

      tx.oncomplete = () => {
        resolve();
      };
      tx.onerror = () => {
        reject(tx.error ?? new Error('[webgpu-search] IDB transaction failed during saveIndexToIDB.'));
      };
      tx.onabort = () => {
        reject(tx.error ?? new Error('[webgpu-search] IDB transaction was aborted during saveIndexToIDB.'));
      };

      const snapStore = tx.objectStore(snapshotStoreName);
      snapStore.put(snapshot, key);

      if (decoupled && docsToStore && docsToStore.length > 0) {
        const docStore = tx.objectStore(docStoreName);
        docStore.put(docsToStore, key);
      }
    });
  } finally {
    if (db) {
      db.close();
    }
  }
}

/**
 * Loads a serialized snapshot (and optional decoupled documents) from IndexedDB.
 */
export async function loadIndexFromIDB<TDoc = Record<string, unknown>>(
  options?: LoadIDBOptions
): Promise<LoadIDBResult<TDoc> | null> {
  const snapshotStoreName = options?.snapshotStoreName ?? DEFAULT_SNAPSHOT_STORE_NAME;
  const docStoreName = options?.docStoreName ?? DEFAULT_DOCUMENT_STORE_NAME;
  const key = options?.key ?? DEFAULT_SNAPSHOT_KEY;
  const shouldLoadDocs = options?.loadDocuments !== false;

  let db: IDBDatabase | null = null;
  try {
    db = await openSearchDatabase(options);
    const storeNames = shouldLoadDocs && db.objectStoreNames.contains(docStoreName)
      ? [snapshotStoreName, docStoreName]
      : [snapshotStoreName];

    return await new Promise<LoadIDBResult<TDoc> | null>((resolve, reject) => {
      const tx = db!.transaction(storeNames, 'readonly');
      let snapshotResult: ArrayBuffer | null = null;
      let documentsResult: TDoc[] | undefined = undefined;

      tx.onerror = () => {
        reject(tx.error ?? new Error('[webgpu-search] IDB transaction failed during loadIndexFromIDB.'));
      };
      tx.onabort = () => {
        reject(tx.error ?? new Error('[webgpu-search] IDB transaction aborted during loadIndexFromIDB.'));
      };
      tx.oncomplete = () => {
        if (!snapshotResult) {
          resolve(null);
        } else {
          resolve({
            snapshot: snapshotResult,
            documents: documentsResult
          });
        }
      };

      const snapStore = tx.objectStore(snapshotStoreName);
      const snapReq = snapStore.get(key);
      snapReq.onsuccess = () => {
        const raw = snapReq.result as unknown;
        // Normalize legacy typed-array records to a plain ArrayBuffer; any
        // other non-empty value is passed through so restore fails closed.
        snapshotResult = raw === undefined || raw === null
          ? null
          : (toSnapshotArrayBuffer(raw) ?? (raw as ArrayBuffer));
      };

      if (shouldLoadDocs && db!.objectStoreNames.contains(docStoreName)) {
        const docStore = tx.objectStore(docStoreName);
        const docReq = docStore.get(key);
        docReq.onsuccess = () => {
          if (Array.isArray(docReq.result) && docReq.result.length > 0) {
            documentsResult = docReq.result as TDoc[];
          }
        };
      }
    });
  } finally {
    if (db) {
      db.close();
    }
  }
}

/**
 * Deletes an index snapshot from IndexedDB.
 */
export async function deleteIndexFromIDB(options?: IDBStorageOptions): Promise<boolean> {
  const snapshotStoreName = options?.snapshotStoreName ?? DEFAULT_SNAPSHOT_STORE_NAME;
  const docStoreName = options?.docStoreName ?? DEFAULT_DOCUMENT_STORE_NAME;
  const key = options?.key ?? DEFAULT_SNAPSHOT_KEY;

  let db: IDBDatabase | null = null;
  try {
    db = await openSearchDatabase(options);
    return await new Promise<boolean>((resolve, reject) => {
      const storeNames = db!.objectStoreNames.contains(docStoreName)
        ? [snapshotStoreName, docStoreName]
        : [snapshotStoreName];

      const tx = db!.transaction(storeNames, 'readwrite');
      tx.onerror = () => {
        reject(tx.error ?? new Error('[webgpu-search] IDB transaction failed during deleteIndexFromIDB.'));
      };
      tx.onabort = () => {
        reject(tx.error ?? new Error('[webgpu-search] IDB transaction aborted during deleteIndexFromIDB.'));
      };
      tx.oncomplete = () => {
        resolve(true);
      };

      const snapStore = tx.objectStore(snapshotStoreName);
      snapStore.delete(key);

      if (db!.objectStoreNames.contains(docStoreName)) {
        const docStore = tx.objectStore(docStoreName);
        docStore.delete(key);
      }
    });
  } finally {
    if (db) {
      db.close();
    }
  }
}

/**
 * High-level helper: loads snapshot from IndexedDB and restores a ready DocumentIndex.
 */
export async function restoreIndexFromIDB<TDoc = Record<string, unknown>>(
  options?: LoadIDBOptions & RestoreDocumentIndexOptions<TDoc>
): Promise<DocumentIndex<TDoc> | null> {
  const loaded = await loadIndexFromIDB<TDoc>(options);
  if (!loaded) {
    return null;
  }

  const restoreOptions: RestoreDocumentIndexOptions<TDoc> = {
    ...options,
    documents: options?.documents ?? loaded.documents
  };

  return restoreSnapshot<TDoc>(loaded.snapshot, restoreOptions);
}
