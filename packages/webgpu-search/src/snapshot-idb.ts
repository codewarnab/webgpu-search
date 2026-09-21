import { restoreSnapshot } from './snapshot-codec';
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

/**
 * Opens the IndexedDB database, provisioning required object stores cleanly.
 */
export function openSearchDatabase(options?: IDBStorageOptions): Promise<IDBDatabase> {
  const idb = resolveIDBFactory(options);
  const dbName = options?.dbName ?? DEFAULT_IDB_DATABASE_NAME;
  const snapshotStoreName = options?.snapshotStoreName ?? DEFAULT_SNAPSHOT_STORE_NAME;
  const docStoreName = options?.docStoreName ?? DEFAULT_DOCUMENT_STORE_NAME;

  return new Promise<IDBDatabase>((resolve, reject) => {
    // Open with version 1 (or allow auto-increment)
    const req = idb.open(dbName, 1);

    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(snapshotStoreName)) {
        db.createObjectStore(snapshotStoreName);
      }
      if (!db.objectStoreNames.contains(docStoreName)) {
        db.createObjectStore(docStoreName);
      }
    };

    req.onsuccess = () => {
      resolve(req.result);
    };

    req.onerror = () => {
      reject(req.error ?? new Error(`[webgpu-search] Failed to open IndexedDB database "${dbName}".`));
    };

    req.onblocked = () => {
      console.warn(`[webgpu-search] IndexedDB database "${dbName}" open request blocked by other open tabs/connections.`);
    };
  });
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
  indexOrBuffer: DocumentIndex<any> | SearchWorkerClient<any> | ArrayBuffer,
  options?: SaveIDBOptions
): Promise<void> {
  if (!indexOrBuffer) {
    throw new TypeError('[webgpu-search] saveIndexToIDB requires a DocumentIndex, SearchWorkerClient, or ArrayBuffer.');
  }

  const decoupled = options?.decoupled === true;
  let snapshot: ArrayBuffer;
  let docsToStore: any[] | undefined = options?.documents;

  // 1. Serialization phase (OUTSIDE transaction to avoid auto-commit)
  const isBuffer =
    indexOrBuffer instanceof ArrayBuffer ||
    (typeof ArrayBuffer !== 'undefined' &&
      typeof (indexOrBuffer as any)?.byteLength === 'number' &&
      typeof (indexOrBuffer as any)?.slice === 'function');

  if (isBuffer) {
    snapshot = indexOrBuffer as ArrayBuffer;
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
        snapshotResult = (snapReq.result as ArrayBuffer) ?? null;
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
