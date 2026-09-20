import type {
  AddOptions,
  DocumentId,
  DocumentIndexOptions,
  DocumentIndexStats,
  DocumentSearchOptions,
  DocumentSearchResponse,
  MutationBatch,
  MutationResult,
  WorkerClientOptions
} from '../types';
import {
  deserializeError,
  type WorkerMessageType,
  type WorkerRequest,
  type WorkerResponse,
  type WorkerSearchPayload
} from './protocol';
import { abortError, throwIfAborted } from '../runtime-guards';

interface InternalFieldDef<TDoc> {
  name: string;
  weight: number;
  getter: (doc: TDoc) => any;
}

interface PendingQuery<TDoc> {
  queryId: number;
  resolve: (res: DocumentSearchResponse<TDoc>) => void;
  reject: (err: any) => void;
  signalCleanup?: () => void;
  filter?: (doc: TDoc) => boolean;
  limit?: number;
}

export const INTERNAL_WORKER_ID_KEY = '__wgpu_id__';

/**
 * First-party asynchronous client managing an off-thread search worker.
 *
 * Implements monotonic query sequencing with immediate AbortError rejection,
 * transferable buffer safety, string-isolated enrichment, and error class rehydration.
 */
export class SearchWorkerClient<TDoc = Record<string, unknown>> {
  readonly options?: WorkerClientOptions;
  private worker: Worker | null = null;
  private ownsWorker: boolean = false;
  private isDestroyed: boolean = false;
  private nextRequestId: number = 1;
  private nextQueryId: number = 1;
  private readonly stringIsolated: boolean;
  private readonly docMap: Map<DocumentId, TDoc> = new Map();
  private getId: (doc: TDoc) => DocumentId = (doc: any) => doc?.id;
  private fieldDefinitions: InternalFieldDef<TDoc>[] = [];
  private messageListener: ((event: MessageEvent) => void) | null = null;
  private errorListener: ((err: any) => void) | null = null;

  private readonly pendingRequests = new Map<number, {
    resolve: (val: any) => void;
    reject: (err: any) => void;
  }>();

  private readonly pendingQueries = new Map<number, PendingQuery<TDoc>>();

  constructor(options?: WorkerClientOptions) {
    this.options = options;
    this.stringIsolated = options?.stringIsolated ?? true;
  }

  private getWorker(): Worker {
    if (this.isDestroyed) {
      throw new Error('[webgpu-search] SearchWorkerClient has been destroyed.');
    }
    if (!this.worker) {
      if (this.options?.worker) {
        this.worker = typeof this.options.worker === 'function' ? this.options.worker() : this.options.worker;
        this.ownsWorker = false;
      } else {
        this.worker = this.createDefaultWorker();
        this.ownsWorker = true;
      }
      this.attachWorkerListeners(this.worker);
    }
    return this.worker;
  }

  private createDefaultWorker(): Worker {
    if (typeof Worker === 'undefined') {
      throw new Error(
        '[webgpu-search] Worker environment not detected. In Node.js or SSR environments, pass a custom worker or worker factory via options.worker.'
      );
    }
    try {
      const isTs = typeof import.meta?.url === 'string' && import.meta.url.endsWith('.ts');
      const workerUrl = new URL(
        isTs ? './search-worker.ts' : './worker.js',
        import.meta?.url || 'http://localhost/'
      );
      return new Worker(workerUrl, { type: 'module' });
    } catch (err) {
      throw new Error(
        `[webgpu-search] Failed to construct default Worker: ${String(err)}. Please supply options.worker.`
      );
    }
  }

  private attachWorkerListeners(w: Worker): void {
    const handleMessage = (event: MessageEvent) => {
      const resp = event?.data as WorkerResponse;
      if (!resp || typeof resp.id !== 'number') return;

      // 1. Check pending search queries
      const pendingQuery = this.pendingQueries.get(resp.id);
      if (pendingQuery) {
        this.pendingQueries.delete(resp.id);
        pendingQuery.signalCleanup?.();

        if (!resp.success) {
          pendingQuery.reject(deserializeError(resp.error!));
        } else {
          try {
            const searchResp = resp.result as DocumentSearchResponse<TDoc>;
            if (searchResp && Array.isArray(searchResp.results)) {
              // String-isolated enrichment: re-attach original doc
              for (let i = 0; i < searchResp.results.length; i++) {
                const item = searchResp.results[i];
                if (item.doc === undefined || item.doc === null) {
                  const doc = this.docMap.get(item.id);
                  if (doc !== undefined) {
                    item.doc = doc;
                  }
                }
              }

              // Apply predicate filter if configured
              if (pendingQuery.filter) {
                searchResp.results = searchResp.results.filter((item) => pendingQuery.filter!(item.doc));
                if (pendingQuery.limit && searchResp.results.length > pendingQuery.limit) {
                  searchResp.results = searchResp.results.slice(0, pendingQuery.limit);
                }
                searchResp.totalMatches = searchResp.results.length;
              }
            }
            pendingQuery.resolve(searchResp);
          } catch (err) {
            pendingQuery.reject(err);
          }
        }
        return;
      }

      // 2. Check pending generic requests
      const pendingReq = this.pendingRequests.get(resp.id);
      if (pendingReq) {
        this.pendingRequests.delete(resp.id);
        if (!resp.success) {
          pendingReq.reject(deserializeError(resp.error!));
        } else {
          pendingReq.resolve(resp.result);
        }
      }
    };

    const handleError = (err: any) => {
      const errorObj = new Error(err?.message || 'Search worker encountered a fatal error.');
      this.isDestroyed = true;
      for (const pending of this.pendingQueries.values()) {
        pending.signalCleanup?.();
        pending.reject(errorObj);
      }
      this.pendingQueries.clear();

      for (const pending of this.pendingRequests.values()) {
        pending.reject(errorObj);
      }
      this.pendingRequests.clear();

      if (this.worker) {
        this.removeWorkerListeners(this.worker);
        if (this.ownsWorker && typeof this.worker.terminate === 'function') {
          this.worker.terminate();
        }
        this.worker = null;
      }
    };

    this.messageListener = handleMessage;
    this.errorListener = handleError;

    if (typeof w.addEventListener === 'function') {
      w.addEventListener('message', handleMessage);
      w.addEventListener('error', handleError);
    } else {
      w.onmessage = handleMessage;
      w.onerror = handleError;
    }
  }

  private removeWorkerListeners(w: Worker): void {
    if (this.messageListener) {
      if (typeof w.removeEventListener === 'function') {
        w.removeEventListener('message', this.messageListener);
      } else {
        w.onmessage = null;
      }
      this.messageListener = null;
    }
    if (this.errorListener) {
      if (typeof w.removeEventListener === 'function') {
        w.removeEventListener('error', this.errorListener);
      } else {
        w.onerror = null;
      }
      this.errorListener = null;
    }
  }

  private sendRequest<T>(type: WorkerMessageType, payload?: any, transfer?: Transferable[]): Promise<T> {
    if (this.isDestroyed) {
      return Promise.reject(new Error('[webgpu-search] SearchWorkerClient has been destroyed.'));
    }
    const worker = this.getWorker();
    const id = this.nextRequestId++;
    return new Promise<T>((resolve, reject) => {
      this.pendingRequests.set(id, { resolve, reject });
      const req: WorkerRequest = { id, type, payload };
      if (transfer && transfer.length > 0) {
        worker.postMessage(req, transfer);
      } else {
        worker.postMessage(req);
      }
    });
  }

  private extractSerializableDoc(doc: TDoc, id: DocumentId): Record<string, unknown> {
    const out: Record<string, unknown> = { [INTERNAL_WORKER_ID_KEY]: id };
    for (let i = 0; i < this.fieldDefinitions.length; i++) {
      const f = this.fieldDefinitions[i];
      const val = f.getter(doc);
      out[f.name] = val !== undefined && val !== null ? val : '';
    }
    return out;
  }

  async init(
    optionsOrRecords?: DocumentIndexOptions<TDoc> | TDoc[],
    maybeOptions?: DocumentIndexOptions<TDoc>
  ): Promise<void> {
    if (this.isDestroyed) {
      throw new Error('[webgpu-search] SearchWorkerClient has been destroyed.');
    }

    let options: DocumentIndexOptions<TDoc>;
    let records: TDoc[] = [];

    if (Array.isArray(optionsOrRecords)) {
      records = optionsOrRecords;
      if (!maybeOptions) {
        throw new TypeError('[webgpu-search] init(records, options) requires options.');
      }
      options = maybeOptions;
    } else if (optionsOrRecords && typeof optionsOrRecords === 'object') {
      options = optionsOrRecords;
      if (Array.isArray((options as any).initialRecords)) {
        records = (options as any).initialRecords;
      }
    } else {
      throw new TypeError('[webgpu-search] init requires DocumentIndexOptions.');
    }

    if (options.idField !== undefined && typeof options.idField !== 'string' && typeof options.idField !== 'function') {
      throw new TypeError('[webgpu-search] idField must be a string property name or function.');
    }

    if (typeof options.idField === 'function') {
      this.getId = options.idField;
    } else if (typeof options.idField === 'string') {
      const prop = options.idField;
      this.getId = (doc: any) => doc[prop];
    } else {
      this.getId = (doc: any) => doc.id;
    }

    if (!Array.isArray(options.fields) || options.fields.length === 0) {
      throw new TypeError('[webgpu-search] DocumentIndex expects options.fields to be a non-empty array.');
    }

    this.fieldDefinitions = options.fields.map((f) => {
      if (typeof f === 'string') {
        if (!f) {
          throw new TypeError('[webgpu-search] Field name cannot be an empty string.');
        }
        return {
          name: f,
          weight: 1.0,
          getter: (doc: any) => doc[f]
        };
      }
      if (!f || typeof f !== 'object' || typeof f.name !== 'string' || !f.name) {
        throw new TypeError('[webgpu-search] Field definition requires non-empty name string.');
      }
      const weight = f.weight ?? 1.0;
      if (typeof weight !== 'number' || !Number.isFinite(weight) || weight <= 0) {
        throw new RangeError(`[webgpu-search] Field weight must be a positive finite number, got ${weight}.`);
      }
      return {
        name: f.name,
        weight,
        getter: f.getter ? f.getter : (doc: any) => doc[f.name]
      };
    });

    if (!this.stringIsolated) {
      if (typeof options.idField === 'function') {
        throw new TypeError('[webgpu-search] stringIsolated: false does not support function idField across worker boundary.');
      }
      for (const f of options.fields) {
        if (typeof f === 'object' && typeof f.getter === 'function') {
          throw new TypeError('[webgpu-search] stringIsolated: false does not support custom getter functions across worker boundary.');
        }
      }
    }

    const nextDocMap = new Map<DocumentId, TDoc>();
    const serializableRecords: Record<string, unknown>[] = [];
    for (let i = 0; i < records.length; i++) {
      const doc = records[i];
      const id = this.getId(doc);
      nextDocMap.set(id, doc);
      if (this.stringIsolated) {
        serializableRecords.push(this.extractSerializableDoc(doc, id));
      } else {
        serializableRecords.push(doc as any);
      }
    }

    const workerFields = this.fieldDefinitions.map((f) => ({
      name: f.name,
      weight: f.weight
    }));

    const workerOptions = {
      ...options,
      idField: this.stringIsolated ? INTERNAL_WORKER_ID_KEY : (options.idField ?? 'id'),
      fields: workerFields
    };

    await this.sendRequest('INIT', {
      options: workerOptions,
      records: serializableRecords
    });

    this.docMap.clear();
    for (const [id, doc] of nextDocMap) {
      this.docMap.set(id, doc);
    }
  }

  async search(
    query: string,
    options?: DocumentSearchOptions<TDoc>
  ): Promise<DocumentSearchResponse<TDoc>> {
    if (this.isDestroyed) {
      throw new Error('[webgpu-search] SearchWorkerClient has been destroyed.');
    }

    throwIfAborted(options?.signal);

    const worker = this.getWorker();
    const reqId = this.nextRequestId++;
    const queryId = this.nextQueryId++;

    // Monotonic query sequencing: immediately reject all prior pending queries with AbortError
    for (const [id, pending] of this.pendingQueries.entries()) {
      if (pending.queryId < queryId) {
        pending.signalCleanup?.();
        pending.reject(abortError());
        this.pendingQueries.delete(id);
        worker.postMessage({
          id: this.nextRequestId++,
          type: 'ABORT',
          payload: { queryId: pending.queryId }
        } satisfies WorkerRequest);
      }
    }

    const { filter, signal, limit, maxResults, ...restOptions } = options || {};
    const requestedLimit = limit ?? maxResults ?? 50;

    // Avoid candidate starvation when predicate filter is applied on main thread
    const workerOptions = {
      ...restOptions,
      limit: filter ? ((options as any)?.candidateCapacity ?? 8192) : requestedLimit
    };

    return new Promise<DocumentSearchResponse<TDoc>>((resolve, reject) => {
      let signalCleanup: (() => void) | undefined;

      if (signal) {
        const onAbort = () => {
          const p = this.pendingQueries.get(reqId);
          if (p) {
            this.pendingQueries.delete(reqId);
            signalCleanup?.();
            p.reject(abortError());
            worker.postMessage({
              id: this.nextRequestId++,
              type: 'ABORT',
              payload: { queryId }
            } satisfies WorkerRequest);
          }
        };
        signal.addEventListener('abort', onAbort, { once: true });
        signalCleanup = () => signal.removeEventListener('abort', onAbort);
      }

      this.pendingQueries.set(reqId, {
        queryId,
        resolve,
        reject,
        signalCleanup,
        filter,
        limit: requestedLimit
      });

      const req: WorkerRequest = {
        id: reqId,
        type: 'SEARCH',
        payload: {
          queryId,
          query,
          options: workerOptions,
          stringIsolated: this.stringIsolated
        } satisfies WorkerSearchPayload<TDoc>
      };

      worker.postMessage(req);
    });
  }

  async applyBatch(batch: MutationBatch<TDoc>, options?: AddOptions): Promise<MutationResult> {
    if (this.isDestroyed) {
      throw new Error('[webgpu-search] SearchWorkerClient has been destroyed.');
    }
    if (!batch || typeof batch !== 'object') {
      throw new TypeError('[webgpu-search] applyBatch expects batch object.');
    }

    const serializableBatch: MutationBatch<any> = {};
    const pendingAdds = new Map<DocumentId, TDoc>();
    const pendingUpdates = new Map<DocumentId, TDoc>();
    const pendingRemoves: DocumentId[] = [];

    if (batch.remove !== undefined) {
      if (!Array.isArray(batch.remove)) {
        throw new TypeError('[webgpu-search] batch.remove must be an array of document IDs.');
      }
      serializableBatch.remove = batch.remove;
      for (const id of batch.remove) {
        pendingRemoves.push(id);
      }
    }

    if (batch.update !== undefined) {
      if (!Array.isArray(batch.update)) {
        throw new TypeError('[webgpu-search] batch.update must be an array of documents.');
      }
      serializableBatch.update = batch.update.map((doc) => {
        const id = this.getId(doc);
        pendingUpdates.set(id, doc);
        return this.stringIsolated ? this.extractSerializableDoc(doc, id) : (doc as any);
      });
    }

    if (batch.add !== undefined) {
      if (!Array.isArray(batch.add)) {
        throw new TypeError('[webgpu-search] batch.add must be an array of documents.');
      }
      serializableBatch.add = batch.add.map((doc) => {
        const id = this.getId(doc);
        pendingAdds.set(id, doc);
        return this.stringIsolated ? this.extractSerializableDoc(doc, id) : (doc as any);
      });
    }

    const res = await this.sendRequest<MutationResult>('MUTATE', {
      batch: serializableBatch,
      options
    });

    for (const id of pendingRemoves) {
      this.docMap.delete(id);
    }
    for (const [id, doc] of pendingUpdates) {
      this.docMap.set(id, doc);
    }
    for (const [id, doc] of pendingAdds) {
      this.docMap.set(id, doc);
    }

    return res;
  }

  async add(docs: TDoc | TDoc[], options?: AddOptions): Promise<MutationResult> {
    const list = Array.isArray(docs) ? docs : [docs];
    return this.applyBatch({ add: list }, options);
  }

  async update(docs: TDoc | TDoc[]): Promise<MutationResult> {
    const list = Array.isArray(docs) ? docs : [docs];
    return this.applyBatch({ update: list });
  }

  async remove(ids: DocumentId | DocumentId[]): Promise<MutationResult> {
    const list = Array.isArray(ids) ? ids : [ids];
    return this.applyBatch({ remove: list });
  }

  async serialize(): Promise<ArrayBuffer> {
    if (this.isDestroyed) {
      throw new Error('[webgpu-search] SearchWorkerClient has been destroyed.');
    }
    return this.sendRequest<ArrayBuffer>('SERIALIZE');
  }

  async restore(buffer: ArrayBuffer, options?: { transfer?: boolean }): Promise<void> {
    if (this.isDestroyed) {
      throw new Error('[webgpu-search] SearchWorkerClient has been destroyed.');
    }
    if (!buffer || typeof (buffer as any).byteLength !== 'number') {
      throw new TypeError('[webgpu-search] restore expects an ArrayBuffer.');
    }
    const shouldTransfer = options?.transfer === true;
    const toSend = shouldTransfer ? buffer : buffer.slice(0);
    await this.sendRequest('RESTORE', { buffer: toSend, options }, [toSend]);
  }

  async getStats(): Promise<DocumentIndexStats> {
    if (this.isDestroyed) {
      throw new Error('[webgpu-search] SearchWorkerClient has been destroyed.');
    }
    return this.sendRequest<DocumentIndexStats>('STATS');
  }

  async destroy(): Promise<void> {
    if (this.isDestroyed) return;
    this.isDestroyed = true;

    for (const pending of this.pendingQueries.values()) {
      pending.signalCleanup?.();
      pending.reject(abortError());
    }
    this.pendingQueries.clear();

    for (const pending of this.pendingRequests.values()) {
      pending.reject(new Error('[webgpu-search] SearchWorkerClient has been destroyed.'));
    }
    this.pendingRequests.clear();


    if (this.worker) {
      try {
        const req: WorkerRequest = {
          id: this.nextRequestId++,
          type: 'DESTROY'
        };
        this.worker.postMessage(req);
      } catch {}

      this.removeWorkerListeners(this.worker);

      if (this.ownsWorker && typeof this.worker.terminate === 'function') {
        this.worker.terminate();
      }
      this.worker = null;
    }
    this.docMap.clear();
  }

  async [Symbol.asyncDispose](): Promise<void> {
    await this.destroy();
  }
}
