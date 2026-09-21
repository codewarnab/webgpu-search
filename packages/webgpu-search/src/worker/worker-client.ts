import type {
  AddOptions,
  DocumentId,
  DocumentIndexOptions,
  DocumentIndexStats,
  DocumentSearchOptions,
  DocumentSearchResponse,
  MutationBatch,
  MutationResult,
  RestoreDocumentIndexOptions,
  SerializeDocumentIndexOptions,
  WorkerClientOptions,
  FilterExpression
} from '../types';
import {
  deserializeError,
  type WorkerMessageType,
  type WorkerRequest,
  type WorkerResponse,
  type WorkerSearchPayload
} from './protocol';
import { abortError, throwIfAborted } from '../runtime-guards';
import { deserializeDocumentSnapshotHeader } from '../persistence';
import { SERIALIZED_DOC_HEADER_BYTES } from '../text-profile';
import { IncompatibleHookError } from '../errors';
import { hasAnyHook, normalizeSearchExtensionHooks } from '../extensions';

/**
 * Fail-closed worker extensions guard: empty `{}` is a no-op (consistent
 * with `normalizeSearchExtensionHooks`), any real hook rejects with
 * `IncompatibleHookError`. Malformed shapes throw `TypeError` via normalize.
 */
function assertNoWorkerExtensions(
  extensions: unknown,
  method: 'init' | 'search' | 'restore'
): void {
  if (extensions === undefined) return;
  const normalized = normalizeSearchExtensionHooks(
    extensions as Parameters<typeof normalizeSearchExtensionHooks>[0]
  );
  if (hasAnyHook(normalized)) {
    throw new IncompatibleHookError(
      'extensions',
      `SearchExtensionHooks contain function closures which cannot be cloned across Web Worker boundaries (${method} rejected fail-closed).`
    );
  }
}

interface InternalFieldDef<TDoc> {
  name: string;
  weight: number;
  getter: (doc: TDoc) => string | string[] | undefined | null;
}

interface PendingQuery<TDoc> {
  queryId: number;
  resolve: (res: DocumentSearchResponse<TDoc>) => void;
  reject: (err: any) => void;
  signalCleanup?: () => void;
  filter?: ((doc: TDoc) => boolean) | FilterExpression;
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
  private filterDefinitions: Array<{ name: string; getter: (doc: any) => any }> = [];
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
              // String-isolated enrichment: re-attach original docs.
              for (let i = 0; i < searchResp.results.length; i++) {
                const item = searchResp.results[i];
                if (item.doc === undefined || item.doc === null) {
                  const doc = this.docMap.get(item.id);
                  if (doc !== undefined) {
                    item.doc = doc;
                  }
                }
              }
              if (Array.isArray(searchResp.suggestions)) {
                for (let i = 0; i < searchResp.suggestions.length; i++) {
                  const s = searchResp.suggestions[i];
                  if (s.doc === undefined || s.doc === null) {
                    const key = s.docId ?? (s as unknown as { id?: DocumentId }).id;
                    if (key !== undefined) {
                      const doc = this.docMap.get(key);
                      if (doc !== undefined) {
                        s.doc = doc;
                      }
                    }
                  }
                }
              }

              // Apply predicate filter if configured as a function.
              // Facets computed worker-side are over the unfiltered set and
              // would go stale, so drop them fail-closed (local path applies
              // the predicate conjunctively in buildFacetResults).
              // Suggestions stay index-wide by design (filters never narrow
              // suggestions), so they are kept as returned.
              if (typeof pendingQuery.filter === 'function') {
                const predicate = pendingQuery.filter;
                searchResp.results = searchResp.results.filter((item) => predicate(item.doc));
                if (pendingQuery.limit && searchResp.results.length > pendingQuery.limit) {
                  searchResp.results = searchResp.results.slice(0, pendingQuery.limit);
                }
                searchResp.totalMatches = searchResp.results.length;
                if ('facets' in searchResp) {
                  delete (searchResp as { facets?: unknown }).facets;
                }
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
      try {
        const req: WorkerRequest = { id, type, payload };
        if (transfer && transfer.length > 0) {
          worker.postMessage(req, transfer);
        } else {
          worker.postMessage(req);
        }
      } catch (err) {
        this.pendingRequests.delete(id);
        reject(err);
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
    for (let i = 0; i < this.filterDefinitions.length; i++) {
      const ff = this.filterDefinitions[i];
      const val = ff.getter(doc);
      if (val !== undefined) {
        out[ff.name] = val;
      }
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

    const filterDefs: Array<{ name: string; getter: (doc: any) => any }> = [];
    if (options.filterFields) {
      for (let i = 0; i < options.filterFields.length; i++) {
        const ff = options.filterFields[i];
        if (typeof ff === 'string') {
          filterDefs.push({ name: ff, getter: (doc: any) => doc[ff] });
        } else if (ff && typeof ff === 'object' && ff.name) {
          filterDefs.push({
            name: ff.name,
            getter: ff.getter ?? ((doc: any) => doc[ff.name])
          });
        }
      }
    }
    this.filterDefinitions = filterDefs;

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

    if (options.extensions) {
      assertNoWorkerExtensions(options.extensions, 'init');
    }

    const workerFields = this.fieldDefinitions.map((f) => ({
      name: f.name,
      weight: f.weight
    }));

    let workerFilterFields = options.filterFields;
    if (workerFilterFields) {
      workerFilterFields = workerFilterFields.map((f) => {
        if (typeof f === 'object' && f !== null) {
          const { getter, ...serializableField } = f;
          return serializableField;
        }
        return f;
      });
    }

    const { extensions, filterFields, ...restInitOptions } = options;
    const workerOptions = {
      ...restInitOptions,
      ...(workerFilterFields ? { filterFields: workerFilterFields } : {}),
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

    if (options?.extensions) {
      assertNoWorkerExtensions(options.extensions, 'search');
    }

    const { filter, signal, limit, maxResults, extensions: _omitExtensions, ...restOptions } = options || {};

    if (filter !== undefined && typeof filter !== 'function') {
      if (typeof filter !== 'object' || filter === null) {
        throw new TypeError('[webgpu-search] options.filter must be a function or FilterExpression.');
      }
    }

    const requestedLimit = limit ?? maxResults ?? 50;

    let workerBudget = restOptions.budget;
    if (workerBudget?.abortSignal) {
      const { abortSignal, ...remainingBudget } = workerBudget;
      workerBudget = remainingBudget;
    }

    const isPredicate = typeof filter === 'function';
    // Avoid candidate starvation when predicate filter is applied on main thread.
    // Facets are withheld from the worker when a predicate is present: the
    // worker cannot apply the closure, so any worker-computed facets would
    // reflect the unfiltered distribution (stale). Caller gets results-only.
    const { facets: _omitFacets, faceting: _omitFaceting, ...nonFacetRest } = restOptions as Record<string, unknown>;
    const workerOptions = {
      ...(isPredicate ? nonFacetRest : restOptions),
      ...(workerBudget ? { budget: workerBudget } : {}),
      ...(filter && !isPredicate ? { filter } : {}),
      limit: isPredicate ? ((options as any)?.candidateCapacity ?? 8192) : requestedLimit
    };

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
        filter: isPredicate ? filter : undefined,
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

      try {
        worker.postMessage(req);
      } catch (err) {
        this.pendingQueries.delete(reqId);
        signalCleanup?.();
        reject(err);
      }
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

  getRecords(): TDoc[] {
    return Array.from(this.docMap.values());
  }

  async serialize(options?: SerializeDocumentIndexOptions): Promise<ArrayBuffer> {
    if (this.isDestroyed) {
      throw new Error('[webgpu-search] SearchWorkerClient has been destroyed.');
    }
    return this.sendRequest<ArrayBuffer>('SERIALIZE', { options });
  }

  async restore(buffer: ArrayBuffer, options?: RestoreDocumentIndexOptions<TDoc>): Promise<void> {
    if (this.isDestroyed) {
      throw new Error('[webgpu-search] SearchWorkerClient has been destroyed.');
    }
    if (!buffer || typeof (buffer as any).byteLength !== 'number') {
      throw new TypeError('[webgpu-search] restore expects an ArrayBuffer.');
    }
    if (options?.options?.extensions) {
      assertNoWorkerExtensions(options.options.extensions, 'restore');
    }

    let stagedFieldDefs = this.fieldDefinitions;
    let stagedFilterDefs = this.filterDefinitions;
    let stagedGetId = this.getId;
    const stagedDocMap = new Map<DocumentId, TDoc>();

    try {
      const header = deserializeDocumentSnapshotHeader(buffer);
      const userFields = options?.options?.fields;
      const userFieldMap = new Map<string, any>();
      if (Array.isArray(userFields)) {
        for (const uf of userFields) {
          if (typeof uf === 'string') {
            userFieldMap.set(uf, uf);
          } else if (uf && typeof uf === 'object' && typeof uf.name === 'string') {
            userFieldMap.set(uf.name, uf);
          }
        }
      }

      if (buffer.byteLength >= SERIALIZED_DOC_HEADER_BYTES + header.schemaByteLength) {
        const schemaBytes = new Uint8Array(buffer, SERIALIZED_DOC_HEADER_BYTES, header.schemaByteLength);
        const schemaStr = new TextDecoder().decode(schemaBytes);
        const schema = JSON.parse(schemaStr);
        if (schema && Array.isArray(schema.fields)) {
          stagedFieldDefs = schema.fields.map((f: any) => {
            const uf = userFieldMap.get(f.name);
            const getter = typeof uf === 'object' && uf !== null && typeof uf.getter === 'function'
              ? uf.getter
              : (doc: any) => doc[f.name];
            return {
              name: f.name,
              weight: f.weight ?? 1.0,
              getter
            };
          });
        }
        if (schema && Array.isArray(schema.filterFields)) {
          const userFilterFields = options?.options?.filterFields;
          const userFilterMap = new Map<string, any>();
          if (Array.isArray(userFilterFields)) {
            for (const uff of userFilterFields) {
              if (typeof uff === 'string') {
                userFilterMap.set(uff, uff);
              } else if (uff && typeof uff === 'object' && typeof uff.name === 'string') {
                userFilterMap.set(uff.name, uff);
              }
            }
          }
          stagedFilterDefs = schema.filterFields.map((ff: any) => {
            const uf = userFilterMap.get(ff.name);
            const getter = typeof uf === 'object' && uf !== null && typeof uf.getter === 'function'
              ? uf.getter
              : (doc: any) => doc[ff.name];
            return {
              name: ff.name,
              getter
            };
          });
        }
        if (typeof options?.options?.idField === 'function') {
          stagedGetId = options.options.idField;
        } else if (typeof options?.options?.idField === 'string') {
          const prop = options.options.idField;
          stagedGetId = (doc: any) => doc[prop] ?? doc[INTERNAL_WORKER_ID_KEY] ?? doc.id;
        } else if (schema.idField && typeof schema.idField === 'string') {
          const prop = schema.idField;
          stagedGetId = (doc: any) => doc[prop] ?? doc[INTERNAL_WORKER_ID_KEY] ?? doc.id;
        } else {
          stagedGetId = (doc: any) => doc[INTERNAL_WORKER_ID_KEY] ?? doc.id;
        }
      }

      // Repopulate stagedDocMap if documents are provided or embedded
      if (options?.documents && Array.isArray(options.documents)) {
        for (const doc of options.documents) {
          const id = stagedGetId(doc);
          stagedDocMap.set(id, doc);
        }
      } else if (header.docsByteLength > 0) {
        const docsOffset =
          SERIALIZED_DOC_HEADER_BYTES +
          header.schemaByteLength +
          header.tokenCount * 4 +
          (header.rowCount + 1) * 4;
        const docsBytes = new Uint8Array(buffer, docsOffset, header.docsByteLength);
        const docsStr = new TextDecoder().decode(docsBytes);
        const docs = JSON.parse(docsStr);
        if (Array.isArray(docs)) {
          for (const doc of docs) {
            const id = stagedGetId(doc);
            if (doc && doc[INTERNAL_WORKER_ID_KEY] !== undefined && doc.id === undefined) {
              doc.id = doc[INTERNAL_WORKER_ID_KEY];
            }
            stagedDocMap.set(id, doc);
          }
        }
      }
    } catch {
      // Allow worker to handle fail-closed validation and throw IncompatibleIndexError
    }

    // Sanitize options to avoid DataCloneError over postMessage.
    // Extensions are rejected fail-closed above; strip them defensively so
    // an empty `{}` no-op never crosses the boundary.
    const sanitizedOptions: RestoreDocumentIndexOptions<TDoc> | undefined = options ? {
      ...options,
      device: undefined,
      options: options.options ? {
        ...options.options,
        device: undefined,
        extensions: undefined,
        fields: options.options.fields?.map((f) =>
          typeof f === 'string' ? f : { name: f.name, weight: f.weight }
        ),
        filterFields: options.options.filterFields?.map((f) => {
          if (typeof f === 'object' && f !== null) {
            const { getter, ...serializableField } = f;
            return serializableField;
          }
          return f;
        }),
        idField: typeof options.options.idField === 'string' ? options.options.idField : undefined
      } : undefined
    } : undefined;

    const shouldTransfer = options?.transfer === true;
    const toSend = shouldTransfer ? buffer : buffer.slice(0);
    await this.sendRequest('RESTORE', { buffer: toSend, options: sanitizedOptions }, [toSend]);

    // Commit only after successful restore response from worker
    this.fieldDefinitions = stagedFieldDefs;
    this.filterDefinitions = stagedFilterDefs;
    this.getId = stagedGetId;
    this.docMap.clear();
    for (const [k, v] of stagedDocMap) {
      this.docMap.set(k, v);
    }
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
