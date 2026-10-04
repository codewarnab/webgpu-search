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
  type WorkerRestoreResult,
  type WorkerSearchPayload
} from './protocol';
import { abortError, throwIfAborted } from '../guard';
import { MAX_SNAPSHOT_BYTES, toSnapshotArrayBuffer } from '../snapshot-codec';
import { IncompatibleIndexError } from '../text-profile';
import { IncompatibleHookError, IncompatibleOptionError } from '../errors';
import { hasAnyHook, normalizeSearchHooks } from '../hooks';

/**
 * Fail-closed worker hooks guard: empty `{}` is a no-op (consistent
 * with `normalizeSearchHooks`), any real hook rejects with
 * `IncompatibleHookError`. Malformed shapes throw `TypeError` via normalize.
 */
function assertNoWorkerExtensions(
  hooks: unknown,
  method: 'init' | 'search' | 'restore'
): void {
  if (hooks === undefined) return;
  const normalized = normalizeSearchHooks(
    hooks as Parameters<typeof normalizeSearchHooks>[0]
  );
  if (hasAnyHook(normalized)) {
    throw new IncompatibleHookError(
      'hooks',
      `SearchHooks contain function closures which cannot be cloned across Web Worker boundaries (${method} rejected fail-closed).`
    );
  }
}

interface InternalFieldDef<TDoc> {
  name: string;
  weight: number;
  getter: (doc: TDoc) => string | string[] | undefined | null;
}

/**
 * Doc-extraction config: how the client derives ids and serializable
 * field/filter values for records it sends to the worker.
 */
interface ClientIndexConfig<TDoc> {
  getId: (doc: TDoc) => DocumentId;
  fieldDefinitions: InternalFieldDef<TDoc>[];
  filterDefinitions: Array<{ name: string; getter: (doc: any) => any }>;
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
  /**
   * Outgoing config: matches the worker state implied by every request
   * posted so far (assuming success). Used to serialize MUTATE payloads.
   */
  private getId: (doc: TDoc) => DocumentId = (doc: any) => doc?.id;
  private fieldDefinitions: InternalFieldDef<TDoc>[] = [];
  private filterDefinitions: Array<{ name: string; getter: (doc: any) => any }> = [];
  /** Config of the most recent INIT/RESTORE the worker confirmed. */
  private committedConfig: ClientIndexConfig<TDoc> = {
    getId: this.getId,
    fieldDefinitions: this.fieldDefinitions,
    filterDefinitions: this.filterDefinitions
  };
  /**
   * Bumped whenever an INIT/RESTORE is posted. The worker processes
   * requests in order and replies in order, so commits happen in completion
   * order; a failed request only rolls the outgoing config back when no
   * later INIT/RESTORE was posted after it.
   */
  private stateGeneration: number = 0;
  /**
   * Settles once every RESTORE posted so far has settled. The restored
   * schema is only known after the worker validates the snapshot, so
   * mutations issued meanwhile wait before serializing their documents.
   */
  private restoreBarrier: Promise<void> | null = null;
  /**
   * Settles once every deferred request has posted. Requests normally post
   * synchronously (worker processes them in arrival order); while a request
   * is deferred (a mutation awaiting a pending RESTORE), later requests queue
   * behind it so the worker still sees them in call order.
   */
  private deferredPosts: Promise<void> | null = null;
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
              // the predicate conjunctively in buildFacetResults). Diagnostics
              // (scannedCandidates / filterSelectivity / hasOverflow / timings)
              // are likewise worker-computed pre-predicate and would go stale,
              // so drop them too — mirroring the facets path. Callers needing
              // predicate + diagnostics should run the local DocumentIndex.
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
                if ('diagnostics' in searchResp) {
                  delete (searchResp as { diagnostics?: unknown }).diagnostics;
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

  private currentConfig(): ClientIndexConfig<TDoc> {
    return {
      getId: this.getId,
      fieldDefinitions: this.fieldDefinitions,
      filterDefinitions: this.filterDefinitions
    };
  }

  private setOutgoingConfig(config: ClientIndexConfig<TDoc>): void {
    this.getId = config.getId;
    this.fieldDefinitions = config.fieldDefinitions;
    this.filterDefinitions = config.filterDefinitions;
  }

  /**
   * Runs `run` (which must post its request synchronously before its first
   * `await`) immediately when nothing is deferred, otherwise after `waitFor`
   * and every earlier deferred request have settled/posted.
   */
  private runInPostOrder<T>(waitFor: Promise<void> | null, run: () => Promise<T>): Promise<T> {
    const prior = this.deferredPosts;
    if (!waitFor && !prior) {
      return run();
    }
    let release!: () => void;
    const posted = new Promise<void>((resolve) => {
      release = resolve;
    });
    this.deferredPosts = posted;
    return Promise.all([waitFor, prior]).then(() => {
      try {
        return run();
      } finally {
        release();
        if (this.deferredPosts === posted) {
          this.deferredPosts = null;
        }
      }
    });
  }

  /** Posts an INIT/RESTORE and commits/rolls back config in completion order. */
  private async sendStateRequest<T>(
    type: 'INIT' | 'RESTORE',
    payload: unknown,
    outgoing: ClientIndexConfig<TDoc> | null,
    transfer?: Transferable[]
  ): Promise<{ result: T; isLatest: boolean }> {
    const generation = ++this.stateGeneration;
    if (outgoing) {
      this.setOutgoingConfig(outgoing);
    }
    try {
      const result = await this.sendRequest<T>(type, payload, transfer);
      return { result, isLatest: this.stateGeneration === generation };
    } catch (err) {
      if (this.stateGeneration === generation) {
        // Worker kept its previous index; mirror it.
        this.setOutgoingConfig(this.committedConfig);
      }
      throw err;
    }
  }

  private extractSerializableDoc(
    doc: TDoc,
    id: DocumentId,
    config: ClientIndexConfig<TDoc> = this.currentConfig()
  ): Record<string, unknown> {
    const out: Record<string, unknown> = { [INTERNAL_WORKER_ID_KEY]: id };
    for (let i = 0; i < config.fieldDefinitions.length; i++) {
      const f = config.fieldDefinitions[i];
      const val = f.getter(doc);
      out[f.name] = val !== undefined && val !== null ? val : '';
    }
    for (let i = 0; i < config.filterDefinitions.length; i++) {
      const ff = config.filterDefinitions[i];
      const val = ff.getter(doc);
      if (val !== undefined) {
        out[ff.name] = val;
      }
    }
    return out;
  }

  init(
    optionsOrRecords?: DocumentIndexOptions<TDoc> | TDoc[],
    maybeOptions?: DocumentIndexOptions<TDoc>
  ): Promise<void> {
    return this.runInPostOrder(null, () => this.initNow(optionsOrRecords, maybeOptions));
  }

  private async initNow(
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

    // Stage config locally; it is only committed once the worker confirms.
    let stagedGetId: (doc: TDoc) => DocumentId;
    if (typeof options.idField === 'function') {
      stagedGetId = options.idField;
    } else if (typeof options.idField === 'string') {
      const prop = options.idField;
      stagedGetId = (doc: any) => doc[prop];
    } else {
      stagedGetId = (doc: any) => doc.id;
    }

    if (!Array.isArray(options.fields) || options.fields.length === 0) {
      throw new TypeError('[webgpu-search] DocumentIndex expects options.fields to be a non-empty array.');
    }

    const stagedFieldDefs: InternalFieldDef<TDoc>[] = options.fields.map((f) => {
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
    const staged: ClientIndexConfig<TDoc> = {
      getId: stagedGetId,
      fieldDefinitions: stagedFieldDefs,
      filterDefinitions: filterDefs
    };

    const nextDocMap = new Map<DocumentId, TDoc>();
    const serializableRecords: Record<string, unknown>[] = [];
    for (let i = 0; i < records.length; i++) {
      const doc = records[i];
      const id = stagedGetId(doc);
      nextDocMap.set(id, doc);
      if (this.stringIsolated) {
        serializableRecords.push(this.extractSerializableDoc(doc, id, staged));
      } else {
        serializableRecords.push(doc as any);
      }
    }

    if ((options as unknown as Record<string, unknown>).extensions !== undefined) {
      throw new IncompatibleHookError(
        'hooks',
        "[webgpu-search] extensions was removed; use hooks."
      );
    }
    if (options.hooks) {
      assertNoWorkerExtensions(options.hooks, 'init');
    }

    const workerFields = stagedFieldDefs.map((f) => ({
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

    const { hooks: _omitHooksInit, filterFields, ...restInitOptions } = options;
    const workerOptions = {
      ...restInitOptions,
      ...(workerFilterFields ? { filterFields: workerFilterFields } : {}),
      idField: this.stringIsolated ? INTERNAL_WORKER_ID_KEY : (options.idField ?? 'id'),
      fields: workerFields
    };

    await this.sendStateRequest('INIT', {
      options: workerOptions,
      records: serializableRecords
    }, staged);

    // Commit in completion order (worker replies in request order).
    this.committedConfig = staged;
    this.docMap.clear();
    for (const [id, doc] of nextDocMap) {
      this.docMap.set(id, doc);
    }
  }

  search(
    query: string,
    options?: DocumentSearchOptions<TDoc>
  ): Promise<DocumentSearchResponse<TDoc>> {
    return this.runInPostOrder(null, () => this.searchNow(query, options));
  }

  private async searchNow(
    query: string,
    options?: DocumentSearchOptions<TDoc>
  ): Promise<DocumentSearchResponse<TDoc>> {
    if (this.isDestroyed) {
      throw new Error('[webgpu-search] SearchWorkerClient has been destroyed.');
    }

    throwIfAborted(options?.signal);

    if ((options as unknown as Record<string, unknown> | undefined)?.extensions !== undefined) {
      throw new IncompatibleHookError(
        'hooks',
        "[webgpu-search] extensions was removed; use hooks."
      );
    }
    if ((options as unknown as Record<string, unknown> | undefined)?.suggest !== undefined) {
      throw new IncompatibleHookError(
        'autocomplete',
        "[webgpu-search] suggest was removed; use autocomplete."
      );
    }
    if ((options as unknown as Record<string, unknown> | undefined)?.cpuAlgorithm !== undefined) {
      throw new IncompatibleOptionError(
        'cpuScorer',
        "[webgpu-search] cpuAlgorithm was removed; use cpuScorer: 'exact' | 'ufuzzy'."
      );
    }
    if (options?.hooks) {
      assertNoWorkerExtensions(options?.hooks, 'search');
    }

    const { filter, signal, limit, maxResults, hooks: _omitHooks, ...restOptions } = (options || {}) as DocumentSearchOptions<TDoc> & { hooks?: unknown };

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

  applyBatch(batch: MutationBatch<TDoc>, options?: AddOptions): Promise<MutationResult> {
    // Field/id definitions are unknown until a pending RESTORE settles, so
    // the mutation (and every request issued after it) waits for it.
    return this.runInPostOrder(this.restoreBarrier, () => this.applyBatchNow(batch, options));
  }

  private async applyBatchNow(batch: MutationBatch<TDoc>, options?: AddOptions): Promise<MutationResult> {
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

  serialize(options?: SerializeDocumentIndexOptions): Promise<ArrayBuffer> {
    return this.runInPostOrder(null, async () => {
      if (this.isDestroyed) {
        throw new Error('[webgpu-search] SearchWorkerClient has been destroyed.');
      }
      return this.sendRequest<ArrayBuffer>('SERIALIZE', { options });
    });
  }

  restore(buffer: ArrayBuffer | ArrayBufferView, options?: RestoreDocumentIndexOptions<TDoc>): Promise<void> {
    return this.runInPostOrder(null, () => this.restoreNow(buffer, options));
  }

  private async restoreNow(buffer: ArrayBuffer | ArrayBufferView, options?: RestoreDocumentIndexOptions<TDoc>): Promise<void> {
    if (this.isDestroyed) {
      throw new Error('[webgpu-search] SearchWorkerClient has been destroyed.');
    }
    if (!buffer || typeof (buffer as any).byteLength !== 'number') {
      throw new TypeError('[webgpu-search] restore expects an ArrayBuffer.');
    }
    if (buffer.byteLength > MAX_SNAPSHOT_BYTES) {
      throw new IncompatibleIndexError(`snapshot-bytes<=${MAX_SNAPSHOT_BYTES}`, buffer.byteLength);
    }
    const snapshotBuffer = toSnapshotArrayBuffer(buffer);
    if (!snapshotBuffer) {
      throw new TypeError('[webgpu-search] restore expects an ArrayBuffer.');
    }
    if ((options?.options as unknown as Record<string, unknown> | undefined)?.extensions !== undefined) {
      throw new IncompatibleHookError(
        'hooks',
        "[webgpu-search] extensions was removed; use hooks."
      );
    }
    if (options?.options?.hooks) {
      assertNoWorkerExtensions(options?.options?.hooks, 'restore');
    }
    // Header, checksum, schema, docIds and documents are validated once,
    // worker-side; client state commits only after the worker succeeds and
    // worker errors (IncompatibleIndexError, ...) propagate rehydrated.

    // Sanitize options to avoid DataCloneError over postMessage.
    // Hooks are rejected fail-closed above; strip them defensively so
    // an empty `{}` no-op never crosses the boundary.
    const sanitizedOptions: RestoreDocumentIndexOptions<TDoc> | undefined = options ? {
      ...options,
      device: undefined,
      options: options.options ? {
        ...options.options,
        device: undefined,
        hooks: undefined,
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

    // Transfer only on opt-in; partial views were already copied by
    // toSnapshotArrayBuffer, so a view's wider backing store is never detached.
    const shouldTransfer = options?.transfer === true;
    const toSend = shouldTransfer ? snapshotBuffer : snapshotBuffer.slice(0);

    const request = this.sendStateRequest<WorkerRestoreResult>(
      'RESTORE',
      { buffer: toSend, options: sanitizedOptions },
      null,
      [toSend]
    );
    const barrier: Promise<void> = Promise.all([this.restoreBarrier, request]).then(
      () => undefined,
      () => undefined
    );
    this.restoreBarrier = barrier;
    let outcome: { result: WorkerRestoreResult; isLatest: boolean };
    try {
      outcome = await request;
    } finally {
      // Clear once the latest pending restore settles (no-op if superseded).
      void barrier.then(() => {
        if (this.restoreBarrier === barrier) this.restoreBarrier = null;
      });
    }

    // Commit only after the worker validated and swapped in the snapshot.
    const staged = this.buildRestoredConfig(outcome.result, options);
    const stagedDocMap = new Map<DocumentId, TDoc>();
    if (options?.documents && Array.isArray(options.documents)) {
      for (const doc of options.documents) {
        stagedDocMap.set(staged.getId(doc), doc);
      }
    } else if (Array.isArray(outcome.result?.records)) {
      for (const doc of outcome.result.records as any[]) {
        if (doc && typeof doc === 'object') {
          if (doc[INTERNAL_WORKER_ID_KEY] !== undefined && doc.id === undefined) {
            doc.id = doc[INTERNAL_WORKER_ID_KEY];
          }
          stagedDocMap.set(staged.getId(doc), doc as TDoc);
        }
      }
    }

    this.committedConfig = staged;
    if (outcome.isLatest) {
      this.setOutgoingConfig(staged);
    }
    this.docMap.clear();
    for (const [k, v] of stagedDocMap) {
      this.docMap.set(k, v);
    }
  }

  private buildRestoredConfig(
    result: WorkerRestoreResult | undefined,
    options?: RestoreDocumentIndexOptions<TDoc>
  ): ClientIndexConfig<TDoc> {
    if (!result || !Array.isArray(result.fields) || !Array.isArray(result.filterFields)) {
      throw new IncompatibleIndexError('worker restore schema', typeof result);
    }
    const userFieldMap = new Map<string, any>();
    for (const uf of options?.options?.fields ?? []) {
      if (typeof uf === 'string') {
        userFieldMap.set(uf, uf);
      } else if (uf && typeof uf === 'object' && typeof uf.name === 'string') {
        userFieldMap.set(uf.name, uf);
      }
    }
    const fieldDefinitions: InternalFieldDef<TDoc>[] = result.fields.map((f) => {
      const uf = userFieldMap.get(f.name);
      const getter = typeof uf === 'object' && uf !== null && typeof uf.getter === 'function'
        ? uf.getter
        : (doc: any) => doc[f.name];
      return { name: f.name, weight: f.weight ?? 1.0, getter };
    });

    const userFilterMap = new Map<string, any>();
    for (const uff of options?.options?.filterFields ?? []) {
      if (typeof uff === 'string') {
        userFilterMap.set(uff, uff);
      } else if (uff && typeof uff === 'object' && typeof uff.name === 'string') {
        userFilterMap.set(uff.name, uff);
      }
    }
    const filterDefinitions = result.filterFields.map((ff) => {
      const uf = userFilterMap.get(ff.name);
      const getter = typeof uf === 'object' && uf !== null && typeof uf.getter === 'function'
        ? uf.getter
        : (doc: any) => doc[ff.name];
      return { name: ff.name, getter };
    });

    let getId: (doc: TDoc) => DocumentId;
    const userIdField = options?.options?.idField;
    if (typeof userIdField === 'function') {
      getId = userIdField;
    } else if (typeof userIdField === 'string') {
      const prop = userIdField;
      getId = (doc: any) => doc[prop] ?? doc[INTERNAL_WORKER_ID_KEY] ?? doc.id;
    } else if (typeof result.idField === 'string' && result.idField) {
      const prop = result.idField;
      getId = (doc: any) => doc[prop] ?? doc[INTERNAL_WORKER_ID_KEY] ?? doc.id;
    } else {
      getId = (doc: any) => doc[INTERNAL_WORKER_ID_KEY] ?? doc.id;
    }
    return { getId, fieldDefinitions, filterDefinitions };
  }

  getStats(): Promise<DocumentIndexStats> {
    return this.runInPostOrder(null, async () => {
      if (this.isDestroyed) {
        throw new Error('[webgpu-search] SearchWorkerClient has been destroyed.');
      }
      return this.sendRequest<DocumentIndexStats>('STATS');
    });
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
