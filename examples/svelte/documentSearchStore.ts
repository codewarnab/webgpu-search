import {
  DocumentIndex,
  SearchWorkerClient,
  type DocumentId,
  type DocumentField,
  type DocumentIndexOptions,
  type DocumentSearchOptions,
  type DocumentSearchResultItem,
  type DocumentSearchResponse,
  type DocumentIndexStats,
  type MutationResult,
  type AddOptions,
  type EngineType,
  type FallbackReason,
  type SearchTimings,
} from 'webgpu-search';

export type Subscriber<T> = (value: T) => void;
export type Unsubscriber = () => void;

export interface ReadableStore<T> {
  subscribe(run: Subscriber<T>): Unsubscriber;
}

export interface DocumentSearchState<TDoc = any> {
  query: string;
  results: DocumentSearchResultItem<TDoc>[];
  totalMatches: number;
  response: DocumentSearchResponse<TDoc> | null;
  isSearching: boolean;
  isIndexing: boolean;
  isReady: boolean;
  error: Error | null;
  stats: DocumentIndexStats | null;
  engine: EngineType | null;
  fallbackReason: FallbackReason | undefined;
  timings: SearchTimings | null;
  mutationEpoch: number;
}

export interface DocumentSearchStoreOptions<TDoc = any> {
  /** Initial array of documents to index */
  initialDocs?: TDoc[];
  /** Options passed to DocumentIndex or SearchWorkerClient */
  indexOptions?: DocumentIndexOptions<TDoc>;
  /** Pre-instantiated DocumentIndex instance */
  index?: DocumentIndex<TDoc>;
  /** Pre-instantiated SearchWorkerClient instance */
  workerClient?: SearchWorkerClient<TDoc>;
  /** Worker factory or Worker instance for off-thread searching */
  worker?: Worker | (() => Worker);
  /** Debounce delay in milliseconds before executing query (default: 150ms) */
  debounceMs?: number;
  /** Default search options (e.g. mode, limit, highlight, tag) */
  searchOptions?: DocumentSearchOptions<TDoc>;
  /** Initial query string (default: '') */
  initialQuery?: string;
  /** Whether search execution is active (default: true) */
  enabled?: boolean;
}

export interface DocumentSearchStore<TDoc = any> extends ReadableStore<DocumentSearchState<TDoc>> {
  /** Set query string and trigger debounced search */
  setQuery(query: string): void;
  /** Trigger an immediate search */
  search(
    queryOverride?: string,
    optionsOverride?: DocumentSearchOptions<TDoc>
  ): Promise<DocumentSearchResponse<TDoc> | null>;
  /** Add documents dynamically to the index */
  add(docs: TDoc | TDoc[], options?: AddOptions): Promise<MutationResult>;
  /** Update existing documents in the index */
  update(docs: TDoc | TDoc[]): Promise<MutationResult>;
  /** Remove documents by ID from the index */
  remove(ids: DocumentId | DocumentId[]): Promise<MutationResult>;
  /** Refresh telemetry statistics */
  refreshStats(): Promise<DocumentIndexStats | null>;
  /** Teardown index or worker resources */
  destroy(): void;
}

/**
 * Creates a Svelte-compatible reactive store for WebGPU-accelerated document search.
 * Implements Svelte's `Readable` contract so it can be consumed using the `$store` syntax.
 */
export function createDocumentSearch<TDoc = any>(
  options: DocumentSearchStoreOptions<TDoc> = {}
): DocumentSearchStore<TDoc> {
  const {
    initialDocs,
    indexOptions,
    index: externalIndex,
    workerClient: externalWorkerClient,
    worker,
    debounceMs = 150,
    searchOptions,
    initialQuery = '',
    enabled = true,
  } = options;

  let state: DocumentSearchState<TDoc> = {
    query: initialQuery,
    results: [],
    totalMatches: 0,
    response: null,
    isSearching: false,
    isIndexing: true,
    isReady: false,
    error: null,
    stats: null,
    engine: null,
    fallbackReason: undefined,
    timings: null,
    mutationEpoch: 0,
  };

  const subscribers = new Set<Subscriber<DocumentSearchState<TDoc>>>();

  function notify(): void {
    const snapshot = { ...state };
    for (const sub of subscribers) {
      sub(snapshot);
    }
  }

  function updateState(partial: Partial<DocumentSearchState<TDoc>>): void {
    state = { ...state, ...partial };
    notify();
  }

  let indexInstance: DocumentIndex<TDoc> | null = null;
  let workerClientInstance: SearchWorkerClient<TDoc> | null = null;
  let isOwned = false;

  let searchId = 0;
  let activeAbortController: AbortController | null = null;
  let debounceTimer: any = null;

  async function refreshStats(): Promise<DocumentIndexStats | null> {
    try {
      let currentStats: DocumentIndexStats | null = null;
      if (workerClientInstance) {
        currentStats = await workerClientInstance.getStats();
      } else if (indexInstance) {
        currentStats = indexInstance.getStats();
      }
      if (currentStats) {
        updateState({
          stats: currentStats,
          engine: currentStats.engine,
          fallbackReason: currentStats.fallbackReason,
          mutationEpoch: currentStats.mutationEpoch,
        });
      }
      return currentStats;
    } catch {
      return null;
    }
  }

  async function search(
    queryOverride?: string,
    optionsOverride?: DocumentSearchOptions<TDoc>
  ): Promise<DocumentSearchResponse<TDoc> | null> {
    const q = queryOverride !== undefined ? queryOverride : state.query;
    const opts = { ...searchOptions, ...optionsOverride };

    if (!indexInstance && !workerClientInstance) return null;

    const currentSearchId = ++searchId;

    if (activeAbortController) {
      activeAbortController.abort();
    }
    const abortController = new AbortController();
    activeAbortController = abortController;

    updateState({ isSearching: true, error: null });

    try {
      let res: DocumentSearchResponse<TDoc>;
      if (workerClientInstance) {
        res = await workerClientInstance.search(q, {
          ...opts,
          signal: abortController.signal,
        });
      } else if (indexInstance) {
        res = await indexInstance.search(q, {
          ...opts,
          signal: abortController.signal,
        });
      } else {
        return null;
      }

      if (currentSearchId === searchId) {
        updateState({
          results: res.results,
          totalMatches: res.totalMatches,
          response: res,
          timings: res.timings,
          engine: res.engine,
          fallbackReason: res.fallbackReason,
          isSearching: false,
        });
      }
      return res;
    } catch (err: any) {
      if (err.name === 'AbortError' || currentSearchId !== searchId) {
        return null;
      }
      updateState({ error: err, isSearching: false });
      return null;
    }
  }

  function setQuery(newQuery: string): void {
    updateState({ query: newQuery });
    if (!state.isReady || !enabled) return;

    if (debounceTimer) {
      clearTimeout(debounceTimer);
    }

    if (debounceMs <= 0) {
      search(newQuery);
    } else {
      debounceTimer = setTimeout(() => {
        search(newQuery);
      }, debounceMs);
    }
  }

  async function add(docs: TDoc | TDoc[], addOpts?: AddOptions): Promise<MutationResult> {
    updateState({ isIndexing: true, error: null });
    try {
      let result: MutationResult;
      if (workerClientInstance) {
        result = await workerClientInstance.add(docs, addOpts);
      } else if (indexInstance) {
        result = await indexInstance.add(docs, addOpts);
      } else {
        throw new Error('Search index is not initialized.');
      }
      updateState({ mutationEpoch: result.mutationEpoch });
      await refreshStats();
      if (state.query.trim().length > 0) {
        await search();
      }
      updateState({ isIndexing: false });
      return result;
    } catch (err: any) {
      updateState({ error: err, isIndexing: false });
      throw err;
    }
  }

  async function update(docs: TDoc | TDoc[]): Promise<MutationResult> {
    updateState({ isIndexing: true, error: null });
    try {
      let result: MutationResult;
      if (workerClientInstance) {
        result = await workerClientInstance.update(docs);
      } else if (indexInstance) {
        result = await indexInstance.update(docs);
      } else {
        throw new Error('Search index is not initialized.');
      }
      updateState({ mutationEpoch: result.mutationEpoch });
      await refreshStats();
      if (state.query.trim().length > 0) {
        await search();
      }
      updateState({ isIndexing: false });
      return result;
    } catch (err: any) {
      updateState({ error: err, isIndexing: false });
      throw err;
    }
  }

  async function remove(ids: DocumentId | DocumentId[]): Promise<MutationResult> {
    updateState({ isIndexing: true, error: null });
    try {
      let result: MutationResult;
      if (workerClientInstance) {
        result = await workerClientInstance.remove(ids);
      } else if (indexInstance) {
        result = await indexInstance.remove(ids);
      } else {
        throw new Error('Search index is not initialized.');
      }
      updateState({ mutationEpoch: result.mutationEpoch });
      await refreshStats();
      if (state.query.trim().length > 0) {
        await search();
      }
      updateState({ isIndexing: false });
      return result;
    } catch (err: any) {
      updateState({ error: err, isIndexing: false });
      throw err;
    }
  }

  function destroy(): void {
    if (activeAbortController) {
      activeAbortController.abort();
      activeAbortController = null;
    }
    if (debounceTimer) {
      clearTimeout(debounceTimer);
      debounceTimer = null;
    }
    if (isOwned) {
      if (workerClientInstance) {
        workerClientInstance.destroy();
        workerClientInstance = null;
      }
      if (indexInstance) {
        indexInstance.destroy();
        indexInstance = null;
      }
    }
    updateState({ isReady: false });
  }

  // Initialize engine
  (async () => {
    updateState({ isIndexing: true, error: null });
    try {
      if (externalWorkerClient) {
        workerClientInstance = externalWorkerClient;
        isOwned = false;
      } else if (externalIndex) {
        indexInstance = externalIndex;
        isOwned = false;
      } else if (worker) {
        const client = new SearchWorkerClient<TDoc>({ worker });
        await client.init(indexOptions);
        if (initialDocs && initialDocs.length > 0) {
          await client.add(initialDocs);
        }
        workerClientInstance = client;
        isOwned = true;
      } else {
        const defaultFields = (indexOptions?.fields ?? ['title', 'text']) as Array<DocumentField<TDoc>>;
        const idx = await DocumentIndex.create(initialDocs ?? [], {
          ...indexOptions,
          fields: defaultFields,
        });
        indexInstance = idx;
        isOwned = true;
      }

      updateState({ isReady: true, isIndexing: false });
      await refreshStats();
      if (initialQuery.trim().length > 0) {
        search(initialQuery);
      }
    } catch (err: any) {
      updateState({ error: err, isIndexing: false });
    }
  })();

  return {
    subscribe(run: Subscriber<DocumentSearchState<TDoc>>): Unsubscriber {
      subscribers.add(run);
      run({ ...state });
      return () => {
        subscribers.delete(run);
      };
    },
    setQuery,
    search,
    add,
    update,
    remove,
    refreshStats,
    destroy,
  };
}
