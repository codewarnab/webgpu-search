import { ref, shallowRef, watch, onUnmounted, getCurrentInstance, type Ref, type ShallowRef } from 'vue';
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

export interface UseSearchOptions<TDoc = any> {
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

export interface UseSearchResult<TDoc = any> {
  /** Reactive query string */
  query: Ref<string>;
  /** Ranked search result items */
  results: ShallowRef<DocumentSearchResultItem<TDoc>[]>;
  /** Total matching documents */
  totalMatches: Ref<number>;
  /** Full raw search response */
  response: ShallowRef<DocumentSearchResponse<TDoc> | null>;
  /** True while search scoring is in flight */
  isSearching: Ref<boolean>;
  /** True while index creation or mutations are in flight */
  isIndexing: Ref<boolean>;
  /** True once the engine is initialized and ready */
  isReady: Ref<boolean>;
  /** Error object if initialization, search, or mutation throws */
  error: Ref<Error | null>;
  /** Telemetry and resource statistics */
  stats: ShallowRef<DocumentIndexStats | null>;
  /** Active engine: 'webgpu' | 'cpu' */
  engine: Ref<EngineType | null>;
  /** CPU fallback reason if running on CPU */
  fallbackReason: Ref<FallbackReason | undefined>;
  /** Execution latency breakdown */
  timings: ShallowRef<SearchTimings | null>;
  /** Current mutation epoch counter */
  mutationEpoch: Ref<number>;
  /** Manually trigger an immediate search */
  search: (
    queryOverride?: string,
    optionsOverride?: DocumentSearchOptions<TDoc>
  ) => Promise<DocumentSearchResponse<TDoc> | null>;
  /** Add documents dynamically to the index */
  add: (docs: TDoc | TDoc[], options?: AddOptions) => Promise<MutationResult>;
  /** Update existing documents in the index */
  update: (docs: TDoc | TDoc[]) => Promise<MutationResult>;
  /** Remove documents by ID */
  remove: (ids: DocumentId | DocumentId[]) => Promise<MutationResult>;
  /** Refresh telemetry statistics */
  refreshStats: () => Promise<DocumentIndexStats | null>;
  /** Teardown index or worker resources */
  destroy: () => void;
}

/**
 * Production-ready Vue 3 composable for WebGPU-accelerated document fuzzy and substring search.
 * Provides reactive refs, automatic query debouncing, mutation handling, and telemetry.
 */
export function useSearch<TDoc = any>(options: UseSearchOptions<TDoc> = {}): UseSearchResult<TDoc> {
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

  const query = ref<string>(initialQuery);
  const results = shallowRef<DocumentSearchResultItem<TDoc>[]>([]);
  const totalMatches = ref<number>(0);
  const response = shallowRef<DocumentSearchResponse<TDoc> | null>(null);
  const isSearching = ref<boolean>(false);
  const isIndexing = ref<boolean>(true);
  const isReady = ref<boolean>(false);
  const error = ref<Error | null>(null);
  const stats = shallowRef<DocumentIndexStats | null>(null);
  const engine = ref<EngineType | null>(null);
  const fallbackReason = ref<FallbackReason | undefined>(undefined);
  const timings = shallowRef<SearchTimings | null>(null);
  const mutationEpoch = ref<number>(0);

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
        stats.value = currentStats;
        engine.value = currentStats.engine;
        fallbackReason.value = currentStats.fallbackReason;
        mutationEpoch.value = currentStats.mutationEpoch;
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
    const q = queryOverride !== undefined ? queryOverride : query.value;
    const defaultSearchOpts: Partial<DocumentSearchOptions<TDoc>> = {
      mode: 'fuzzy',
      highlight: true,
      tag: 'mark',
      escapeHtml: true,
      limit: 20,
    };
    const opts = { ...defaultSearchOpts, ...searchOptions, ...optionsOverride };

    if (!indexInstance && !workerClientInstance) return null;

    const currentSearchId = ++searchId;

    if (activeAbortController) {
      activeAbortController.abort();
    }
    const abortController = new AbortController();
    activeAbortController = abortController;

    isSearching.value = true;
    error.value = null;

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
        results.value = res.results;
        totalMatches.value = res.totalMatches;
        response.value = res;
        timings.value = res.timings;
        engine.value = res.engine;
        fallbackReason.value = res.fallbackReason;
        isSearching.value = false;
      }
      return res;
    } catch (err: any) {
      if (err.name === 'AbortError' || currentSearchId !== searchId) {
        return null;
      }
      error.value = err;
      isSearching.value = false;
      return null;
    }
  }

  async function add(docs: TDoc | TDoc[], addOpts?: AddOptions): Promise<MutationResult> {
    isIndexing.value = true;
    error.value = null;
    try {
      let result: MutationResult;
      if (workerClientInstance) {
        result = await workerClientInstance.add(docs, addOpts);
      } else if (indexInstance) {
        result = await indexInstance.add(docs, addOpts);
      } else {
        throw new Error('Search index is not initialized.');
      }
      mutationEpoch.value = result.mutationEpoch;
      await refreshStats();
      if (query.value.trim().length > 0) {
        await search();
      }
      isIndexing.value = false;
      return result;
    } catch (err: any) {
      error.value = err;
      isIndexing.value = false;
      throw err;
    }
  }

  async function update(docs: TDoc | TDoc[]): Promise<MutationResult> {
    isIndexing.value = true;
    error.value = null;
    try {
      let result: MutationResult;
      if (workerClientInstance) {
        result = await workerClientInstance.update(docs);
      } else if (indexInstance) {
        result = await indexInstance.update(docs);
      } else {
        throw new Error('Search index is not initialized.');
      }
      mutationEpoch.value = result.mutationEpoch;
      await refreshStats();
      if (query.value.trim().length > 0) {
        await search();
      }
      isIndexing.value = false;
      return result;
    } catch (err: any) {
      error.value = err;
      isIndexing.value = false;
      throw err;
    }
  }

  async function remove(ids: DocumentId | DocumentId[]): Promise<MutationResult> {
    isIndexing.value = true;
    error.value = null;
    try {
      let result: MutationResult;
      if (workerClientInstance) {
        result = await workerClientInstance.remove(ids);
      } else if (indexInstance) {
        result = await indexInstance.remove(ids);
      } else {
        throw new Error('Search index is not initialized.');
      }
      mutationEpoch.value = result.mutationEpoch;
      await refreshStats();
      if (query.value.trim().length > 0) {
        await search();
      }
      isIndexing.value = false;
      return result;
    } catch (err: any) {
      error.value = err;
      isIndexing.value = false;
      throw err;
    }
  }

  let isDestroyed = false;

  function destroy(): void {
    isDestroyed = true;
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
    isReady.value = false;
  }

  // Initialization
  (async () => {
    isIndexing.value = true;
    error.value = null;
    try {
      const defaultFields = (indexOptions?.fields ?? ['title', 'text']) as Array<DocumentField<TDoc>>;
      const resolvedOptions: DocumentIndexOptions<TDoc> = {
        ...indexOptions,
        fields: defaultFields,
      };

      if (externalWorkerClient) {
        workerClientInstance = externalWorkerClient;
        isOwned = false;
      } else if (externalIndex) {
        indexInstance = externalIndex;
        isOwned = false;
      } else if (worker) {
        const client = new SearchWorkerClient<TDoc>({ worker });
        await client.init(resolvedOptions);
        if (initialDocs && initialDocs.length > 0) {
          await client.add(initialDocs);
        }
        if (isDestroyed) {
          client.destroy();
          return;
        }
        workerClientInstance = client;
        isOwned = true;
      } else {
        const idx = await DocumentIndex.create(initialDocs ?? [], resolvedOptions);
        if (isDestroyed) {
          idx.destroy();
          return;
        }
        indexInstance = idx;
        isOwned = true;
      }

      if (isDestroyed) return;

      isReady.value = true;
      isIndexing.value = false;
      await refreshStats();
      const currentQ = query.value.trim();
      if (currentQ.length > 0) {
        search(currentQ);
      } else if (initialQuery.trim().length > 0) {
        search(initialQuery);
      }
    } catch (err: any) {
      if (!isDestroyed) {
        error.value = err;
        isIndexing.value = false;
      }
    }
  })();

  // Debounced watcher on query
  watch(query, (newQ) => {
    if (!isReady.value || !enabled) return;

    if (debounceTimer) {
      clearTimeout(debounceTimer);
    }

    if (debounceMs <= 0) {
      search(newQ);
    } else {
      debounceTimer = setTimeout(() => {
        search(newQ);
      }, debounceMs);
    }
  });

  if (getCurrentInstance()) {
    onUnmounted(() => {
      destroy();
    });
  }

  return {
    query,
    results,
    totalMatches,
    response,
    isSearching,
    isIndexing,
    isReady,
    error,
    stats,
    engine,
    fallbackReason,
    timings,
    mutationEpoch,
    search,
    add,
    update,
    remove,
    refreshStats,
    destroy,
  };
}

/** Alias for useSearch matching useDocumentSearch convention */
export const useDocumentSearch = useSearch;
