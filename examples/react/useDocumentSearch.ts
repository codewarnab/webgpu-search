import { useState, useEffect, useRef, useCallback } from 'react';
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

export interface UseDocumentSearchOptions<TDoc = any> {
  /** Initial array of documents to index on mount */
  initialDocs?: TDoc[];
  /** Options passed to DocumentIndex or SearchWorkerClient */
  indexOptions?: DocumentIndexOptions<TDoc>;
  /** External DocumentIndex instance (takes precedence over worker and initialDocs) */
  index?: DocumentIndex<TDoc>;
  /** External SearchWorkerClient instance */
  workerClient?: SearchWorkerClient<TDoc>;
  /** Worker factory or Worker instance to run searches off the main thread */
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

export interface UseDocumentSearchResult<TDoc = any> {
  /** Current search query */
  query: string;
  /** Setter for search query (triggers debounced search) */
  setQuery: (query: string) => void;
  /** Ranked search result items */
  results: DocumentSearchResultItem<TDoc>[];
  /** Total matching documents (before limit truncation) */
  totalMatches: number;
  /** Full raw search response */
  response: DocumentSearchResponse<TDoc> | null;
  /** True while search dispatch and scoring are in flight */
  isSearching: boolean;
  /** True while index creation or mutations are in flight */
  isIndexing: boolean;
  /** True once the search engine is initialized and ready */
  isReady: boolean;
  /** Error object if initialization, mutation, or search fails */
  error: Error | null;
  /** Index telemetry and resource stats (VRAM, RAM, rows, tombstones) */
  stats: DocumentIndexStats | null;
  /** Active engine: 'webgpu' | 'cpu' */
  engine: EngineType | null;
  /** CPU fallback reason if running on CPU */
  fallbackReason: FallbackReason | undefined;
  /** Execution timings breakdown */
  timings: SearchTimings | null;
  /** Current index mutation epoch counter */
  mutationEpoch: number;
  /** Manually trigger an immediate search with optional query or options overrides */
  search: (
    queryOverride?: string,
    optionsOverride?: DocumentSearchOptions<TDoc>
  ) => Promise<DocumentSearchResponse<TDoc> | null>;
  /** Add documents to the index */
  add: (docs: TDoc | TDoc[], options?: AddOptions) => Promise<MutationResult>;
  /** Update existing documents in the index */
  update: (docs: TDoc | TDoc[]) => Promise<MutationResult>;
  /** Remove documents by ID from the index */
  remove: (ids: DocumentId | DocumentId[]) => Promise<MutationResult>;
  /** Refresh index telemetry and memory stats */
  refreshStats: () => Promise<DocumentIndexStats | null>;
}

/**
 * Production-ready React hook for WebGPU-accelerated document fuzzy and substring search.
 * Supports off-thread Web Workers, live dynamic mutations, highlighting, and observability telemetry.
 */
export function useDocumentSearch<TDoc = any>(
  options: UseDocumentSearchOptions<TDoc> = {}
): UseDocumentSearchResult<TDoc> {
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

  const [query, setQuery] = useState<string>(initialQuery);
  const [results, setResults] = useState<DocumentSearchResultItem<TDoc>[]>([]);
  const [totalMatches, setTotalMatches] = useState<number>(0);
  const [response, setResponse] = useState<DocumentSearchResponse<TDoc> | null>(null);
  const [isSearching, setIsSearching] = useState<boolean>(false);
  const [isIndexing, setIsIndexing] = useState<boolean>(true);
  const [isReady, setIsReady] = useState<boolean>(false);
  const [error, setError] = useState<Error | null>(null);
  const [stats, setStats] = useState<DocumentIndexStats | null>(null);
  const [engine, setEngine] = useState<EngineType | null>(null);
  const [fallbackReason, setFallbackReason] = useState<FallbackReason | undefined>(undefined);
  const [timings, setTimings] = useState<SearchTimings | null>(null);
  const [mutationEpoch, setMutationEpoch] = useState<number>(0);

  // Active instance refs
  const indexRef = useRef<DocumentIndex<TDoc> | null>(null);
  const workerClientRef = useRef<SearchWorkerClient<TDoc> | null>(null);
  const isOwnedInstanceRef = useRef<boolean>(false);

  // Concurrency & debounce guards
  const searchIdCounterRef = useRef<number>(0);
  const abortControllerRef = useRef<AbortController | null>(null);
  const debounceTimerRef = useRef<any>(null);

  // Keep latest search options and query in refs for callbacks
  const searchOptionsRef = useRef(searchOptions);
  searchOptionsRef.current = searchOptions;
  const queryRef = useRef(query);
  queryRef.current = query;

  // Refresh index telemetry stats
  const refreshStats = useCallback(async (): Promise<DocumentIndexStats | null> => {
    try {
      let currentStats: DocumentIndexStats | null = null;
      if (workerClientRef.current) {
        currentStats = await workerClientRef.current.getStats();
      } else if (indexRef.current) {
        currentStats = indexRef.current.getStats();
      }
      if (currentStats) {
        setStats(currentStats);
        setEngine(currentStats.engine);
        setFallbackReason(currentStats.fallbackReason);
        setMutationEpoch(currentStats.mutationEpoch);
      }
      return currentStats;
    } catch {
      return null;
    }
  }, []);

  // Execute search immediately
  const executeSearch = useCallback(
    async (
      queryOverride?: string,
      optionsOverride?: DocumentSearchOptions<TDoc>
    ): Promise<DocumentSearchResponse<TDoc> | null> => {
      const q = queryOverride !== undefined ? queryOverride : queryRef.current;
      const defaultSearchOpts: Partial<DocumentSearchOptions<TDoc>> = {
        mode: 'fuzzy',
        highlight: true,
        tag: 'mark',
        escapeHtml: true,
        limit: 20,
      };
      const opts = { ...defaultSearchOpts, ...searchOptionsRef.current, ...optionsOverride };

      if (!indexRef.current && !workerClientRef.current) {
        return null;
      }

      // Increment monotonic search sequence
      const searchId = ++searchIdCounterRef.current;

      // Abort previous in-flight request
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      const abortController = new AbortController();
      abortControllerRef.current = abortController;

      setIsSearching(true);
      setError(null);

      try {
        let res: DocumentSearchResponse<TDoc>;
        if (workerClientRef.current) {
          res = await workerClientRef.current.search(q, {
            ...opts,
            signal: abortController.signal,
          });
        } else if (indexRef.current) {
          res = await indexRef.current.search(q, {
            ...opts,
            signal: abortController.signal,
          });
        } else {
          return null;
        }

        // Only commit if this is still the newest request
        if (searchId === searchIdCounterRef.current) {
          setResults(res.results);
          setTotalMatches(res.totalMatches);
          setResponse(res);
          setTimings(res.timings);
          setEngine(res.engine);
          setFallbackReason(res.fallbackReason);
          setIsSearching(false);
        }
        return res;
      } catch (err: any) {
        if (err.name === 'AbortError' || searchId !== searchIdCounterRef.current) {
          // Superseded or explicitly aborted: ignore
          return null;
        }
        setError(err);
        setIsSearching(false);
        return null;
      }
    },
    []
  );

  // Add documents
  const add = useCallback(
    async (docs: TDoc | TDoc[], addOpts?: AddOptions): Promise<MutationResult> => {
      setIsIndexing(true);
      setError(null);
      try {
        let result: MutationResult;
        if (workerClientRef.current) {
          result = await workerClientRef.current.add(docs, addOpts);
        } else if (indexRef.current) {
          result = await indexRef.current.add(docs, addOpts);
        } else {
          throw new Error('Search index is not initialized.');
        }
        setMutationEpoch(result.mutationEpoch);
        await refreshStats();
        if (queryRef.current.trim().length > 0) {
          await executeSearch();
        }
        setIsIndexing(false);
        return result;
      } catch (err: any) {
        setError(err);
        setIsIndexing(false);
        throw err;
      }
    },
    [executeSearch, refreshStats]
  );

  // Update documents
  const update = useCallback(
    async (docs: TDoc | TDoc[]): Promise<MutationResult> => {
      setIsIndexing(true);
      setError(null);
      try {
        let result: MutationResult;
        if (workerClientRef.current) {
          result = await workerClientRef.current.update(docs);
        } else if (indexRef.current) {
          result = await indexRef.current.update(docs);
        } else {
          throw new Error('Search index is not initialized.');
        }
        setMutationEpoch(result.mutationEpoch);
        await refreshStats();
        if (queryRef.current.trim().length > 0) {
          await executeSearch();
        }
        setIsIndexing(false);
        return result;
      } catch (err: any) {
        setError(err);
        setIsIndexing(false);
        throw err;
      }
    },
    [executeSearch, refreshStats]
  );

  // Remove documents
  const remove = useCallback(
    async (ids: DocumentId | DocumentId[]): Promise<MutationResult> => {
      setIsIndexing(true);
      setError(null);
      try {
        let result: MutationResult;
        if (workerClientRef.current) {
          result = await workerClientRef.current.remove(ids);
        } else if (indexRef.current) {
          result = await indexRef.current.remove(ids);
        } else {
          throw new Error('Search index is not initialized.');
        }
        setMutationEpoch(result.mutationEpoch);
        await refreshStats();
        if (queryRef.current.trim().length > 0) {
          await executeSearch();
        }
        setIsIndexing(false);
        return result;
      } catch (err: any) {
        setError(err);
        setIsIndexing(false);
        throw err;
      }
    },
    [executeSearch, refreshStats]
  );

  // Initialize engine on mount
  useEffect(() => {
    let isCancelled = false;

    async function initEngine() {
      setIsIndexing(true);
      setError(null);

      try {
        const defaultFields = (indexOptions?.fields ?? ['title', 'text']) as Array<DocumentField<TDoc>>;
        const resolvedOptions: DocumentIndexOptions<TDoc> = {
          ...indexOptions,
          fields: defaultFields,
        };

        if (externalWorkerClient) {
          workerClientRef.current = externalWorkerClient;
          isOwnedInstanceRef.current = false;
        } else if (externalIndex) {
          indexRef.current = externalIndex;
          isOwnedInstanceRef.current = false;
        } else if (worker) {
          const client = new SearchWorkerClient<TDoc>({ worker });
          await client.init(resolvedOptions);
          if (initialDocs && initialDocs.length > 0) {
            await client.add(initialDocs);
          }
          if (isCancelled) {
            client.destroy();
            return;
          }
          workerClientRef.current = client;
          isOwnedInstanceRef.current = true;
        } else {
          const idx = await DocumentIndex.create(initialDocs ?? [], resolvedOptions);
          if (isCancelled) {
            idx.destroy();
            return;
          }
          indexRef.current = idx;
          isOwnedInstanceRef.current = true;
        }

        if (!isCancelled) {
          setIsReady(true);
          setIsIndexing(false);
          await refreshStats();
        }
      } catch (err: any) {
        if (!isCancelled) {
          setError(err);
          setIsIndexing(false);
        }
      }
    }

    initEngine();

    return () => {
      isCancelled = true;
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
      if (isOwnedInstanceRef.current) {
        if (workerClientRef.current) {
          workerClientRef.current.destroy();
          workerClientRef.current = null;
        }
        if (indexRef.current) {
          indexRef.current.destroy();
          indexRef.current = null;
        }
      }
    };
  }, []);

  // Debounced search trigger on query changes
  useEffect(() => {
    if (!isReady || !enabled) return;

    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    if (debounceMs <= 0) {
      executeSearch(query);
    } else {
      debounceTimerRef.current = setTimeout(() => {
        executeSearch(query);
      }, debounceMs);
    }

    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, [query, isReady, enabled, debounceMs, executeSearch]);

  return {
    query,
    setQuery,
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
    search: executeSearch,
    add,
    update,
    remove,
    refreshStats,
  };
}
