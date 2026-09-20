import { WebGPUEngine } from './webgpu-engine';
import { CPUEngine } from './cpu-engine';
import { WebGPUContextManager } from './context-manager';
import { packUnicodeToGPUBuffer, checkMemoryBudget } from './buffer';
import { normalizeText } from './unicode-preprocess';
import { searchCpuReference } from './cpu-reference';
import {
  clampLimit,
  throwIfAborted,
} from './runtime-guards';
import {
  FORMAT_VERSION,
  QUERY_TOKENS_MAX,
  RESULT_LIMIT_MAX,
  SCORING_VERSION,
  UNICODE_VERSION,
  IncompatibleOptionError,
  ProfileMismatchError,
  QueryTooLongError,
  type TextProfileId,
} from './text-profile';
import type {
  EngineType,
  FallbackReason,
  IndexOptions,
  IndexStats,
  SearchOptions,
  SearchResponse,
  SearchResultItem,
  SearchTimings
} from './types';

export class SearchIndex {
  private items: string[];
  private isDestroyed: boolean = false;
  private engineType: EngineType = 'cpu';
  private fallbackReason?: FallbackReason;
  private gpuEngine: WebGPUEngine | null = null;
  private cpuEngine: CPUEngine;
  private vramAllocatedBytes: number = 0;
  private unsubscribeDeviceLost?: () => void;
  private readonly profileId: TextProfileId = 'unicode-default';
  private readonly folded: boolean;
  private readonly preferGpu: boolean;
  private tokenCount: number = 0;
  private recordTokens: Uint32Array[] = [];

  private constructor(
    items: string[],
    folded: boolean,
    preferGpu: boolean,
    recordTokens: Uint32Array[] = [],
  ) {
    this.items = items;
    this.folded = folded;
    this.preferGpu = preferGpu;
    this.recordTokens = recordTokens;
    this.cpuEngine = new CPUEngine();
  }

  /**
   * Factory method to create a SearchIndex with dynamic routing between WebGPU and CPU.
   */
  static async create(items: string[], options: IndexOptions = {}): Promise<SearchIndex> {
    if (!Array.isArray(items)) {
      throw new TypeError(
        '[webgpu-search] SearchIndex.create expects items: string[].',
      );
    }
    if (options.slotBytes !== undefined) {
      throw new IncompatibleOptionError(
        'slotBytes',
        '[webgpu-search] slotBytes throw-on-use in v0.2 (fixed slots removed; removal in v0.3).'
      );
    }
    if (options.textProfile !== undefined && options.textProfile !== 'unicode-default') {
      throw new ProfileMismatchError('unicode-default', options.textProfile, 'textProfile');
    }
    const folded = !(options.caseSensitive ?? false);
    const preferGpu = options.preferGpu ?? false;
    // M2: exact post-fold tokenization shared by query gate + tokenCount.
    // One normalizeText path for records; queries reuse it in search().
    // Defensive copy: callers mutating the input array must not desync
    // items[] from recordTokens[].
    const ownedItems: string[] = items.slice();
    const recordTokens: Uint32Array[] = new Array(ownedItems.length);
    let corpusTokens = 0;
    for (let i = 0; i < ownedItems.length; i++) {
      const el: unknown = ownedItems[i];
      if (typeof el !== 'string') {
        throw new TypeError(
          `[webgpu-search] SearchIndex.create expects string items, got ${typeof el} at index ${i}.`,
        );
      }
      const norm = normalizeText(el as string, folded);
      recordTokens[i] = norm.tokens;
      corpusTokens += norm.tokenCount;
    }
    const index = new SearchIndex(ownedItems, folded, preferGpu, recordTokens);
    index.tokenCount = corpusTokens;

    // Empty dataset fast path: zero allocation, route immediately to CPU
    if (ownedItems.length === 0) {
      index.engineType = 'cpu';
      index.fallbackReason = options.preferGpu === false ? 'prefer-cpu' : 'below-threshold';
      return index;
    }

    const threshold = options.threshold ?? 30_000;
    // Explicit preferGpu:false forces CPU even on large corpora (callers
    // must not need undocumented threshold:Infinity to stay on CPU).
    const shouldAttemptGpu =
      options.preferGpu === false
        ? false
        : preferGpu || ownedItems.length >= threshold;

    if (!shouldAttemptGpu) {
      if (options.preferGpu === false) {
        index.fallbackReason = 'prefer-cpu';
      } else {
        index.fallbackReason = 'below-threshold';
      }
    }

    if (shouldAttemptGpu) {
      // Phase 1 (fail-fast pre-init): exact post-fold tokenCount, no 64-B
      // fiction. Device usually undefined here (128 MB fiction stands), but
      // absurd corpora fail fast without churning a GPU context (rejected:
      // acquire-always). Phase 2 re-checks post-init vs real limits (engine).
      const measuredAvgBytes: number = (corpusTokens * 4) / ownedItems.length;
      const budget = checkMemoryBudget(ownedItems.length, measuredAvgBytes, options.device);
      if (!budget.allowed) {
        console.warn(`[webgpu-search] ${budget.reason} Falling back to CPU.`);
        index.engineType = 'cpu';
        index.fallbackReason = 'memory-budget-exceeded';
        return index;
      }

      try {
        const gpu = new WebGPUEngine();
        const initialized = await gpu.init(options.device);

        if (initialized && gpu.isReady) {
          // M3: pack the cached pre-tokenized streams directly — zero
          // second normalizeText pass. Token-only upload (no strings
          // duplication); the engine resolves `text` to `''` and search()
          // enriches from `items` below.
          const packed = packUnicodeToGPUBuffer(recordTokens, { folded, totalTokens: corpusTokens });
          await gpu.loadDataset(packed);

          index.gpuEngine = gpu;
          index.engineType = 'webgpu';
          index.fallbackReason = undefined;
          index.vramAllocatedBytes = packed.recordsByteLength + packed.offsetsByteLength;

          // Subscribe to device loss for automatic graceful fallback
          index.unsubscribeDeviceLost = WebGPUContextManager.onDeviceLost(() => {
            console.warn('[webgpu-search] GPU device lost, falling back to CPU.');
            if (index.gpuEngine) {
              index.gpuEngine.destroy();
              index.gpuEngine = null;
            }
            index.engineType = 'cpu';
            index.fallbackReason = 'device-lost';
          });

          return index;
        } else {
          index.fallbackReason =
            typeof navigator === 'undefined' || !('gpu' in navigator) || !navigator.gpu
              ? 'webgpu-unsupported'
              : 'device-request-failed';
        }
      } catch (gpuErr) {
        console.warn('[webgpu-search] WebGPU initialization failed, falling back to CPU:', gpuErr);
        index.fallbackReason =
          typeof navigator === 'undefined' || !('gpu' in navigator) || !navigator.gpu
            ? 'webgpu-unsupported'
            : 'device-request-failed';
      }
    }

    // Default CPU engine fallback
    index.engineType = 'cpu';
    return index;
  }

  /**
   * Execute search across the indexed strings.
   */
  async search(query: string, options: SearchOptions = {}): Promise<SearchResponse> {
    const {
      mode = 'fuzzy',
      caseSensitive = false,
      signal,
      cpuAlgorithm = 'parity',
      onQueryTooLong = 'throw',
    } = options;
    if (this.isDestroyed) {
      throw new Error('[webgpu-search] SearchIndex has been destroyed.');
    }
    // Abort wins over size gates (cheap check first, no heavy work first).
    throwIfAborted(signal);
    if (typeof query !== 'string') {
      throw new TypeError(
        `[webgpu-search] search expects query: string, got ${typeof query}.`,
      );
    }
    // Pack-time vs query-time mode guard. folded = !indexCaseSensitive, so
    // throw iff queryCaseSensitive !== indexCaseSensitive.
    // Truth table (indexCS → folded → queryCS → behavior):
    //   false → true  + false → pass (default/default)
    //   false → true  + true  → throw (mismatch)
    //   true  → false + false → throw (mismatch)
    //   true  → false + true  → pass (match)
    if (caseSensitive === this.folded) {
      // folded=true means index packed case-insensitive (caseSensitive:false).
      // A per-query flag that disagrees with pack-time mode would silently mismatch.
      // Breaking v0.2 change: build one index per mode instead of varying per query.
      throw new ProfileMismatchError(!this.folded, caseSensitive);
    }
    if (this.preferGpu && cpuAlgorithm === 'ufuzzy') {
      throw new IncompatibleOptionError(
        'cpuAlgorithm',
        "cpuAlgorithm:'ufuzzy' is CPU-only; use preferGpu:false or cpuAlgorithm:'parity'."
      );
    }
    // Cheap pre-gate before the expensive NFC+fold pipeline: fold expands
    // at most 1->3, so raw code points beyond 3x the cap are definitely over.
    // Gigantic queries (>1M UTF-16 units) throw on the upper-bound estimate
    // to avoid materializing NFC+fold work (DoS); all normal sizes fall
    // through to the exact post-fold gate below so `actual` stays exact.
    // Code points are counted without allocating a spread array.
    const rawTrimmed: string = query.trim();
    let forceCpu = false;
    if (rawTrimmed.length > QUERY_TOKENS_MAX * 4) {
      if (rawTrimmed.length > 1_000_000) {
        let cpCount = 0;
        for (const _ch of rawTrimmed) cpCount++;
        // Upper-bound estimate (fold expands at most 1->3): `actual` on this
        // gigantic-query path is an estimate, not the exact post-fold count
        // (exact gate below covers all normal sizes). cpu-fallback here still
        // forces an exhaustive CPU scan — callers opt into the cost.
        const est: number = cpCount * 3;
        if (est > QUERY_TOKENS_MAX) {
          if (onQueryTooLong !== 'cpu-fallback') {
            throw new QueryTooLongError(QUERY_TOKENS_MAX, est, this.profileId);
          }
          forceCpu = true;
        }
      }
    }
    // M2 query-length gate on exact post-fold token count (shared pipeline).
    const normalizedQuery = normalizeText(query, this.folded);
    const queryTokenCount = normalizedQuery.tokenCount;
    if (queryTokenCount > QUERY_TOKENS_MAX) {
      if (onQueryTooLong === 'cpu-fallback') {
        forceCpu = true;
      } else {
        throw new QueryTooLongError(QUERY_TOKENS_MAX, queryTokenCount, this.profileId);
      }
    }
    const requestedLimit = options.limit ?? options.maxResults ?? 50;
    // Shared clamp: finite + floor + 1..8192, NaN/non-number -> 50.
    const clampedLimit = clampLimit(requestedLimit);

    throwIfAborted(signal);

    // Degenerate post-processing queries that normalize to zero post-fold
    // tokens (whitespace-only, U+3000-only, empty) return unified empty
    // results with query:'' on both paths. Lone-mark / VS / ZWJ /
    // tatweel-only inputs survive NFC+C+F as single tokens (pinned §8b), so
    // they are NOT empty here — they search normally echoing the original.
    const noHits = (q: string): SearchResponse => ({
      results: [], totalMatches: 0, query: q, mode, engine: 'cpu',
      candidateCount: 0, hasOverflow: false,
      timings: { queryUploadMs: 0, encodeSubmitMs: 0, gpuExecutionMs: null, readbackMs: 0, totalMs: 0, gpuDispatchMs: 0 },
      profileId: this.profileId, scoringVersion: SCORING_VERSION, cpuAlgorithm,
      fallbackReason: forceCpu ? 'query-too-long' : this.fallbackReason,
    });
    if (normalizedQuery.isEmpty) {
      return noHits('');
    }

    // Empty dataset edge case: 0 items (non-empty query echoes original).
    if (this.items.length === 0) {
      return noHits(query);
    }

    // 1. WebGPU execution path (M3 parity): all valid queries
    // (queryTokenCount <= 128, enforced above) route to WebGPU when
    // available and ready. Explicit ufuzzy and over-limit cpu-fallback
    // force CPU. Failures fall through to the parity CPU scorer below.
    const gpuHandle = this.gpuEngine;
    const useGpu: boolean =
      !forceCpu &&
      cpuAlgorithm !== 'ufuzzy' &&
      this.engineType === 'webgpu' &&
      gpuHandle !== null &&
      gpuHandle.isReady;
    if (useGpu && gpuHandle !== null) {
      try {
        const gpuResult = await gpuHandle.search(query, {
          ...options,
          mode,
          limit: clampedLimit,
          caseSensitive,
          signal
        });

        throwIfAborted(signal);

        // Remap results to include actual text strings
        const enrichedResults = gpuResult.results.map((item) => ({
          ...item,
          text: this.items[item.index] ?? ''
        }));

        return {
          query: gpuResult.query,
          mode: gpuResult.mode,
          engine: 'webgpu',
          totalMatches: gpuResult.totalMatches,
          candidateCount: gpuResult.candidateCount,
          hasOverflow: gpuResult.hasOverflow,
          results: enrichedResults,
          timings: gpuResult.timings,
          profileId: this.profileId,
          scoringVersion: SCORING_VERSION,
          cpuAlgorithm
        };
      } catch (err: any) {
        if (err.name === 'AbortError') {
          throw err;
        }
        console.warn('[webgpu-search] GPU search failed, CPU fallback:', err);
        this.engineType = 'cpu';
        this.fallbackReason = 'gpu-execution-error';
        if (this.gpuEngine) {
          try { this.gpuEngine.destroy(); } catch {}
          this.gpuEngine = null;
        }
      }
    }

    // 2. CPU execution path.
    // uFuzzy is quarantined to explicit cpuAlgorithm:'ufuzzy' (CPU-only,
    // never in the differential matrix; explicitly non-conforming scores).
    // Default 'parity' serves the shared-pipeline reference scorer; GPU
    // failures also land here with identical parity semantics.
    throwIfAborted(signal);

    if (cpuAlgorithm === 'ufuzzy') {
      let legacyResult: {
        query: string;
        totalMatches: number;
        results: SearchResultItem[];
        durationMs: number;
      };
      if (mode === 'fuzzy') {
        legacyResult = this.cpuEngine.searchUFuzzy(this.items, query, clampedLimit, caseSensitive);
      } else {
        legacyResult = this.cpuEngine.searchNative(this.items, query, clampedLimit, caseSensitive);
      }
      throwIfAborted(signal);
      const timings: SearchTimings = {
        queryUploadMs: 0,
        encodeSubmitMs: 0,
        gpuExecutionMs: null,
        readbackMs: 0,
        totalMs: legacyResult.durationMs,
        gpuDispatchMs: 0
      };
      return {
        query: legacyResult.query,
        mode,
        engine: 'cpu',
        totalMatches: legacyResult.totalMatches,
        candidateCount: Math.min(legacyResult.totalMatches, RESULT_LIMIT_MAX),
        hasOverflow: legacyResult.totalMatches > RESULT_LIMIT_MAX,
        results: legacyResult.results,
        timings,
        profileId: this.profileId,
        scoringVersion: SCORING_VERSION,
        cpuAlgorithm,
        fallbackReason: 'cpu-algorithm-requested'
      };
    }

    const parityResult = searchCpuReference(
      this.recordTokens,
      normalizedQuery.tokens,
      mode,
      clampedLimit,
      this.items,
    );

    throwIfAborted(signal);

    const timings: SearchTimings = {
      queryUploadMs: 0,
      encodeSubmitMs: 0,
      gpuExecutionMs: null,
      readbackMs: 0,
      totalMs: parityResult.durationMs,
      gpuDispatchMs: 0
    };

    let effectiveFallbackReason = this.fallbackReason;
    if (forceCpu) {
      effectiveFallbackReason = 'query-too-long';
    } else if (useGpu && gpuHandle !== null) {
      effectiveFallbackReason = 'gpu-execution-error';
    }

    return {
      query,
      mode,
      engine: 'cpu',
      totalMatches: parityResult.totalMatches,
      candidateCount: Math.min(parityResult.totalMatches, RESULT_LIMIT_MAX),
      hasOverflow: parityResult.totalMatches > RESULT_LIMIT_MAX,
      results: parityResult.results,
      timings,
      profileId: this.profileId,
      scoringVersion: SCORING_VERSION,
      cpuAlgorithm,
      fallbackReason: effectiveFallbackReason
    };
  }

  /**
   * Inspect current index resource allocations and engine state.
   */
  getStats(): IndexStats {
    const adapter = this.gpuEngine?.adapterInfo;
    const ramBytes = this.tokenCount * 4;
    return {
      size: this.items.length,
      engine: this.engineType,
      vramAllocatedBytes: this.vramAllocatedBytes,
      adapterVendor: adapter?.vendor,
      adapterRenderer: adapter?.renderer,
      profileId: this.profileId,
      unicodeVersion: UNICODE_VERSION,
      scoringVersion: SCORING_VERSION,
      tokenCount: this.tokenCount,
      folded: this.folded,
      formatVersion: FORMAT_VERSION,
      fallbackReason: this.fallbackReason,
      memory: {
        vramBytes: this.vramAllocatedBytes,
        ramBytes,
        totalBytes: this.vramAllocatedBytes + ramBytes
      }
    };
  }

  /**
   * Release all GPU and context resources. Fail-closed: clears CPU residency
   * (`items`/`recordTokens`/`tokenCount` reset, engine drops to cpu) and
   * subsequent `search()` throws; post-destroy `getStats()` reads 0/empty.
   */
  destroy(): void {
    this.isDestroyed = true;
    if (this.unsubscribeDeviceLost) {
      this.unsubscribeDeviceLost();
      this.unsubscribeDeviceLost = undefined;
    }
    if (this.gpuEngine) {
      this.gpuEngine.destroy();
      this.gpuEngine = null;
    }
    // Release CPU residency so post-destroy searches fail closed instead
    // of silently succeeding against leaked memory.
    this.recordTokens = [];
    this.items = [];
    this.tokenCount = 0;
    this.engineType = 'cpu';
    this.vramAllocatedBytes = 0;
  }
}

