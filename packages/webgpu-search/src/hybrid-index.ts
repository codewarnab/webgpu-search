import { WebGPUEngine } from './webgpu-engine';
import { CPUEngine } from './cpu-engine';
import { WebGPUContextManager } from './context-manager';
import { packStringsToGPUBuffer, checkMemoryBudget } from './buffer';
import { normalizeText } from './unicode-preprocess';
import { searchCpuReference } from './cpu-reference';
import {
  clampLimit,
  isPrintableAsciiTokens,
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
  private gpuEngine: WebGPUEngine | null = null;
  private cpuEngine: CPUEngine;
  private vramAllocatedBytes: number = 0;
  private unsubscribeDeviceLost?: () => void;
  private readonly profileId: TextProfileId = 'unicode-default';
  private readonly folded: boolean;
  private readonly preferGpu: boolean;
  private tokenCount: number = 0;
  private recordTokens: Uint32Array[] = [];
  private corpusAsciiOnly: boolean = true;

  private constructor(
    items: string[],
    folded: boolean,
    preferGpu: boolean,
    recordTokens: Uint32Array[] = [],
    corpusAsciiOnly: boolean = true,
  ) {
    this.items = items;
    this.folded = folded;
    this.preferGpu = preferGpu;
    this.recordTokens = recordTokens;
    this.corpusAsciiOnly = corpusAsciiOnly;
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
        '[webgpu-search] IndexOptions.slotBytes is throw-on-use in v0.2 (dynamic variable-length indexing replaced fixed slots; removal in v0.3). Remove it and rebuild the index.'
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
    let corpusAsciiOnly = true;
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
      if (corpusAsciiOnly && !isPrintableAsciiTokens(norm.tokens)) corpusAsciiOnly = false;
    }
    const index = new SearchIndex(ownedItems, folded, preferGpu, recordTokens, corpusAsciiOnly);
    index.tokenCount = corpusTokens;

    // Empty dataset fast path: zero allocation, route immediately to CPU
    if (ownedItems.length === 0) {
      index.engineType = 'cpu';
      return index;
    }

    const threshold = options.threshold ?? 30_000;
    // Explicit preferGpu:false forces CPU even on large corpora (callers
    // must not need undocumented threshold:Infinity to stay on CPU).
    const shouldAttemptGpu =
      options.preferGpu === false
        ? false
        : preferGpu || ownedItems.length >= threshold;

    if (shouldAttemptGpu) {
      // Measured estimate from the exact post-fold token total instead of
      // the legacy 64 B/record fiction (70k-char corpora must fail fast).
      const measuredAvgBytes: number =
        ownedItems.length === 0 ? 64 : corpusTokens / ownedItems.length;
      const budget = checkMemoryBudget(ownedItems.length, measuredAvgBytes, options.device);
      if (!budget.allowed) {
        console.warn(
          `[webgpu-search] ${budget.reason} Gracefully falling back to CPU engine.`
        );
        index.engineType = 'cpu';
        return index;
      }

      try {
        const gpu = new WebGPUEngine();
        const initialized = await gpu.init(options.device);

        if (initialized && gpu.isReady) {
          // M2 limitation (honest scaffolding, engine swap is M3): the GPU
          // packer is the legacy sanitized-byte path (NFKD strip + `?`),
          // NOT the folded-token stream. CPU parity only; GPU stays legacy
          // until packUnicodeToGPUBuffer + scalar WGSL land.
          const packed = packStringsToGPUBuffer(ownedItems);
          await gpu.loadDataset({
            size: ownedItems.length,
            strings: ownedItems,
            recordsBufferData: packed.recordsBufferData,
            recordsByteLength: packed.recordsByteLength,
            offsetsBufferData: packed.offsetsBufferData,
            offsetsByteLength: packed.offsetsByteLength,
            gpuBufferData: packed.bufferData,
            byteLength: packed.byteLength
          });

          index.gpuEngine = gpu;
          index.engineType = 'webgpu';
          index.vramAllocatedBytes = packed.byteLength;
          // tokenCount is the M2 post-fold total computed at construction;
          // packed.totalChars is legacy sanitized length (ignored for stats).

          // Subscribe to device loss for automatic graceful fallback
          index.unsubscribeDeviceLost = WebGPUContextManager.onDeviceLost(() => {
            console.warn('[webgpu-search] GPU device lost during runtime. Falling back to CPU engine.');
            index.engineType = 'cpu';
          });

          return index;
        }
      } catch (gpuErr) {
        console.warn('[webgpu-search] WebGPU initialization failed, falling back to CPU:', gpuErr);
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
        "cpuAlgorithm:'ufuzzy' is explicit CPU-only and cannot be combined with preferGpu:true. Use preferGpu:false or cpuAlgorithm:'parity'."
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
    // results with query:'' on both paths (resolves '' vs original echo
    // divergence). Note: lone-mark / VS / ZWJ / tatweel-only inputs survive
    // NFC+C+F as single tokens per the survival rule (acceptance checklist),
    // so they are NOT empty here — they search normally (usually 0 hits,
    // echoing the original query). The contract parenthetical listing them
    // as "become empty" is inaccurate for the frozen pipeline (no stripping
    // step); this behavior is pinned by test-m2-preprocess §8b.
    if (normalizedQuery.isEmpty) {
      const timings: SearchTimings = {
        queryUploadMs: 0,
        encodeSubmitMs: 0,
        gpuExecutionMs: null,
        readbackMs: 0,
        totalMs: 0,
        gpuDispatchMs: 0
      };
      return {
        results: [],
        totalMatches: 0,
        query: '',
        mode,
        engine: 'cpu',
        candidateCount: 0,
        hasOverflow: false,
        timings,
        profileId: this.profileId,
        scoringVersion: SCORING_VERSION,
        cpuAlgorithm
      };
    }

    // Empty dataset edge case: 0 items (non-empty query echoes original).
    if (this.items.length === 0) {
      const timings: SearchTimings = {
        queryUploadMs: 0,
        encodeSubmitMs: 0,
        gpuExecutionMs: null,
        readbackMs: 0,
        totalMs: 0,
        gpuDispatchMs: 0
      };
      return {
        results: [],
        totalMatches: 0,
        query,
        mode,
        engine: 'cpu',
        candidateCount: 0,
        hasOverflow: false,
        timings,
        profileId: this.profileId,
        scoringVersion: SCORING_VERSION,
        cpuAlgorithm
      };
    }

    // 1. WebGPU execution path — M2 LEGACY (not parity): the GPU packer maps
    // everything outside printable ASCII (0x20..0x7E) to '?' and truncates
    // queries to 59 chars, while CPU is folded tokens. Route to GPU only when
    // it cannot diverge: printable-ASCII corpus + query and post-fold query
    // <=59 tokens. Otherwise stay on CPU parity. Explicit ufuzzy and
    // over-limit cpu-fallback also force CPU.
    const gpuAsciiSafe: boolean =
      this.corpusAsciiOnly && isPrintableAsciiTokens(normalizedQuery.tokens);
    const gpuLengthSafe: boolean = queryTokenCount <= 59;
    const gpuHandle = this.gpuEngine;
    const useGpu: boolean =
      !forceCpu &&
      cpuAlgorithm !== 'ufuzzy' &&
      gpuAsciiSafe &&
      gpuLengthSafe &&
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
        console.warn('[webgpu-search] GPU search failed, falling back to CPU for this query:', err);
        // Fallthrough to CPU
      }
    }

    // 2. CPU execution path.
    // uFuzzy is quarantined to explicit cpuAlgorithm:'ufuzzy' (CPU-only,
    // never in the differential matrix; explicitly non-conforming scores).
    // Default 'parity' serves the shared-pipeline reference scorer; GPU
    // failures and GPU-unsafe (non-ASCII/long) queries also land here —
    // note fallback can re-score vs legacy GPU (documented M2 limitation).
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
        cpuAlgorithm
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
      cpuAlgorithm
    };
  }

  /**
   * Inspect current index resource allocations and engine state.
   */
  getStats(): IndexStats {
    const adapter = this.gpuEngine?.adapterInfo;
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
      formatVersion: FORMAT_VERSION
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

