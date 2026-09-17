import { WebGPUEngine } from './webgpu-engine';
import { CPUEngine } from './cpu-engine';
import { WebGPUContextManager } from './context-manager';
import { packStringsToGPUBuffer, checkMemoryBudget } from './buffer';
import { normalizeText } from './unicode-preprocess';
import { searchCpuReference } from './cpu-reference';
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
    if (options.slotBytes !== undefined) {
      throw new IncompatibleOptionError(
        'slotBytes',
        '[webgpu-search] IndexOptions.slotBytes is throw-on-use in v0.2 (dynamic variable-length indexing replaced fixed slots; removal in v0.3). Remove it and rebuild the index.'
      );
    }
    if (options.textProfile !== undefined && options.textProfile !== 'unicode-default') {
      throw new ProfileMismatchError('unicode-default', options.textProfile);
    }
    const folded = !(options.caseSensitive ?? false);
    const preferGpu = options.preferGpu ?? false;
    // M2: exact post-fold tokenization shared by query gate + tokenCount.
    // One normalizeText path for records; queries reuse it in search().
    const recordTokens: Uint32Array[] = new Array(items.length);
    let corpusTokens = 0;
    for (let i = 0; i < items.length; i++) {
      const norm = normalizeText(items[i] ?? '', folded);
      recordTokens[i] = norm.tokens;
      corpusTokens += norm.tokenCount;
    }
    const index = new SearchIndex(items, folded, preferGpu, recordTokens);
    index.tokenCount = corpusTokens;

    // Empty dataset fast path: zero allocation, route immediately to CPU
    if (items.length === 0) {
      index.engineType = 'cpu';
      return index;
    }

    const threshold = options.threshold ?? 30_000;
    const shouldAttemptGpu = preferGpu || items.length >= threshold;

    if (shouldAttemptGpu) {
      const budget = checkMemoryBudget(items.length, 64, options.device);
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
          const packed = packStringsToGPUBuffer(items);
          await gpu.loadDataset({
            size: items.length,
            strings: items,
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
    // M2 query-length gate on exact post-fold token count (shared pipeline).
    const normalizedQuery = normalizeText(query ?? '', this.folded);
    const queryTokenCount = normalizedQuery.tokenCount;
    let forceCpu = false;
    if (queryTokenCount > QUERY_TOKENS_MAX) {
      if (onQueryTooLong === 'cpu-fallback') {
        forceCpu = true;
      } else {
        throw new QueryTooLongError(QUERY_TOKENS_MAX, queryTokenCount, this.profileId);
      }
    }
    const requestedLimit = options.limit ?? options.maxResults ?? 50;
    // Guard against NaN (e.g. a failed parseInt): Math.min/Math.max propagate
    // NaN, which would poison downstream slicing. Treat it as "not provided".
    const clampedLimit = Math.max(1, Math.min(Number.isNaN(requestedLimit) ? 50 : requestedLimit, RESULT_LIMIT_MAX));

    if (signal?.aborted) {
      throw new DOMException('Search aborted', 'AbortError');
    }

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

    // 1. WebGPU execution path (parity scorer; explicit ufuzzy always serves CPU;
    // over-limit with onQueryTooLong:'cpu-fallback' also forces CPU)
    if (!forceCpu && cpuAlgorithm !== 'ufuzzy' && this.engineType === 'webgpu' && this.gpuEngine && this.gpuEngine.isReady) {
      try {
        const gpuResult = await this.gpuEngine.search(query, {
          ...options,
          mode,
          limit: clampedLimit,
          caseSensitive,
          signal
        });

        if (signal?.aborted) {
          throw new DOMException('Search aborted', 'AbortError');
        }

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
    // never in the differential matrix). Default 'parity' serves the
    // shared-pipeline reference scorer; GPU failures also land here.
    if (signal?.aborted) {
      throw new DOMException('Search aborted', 'AbortError');
    }

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
      if (signal?.aborted) {
        throw new DOMException('Search aborted', 'AbortError');
      }
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

    if (signal?.aborted) {
      throw new DOMException('Search aborted', 'AbortError');
    }

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
   * Release all GPU and context resources.
   */
  destroy(): void {
    if (this.unsubscribeDeviceLost) {
      this.unsubscribeDeviceLost();
      this.unsubscribeDeviceLost = undefined;
    }
    if (this.gpuEngine) {
      this.gpuEngine.destroy();
      this.gpuEngine = null;
    }
    this.vramAllocatedBytes = 0;
  }
}

