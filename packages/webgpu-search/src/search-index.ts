import { WebGPUEngine } from './webgpu-engine';
import { CPUEngine } from './cpu-engine';
import { GpuDevicePool, assertValidPowerPreference } from './gpu-device-pool';
import { packDataset, checkMemoryBudget } from './dataset-packing';
import { normalizeText } from './text-normalization';
import { scoreExactMatches } from './exact-scorer';
import {
  initialEngineState,
  transitionEngineState,
  type EngineState,
} from './engine-state';
import {
  normalizeTypoTolerance,
  type NormalizedTypoOptions
} from './search/typo-tolerance';
import {
  normalizeTokenMatchOptions,
  type NormalizedTokenMatchOptions
} from './search/token-search';
import {
  normalizePrefixOptions,
  assertPrefixLengthForQuery,
  type NormalizedPrefixOptions
} from './search/prefix-search';
import {
  clampLimit,
  throwIfAborted,
  nowMs,
  assertValidMode,
} from './guard';
import {
  normalizeCostBudgetOptions,
  throwIfBudgetAborted,
  assertTimeBudget,
  assertCandidateBudget,
  isBroadQueryHeuristic,
  isBroadSelectivity,
  broadQueryRouteWarning,
  broadSelectivityWarning,
  candidateOverflowWarning,
} from './diagnostics';
import {
  DATASET_FORMAT_VERSION,
  QUERY_TOKENS_MAX,
  RESULT_LIMIT_MAX,
  SCORING_VERSION,
  UNICODE_VERSION,
  normalizeCpuScorer,
  IncompatibleOptionError,
  ProfileMismatchError,
  QueryTooLongError,
  type TextProfileId,
} from './text-profile';
import type {
  EngineType,
  IndexOptions,
  IndexStats,
  QueryDiagnostics,
  SearchOptions,
  SearchResponse,
  SearchResultItem,
  SearchTimings
} from './types';

export class SearchIndex {
  private items: string[];
  private isDestroyed: boolean = false;
  private engineState: EngineState = initialEngineState();
  private gpuEngine: WebGPUEngine | null = null;
  private cpuEngine: CPUEngine;
  private vramAllocatedBytes: number = 0;
  private unsubscribeDeviceLost?: () => void;
  private readonly profileId: TextProfileId = 'unicode-default';
  private readonly normalized: boolean;
  private readonly preferGpu: boolean;
  private tokenCount: number = 0;
  private recordTokens: Uint32Array[] = [];
  private creationDevice?: GPUDevice;
  private creationPowerPreference?: GPUPowerPreference;

  private constructor(
    items: string[],
    normalized: boolean,
    preferGpu: boolean,
    recordTokens: Uint32Array[] = [],
  ) {
    this.items = items;
    this.normalized = normalized;
    this.preferGpu = preferGpu;
    this.recordTokens = recordTokens;
    this.cpuEngine = new CPUEngine();
  }

  private subscribeDeviceLost(): void {
    if (this.unsubscribeDeviceLost) {
      this.unsubscribeDeviceLost();
      this.unsubscribeDeviceLost = undefined;
    }
    this.unsubscribeDeviceLost = GpuDevicePool.onDeviceLost(() => {
      console.warn('[webgpu-search] GPU device lost, falling back to CPU.');
      if (this.gpuEngine) {
        this.gpuEngine.destroy();
        this.gpuEngine = null;
      }
      this.vramAllocatedBytes = 0;
      this.engineState = transitionEngineState(this.engineState, 'cpu', 'device-lost');
    });
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
        '[webgpu-search] slotBytes is not supported (fixed slots removed).'
      );
    }
    if (options.textProfile !== undefined && options.textProfile !== 'unicode-default') {
      throw new ProfileMismatchError('unicode-default', options.textProfile, 'textProfile');
    }
    // Fail-closed: unknown powerPreference throws even on CPU-only paths.
    assertValidPowerPreference(options.powerPreference);
    const normalized = !(options.caseSensitive ?? false);
    const preferGpu = options.preferGpu ?? false;
    // exact post-normalization tokenization shared by query gate + tokenCount.
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
      const norm = normalizeText(el as string, normalized);
      recordTokens[i] = norm.tokens;
      corpusTokens += norm.tokenCount;
    }
    const index = new SearchIndex(ownedItems, normalized, preferGpu, recordTokens);
    index.tokenCount = corpusTokens;
    index.creationDevice = options.device;
    index.creationPowerPreference = options.powerPreference;

    // Empty dataset fast path: zero allocation, route immediately to CPU
    if (ownedItems.length === 0) {
      index.engineState = transitionEngineState(
        index.engineState,
        'cpu',
        options.preferGpu === false ? 'prefer-cpu' : 'below-threshold',
      );
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
        index.engineState = transitionEngineState(index.engineState, 'cpu', 'prefer-cpu');
      } else {
        index.engineState = transitionEngineState(index.engineState, 'cpu', 'below-threshold');
      }
    }

    if (shouldAttemptGpu) {
      // Phase 1 (fail-fast pre-init): exact post-normalization tokenCount, no 64-B
      // fiction. Device usually undefined here (128 MB fiction stands), but
      // absurd corpora fail fast without churning a GPU context (rejected:
      // acquire-always). Phase 2 re-checks post-init vs real limits (engine).
      const measuredAvgBytes: number = (corpusTokens * 4) / ownedItems.length;
      const budget = checkMemoryBudget(ownedItems.length, measuredAvgBytes, options.device);
      if (!budget.allowed) {
        console.warn(`[webgpu-search] ${budget.reason} Falling back to CPU.`);
        index.engineState = transitionEngineState(index.engineState, 'cpu', 'memory-budget-exceeded');
        return index;
      }

      try {
        const gpu = new WebGPUEngine();
        const initialized = await gpu.init(
          options.device !== undefined || options.powerPreference !== undefined
            ? { device: options.device, powerPreference: options.powerPreference }
            : undefined
        );

        if (initialized && gpu.isReady) {
          // pack the cached pre-tokenized streams directly — zero
          // second normalizeText pass. Token-only upload (no strings
          // duplication); the engine resolves `text` to `''` and search()
          // enriches from `items` below.
          const packed = packDataset(recordTokens, { normalized, totalTokens: corpusTokens });
          await gpu.loadDataset(packed);

          index.gpuEngine = gpu;
          index.engineState = transitionEngineState(index.engineState, 'webgpu');
          index.vramAllocatedBytes = packed.recordsByteLength + packed.offsetsByteLength;

          // Subscribe to device loss for automatic graceful fallback.
          // Single subscription (re-subscribes replace the old one) so
          // rebuild cycles never leak listeners.
          index.subscribeDeviceLost();

          return index;
        } else {
          index.engineState = transitionEngineState(
            index.engineState,
            'cpu',
            typeof navigator === 'undefined' || !('gpu' in navigator) || !navigator.gpu
              ? 'webgpu-unsupported'
              : 'device-request-failed',
          );
        }
      } catch (gpuErr) {
        console.warn('[webgpu-search] WebGPU initialization failed, falling back to CPU:', gpuErr);
        index.engineState = transitionEngineState(
          index.engineState,
          'cpu',
          typeof navigator === 'undefined' || !('gpu' in navigator) || !navigator.gpu
            ? 'webgpu-unsupported'
            : 'device-request-failed',
        );
      }
    }

    // Default CPU engine fallback
    index.engineState = transitionEngineState(
      index.engineState,
      'cpu',
      index.engineState.fallbackReason,
    );
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
      onQueryTooLong = 'throw',
    } = options;
    if ((options as unknown as Record<string, unknown>).cpuAlgorithm !== undefined) {
      throw new IncompatibleOptionError(
        'cpuScorer',
        "[webgpu-search] cpuAlgorithm was removed; use cpuScorer: 'exact' | 'ufuzzy'."
      );
    }
    const cpuScorer = normalizeCpuScorer(options.cpuScorer) ?? 'exact';
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
    assertValidMode(mode);
    // fail-closed option validation up front so malformed
    // token/prefix/typo shapes throw identically on GPU and CPU paths.
    // 'fuzzy' validates typo shape but ignores it (subsequence matching is
    // inherently typo-tolerant); 'substring'/'token'/'prefix' honor it.
    const tokenOpts: NormalizedTokenMatchOptions = normalizeTokenMatchOptions(options.tokenMatch);
    const prefixOpts: NormalizedPrefixOptions = normalizePrefixOptions(options.prefixMatch);
    const typo: NormalizedTypoOptions = normalizeTypoTolerance(options.typoTolerance);
    // Prefix polarity: only enforce when the caller explicitly set exactCase.
    // The default (prefixMatch undefined) follows the query caseSensitive flag
    // so `create({caseSensitive:true}).search(q,{mode:'prefix',caseSensitive:true})`
    // works without redundant `prefixMatch:{exactCase:true}`.
    if (mode === 'prefix' && options.prefixMatch?.exactCase !== undefined && prefixOpts.exactCase !== caseSensitive) {
      throw new ProfileMismatchError(caseSensitive, prefixOpts.exactCase, 'prefixMatch.exactCase');
    }
    // Legacy ufuzzy/native scorers only implement fuzzy/substring-exact.
    if (cpuScorer === 'ufuzzy' && (mode === 'token' || mode === 'prefix' || typo.enabled)) {
      throw new IncompatibleOptionError(
        'cpuScorer',
        `cpuScorer:'ufuzzy' supports only exact 'fuzzy'/'substring' modes without typo tolerance (got mode '${mode}'${typo.enabled ? ' with typoTolerance' : ''}). Use cpuScorer:'exact'.`
      );
    }
    // Pack-time vs query-time mode guard. normalized = !indexCaseSensitive, so
    // throw iff queryCaseSensitive !== indexCaseSensitive.
    // Truth table (indexCS → normalized → queryCS → behavior):
    // false → true + false → pass (default/default)
    // false → true + true → throw (mismatch)
    // true → false + false → throw (mismatch)
    // true → false + true → pass (match)
    if (caseSensitive === this.normalized) {
      // normalized=true means index packed case-insensitive (caseSensitive:false).
      // A per-query flag that disagrees with pack-time mode would silently mismatch.
      // Breaking change: build one index per mode instead of varying per query.
      throw new ProfileMismatchError(!this.normalized, caseSensitive);
    }
    if (this.preferGpu && cpuScorer === 'ufuzzy') {
      throw new IncompatibleOptionError(
        'cpuScorer',
        "cpuScorer:'ufuzzy' is CPU-only; use preferGpu:false or cpuScorer:'exact'."
      );
    }
    // fail-closed cost-budget + diagnostics validation first so
    // malformed budgets throw identically on GPU/CPU paths, empty corpora,
    // and overlong queries (same precedence as DocumentIndex: budget before
    // the query-too-long gate and before expensive NFC+fold work).
    const budget = normalizeCostBudgetOptions(options.budget);
    if (options.diagnostics !== undefined && typeof options.diagnostics !== 'boolean') {
      throw new TypeError('[webgpu-search] options.diagnostics must be a boolean.');
    }
    const wantsDiagnostics = options.diagnostics === true;
    const needsClock = wantsDiagnostics || budget?.maxExecutionTimeMs !== undefined;
    const queryStartMs = needsClock ? nowMs() : 0;
    const diagWarnings: string[] = [];
    // Cheap pre-gate before the expensive NFC+folding pipeline: folding expands
    // at most 1->3, so raw code points beyond 3x the cap are definitely over.
    // Gigantic queries (>1M UTF-16 units) throw on the upper-bound estimate
    // to avoid materializing NFC+fold work (DoS); all normal sizes fall
    // through to the exact post-normalization gate below so `actual` stays exact.
    // Code points are counted without allocating a spread array.
    const rawTrimmed: string = query.trim();
    let forceCpu = false;
    if (rawTrimmed.length > QUERY_TOKENS_MAX * 4) {
      if (rawTrimmed.length > 1_000_000) {
        let cpCount = 0;
        for (const _ch of rawTrimmed) cpCount++;
        // Upper-bound estimate (fold expands at most 1->3): `actual` on this
        // gigantic-query path is an estimate, not the exact post-normalization count
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
    // query-length gate on exact post-normalization token count (shared pipeline).
    const normalizedQuery = normalizeText(query, this.normalized);
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

    /**
     * assemble response diagnostics (undefined unless requested).
     * The string index has no columnar filters, so `filterSelectivity` is
     * always 1.0 and `filteringMs` is always 0; `highlightMs` is 0 (no
     * highlight enrichment on this path); `facetingMs` is never emitted
     * (string index has no facets — intentional key absence).
     */
    const buildDiagnostics = (
      routedEngine: EngineType,
      scoringMs: number,
      hasOverflow: boolean
    ): QueryDiagnostics | undefined => {
      if (!wantsDiagnostics) return undefined;
      const scanned = this.items.length;
      const diag: QueryDiagnostics = {
        scannedCandidates: scanned,
        filterSelectivity: 1.0,
        routedEngine,
        hasOverflow,
        timings: {
          filteringMs: 0,
          scoringMs,
          highlightMs: 0,
          totalMs: nowMs() - queryStartMs
        }
      };
      if (diagWarnings.length > 0) diag.warnings = [...diagWarnings];
      return diag;
    };
    /**
     * post-hoc broad-query + overflow warnings (non-fatal, gated on
     * `diagnostics:true`). The string index has no facets, so the overflow
     * facets clause is suppressed (`facetsRequested:false`); the message
     * names the `RESULT_LIMIT_MAX` pool and remediation.
     */
    const pushPostHocWarnings = (totalMatches: number, hasOverflow: boolean, capacity: number): void => {
      if (!wantsDiagnostics) return;
      const scanned = this.items.length;
      const selectivity = scanned > 0 ? totalMatches / scanned : 0;
      if (isBroadSelectivity(selectivity, scanned)) {
        diagWarnings.push(broadSelectivityWarning(selectivity, scanned));
      }
      if (hasOverflow) {
        diagWarnings.push(
          candidateOverflowWarning(totalMatches, capacity, { facetsRequested: false })
        );
      }
    };

    throwIfAborted(signal);
    throwIfBudgetAborted(budget);

    // Hoisted prefixLength check: fail-closed even on empty queries/corpora
    // (scorePrefixTokens throws per-record, which empty scans would skip).
    // Skipped for empty queries to match the scorer's early noMatch.
    if (mode === 'prefix' && normalizedQuery.tokens.length > 0) {
      assertPrefixLengthForQuery(prefixOpts, normalizedQuery.tokens.length);
    }

    // Degenerate post-processing queries that normalize to zero post-normalization
    // tokens (whitespace-only, U+3000-only, empty) return unified empty
    // results with query:'' on both paths. Lone-mark / VS / ZWJ /
    // tatweel-only inputs survive NFC+C+F as single tokens (pinned §8b), so
    // they are NOT empty here — they search normally echoing the original.
    const noHits = (q: string): SearchResponse => {
      // trivial exits still honor caller aborts and time budgets.
      throwIfAborted(signal);
      throwIfBudgetAborted(budget);
      assertTimeBudget(queryStartMs, budget);
      const diag = buildDiagnostics('cpu', 0, false);
      // Public totalMs is scorer wall-clock (0 on no-hit); diagnostics
      // carries the wall-clock including validation.
      return {
        results: [], totalMatches: 0, query: q, mode, engine: 'cpu',
        candidateCount: 0, hasOverflow: false,
        timings: { queryUploadMs: 0, encodeSubmitMs: 0, gpuExecutionMs: null, readbackMs: 0, totalMs: 0, gpuDispatchMs: 0 },
        profileId: this.profileId, scoringVersion: SCORING_VERSION, cpuScorer,
        fallbackReason: forceCpu ? 'query-too-long' : this.engineState.fallbackReason,
        ...(diag ? { diagnostics: diag } : {})
      };
    };
    if (normalizedQuery.isEmpty) {
      return noHits('');
    }

    // Empty dataset edge case: 0 items (non-empty query echoes original).
    if (this.items.length === 0) {
      return noHits(query);
    }

    // candidate ceiling + broad-query pre-dispatch guard. Short
    // queries over massive corpora route to the CPU streaming scan to avoid
    // GPU buffer saturation and driver timeouts (TDR). The string index has
    // a fixed RESULT_LIMIT_MAX=8192 pool while DocumentIndex scales
    // min(32768, max(8192, …)) — same corpus can overflow here but not there.
    assertCandidateBudget(this.items.length, budget);
    // GPU-eligibility is needed for the warning-suppression decision below,
    // so compute it before the pre-dispatch guard (routing itself stays below).
    const isGpuSupportedMode: boolean =
      (mode === 'fuzzy' || mode === 'substring') && !typo.enabled;
    let broadQueryCpuRoute = false;
    if (isBroadQueryHeuristic(normalizedQuery.tokens.length, this.items.length)) {
      broadQueryCpuRoute = true;
      const cpuByDesign =
        !isGpuSupportedMode || cpuScorer === 'ufuzzy' || forceCpu;
      if (wantsDiagnostics && !cpuByDesign) {
        diagWarnings.push(broadQueryRouteWarning(this.items.length, normalizedQuery.tokens.length));
      }
    }
    assertTimeBudget(queryStartMs, budget);

    // 1. WebGPU execution path: exact fuzzy/substring queries
    // (queryTokenCount <= 128, enforced above) route to WebGPU when
    // available and ready. Token/prefix modes and typo-tolerant queries are
    // CPU-only in (exact-only WGSL kernels) and skip dispatch with
    // fallbackReason 'unsupported-mode'. Explicit ufuzzy and over-limit
    // cpu-fallback force CPU. Failures fall through to the exact CPU
    // scorer below.
    const gpuHandle = this.gpuEngine;
    const useGpu: boolean =
      !forceCpu &&
      !broadQueryCpuRoute &&
      isGpuSupportedMode &&
      cpuScorer !== 'ufuzzy' &&
      this.engineState.engine === 'webgpu' &&
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

        // deadline enforcement + post-hoc warnings before returning.
        throwIfBudgetAborted(budget);
        assertTimeBudget(queryStartMs, budget);
        pushPostHocWarnings(gpuResult.totalMatches, gpuResult.hasOverflow, RESULT_LIMIT_MAX);
        const gpuDiag = buildDiagnostics('webgpu', gpuResult.timings.totalMs, gpuResult.hasOverflow);

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
          cpuScorer,
          ...(gpuDiag ? { diagnostics: gpuDiag } : {})
        };
      } catch (err: any) {
        if (err.name === 'AbortError') {
          throw err;
        }
        // Mode-routed rejections (token/prefix/typo reaching the engine via
        // direct or raced calls) are not engine failures: the healthy GPU
        // stays up and the query falls through to exact CPU below.
        const modeRouted: boolean =
          err instanceof IncompatibleOptionError && (err.option === 'mode' || err.option === 'typoTolerance');
        if (!modeRouted) {
          console.warn('[webgpu-search] GPU search failed, CPU fallback:', err);
          this.engineState = transitionEngineState(this.engineState, 'cpu', 'gpu-execution-error');
          if (this.gpuEngine) {
            try { this.gpuEngine.destroy(); } catch {}
            this.gpuEngine = null;
          }
        }
      }
    }

    // 2. CPU execution path.
    // uFuzzy is quarantined to explicit cpuScorer:'ufuzzy' (CPU-only,
    // never in the differential matrix; explicitly non-conforming scores).
    // Default 'exact' serves the shared-pipeline reference scorer; GPU
    // failures also land here with identical exact semantics.
    throwIfAborted(signal);
    throwIfBudgetAborted(budget);
    assertTimeBudget(queryStartMs, budget);

    if (cpuScorer === 'ufuzzy') {
      let legacyResult: {
        query: string;
        totalMatches: number;
        results: SearchResultItem[];
        durationMs: number;
      };
      if (mode === 'fuzzy') {
        legacyResult = this.cpuEngine.searchWithUFuzzy(this.items, query, clampedLimit, caseSensitive);
      } else {
        legacyResult = this.cpuEngine.searchNaiveScan(this.items, query, clampedLimit, caseSensitive);
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
      // deadline enforcement + post-hoc warnings before returning.
      throwIfBudgetAborted(budget);
      assertTimeBudget(queryStartMs, budget);
      const legacyHasOverflow = legacyResult.totalMatches > RESULT_LIMIT_MAX;
      pushPostHocWarnings(legacyResult.totalMatches, legacyHasOverflow, RESULT_LIMIT_MAX);
      const legacyDiag = buildDiagnostics('cpu', legacyResult.durationMs, legacyHasOverflow);
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
        cpuScorer,
        fallbackReason: 'cpu-algorithm-requested',
        ...(legacyDiag ? { diagnostics: legacyDiag } : {})
      };
    }

    const parityResult = scoreExactMatches(
      this.recordTokens,
      normalizedQuery.tokens,
      mode,
      clampedLimit,
      this.items,
      {
        tokenMatch: { operator: tokenOpts.operator, minMatchCount: tokenOpts.minMatchCount },
        prefixMatch: prefixOpts.prefixLength !== undefined
          ? { prefixLength: prefixOpts.prefixLength, exactCase: prefixOpts.exactCase }
          : { exactCase: prefixOpts.exactCase },
        typoTolerance: {
          enabled: typo.enabled,
          maxDistance: typo.maxDistance,
          minWordLengthForOneTypo: typo.minWordLengthForOneTypo,
          minWordLengthForTwoTypos: typo.minWordLengthForTwoTypos,
          prefixExactLength: typo.prefixExactLength
        }
      },
    );

    throwIfAborted(signal);
    throwIfBudgetAborted(budget);
    assertTimeBudget(queryStartMs, budget);

    const timings: SearchTimings = {
      queryUploadMs: 0,
      encodeSubmitMs: 0,
      gpuExecutionMs: null,
      readbackMs: 0,
      totalMs: parityResult.durationMs,
      gpuDispatchMs: 0
    };

    let effectiveFallbackReason = this.engineState.fallbackReason;
    if (forceCpu) {
      effectiveFallbackReason = 'query-too-long';
    } else if (!isGpuSupportedMode) {
      // token/prefix/typo queries are CPU-by-design (exact-only
      // WGSL kernels) — recorded per the exact boundary.
      effectiveFallbackReason = 'unsupported-mode';
    } else if (useGpu && gpuHandle !== null) {
      effectiveFallbackReason = 'gpu-execution-error';
    }
    // broad-query CPU routing is a routing decision (like the exact
    // note above for exhausted GPU), not a scorer request — surfaced in
    // diagnostics.warnings instead of fallbackReason.

    // deadline enforcement + post-hoc warnings before returning.
    const parityHasOverflow = parityResult.totalMatches > RESULT_LIMIT_MAX;
    pushPostHocWarnings(parityResult.totalMatches, parityHasOverflow, RESULT_LIMIT_MAX);
    const parityDiag = buildDiagnostics('cpu', parityResult.durationMs, parityHasOverflow);

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
      cpuScorer,
      fallbackReason: effectiveFallbackReason,
      ...(parityDiag ? { diagnostics: parityDiag } : {})
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
      engine: this.engineState.engine,
      vramAllocatedBytes: this.vramAllocatedBytes,
      adapterVendor: adapter?.vendor,
      adapterRenderer: adapter?.renderer,
      profileId: this.profileId,
      unicodeVersion: UNICODE_VERSION,
      scoringVersion: SCORING_VERSION,
      tokenCount: this.tokenCount,
      normalized: this.normalized,
      formatVersion: DATASET_FORMAT_VERSION,
      fallbackReason: this.engineState.fallbackReason,
      memory: {
        vramBytes: this.vramAllocatedBytes,
        ramBytes,
        totalBytes: this.vramAllocatedBytes + ramBytes
      }
    };
  }

  /**
   * Explicit device-loss rebuild path (Phase 2 reliability proof).
   *
   * After a `fallbackReason: 'device-lost'` (or `gpu-execution-error`)
   * transition, attempts to re-acquire a GPU device and re-upload the
   * in-memory corpus (`recordTokens` are retained across loss — only GPU
   * buffers are torn down). On success the engine returns to `'webgpu'`
   * with identical match semantics (same packed tokens, same scorer);
   * on failure the index stays on CPU with an explicit `fallbackReason`
   * and returns `false` (never throws for GPU-unavailable).
   *
   * - `preferGpu: false` indexes are CPU-by-design: rebuild is a no-op
   *   returning `false`.
   * - Pass a fresh injected `device` after loss (e.g. a new mock device
   *   in headless tests); otherwise the creation-time device (if any) is
   *   reused, else the shared pool is tried.
   * - Re-subscribes the single device-loss listener (no leak on cycles).
   * - Throws only when the index itself is destroyed.
   */
  async rebuildGpu(options?: { device?: GPUDevice; powerPreference?: GPUPowerPreference }): Promise<boolean> {
    if (this.isDestroyed) {
      throw new Error('[webgpu-search] SearchIndex has been destroyed.');
    }
    if (this.gpuEngine !== null && this.gpuEngine.isReady && this.engineState.engine === 'webgpu') {
      return true;
    }
    if (this.preferGpu === false) {
      return false;
    }
    if (this.items.length === 0) {
      return false;
    }
    const deviceToUse = options?.device ?? this.creationDevice;
    const powerPreference = options?.powerPreference ?? this.creationPowerPreference;
    assertValidPowerPreference(powerPreference);
    if (this.gpuEngine) {
      try { this.gpuEngine.destroy(); } catch {}
      this.gpuEngine = null;
    }
    this.vramAllocatedBytes = 0;
    try {
      const gpu = new WebGPUEngine();
      const initialized = await gpu.init(
        deviceToUse !== undefined || powerPreference !== undefined
          ? { device: deviceToUse, powerPreference }
          : undefined
      );
      if (!initialized || !gpu.isReady) {
        this.engineState = transitionEngineState(
          this.engineState,
          'cpu',
          typeof navigator === 'undefined' || !('gpu' in navigator) || !navigator.gpu
            ? 'webgpu-unsupported'
            : 'device-request-failed',
        );
        try { gpu.destroy(); } catch {}
        return false;
      }
      const packed = packDataset(this.recordTokens, { normalized: this.normalized, totalTokens: this.tokenCount });
      await gpu.loadDataset(packed);
      this.gpuEngine = gpu;
      this.engineState = transitionEngineState(this.engineState, 'webgpu');
      this.vramAllocatedBytes = packed.recordsByteLength + packed.offsetsByteLength;
      if (deviceToUse !== undefined) {
        this.creationDevice = deviceToUse;
      }
      if (powerPreference !== undefined) {
        this.creationPowerPreference = powerPreference;
      }
      this.subscribeDeviceLost();
      return true;
    } catch {
      this.engineState = transitionEngineState(
        this.engineState,
        'cpu',
        typeof navigator === 'undefined' || !('gpu' in navigator) || !navigator.gpu
          ? 'webgpu-unsupported'
          : 'device-request-failed',
      );
      return false;
    }
  }

  /**
   * Release all GPU and context resources. Fail-closed: clears CPU residency
   * (`items`/`recordTokens`/`tokenCount` reset, engine drops to cpu) and
   * subsequent `search()` throws; post-destroy `getStats()` reads 0/empty.
   * Idempotent: double-destroy is safe and unsubscribes exactly once.
   */
  destroy(): void {
    if (this.isDestroyed) {
      return;
    }
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
    this.engineState = transitionEngineState(this.engineState, 'cpu', this.engineState.fallbackReason);
    this.vramAllocatedBytes = 0;
  }

  [Symbol.dispose](): void {
    this.destroy();
  }
}

