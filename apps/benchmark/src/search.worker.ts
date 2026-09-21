import {
  WebGPUEngine,
  CPUEngine,
  scoreExactMatches,
  normalizeText,
  packDataset,
  deserializeDataset,
  type SearchResult,
  type CPUSearchResult,
  type PackedDataset,
  type SearchMode
} from 'webgpu-search';

let gpuEngine: WebGPUEngine | null = null;
let cpuEngine: CPUEngine | null = null;
let datasetStrings: string[] = [];
let datasetRecordTokens: Uint32Array[] = [];
let datasetSize = 0;
// Monotonic dataset generation: bumped only on successful LOAD_DATASET
// commit. SEARCH_RESULTS echoes it so the main thread can drop hits that
// belong to a stale dataset (load vs in-flight search race).
let datasetGeneration = 0;
let latestQueryId = 0;
let activeAbortController: AbortController | null = null;

/**
 * String-isolated enrichment.
 *
 * When true the worker loads token-only datasets into the GPU engine (no
 * `strings` duplication in VRAM residency) and returns compact
 * `{index,score}[]` hits on the SEARCH path; the main thread maps `text`
 * from its own `currentDataset.strings` copy. This removes the per-keystroke
 * worker->main text clone: a 1000-hit full `SearchResult` carries ~50 KB of
 * string payload back, the compact form carries `1000 x 8 B ~ 8 KB`
 * (index+score only) -- ~6x smaller structured-clone per keystroke
 * (browser-measured confirmation is pending-hardware; the byte math is
 * structural: `hits x avgTextLen` vs `hits x 8`. Note: per-object
 * structured-clone overhead means `{index,score}` objects cost more than
 * 8 B on the wire and text length is UTF-16 units, not bytes -- treat the
 * ratio as order-of-magnitude, not exact).
 *
 * Toggle to `false` for the legacy full-text return path when doing
 * before/after measurement. LOAD_DATASET still receives `strings` (needed
 * for in-worker CPU comparison); the GPU load itself is token-only only
 * when the flag is true.
 */
export const STRING_ISOLATED_ENRICHMENT = true;

export interface CompactHit {
  index: number;
  score: number;
}

export interface WorkerGpuMeta {
  query: string;
  mode: SearchMode;
  engine: 'webgpu';
  totalMatches: number;
  candidateCount: number;
  hasOverflow: boolean;
  timings: SearchResult['timings'];
}

function countChars(strings: readonly string[]): number {
  let n = 0;
  for (let i = 0; i < strings.length; i++) n += (strings[i] ?? '').length;
  return n;
}

self.onmessage = async (e: MessageEvent) => {
  const { type, payload } = e.data;

  if (type === 'INIT') {
    gpuEngine = new WebGPUEngine();
    cpuEngine = new CPUEngine();
    let isSupported = false;

    try {
      isSupported = await gpuEngine.init();
    } catch (err) {
      console.warn('[search.worker] WebGPU initialization failed:', err);
    }

    self.postMessage({
      type: 'INIT_DONE',
      payload: {
        isSupported,
        adapterInfo: gpuEngine?.adapterInfo ?? null
      }
    });
    return;
  }

  if (type === 'LOAD_DATASET') {
    const strings: string[] | undefined = payload?.strings;
    const serialized: ArrayBuffer | undefined = payload?.serialized;
    const sentTimestamp: number | undefined = typeof payload?.sentTimestamp === 'number' ? payload.sentTimestamp : undefined;
    const transferLatencyMs = sentTimestamp !== undefined ? performance.now() - sentTimestamp : undefined;
    // Adopt the main-thread epoch when provided (stale LOADs with an older
    // epoch are dropped to avoid out-of-order commit going backwards).
    // Harness LOADs omit it, in which case the worker bumps its own counter.
    const incomingGeneration: number | undefined =
      typeof payload?.datasetGeneration === 'number' ? payload.datasetGeneration : undefined;
    if (incomingGeneration !== undefined && incomingGeneration < datasetGeneration) {
      self.postMessage({
        type: 'DATASET_LOADED',
        payload: {
          size: datasetSize,
          uploadTimeMs: 0,
          packMs: 0,
          datasetGeneration,
          error: `[search.worker] stale LOAD_DATASET generation ${incomingGeneration} < ${datasetGeneration}; dropped.`
        }
      });
      return;
    }

    // Fail fast on legacy byte buffers. The engine
    // silently ignores `recordsBufferData`/`offsetsBufferData` whenever
    // `strings` is present and repacks from `strings` -- so sending them is
    // pure structured-clone waste plus a dead packer. Reject explicitly so
    // callers migrate to `{strings}` / `{serialized}` (dataset) instead of
    // silently paying the clone. Rejected even when combined with new
    // fields: the combo still pays the exact clone waste this guard exists
    // to kill.
    if (
      payload?.recordsBufferData !== undefined ||
      payload?.offsetsBufferData !== undefined ||
      payload?.gpuBufferData !== undefined
    ) {
      self.postMessage({
        type: 'DATASET_LOADED',
        payload: {
          size: 0,
          uploadTimeMs: 0,
          packMs: 0,
          datasetGeneration,
          error:
            '[search.worker] legacy byte buffers rejected (no dataset magic). ' +
            'Send {strings} or {serialized} (packDataset/serialize). Rebuild required.'
        }
      });
      return;
    }

    // Preferred path: dataset buffer (transferable; the `.slice(0)`
    // copy in main.ts is moved with a transfer list -- copy-then-move, not
    // zero-copy, since the original is retained for reuse. `strings` may
    // ride along for in-worker CPU comparison.
    if (serialized !== undefined) {
      // Neutered-buffer guard: a transferred-then-reused or detached buffer
      // reports byteLength 0 -- explicit re-create path, never a silent empty
      // index (matches library `deserializeDataset` fail-closed rule).
      // Duck-typed (byteLength number + slice function) to match the
      // library cross-realm rule; foreign-realm ArrayBuffers are accepted,
      // non-buffers and neutered views are rejected.
      const asBuf = serialized as unknown as { byteLength?: unknown; slice?: unknown };
      const byteLen: number =
        typeof asBuf?.byteLength === 'number' && typeof asBuf?.slice === 'function'
          ? (asBuf.byteLength as number)
          : 0;
      if (byteLen === 0) {
        self.postMessage({
          type: 'DATASET_LOADED',
          payload: {
            size: 0,
            uploadTimeMs: 0,
            packMs: 0,
            datasetGeneration,
            error:
              '[search.worker] neutered serialized buffer (byteLength 0 post-transfer). Re-create required.'
          }
        });
        return;
      }
      const t0 = performance.now();
      try {
        const packed = deserializeDataset(serialized);
        const packMs = performance.now() - t0;
        const nextStrings = Array.isArray(strings) ? strings : [];
        const nextSize = packed.rowCount;
        const nextTokens = new Array<Uint32Array>(packed.rowCount);
        for (let i = 0; i < packed.rowCount; i++) {
          nextTokens[i] = packed.tokens.subarray(packed.offsets[i]!, packed.offsets[i + 1]!);
        }
        let uploadTimeMs = 0;
        if (gpuEngine?.isReady) {
          try {
            // Honor STRING_ISOLATED_ENRICHMENT on this path too: false means
            // full-text load when strings are available (before/after
            // measurement toggle); otherwise token-only packed load.
            const loadable =
              STRING_ISOLATED_ENRICHMENT || nextStrings.length === 0
                ? packed
                : { size: nextStrings.length, strings: nextStrings };
            const res = await gpuEngine.loadDataset(loadable);
            uploadTimeMs = res.uploadTimeMs;
          } catch (loadErr) {
            console.error('[search.worker] GPU loadDataset failed:', loadErr);
            // Fail-closed: keep the previous dataset (do not commit partial
            // state) so subsequent SEARCH hits stay consistent with the GPU.
            self.postMessage({
              type: 'DATASET_LOADED',
              payload: {
                size: datasetSize,
                uploadTimeMs,
                packMs,
                serializedBytes: byteLen,
                stringsChars: countChars(nextStrings),
                datasetGeneration,
                error: String(loadErr)
              }
            });
            return;
          }
        }
        // Commit only after successful deserialize + GPU load.
        datasetStrings = nextStrings;
        datasetRecordTokens = nextTokens;
        datasetSize = nextSize;
        datasetGeneration =
          incomingGeneration !== undefined ? incomingGeneration : datasetGeneration + 1;
        self.postMessage({
          type: 'DATASET_LOADED',
          payload: {
            size: datasetSize,
            uploadTimeMs,
            packMs,
            serializedBytes: byteLen,
            stringsChars: countChars(datasetStrings),
            tokenCount: packed.tokenCount,
            normalized: packed.normalized,
            folded: packed.folded,
            transferLatencyMs: transferLatencyMs !== undefined ? Number(transferLatencyMs.toFixed(2)) : undefined,
            datasetGeneration
          }
        });
      } catch (err) {
        // deserializeDataset throws IncompatibleIndexError on
        // magic/version/checksum/shape failures -- surface, don't swallow.
        self.postMessage({
          type: 'DATASET_LOADED',
          payload: {
            size: 0,
            uploadTimeMs: 0,
            packMs: performance.now() - t0,
            datasetGeneration,
            error: String(err)
          }
        });
      }
      return;
    }

    // Fallback path: raw strings (structured-clone cost is real -- the array
    // is fully copied into the worker; see DATASET_LOADED `stringsChars`).
    // Packs via the unicode pipeline (the legacy `packStringsToGPUBuffer`
    // ASCII-mangling packer is deleted from this flow).
    // Fail-closed: missing/non-array `strings` (with no `serialized`) never
    // wipes a good index with an empty success. An explicit `[]` is allowed
    // (clears the index); anything else posts an error and keeps state.
    if (!Array.isArray(strings)) {
      self.postMessage({
        type: 'DATASET_LOADED',
        payload: {
          size: datasetSize,
          uploadTimeMs: 0,
          packMs: 0,
          datasetGeneration,
          error: '[search.worker] LOAD_DATASET requires {strings: string[]} or {serialized: ArrayBuffer}. Got neither; keeping previous dataset.'
        }
      });
      return;
    }
    const list: string[] = strings;
    // Always pack for metrics, even when the GPU is not ready (CPU-only
    // worker still reports tokenCount instead of a misleading 0).
    const t0 = performance.now();
    let packedForMetrics: PackedDataset | null = null;
    try {
      packedForMetrics = packDataset(list, { normalized: true });
    } catch (packErr) {
      self.postMessage({
        type: 'DATASET_LOADED',
        payload: {
          size: datasetSize,
          uploadTimeMs: 0,
          packMs: performance.now() - t0,
          datasetGeneration,
          error: String(packErr)
        }
      });
      return;
    }
    const packMs = performance.now() - t0;
    const tokenCount = packedForMetrics.tokenCount;
    let uploadTimeMs = 0;
    if (gpuEngine?.isReady) {
      try {
        // String-isolated: token-only load (engine resolves `text` to ``;
        // SEARCH returns compact hits, main enriches). Legacy path loads
        // `strings` into the engine for full-text returns.
        const loadable = STRING_ISOLATED_ENRICHMENT
          ? packedForMetrics
          : { size: list.length, strings: list };
        const res = await gpuEngine.loadDataset(loadable);
        uploadTimeMs = res.uploadTimeMs;
      } catch (loadErr) {
        console.error('[search.worker] GPU loadDataset failed:', loadErr);
        // Keep previous dataset on failure; surface the error.
        self.postMessage({
          type: 'DATASET_LOADED',
          payload: {
            size: datasetSize,
            uploadTimeMs: 0,
            packMs,
            stringsChars: countChars(list),
            tokenCount,
            stringIsolated: STRING_ISOLATED_ENRICHMENT,
            datasetGeneration,
            error: String(loadErr)
          }
        });
        return;
      }
    }

    // Commit only after successful pack + GPU load.
    datasetStrings = list;
    datasetSize = list.length;
    if (packedForMetrics) {
      const nextTokens = new Array<Uint32Array>(packedForMetrics.rowCount);
      for (let i = 0; i < packedForMetrics.rowCount; i++) {
        nextTokens[i] = packedForMetrics.tokens.subarray(packedForMetrics.offsets[i]!, packedForMetrics.offsets[i + 1]!);
      }
      datasetRecordTokens = nextTokens;
    }
    datasetGeneration =
      incomingGeneration !== undefined ? incomingGeneration : datasetGeneration + 1;

    self.postMessage({
      type: 'DATASET_LOADED',
      payload: {
        size: datasetSize,
        uploadTimeMs,
        packMs,
        stringsChars: countChars(list),
        tokenCount,
        stringIsolated: STRING_ISOLATED_ENRICHMENT,
        transferLatencyMs: transferLatencyMs !== undefined ? Number(transferLatencyMs.toFixed(2)) : undefined,
        datasetGeneration
      }
    });
    return;
  }

  if (type === 'SEARCH') {
    const { queryId, query, mode, limit = 1000, runCpuComparison = false, stringIsolated } = payload ?? {};
    const isStringIsolated = typeof stringIsolated === 'boolean' ? stringIsolated : STRING_ISOLATED_ENRICHMENT;
    const requestGeneration: number | undefined =
      typeof payload?.datasetGeneration === 'number' ? payload.datasetGeneration : undefined;

    // Fail-closed payload validation: a malformed queryId must post
    // SEARCH_ERROR and must never poison latestQueryId (otherwise all
    // future `<` comparisons go false and the drop logic dies).
    if (!Number.isInteger(queryId) || (queryId as number) < 0) {
      self.postMessage({
        type: 'SEARCH_ERROR',
        payload: {
          queryId: typeof queryId === 'number' ? queryId : null,
          query: typeof query === 'string' ? query : '',
          datasetGeneration,
          error: '[search.worker] SEARCH requires integer queryId >= 0.'
        }
      });
      return;
    }

    if (queryId < latestQueryId) {
      return; // Superseded by a newer query
    }
    latestQueryId = queryId;

    if (activeAbortController) {
      activeAbortController.abort();
    }
    activeAbortController = new AbortController();
    const signal = activeAbortController.signal;

    let gpuResult: SearchResult | null = null;
    let gpuCompact: CompactHit[] | null = null;
    let gpuMeta: WorkerGpuMeta | null = null;
    let ufuzzyResult: CPUSearchResult | null = null;
    let nativeResult: CPUSearchResult | null = null;
    let workerSerializationMs = 0;

    try {
      // Validate mode upfront (independent of engine readiness) so invalid
      // modes surface as SEARCH_ERROR instead of silent empty results.
      if (mode !== 'substring' && mode !== 'fuzzy') {
        throw new TypeError(`[search.worker] SEARCH mode must be 'substring'|'fuzzy', got ${String(mode)}.`);
      }
      // Validate compact-hit bounds lazily at enrich time in main; here just
      // ensure limit is sane via the shared clamp (engine clamps again).
      if (gpuEngine?.isReady && query) {
        const full = await gpuEngine.search(query, {
          mode,
          limit,
          signal
        });
        const tSer0 = performance.now();
        if (isStringIsolated) {
          // Strip `text` at the boundary: the main thread owns display
          // strings and re-attaches them by index (same contract as
          // SearchIndex enrichment over token-only engine results).
          // Validate indices before posting (defensive: engine is trusted,
          // but a corrupt index must not become a main-thread OOB read).
          for (const r of full.results) {
            if (!Number.isInteger(r.index) || r.index < 0 || r.index >= datasetSize) {
              throw new Error(`[search.worker] compact hit index out of bounds: ${r.index} (size ${datasetSize}).`);
            }
          }
          gpuCompact = full.results.map((r) => ({ index: r.index, score: r.score }));
          gpuMeta = {
            query: full.query,
            mode: full.mode,
            engine: 'webgpu',
            totalMatches: full.totalMatches,
            candidateCount: full.candidateCount,
            hasOverflow: full.hasOverflow,
            timings: full.timings
          };
        } else {
          gpuResult = full;
        }
        workerSerializationMs = performance.now() - tSer0;
      }

      if (signal.aborted || queryId < latestQueryId) {
        return;
      }

      let parityResult: { durationMs: number; totalMatches: number; results: any[] } | null = null;
      if (runCpuComparison && query) {
        if (datasetRecordTokens.length > 0) {
          const norm = normalizeText(query, true);
          parityResult = scoreExactMatches(
            datasetRecordTokens,
            norm.tokens,
            mode,
            limit,
            datasetStrings
          );
        }
        if (cpuEngine && datasetStrings.length > 0) {
          ufuzzyResult = cpuEngine.searchUFuzzy(datasetStrings, query, limit);
          nativeResult = cpuEngine.searchNative(datasetStrings, query, limit);
        }
      }

      if (signal.aborted || queryId < latestQueryId) {
        return;
      }

      self.postMessage({
        type: 'SEARCH_RESULTS',
        payload: {
          queryId,
          query,
          datasetGeneration,
          requestGeneration,
          gpuResult,
          gpuCompact,
          gpuMeta,
          parityResult,
          ufuzzyResult,
          nativeResult,
          workerPostTimestamp: performance.now(),
          workerSerializationMs: Number(workerSerializationMs.toFixed(2)),
          stringIsolated: isStringIsolated
        }
      });
    } catch (err: unknown) {
      if ((err as { name?: string })?.name === 'AbortError') {
        return; // Expected abort on rapid user keystrokes
      }
      console.error('[search.worker] Search error:', err);
      self.postMessage({
        type: 'SEARCH_ERROR',
        payload: {
          queryId,
          query: typeof query === 'string' ? query : '',
          datasetGeneration,
          error: String(err)
        }
      });
    }
    return;
  }
};
