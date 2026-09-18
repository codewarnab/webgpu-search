import {
  WebGPUEngine,
  CPUEngine,
  packUnicodeToGPUBuffer,
  deserializeUnicodeDataset,
  type SearchResult,
  type CPUSearchResult
} from 'webgpu-search';

let gpuEngine: WebGPUEngine | null = null;
let cpuEngine: CPUEngine | null = null;
let datasetStrings: string[] = [];
let datasetSize = 0;
let latestQueryId = 0;
let activeAbortController: AbortController | null = null;

/**
 * M4 string-isolated enrichment (Issue #7).
 *
 * When true the worker loads token-only datasets into the GPU engine (no
 * `strings` duplication in VRAM residency) and returns compact
 * `{index,score}[]` hits on the SEARCH path; the main thread maps `text`
 * from its own `currentDataset.strings` copy. This removes the per-keystroke
 * worker→main text clone: a 1000-hit full `SearchResult` carries ~50 KB of
 * string payload back, the compact form carries `1000 × 8 B ≈ 8 KB`
 * (index+score only) — ~6× smaller structured-clone per keystroke
 * (browser-measured confirmation is pending-hardware; the byte math is
 * structural: `hits × avgTextLen` vs `hits × 8`).
 *
 * Toggle to `false` for the legacy full-text return path when doing
 * before/after measurement. LOAD_DATASET still receives `strings` (needed
 * for in-worker CPU comparison); the GPU load itself is token-only.
 */
export const STRING_ISOLATED_ENRICHMENT = true;

export interface CompactHit {
  index: number;
  score: number;
}

export interface WorkerGpuMeta {
  query: string;
  mode: 'substring' | 'fuzzy';
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

    // Fail fast on legacy v0.1 byte buffers (M4 blocker fix). The M3 engine
    // silently ignores `recordsBufferData`/`offsetsBufferData` whenever
    // `strings` is present and repacks from `strings` — so sending them is
    // pure structured-clone waste plus a dead packer. Reject explicitly so
    // callers migrate to `{strings}` / `{serialized}` (U2F2) instead of
    // silently paying the clone.
    if (
      strings === undefined &&
      serialized === undefined &&
      (payload?.recordsBufferData !== undefined ||
        payload?.offsetsBufferData !== undefined ||
        payload?.gpuBufferData !== undefined)
    ) {
      self.postMessage({
        type: 'DATASET_LOADED',
        payload: {
          size: 0,
          uploadTimeMs: 0,
          packMs: 0,
          error:
            '[search.worker] legacy v0.1 byte buffers rejected (no U2F2 magic). ' +
            'Send {strings} or {serialized} (packUnicodeToGPUBuffer/serialize). Rebuild required.'
        }
      });
      return;
    }

    // Preferred path: U2F2 serialized buffer (transferable, zero-copy with a
    // transfer list). `strings` may ride along for in-worker CPU comparison.
    if (serialized !== undefined) {
      // Neutered-buffer guard: a transferred-then-reused or detached buffer
      // reports byteLength 0 — explicit re-create path, never a silent empty
      // index (matches library `deserializeUnicodeDataset` fail-closed rule).
      const byteLen: number =
        serialized instanceof ArrayBuffer ? serialized.byteLength : 0;
      if (byteLen === 0) {
        self.postMessage({
          type: 'DATASET_LOADED',
          payload: {
            size: 0,
            uploadTimeMs: 0,
            packMs: 0,
            error:
              '[search.worker] neutered serialized buffer (byteLength 0 post-transfer). Re-create required.'
          }
        });
        return;
      }
      const t0 = performance.now();
      try {
        const packed = deserializeUnicodeDataset(serialized);
        const packMs = performance.now() - t0;
        datasetStrings = Array.isArray(strings) ? strings : [];
        datasetSize = packed.rowCount;
        let uploadTimeMs = 0;
        if (gpuEngine?.isReady) {
          try {
            const res = await gpuEngine.loadDataset(packed);
            uploadTimeMs = res.uploadTimeMs;
          } catch (loadErr) {
            console.error('[search.worker] GPU loadDataset failed:', loadErr);
            self.postMessage({
              type: 'DATASET_LOADED',
              payload: {
                size: datasetSize,
                uploadTimeMs,
                packMs,
                serializedBytes: byteLen,
                stringsChars: countChars(datasetStrings),
                error: String(loadErr)
              }
            });
            return;
          }
        }
        self.postMessage({
          type: 'DATASET_LOADED',
          payload: {
            size: datasetSize,
            uploadTimeMs,
            packMs,
            serializedBytes: byteLen,
            stringsChars: countChars(datasetStrings),
            tokenCount: packed.tokenCount,
            folded: packed.folded
          }
        });
      } catch (err) {
        // deserializeUnicodeDataset throws IncompatibleIndexError on
        // magic/version/checksum/shape failures — surface, don't swallow.
        self.postMessage({
          type: 'DATASET_LOADED',
          payload: {
            size: 0,
            uploadTimeMs: 0,
            packMs: performance.now() - t0,
            error: String(err)
          }
        });
      }
      return;
    }

    // Fallback path: raw strings (structured-clone cost is real — the array
    // is fully copied into the worker; see DATASET_LOADED `stringsChars`).
    // Packs via the unicode pipeline (M4; the legacy `packStringsToGPUBuffer`
    // ASCII-mangling packer is deleted from this flow).
    const list: string[] = Array.isArray(strings) ? strings : [];
    datasetStrings = list;
    datasetSize = list.length;

    const t0 = performance.now();
    let packMs = 0;
    let uploadTimeMs = 0;
    let tokenCount = 0;
    if (gpuEngine?.isReady) {
      try {
        const packed = packUnicodeToGPUBuffer(list, { folded: true });
        packMs = performance.now() - t0;
        tokenCount = packed.tokenCount;
        // String-isolated: token-only load (engine resolves `text` to `''`;
        // SEARCH returns compact hits, main enriches). Legacy path loads
        // `strings` into the engine for full-text returns.
        const res = await gpuEngine.loadDataset(
          STRING_ISOLATED_ENRICHMENT ? packed : { size: list.length, strings: list }
        );
        uploadTimeMs = res.uploadTimeMs;
      } catch (loadErr) {
        console.error('[search.worker] GPU loadDataset failed:', loadErr);
      }
    }

    self.postMessage({
      type: 'DATASET_LOADED',
      payload: {
        size: datasetSize,
        uploadTimeMs,
        packMs,
        stringsChars: countChars(list),
        tokenCount,
        stringIsolated: STRING_ISOLATED_ENRICHMENT
      }
    });
    return;
  }

  if (type === 'SEARCH') {
    const { queryId, query, mode, limit = 1000, runCpuComparison = false } = payload;

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

    try {
      if (gpuEngine?.isReady && query) {
        const full = await gpuEngine.search(query, {
          mode,
          limit,
          signal
        });
        if (STRING_ISOLATED_ENRICHMENT) {
          // Strip `text` at the boundary: the main thread owns display
          // strings and re-attaches them by index (same contract as
          // SearchIndex enrichment over token-only engine results).
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
      }

      if (signal.aborted || queryId < latestQueryId) {
        return;
      }

      if (runCpuComparison && cpuEngine && query && datasetStrings.length > 0) {
        ufuzzyResult = cpuEngine.searchUFuzzy(datasetStrings, query, limit);
        nativeResult = cpuEngine.searchNative(datasetStrings, query, limit);
      }

      if (signal.aborted || queryId < latestQueryId) {
        return;
      }

      self.postMessage({
        type: 'SEARCH_RESULTS',
        payload: {
          queryId,
          query,
          gpuResult,
          gpuCompact,
          gpuMeta,
          ufuzzyResult,
          nativeResult
        }
      });
    } catch (err: unknown) {
      if ((err as { name?: string })?.name === 'AbortError') {
        return; // Expected abort on rapid user keystrokes
      }
      console.error('[search.worker] Search error:', err);
    }
    return;
  }
};
