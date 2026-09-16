import {
  WebGPUEngine,
  CPUEngine,
  packStringsToGPUBuffer,
  type SearchResult,
  type CPUSearchResult
} from 'webgpu-search';

let gpuEngine: WebGPUEngine | null = null;
let cpuEngine: CPUEngine | null = null;
let datasetStrings: string[] = [];
let latestQueryId = 0;
let activeAbortController: AbortController | null = null;

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
    const { strings, recordsBufferData, offsetsBufferData } = payload;
    datasetStrings = strings;

    let uploadTimeMs = 0;
    if (gpuEngine?.isReady) {
      let recordsData: ArrayBuffer = recordsBufferData;
      let offsetsData: ArrayBuffer = offsetsBufferData;
      let byteLength = (recordsData?.byteLength ?? 0) + (offsetsData?.byteLength ?? 0);

      if (!recordsData || !offsetsData) {
        const packed = packStringsToGPUBuffer(strings);
        recordsData = packed.recordsBufferData;
        offsetsData = packed.offsetsBufferData;
        byteLength = packed.byteLength;
      }

      try {
        const res = await gpuEngine.loadDataset({
          size: strings.length,
          strings,
          recordsBufferData: recordsData,
          recordsByteLength: recordsData.byteLength,
          offsetsBufferData: offsetsData,
          offsetsByteLength: offsetsData.byteLength,
          gpuBufferData: recordsData,
          byteLength
        });
        uploadTimeMs = res.uploadTimeMs;
      } catch (loadErr) {
        console.error('[search.worker] GPU loadDataset failed:', loadErr);
      }
    }

    self.postMessage({
      type: 'DATASET_LOADED',
      payload: {
        size: strings.length,
        uploadTimeMs
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
    let ufuzzyResult: CPUSearchResult | null = null;
    let nativeResult: CPUSearchResult | null = null;

    try {
      if (gpuEngine?.isReady && query) {
        gpuResult = await gpuEngine.search(query, {
          mode,
          limit,
          signal
        });
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
          ufuzzyResult,
          nativeResult
        }
      });
    } catch (err: any) {
      if (err?.name === 'AbortError') {
        return; // Expected abort on rapid user keystrokes
      }
      console.error('[search.worker] Search error:', err);
    }
    return;
  }
};
