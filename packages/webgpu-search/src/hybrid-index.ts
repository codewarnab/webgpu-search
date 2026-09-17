import { WebGPUEngine } from './webgpu-engine';
import { CPUEngine } from './cpu-engine';
import { WebGPUContextManager } from './context-manager';
import { packStringsToGPUBuffer, checkMemoryBudget } from './buffer';
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

  private constructor(items: string[]) {
    this.items = items;
    this.cpuEngine = new CPUEngine();
  }

  /**
   * Factory method to create a SearchIndex with dynamic routing between WebGPU and CPU.
   */
  static async create(items: string[], options: IndexOptions = {}): Promise<SearchIndex> {
    const index = new SearchIndex(items);

    // Empty dataset fast path: zero allocation, route immediately to CPU
    if (items.length === 0) {
      index.engineType = 'cpu';
      return index;
    }

    const threshold = options.threshold ?? 30_000;
    const preferGpu = options.preferGpu ?? false;
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
    const { mode = 'fuzzy', caseSensitive = false, signal } = options;
    const requestedLimit = options.limit ?? options.maxResults ?? 50;
    // Guard against NaN (e.g. a failed parseInt): Math.min/Math.max propagate
    // NaN, which would poison downstream slicing. Treat it as "not provided".
    const clampedLimit = Math.max(1, Math.min(Number.isNaN(requestedLimit) ? 50 : requestedLimit, 8192));

    if (signal?.aborted) {
      throw new DOMException('Search aborted', 'AbortError');
    }

    // Empty dataset edge case: 0 items
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
        timings
      };
    }

    // 1. WebGPU execution path
    if (this.engineType === 'webgpu' && this.gpuEngine && this.gpuEngine.isReady) {
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
          timings: gpuResult.timings
        };
      } catch (err: any) {
        if (err.name === 'AbortError') {
          throw err;
        }
        console.warn('[webgpu-search] GPU search failed, falling back to CPU for this query:', err);
        // Fallthrough to CPU
      }
    }

    // 2. CPU execution path
    if (signal?.aborted) {
      throw new DOMException('Search aborted', 'AbortError');
    }

    let cpuResult: {
      query: string;
      totalMatches: number;
      results: SearchResultItem[];
      durationMs: number;
    };

    if (mode === 'fuzzy') {
      cpuResult = this.cpuEngine.searchUFuzzy(this.items, query, clampedLimit, caseSensitive);
    } else {
      cpuResult = this.cpuEngine.searchNative(this.items, query, clampedLimit, caseSensitive);
    }

    if (signal?.aborted) {
      throw new DOMException('Search aborted', 'AbortError');
    }

    const timings: SearchTimings = {
      queryUploadMs: 0,
      encodeSubmitMs: 0,
      gpuExecutionMs: null,
      readbackMs: 0,
      totalMs: cpuResult.durationMs,
      gpuDispatchMs: 0
    };

    return {
      query: cpuResult.query,
      mode,
      engine: 'cpu',
      totalMatches: cpuResult.totalMatches,
      candidateCount: Math.min(cpuResult.totalMatches, 8192),
      hasOverflow: false,
      results: cpuResult.results,
      timings
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
      adapterRenderer: adapter?.renderer
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

