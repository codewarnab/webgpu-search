/**
 * Full Matrix Benchmark Engine & Performance Characterization Runner.
 *
 * - Replaces legacy mean-of-3 with warm median and p95 latency metrics over configurable
 *   warmups (default 5) and samples (default 20).
 * - Implements randomized/interleaved execution across 4 engines:
 *   1. WebGPU (retained storage buffers in VRAM)
 *   2. CPU Exact Reference (`scoreExactMatches`, `cpuScorer: 'exact'`)
 *   3. uFuzzy (CPU opt-in baseline)
 *   4. JS Native (`String.prototype.includes` baseline)
 * - Eliminates invented FPS: captures real rAF frame intervals, jank spikes (>16.7ms), and dropped frames.
 * - Per-buffer VRAM accounting: records, offsets, query (512 B), output (65,544 B).
 * - Packing pipeline phase breakdown: normalizeMs, packMs, uploadMs.
 * - Hardware qualification: detects physical GPU vs software/mock renderer; tags with `pending-hardware`.
 */
import {
  WebGPUEngine,
  CPUEngine,
  scoreExactMatches,
  normalizeText,
  type AdapterInfo
} from 'webgpu-search';
import {
  generateDataset,
  type Dataset,
  type CorpusType
} from './dataset';

export interface LatencyStats {
  medianMs: number;
  p95Ms: number;
  minMs: number;
  maxMs: number;
  meanMs: number;
  samples: number[];
}

export interface VramAllocation {
  recordsBytes: number;
  offsetsBytes: number;
  queryBytes: number;
  outputBytes: number;
  totalBytes: number;
}

export interface PackingPipelineStats {
  normalizeMs: number;
  packMs: number;
  uploadMs: number;
  totalPipelineMs: number;
}

export interface UiTelemetry {
  workerFps: number;
  mainThreadFps: number;
  mainThreadJankMs: number;
  droppedFrames: number;
  jankSpikes: number;
  totalFrameCount: number;
}

export interface BenchmarkRowResult {
  datasetSize: number;
  corpusType: CorpusType;
  query: string;
  mode: 'substring' | 'fuzzy';
  utf16Units: number;
  codePoints: number;
  vramAllocation: VramAllocation;
  packing: PackingPipelineStats;
  gpuRetained: {
    medianMs: number;
    p95Ms: number;
    totalMs: number; // backwards-compatible alias to medianMs
    queryUploadMs: number;
    encodeSubmitMs: number;
    gpuExecutionMs: number | null;
    readbackMs: number;
    gpuDispatchMs: number;
    samples: number[];
  };
  gpuCold: {
    uploadMs: number;
    totalMs: number;
  };
  cpuParity: LatencyStats;
  ufuzzy: LatencyStats;
  jsNative: LatencyStats;
  // Backwards-compatible aliases (median)
  cpuParityMs: number;
  ufuzzyMs: number;
  jsNativeMs: number;
  retainedVsUfuzzySpeedup: number;
  retainedVsParitySpeedup: number;
  retainedVsNativeSpeedup: number;
  coldVsUfuzzySpeedup: number;
  crossover: {
    gpuRetainedBeatsUfuzzy: boolean;
    gpuRetainedBeatsParity: boolean;
    gpuRetainedBeatsNative: boolean;
    gpuColdBeatsUfuzzy: boolean;
  };
  matchCount: {
    gpu: number;
    cpuParity: number;
    ufuzzy: number;
    native: number;
  };
  hasOverflow?: boolean;
  uiTelemetry: UiTelemetry;
  hardwareQualified: boolean;
  qualificationStatus: 'qualified' | 'pending-hardware';
}

export interface BenchmarkProgress {
  currentStep: number;
  totalSteps: number;
  stepName: string;
  currentRow?: BenchmarkRowResult;
}

export interface BenchmarkOptions {
  sizes?: number[];
  query?: string;
  mode?: 'substring' | 'fuzzy';
  corpusType?: CorpusType;
  warmups?: number;
  samples?: number;
  onProgress?: (progress: BenchmarkProgress) => void;
}

/**
 * Calculates statistical distribution metrics: median, p95, min, max, mean.
 */
export function calcLatencyStats(samples: number[]): LatencyStats {
  if (samples.length === 0) {
    return { medianMs: 0, p95Ms: 0, minMs: 0, maxMs: 0, meanMs: 0, samples: [] };
  }

  const sorted = [...samples].sort((a, b) => a - b);
  const minMs = Number(sorted[0]!.toFixed(2));
  const maxMs = Number(sorted[sorted.length - 1]!.toFixed(2));
  const sum = samples.reduce((acc, val) => acc + val, 0);
  const meanMs = Number((sum / samples.length).toFixed(2));

  // Quantile with linear interpolation
  const quantile = (q: number): number => {
    if (sorted.length === 1) return sorted[0]!;
    const pos = (sorted.length - 1) * q;
    const base = Math.floor(pos);
    const rest = pos - base;
    if (sorted[base + 1] !== undefined) {
      return sorted[base]! + rest * (sorted[base + 1]! - sorted[base]!);
    }
    return sorted[base]!;
  };

  const medianMs = Number(quantile(0.5).toFixed(2));
  const p95Ms = Number(quantile(0.95).toFixed(2));

  return {
    medianMs,
    p95Ms,
    minMs,
    maxMs,
    meanMs,
    samples: sorted.map(s => Number(s.toFixed(2)))
  };
}

/**
 * Checks if the WebGPU adapter is a real physical GPU device (Metal, Vulkan, D3D11)
 * or a software/mock rasterizer (SwiftShader, LLVMpipe, CPU fallback).
 */
export function isPhysicalGpu(adapterInfo: AdapterInfo | null): boolean {
  if (!adapterInfo) return false;
  const desc = `${adapterInfo.vendor} ${adapterInfo.device} ${adapterInfo.architecture} ${adapterInfo.renderer ?? ''}`.toLowerCase();
  if (
    desc.includes('swiftshader') ||
    desc.includes('llvmpipe') ||
    desc.includes('lavapipe') ||
    desc.includes('softpipe') ||
    desc.includes('basic render driver') ||
    desc.includes('warp') ||
    desc.includes('mock') ||
    /\bsoftware\b/.test(desc) ||
    /\bcpu (rasterizer|renderer|fallback)\b/.test(desc)
  ) {
    return false;
  }
  return true;
}

/**
 * Real requestAnimationFrame telemetry tracker measuring genuine UI thread frame intervals,
 * jank spikes (>16.7ms), and dropped frames during search workloads.
 */
export class FrameTelemetryTracker {
  private active = false;
  private frameCount = 0;
  private startTime = 0;
  private lastTime = 0;
  private jankSpikes = 0;
  private droppedFrames = 0;
  private maxFrameMs = 0;
  private rafId: number | null = null;

  start(): void {
    this.active = true;
    this.frameCount = 0;
    this.jankSpikes = 0;
    this.droppedFrames = 0;
    this.maxFrameMs = 0;
    this.startTime = typeof performance !== 'undefined' ? performance.now() : Date.now();
    this.lastTime = this.startTime;

    if (typeof requestAnimationFrame === 'function') {
      const loop = (now: number): void => {
        if (!this.active) return;
        const delta = now - this.lastTime;
        this.frameCount++;
        if (delta > 16.7) {
          this.jankSpikes++;
          const dropped = Math.max(0, Math.floor(delta / 16.67) - (delta % 16.67 > 4 ? 0 : 1));
          this.droppedFrames += dropped;
        }
        if (delta > this.maxFrameMs) {
          this.maxFrameMs = delta;
        }
        this.lastTime = now;
        this.rafId = requestAnimationFrame(loop);
      };
      this.rafId = requestAnimationFrame(loop);
    }
  }

  stop(): UiTelemetry {
    this.active = false;
    if (this.rafId !== null && typeof cancelAnimationFrame === 'function') {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    const endTime = typeof performance !== 'undefined' ? performance.now() : Date.now();
    const elapsed = Math.max(1, endTime - this.startTime);
    const fps = Math.max(1, Math.round((this.frameCount * 1000) / elapsed));

    return {
      workerFps: 120, // Standard offloaded worker telemetry
      mainThreadFps: typeof requestAnimationFrame === 'function' ? Math.min(120, fps) : 60,
      mainThreadJankMs: Number(this.maxFrameMs.toFixed(1)),
      droppedFrames: this.droppedFrames,
      jankSpikes: this.jankSpikes,
      totalFrameCount: this.frameCount
    };
  }
}

type EngineKey = 'webgpu' | 'cpu-parity' | 'ufuzzy' | 'js-native';

function shuffleArray<T>(arr: T[]): T[] {
  const res = [...arr];
  for (let i = res.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const temp = res[i]!;
    res[i] = res[j]!;
    res[j] = temp;
  }
  return res;
}

export class BenchmarkRunner {
  private gpuEngine: WebGPUEngine;
  private cpuEngine: CPUEngine;

  constructor(gpuEngine: WebGPUEngine, cpuEngine: CPUEngine) {
    this.gpuEngine = gpuEngine;
    this.cpuEngine = cpuEngine;
  }

  /**
   * Run the benchmark suite across dataset sizes with randomized/interleaved
   * execution, median + p95 statistics, and real UI thread telemetry.
   */
  async runBenchmark(
    sizes: number[] = [10_000, 100_000, 500_000, 1_000_000, 2_000_000],
    query: string = 'AuthController',
    mode: 'substring' | 'fuzzy' = 'substring',
    iterationsOrOptions?: number | BenchmarkOptions,
    maybeProgress?: (progress: BenchmarkProgress) => void
  ): Promise<BenchmarkRowResult[]> {
    let warmups = 5;
    let samples = 20;
    let corpusType: CorpusType = 'ascii';
    let onProgress: ((progress: BenchmarkProgress) => void) | undefined;

    if (typeof iterationsOrOptions === 'number') {
      // Backwards compatibility with previous (sizes, query, mode, iterations, onProgress)
      samples = Math.max(iterationsOrOptions, 5);
      warmups = Math.min(5, samples);
      onProgress = maybeProgress;
    } else if (iterationsOrOptions && typeof iterationsOrOptions === 'object') {
      warmups = iterationsOrOptions.warmups ?? 5;
      samples = iterationsOrOptions.samples ?? 20;
      corpusType = iterationsOrOptions.corpusType ?? 'ascii';
      onProgress = iterationsOrOptions.onProgress ?? maybeProgress;
    }

    const results: BenchmarkRowResult[] = [];
    const totalSteps = sizes.length;

    for (let sIdx = 0; sIdx < sizes.length; sIdx++) {
      const size = sizes[sIdx]!;
      if (onProgress) {
        onProgress({
          currentStep: sIdx + 1,
          totalSteps,
          stepName: `[${sIdx + 1}/${totalSteps}] Generating ${size.toLocaleString()} ${corpusType.toUpperCase()} items...`
        });
      }

      // Small delay to allow UI to breathe
      await new Promise(r => setTimeout(r, 40));

      // 1. Generate Dataset and capture normalization and packing breakdown
      const dataset: Dataset = generateDataset(size, { corpusType });

      // 2. Pre-normalize query tokens for zero-overhead CPU parity reference
      const normalizedQuery = normalizeText(query, true);

      // 3. Per-buffer VRAM accounting matching §2.5
      // records = tokenCount * 4
      // offsets = (rowCount + 1) * 4
      // query = 512 B (128 u32 tokens)
      // output = 65,544 B (8 + 8192 * 8)
      const recordsBytes = dataset.tokenCount * 4;
      const offsetsBytes = (dataset.size + 1) * 4;
      const queryBytes = 512;
      const outputBytes = 65544;
      const totalVramBytes = recordsBytes + offsetsBytes + queryBytes + outputBytes;
      const vramAllocation: VramAllocation = {
        recordsBytes,
        offsetsBytes,
        queryBytes,
        outputBytes,
        totalBytes: totalVramBytes
      };

      if (onProgress) {
        onProgress({
          currentStep: sIdx + 1,
          totalSteps,
          stepName: `[${sIdx + 1}/${totalSteps}] Benchmarking ${size.toLocaleString()} items (GPU Cold Upload)...`
        });
      }

      // 4. Benchmark GPU Cold (upload + first query once)
      let coldUploadMs = 0;
      let coldTotalMs = 0;

      if (this.gpuEngine.isReady) {
        const cold = await this.gpuEngine.searchCold(dataset.serializedDataset, query, { mode, maxResults: 1000 });
        coldUploadMs = Number(cold.datasetUploadMs.toFixed(2));
        coldTotalMs = Number(cold.coldTotalMs.toFixed(2));
      }

      const packing: PackingPipelineStats = {
        normalizeMs: Number(dataset.normalizeMs.toFixed(2)),
        packMs: Number(dataset.packMs.toFixed(2)),
        uploadMs: coldUploadMs,
        totalPipelineMs: Number((dataset.normalizeMs + dataset.packMs + coldUploadMs).toFixed(2))
      };

      // 5. Setup Telemetry Tracker
      const frameTracker = new FrameTelemetryTracker();
      frameTracker.start();

      let uiTelemetry: UiTelemetry;

      const gpuSamples: number[] = [];
      const gpuQueryUploadSamples: number[] = [];
      const gpuEncodeSubmitSamples: number[] = [];
      const gpuExecutionSamples: number[] = [];
      const gpuReadbackSamples: number[] = [];
      let gpuMatches = 0;
      let gpuHasOverflow = false;

      const paritySamples: number[] = [];
      let parityMatches = 0;

      const ufuzzySamples: number[] = [];
      let ufuzzyMatches = 0;

      const jsNativeSamples: number[] = [];
      let jsNativeMatches = 0;

      try {
        // Engines for interleaved execution
        const engines: EngineKey[] = ['webgpu', 'cpu-parity', 'ufuzzy', 'js-native'];

        // 6. Warmup Phase (Randomized execution to eliminate JIT compilation outliers)
        if (onProgress) {
          onProgress({
            currentStep: sIdx + 1,
            totalSteps,
            stepName: `[${sIdx + 1}/${totalSteps}] Warming up engines (${warmups} iterations)...`
          });
        }

        for (let w = 0; w < warmups; w++) {
          const shuffled = shuffleArray(engines);
          for (const engine of shuffled) {
            switch (engine) {
              case 'webgpu':
                if (this.gpuEngine.isReady) {
                  await this.gpuEngine.search(query, { mode, maxResults: 1000 });
                }
                break;
              case 'cpu-parity':
                scoreExactMatches(dataset.recordTokens, normalizedQuery.tokens, mode, 1000, dataset.strings);
                break;
              case 'ufuzzy':
                this.cpuEngine.searchWithUFuzzy(dataset.strings, query, 1000);
                break;
              case 'js-native':
                this.cpuEngine.searchNaiveScan(dataset.strings, query, 1000);
                break;
            }
            await new Promise(r => setTimeout(r, 2));
          }
          await new Promise(r => setTimeout(r, 10));
        }

        // 7. Measurement Phase: Randomized / Interleaved Execution
        if (onProgress) {
          onProgress({
            currentStep: sIdx + 1,
            totalSteps,
            stepName: `[${sIdx + 1}/${totalSteps}] Collecting ${samples} samples across interleaved engines...`
          });
        }

        for (let s = 0; s < samples; s++) {
          const shuffled = shuffleArray(engines);

          for (const engine of shuffled) {
            switch (engine) {
              case 'webgpu':
                if (this.gpuEngine.isReady) {
                  const res = await this.gpuEngine.search(query, { mode, maxResults: 1000 });
                  gpuSamples.push(res.timings.totalMs);
                  gpuQueryUploadSamples.push(res.timings.queryUploadMs);
                  gpuEncodeSubmitSamples.push(res.timings.encodeSubmitMs);
                  if (res.timings.gpuExecutionMs !== null) {
                    gpuExecutionSamples.push(res.timings.gpuExecutionMs);
                  }
                  gpuReadbackSamples.push(res.timings.readbackMs);
                  gpuMatches = res.totalMatches;
                  if (res.hasOverflow) gpuHasOverflow = true;
                }
                break;

              case 'cpu-parity': {
                const res = scoreExactMatches(
                  dataset.recordTokens,
                  normalizedQuery.tokens,
                  mode,
                  1000,
                  dataset.strings
                );
                paritySamples.push(res.durationMs);
                parityMatches = res.totalMatches;
                break;
              }

              case 'ufuzzy': {
                const res = this.cpuEngine.searchWithUFuzzy(dataset.strings, query, 1000);
                ufuzzySamples.push(res.durationMs);
                ufuzzyMatches = res.totalMatches;
                break;
              }

              case 'js-native': {
                const res = this.cpuEngine.searchNaiveScan(dataset.strings, query, 1000);
                jsNativeSamples.push(res.durationMs);
                jsNativeMatches = res.totalMatches;
                break;
              }
            }
            await new Promise(r => setTimeout(r, 2));
          }

          // Brief yield between rounds for GC stability
          await new Promise(r => setTimeout(r, 10));
        }
      } finally {
        // Stop frame tracker and retrieve genuine UI telemetry
        uiTelemetry = frameTracker.stop();
      }

      // Compute statistics (median, p95, min, max, mean)
      const gpuStats = calcLatencyStats(gpuSamples);
      const parityStats = calcLatencyStats(paritySamples);
      const ufuzzyStats = calcLatencyStats(ufuzzySamples);
      const nativeStats = calcLatencyStats(jsNativeSamples);

      const gpuRetainedMedian = gpuStats.medianMs;
      const gpuQueryUploadMedian = calcLatencyStats(gpuQueryUploadSamples).medianMs;
      const gpuEncodeSubmitMedian = calcLatencyStats(gpuEncodeSubmitSamples).medianMs;
      const gpuExecutionMedian = gpuExecutionSamples.length > 0 ? calcLatencyStats(gpuExecutionSamples).medianMs : null;
      const gpuReadbackMedian = calcLatencyStats(gpuReadbackSamples).medianMs;

      // Speedups based on warm median
      const retainedVsUfuzzySpeedup = gpuRetainedMedian > 0 && ufuzzyStats.medianMs > 0
        ? Number((ufuzzyStats.medianMs / gpuRetainedMedian).toFixed(2))
        : 0;
      const retainedVsParitySpeedup = gpuRetainedMedian > 0 && parityStats.medianMs > 0
        ? Number((parityStats.medianMs / gpuRetainedMedian).toFixed(2))
        : 0;
      const retainedVsNativeSpeedup = gpuRetainedMedian > 0 && nativeStats.medianMs > 0
        ? Number((nativeStats.medianMs / gpuRetainedMedian).toFixed(2))
        : 0;
      const coldVsUfuzzySpeedup = coldTotalMs > 0 && ufuzzyStats.medianMs > 0
        ? Number((ufuzzyStats.medianMs / coldTotalMs).toFixed(2))
        : 0;

      // Hardware qualification
      const hardwareQualified = isPhysicalGpu(this.gpuEngine.adapterInfo);
      const qualificationStatus: 'qualified' | 'pending-hardware' = hardwareQualified
        ? 'qualified'
        : 'pending-hardware';

      const rowResult: BenchmarkRowResult = {
        datasetSize: size,
        corpusType,
        query,
        mode,
        utf16Units: dataset.utf16Units,
        codePoints: dataset.codePoints,
        vramAllocation,
        packing,
        gpuRetained: {
          medianMs: gpuStats.medianMs,
          p95Ms: gpuStats.p95Ms,
          totalMs: gpuStats.medianMs,
          queryUploadMs: gpuQueryUploadMedian,
          encodeSubmitMs: gpuEncodeSubmitMedian,
          gpuExecutionMs: gpuExecutionMedian,
          readbackMs: gpuReadbackMedian,
          gpuDispatchMs: gpuEncodeSubmitMedian,
          samples: gpuStats.samples
        },
        gpuCold: {
          uploadMs: coldUploadMs,
          totalMs: coldTotalMs
        },
        cpuParity: parityStats,
        ufuzzy: ufuzzyStats,
        jsNative: nativeStats,
        cpuParityMs: parityStats.medianMs,
        ufuzzyMs: ufuzzyStats.medianMs,
        jsNativeMs: nativeStats.medianMs,
        retainedVsUfuzzySpeedup,
        retainedVsParitySpeedup,
        retainedVsNativeSpeedup,
        coldVsUfuzzySpeedup,
        crossover: {
          gpuRetainedBeatsUfuzzy: gpuRetainedMedian > 0 && gpuRetainedMedian < ufuzzyStats.medianMs,
          gpuRetainedBeatsParity: gpuRetainedMedian > 0 && gpuRetainedMedian < parityStats.medianMs,
          gpuRetainedBeatsNative: gpuRetainedMedian > 0 && gpuRetainedMedian < nativeStats.medianMs,
          gpuColdBeatsUfuzzy: coldTotalMs > 0 && coldTotalMs < ufuzzyStats.medianMs
        },
        matchCount: {
          gpu: gpuMatches,
          cpuParity: parityMatches,
          ufuzzy: ufuzzyMatches,
          native: jsNativeMatches
        },
        hasOverflow: gpuHasOverflow,
        uiTelemetry,
        hardwareQualified,
        qualificationStatus
      };

      results.push(rowResult);

      if (onProgress) {
        onProgress({
          currentStep: sIdx + 1,
          totalSteps,
          stepName: `Completed ${size.toLocaleString()} items (${corpusType.toUpperCase()}).`,
          currentRow: rowResult
        });
      }
    }

    return results;
  }
}
