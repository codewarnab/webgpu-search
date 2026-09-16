import { WebGPUEngine } from './webgpu-engine.ts';
import { CPUEngine } from './cpu-engine.ts';
import { generateDataset, type Dataset } from './dataset.ts';

export interface BenchmarkRowResult {
    datasetSize: number;
    query: string;
    mode: 'substring' | 'fuzzy';
    gpuRetained: {
        totalMs: number;
        queryUploadMs: number;
        encodeSubmitMs: number;
        gpuExecutionMs: number | null;
        readbackMs: number;
        gpuDispatchMs: number;
    };
    gpuCold: {
        uploadMs: number;
        totalMs: number;
    };
    ufuzzyMs: number;
    jsNativeMs: number;
    retainedVsUfuzzySpeedup: number;
    retainedVsNativeSpeedup: number;
    coldVsUfuzzySpeedup: number;
    crossover: {
        gpuRetainedBeatsUfuzzy: boolean;
        gpuRetainedBeatsNative: boolean;
        gpuColdBeatsUfuzzy: boolean;
    };
    matchCount: {
        gpu: number;
        ufuzzy: number;
        native: number;
    };
    hasOverflow?: boolean;
}

export interface BenchmarkProgress {
    currentStep: number;
    totalSteps: number;
    stepName: string;
    currentRow?: BenchmarkRowResult;
}

export class BenchmarkRunner {
    private gpuEngine: WebGPUEngine;
    private cpuEngine: CPUEngine;

    constructor(gpuEngine: WebGPUEngine, cpuEngine: CPUEngine) {
        this.gpuEngine = gpuEngine;
        this.cpuEngine = cpuEngine;
    }

    async runBenchmark(
        sizes: number[] = [10_000, 100_000, 500_000, 1_000_000, 2_000_000],
        query: string = 'AuthController',
        mode: 'substring' | 'fuzzy' = 'substring',
        iterations: number = 3,
        onProgress?: (progress: BenchmarkProgress) => void
    ): Promise<BenchmarkRowResult[]> {
        const results: BenchmarkRowResult[] = [];
        const totalSteps = sizes.length;

        for (let sIdx = 0; sIdx < sizes.length; sIdx++) {
            const size = sizes[sIdx];
            if (onProgress) {
                onProgress({
                    currentStep: sIdx + 1,
                    totalSteps,
                    stepName: `Generating dataset of ${size.toLocaleString()} items...`
                });
            }

            // Small delay to allow UI to breathe
            await new Promise(r => setTimeout(r, 40));

            const dataset: Dataset = generateDataset(size);

            if (onProgress) {
                onProgress({
                    currentStep: sIdx + 1,
                    totalSteps,
                    stepName: `Benchmarking ${size.toLocaleString()} items on GPU & CPU...`
                });
            }

            // 1. Benchmark GPU Cold & Retained (if GPU available)
            let coldRes = { uploadMs: 0, coldTotalMs: 0 };
            let gpuRetainedTotal = 0;
            let gpuQueryUpload = 0;
            let gpuEncodeSubmit = 0;
            let gpuExecutionSum = 0;
            let gpuExecutionCount = 0;
            let gpuReadback = 0;
            let gpuMatches = 0;
            let gpuHasOverflow = false;

            if (this.gpuEngine.isReady) {
                // 1. Benchmark GPU Cold (measure upload + query once)
                const cold = await this.gpuEngine.searchCold(dataset, query, { mode, maxResults: 1000 });
                coldRes = { uploadMs: cold.datasetUploadMs, coldTotalMs: cold.coldTotalMs };

                // 2. Benchmark GPU Retained (data stays in VRAM, query multiple times)
                // Warmup
                await this.gpuEngine.search(query, { mode, maxResults: 1000 });

                for (let i = 0; i < iterations; i++) {
                    const res = await this.gpuEngine.search(query, { mode, maxResults: 1000 });
                    gpuRetainedTotal += res.timings.totalMs;
                    gpuQueryUpload += res.timings.queryUploadMs;
                    gpuEncodeSubmit += res.timings.encodeSubmitMs;
                    if (res.timings.gpuExecutionMs !== null) {
                        gpuExecutionSum += res.timings.gpuExecutionMs;
                        gpuExecutionCount++;
                    }
                    gpuReadback += res.timings.readbackMs;
                    gpuMatches = res.totalMatches;
                    if (res.hasOverflow) gpuHasOverflow = true;
                }

                gpuRetainedTotal /= iterations;
                gpuQueryUpload /= iterations;
                gpuEncodeSubmit /= iterations;
                gpuReadback /= iterations;
            }

            // 3. Benchmark CPU uFuzzy
            // Warmup
            this.cpuEngine.searchUFuzzy(dataset.strings, query, 1000);

            let ufuzzyTotal = 0;
            let ufuzzyMatches = 0;
            for (let i = 0; i < iterations; i++) {
                const res = this.cpuEngine.searchUFuzzy(dataset.strings, query, 1000);
                ufuzzyTotal += res.durationMs;
                ufuzzyMatches = res.totalMatches;
            }
            ufuzzyTotal /= iterations;

            // 4. Benchmark CPU Native JS
            // Warmup
            this.cpuEngine.searchNative(dataset.strings, query, 1000);

            let jsNativeTotal = 0;
            let jsNativeMatches = 0;
            for (let i = 0; i < iterations; i++) {
                const res = this.cpuEngine.searchNative(dataset.strings, query, 1000);
                jsNativeTotal += res.durationMs;
                jsNativeMatches = res.totalMatches;
            }
            jsNativeTotal /= iterations;

            const retainedVsUfuzzySpeedup = gpuRetainedTotal > 0 ? Number((ufuzzyTotal / gpuRetainedTotal).toFixed(2)) : 0;
            const retainedVsNativeSpeedup = gpuRetainedTotal > 0 ? Number((jsNativeTotal / gpuRetainedTotal).toFixed(2)) : 0;
            const coldVsUfuzzySpeedup = coldRes.coldTotalMs > 0 ? Number((ufuzzyTotal / coldRes.coldTotalMs).toFixed(2)) : 0;

            const rowResult: BenchmarkRowResult = {
                datasetSize: size,
                query,
                mode,
                gpuRetained: {
                    totalMs: Number(gpuRetainedTotal.toFixed(2)),
                    queryUploadMs: Number(gpuQueryUpload.toFixed(2)),
                    encodeSubmitMs: Number(gpuEncodeSubmit.toFixed(2)),
                    gpuExecutionMs: gpuExecutionCount > 0 ? Number((gpuExecutionSum / gpuExecutionCount).toFixed(2)) : null,
                    readbackMs: Number(gpuReadback.toFixed(2)),
                    gpuDispatchMs: Number(gpuEncodeSubmit.toFixed(2))
                },
                gpuCold: {
                    uploadMs: Number(coldRes.uploadMs.toFixed(2)),
                    totalMs: Number(coldRes.coldTotalMs.toFixed(2))
                },
                ufuzzyMs: Number(ufuzzyTotal.toFixed(2)),
                jsNativeMs: Number(jsNativeTotal.toFixed(2)),
                retainedVsUfuzzySpeedup,
                retainedVsNativeSpeedup,
                coldVsUfuzzySpeedup,
                crossover: {
                    gpuRetainedBeatsUfuzzy: gpuRetainedTotal > 0 && gpuRetainedTotal < ufuzzyTotal,
                    gpuRetainedBeatsNative: gpuRetainedTotal > 0 && gpuRetainedTotal < jsNativeTotal,
                    gpuColdBeatsUfuzzy: coldRes.coldTotalMs > 0 && coldRes.coldTotalMs < ufuzzyTotal
                },
                matchCount: {
                    gpu: gpuMatches,
                    ufuzzy: ufuzzyMatches,
                    native: jsNativeMatches
                },
                hasOverflow: gpuHasOverflow
            };

            results.push(rowResult);

            if (onProgress) {
                onProgress({
                    currentStep: sIdx + 1,
                    totalSteps,
                    stepName: `Completed ${size.toLocaleString()} items.`,
                    currentRow: rowResult
                });
            }
        }

        return results;
    }
}
