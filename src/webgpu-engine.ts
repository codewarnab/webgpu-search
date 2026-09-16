import { SUBSTRING_WGSL } from './shaders/substring.wgsl.ts';
import { FUZZY_WGSL } from './shaders/fuzzy.wgsl.ts';
import type { Dataset } from './dataset.ts';

export interface SearchOptions {
    mode: 'substring' | 'fuzzy';
    caseSensitive?: boolean;
    maxResults?: number;
}

export interface SearchTimings {
    queryUploadMs: number;
    encodeSubmitMs: number;
    gpuExecutionMs: number | null;
    readbackMs: number;
    totalMs: number;
    /** Backwards-compatible alias for encodeSubmitMs */
    gpuDispatchMs: number;
}

export interface SearchResult {
    query: string;
    mode: 'substring' | 'fuzzy';
    totalMatches: number;
    candidateCount: number;
    hasOverflow: boolean;
    results: Array<{ index: number; score: number; text?: string }>;
    timings: SearchTimings;
}

export interface ColdSearchResult extends SearchResult {
    datasetUploadMs: number;
    coldTotalMs: number;
}

export interface AdapterInfo {
    vendor: string;
    architecture: string;
    device: string;
    description: string;
    renderer: string;
    maxBufferSizeMB: number;
    maxStorageBindingSizeMB: number;
    maxComputeWorkgroupsPerDimension: number;
    maxComputeInvocationsPerWorkgroup: number;
    hasTimestampQuery: boolean;
}

export class WebGPUEngine {
    private adapter: GPUAdapter | null = null;
    private device: GPUDevice | null = null;
    private substringPipeline: GPUComputePipeline | null = null;
    private fuzzyPipeline: GPUComputePipeline | null = null;

    private recordsBuffer: GPUBuffer | null = null;
    private uniformBuffer: GPUBuffer | null = null;
    private outputBuffer: GPUBuffer | null = null;
    private stagingBuffer: GPUBuffer | null = null;

    private querySet: GPUQuerySet | null = null;
    private queryResolveBuffer: GPUBuffer | null = null;
    private queryStagingBuffer: GPUBuffer | null = null;

    private currentDatasetSize: number = 0;
    private currentStrings: string[] | null = null;
    private candidateCapacity: number = 8192;
    private outputByteLength: number = 0;

    public adapterInfo: AdapterInfo | null = null;

    get isReady(): boolean {
        return this.device !== null;
    }

    /**
     * Check WebGPU support and initialize device and pipelines
     */
    async init(): Promise<boolean> {
        if (!navigator.gpu) {
            console.error('WebGPU is not supported in this browser.');
            return false;
        }

        try {
            // First try high-performance (discrete GPU)
            this.adapter = await navigator.gpu.requestAdapter({
                powerPreference: 'high-performance'
            });

            // Fallback to default/integrated GPU if high-performance is null
            if (!this.adapter) {
                this.adapter = await navigator.gpu.requestAdapter();
            }

            if (!this.adapter) {
                console.error('No suitable GPUAdapter found.');
                return false;
            }

            // Inspect adapter info
            let info: any = (this.adapter as any).info || {};
            if ((!info.vendor && !info.device) && 'requestAdapterInfo' in this.adapter) {
                try {
                    info = await (this.adapter as any).requestAdapterInfo();
                } catch {
                    info = {};
                }
            }

            // Fallback to WebGL unmasked renderer if WebGPU info is sanitized by browser
            let unmaskedRenderer = '';
            try {
                const canvas = document.createElement('canvas');
                const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
                if (gl) {
                    const ext = (gl as any).getExtension('WEBGL_debug_renderer_info');
                    if (ext) {
                        unmaskedRenderer = (gl as any).getParameter(ext.UNMASKED_RENDERER_WEBGL) || '';
                    }
                }
            } catch {
                // ignore
            }

            let vendor = info.vendor || '';
            let device = info.device || '';
            let architecture = info.architecture || '';

            if (!device && unmaskedRenderer) {
                device = unmaskedRenderer;
            }
            if (!vendor) {
                if (/nvidia/i.test(unmaskedRenderer) || /nvidia/i.test(device)) vendor = 'NVIDIA';
                else if (/intel/i.test(unmaskedRenderer) || /intel/i.test(device)) vendor = 'Intel';
                else if (/amd|radeon/i.test(unmaskedRenderer) || /amd|radeon/i.test(device)) vendor = 'AMD';
                else if (/apple/i.test(unmaskedRenderer) || /apple/i.test(device)) vendor = 'Apple';
                else vendor = 'Unknown GPU Vendor';
            }
            if (!device) device = 'WebGPU Generic Device';
            if (!architecture) architecture = 'Default';

            const limits = this.adapter.limits;
            const requiredFeatures: GPUFeatureName[] = [];
            const hasTimestamp = this.adapter.features.has('timestamp-query');
            if (hasTimestamp) {
                requiredFeatures.push('timestamp-query');
            }

            this.adapterInfo = {
                vendor,
                architecture,
                device,
                description: info.description || unmaskedRenderer || navigator.userAgent,
                renderer: unmaskedRenderer || device,
                maxBufferSizeMB: Math.round(limits.maxBufferSize / (1024 * 1024)),
                maxStorageBindingSizeMB: Math.round(limits.maxStorageBufferBindingSize / (1024 * 1024)),
                maxComputeWorkgroupsPerDimension: limits.maxComputeWorkgroupsPerDimension,
                maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup,
                hasTimestampQuery: hasTimestamp
            };

            // Request device with adapter limits, fallback to standard if driver rejects high limits
            try {
                this.device = await this.adapter.requestDevice({
                    requiredFeatures,
                    requiredLimits: {
                        maxBufferSize: limits.maxBufferSize,
                        maxStorageBufferBindingSize: limits.maxStorageBufferBindingSize,
                        maxComputeWorkgroupsPerDimension: limits.maxComputeWorkgroupsPerDimension
                    }
                });
            } catch (limitErr) {
                console.warn('requestDevice with custom limits failed, falling back to default limits:', limitErr);
                try {
                    this.device = await this.adapter.requestDevice({ requiredFeatures });
                } catch (featErr) {
                    console.warn('requestDevice with requiredFeatures failed, falling back without features:', featErr);
                    this.device = await this.adapter.requestDevice();
                }
            }

            // Create compute pipelines
            const substringModule = this.device.createShaderModule({
                label: 'Substring Search Module',
                code: SUBSTRING_WGSL
            });

            const fuzzyModule = this.device.createShaderModule({
                label: 'Fuzzy Search Module',
                code: FUZZY_WGSL
            });

            this.substringPipeline = await this.device.createComputePipelineAsync({
                label: 'Substring Pipeline',
                layout: 'auto',
                compute: {
                    module: substringModule,
                    entryPoint: 'main'
                }
            });

            this.fuzzyPipeline = await this.device.createComputePipelineAsync({
                label: 'Fuzzy Pipeline',
                layout: 'auto',
                compute: {
                    module: fuzzyModule,
                    entryPoint: 'main'
                }
            });

            // Uniform buffer: 272 bytes (16 bytes header + 256 bytes query array)
            this.uniformBuffer = this.device.createBuffer({
                label: 'Uniform Buffer',
                size: 272,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
            });

            // Setup timestamp queries if supported
            if (this.device.features.has('timestamp-query')) {
                try {
                    this.querySet = this.device.createQuerySet({
                        label: 'Search Timestamp QuerySet',
                        type: 'timestamp',
                        count: 2
                    });
                    this.queryResolveBuffer = this.device.createBuffer({
                        label: 'Timestamp Resolve Buffer',
                        size: 16,
                        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
                    });
                    this.queryStagingBuffer = this.device.createBuffer({
                        label: 'Timestamp Staging Buffer',
                        size: 16,
                        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
                    });
                } catch (qErr) {
                    console.warn('Failed to allocate timestamp query set:', qErr);
                    this.querySet = null;
                    this.queryResolveBuffer = null;
                    this.queryStagingBuffer = null;
                }
            }

            this.allocateOutputBuffers(this.candidateCapacity);

            return true;
        } catch (err) {
            console.error('Failed to initialize WebGPU:', err);
            return false;
        }
    }

    private searchMutex: Promise<any> = Promise.resolve();

    private allocateOutputBuffers(candidateCapacity: number = 8192) {
        if (!this.device) return;
        if (this.outputBuffer && candidateCapacity <= this.candidateCapacity) return;
        this.candidateCapacity = Math.max(candidateCapacity, 8192);
        // count (4 bytes) + pad (4 bytes) + candidateCapacity * (index 4 bytes + score 4 bytes)
        this.outputByteLength = 8 + this.candidateCapacity * 8;

        if (this.outputBuffer) this.outputBuffer.destroy();
        if (this.stagingBuffer) this.stagingBuffer.destroy();

        this.outputBuffer = this.device.createBuffer({
            label: 'Output Buffer',
            size: this.outputByteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });

        this.stagingBuffer = this.device.createBuffer({
            label: 'Staging Buffer',
            size: this.outputByteLength,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
    }

    /**
     * Upload dataset to GPU VRAM for retained search
     */
    async loadDataset(dataset: Dataset): Promise<{ uploadTimeMs: number }> {
        if (!this.device) {
            this.currentDatasetSize = dataset.size;
            this.currentStrings = dataset.strings;
            return { uploadTimeMs: 0 };
        }

        if (this.recordsBuffer) {
            this.recordsBuffer.destroy();
            this.recordsBuffer = null;
        }

        const t0 = performance.now();

        this.recordsBuffer = this.device.createBuffer({
            label: `Records Buffer (${dataset.size} items)`,
            size: dataset.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });

        this.device.queue.writeBuffer(this.recordsBuffer, 0, dataset.gpuBufferData);
        // Wait for upload queue to flush
        await this.device.queue.onSubmittedWorkDone();

        const uploadTimeMs = performance.now() - t0;
        this.currentDatasetSize = dataset.size;
        this.currentStrings = dataset.strings;

        return { uploadTimeMs };
    }

    /**
     * Execute search against pre-loaded VRAM records (mutex-protected to prevent buffer race conditions)
     */
    async search(query: string, options: SearchOptions): Promise<SearchResult> {
        const execute = () => this.searchInternal(query, options);
        const p = this.searchMutex.then(execute, execute);
        this.searchMutex = p.catch(() => {});
        return p;
    }

    private async searchInternal(query: string, options: SearchOptions): Promise<SearchResult> {
        const emptyTimings: SearchTimings = {
            queryUploadMs: 0,
            encodeSubmitMs: 0,
            gpuExecutionMs: null,
            readbackMs: 0,
            totalMs: 0,
            gpuDispatchMs: 0
        };

        if (!this.device || !this.recordsBuffer || !this.uniformBuffer || !this.outputBuffer || !this.stagingBuffer) {
            return {
                query,
                mode: options.mode,
                totalMatches: 0,
                candidateCount: 0,
                hasOverflow: false,
                results: [],
                timings: emptyTimings
            };
        }

        const cleanQuery = query.trim();
        if (cleanQuery.length === 0) {
            return {
                query: '',
                mode: options.mode,
                totalMatches: 0,
                candidateCount: 0,
                hasOverflow: false,
                results: [],
                timings: emptyTimings
            };
        }

        const maxResults = options.maxResults ?? 1000;
        if (maxResults > this.candidateCapacity) {
            this.allocateOutputBuffers(maxResults);
        }
        const pipeline = options.mode === 'fuzzy' ? this.fuzzyPipeline! : this.substringPipeline!;
        const caseSensitive = !!options.caseSensitive;

        const totalStart = performance.now();

        // 1. Upload Query Uniforms
        const tQueryStart = performance.now();
        const uniformData = new ArrayBuffer(272);
        const u32Uniform = new Uint32Array(uniformData);
        const queryLen = Math.min(cleanQuery.length, 59);
        u32Uniform[0] = this.currentDatasetSize;
        u32Uniform[1] = queryLen;
        u32Uniform[2] = this.candidateCapacity;
        u32Uniform[3] = caseSensitive ? 1 : 0;

        for (let i = 0; i < queryLen; i++) {
            u32Uniform[4 + i] = cleanQuery.charCodeAt(i) & 0xFF;
        }

        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);
        const queryUploadMs = performance.now() - tQueryStart;

        // 2. Dispatch Compute
        const tDispatchStart = performance.now();
        const commandEncoder = this.device.createCommandEncoder({ label: 'Search Command Encoder' });

        // Clear output count and header
        commandEncoder.clearBuffer(this.outputBuffer, 0, this.outputByteLength);

        const bindGroup = this.device.createBindGroup({
            label: 'Search BindGroup',
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: this.recordsBuffer } },
                { binding: 2, resource: { buffer: this.outputBuffer } }
            ]
        });

        const passDesc: GPUComputePassDescriptor = {
            label: 'Search Compute Pass'
        };
        if (this.querySet) {
            (passDesc as any).timestampWrites = {
                querySet: this.querySet,
                beginningOfPassWriteIndex: 0,
                endOfPassWriteIndex: 1
            };
        }

        const passEncoder = commandEncoder.beginComputePass(passDesc);
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, bindGroup);

        const workgroupSize = 128;
        const workgroupCount = Math.ceil(this.currentDatasetSize / workgroupSize);
        passEncoder.dispatchWorkgroups(workgroupCount);
        passEncoder.end();

        if (this.querySet && this.queryResolveBuffer && this.queryStagingBuffer) {
            commandEncoder.resolveQuerySet(this.querySet, 0, 2, this.queryResolveBuffer, 0);
            commandEncoder.copyBufferToBuffer(this.queryResolveBuffer, 0, this.queryStagingBuffer, 0, 16);
        }

        // Copy output buffer to staging buffer for CPU readback
        commandEncoder.copyBufferToBuffer(this.outputBuffer, 0, this.stagingBuffer, 0, this.outputByteLength);

        this.device.queue.submit([commandEncoder.finish()]);
        const encodeSubmitMs = performance.now() - tDispatchStart;

        // 3. MapAsync Readback
        const tReadbackStart = performance.now();
        const mapPromises: Promise<void>[] = [this.stagingBuffer.mapAsync(GPUMapMode.READ)];
        if (this.queryStagingBuffer) {
            mapPromises.push(this.queryStagingBuffer.mapAsync(GPUMapMode.READ));
        }

        await Promise.all(mapPromises);

        let gpuExecutionMs: number | null = null;
        if (this.queryStagingBuffer) {
            try {
                const timeBuffer = this.queryStagingBuffer.getMappedRange();
                const timeU64 = new BigUint64Array(timeBuffer);
                const t0 = timeU64[0];
                const t1 = timeU64[1];
                if (t1 >= t0 && t0 > 0n) {
                    gpuExecutionMs = Number(t1 - t0) / 1_000_000;
                }
            } catch (tsErr) {
                console.warn('Failed to read timestamp query:', tsErr);
            } finally {
                this.queryStagingBuffer.unmap();
            }
        }

        let totalMatches = 0;
        let candidateCount = 0;
        let hasOverflow = false;
        let candidates: Array<{ index: number; score: number; text?: string }> = [];

        try {
            const arrayBuffer = this.stagingBuffer.getMappedRange();
            const u32Read = new Uint32Array(arrayBuffer);
            const i32Read = new Int32Array(arrayBuffer);

            totalMatches = u32Read[0];
            candidateCount = Math.min(totalMatches, this.candidateCapacity);
            hasOverflow = totalMatches > this.candidateCapacity;

            candidates = new Array(candidateCount);
            for (let i = 0; i < candidateCount; i++) {
                const index = u32Read[2 + i * 2];
                const score = i32Read[3 + i * 2];
                candidates[i] = {
                    index,
                    score,
                    text: this.currentStrings ? this.currentStrings[index] : undefined
                };
            }
        } finally {
            this.stagingBuffer.unmap();
        }

        const readbackMs = performance.now() - tReadbackStart;
        const totalMs = performance.now() - totalStart;

        // Sort candidates descending by score on CPU and slice top-K
        candidates.sort((a, b) => b.score - a.score);
        const results = candidates.slice(0, maxResults);

        return {
            query,
            mode: options.mode,
            totalMatches,
            candidateCount,
            hasOverflow,
            results,
            timings: {
                queryUploadMs,
                encodeSubmitMs,
                gpuExecutionMs,
                readbackMs,
                totalMs,
                gpuDispatchMs: encodeSubmitMs
            }
        };
    }

    /**
     * Cold search: includes full dataset creation & upload in the timing measurement
     */
    async searchCold(dataset: Dataset, query: string, options: SearchOptions): Promise<ColdSearchResult> {
        if (!this.device) {
            const warmResult = await this.search(query, options);
            return {
                ...warmResult,
                datasetUploadMs: 0,
                coldTotalMs: 0
            };
        }

        const tUploadStart = performance.now();
        await this.loadDataset(dataset);
        const datasetUploadMs = performance.now() - tUploadStart;

        const warmResult = await this.search(query, options);

        return {
            ...warmResult,
            datasetUploadMs,
            coldTotalMs: datasetUploadMs + warmResult.timings.totalMs
        };
    }

    destroy() {
        if (this.recordsBuffer) this.recordsBuffer.destroy();
        if (this.uniformBuffer) this.uniformBuffer.destroy();
        if (this.outputBuffer) this.outputBuffer.destroy();
        if (this.stagingBuffer) this.stagingBuffer.destroy();
        if (this.queryResolveBuffer) this.queryResolveBuffer.destroy();
        if (this.queryStagingBuffer) this.queryStagingBuffer.destroy();
        if (this.querySet) this.querySet.destroy();
        if (this.device) this.device.destroy();
    }
}
