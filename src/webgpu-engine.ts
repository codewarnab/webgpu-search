import { SUBSTRING_WGSL } from './shaders/substring.wgsl.ts';
import { FUZZY_WGSL } from './shaders/fuzzy.wgsl.ts';
import type { Dataset } from './dataset.ts';

export interface SearchOptions {
    mode: 'substring' | 'fuzzy';
    caseSensitive?: boolean;
    maxResults?: number;
}

export interface SearchResult {
    query: string;
    mode: 'substring' | 'fuzzy';
    totalMatches: number;
    results: Array<{ index: number; score: number; text?: string }>;
    timings: {
        queryUploadMs: number;
        gpuDispatchMs: number;
        readbackMs: number;
        totalMs: number;
    };
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

    private currentDatasetSize: number = 0;
    private currentStrings: string[] | null = null;
    private maxResults: number = 1000;
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
                this.device = await this.adapter.requestDevice({ requiredFeatures });
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

            this.allocateOutputBuffers(this.maxResults);

            return true;
        } catch (err) {
            console.error('Failed to initialize WebGPU:', err);
            return false;
        }
    }

    private searchMutex: Promise<any> = Promise.resolve();

    private allocateOutputBuffers(maxResults: number) {
        if (!this.device) return;
        if (this.outputBuffer && maxResults <= this.maxResults) return;
        this.maxResults = Math.max(maxResults, 1000);
        // count (4 bytes) + pad (4 bytes) + maxResults * (index 4 bytes + score 4 bytes)
        this.outputByteLength = 8 + this.maxResults * 8;

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
        if (!this.device || !this.recordsBuffer || !this.uniformBuffer || !this.outputBuffer || !this.stagingBuffer) {
            return {
                query,
                mode: options.mode,
                totalMatches: 0,
                results: [],
                timings: {
                    queryUploadMs: 0,
                    gpuDispatchMs: 0,
                    readbackMs: 0,
                    totalMs: 0
                }
            };
        }

        const cleanQuery = query.trim();
        if (cleanQuery.length === 0) {
            return {
                query: '',
                mode: options.mode,
                totalMatches: 0,
                results: [],
                timings: {
                    queryUploadMs: 0,
                    gpuDispatchMs: 0,
                    readbackMs: 0,
                    totalMs: 0
                }
            };
        }

        const maxResults = Math.min(options.maxResults ?? 1000, this.maxResults);
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
        u32Uniform[2] = maxResults;
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

        const passEncoder = commandEncoder.beginComputePass({ label: 'Search Compute Pass' });
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, bindGroup);

        const workgroupSize = 128;
        const workgroupCount = Math.ceil(this.currentDatasetSize / workgroupSize);
        passEncoder.dispatchWorkgroups(workgroupCount);
        passEncoder.end();

        // Copy output buffer to staging buffer for CPU readback
        commandEncoder.copyBufferToBuffer(this.outputBuffer, 0, this.stagingBuffer, 0, this.outputByteLength);

        this.device.queue.submit([commandEncoder.finish()]);
        const gpuDispatchMs = performance.now() - tDispatchStart;

        // 3. MapAsync Readback
        const tReadbackStart = performance.now();
        await this.stagingBuffer.mapAsync(GPUMapMode.READ);
        const arrayBuffer = this.stagingBuffer.getMappedRange();

        const u32Read = new Uint32Array(arrayBuffer);
        const i32Read = new Int32Array(arrayBuffer);

        const totalMatches = u32Read[0];
        const numItems = Math.min(totalMatches, maxResults);
        const results: Array<{ index: number; score: number; text?: string }> = new Array(numItems);

        for (let i = 0; i < numItems; i++) {
            const index = u32Read[2 + i * 2];
            const score = i32Read[3 + i * 2];
            results[i] = {
                index,
                score,
                text: this.currentStrings ? this.currentStrings[index] : undefined
            };
        }

        this.stagingBuffer.unmap();
        const readbackMs = performance.now() - tReadbackStart;
        const totalMs = performance.now() - totalStart;

        // Sort results by score descending
        results.sort((a, b) => b.score - a.score);

        return {
            query,
            mode: options.mode,
            totalMatches,
            results,
            timings: {
                queryUploadMs,
                gpuDispatchMs,
                readbackMs,
                totalMs
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
        if (this.device) this.device.destroy();
    }
}
