import SUBSTRING_WGSL from './shaders/substring.wgsl';
import FUZZY_WGSL from './shaders/fuzzy.wgsl';
import { WebGPUContextManager } from './context-manager';
import { sanitizeStringForSlot, packStringsToGPUBuffer } from './buffer';
import type { AdapterInfo, SearchOptions, SearchResultItem, SearchTimings, SearchMode } from './types';

const BufferUsage = (typeof globalThis !== 'undefined' && 'GPUBufferUsage' in globalThis ? (globalThis as any).GPUBufferUsage : {
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  INDIRECT: 0x0100,
  QUERY_RESOLVE: 0x0200
});

const MapMode = (typeof globalThis !== 'undefined' && 'GPUMapMode' in globalThis ? (globalThis as any).GPUMapMode : {
  READ: 0x0001,
  WRITE: 0x0002
});

export interface DatasetLike {
  size: number;
  strings: string[];
  recordsBufferData?: ArrayBuffer;
  recordsByteLength?: number;
  offsetsBufferData?: ArrayBuffer;
  offsetsByteLength?: number;
  gpuBufferData?: ArrayBuffer;
  byteLength?: number;
}

export interface WebGPUSearchResult {
  query: string;
  mode: SearchMode;
  totalMatches: number;
  candidateCount: number;
  hasOverflow: boolean;
  results: SearchResultItem[];
  timings: SearchTimings;
}

export type SearchResult = WebGPUSearchResult;

export interface ColdSearchResult extends WebGPUSearchResult {
  datasetUploadMs: number;
  coldTotalMs: number;
}

export class WebGPUEngine {
  private device: GPUDevice | null = null;
  private isSharedDevice: boolean = false;
  private substringPipeline: GPUComputePipeline | null = null;
  private fuzzyPipeline: GPUComputePipeline | null = null;

  private offsetsBuffer: GPUBuffer | null = null;
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
  private searchMutex: Promise<any> = Promise.resolve();

  public adapterInfo: AdapterInfo | null = null;

  get isReady(): boolean {
    return this.device !== null;
  }

  get currentSize(): number {
    return this.currentDatasetSize;
  }

  /**
   * Check WebGPU support and initialize device and pipelines.
   * Supports custom injected device or automatically acquires a shared device.
   */
  async init(customDevice?: GPUDevice): Promise<boolean> {
    if (customDevice) {
      this.device = customDevice;
      this.isSharedDevice = false;
      this.adapterInfo = {
        vendor: 'Mock Vendor',
        architecture: 'Mock Architecture',
        device: 'Mock WebGPU Device',
        description: 'Deterministic Mock Device (vgpu/mock) for CI / Headless Unit Testing',
        renderer: 'Mock WebGPU Device',
        maxBufferSizeMB: 256,
        maxStorageBindingSizeMB: 128,
        maxComputeWorkgroupsPerDimension: 65535,
        maxComputeInvocationsPerWorkgroup: 256,
        hasTimestampQuery: customDevice.features ? customDevice.features.has('timestamp-query') : false
      };
      return await this.setupPipelinesAndBuffers();
    }

    const acquired = await WebGPUContextManager.acquireDevice();
    if (!acquired) {
      return false;
    }

    this.device = acquired.device;
    this.adapterInfo = acquired.adapterInfo;
    this.isSharedDevice = acquired.isShared;

    return await this.setupPipelinesAndBuffers();
  }

  private async setupPipelinesAndBuffers(): Promise<boolean> {
    if (!this.device) return false;

    // Create compute pipelines
    const substringModule = this.device.createShaderModule({
      label: 'Substring Search Module',
      code: SUBSTRING_WGSL
    });

    const fuzzyModule = this.device.createShaderModule({
      label: 'Fuzzy Search Module',
      code: FUZZY_WGSL
    });

    const createPipeline = async (desc: GPUComputePipelineDescriptor) => {
      if (typeof (this.device as any).createComputePipelineAsync === 'function') {
        return await this.device!.createComputePipelineAsync(desc);
      }
      return (this.device as any).createComputePipeline(desc);
    };

    this.substringPipeline = await createPipeline({
      label: 'Substring Pipeline',
      layout: 'auto',
      compute: {
        module: substringModule,
        entryPoint: 'main'
      }
    });

    this.fuzzyPipeline = await createPipeline({
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
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });

    // Setup timestamp queries if supported
    if (this.device.features && this.device.features.has('timestamp-query')) {
      try {
        this.querySet = this.device.createQuerySet({
          label: 'Search Timestamp QuerySet',
          type: 'timestamp',
          count: 2
        });
        this.queryResolveBuffer = this.device.createBuffer({
          label: 'Timestamp Resolve Buffer',
          size: 16,
          usage: BufferUsage.QUERY_RESOLVE | BufferUsage.COPY_SRC
        });
        this.queryStagingBuffer = this.device.createBuffer({
          label: 'Timestamp Staging Buffer',
          size: 16,
          usage: BufferUsage.MAP_READ | BufferUsage.COPY_DST
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
  }

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
      usage: BufferUsage.STORAGE | BufferUsage.COPY_SRC | BufferUsage.COPY_DST
    });

    this.stagingBuffer = this.device.createBuffer({
      label: 'Staging Buffer',
      size: this.outputByteLength,
      usage: BufferUsage.MAP_READ | BufferUsage.COPY_DST
    });
  }

  /**
   * Upload dataset to GPU VRAM for retained search.
   * Serialized through searchMutex to prevent buffer destruction races.
   */
  async loadDataset(dataset: DatasetLike | string[]): Promise<{ uploadTimeMs: number }> {
    const execute = () => this.loadDatasetInternal(dataset);
    const p = this.searchMutex.then(execute, execute);
    this.searchMutex = p.catch(() => {});
    return p;
  }

  private async loadDatasetInternal(dataset: DatasetLike | string[]): Promise<{ uploadTimeMs: number }> {
    let strings: string[];
    let recordsData: ArrayBuffer;
    let recordsLen: number;
    let offsetsData: ArrayBuffer;
    let offsetsLen: number;
    let count: number;

    if (Array.isArray(dataset)) {
      strings = dataset;
      count = dataset.length;
      const packed = packStringsToGPUBuffer(strings);
      recordsData = packed.recordsBufferData;
      recordsLen = packed.recordsByteLength;
      offsetsData = packed.offsetsBufferData;
      offsetsLen = packed.offsetsByteLength;
    } else {
      strings = dataset.strings;
      count = dataset.size;
      if (dataset.recordsBufferData && dataset.offsetsBufferData) {
        recordsData = dataset.recordsBufferData;
        recordsLen = dataset.recordsByteLength ?? dataset.recordsBufferData.byteLength;
        offsetsData = dataset.offsetsBufferData;
        offsetsLen = dataset.offsetsByteLength ?? dataset.offsetsBufferData.byteLength;
      } else {
        const packed = packStringsToGPUBuffer(strings);
        recordsData = packed.recordsBufferData;
        recordsLen = packed.recordsByteLength;
        offsetsData = packed.offsetsBufferData;
        offsetsLen = packed.offsetsByteLength;
      }
    }

    if (!this.device) {
      this.currentDatasetSize = count;
      this.currentStrings = strings;
      return { uploadTimeMs: 0 };
    }

    if (this.offsetsBuffer) {
      try { this.offsetsBuffer.destroy(); } catch {}
      this.offsetsBuffer = null;
    }
    if (this.recordsBuffer) {
      try { this.recordsBuffer.destroy(); } catch {}
      this.recordsBuffer = null;
    }

    const t0 = performance.now();

    this.offsetsBuffer = this.device.createBuffer({
      label: `Offsets Buffer (${count} items)`,
      size: Math.max(offsetsLen, 16),
      usage: BufferUsage.STORAGE | BufferUsage.COPY_DST
    });

    this.recordsBuffer = this.device.createBuffer({
      label: `Records Buffer (${count} items, ${recordsLen} bytes)`,
      size: Math.max(recordsLen, 16),
      usage: BufferUsage.STORAGE | BufferUsage.COPY_DST
    });

    this.device.queue.writeBuffer(this.offsetsBuffer, 0, offsetsData);
    this.device.queue.writeBuffer(this.recordsBuffer, 0, recordsData);
    await this.device.queue.onSubmittedWorkDone();

    const uploadTimeMs = performance.now() - t0;
    this.currentDatasetSize = count;
    this.currentStrings = strings;

    return { uploadTimeMs };
  }

  /**
   * Execute search against pre-loaded VRAM records (mutex-protected to prevent buffer race conditions)
   */
  async search(query: string, options: SearchOptions): Promise<WebGPUSearchResult> {
    const execute = () => this.searchInternal(query, options);
    const p = this.searchMutex.then(execute, execute);
    this.searchMutex = p.catch(() => {});
    return p;
  }

  private async searchInternal(query: string, options: SearchOptions): Promise<WebGPUSearchResult> {
    const mode = options.mode ?? 'fuzzy';
    const limit = Math.max(1, Math.min(options.limit ?? options.maxResults ?? 50, 8192));

    const emptyTimings: SearchTimings = {
      queryUploadMs: 0,
      encodeSubmitMs: 0,
      gpuExecutionMs: null,
      readbackMs: 0,
      totalMs: 0,
      gpuDispatchMs: 0
    };

    if (!this.device || !this.recordsBuffer || !this.offsetsBuffer || !this.uniformBuffer || !this.outputBuffer || !this.stagingBuffer) {
      return {
        query,
        mode,
        totalMatches: 0,
        candidateCount: 0,
        hasOverflow: false,
        results: [],
        timings: emptyTimings
      };
    }

    if (options.signal?.aborted) {
      throw new DOMException('Search aborted', 'AbortError');
    }

    const cleanQuery = query.trim();
    if (cleanQuery.length === 0) {
      return {
        query: '',
        mode,
        totalMatches: 0,
        candidateCount: 0,
        hasOverflow: false,
        results: [],
        timings: emptyTimings
      };
    }

    if (limit > this.candidateCapacity) {
      this.allocateOutputBuffers(limit);
    }
    const pipeline = mode === 'fuzzy' ? this.fuzzyPipeline! : this.substringPipeline!;
    const caseSensitive = !!options.caseSensitive;

    const totalStart = performance.now();

    // 1. Normalize Query and Upload Uniforms
    const tQueryStart = performance.now();
    const uniformData = new ArrayBuffer(272);
    const u32Uniform = new Uint32Array(uniformData);

    // Normalize query string (remove diacritics / sanitize) to match recordsBuffer encoding
    const normalizedQuery = sanitizeStringForSlot(cleanQuery, 59);
    const queryLen = normalizedQuery.length;

    u32Uniform[0] = this.currentDatasetSize;
    u32Uniform[1] = queryLen;
    u32Uniform[2] = this.candidateCapacity;
    u32Uniform[3] = caseSensitive ? 1 : 0;

    for (let i = 0; i < queryLen; i++) {
      u32Uniform[4 + i] = normalizedQuery.charCodeAt(i) & 0xFF;
    }

    this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);
    const queryUploadMs = performance.now() - tQueryStart;

    if (options.signal?.aborted) {
      throw new DOMException('Search aborted', 'AbortError');
    }

    // 2. Dispatch Compute
    const tDispatchStart = performance.now();
    const commandEncoder = this.device.createCommandEncoder({ label: 'Search Command Encoder' });

    if (typeof (commandEncoder as any).clearBuffer === 'function') {
      commandEncoder.clearBuffer(this.outputBuffer, 0, this.outputByteLength);
    }

    const bindGroup = this.device.createBindGroup({
      label: 'Search BindGroup',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.offsetsBuffer } },
        { binding: 2, resource: { buffer: this.recordsBuffer } },
        { binding: 3, resource: { buffer: this.outputBuffer } }
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

    commandEncoder.copyBufferToBuffer(this.outputBuffer, 0, this.stagingBuffer, 0, this.outputByteLength);
    this.device.queue.submit([commandEncoder.finish()]);
    const encodeSubmitMs = performance.now() - tDispatchStart;

    if (options.signal?.aborted) {
      throw new DOMException('Search aborted', 'AbortError');
    }

    // 3. MapAsync Readback with Scoped Cleanup
    const tReadbackStart = performance.now();
    let gpuExecutionMs: number | null = null;

    if (this.queryStagingBuffer) {
      try {
        await this.queryStagingBuffer.mapAsync(MapMode.READ);
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
        try { this.queryStagingBuffer.unmap(); } catch {}
      }
    }

    let totalMatches = 0;
    let candidateCount = 0;
    let hasOverflow = false;
    let candidates: SearchResultItem[] = [];

    await this.stagingBuffer.mapAsync(MapMode.READ);
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
          text: this.currentStrings ? (this.currentStrings[index] ?? '') : ''
        };
      }
    } finally {
      this.stagingBuffer.unmap();
    }

    if (options.signal?.aborted) {
      throw new DOMException('Search aborted', 'AbortError');
    }

    const readbackMs = performance.now() - tReadbackStart;
    const totalMs = performance.now() - totalStart;

    candidates.sort((a, b) => b.score - a.score);
    const results = candidates.slice(0, limit);

    return {
      query,
      mode,
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
  async searchCold(dataset: DatasetLike | string[], query: string, options: SearchOptions): Promise<ColdSearchResult> {
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
    if (this.offsetsBuffer) {
      try { this.offsetsBuffer.destroy(); } catch {}
      this.offsetsBuffer = null;
    }
    if (this.recordsBuffer) {
      try { this.recordsBuffer.destroy(); } catch {}
      this.recordsBuffer = null;
    }
    if (this.uniformBuffer) {
      try { this.uniformBuffer.destroy(); } catch {}
      this.uniformBuffer = null;
    }
    if (this.outputBuffer) {
      try { this.outputBuffer.destroy(); } catch {}
      this.outputBuffer = null;
    }
    if (this.stagingBuffer) {
      try { this.stagingBuffer.destroy(); } catch {}
      this.stagingBuffer = null;
    }
    if (this.queryResolveBuffer) {
      try { this.queryResolveBuffer.destroy(); } catch {}
      this.queryResolveBuffer = null;
    }
    if (this.queryStagingBuffer) {
      try { this.queryStagingBuffer.destroy(); } catch {}
      this.queryStagingBuffer = null;
    }
    if (this.querySet) {
      try { this.querySet.destroy(); } catch {}
      this.querySet = null;
    }
    if (this.device) {
      WebGPUContextManager.releaseDevice(this.device, this.isSharedDevice);
      this.device = null;
    }
  }
}
