import SUBSTRING_WGSL from './shaders/substring.wgsl';
import FUZZY_WGSL from './shaders/fuzzy.wgsl';
import { WebGPUContextManager } from './context-manager';
import {
  checkMemoryBudget,
  deserializeUnicodeDataset,
  packUnicodeToGPUBuffer,
  type PackedUnicodeBufferV2,
} from './buffer';
import { normalizeText } from './unicode-preprocess';
import { compareParityResults } from './cpu-reference';
import { clampLimit, nowMs, throwIfAborted, abortError } from './runtime-guards';
import {
  PROFILE_TO_ENUM,
  QUERY_TOKENS_MAX,
  SCORING_TO_ENUM,
  SCORING_VERSION,
  UNICODE_VERSION,
  UNICODE_VERSION_TO_ENUM,
  IncompatibleIndexError,
  ProfileMismatchError,
  QueryTooLongError,
} from './text-profile';
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
  // `strings?` optional: hybrid enriches `text` from its own `items`, so
  // forcing strings through the engine duplicates the corpus. Token-only
  // datasets resolve `text` to `''` (documented). Rejected: strings-required
  // (2x residency), placeholder synthesis (dishonest). `string[]` overload
  // routes via `normalizeText` — single code path, no legacy sanitizer.
  size: number;
  strings?: string[];
  recordsBufferData?: ArrayBuffer;
  recordsByteLength?: number;
  offsetsBufferData?: ArrayBuffer;
  offsetsByteLength?: number;
  gpuBufferData?: ArrayBuffer;
  byteLength?: number;
}

export type EngineDataset = PackedUnicodeBufferV2 | ArrayBuffer | DatasetLike | string[];

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
  private deviceReleased: boolean = false;
  private substringPipeline: GPUComputePipeline | null = null;
  private fuzzyPipeline: GPUComputePipeline | null = null;

  private offsetsBuffer: GPUBuffer | null = null;
  private recordsBuffer: GPUBuffer | null = null;
  private uniformBuffer: GPUBuffer | null = null;
  private queryBuffer: GPUBuffer | null = null;
  private outputBuffer: GPUBuffer | null = null;
  private stagingBuffer: GPUBuffer | null = null;

  private querySet: GPUQuerySet | null = null;
  private queryResolveBuffer: GPUBuffer | null = null;
  private queryStagingBuffer: GPUBuffer | null = null;

  private currentDatasetSize: number = 0;
  private currentStrings: string[] | null = null;
  private currentTokens: Uint32Array | null = null;
  private currentOffsets: Uint32Array | null = null;
  private folded: boolean = true;
  private profileId: string = 'unicode-default';
  private unicodeVersion: string = UNICODE_VERSION;
  private scoringVersion: string = SCORING_VERSION;
  private candidateCapacity: number = 8192;
  private outputByteLength: number = 0;
  private searchMutex: Promise<any> = Promise.resolve();
  private generation: number = 0;

  public adapterInfo: AdapterInfo | null = null;

  get isReady(): boolean {
    return this.device !== null;
  }

  get currentSize(): number {
    return this.currentDatasetSize;
  }

  async init(customDevice?: GPUDevice): Promise<boolean> {
    if (customDevice) {
      this.device = customDevice;
      this.isSharedDevice = false;
      this.deviceReleased = false;
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
      this.attachLostHandler();
      return await this.setupPipelinesAndBuffers();
    }

    const acquired = await WebGPUContextManager.acquireDevice();
    if (!acquired) {
      return false;
    }

    this.device = acquired.device;
    this.adapterInfo = acquired.adapterInfo;
    this.isSharedDevice = acquired.isShared;
    this.deviceReleased = false;
    this.attachLostHandler();
    return await this.setupPipelinesAndBuffers();
  }

  private attachLostHandler(): void {
    const dev = this.device as unknown as { lost?: Promise<any> } | null;
    if (!dev?.lost?.then) return;
    dev.lost.then(() => {
      // Device loss: null handles, mark unready. Shared refCount decremented
      // once (deviceReleased guard) so live siblings keep the device.
      this.substringPipeline = this.fuzzyPipeline = null;
      this.offsetsBuffer = this.recordsBuffer = this.uniformBuffer = this.queryBuffer = null;
      this.outputBuffer = this.stagingBuffer = this.queryResolveBuffer = this.queryStagingBuffer = null;
      this.querySet = null;
      if (this.device !== null && !this.deviceReleased) {
        this.deviceReleased = true;
        try {
          WebGPUContextManager.releaseDevice(this.device, this.isSharedDevice);
        } catch {}
      }
      this.device = null;
      this.generation++;
    });
  }

  private async setupPipelinesAndBuffers(): Promise<boolean> {
    if (!this.device) return false;

    const substringModule = this.device.createShaderModule({
      label: 'substring',
      code: SUBSTRING_WGSL
    });

    const fuzzyModule = this.device.createShaderModule({
      label: 'fuzzy',
      code: FUZZY_WGSL
    });

    const createPipeline = async (desc: GPUComputePipelineDescriptor) => {
      if (typeof (this.device as any).createComputePipelineAsync === 'function') {
        return await this.device!.createComputePipelineAsync(desc);
      }
      return (this.device as any).createComputePipeline(desc);
    };

    this.substringPipeline = await createPipeline({
      label: 'substring',
      layout: 'auto',
      compute: {
        module: substringModule,
        entryPoint: 'main'
      }
    });

    this.fuzzyPipeline = await createPipeline({
      label: 'fuzzy',
      layout: 'auto',
      compute: {
        module: fuzzyModule,
        entryPoint: 'main'
      }
    });

    // Uniform header: 32 B, 16 B-aligned. Query buffer: 512 B persistent
    // storage (QUERY_TOKENS_MAX*4, min 16 B), written per query.
    this.uniformBuffer = this.device.createBuffer({
      label: 'Uniform Buffer',
      size: 32,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });

    // Persistent storage query buffer: QUERY_TOKENS_MAX * 4 = 512 B, min 16 B,
    // written per query via queue.writeBuffer, destroyed under mutex.
    this.queryBuffer = this.device.createBuffer({
      label: 'Query Buffer',
      size: Math.max(QUERY_TOKENS_MAX * 4, 16),
      usage: BufferUsage.STORAGE | BufferUsage.COPY_DST
    });

    if (this.device.features && this.device.features.has('timestamp-query')) {
      try {
        this.querySet = this.device.createQuerySet({
          label: 'ts-query',
          type: 'timestamp',
          count: 2
        });
        this.queryResolveBuffer = this.device.createBuffer({
          label: 'ts-resolve',
          size: 16,
          usage: BufferUsage.QUERY_RESOLVE | BufferUsage.COPY_SRC
        });
        this.queryStagingBuffer = this.device.createBuffer({
          label: 'ts-staging',
          size: 16,
          usage: BufferUsage.MAP_READ | BufferUsage.COPY_DST
        });
      } catch (qErr) {
        console.warn('timestamp alloc failed:', qErr);
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

  private queued<T>(fn: () => Promise<T>): Promise<T> {
    const p = this.searchMutex.then(fn, fn);
    this.searchMutex = p.catch(() => {});
    return p;
  }

  async loadDataset(dataset: EngineDataset): Promise<{ uploadTimeMs: number }> {
    return this.queued(() => this.loadDatasetInternal(dataset));
  }

  private resolvePacked(dataset: EngineDataset): { packed: PackedUnicodeBufferV2; strings: string[] | null } {
    if (Array.isArray(dataset)) {
      const arr = dataset as unknown[];
      if (arr.length === 0 || typeof arr[0] === 'string') {
        const packed = packUnicodeToGPUBuffer(arr as string[], { folded: this.folded });
        return { packed, strings: (arr as string[]).slice() };
      }
      throw new TypeError('[webgpu-search] loadDataset expects string[].');
    }
    if (dataset instanceof ArrayBuffer) {
      const packed = deserializeUnicodeDataset(dataset);
      return { packed, strings: null };
    }
    const d = dataset as unknown as Record<string, unknown>;
    if (d['tokens'] instanceof Uint32Array && d['offsets'] instanceof Uint32Array && typeof d['rowCount'] === 'number') {
      // Trust boundary note: objects straight from our packer are monotonic
      // by construction, so only lengths + endpoints are re-checked here.
      // Untrusted bytes get the full loop in deserializeUnicodeDataset.
      const packed = d as unknown as PackedUnicodeBufferV2;
      const rc = packed.rowCount;
      const tc = packed.tokenCount;
      if (!(packed.offsets.length === rc + 1) || !(packed.tokens.length === tc)) {
        throw new IncompatibleIndexError('packed-shape', `${rc}/${tc}`);
      }
      if ((packed.offsets[0]) !== 0 || (packed.offsets[rc]) !== tc) {
        throw new IncompatibleIndexError(tc, packed.offsets[rc]);
      }
      const strings = Array.isArray(d['strings']) ? (d['strings'] as string[]) : null;
      return { packed, strings };
    }
    if (Array.isArray(d['strings'])) {
      const strings = d['strings'] as string[];
      const size = d['size'];
      if (typeof size === 'number' && size !== strings.length) {
        throw new IncompatibleIndexError(strings.length, size);
      }
      const packed = packUnicodeToGPUBuffer(strings, { folded: this.folded });
      return { packed, strings };
    }
    // Legacy v0.1 byte buffers carry no U2F2 magic — fail closed, rebuild.
    const actual = typeof d['byteLength'] === 'number' ? d['byteLength'] : (d['size'] ?? 'legacy-v0.1');
    throw new IncompatibleIndexError(0x55324632, actual);
  }

  private async loadDatasetInternal(dataset: EngineDataset): Promise<{ uploadTimeMs: number }> {
    this.generation++;
    const { packed, strings } = this.resolvePacked(dataset);
    const count = packed.rowCount;

    this.currentDatasetSize = count;
    this.currentStrings = strings;
    this.currentTokens = packed.tokens;
    this.currentOffsets = packed.offsets;
    this.folded = packed.folded;
    this.profileId = packed.profileId;
    this.unicodeVersion = packed.unicodeVersion;
    this.scoringVersion = packed.scoringVersion;
    // Retained CPU-side for re-upload after device loss + graceful fallback.
    void this.currentTokens;
    void this.currentOffsets;

    if (!this.device) {
      return { uploadTimeMs: 0 };
    }

    // Phase 2 (authoritative post-init): exact per-buffer check vs real
    // device.limits before allocating. Phase 1 (hybrid create()) fail-fasts
    // pre-init with the exact post-fold tokenCount; acquiring a device just
    // to check would churn GPU contexts (rejected: acquire-always), while
    // allocation-time-only would OOM mid-load leaving half-state (rejected).
    const avgBytes = count === 0 ? 0 : packed.recordsByteLength / count;
    const budget = checkMemoryBudget(count, avgBytes, this.device, this.candidateCapacity);
    if (!budget.allowed) {
      throw new Error(`[webgpu-search] ${budget.reason} Falling back to CPU.`);
    }

    for (const b of [this.offsetsBuffer, this.recordsBuffer]) {
      try { b?.destroy(); } catch {}
    }
    this.offsetsBuffer = this.recordsBuffer = null;

    const t0 = nowMs();

    this.offsetsBuffer = this.device.createBuffer({
      label: `Offsets (${count})`,
      size: Math.max(packed.offsetsByteLength, 16),
      usage: BufferUsage.STORAGE | BufferUsage.COPY_DST
    });

    this.recordsBuffer = this.device.createBuffer({
      label: `Records (${count})`,
      size: Math.max(packed.recordsByteLength, 16),
      usage: BufferUsage.STORAGE | BufferUsage.COPY_DST
    });

    this.device.queue.writeBuffer(this.offsetsBuffer, 0, packed.offsetsBufferData);
    if (packed.recordsByteLength > 0) {
      this.device.queue.writeBuffer(this.recordsBuffer, 0, packed.recordsBufferData);
    }
    await this.device.queue.onSubmittedWorkDone();

    const uploadTimeMs = nowMs() - t0;

    return { uploadTimeMs };
  }

  async search(query: string, options: SearchOptions): Promise<WebGPUSearchResult> {
    return this.queued(() => this.searchInternal(query, options));
  }

  private async searchInternal(query: string, options: SearchOptions): Promise<WebGPUSearchResult> {
    const mode = options.mode ?? 'fuzzy';
    const limit = clampLimit(options.limit ?? options.maxResults ?? 50);

    const noHits = (q: string): WebGPUSearchResult => ({
      query: q, mode, totalMatches: 0, candidateCount: 0, hasOverflow: false,
      results: [], timings: { queryUploadMs: 0, encodeSubmitMs: 0, gpuExecutionMs: null, readbackMs: 0, totalMs: 0, gpuDispatchMs: 0 },
    });

    if (!this.device || !this.recordsBuffer || !this.offsetsBuffer || !this.uniformBuffer || !this.queryBuffer || !this.outputBuffer || !this.stagingBuffer) {
      return noHits(query);
    }

    throwIfAborted(options.signal);

    // Defensive post-fold gate (hybrid owns the exact gate + cpu-fallback;
    // engine never silently falls back — it throws).
    const nq = normalizeText(query, this.folded);
    if (nq.isEmpty) {
      return noHits('');
    }
    if (nq.tokenCount > QUERY_TOKENS_MAX) {
      throw new QueryTooLongError(QUERY_TOKENS_MAX, nq.tokenCount, this.profileId);
    }
    if (options.caseSensitive !== undefined && (options.caseSensitive === this.folded)) {
      throw new ProfileMismatchError(!this.folded, options.caseSensitive);
    }

    // Zero-dispatch fast path: no rows means no compute work to submit.
    if (this.currentDatasetSize === 0) {
      return noHits(query);
    }

    if (limit > this.candidateCapacity) {
      this.allocateOutputBuffers(limit);
    }
    const pipeline = mode === 'fuzzy' ? this.fuzzyPipeline! : this.substringPipeline!;
    const caseSensitive = !!options.caseSensitive;
    const queryLen = nq.tokens.length;

    const totalStart = nowMs();

    const tQueryStart = nowMs();
    this.device.queue.writeBuffer(this.queryBuffer, 0, nq.tokens);
    const udata = new ArrayBuffer(32);
    const U32 = new Uint32Array(udata);
    U32[0] = this.currentDatasetSize;
    U32[1] = queryLen;
    U32[2] = this.candidateCapacity;
    // flagsAndProfile: caseSensitive:1b + folded:1b + profile:6b + unicode:8b + scoring:8b (§2.5).
    const pEnum = (PROFILE_TO_ENUM as Record<string, number>)[this.profileId] ?? 0;
    const uEnum = (UNICODE_VERSION_TO_ENUM as Record<string, number>)[this.unicodeVersion] ?? 0;
    const sEnum = (SCORING_TO_ENUM as Record<string, number>)[this.scoringVersion] ?? 0;
    U32[3] = (((caseSensitive ? 1 : 0) | (this.folded ? 2 : 0) | ((pEnum & 0x3f) << 2) | ((uEnum & 0xff) << 8) | ((sEnum & 0xff) << 16)) >>> 0);
    U32[4] = 0;
    U32[5] = 0;
    U32[6] = 0;
    U32[7] = 0;
    this.device.queue.writeBuffer(this.uniformBuffer, 0, udata);
    const queryUploadMs = nowMs() - tQueryStart;

    throwIfAborted(options.signal);

    const tDispatchStart = nowMs();
    let skipTimestamps = false;

    const bg = this.device.createBindGroup({
      label: 'search-bg',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: { buffer: this.offsetsBuffer } },
        { binding: 2, resource: { buffer: this.recordsBuffer } },
        { binding: 3, resource: { buffer: this.queryBuffer } },
        { binding: 4, resource: { buffer: this.outputBuffer } }
      ]
    });

    // Chunked dispatch: when ceil(rows/128) exceeds the device dimension
    // limit, each chunk re-writes the uniform (_pad.x = base row) and submits
    // its own pass; the shared atomic counter accumulates. Fast path keeps
    // timestamps.
    const md = (this.device.limits as unknown as Record<string, unknown>).maxComputeWorkgroupsPerDimension;
    const maxDim = typeof md === 'number' ? md : 65535;
    const workgroupSize = 128;
    const totalWorkgroups = Math.ceil(this.currentDatasetSize / workgroupSize);

    if (totalWorkgroups <= maxDim) {
      const passDesc: GPUComputePassDescriptor = {
        label: 'search-pass'
      };
      if (this.querySet) {
        (passDesc as any).timestampWrites = {
          querySet: this.querySet,
          beginningOfPassWriteIndex: 0,
          endOfPassWriteIndex: 1
        };
      }
      const enc = this.device.createCommandEncoder({ label: 'search' });

      if (typeof (enc as any).clearBuffer === 'function') {
        enc.clearBuffer(this.outputBuffer, 0, this.outputByteLength);
      }

      const pass = enc.beginComputePass(passDesc);
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(totalWorkgroups);
      pass.end();

      if (this.querySet && this.queryResolveBuffer && this.queryStagingBuffer) {
        enc.resolveQuerySet(this.querySet, 0, 2, this.queryResolveBuffer, 0);
        enc.copyBufferToBuffer(this.queryResolveBuffer, 0, this.queryStagingBuffer, 0, 16);
      }

      enc.copyBufferToBuffer(this.outputBuffer, 0, this.stagingBuffer, 0, this.outputByteLength);
      this.device.queue.submit([enc.finish()]);
    } else {
      const step = maxDim * workgroupSize;
      const clr = this.device.createCommandEncoder({ label: 'clear' });
      if (typeof (clr as any).clearBuffer === 'function') {
        (clr as any).clearBuffer(this.outputBuffer, 0, this.outputByteLength);
        this.device.queue.submit([clr.finish()]);
      }
      let base = 0;
      let left = this.currentDatasetSize;
      while (left > 0) {
        const n = left > step ? step : left;
        const wgs = Math.ceil(n / workgroupSize);
        U32[4] = base;
        this.device.queue.writeBuffer(this.uniformBuffer, 0, udata);
        const chunkEnc = this.device.createCommandEncoder({ label: 'chunk' });
        const chunkPass = chunkEnc.beginComputePass({ label: 'chunk-pass' });
        chunkPass.setPipeline(pipeline);
        chunkPass.setBindGroup(0, bg);
        chunkPass.dispatchWorkgroups(wgs);
        chunkPass.end();
        this.device.queue.submit([chunkEnc.finish()]);
        base += n;
        left -= n;
      }
      const cp = this.device.createCommandEncoder({ label: 'copy' });
      cp.copyBufferToBuffer(this.outputBuffer, 0, this.stagingBuffer, 0, this.outputByteLength);
      this.device.queue.submit([cp.finish()]);
      // Chunked corpora exceed any latency budget; timestamps skipped (null).
      skipTimestamps = true;
    }
    const encodeSubmitMs = nowMs() - tDispatchStart;

    throwIfAborted(options.signal);

    const tReadbackStart = nowMs();
    let gpuExecutionMs: number | null = null;

    if (this.queryStagingBuffer && !skipTimestamps) {
      try {
        await this.queryStagingBuffer.mapAsync(MapMode.READ);
        const tu = new BigUint64Array(this.queryStagingBuffer.getMappedRange());
        if (tu[1] >= (tu[0]) && (tu[0]) > 0n) {
          gpuExecutionMs = Number((tu[1]) - (tu[0])) / 1_000_000;
        }
      } catch (tsErr) {
        console.warn('timestamp read failed:', tsErr);
      } finally {
        try { this.queryStagingBuffer.unmap(); } catch {}
      }
    }

    let totalMatches = 0;
    let candidateCount = 0;
    let hasOverflow = false;
    let candidates: SearchResultItem[] = [];

    // Generation-epoch abort for in-flight mapAsync (takes no signal, so
    // abort is cooperative discard). Capture before await; on mismatch unmap
    // and discard (stale dataset / torn-down engine). Rejected: bare
    // Promise.race — abandons the mapping, later destroy-while-mapped error.
    const gen = this.generation;
    await this.stagingBuffer.mapAsync(MapMode.READ);
    try {
      if (gen !== this.generation) {
        try { this.stagingBuffer.unmap(); } catch {}
        throw abortError();
      }
      throwIfAborted(options.signal);
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
        // Token-only datasets carry no strings: resolve `text` to `''`
        // (documented degradation; see DatasetLike).
        candidates[i] = {
          index,
          score,
          text: this.currentStrings ? (this.currentStrings[index] ?? '') : ''
        };
      }
    } finally {
      try { this.stagingBuffer.unmap(); } catch {}
    }

    throwIfAborted(options.signal);

    const readbackMs = nowMs() - tReadbackStart;
    const totalMs = nowMs() - totalStart;

    candidates.sort(compareParityResults);
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
   * Cold search: upload + search in a single mutex acquisition (A never
   * observes B's rows — guaranteed by the single-mutex interleave rule).
   */
  async searchCold(dataset: EngineDataset, query: string, options: SearchOptions): Promise<ColdSearchResult> {
    return this.queued(async () => {
      const tUploadStart = nowMs();
      await this.loadDatasetInternal(dataset);
      const datasetUploadMs = nowMs() - tUploadStart;
      const warmResult = await this.searchInternal(query, options);
      return { ...warmResult, datasetUploadMs, coldTotalMs: datasetUploadMs + warmResult.timings.totalMs };
    });
  }

  destroy() {
    // Epoch bump first so in-flight mapAsync continuations discard (see
    // readback site; rejected: Promise.race abandons the mapping). Unmap
    // (never destroy a mapped buffer), destroy, then park a barrier on the
    // mutex so queued ops observe the torn-down state.
    this.generation++;
    try { this.stagingBuffer?.unmap(); } catch {}
    try { this.queryStagingBuffer?.unmap(); } catch {}
    const bufs = [this.offsetsBuffer, this.recordsBuffer, this.uniformBuffer, this.queryBuffer, this.outputBuffer, this.stagingBuffer, this.queryResolveBuffer, this.queryStagingBuffer];
    for (const b of bufs) { try { b?.destroy(); } catch {} }
    this.offsetsBuffer = this.recordsBuffer = this.uniformBuffer = this.queryBuffer = null;
    this.outputBuffer = this.stagingBuffer = this.queryResolveBuffer = this.queryStagingBuffer = null;
    if (this.querySet) {
      try { this.querySet.destroy(); } catch {}
      this.querySet = null;
    }
    this.substringPipeline = null;
    this.fuzzyPipeline = null;
    this.currentStrings = null;
    this.currentTokens = null;
    this.currentOffsets = null;
    this.currentDatasetSize = 0;
    if (this.device && !this.deviceReleased) {
      this.deviceReleased = true;
      try {
        WebGPUContextManager.releaseDevice(this.device, this.isSharedDevice);
      } catch {}
    }
    this.device = null;
    const p = this.searchMutex.then(() => {}, () => {});
    this.searchMutex = p.catch(() => {});
  }
}
