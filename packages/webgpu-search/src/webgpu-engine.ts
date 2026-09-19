import SUBSTRING_WGSL from './shaders/substring.wgsl';
import FUZZY_WGSL from './shaders/fuzzy.wgsl';
import { WebGPUContextManager } from './context-manager';
import {
  checkMemoryBudget,
  computeClampedHeadroomBytes,
  deserializeUnicodeDataset,
  packUnicodeToGPUBuffer,
  validatePackedOffsets,
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
  /** @deprecated v0.2: ignored when `strings` is present, rejected otherwise. Use PackedUnicodeBufferV2. Removal in v0.3. */
  recordsBufferData?: ArrayBuffer;
  /** @deprecated v0.2: see recordsBufferData. */
  recordsByteLength?: number;
  /** @deprecated v0.2: see recordsBufferData. */
  offsetsBufferData?: ArrayBuffer;
  /** @deprecated v0.2: see recordsBufferData. */
  offsetsByteLength?: number;
  /** @deprecated v0.2: legacy v0.1 shape, always rejected. */
  gpuBufferData?: ArrayBuffer;
  /** @deprecated v0.2: legacy v0.1 shape, always rejected. */
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

  private allocatedOffsetsByteLength: number = 0;
  private allocatedRecordsByteLength: number = 0;

  get allocatedOffsetsBytes(): number {
    return this.allocatedOffsetsByteLength;
  }

  get allocatedRecordsBytes(): number {
    return this.allocatedRecordsByteLength;
  }

  get vramAllocatedBytes(): number {
    return this.allocatedOffsetsByteLength + this.allocatedRecordsByteLength;
  }

  canFitHeadroom(requiredRows: number, requiredTokens: number): boolean {
    const needOffsets = (requiredRows + 1) * 4;
    const needRecords = requiredTokens * 4;
    return (
      this.offsetsBuffer !== null &&
      this.recordsBuffer !== null &&
      needOffsets <= this.allocatedOffsetsByteLength &&
      needRecords <= this.allocatedRecordsByteLength
    );
  }

  isHeadroomExhausted(requiredRows: number, requiredTokens: number): boolean {
    return !this.canFitHeadroom(requiredRows, requiredTokens);
  }

  async init(customDevice?: GPUDevice): Promise<boolean> {
    // Guard re-entry: dispose existing GPU buffers before re-creating so
    // init() twice does not leak the first set (benchmark/power users).
    if (this.device) {
      this.disposeGpuBuffers();
    }
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
    // Capture identity: re-init() attaches a new handler per device; a stale
    // handler for a previous device must not tear down the live engine.
    const mine = this.device;
    dev.lost.then(() => {
      if (this.device !== mine) return;
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
    }, () => {});
  }

  private disposeGpuBuffers(): void {
    for (const b of [this.offsetsBuffer, this.recordsBuffer, this.uniformBuffer, this.queryBuffer, this.outputBuffer, this.stagingBuffer, this.queryResolveBuffer, this.queryStagingBuffer]) {
      try { (b as GPUBuffer | null)?.unmap?.(); } catch {}
      try { (b as GPUBuffer | null)?.destroy(); } catch {}
    }
    this.offsetsBuffer = this.recordsBuffer = this.uniformBuffer = this.queryBuffer = null;
    this.outputBuffer = this.stagingBuffer = this.queryResolveBuffer = this.queryStagingBuffer = null;
    if (this.querySet) {
      try { this.querySet.destroy(); } catch {}
      this.querySet = null;
    }
    this.substringPipeline = this.fuzzyPipeline = null;
    this.allocatedOffsetsByteLength = 0;
    this.allocatedRecordsByteLength = 0;
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

  /**
   * Current candidate capacity allocated for the output buffer.
   */
  get currentCandidateCapacity(): number {
    return this.candidateCapacity;
  }

  /**
   * Ensure candidate capacity is scaled to at least the requested capacity
   * (clamped between 8,192 and 32,768).
   */
  ensureCandidateCapacity(capacity: number): void {
    const safeCap = typeof capacity === 'number' && Number.isFinite(capacity) && capacity > 0
      ? Math.floor(capacity)
      : 8192;
    const clamped = Math.min(32768, Math.max(8192, safeCap));
    if (clamped !== this.candidateCapacity || !this.outputBuffer) {
      this.candidateCapacity = clamped;
      if (this.device) {
        this.allocateOutputBuffers(clamped);
      }
    }
  }

  allocateOutputBuffers(candidateCapacity: number = 8192): void {
    if (!this.device) return;
    const safeCap = typeof candidateCapacity === 'number' && Number.isFinite(candidateCapacity) && candidateCapacity > 0
      ? Math.floor(candidateCapacity)
      : 8192;
    const clampedCap = Math.min(32768, Math.max(safeCap, 8192));
    if (this.outputBuffer && clampedCap <= this.candidateCapacity && this.outputByteLength >= 8 + clampedCap * 8) return;
    this.candidateCapacity = clampedCap;
    this.outputByteLength = 8 + this.candidateCapacity * 8;

    // Advance generation to discard any in-flight reads against old staging buffers
    this.generation++;

    // Unmap-before-destroy (symmetric with destroy()): destroying a mapped
    // buffer throws / leaves torn state on some implementations.
    for (const b of [this.outputBuffer, this.stagingBuffer]) {
      try { (b as GPUBuffer | null)?.unmap?.(); } catch {}
    }
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

  async loadDataset(
    dataset: EngineDataset,
    options?: { rowCapacity?: number; tokenCapacity?: number; growthFactor?: number }
  ): Promise<{ uploadTimeMs: number }> {
    return this.queued(() => this.loadDatasetInternal(dataset, options));
  }

  private resolvePacked(dataset: EngineDataset): { packed: PackedUnicodeBufferV2; strings: string[] | null } {
    if (Array.isArray(dataset)) {
      const arr = dataset as unknown[];
      if (arr.length === 0 || typeof arr[0] === 'string') {
        const packed = packUnicodeToGPUBuffer(arr as string[], { folded: this.folded });
        return { packed, strings: (arr as string[]).slice() };
      }
      // Symmetric with packUnicodeToGPUBuffer: pre-tokenized Uint32Array[]
      // takes the zero-renorm path (previously rejected here).
      if (arr[0] instanceof Uint32Array || (typeof (arr[0] as any)?.length === 'number' && typeof (arr[0] as any)?.set === 'function')) {
        const packed = packUnicodeToGPUBuffer(arr as unknown as Uint32Array[], { folded: this.folded });
        return { packed, strings: null };
      }
      throw new TypeError('[webgpu-search] loadDataset expects string[] or Uint32Array[].');
    }
    // Duck-type ArrayBuffer for cross-realm buffers (iframe/worker) and
    // Node Buffer-backed views: accept byteLength+slice instead of instanceof.
    const asBuf = dataset as unknown as { byteLength?: unknown; slice?: unknown };
    if (typeof asBuf?.byteLength === 'number' && typeof asBuf?.slice === 'function' && !(dataset as any).tokens) {
      const packed = deserializeUnicodeDataset(dataset as unknown as ArrayBuffer);
      return { packed, strings: null };
    }
    const d = dataset as unknown as Record<string, unknown>;
    const tokensLike = d['tokens'] as unknown;
    const offsetsLike = d['offsets'] as unknown;
    const isU32View = (v: unknown): v is Uint32Array =>
      v instanceof Uint32Array || (typeof (v as any)?.length === 'number' && (v as any)?.constructor?.name === 'Uint32Array');
    if (isU32View(tokensLike) && isU32View(offsetsLike) && typeof d['rowCount'] === 'number') {
      // Fail-closed: run the same monotonicity + terminal + bounds loop as
      // deserializeUnicodeDataset. Interior entries are NOT trusted — a
      // crafted offsets array would otherwise underflow (t1-t0 wraps u32)
      // and hang the dispatch / OOB-read in WGSL.
      const packed = d as unknown as PackedUnicodeBufferV2;
      const rc = packed.rowCount;
      const tc = packed.tokenCount;
      if (!Number.isInteger(rc) || !Number.isInteger(tc) || rc < 0 || tc < 0) {
        throw new IncompatibleIndexError('packed-shape', `${rc}/${tc}`);
      }
      validatePackedOffsets(packed.offsets, rc, tc);
      if (packed.tokens.length !== tc) {
        throw new IncompatibleIndexError('packed-shape', `${rc}/${tc}`);
      }
      // Derive byte lengths from the views (don't trust caller fields, which
      // would otherwise cause short upload or writeBuffer RangeError).
      const wantRecords = tc * 4;
      const wantOffsets = (rc + 1) * 4;
      if (packed.recordsByteLength !== wantRecords || packed.offsetsByteLength !== wantOffsets) {
        throw new IncompatibleIndexError(`${wantRecords}/${wantOffsets}`, `${packed.recordsByteLength}/${packed.offsetsByteLength}`);
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

  private async loadDatasetInternal(
    dataset: EngineDataset,
    options?: { rowCapacity?: number; tokenCapacity?: number; growthFactor?: number }
  ): Promise<{ uploadTimeMs: number }> {
    this.generation++;
    const { packed, strings } = this.resolvePacked(dataset);
    const count = packed.rowCount;

    // Snapshot CPU metadata so budget/OOM throws restore instead of leaving
    // phantom sizes with null buffers (half-state).
    const prev = {
      size: this.currentDatasetSize,
      strings: this.currentStrings,
      tokens: this.currentTokens,
      offsets: this.currentOffsets,
      folded: this.folded,
      profileId: this.profileId,
      unicodeVersion: this.unicodeVersion,
      scoringVersion: this.scoringVersion,
    };
    this.currentDatasetSize = count;
    this.currentStrings = strings;
    this.currentTokens = packed.tokens;
    this.currentOffsets = packed.offsets;
    this.folded = packed.folded;
    this.profileId = packed.profileId;
    this.unicodeVersion = packed.unicodeVersion;
    this.scoringVersion = packed.scoringVersion;

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
      Object.assign(this, {
        currentDatasetSize: prev.size,
        currentStrings: prev.strings,
        currentTokens: prev.tokens,
        currentOffsets: prev.offsets,
        folded: prev.folded,
        profileId: prev.profileId,
        unicodeVersion: prev.unicodeVersion,
        scoringVersion: prev.scoringVersion,
      });
      throw new Error(`[webgpu-search] ${budget.reason} Falling back to CPU.`);
    }

    const reqOffsets = Math.max(packed.offsetsByteLength, 16);
    const reqRecords = Math.max(packed.recordsByteLength, 16);

    const targetOffsets = computeClampedHeadroomBytes(
      Math.max((options?.rowCapacity !== undefined ? options.rowCapacity + 1 : count + 1) * 4, reqOffsets),
      { growthFactor: options?.growthFactor, device: this.device }
    );
    const targetRecords = computeClampedHeadroomBytes(
      Math.max((options?.tokenCapacity !== undefined ? options.tokenCapacity : packed.tokenCount) * 4, reqRecords),
      { growthFactor: options?.growthFactor, device: this.device }
    );

    const allocOffsets = Math.max(reqOffsets, targetOffsets);
    const allocRecords = Math.max(reqRecords, targetRecords);

    const t0 = nowMs();
    let newOffsetsBuffer: GPUBuffer | null = null;
    let newRecordsBuffer: GPUBuffer | null = null;

    try {
      newOffsetsBuffer = this.device.createBuffer({
        label: `Offsets (${count})`,
        size: allocOffsets,
        usage: BufferUsage.STORAGE | BufferUsage.COPY_DST
      });

      newRecordsBuffer = this.device.createBuffer({
        label: `Records (${count})`,
        size: allocRecords,
        usage: BufferUsage.STORAGE | BufferUsage.COPY_DST
      });

      this.device.queue.writeBuffer(newOffsetsBuffer, 0, packed.offsetsBufferData);
      if (packed.recordsByteLength > 0) {
        this.device.queue.writeBuffer(newRecordsBuffer, 0, packed.recordsBufferData);
      }

      // Safe swap: only unmap/destroy previous buffers after new allocations succeed
      for (const b of [this.offsetsBuffer, this.recordsBuffer]) {
        try { (b as GPUBuffer | null)?.unmap?.(); } catch {}
        try { b?.destroy(); } catch {}
      }
      this.offsetsBuffer = newOffsetsBuffer;
      this.recordsBuffer = newRecordsBuffer;
      this.allocatedOffsetsByteLength = allocOffsets;
      this.allocatedRecordsByteLength = allocRecords;
    } catch (allocErr) {
      try { newOffsetsBuffer?.destroy(); } catch {}
      try { newRecordsBuffer?.destroy(); } catch {}
      // Restore previous CPU metadata so stats don't lie after OOM.
      Object.assign(this, {
        currentDatasetSize: prev.size,
        currentStrings: prev.strings,
        currentTokens: prev.tokens,
        currentOffsets: prev.offsets,
        folded: prev.folded,
        profileId: prev.profileId,
        unicodeVersion: prev.unicodeVersion,
        scoringVersion: prev.scoringVersion,
      });
      throw allocErr;
    }

    const uploadTimeMs = nowMs() - t0;

    return { uploadTimeMs };
  }

  async appendRows(
    newTokens: Uint32Array,
    newOffsets: Uint32Array,
    newTotalRows: number
  ): Promise<void> {
    return this.queued(async () => {
      if (!this.device || !this.offsetsBuffer || !this.recordsBuffer) {
        throw new Error('[webgpu-search] Cannot append rows: GPU buffers not initialized.');
      }
      const prevRows = this.currentDatasetSize;
      const prevTokens = this.currentTokens ? this.currentTokens.length : 0;
      const needOffsetsBytes = (prevRows + 1) * 4 + newOffsets.byteLength;
      const needRecordsBytes = prevTokens * 4 + newTokens.byteLength;
      if (needOffsetsBytes > this.allocatedOffsetsByteLength || needRecordsBytes > this.allocatedRecordsByteLength) {
        throw new Error('[webgpu-search] appendRows exceeds allocated buffer headroom.');
      }
      this.generation++;

      // Write new offsets: row offsets starting at (prevRows + 1) * 4
      this.device.queue.writeBuffer(this.offsetsBuffer, (prevRows + 1) * 4, newOffsets);
      // Write new tokens at prevTokens * 4
      if (newTokens.byteLength > 0) {
        this.device.queue.writeBuffer(this.recordsBuffer, prevTokens * 4, newTokens);
      }

      const combinedTokens = new Uint32Array(prevTokens + newTokens.length);
      if (this.currentTokens) combinedTokens.set(this.currentTokens, 0);
      combinedTokens.set(newTokens, prevTokens);
      this.currentTokens = combinedTokens;

      const combinedOffsets = new Uint32Array(newTotalRows + 1);
      if (this.currentOffsets) combinedOffsets.set(this.currentOffsets, 0);
      combinedOffsets.set(newOffsets, prevRows + 1);
      this.currentOffsets = combinedOffsets;

      this.currentDatasetSize = newTotalRows;
    });
  }

  async search(query: string, options: SearchOptions): Promise<WebGPUSearchResult> {
    return this.queued(() => this.searchInternal(query, options));
  }

  private async searchInternal(query: string, options: SearchOptions): Promise<WebGPUSearchResult> {
    const rawMode = options.mode ?? 'fuzzy';
    if (rawMode !== 'fuzzy' && rawMode !== 'substring') {
      throw new TypeError(`[webgpu-search] search mode must be 'fuzzy'|'substring', got ${String(rawMode)}.`);
    }
    const mode = rawMode;
    const limit = clampLimit(options.limit ?? options.maxResults ?? 50);
    // Strict boolean gate (matches hybrid-index default caseSensitive=false):
    // forged truthy (1, 'true') fails closed instead of bypassing then
    // dispatching as the opposite polarity.
    if (options.caseSensitive !== undefined && typeof options.caseSensitive !== 'boolean') {
      throw new TypeError(`[webgpu-search] search caseSensitive must be boolean, got ${typeof options.caseSensitive}.`);
    }

    const noHits = (q: string): WebGPUSearchResult => ({
      query: q, mode, totalMatches: 0, candidateCount: 0, hasOverflow: false,
      results: [], timings: { queryUploadMs: 0, encodeSubmitMs: 0, gpuExecutionMs: null, readbackMs: 0, totalMs: 0, gpuDispatchMs: 0 },
    });

    throwIfAborted(options.signal);

    // Defensive post-fold gates run BEFORE the no-device early return so
    // direct-engine callers get identical throw/echo semantics with or
    // without a device (hybrid already gates, but engine must not diverge).
    // Engine never silently falls back — it throws.
    const nq = normalizeText(query, this.folded);
    if (nq.isEmpty) {
      return noHits('');
    }
    if (nq.tokenCount > QUERY_TOKENS_MAX) {
      throw new QueryTooLongError(QUERY_TOKENS_MAX, nq.tokenCount, this.profileId);
    }
    // Default omitted flag to false (same as hybrid-index) so direct-engine
    // callers get identical ProfileMismatch semantics (folded=false index +
    // omitted flag throws, instead of silently running case-sensitive).
    const qcs = options.caseSensitive ?? false;
    if (qcs === this.folded) {
      throw new ProfileMismatchError(!this.folded, options.caseSensitive);
    }

    if (!this.device || !this.recordsBuffer || !this.offsetsBuffer || !this.uniformBuffer || !this.queryBuffer || !this.outputBuffer || !this.stagingBuffer || !this.substringPipeline || !this.fuzzyPipeline) {
      return noHits(query);
    }

    // Zero-dispatch fast path: no rows means no compute work to submit.
    // Note: empty already returned above with unified echo ''.
    if (this.currentDatasetSize === 0) {
      return noHits(query);
    }

    const pipeline = mode === 'fuzzy' ? this.fuzzyPipeline! : this.substringPipeline!;
    const caseSensitive = qcs;
    const queryLen = nq.tokens.length;

    const totalStart = nowMs();

    const tQueryStart = nowMs();
    this.device.queue.writeBuffer(this.queryBuffer, 0, nq.tokens);
    const udata = new ArrayBuffer(32);
    const U32 = new Uint32Array(udata);
    U32[0] = this.currentDatasetSize;
    U32[1] = queryLen;
    U32[2] = this.candidateCapacity;
    // flagsAndProfile: caseSensitive:1b + folded:1b + profile:6b + unicode:8b
    // + scoring:8b. Reserved for M4 harness/debug — shaders are pure-`==` by
    // construction and do not read it (host enforces ProfileMismatchError).
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
    // timestamps. Queue-ordering dependency: WebGPU guarantees serial
    // queue execution, so each dispatch samples only prior uniform writes
    // (chunk counts are tiny in practice: step ≈ 8.4M rows/chunk).
    const md = (this.device.limits as unknown as Record<string, unknown>).maxComputeWorkgroupsPerDimension;
    const maxDim = typeof md === 'number' && Number.isFinite(md) && md >= 1 ? Math.floor(md) : 65535;
    const workgroupSize = 128;
    const totalWorkgroups = Math.ceil(this.currentDatasetSize / workgroupSize);

    // Zero the atomic counter. clearBuffer may not exist on older Safari —
    // fall back to an 8-byte zero write (count + _pad0), queue-ordered
    // before the dispatch. Never skip the reset (stale count corruption).
    const zeroCounterFallback = () => {
      this.device!.queue.writeBuffer(this.outputBuffer!, 0, new Uint8Array(8));
    };

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
      } else {
        // No clearBuffer: encode the zero via copy path — submit the
        // queue write before the compute submit (serial ordering).
        zeroCounterFallback();
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
      } else {
        zeroCounterFallback();
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
    // mapAsync rejection itself (e.g. destroy raced it via unmap) also maps
    // to AbortError when the epoch moved, so hybrid doesn't misclassify it
    // as fallback-eligible GPU failure.
    const gen = this.generation;
    try {
      await this.stagingBuffer.mapAsync(MapMode.READ);
    } catch (mapErr) {
      if (gen !== this.generation) {
        try { this.stagingBuffer.unmap(); } catch {}
        throw abortError();
      }
      throw mapErr;
    }
    try {
      if (gen !== this.generation) {
        try { this.stagingBuffer.unmap(); } catch {}
        throw abortError();
      }
      // Re-check liveness after the await: destroy() nulls device/buffers.
      if (!this.device || !this.stagingBuffer) {
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
    this.allocatedOffsetsByteLength = 0;
    this.allocatedRecordsByteLength = 0;
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
