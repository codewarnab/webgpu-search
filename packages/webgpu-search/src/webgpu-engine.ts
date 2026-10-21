import { GpuDevicePool, assertValidPowerPreference } from './gpu-device-pool';
import {
  checkMemoryBudget,
  computeClampedHeadroomBytes,
  deserializeDataset,
  resolveStorageBufferLimit,
  packDataset,
  validatePackedOffsets,
  type PackedDataset,
} from './dataset-packing';
import { normalizeText } from './text-normalization';
import { compareExactResults } from './exact-scorer';
import { clampLimit, nowMs, throwIfAborted, assertValidMode } from './guard';
import {
  PROFILE_TO_ENUM,
  QUERY_TOKENS_MAX,
  SCORING_TO_ENUM,
  SCORING_VERSION,
  UNICODE_VERSION,
  UNICODE_VERSION_TO_ENUM,
  IncompatibleIndexError,
  IncompatibleOptionError,
  ProfileMismatchError,
  QueryTooLongError,
} from './text-profile';
import { normalizeTypoTolerance } from './search/typo-tolerance';
import { WebGPUSearchError } from './errors';
import type { AdapterInfo, SearchOptions, SearchResultItem, SearchTimings, SearchMode } from './types';
import { createGpuEngineState } from './gpu/state';
import { GpuBufferManager } from './gpu/buffers';
import { GpuPipelineManager } from './gpu/pipelines';
import { GpuDispatcher, WORKGROUP_SIZE } from './gpu/dispatch';

export interface DatasetLike {
  // `strings?` optional: search-index enriches `text` from its own `items`, so
  // forcing strings through the engine duplicates the corpus. Token-only
  // datasets resolve `text` to `''` (documented). Rejected: strings-required
  // (2x residency), placeholder synthesis (dishonest). `string[]` overload
  // routes via `normalizeText` — single code path, no legacy sanitizer.
  size: number;
  strings?: string[];
  /** @deprecated Ignored when `strings` is present, rejected otherwise. Use PackedDataset. */
  recordsBufferData?: ArrayBuffer;
  /** @deprecated See recordsBufferData. */
  recordsByteLength?: number;
  /** @deprecated See recordsBufferData. */
  offsetsBufferData?: ArrayBuffer;
  /** @deprecated See recordsBufferData. */
  offsetsByteLength?: number;
  /** @deprecated Legacy shape, always rejected. */
  gpuBufferData?: ArrayBuffer;
  /** @deprecated Legacy shape, always rejected. */
  byteLength?: number;
}

export type EngineDataset = PackedDataset | ArrayBuffer | DatasetLike | string[];

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

/** Search-engine facade over the gpu/ managers: device pool, CPU dataset metadata, mutex, fail-closed gates. */
export class WebGPUEngine {
  private readonly state = createGpuEngineState();
  private readonly pipelines = new GpuPipelineManager(this.state);
  private readonly buffers = new GpuBufferManager(this.state);
  private readonly dispatch = new GpuDispatcher(this.state);

  private isSharedDevice: boolean = false;
  private deviceReleased: boolean = false;

  // Dataset CPU metadata (engine-owned; buffer writes flow through managers).
  private currentDatasetSize: number = 0;
  private currentStrings: string[] | null = null;
  private currentTokens: Uint32Array | null = null;
  private currentOffsets: Uint32Array | null = null;
  private normalized: boolean = true;
  private profileId: string = 'unicode-default';
  private unicodeVersion: string = UNICODE_VERSION;
  private scoringVersion: string = SCORING_VERSION;
  private searchMutex: Promise<any> = Promise.resolve();

  public adapterInfo: AdapterInfo | null = null;

  get isReady(): boolean {
    return this.state.device !== null;
  }

  get currentSize(): number {
    return this.currentDatasetSize;
  }

  get allocatedOffsetsBytes(): number {
    return this.state.allocatedOffsetsByteLength;
  }

  get allocatedRecordsBytes(): number {
    return this.state.allocatedRecordsByteLength;
  }

  get vramAllocatedBytes(): number {
    return this.state.allocatedOffsetsByteLength + this.state.allocatedRecordsByteLength;
  }

  canFitHeadroom(requiredRows: number, requiredTokens: number): boolean {
    return this.buffers.hasDatasetBufferCapacity(requiredRows, requiredTokens);
  }

  isHeadroomExhausted(requiredRows: number, requiredTokens: number): boolean {
    return !this.canFitHeadroom(requiredRows, requiredTokens);
  }

  async init(
    customDevice?: GPUDevice | { device?: GPUDevice; powerPreference?: GPUPowerPreference }
  ): Promise<boolean> {
    // Guard re-entry: dispose existing GPU buffers AND release the previous
    // device reference before re-acquiring, so init() twice leaks neither the
    // first buffer set nor a pool refcount (benchmark/power users).
    if (this.state.device) {
      this.state.generation++;
      this.buffers.disposeForReinit();
      this.pipelines.drop();
      this.releaseCurrentDevice();
    }
    // Normalize overloads: bare GPUDevice (legacy) or options bag.
    // powerPreference is validated fail-closed (unknown throws) and forwarded
    // to GpuDevicePool.acquireDevice; ignored when a custom device is injected.
    let injected: GPUDevice | undefined;
    let powerPreference: GPUPowerPreference | undefined;
    if (
      customDevice !== undefined &&
      customDevice !== null &&
      typeof customDevice === 'object' &&
      ('powerPreference' in (customDevice as Record<string, unknown>) ||
        ('device' in (customDevice as Record<string, unknown>) &&
          !('queue' in (customDevice as Record<string, unknown>))))
    ) {
      const opts = customDevice as { device?: GPUDevice; powerPreference?: GPUPowerPreference };
      injected = opts.device;
      powerPreference = opts.powerPreference;
    } else if (customDevice !== undefined && customDevice !== null) {
      injected = customDevice as GPUDevice;
    }
    if (injected) {
      assertValidPowerPreference(powerPreference);
      this.state.device = injected;
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
        hasTimestampQuery: injected.features ? injected.features.has('timestamp-query') : false
      };
      this.attachLostHandler();
      return await this.setupOrTeardown();
    }

    const acquired = await GpuDevicePool.acquireDevice(
      powerPreference !== undefined ? { powerPreference } : undefined
    );
    if (!acquired) {
      return false;
    }

    this.state.device = acquired.device;
    this.adapterInfo = acquired.adapterInfo;
    this.isSharedDevice = acquired.isShared;
    this.deviceReleased = false;
    this.attachLostHandler();
    return await this.setupOrTeardown();
  }

  /**
   * Pipeline/buffer setup with fail-closed teardown: a throw (shader
   * compile, pipeline creation, OOM) destroys partial buffers and releases
   * the device reference so a failed init never leaks a pool refcount.
   */
  private async setupOrTeardown(): Promise<boolean> {
    try {
      return await this.setupPipelinesAndBuffers();
    } catch (err) {
      this.destroy();
      throw err;
    }
  }

  /** Release the held device reference exactly once (deviceReleased guard). */
  private releaseCurrentDevice(): void {
    if (this.state.device && !this.deviceReleased) {
      this.deviceReleased = true;
      try {
        GpuDevicePool.releaseDevice(this.state.device, this.isSharedDevice);
      } catch {}
    }
    this.state.device = null;
  }

  private attachLostHandler(): void {
    const dev = this.state.device as unknown as { lost?: Promise<any> } | null;
    if (!dev?.lost?.then) return;
    // Capture identity: re-init() attaches a new handler per device; a stale
    // handler for a previous device must not tear down the live engine.
    const mine = this.state.device;
    dev.lost.then(() => {
      if (this.state.device !== mine) return;
      // Device loss: null handles, mark unready. Shared refCount decremented
      // once (deviceReleased guard) so live siblings keep the device.
      this.pipelines.drop();
      this.buffers.dropDeviceRefs();
      this.releaseCurrentDevice();
      this.state.generation++;
    }, () => {});
  }

  private async setupPipelinesAndBuffers(): Promise<boolean> {
    if (!this.state.device) return false;

    // Verbatim original order: substring module → fuzzy module → substring pipeline → fuzzy pipeline → uniform → query → timestamps → output buffers (mock/spy compatibility).
    await this.pipelines.setup();

    this.buffers.createFixedBuffers();
    this.buffers.createTimestampResources();

    this.buffers.allocateOutputBuffers(this.state.candidateCapacity);
    return true;
  }

  /**
   * Current candidate capacity allocated for the output buffer.
   */
  get currentCandidateCapacity(): number {
    return this.state.candidateCapacity;
  }

  /**
   * Ensure candidate capacity is scaled to at least the requested capacity
   * (clamped between 8,192 and 32,768).
   */
  ensureCandidateCapacity(capacity: number): void {
    this.buffers.ensureCandidateCapacity(capacity);
  }

  allocateOutputBuffers(candidateCapacity: number = 8192): void {
    this.buffers.allocateOutputBuffers(candidateCapacity);
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

  private resolvePacked(dataset: EngineDataset): { packed: PackedDataset; strings: string[] | null } {
    if (Array.isArray(dataset)) {
      const arr = dataset as unknown[];
      if (arr.length === 0 || typeof arr[0] === 'string') {
        const packed = packDataset(arr as string[], { normalized: this.normalized });
        return { packed, strings: (arr as string[]).slice() };
      }
      // Symmetric with packDataset: pre-tokenized Uint32Array[]
      // takes the zero-renorm path (previously rejected here).
      if (arr[0] instanceof Uint32Array || (typeof (arr[0] as any)?.length === 'number' && typeof (arr[0] as any)?.set === 'function')) {
        const packed = packDataset(arr as unknown as Uint32Array[], { normalized: this.normalized });
        return { packed, strings: null };
      }
      throw new TypeError('[webgpu-search] loadDataset expects string[] or Uint32Array[].');
    }
    // Duck-type ArrayBuffer for cross-realm buffers (iframe/worker) and
    // Node Buffer-backed views: accept byteLength+slice instead of instanceof.
    const asBuf = dataset as unknown as { byteLength?: unknown; slice?: unknown };
    if (typeof asBuf?.byteLength === 'number' && typeof asBuf?.slice === 'function' && !(dataset as any).tokens) {
      const packed = deserializeDataset(dataset as unknown as ArrayBuffer);
      return { packed, strings: null };
    }
    const d = dataset as unknown as Record<string, unknown>;
    const tokensLike = d['tokens'] as unknown;
    const offsetsLike = d['offsets'] as unknown;
    const isU32View = (v: unknown): v is Uint32Array =>
      v instanceof Uint32Array || (typeof (v as any)?.length === 'number' && (v as any)?.constructor?.name === 'Uint32Array');
    if (isU32View(tokensLike) && isU32View(offsetsLike) && typeof d['rowCount'] === 'number') {
      // Fail-closed: run the same monotonicity + terminal + bounds loop as
      // deserializeDataset. Interior entries are NOT trusted — a
      // crafted offsets array would otherwise underflow (t1-t0 wraps u32)
      // and hang the dispatch / OOB-read in WGSL.
      const packed = d as unknown as PackedDataset;
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
      const packed = packDataset(strings, { normalized: this.normalized });
      return { packed, strings };
    }
    // Legacy byte buffers carry no dataset magic — fail closed, rebuild.
    const actual = typeof d['byteLength'] === 'number' ? d['byteLength'] : (d['size'] ?? 'legacy');
    throw new IncompatibleIndexError(0x55324632, actual);
  }

  private async loadDatasetInternal(
    dataset: EngineDataset,
    options?: { rowCapacity?: number; tokenCapacity?: number; growthFactor?: number }
  ): Promise<{ uploadTimeMs: number }> {
    this.state.generation++;
    const { packed, strings } = this.resolvePacked(dataset);
    const count = packed.rowCount;

    // Snapshot CPU metadata so budget/OOM throws restore instead of leaving
    // phantom sizes with null buffers (half-state).
    const prev = {
      size: this.currentDatasetSize,
      strings: this.currentStrings,
      tokens: this.currentTokens,
      offsets: this.currentOffsets,
      normalized: this.normalized,
      profileId: this.profileId,
      unicodeVersion: this.unicodeVersion,
      scoringVersion: this.scoringVersion,
    };
    this.currentDatasetSize = count;
    this.currentStrings = strings;
    this.currentTokens = packed.tokens;
    this.currentOffsets = packed.offsets;
    this.normalized = packed.normalized ?? packed.folded;
    this.profileId = packed.profileId;
    this.unicodeVersion = packed.unicodeVersion;
    this.scoringVersion = packed.scoringVersion;

    if (!this.state.device) {
      Object.assign(this, {
        currentDatasetSize: prev.size,
        currentStrings: prev.strings,
        currentTokens: prev.tokens,
        currentOffsets: prev.offsets,
        normalized: prev.normalized,
        profileId: prev.profileId,
        unicodeVersion: prev.unicodeVersion,
        scoringVersion: prev.scoringVersion,
      });
      throw new WebGPUSearchError('[webgpu-search] loadDataset requires an initialized GPU device. Call init() first.');
    }

    // Staged validation (authoritative post-init): exact per-buffer check vs real
    // device.limits before allocating. Staged validation (search-index create()) fail-fasts
    // pre-init with the exact post-normalization tokenCount; acquiring a device just
    // to check would churn GPU contexts (rejected: acquire-always), while
    // allocation-time-only would OOM mid-load leaving half-state (rejected).
    const avgBytes = count === 0 ? 0 : packed.recordsByteLength / count;
    const budget = checkMemoryBudget(count, avgBytes, this.state.device, this.state.candidateCapacity);
    if (!budget.allowed) {
      Object.assign(this, {
        currentDatasetSize: prev.size,
        currentStrings: prev.strings,
        currentTokens: prev.tokens,
        currentOffsets: prev.offsets,
        normalized: prev.normalized,
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
      { growthFactor: options?.growthFactor, device: this.state.device }
    );
    const targetRecords = computeClampedHeadroomBytes(
      Math.max((options?.tokenCapacity !== undefined ? options.tokenCapacity : packed.tokenCount) * 4, reqRecords),
      { growthFactor: options?.growthFactor, device: this.state.device }
    );

    const allocOffsets = Math.max(reqOffsets, targetOffsets);
    const allocRecords = Math.max(reqRecords, targetRecords);

    // Per-buffer binding limit: headroom is clamped to the limit, so only an
    // actual corpus above the limit can land here. Throw the budget error
    // (callers fall back to CPU) instead of creating an oversized binding
    // that fails validation and silently yields 0 results.
    const bindingLimit = resolveStorageBufferLimit(this.state.device);
    if (allocOffsets > bindingLimit || allocRecords > bindingLimit) {
      Object.assign(this, {
        currentDatasetSize: prev.size,
        currentStrings: prev.strings,
        currentTokens: prev.tokens,
        currentOffsets: prev.offsets,
        normalized: prev.normalized,
        profileId: prev.profileId,
        unicodeVersion: prev.unicodeVersion,
        scoringVersion: prev.scoringVersion,
      });
      const over = allocRecords > bindingLimit ? 'records' : 'offsets';
      throw new Error(`[webgpu-search] Dataset ${over} buffer too large. Falling back to CPU.`);
    }

    const t0 = nowMs();
    try {
      this.buffers.swapDatasetBuffers(count, packed, allocOffsets, allocRecords);
    } catch (allocErr) {
      // Restore previous CPU metadata so stats don't lie after OOM.
      Object.assign(this, {
        currentDatasetSize: prev.size,
        currentStrings: prev.strings,
        currentTokens: prev.tokens,
        currentOffsets: prev.offsets,
        normalized: prev.normalized,
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
      const prevRows = this.currentDatasetSize;
      const prevTokens = this.currentTokens ? this.currentTokens.length : 0;
      this.buffers.writeAppendedRows(prevRows, prevTokens, newTokens, newOffsets);

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
    // Unified with hybrid/document/exact-scorer: unknown modes throw
    // IncompatibleOptionError('mode') via the shared guard.
    assertValidMode(rawMode);
    // WGSL shaders are exact-only ('fuzzy'/'substring'). Token and
    // prefix modes route to the CPU reference engine — the hybrid/document
    // indexes catch this and fall back with fallbackReason 'unsupported-mode'
    // (scoring-parity boundary). Direct engine callers must route
    // themselves; the engine never silently serves approximate results.
    if (rawMode === 'token' || rawMode === 'prefix') {
      throw new IncompatibleOptionError(
        'mode',
        `Search mode '${rawMode}' is CPU-only (no WGSL kernel). Route token/prefix queries to the CPU exact scorer.`
      );
    }
    // Typo-tolerant queries cannot run on the exact-only shaders either.
    // Normalized here (fail-closed on malformed shapes) so direct callers
    // get identical validation with or without a device.
    const engineTypo = normalizeTypoTolerance(options.typoTolerance);
    if (engineTypo.enabled) {
      throw new IncompatibleOptionError(
        'typoTolerance',
        '[webgpu-search] Typo-tolerant queries are CPU-only (WGSL kernels are exact-only). Route to the CPU exact scorer.'
      );
    }
    const mode = rawMode;
    const limit = clampLimit(options.limit ?? options.maxResults ?? 50);
    // Strict boolean gate (matches search-index default caseSensitive=false):
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

    // Defensive post-normalization gates run BEFORE the no-device early return so
    // direct-engine callers get identical throw/echo semantics with or
    // without a device (hybrid already gates, but engine must not diverge).
    // Engine never silently falls back — it throws.
    const nq = normalizeText(query, this.normalized);
    if (nq.isEmpty) {
      return noHits('');
    }
    if (nq.tokenCount > QUERY_TOKENS_MAX) {
      throw new QueryTooLongError(QUERY_TOKENS_MAX, nq.tokenCount, this.profileId);
    }
    // Default omitted flag to false (same as search-index) so direct-engine
    // callers get identical ProfileMismatch semantics (normalized=false index +
    // omitted flag throws, instead of silently running case-sensitive).
    const qcs = options.caseSensitive ?? false;
    if (qcs === this.normalized) {
      throw new ProfileMismatchError(!this.normalized, options.caseSensitive);
    }

    if (!this.buffers.isSearchReady()) {
      return noHits(query);
    }

    // Zero-dispatch fast path: no rows means no compute work to submit.
    // Note: empty already returned above with unified echo ''.
    if (this.currentDatasetSize === 0) {
      return noHits(query);
    }

    const pipeline = mode === 'fuzzy' ? this.state.fuzzyPipeline! : this.state.substringPipeline!;
    const caseSensitive = qcs;
    const queryLen = nq.tokens.length;

    const totalStart = nowMs();

    const tQueryStart = nowMs();
    const udata = new ArrayBuffer(32);
    const U32 = new Uint32Array(udata);
    U32[0] = this.currentDatasetSize;
    U32[1] = queryLen;
    U32[2] = this.state.candidateCapacity;
    // flagsAndProfile: caseSensitive:1b + normalized:1b + profile:6b + unicode:8b
    // + scoring:8b. Reserved for harness/debug — shaders are pure-`==` by
    // construction and do not read it (host enforces ProfileMismatchError).
    const pEnum = (PROFILE_TO_ENUM as Record<string, number>)[this.profileId] ?? 0;
    const uEnum = (UNICODE_VERSION_TO_ENUM as Record<string, number>)[this.unicodeVersion] ?? 0;
    const sEnum = (SCORING_TO_ENUM as Record<string, number>)[this.scoringVersion] ?? 0;
    U32[3] = (((caseSensitive ? 1 : 0) | (this.normalized ? 2 : 0) | ((pEnum & 0x3f) << 2) | ((uEnum & 0xff) << 8) | ((sEnum & 0xff) << 16)) >>> 0);
    U32[4] = 0;
    U32[5] = 0;
    U32[6] = 0;
    U32[7] = 0;
    this.dispatch.uploadSearchRequest(nq.tokens, udata);
    const queryUploadMs = nowMs() - tQueryStart;

    throwIfAborted(options.signal);

    const tDispatchStart = nowMs();

    const bg = this.dispatch.createSearchBindGroup(pipeline);

    // Chunked dispatch: when ceil(rows/128) exceeds the device dimension
    // limit, each chunk re-writes the uniform (_pad.x = base row) and submits
    // its own pass; the shared atomic counter accumulates. Fast path keeps
    // timestamps. Queue-ordering dependency: WebGPU guarantees serial
    // queue execution, so each dispatch samples only prior uniform writes
    // (chunk counts are tiny in practice: step ≈ 8.4M rows/chunk).
    const md = (this.state.device!.limits as unknown as Record<string, unknown>).maxComputeWorkgroupsPerDimension;
    const maxDim = typeof md === 'number' && Number.isFinite(md) && md >= 1 ? Math.floor(md) : 65535;
    const workgroupSize = WORKGROUP_SIZE;
    const totalWorkgroups = Math.ceil(this.currentDatasetSize / workgroupSize);

    // searchInternal only routes; single vs chunked lives above.
    // Chunked corpora exceed any latency budget; timestamps skipped (null).
    let skipTimestamps = false;
    if (totalWorkgroups <= maxDim) {
      this.dispatch.dispatchSinglePass(pipeline, bg, totalWorkgroups);
    } else {
      this.dispatch.dispatchChunked(pipeline, bg, udata, U32, maxDim, this.currentDatasetSize);
      skipTimestamps = true;
    }
    const encodeSubmitMs = nowMs() - tDispatchStart;

    throwIfAborted(options.signal);

    const tReadbackStart = nowMs();
    let gpuExecutionMs: number | null = null;

    // Generation-epoch abort for in-flight mapAsync (takes no signal, so
    // abort is cooperative discard). Captured BEFORE the first await (the
    // timestamp readback): destroy()/ensureCandidateCapacity()/loadDataset()
    // during that await bump the epoch and null/replace buffers, which must
    // surface as AbortError — never a TypeError or a read of a fresh
    // (zeroed) staging buffer. Buffers are pinned locally for the same reason.
    // Rejected: bare Promise.race — abandons the mapping, later
    // destroy-while-mapped error. mapAsync rejection itself (e.g. destroy
    // raced it via unmap) also maps to AbortError when the epoch moved, so
    // hybrid doesn't misclassify it as fallback-eligible GPU failure.
    const gen = this.state.generation;
    const stagingBuffer = this.state.stagingBuffer!;
    const tsStaging = this.state.queryStagingBuffer;

    if (tsStaging && !skipTimestamps) {
      gpuExecutionMs = await this.dispatch.readTimestampGpuMs(tsStaging, gen);
    }

    // Torn down / reallocated during the timestamp await.
    this.dispatch.assertStagingLive(stagingBuffer, gen);

    let totalMatches = 0;
    let candidateCount = 0;
    let hasOverflow = false;
    let candidates: SearchResultItem[] = [];

    await this.dispatch.mapStagedOutput(stagingBuffer, gen);
    try {
      // Re-check epoch + liveness after the await: destroy() nulls
      // device/buffers, reallocation swaps them.
      this.dispatch.assertStagingLive(stagingBuffer, gen);
      throwIfAborted(options.signal);
      const decoded = this.dispatch.readMappedOutput(
        stagingBuffer,
        this.state.candidateCapacity,
        (index: number) => (this.currentStrings ? (this.currentStrings[index] ?? '') : '')
      );
      totalMatches = decoded.totalMatches;
      candidateCount = decoded.candidateCount;
      hasOverflow = decoded.hasOverflow;
      candidates = decoded.candidates;
    } finally {
      try { stagingBuffer.unmap(); } catch {}
    }

    throwIfAborted(options.signal);

    const readbackMs = nowMs() - tReadbackStart;
    const totalMs = nowMs() - totalStart;

    candidates.sort(compareExactResults);
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
    this.state.generation++;
    this.buffers.destroyForTeardown();
    this.pipelines.drop();
    this.currentStrings = null;
    this.currentTokens = null;
    this.currentOffsets = null;
    this.currentDatasetSize = 0;
    this.releaseCurrentDevice();
    const p = this.searchMutex.then(() => {}, () => {});
    this.searchMutex = p.catch(() => {});
  }
}
