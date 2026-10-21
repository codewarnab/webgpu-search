/**
 * GPU buffer lifecycle for the WebGPU engine (verbatim move of the buffer
 * half of the facade): fixed uniform/query allocations, timestamp query
 * resources, output/staging (re)allocation, dataset buffer swap with a safe
 * swap, append tail writes, and the three teardown paths (re-init =
 * unmap-each; explicit destroy = unmap staging pair; device loss = null
 * refs, allocated sizes kept).
 */

import { QUERY_TOKENS_MAX } from '../text-profile';
import type { PackedDataset } from '../dataset-packing';
import { BufferUsage, type GpuEngineState } from './state';

export class GpuBufferManager {
  constructor(private readonly state: GpuEngineState) {}

  /** Uniform 32 B; query QUERY_TOKENS_MAX*4 (min 16 B), destroyed under mutex. */
  createFixedBuffers(): void {
    const device = this.state.device!;

    this.state.uniformBuffer = device.createBuffer({
      label: 'Uniform Buffer',
      size: 32,
      usage: BufferUsage.UNIFORM | BufferUsage.COPY_DST
    });

    this.state.queryBuffer = device.createBuffer({
      label: 'Query Buffer',
      size: Math.max(QUERY_TOKENS_MAX * 4, 16),
      usage: BufferUsage.STORAGE | BufferUsage.COPY_DST
    });
  }

  /** Timestamp query + resolve/staging, degraded to null on alloc failure (warn). */
  createTimestampResources(): void {
    const device = this.state.device!;
    if (device.features && device.features.has('timestamp-query')) {
      try {
        this.state.querySet = device.createQuerySet({
          label: 'ts-query',
          type: 'timestamp',
          count: 2
        });
        this.state.queryResolveBuffer = device.createBuffer({
          label: 'ts-resolve',
          size: 16,
          usage: BufferUsage.QUERY_RESOLVE | BufferUsage.COPY_SRC
        });
        this.state.queryStagingBuffer = device.createBuffer({
          label: 'ts-staging',
          size: 16,
          usage: BufferUsage.MAP_READ | BufferUsage.COPY_DST
        });
      } catch (qErr) {
        console.warn('timestamp alloc failed:', qErr);
        this.state.querySet = null;
        this.state.queryResolveBuffer = null;
        this.state.queryStagingBuffer = null;
      }
    }
  }

  ensureCandidateCapacity(capacity: number): void {
    const state = this.state;
    const safeCap = typeof capacity === 'number' && Number.isFinite(capacity) && capacity > 0
      ? Math.floor(capacity)
      : 8192;
    const clamped = Math.min(32768, Math.max(8192, safeCap));
    if (clamped !== state.candidateCapacity || !state.outputBuffer) {
      state.candidateCapacity = clamped;
      if (state.device) {
        this.allocateOutputBuffers(clamped);
      }
    }
  }

  allocateOutputBuffers(candidateCapacity: number = 8192): void {
    const state = this.state;
    const device = state.device;
    if (!device) return;
    const safeCap = typeof candidateCapacity === 'number' && Number.isFinite(candidateCapacity) && candidateCapacity > 0
      ? Math.floor(candidateCapacity)
      : 8192;
    const clampedCap = Math.min(32768, Math.max(safeCap, 8192));
    if (state.outputBuffer && clampedCap <= state.candidateCapacity && state.outputByteLength >= 8 + clampedCap * 8) return;
    state.candidateCapacity = clampedCap;
    state.outputByteLength = 8 + state.candidateCapacity * 8;

    // Advance generation to discard any in-flight reads against old staging buffers
    state.generation++;

    // Unmap-before-destroy (symmetric with destroy()): destroying a mapped
    // buffer throws / leaves torn state on some implementations.
    for (const b of [state.outputBuffer, state.stagingBuffer]) {
      try { (b as GPUBuffer | null)?.unmap?.(); } catch {}
    }
    if (state.outputBuffer) state.outputBuffer.destroy();
    if (state.stagingBuffer) state.stagingBuffer.destroy();

    state.outputBuffer = device.createBuffer({
      label: 'Output Buffer',
      size: state.outputByteLength,
      usage: BufferUsage.STORAGE | BufferUsage.COPY_SRC | BufferUsage.COPY_DST
    });

    state.stagingBuffer = device.createBuffer({
      label: 'Staging Buffer',
      size: state.outputByteLength,
      usage: BufferUsage.MAP_READ | BufferUsage.COPY_DST
    });
  }

  /** True when every buffer/pipeline the search pass needs is live. */
  isSearchReady(): boolean {
    const s = this.state;
    return (
      s.device !== null &&
      s.recordsBuffer !== null &&
      s.offsetsBuffer !== null &&
      s.uniformBuffer !== null &&
      s.queryBuffer !== null &&
      s.outputBuffer !== null &&
      s.stagingBuffer !== null &&
      s.substringPipeline !== null &&
      s.fuzzyPipeline !== null
    );
  }

  /** Headroom check: dataset bindings must fit `requiredRows` offsets + `requiredTokens` records. */
  hasDatasetBufferCapacity(requiredRows: number, requiredTokens: number): boolean {
    const needOffsets = (requiredRows + 1) * 4;
    const needRecords = requiredTokens * 4;
    const s = this.state;
    return (
      s.offsetsBuffer !== null &&
      s.recordsBuffer !== null &&
      needOffsets <= s.allocatedOffsetsByteLength &&
      needRecords <= s.allocatedRecordsByteLength
    );
  }

  /**
   * Create + upload the packed dataset's storage buffers, then swap them for
   * the previous pair (old buffers unmap/destroy only after new allocations
   * succeed). Destroys partials and rethrows on failure — the facade restores
   * CPU metadata (same split as the original inlined block).
   */
  swapDatasetBuffers(
    count: number,
    packed: PackedDataset,
    allocOffsets: number,
    allocRecords: number
  ): void {
    let newOffsetsBuffer: GPUBuffer | null = null;
    let newRecordsBuffer: GPUBuffer | null = null;
    const device = this.state.device!;
    const state = this.state;

    try {
      newOffsetsBuffer = device.createBuffer({
        label: `Offsets (${count})`,
        size: allocOffsets,
        usage: BufferUsage.STORAGE | BufferUsage.COPY_DST
      });

      newRecordsBuffer = device.createBuffer({
        label: `Records (${count})`,
        size: allocRecords,
        usage: BufferUsage.STORAGE | BufferUsage.COPY_DST
      });

      device.queue.writeBuffer(newOffsetsBuffer, 0, packed.offsetsBufferData);
      if (packed.recordsByteLength > 0) {
        device.queue.writeBuffer(newRecordsBuffer, 0, packed.recordsBufferData);
      }

      // Safe swap: only unmap/destroy previous buffers after new allocations succeed
      for (const b of [state.offsetsBuffer, state.recordsBuffer]) {
        try { (b as GPUBuffer | null)?.unmap?.(); } catch {}
        try { b?.destroy(); } catch {}
      }
      state.offsetsBuffer = newOffsetsBuffer;
      state.recordsBuffer = newRecordsBuffer;
      state.allocatedOffsetsByteLength = allocOffsets;
      state.allocatedRecordsByteLength = allocRecords;
    } catch (allocErr) {
      try { newOffsetsBuffer?.destroy(); } catch {}
      try { newRecordsBuffer?.destroy(); } catch {}
      throw allocErr;
    }
  }

  /**
   * Guards + generation bump + queue writes for appended rows (GPU half of
   * appendRows; the facade does the CPU merges). Counts are pre-append.
   */
  writeAppendedRows(
    prevRows: number,
    prevTokens: number,
    newTokens: Uint32Array,
    newOffsets: Uint32Array
  ): void {
    const state = this.state;
    if (!state.device || !state.offsetsBuffer || !state.recordsBuffer) {
      throw new Error('[webgpu-search] Cannot append rows: GPU buffers not initialized.');
    }
    const needOffsetsBytes = (prevRows + 1) * 4 + newOffsets.byteLength;
    const needRecordsBytes = prevTokens * 4 + newTokens.byteLength;
    if (needOffsetsBytes > state.allocatedOffsetsByteLength || needRecordsBytes > state.allocatedRecordsByteLength) {
      throw new Error('[webgpu-search] appendRows exceeds allocated buffer headroom.');
    }
    state.generation++;

    // Write new offsets: row offsets starting at (prevRows + 1) * 4
    state.device.queue.writeBuffer(state.offsetsBuffer, (prevRows + 1) * 4, newOffsets);
    // Write new tokens at prevTokens * 4
    if (newTokens.byteLength > 0) {
      state.device.queue.writeBuffer(state.recordsBuffer, prevTokens * 4, newTokens);
    }
  }

  /** init() re-entry: unmap-each + destroy all buffers, zero allocated sizes. */
  disposeForReinit(): void {
    for (const b of [
      this.state.offsetsBuffer, this.state.recordsBuffer, this.state.uniformBuffer,
      this.state.queryBuffer, this.state.outputBuffer, this.state.stagingBuffer,
      this.state.queryResolveBuffer, this.state.queryStagingBuffer
    ]) {
      try { (b as GPUBuffer | null)?.unmap?.(); } catch {}
      try { (b as GPUBuffer | null)?.destroy(); } catch {}
    }
    this.state.offsetsBuffer = this.state.recordsBuffer = this.state.uniformBuffer = this.state.queryBuffer = null;
    this.state.outputBuffer = this.state.stagingBuffer = this.state.queryResolveBuffer = this.state.queryStagingBuffer = null;
    if (this.state.querySet) {
      try { this.state.querySet.destroy(); } catch {}
      this.state.querySet = null;
    }
    this.state.allocatedOffsetsByteLength = 0;
    this.state.allocatedRecordsByteLength = 0;
  }

  /**
   * Explicit destroy(): unmap the (possibly mapped) staging pair first, then
   * destroy all buffers + querySet and zero allocated sizes. Does not unmap
   * dataset/uniform buffers (original did too); pipeline handles + CPU
   * metadata parked by the facade.
   */
  destroyForTeardown(): void {
    try { this.state.stagingBuffer?.unmap(); } catch {}
    try { this.state.queryStagingBuffer?.unmap(); } catch {}
    const bufs = [
      this.state.offsetsBuffer, this.state.recordsBuffer, this.state.uniformBuffer,
      this.state.queryBuffer, this.state.outputBuffer, this.state.stagingBuffer,
      this.state.queryResolveBuffer, this.state.queryStagingBuffer
    ];
    for (const b of bufs) { try { b?.destroy(); } catch {} }
    this.state.offsetsBuffer = this.state.recordsBuffer = this.state.uniformBuffer = this.state.queryBuffer = null;
    this.state.outputBuffer = this.state.stagingBuffer = this.state.queryResolveBuffer = this.state.queryStagingBuffer = null;
    if (this.state.querySet) {
      try { this.state.querySet.destroy(); } catch {}
      this.state.querySet = null;
    }
    this.state.allocatedOffsetsByteLength = 0;
    this.state.allocatedRecordsByteLength = 0;
  }

  /**
   * Device loss: null refs without destroying (device is already lost).
   * Keeps allocated sizes (stale `vramAllocatedBytes` is pre-existing).
   */
  dropDeviceRefs(): void {
    this.state.offsetsBuffer = this.state.recordsBuffer = this.state.uniformBuffer = this.state.queryBuffer = null;
    this.state.outputBuffer = this.state.stagingBuffer = this.state.queryResolveBuffer = this.state.queryStagingBuffer = null;
    this.state.querySet = null;
  }
}
