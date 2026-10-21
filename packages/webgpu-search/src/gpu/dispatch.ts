/**
 * Search submission + readback for the WebGPU engine (verbatim move of the
 * facade's dispatch/readback helpers): queue uploads, bind group creation,
 * single-pass vs chunked dispatch with counter-reset fallbacks, and the
 * generation-epoch-guarded mapAsync readbacks. Sync blocks use a device
 * snapshot (no awaits inside); readback paths keep live epoch reads.
 */

import { abortError } from '../guard';
import type { SearchResultItem } from '../types';
import { MapMode, type GpuEngineState } from './state';

/** Compute workgroup size the WGSL kernels assume (dispatch math: ceil(rows/128)). */
export const WORKGROUP_SIZE = 128;

/** Decoded staging buffer contents for one search. */
export interface MappedSearchOutput {
  totalMatches: number;
  candidateCount: number;
  hasOverflow: boolean;
  candidates: SearchResultItem[];
}

export class GpuDispatcher {
  constructor(private readonly state: GpuEngineState) {}

  /** Queue uploads for one search: query tokens, then the 32 B uniform header. */
  uploadSearchRequest(tokens: Uint32Array, uniform: ArrayBuffer): void {
    const device = this.state.device!;
    device.queue.writeBuffer(this.state.queryBuffer!, 0, tokens);
    device.queue.writeBuffer(this.state.uniformBuffer!, 0, uniform);
  }

  /** Bind group over the fixed search bindings (auto-layout pipeline). */
  createSearchBindGroup(pipeline: GPUComputePipeline): GPUBindGroup {
    const s = this.state;
    return s.device!.createBindGroup({
      label: 'search-bg',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: s.uniformBuffer! } },
        { binding: 1, resource: { buffer: s.offsetsBuffer! } },
        { binding: 2, resource: { buffer: s.recordsBuffer! } },
        { binding: 3, resource: { buffer: s.queryBuffer! } },
        { binding: 4, resource: { buffer: s.outputBuffer! } }
      ]
    });
  }

  /**
   * Zero the atomic counter fallback for Safari without `clearBuffer`.
   * 8-byte zero write (count + _pad0), queue-ordered before dispatch.
   * Never skip the reset (stale count corruption). No scoring change.
   */
  private writeZeroCounterFallback(): void {
    this.state.device!.queue.writeBuffer(this.state.outputBuffer!, 0, new Uint8Array(8));
  }

  /**
   * Single-pass dispatch: counter reset + compute pass + timestamp resolve +
   * readback copy in one command buffer. No scoring change.
   */
  dispatchSinglePass(
    pipeline: GPUComputePipeline,
    bg: GPUBindGroup,
    totalWorkgroups: number
  ): void {
    const state = this.state;
    const device = state.device!;
    const passDesc: GPUComputePassDescriptor = {
      label: 'search-pass'
    };
    if (state.querySet) {
      (passDesc as any).timestampWrites = {
        querySet: state.querySet,
        beginningOfPassWriteIndex: 0,
        endOfPassWriteIndex: 1
      };
    }
    const enc = device.createCommandEncoder({ label: 'search' });

    if (typeof (enc as any).clearBuffer === 'function') {
      enc.clearBuffer(state.outputBuffer!, 0, state.outputByteLength);
    } else {
      // No clearBuffer: encode the zero via copy path — submit the
      // queue write before the compute submit (serial ordering).
      this.writeZeroCounterFallback();
    }

    const pass = enc.beginComputePass(passDesc);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(totalWorkgroups);
    pass.end();

    if (state.querySet && state.queryResolveBuffer && state.queryStagingBuffer) {
      enc.resolveQuerySet(state.querySet, 0, 2, state.queryResolveBuffer, 0);
      enc.copyBufferToBuffer(state.queryResolveBuffer, 0, state.queryStagingBuffer, 0, 16);
    }

    enc.copyBufferToBuffer(state.outputBuffer!, 0, state.stagingBuffer!, 0, state.outputByteLength);
    device.queue.submit([enc.finish()]);
  }

  /**
   * Chunked dispatch: when ceil(rows/128) exceeds the device dimension
   * limit, each chunk re-writes the uniform (_pad.x = base row) and submits
   * its own pass; the shared atomic counter accumulates. WebGPU's serial
   * queue guarantees each dispatch samples only prior uniform writes.
   * Timestamps skipped (null). No scoring change.
   */
  dispatchChunked(
    pipeline: GPUComputePipeline,
    bg: GPUBindGroup,
    uniform: ArrayBuffer,
    U32: Uint32Array,
    maxDim: number,
    totalRows: number
  ): void {
    const state = this.state;
    const device = state.device!;
    const step = maxDim * WORKGROUP_SIZE;
    const clr = device.createCommandEncoder({ label: 'clear' });
    if (typeof (clr as any).clearBuffer === 'function') {
      (clr as any).clearBuffer(state.outputBuffer!, 0, state.outputByteLength);
      device.queue.submit([clr.finish()]);
    } else {
      this.writeZeroCounterFallback();
    }
    let base = 0;
    let left = totalRows;
    while (left > 0) {
      const n = left > step ? step : left;
      const wgs = Math.ceil(n / WORKGROUP_SIZE);
      U32[4] = base;
      device.queue.writeBuffer(state.uniformBuffer!, 0, uniform);
      const chunkEnc = device.createCommandEncoder({ label: 'chunk' });
      const chunkPass = chunkEnc.beginComputePass({ label: 'chunk-pass' });
      chunkPass.setPipeline(pipeline);
      chunkPass.setBindGroup(0, bg);
      chunkPass.dispatchWorkgroups(wgs);
      chunkPass.end();
      device.queue.submit([chunkEnc.finish()]);
      base += n;
      left -= n;
    }
    const cp = device.createCommandEncoder({ label: 'copy' });
    cp.copyBufferToBuffer(state.outputBuffer!, 0, state.stagingBuffer!, 0, state.outputByteLength);
    device.queue.submit([cp.finish()]);
  }

  /**
   * Map the timestamp staging buffer and extract the GPU execution delta.
   * Epoch-only guard: a teardown during the await discards the stale sample
   * (null); warn-suppressed once the epoch moved too.
   */
  async readTimestampGpuMs(tsStaging: GPUBuffer, epoch: number): Promise<number | null> {
    let gpuExecutionMs: number | null = null;
    try {
      await tsStaging.mapAsync(MapMode.READ);
      if (epoch === this.state.generation) {
        const tu = new BigUint64Array(tsStaging.getMappedRange());
        if (tu[1] >= (tu[0]) && (tu[0]) > 0n) {
          gpuExecutionMs = Number((tu[1]) - (tu[0])) / 1_000_000;
        }
      }
    } catch (tsErr) {
      if (epoch === this.state.generation) console.warn('timestamp read failed:', tsErr);
    } finally {
      try { tsStaging.unmap(); } catch {}
    }
    return gpuExecutionMs;
  }

  /**
   * Map the output staging buffer. An epoch that moved during the await
   * (destroy/reallocation raced the map) surfaces as AbortError — never a
   * fallback-eligible GPU failure; genuine map failures rethrow untouched.
   */
  async mapStagedOutput(stagingBuffer: GPUBuffer, epoch: number): Promise<void> {
    try {
      await stagingBuffer.mapAsync(MapMode.READ);
    } catch (mapErr) {
      if (epoch !== this.state.generation) {
        try { stagingBuffer.unmap(); } catch {}
        throw abortError();
      }
      throw mapErr;
    }
  }

  /** Re-check epoch + device + staging identity after an await; drift → AbortError. */
  assertStagingLive(stagingBuffer: GPUBuffer, epoch: number): void {
    const s = this.state;
    if (epoch !== s.generation || !s.device || s.stagingBuffer !== stagingBuffer) {
      throw abortError();
    }
  }

  /**
   * Decode one mapped staging buffer: u32[0] = total matches, then
   * per-candidate pairs (u32 index, i32 score). `resolveText` maps rows to
   * display text (token-only datasets resolve `''`; see DatasetLike).
   */
  readMappedOutput(
    stagingBuffer: GPUBuffer,
    candidateCapacity: number,
    resolveText: (index: number) => string
  ): MappedSearchOutput {
    const arrayBuffer = stagingBuffer.getMappedRange();
    const u32Read = new Uint32Array(arrayBuffer);
    const i32Read = new Int32Array(arrayBuffer);

    const totalMatches = u32Read[0];
    const candidateCount = Math.min(totalMatches, candidateCapacity);
    const hasOverflow = totalMatches > candidateCapacity;

    const candidates: SearchResultItem[] = new Array(candidateCount);
    for (let i = 0; i < candidateCount; i++) {
      const index = u32Read[2 + i * 2];
      const score = i32Read[3 + i * 2];
      candidates[i] = { index, score, text: resolveText(index) };
    }
    return { totalMatches, candidateCount, hasOverflow, candidates };
  }
}
