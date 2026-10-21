/**
 * Shared GPU state + portable GPU constant tables for the extracted engine
 * subsystems (`gpu/buffers.ts`, `gpu/pipelines.ts`, `gpu/dispatch.ts`).
 * One `GpuEngineState` per `WebGPUEngine`; managers mutate it by reference.
 * Dataset CPU metadata stays on the facade. Portable: no DOM refs.
 */

/** GPUBufferUsage flags: global table under WebGPU runtimes, constants under Node/Bun. */
export const BufferUsage = (typeof globalThis !== 'undefined' && 'GPUBufferUsage' in globalThis ? (globalThis as any).GPUBufferUsage : {
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

/** GPUMapMode flags: same portability contract as BufferUsage. */
export const MapMode = (typeof globalThis !== 'undefined' && 'GPUMapMode' in globalThis ? (globalThis as any).GPUMapMode : {
  READ: 0x0001,
  WRITE: 0x0002
});

/** All mutable GPU-side engine state (buffers/pipelines/dispatch/dispose + epoch). */
export interface GpuEngineState {
  device: GPUDevice | null;                 // null until init / after device loss / teardown
  substringPipeline: GPUComputePipeline | null;
  fuzzyPipeline: GPUComputePipeline | null;
  offsetsBuffer: GPUBuffer | null;
  recordsBuffer: GPUBuffer | null;
  uniformBuffer: GPUBuffer | null;          // 32 B uniform header, written per query
  queryBuffer: GPUBuffer | null;            // QUERY_TOKENS_MAX*4 (min 16 B), written per query
  outputBuffer: GPUBuffer | null;           // atomic counter + candidates: 8 + cap*8 bytes
  stagingBuffer: GPUBuffer | null;          // MAP_READ twin of outputBuffer, remapped per search
  querySet: GPUQuerySet | null;             // timestamp resources (null when unsupported/failure)
  queryResolveBuffer: GPUBuffer | null;
  queryStagingBuffer: GPUBuffer | null;
  candidateCapacity: number;                // output/staging sizing, clamped 8192..32768
  outputByteLength: number;                 // 8 + candidateCapacity * 8
  /** Epoch bumped on every teardown/realloc/load/append; in-flight readbacks discard. */
  generation: number;
  /**
   * Reserved dataset binding sizes. NOT zeroed on device loss (stale
   * `vramAllocatedBytes` is pre-existing); explicit destroy()/dispose zero them.
   */
  allocatedOffsetsByteLength: number;
  allocatedRecordsByteLength: number;
}

export function createGpuEngineState(): GpuEngineState {
  return {
    device: null,
    substringPipeline: null,
    fuzzyPipeline: null,
    offsetsBuffer: null,
    recordsBuffer: null,
    uniformBuffer: null,
    queryBuffer: null,
    outputBuffer: null,
    stagingBuffer: null,
    querySet: null,
    queryResolveBuffer: null,
    queryStagingBuffer: null,
    candidateCapacity: 8192,
    outputByteLength: 0,
    generation: 0,
    allocatedOffsetsByteLength: 0,
    allocatedRecordsByteLength: 0,
  };
}
