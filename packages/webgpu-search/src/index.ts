// High-Level Primary API
export { SearchIndex } from './hybrid-index';

// Low-Level Engines & Hardware Utilities for Power Users & Benchmarks
export { WebGPUEngine, type DatasetLike, type ColdSearchResult, type WebGPUSearchResult, type SearchResult } from './webgpu-engine';
export { CPUEngine, type CPUSearchResult } from './cpu-engine';
export { WebGPUContextManager, type AcquiredDeviceContext } from './context-manager';
export {
  packStringsToGPUBuffer,
  sanitizeStringForSlot,
  checkMemoryBudget,
  type PackedGPUBuffer,
  type MemoryBudgetCheck
} from './buffer';

// Unified Types
export type * from './types';
