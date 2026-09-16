# webgpu-search

[![npm version](https://img.shields.io/npm/v/webgpu-search.svg)](https://www.npmjs.com/package/webgpu-search)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Ultra-fast hybrid fuzzy and substring search library for the web. Powered by parallel **WebGPU compute shaders (WGSL)** on retained VRAM with automatic seamless fallback to **uFuzzy (CPU)** for small datasets, older browsers, Web Workers, Node.js, and SSR.

---

## ⚡ Features

- **Sub-2ms Searches**: Parallel WGSL compute shaders process millions of rows directly in GPU VRAM.
- **Smart Dynamic Routing**: Executes in WebGPU when dataset scale benefits from parallel GPU hardware (default: $\ge$ 30,000 items), and routes to uFuzzy on CPU when small or when WebGPU is unavailable.
- **Zero-Stutter Concurrency**: Drops stale readbacks during rapid keystrokes with `AbortSignal` and monotonic sequence tagging.
- **Context Multiplexing**: Single shared `GPUDevice` pooled across all search indexes to prevent browser context exhaustion.
- **Worker & SSR Safe**: 100% compatible with Web Workers, Node.js, and SSR environments without DOM crashes.
- **Unified Scoring API**: Identical normalized `SearchResultItem` output across both WebGPU and CPU engines.

---

## 📦 Installation

```bash
npm install webgpu-search
# or
bun add webgpu-search
# or
pnpm add webgpu-search
```

---

## 🚀 Quickstart

```ts
import { SearchIndex } from 'webgpu-search';

// 1. Create index from an array of strings
const index = await SearchIndex.create(items, { threshold: 30_000 });

// 2. Query with search mode and limit
const response = await index.search('AuthController', { mode: 'fuzzy', limit: 25 });

console.log(`Found ${response.totalMatches} matches in ${response.timings.totalMs.toFixed(2)}ms via ${response.engine}`);
for (const item of response.results) {
  console.log(`[Score: ${item.score}] #${item.index}: ${item.text}`);
}

// 3. Free GPU memory when finished
index.destroy();
```

---

## 🔄 How Dynamic Routing Works

Search performance depends on dataset size:
- **Small Datasets (< 30k rows)**: CPU overhead is negligible (< 1ms). WebGPU kernel dispatch and buffer mapping would add unnecessary latency.
- **Large Datasets ($\ge$ 30k to 2M+ rows)**: Parallel GPU compute pipelines scan entire datasets in parallel in **1.5ms – 3ms**, whereas CPU single-threaded matching scales linearly into dozens or hundreds of milliseconds.

| Dataset Size | CPU (uFuzzy) | WebGPU (Retained VRAM) | Speedup | Servicing Engine |
| :--- | :--- | :--- | :--- | :--- |
| **10,000** | 0.8 ms | 1.8 ms | 0.44x | **CPU (uFuzzy)** |
| **100,000** | 7.9 ms | 1.9 ms | **4.1x** | **WebGPU** |
| **500,000** | 42.5 ms | 2.1 ms | **20.2x** | **WebGPU** |
| **1,000,000** | 89.2 ms | 2.6 ms | **34.3x** | **WebGPU** |
| **2,000,000** | 185.0 ms | 3.8 ms | **48.7x** | **WebGPU** |

---

## 🛠️ Web Worker Usage (Zero Main-Thread Latency)

Run heavy search completely off the main thread:

```ts
// search.worker.ts
import { SearchIndex } from 'webgpu-search';

let index: SearchIndex | null = null;

self.onmessage = async (e: MessageEvent) => {
  const { id, type, payload } = e.data;

  if (type === 'INIT') {
    index = await SearchIndex.create(payload.items, payload.options);
    self.postMessage({ id, type: 'READY' });
  } else if (type === 'SEARCH' && index) {
    const results = await index.search(payload.query, payload.options);
    self.postMessage({ id, type: 'RESULTS', payload: results });
  } else if (type === 'DESTROY' && index) {
    index.destroy();
    index = null;
  }
};
```

---

## 🎛️ Advanced: Low-Level Engine Access

For custom benchmarks, fine-grained buffer manipulation, or custom GPU pipelines:

```ts
import { WebGPUEngine, CPUEngine, packStringsToGPUBuffer } from 'webgpu-search';

// 1. Pack arbitrary strings into GPU byte layout (64-byte or 128-byte slots)
const packed = packStringsToGPUBuffer(myStrings, 64);

// 2. Direct WebGPU compute pipeline
const gpu = new WebGPUEngine();
await gpu.init();
await gpu.loadDataset({
  size: myStrings.length,
  strings: myStrings,
  gpuBufferData: packed.bufferData,
  byteLength: packed.byteLength
});

const gpuRes = await gpu.search('searchQuery', { mode: 'fuzzy', limit: 100 });
console.log(gpuRes.timings); // Detailed breakdown: upload, submit, gpuExecution, readback
gpu.destroy();
```

---

## 📜 API Reference

### `SearchIndex.create(items, options?)`
- `items: string[]`: Array of strings to index.
- `options.threshold`: Item cutoff to use WebGPU vs CPU (default: `30000`).
- `options.preferGpu`: Force WebGPU if available (default: `false`).
- `options.slotBytes`: Row width, `64` (59 chars) or `128` (123 chars) (default: `64`).
- `options.device`: Custom injected `GPUDevice`.

### `index.search(query, options?)`
- `query: string`: Query string.
- `options.mode`: `'fuzzy'` (subsequence + word-boundary scoring) or `'substring'` (case-insensitive substring).
- `options.limit`: Maximum results to return. Defaults to `50` and is clamped to the inclusive range `1..8192` on both CPU and WebGPU.
- `options.maxResults`: Backwards-compatible alias for `limit`; `limit` takes precedence when both are provided.
- `options.caseSensitive`: Case sensitivity flag (default: `false`).
- `options.signal`: `AbortSignal` to cancel stale query readback during fast typing.

### `index.getStats()`
Returns `{ size, engine, vramAllocatedBytes, adapterVendor, adapterRenderer }`.

### `index.destroy()`
Releases GPU buffers and releases reference from the shared context manager.

---

## 📄 License

MIT © 2026

