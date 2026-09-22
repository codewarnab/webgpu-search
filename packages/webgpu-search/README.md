# webgpu-search

[![npm version](https://img.shields.io/npm/v/webgpu-search.svg)](https://www.npmjs.com/package/webgpu-search)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Ultra-fast hybrid fuzzy and substring search library for the web. Powered by parallel **WebGPU compute shaders (WGSL)** on retained VRAM with automatic seamless fallback to **uFuzzy (CPU)** for small datasets, older browsers, Web Workers, Node.js, and SSR.

---

## ⚡ Features

- **Measured GPU Acceleration**: On the documented Intel Iris Xe / Windows 11 / D3D11 ANGLE run, retained end-to-end WebGPU searches measured 3.13-17.27 ms for 100,000-2,000,000 rows, excluding one documented 59.10 ms outlier. Results vary by device, browser, query, dataset, timestamp-query support, and candidate overflow.
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
- **Large Datasets ($\ge$ 30k to 2M+ rows)**: Parallel GPU compute can outperform CPU matching, but the crossover and latency depend on the device, browser, query, dataset, timestamp-query support, and candidate overflow. See the root README for the current measured run and methodology.

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
import {
  WebGPUEngine,
  CPUEngine,
  packDataset,
  serializeDataset,
  deserializeDataset
} from 'webgpu-search';

// 1. Pack arbitrary strings into normalized u32 Unicode scalar tokens
const packed = packDataset(myStrings, { normalized: true });

// 2. Direct WebGPU compute pipeline with packed or deserialized dataset
const gpu = new WebGPUEngine();
await gpu.init();
await gpu.loadDataset(packed);

const gpuRes = await gpu.search('searchQuery', { mode: 'fuzzy', limit: 100 });
console.log(gpuRes.timings); // Detailed breakdown: queryUpload, encodeSubmit, gpuExecution, readback
gpu.destroy();
```

---

## 📜 API Reference

### `SearchIndex.create(items, options?)`
- `items: string[]`: Array of strings to index.
- `options.threshold`: Item cutoff to use WebGPU vs CPU (default: `30000`).
- `options.preferGpu`: Force WebGPU if available (default: `false`). Conflicts with `search({ cpuScorer: 'ufuzzy' })` → `IncompatibleOptionError`.
- `options.slotBytes`: throw-on-use (`IncompatibleOptionError`; dynamic variable-length indexing replaced fixed slots). Remove it and rebuild.
- `options.textProfile`: Index-level immutable profile (default: `'unicode-default'`; unknown values throw `ProfileMismatchError`).
- `options.caseSensitive`: Pack-time normalization control (default: `false` = normalized/NFC+C+F; `true` = NFC-only). Fixed at construction.
- `options.device`: Custom injected `GPUDevice`.
- `options.powerPreference`: `'high-performance'` | `'low-power'`. Forwarded to `GpuDevicePool.acquireDevice` (`navigator.gpu.requestAdapter({ powerPreference })`). Unknown values throw `IncompatibleOptionError` fail-closed, even on CPU-only paths. Ignored when `device` is injected (no adapter request is made).
- `options.hooks`: `SearchHooks` for custom tokenization, scoring boosts, or predicates (`extensions` / `SearchExtensionHooks` were removed; unknown hooks throw fail-closed).
- `options.autocomplete`: `AutocompleteOptions | boolean` for inline suggestions on `DocumentIndex.search()` (`suggest` / `SuggestOptions` were removed; use `autocomplete()` / `AutocompleteOptions` / `SuggestionItem`).

### `index.search(query, options?)`
- `query: string`: Query string.
- `options.mode`: `'fuzzy'` (subsequence + word-boundary scoring) or `'substring'` (case-insensitive substring).
- `options.limit`: Maximum results to return. Defaults to `50` and is clamped to the inclusive range `1..8192` (`RESULT_LIMIT_MAX`) on both CPU and WebGPU.
- `options.maxResults`: Backwards-compatible alias for `limit`; `limit` takes precedence when both are provided.
- `options.caseSensitive`: Must match the index packed mode (default: `false`). Mismatch throws `ProfileMismatchError` — build one index per mode instead of varying per query.
- `options.cpuScorer`: `'exact'` (default) or `'ufuzzy'` (explicit opt-in CPU-only, skips GPU). `cpuAlgorithm` / `'parity'` were removed — unknown scorers throw `IncompatibleOptionError` fail-closed.
- `options.onQueryTooLong`: `'throw'` (default, throws `QueryTooLongError` over `QUERY_TOKENS_MAX=128` tokens) or `'cpu-fallback'` (forces CPU for that query).
- `options.signal`: `AbortSignal` to cancel stale query readback during fast typing.
- Returns `SearchResponse` with `profileId`/`scoringVersion`/`cpuScorer` echo.

### `index.getStats()`
Returns `{ size, engine, vramAllocatedBytes, adapterVendor, adapterRenderer, profileId, unicodeVersion, scoringVersion, tokenCount, normalized, formatVersion }`. `tokenCount` is the post-normalization Unicode scalar count.

### Snapshot format
Records are packed as u32 scalar tokens with token offsets; the binary snapshot format is versioned (`SNAPSHOT_MAGIC 0x55324434`, legacy `LEGACY_SNAPSHOT_MAGIC 0x55324433` read-only). See the full [**Snapshot Format (`docs/snapshot-format.md`)**](../../docs/snapshot-format.md).

### `index.destroy()`
Releases GPU buffers and releases reference from the shared context manager.

---

## 📄 License

MIT © 2026

