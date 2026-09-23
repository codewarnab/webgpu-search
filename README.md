# Browser-Native Fuzzy Search with WebGPU & CPU

[![Live Demo](https://img.shields.io/badge/demo-online-brightgreen.svg)](https://webgpu-fuzzy-search.vercel.app)
[![npm version](https://img.shields.io/npm/v/webgpu-search.svg)](https://www.npmjs.com/package/webgpu-search)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An ultra-fast fuzzy and substring search engine powered by parallel **WebGPU compute shaders (WGSL)** on retained VRAM, backed by a bit-exact **CPU exact scorer** and opt-in **uFuzzy**, with Web Worker offloading that keeps search work completely off the main thread.

> ℹ️ **Text pipeline:** every string is normalized through a spec-compliant, code-point-safe Unicode pipeline (`u32` scalar packing, `CaseFolding-16.0.0`, dataset container). See [`docs/text-normalization.md`](./docs/text-normalization.md) and [`docs/snapshot-format.md`](./docs/snapshot-format.md).

👉 **[Product site](https://webgpu-fuzzy-search.vercel.app)** · [Try the homepage benchmark](https://webgpu-fuzzy-search.vercel.app/#benchmark) · [Run the full benchmark](https://webgpu-fuzzy-search.vercel.app/benchmark/)

---

## ⚡ Monorepo Structure

- **[`packages/webgpu-search`](./packages/webgpu-search)**: Zero-dependency core library published to npm. Provides high-level `SearchIndex` with dynamic crossover routing, low-level `WebGPUEngine` and `CPUEngine`, dataset binary serialization, and SSR/Worker-safe memory primitives.
- **[`apps/site`](./apps/site)**: Monochrome product homepage and a local WebGPU-versus-exact-CPU comparison page.
- **[`apps/benchmark`](./apps/benchmark)**: Full local benchmark across 10,000 to 2,000,000 records and ASCII, CJK, and emoji/mixed-script corpora.
- **Example apps**: Docs Search, Code Palette, and Log Viewer are mounted under `/examples/` in the same deployment.

---

## 🚀 Quick Start

```bash
# Clone and install dependencies
git clone https://github.com/codewarnab/webgpu-fuzzy-search.git
cd webgpu-fuzzy-search
bun install

# Run the site and workspace dev servers
bun run dev
```

Open the homepage at **`http://localhost:5178`**. The dev server proxies `/benchmark/` and `/examples/.../` to their workspace apps. Run `bun run build` to produce the complete static site at `apps/site/dist`.

---

## 💻 Library Usage

```ts
import {
  SearchIndex,
  QueryTooLongError,
  IncompatibleIndexError,
  ProfileMismatchError,
  IncompatibleOptionError
} from 'webgpu-search';

// 1. Create index from an array of strings
const index = await SearchIndex.create([
  'packages/core/src/AuthController.ts',
  'src/views/ユーザー設定/Profile.vue',
  'docs/api/Straße_v2.md',
  'assets/icons/🧑‍💻_developer.png'
], {
  threshold: 30_000,           // WebGPU vs CPU crossover threshold
  caseSensitive: false,        // Fixed at construction: NFC + C+F fold
  textProfile: 'unicode-default'
});

// 2. Query with search mode and limit
const response = await index.search('ユーザー', {
  mode: 'fuzzy',               // 'fuzzy' or 'substring'
  limit: 25,                   // Clamped to 1..8192
  onQueryTooLong: 'throw'      // 'throw' (QueryTooLongError) or 'cpu-fallback'
});

console.log(`Matched ${response.totalMatches} records in ${response.timings.totalMs.toFixed(2)}ms via ${response.engine}`);
for (const hit of response.results) {
  console.log(`[Score: ${hit.score}] #${hit.index}: ${hit.text}`);
}

// 3. Inspect telemetry and VRAM footprint
const stats = index.getStats();
console.log(`Allocated VRAM: ${(stats.vramAllocatedBytes / 1024).toFixed(1)} KB for ${stats.tokenCount} tokens`);

// 4. Free GPU memory when finished
index.destroy();
```

### Low-Level Pipeline & Binary Serialization

```ts
import {
  WebGPUEngine,
  CPUEngine,
  packDataset,
  serializeDataset,
  deserializeDataset
} from 'webgpu-search';

// 1. Pack strings into normalized u32 Unicode scalar tokens
const packed = packDataset(rawStrings, { normalized: true });

// 2. Serialize into the dataset binary container (magic 0x55324632 + CRC32 checksum)
const buffer: ArrayBuffer = serializeDataset(packed);

// 3. Restore and validate with fail-closed integrity checks
const restored = deserializeDataset(buffer);

// 4. Load directly into WebGPU compute engine
const gpu = new WebGPUEngine();
await gpu.init();
await gpu.loadDataset(restored);

const result = await gpu.search('test', { mode: 'substring', limit: 50 });
console.log(`GPU execution: ${result.timings.gpuExecutionMs}ms, Readback: ${result.timings.readbackMs}ms`);
gpu.destroy();
```

---

## 📊 Reproducible Multi-Corpus Benchmark Results

Below are benchmark results generated via the automated browser benchmark runner (`bun run test:benchmark`).

> **⚠️ Hardware Qualification Notice**:
> The automated suite runs in headless Chromium using software Vulkan (`Google SwiftShader / LLVMpipe`) in containerized environments. Cells below are explicitly qualified as **`pending-hardware`** per [`docs/text-normalization.md`](./docs/text-normalization.md) §5. On physical dedicated GPUs (NVIDIA RTX, Apple Silicon Metal, Intel Iris Xe), GPU compute times are dramatically faster due to hardware memory bandwidth and parallel execution units.

### 1. Exact Substring Benchmark Matrix

| Dataset Size | Corpus | Packed VRAM | GPU Retained (med / p95) | CPU Exact (med / p95) | uFuzzy (med) | JS Native (med) | Qualification |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **10,000** | ASCII | 1.71 MB | 8.50 / 10.18 ms | 1.50 / 1.60 ms | 1.70 ms | 1.10 ms | pending-hardware |
| **100,000** | ASCII | 16.88 MB | 32.60 / 36.70 ms | 15.00 / 16.18 ms | 15.80 ms | 11.40 ms | pending-hardware |
| **10,000** | CJK Hanzi/Kana | 1.46 MB | 7.40 / 8.88 ms | 1.70 / 2.18 ms | N/A* | 2.90 ms | pending-hardware |
| **100,000** | CJK Hanzi/Kana | 14.41 MB | 32.30 / 37.44 ms | 16.50 / 18.74 ms | N/A* | 25.50 ms | pending-hardware |
| **10,000** | Emoji Astral | 2.03 MB | 7.50 / 10.16 ms | 2.40 / 4.22 ms | N/A* | 4.00 ms | pending-hardware |
| **100,000** | Emoji Astral | 20.08 MB | 45.00 / 75.82 ms | 21.20 / 30.50 ms | N/A* | 36.90 ms | pending-hardware |

*\*Note: uFuzzy is designed primarily for Latin/ASCII tokenized text; non-Latin/Emoji scripts yield 0 hits without custom tokenizers.*

### 2. Fuzzy Subsequence Benchmark Matrix

| Dataset Size | Corpus | Packed VRAM | GPU Retained (med / p95) | CPU Exact (med / p95) | uFuzzy (med) | JS Native (med) | Qualification |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **10,000** | ASCII | 1.71 MB | 9.00 / 9.40 ms | 1.60 / 3.38 ms | 1.60 ms | 1.10 ms | pending-hardware |
| **100,000** | ASCII | 16.88 MB | 58.00 / 63.46 ms | 16.10 / 34.22 ms | 15.40 ms | 10.70 ms | pending-hardware |
| **10,000** | CJK Hanzi/Kana | 1.46 MB | 8.90 / 11.82 ms | 2.10 / 2.20 ms | N/A* | 2.90 ms | pending-hardware |
| **100,000** | CJK Hanzi/Kana | 14.41 MB | 48.70 / 84.86 ms | 14.10 / 14.88 ms | N/A* | 23.70 ms | pending-hardware |
| **10,000** | Emoji Astral | 2.03 MB | 8.00 / 14.94 ms | 2.10 / 3.90 ms | N/A* | 3.70 ms | pending-hardware |
| **100,000** | Emoji Astral | 20.08 MB | 68.80 / 76.48 ms | 17.80 / 18.10 ms | N/A* | 37.40 ms | pending-hardware |

---

## 🔬 Architectural Deep-Dive

### 1. The Retained VRAM vs Cold Upload Reality
- **Cold Upload**: Transferring 1,000,000 records (~64 MB packed `u32` tokens) over the PCIe bus via `queue.writeBuffer` takes **~8–20 ms**. If an application re-uploads data on every keystroke, CPU search will always win.
- **Retained VRAM**: When dataset storage buffers are pre-loaded once into GPU VRAM, subsequent keystrokes only upload a **32-byte uniform header** and a **512-byte persistent storage query buffer** (up to 128 `u32` code points).

### 2. Candidate Compaction & Deterministic Sorting
- WebGPU buffer readbacks have an inherent synchronization floor (~1ms to 3ms). Below 30,000 records, the CPU exact scorer completes in <2ms, making CPU faster at small scales.
- **Candidate Pool Compaction**: The WGSL compute shaders write up to **8,192 scored candidate matches** (`{ index: u32, score: i32 }`) into a compact 65,544-byte output buffer using atomic counters (`atomicAdd(&out.count, 1u)`).
- **Strict Deterministic Tie-Breaking**: Both WebGPU readback and CPU exact fallback sort matches by `(score DESC, index ASC)` using signed 32-bit integer arithmetic. When `hasOverflow === false`, result rankings and scores are bit-exact identical between WebGPU and CPU.

### 3. Web Worker Offloading and String-Isolated Enrichment
Searching millions of records directly on the browser main thread can cause frame drops and UI freezes.
- **String-Isolated Enrichment**: The worker executes compute pipelines and returns compact `{ index, score }[]` hit arrays (~8 bytes per match), while the main thread resolves the display strings (`text`) locally. This eliminates massive structured-clone serialization stalls between worker and UI thread.
- **Cancellation & Freshness**: Monotonic query IDs (`latestQueryId`) and `AbortController` signals drop stale in-flight passes during rapid user typing.

---

## 🛠️ Verification & Quality Assurance Gates

All verification commands are executable via Bun or npm:

```bash
# 1. Validate offline WGSL compute shader syntax and uniform alignment
bun run check:shaders

# 2. Run TypeScript typechecks across all monorepo workspaces
bun run typecheck

# 3. Execute in-memory headless unit tests (25/25 suites via vgpu/mock)
bun run test:headless

# 4. Execute differential parity suite (117 assertions comparing CPU and GPU contracts)
bun run test:parity

# 5. Run headless browser regression tests in Chrome
bun run test:browser

# 6. Execute full multi-corpus benchmark suite in headless Chrome
bun run test:benchmark -- --fast

# 7. Verify bundle size budget
bun run check:bundle-size
```

---

## ⚠️ Breaking Changes (next major)

The deprecated aliases were removed. Update call sites before upgrading:

- `cpuAlgorithm` / `'parity'` → `cpuScorer: 'exact' | 'ufuzzy'` (default `'exact'`). Unknown scorers throw `IncompatibleOptionError` fail-closed.
- `extensions` / `SearchExtensionHooks` / `normalizeSearchExtensionHooks` → `hooks` / `SearchHooks` / `normalizeSearchHooks`.
- `suggest` option / `SuggestOptions` / `SUGGEST_*` / `normalizeSuggestOptions` → `autocomplete` / `AutocompleteOptions` / `AUTOCOMPLETE_*` / `normalizeAutocompleteOptions` (`SuggestionItem` / `SuggestResponse` / `autocomplete()` stay canonical).
- `folded` public fields → `normalized` (e.g. `packDataset(..., { normalized: true })`, `getStats().normalized`).
- `FORMAT_VERSION*` / `U2D4_*` / `SERIALIZED_*` → `DATASET_*` / `SNAPSHOT_*` / `LEGACY_SNAPSHOT_*` (wire magic bytes unchanged).
- `countUnicodeCodePoints`, `isAsciiTokens` / `isPrintableAsciiTokens`, `CPUEngine.searchUFuzzy` / `searchNative` removed (low-level CPU entry points are `searchWithUFuzzy` / `searchNaiveScan`).
- `powerPreference` is now forwarded to `GpuDevicePool.acquireDevice` (`navigator.gpu.requestAdapter({ powerPreference })`); unknown values throw `IncompatibleOptionError` fail-closed, even on CPU-only paths.

`SCORING_VERSION = 'parity-v1'` keeps its value; differential "parity harness" prose is unchanged.

---

## 📄 Documentation

- [**Snapshot format (`docs/snapshot-format.md`)**](./docs/snapshot-format.md): Versioned binary persistence, compatibility, and IndexedDB notes.
- [**Public API freeze (`docs/public-api.md`)**](./docs/public-api.md): Frozen 1.x contract — entry points, class contracts, result ordering, response echoes, errors, versioning, and deprecation policy.
- [**Support matrix + CPU baseline (`docs/support-matrix.md`)**](./docs/support-matrix.md): Browser/OS/runtime support, tested CPU-only baseline, `ufuzzy` opt-in rules, and the named compatibility suite.
- [**Text normalization (`docs/text-normalization.md`)**](./docs/text-normalization.md): Normative specification for preprocessing pipeline, version caps, delimiter sets, and scoring formulas.
- [**Naming conventions (`docs/naming-conventions.md`)**](./docs/naming-conventions.md): Domain-first naming rules enforced by `lint:naming`.

---

## 📄 License

MIT © [codewarnab](https://github.com/codewarnab)
