# webgpu-search

[![Live Demo](https://img.shields.io/badge/demo-online-brightgreen.svg)](https://webgpu-search.vercel.app)
[![npm version](https://img.shields.io/npm/v/webgpu-search.svg)](https://www.npmjs.com/package/webgpu-search)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Fuzzy and substring search for the browser, powered by WebGPU compute shaders with a bit-exact CPU fallback and first-class Web Worker support.

Datasets are Unicode-normalized once and packed into `u32` token streams loaded into **retained VRAM**, so each keystroke uploads a 32-byte uniform header instead of the corpus. WGSL compute shaders score up to 8,192 candidates per query into a single compact readback buffer. Below the crossover threshold (default 30,000 records), or wherever WebGPU isn't available, a **CPU exact scorer** runs the same deterministic ranking — results are identical to GPU output by `(score DESC, index ASC)`.

**[Live demo](https://webgpu-search.vercel.app)** — includes a benchmark comparing WebGPU, exact CPU, uFuzzy, and Fuse.js, measured live in your browser.

## Features

- **WebGPU compute search** — retained buffer scanning in WGSL; a 32-byte uniform upload per query, not a dataset re-upload.
- **Deterministic CPU/GPU parity** — a bit-exact CPU scorer with strict tie-breaking, so results are byte-identical between engines.
- **Automatic routing** — configurable crossover threshold (`threshold`, default 30,000 records); CPU-only fallback for `token`/`prefix` modes, typo tolerance, and devices without WebGPU.
- **Typed documents** — `DocumentIndex` for multi-field, weighted search with columnar filters, facet aggregation, autocomplete/did-you-mean, Unicode-safe highlighting, live mutations, and CRC32-checked snapshot persistence.
- **Off the main thread** — `webgpu-search/worker` entry and `SearchWorkerClient` run the whole index in a worker, returning compact `{ index, score }` hits enriched on the main thread; stale in-flight queries abort on rapid typing.
- **Portable** — ESM + CJS, SSR/Node/Worker-safe (no unguarded DOM globals); one runtime dependency (`@leeoniya/ufuzzy`, the opt-in CPU fuzzy backend).

## Installation

```bash
npm install webgpu-search
```

Runs anywhere JavaScript does; the GPU path needs a WebGPU-capable browser. See the [support matrix](./docs/support-matrix.md) for tested browsers and the CPU-only baseline.

## Quick start

```ts
import { SearchIndex } from 'webgpu-search';

// 1. Create an index from an array of strings
const index = await SearchIndex.create([
  'packages/core/src/AuthController.ts',
  'src/views/ユーザー設定/Profile.vue',
  'docs/api/Straße_v2.md',
  'assets/icons/🧑‍💻_developer.png'
], {
  threshold: 30_000,      // records below this route to the CPU scorer
  caseSensitive: false    // fixed at construction: NFC + CaseFolding 16.0.0
});

// 2. Query
const response = await index.search('ユーザー', {
  mode: 'fuzzy',                // 'fuzzy' | 'substring' | 'token' | 'prefix'
  limit: 25,                    // clamped to 1..8192
  onQueryTooLong: 'throw'       // 'throw' (QueryTooLongError) or 'cpu-fallback'
});

console.log(`${response.totalMatches} matches in ${response.timings.totalMs.toFixed(2)}ms via ${response.engine}`);
for (const hit of response.results) {
  console.log(`[Score: ${hit.score}] #${hit.index}: ${hit.text}`);
}

// 3. Inspect memory footprint and free VRAM when done
const stats = index.getStats();
console.log(`VRAM: ${(stats.vramAllocatedBytes / 1024).toFixed(1)} KB for ${stats.tokenCount} tokens`);
index.destroy();
```

## Documents, filters, facets, autocomplete

For anything beyond a flat string list, use `DocumentIndex`: weighted multi-field search over typed records, structured filters and facets, autocomplete suggestions, highlighting, incremental mutations, and versioned snapshots.

```ts
import { DocumentIndex } from 'webgpu-search';

const index = await DocumentIndex.create(records, {
  fields: [
    'title',                          // weight 1 by default
    { name: 'summary', weight: 0.5 }
  ],
  filterFields: [
    { name: 'category', type: 'string' }
  ]
});

const response = await index.search('gpu shader', {
  limit: 10,
  filter: { category: 'GPU' },
  facets: { category: { type: 'terms', field: 'category' } },
  autocomplete: true                  // prefix + did-you-mean suggestions
});

for (const hit of response.results) {
  console.log(hit.id, hit.score, hit.matchedField); // hit.doc is the original record
}

const suggestions = await index.autocomplete('gpu');
```

See [`docs/public-api.md`](./docs/public-api.md) for the frozen 1.x contract (field schemas, filter expression syntax, error types, versioning policy).

## Worker offloading

`SearchWorkerClient` mirrors `DocumentIndex` over `postMessage`:

```ts
import { SearchWorkerClient } from 'webgpu-search';

const client = new SearchWorkerClient();
await client.init(records, { fields: ['title', 'summary'] });
const response = await client.search('gpu', { limit: 10, signal: abortController.signal });
await client.destroy(); // terminates the owned worker
```

The worker returns compact `{ index, score }` hit arrays; only matched documents are cloned back, so multi-million-record corpora never stall the UI thread. For custom worker bundling, use the dedicated `webgpu-search/worker` entry (`startSearchWorker`, `isDedicatedWorker`); it only calls `postMessage` inside a dedicated worker scope.

## Low-level pipeline

`SearchIndex` composes public primitives that you can drive directly — packing, binary serialization, and the engines:

```ts
import { WebGPUEngine, CPUEngine, packDataset, serializeDataset, deserializeDataset } from 'webgpu-search';

// 1. Pack strings into normalized u32 Unicode scalar tokens
const packed = packDataset(rawStrings, { normalized: true });

// 2. Serialize into the dataset binary container (magic 0x55324632 + CRC32 checksum)
const buffer: ArrayBuffer = serializeDataset(packed);

// 3. Restore with fail-closed integrity checks
const restored = deserializeDataset(buffer);

// 4. Load into the WebGPU compute engine
const gpu = new WebGPUEngine();
await gpu.init();
await gpu.loadDataset(restored);
const result = await gpu.search('test', { mode: 'substring', limit: 50 });

console.log(`GPU: ${result.timings.gpuExecutionMs}ms, readback: ${result.timings.readbackMs}ms`);
gpu.destroy();
```

`CPUEngine` (`searchWithUFuzzy` / `searchNaiveScan`) implements the same contract without a GPU.

## How it works

**Retained VRAM vs cold upload.** Transferring ~64 MB of packed `u32` tokens over the PCIe bus costs ~8–20 ms; re-uploading on every keystroke would make CPU search always win. Retained datasets change the per-query cost to a 32-byte uniform header plus a 512-byte query buffer (up to 128 `u32` code points).

**Candidate compaction & deterministic sorting.** WGSL shaders write up to 8,192 scored candidates (`{ index: u32, score: i32 }`) into a compact 65,544-byte output buffer via atomic counter. Both engines sort by `(score DESC, index ASC)` in signed 32-bit integer arithmetic — outside buffer overflow, rankings are bit-identical between WebGPU and CPU.

**Device safety.** Device acquisition routes through `WebGPUContextManager` / `GpuDevicePool`; device loss triggers automatic re-acquisition via `rebuildGpu()`, never a plain `navigator.gpu.requestDevice()` per query.

## Benchmarks

Skip hand-gathered numbers — run the comparison in your own browser: the [live demo](https://webgpu-search.vercel.app/#benchmark) measures WebGPU vs exact CPU vs uFuzzy vs Fuse.js against your real GPU and reports per-engine timings.

Reproducible multi-corpus runs (10k–2M records; ASCII, CJK, emoji) come from the automated headless runner:

```bash
bun run test:benchmark -- --fast
```

Headless CI uses software Vulkan (SwiftShader), so its GPU columns are qualified `pending-hardware`; on real hardware (NVIDIA, Apple Silicon, Intel) GPU times drop dramatically. Frozen fixtures and methodology: [`docs/benchmarks.md`](./docs/benchmarks.md).

## Repository layout

- [`packages/webgpu-search`](./packages/webgpu-search) — the core library published to npm.
- [`apps/site`](./apps/site) — product homepage with the embedded live benchmark (`/#benchmark`; built to `apps/site/dist`, dev server on `:5178`).
- [`apps/benchmark`](./apps/benchmark) — full local benchmark harness (dev server on `:5173`; used by `test:browser` / `test:benchmark`).
- [`examples/`](./examples) — integration recipes for React, Vue, Svelte, and vanilla.
- [`scripts/bench-data/`](./scripts/bench-data) — deterministic benchmark fixture generators.

## Verification

```bash
bun install

bun run check:shaders    # validate offline WGSL syntax + uniform alignment
bun run typecheck        # TypeScript across all workspaces
bun run test:headless    # in-memory unit suites (vgpu/mock)
bun run test:parity      # differential CPU/GPU parity harness
bun run test:browser     # headless browser regression tests
bun run test:benchmark   # full multi-corpus benchmark suite
bun run check:bundle-size
```

## Documentation

- [Snapshot format](./docs/snapshot-format.md) — versioned binary persistence, compatibility, IndexedDB notes.
- [Public API freeze](./docs/public-api.md) — frozen 1.x contract and deprecation policy.
- [Support matrix + CPU baseline](./docs/support-matrix.md) — browsers, OS, runtimes, `ufuzzy` opt-in rules.
- [Text normalization](./docs/text-normalization.md) — normative preprocessing spec, version caps, scoring formulas.
- [Naming conventions](./docs/naming-conventions.md) — domain-first naming rules enforced by `lint:naming`.
- [Benchmarks](./docs/benchmarks.md) — frozen fixtures, headless CPU baseline, hardware qualification rules.
- [Contributing](./CONTRIBUTING.md) · [Security](./SECURITY.md) · [Code of Conduct](./CODE_OF_CONDUCT.md)

## License

[MIT](./LICENSE) — © [codewarnab](https://github.com/codewarnab) and webgpu-search contributors.
