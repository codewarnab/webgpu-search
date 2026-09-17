# Turborepo Migration & NPM Package Launch Plan (Enhanced)

## 1. Executive Summary & Core Objectives

This plan outlines the architecture and execution roadmap for transforming `webgpu-fuzzy-search` from an interactive benchmark prototype into a production-grade **Turborepo monorepo** with two first-class workspaces:

1. **`packages/webgpu-fuzzy`**: A zero-config, highly-optimized hybrid search library published to npm. It dynamically routes queries between **WebGPU compute shaders (WGSL)** and **uFuzzy (CPU)**, accelerating large retained datasets while providing seamless fallback in non-WebGPU environments (Node.js, SSR, older browsers).
2. **`apps/benchmark`**: The interactive web dashboard, hardware evaluation suite, SVG/PNG chart renderer, and multi-size crossover matrix runner.

---

## 2. Identified Gaps in Original Plan & Architectural Solutions

An in-depth first-principles and software design audit of the initial migration plan revealed **10 critical technical gaps**. This improved plan resolves every one of them:

| # | Domain | Identified Gap in Original Plan | Concrete Solution in Enhanced Plan |
| :--- | :--- | :--- | :--- |
| **1** | **Consumer Breakage** | Original plan hid all low-level engines behind `SearchIndex`. `apps/benchmark` and `test-vgpu-mock.ts` rely on `WebGPUEngine`, `CPUEngine`, timestamp queries, and `generateDataset`. | Export high-level `SearchIndex` as the primary API, and export low-level engines (`WebGPUEngine`, `CPUEngine`, `WebGPUContextManager`, buffer packers) via named exports. Retain `dataset.ts` for synthetic generation. |
| **2** | **Runtime Portability** | `webgpu-engine.ts` invokes `document.createElement('canvas')` to detect unmasked GPU renderers. This throws a fatal `ReferenceError` inside **Web Workers** and **Node/SSR**. | Guard DOM calls with `typeof document !== 'undefined'`, support `OffscreenCanvas`, and ensure 100% Web Worker compatibility (the primary home for web search libraries). |
| **3** | **Hardware Resources** | Every `SearchIndex.create()` called `requestDevice()`. Browsers enforce strict limits (4–8 active `GPUDevice`s max). Multiple search indexes crashed WebGPU with device loss. | Implement a shared `WebGPUContextManager` singleton that multiplexes a single `GPUDevice` across multiple search indexes, with support for custom injected devices. |
| **4** | **API Symmetry** | `WebGPUEngine` returned `{ index, score, text }`, but `CPUEngine` returned `{ index, text }` (no score). Routing between CPU and GPU broke downstream score consumers. | Normalize CPU search outputs to calculate equivalent score rankings, establishing a unified `SearchResultItem` interface across both engines. |
| **5** | **Memory & String Bounds** | WGSL uses 64-byte slots (max 59 ASCII chars). Arbitrary user strings > 59 chars were silently truncated. Datasets > 2M rows exceed default 128MB `maxStorageBufferBindingSize`. | Add explicit string validation, configurable slot modes (64-byte default / 128-byte extended), and memory budget checks against hardware limits with automatic CPU fallback. |
| **6** | **Search Concurrency** | Current FIFO mutex queues every keystroke sequentially. Typing "auth" queued 4 GPU passes, delaying the final result and freezing the main thread during readbacks. | Introduce monotonic sequence tagging and `AbortSignal` cancellation to immediately drop stale query readbacks when newer keystrokes arrive. |
| **7** | **Shader Drift** | Shaders existed as both `.wgsl` and `.wgsl.ts`. Editing `.wgsl` created silent drift. Turbo task pipeline had no shader codegen dependency. | Use `tsup` raw text loading (`loader: { '.wgsl': 'text' }`) or an automated `codegen:shaders` step. Wire `check:shaders` into `turbo build`. |
| **8** | **Monorepo Dev Loop** | `apps/benchmark` consuming `webgpu-fuzzy` via `dist/` required background watch builds to see changes. | Configure Vite alias in `apps/benchmark` to resolve directly to `packages/webgpu-fuzzy/src/index.ts` in development for instant HMR. |
| **9** | **CI Portability** | `test-regression.ts` hardcoded a Windows Chrome path (`C:\Program Files\...`), causing immediate CI failure on Linux GitHub Actions runners. | Parameterize Chrome binary discovery via `CHROME_BIN` / `@puppeteer/browsers` and define a multi-tiered CI workflow (fast headless mock + browser regression). |
| **10** | **NPM Package Polish** | Missing package metadata: `"sideEffects": false`, `"files": ["dist"]`, dual ESM/CJS export map, TypeScript declaration maps, and bundle size budget. | Complete production-ready `package.json`, `tsup.config.ts`, and bundle size threshold (< 20 KB min+gzip). |

---

## 3. Target Directory Structure

```
webgpu-fuzzy-search/
├── turbo.json                          # Turborepo v2 pipeline configuration
├── package.json                        # Root workspace configuration & global dev scripts
├── bun.lock                            # Unified lockfile
├── tsconfig.base.json                  # Shared TypeScript compiler options
├── PLAN.md                             # This architectural roadmap
├── .github/
│   └── workflows/
│       └── ci.yml                      # Automated CI: Typecheck, Shader check, Mock test
│
├── apps/
│   └── benchmark/                      # Interactive benchmark UI & evaluation suite
│       ├── package.json                # Depends on "webgpu-fuzzy": "workspace:*"
│       ├── tsconfig.json               # Extends ../../tsconfig.base.json
│       ├── vite.config.ts              # Vite bundler with workspace dev resolution
│       ├── index.html                  # Dashboard entry HTML
│       ├── public/                     # Static icons, favicons, fonts
│       │   ├── favicon.svg
│       │   └── icons.svg
│       └── src/
│           ├── main.ts                 # Dashboard UI controller & SVG chart renderer
│           ├── benchmark.ts            # Matrix runner & crossover calculator
│           ├── dataset.ts              # Synthetic code path generator
│           ├── export.ts               # CSV and SVG/PNG chart export utilities
│           └── style.css               # Dashboard styling
│
├── packages/
│   └── webgpu-fuzzy/                   # Core publishable NPM package
│       ├── package.json                # Package metadata, exports map, types
│       ├── tsconfig.json               # Extends ../../tsconfig.base.json
│       ├── tsup.config.ts              # Dual ESM/CJS, .d.ts, raw WGSL bundling
│       ├── README.md                   # Public npm documentation & quickstart
│       └── src/
│           ├── index.ts                # Public entry point: SearchIndex & low-level exports
│           ├── types.ts                # Unified public types & interface contracts
│           ├── context-manager.ts      # Shared GPUDevice singleton & context pooling
│           ├── hybrid-index.ts         # High-level SearchIndex with auto-routing
│           ├── buffer.ts               # String packing, UTF-8 sanitation & GPU byte layouts
│           ├── webgpu-engine.ts        # Retained VRAM WebGPU compute pipeline
│           ├── cpu-engine.ts           # uFuzzy & native CPU search engine
│           └── shaders/
│               ├── fuzzy.wgsl          # Subsequence & word-boundary scoring shader
│               ├── substring.wgsl      # Exact/case-insensitive substring shader
│               ├── fuzzy.wgsl.ts       # Inlined fallback string
│               └── substring.wgsl.ts   # Inlined fallback string
│
└── scripts/
    ├── test-vgpu-mock.ts               # Headless mock device unit tests (vgpu/mock)
    └── test-regression.ts              # Cross-platform Puppeteer browser tests
```

---

## 4. Shared Monorepo Tooling Configuration

### 4.1. Root `package.json`

```json
{
  "name": "webgpu-fuzzy-monorepo",
  "private": true,
  "type": "module",
  "workspaces": [
    "apps/*",
    "packages/*"
  ],
  "scripts": {
    "dev": "turbo dev",
    "build": "turbo build",
    "test": "turbo test",
    "test:mock": "bun scripts/test-vgpu-mock.ts",
    "test:browser": "bun scripts/test-regression.ts",
    "check:shaders": "turbo check:shaders",
    "typecheck": "turbo typecheck",
    "lint": "turbo lint",
    "clean": "turbo clean && rm -rf node_modules apps/*/dist packages/*/dist"
  },
  "devDependencies": {
    "@webgpu/types": "^0.1.72",
    "puppeteer-core": "^25.10.0",
    "turbo": "^2.4.4",
    "typescript": "~6.0.2",
    "vgpu": "^0.5.0"
  },
  "engines": {
    "node": ">=18.0.0",
    "bun": ">=1.1.0"
  }
}
```

### 4.2. Turborepo Orchestration (`turbo.json`)

```json
{
  "$schema": "https://turbo.build/schema.json",
  "tasks": {
    "check:shaders": {
      "inputs": ["src/shaders/**/*.wgsl"],
      "outputs": []
    },
    "typecheck": {
      "dependsOn": ["^build"],
      "inputs": ["src/**/*.ts", "tsconfig.json"],
      "outputs": []
    },
    "build": {
      "dependsOn": ["^build", "check:shaders"],
      "inputs": ["src/**", "package.json", "tsconfig.json", "tsup.config.ts"],
      "outputs": ["dist/**"]
    },
    "test": {
      "dependsOn": ["build"],
      "inputs": ["src/**", "scripts/**"],
      "outputs": []
    },
    "dev": {
      "persistent": true,
      "cache": false
    },
    "lint": {
      "inputs": ["src/**"]
    }
  }
}
```

### 4.3. Base TypeScript Config (`tsconfig.base.json`)

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "strict": true,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "skipLibCheck": true,
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "forceConsistentCasingInFileNames": true,
    "verbatimModuleSyntax": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true
  }
}
```

---

## 5. Core Library Design (`packages/webgpu-fuzzy`)

### 5.1. Public TypeScript Contracts (`src/types.ts`)

```ts
export type SearchMode = 'fuzzy' | 'substring';
export type EngineType = 'webgpu' | 'cpu';

export interface SearchOptions {
  mode?: SearchMode;              // Default: 'fuzzy'
  limit?: number;                 // Max results (Default: 50, clamp: 8192)
  caseSensitive?: boolean;        // Default: false
  signal?: AbortSignal;           // Cancel stale queries during rapid typing
}

export interface SearchResultItem {
  index: number;                  // Original index in dataset
  score: number;                  // Higher is better (normalized integer)
  text: string;                   // Resolved match string
}

export interface SearchTimings {
  queryUploadMs: number;          // Uniform write latency
  encodeSubmitMs: number;         // Command recording & submit latency
  gpuExecutionMs: number | null;  // Hardware timestamp query (null if unsupported)
  readbackMs: number;             // mapAsync() & CPU candidate slice latency
  totalMs: number;                // Total wall-clock query duration
}

export interface SearchResponse {
  query: string;
  mode: SearchMode;
  engine: EngineType;             // Which engine serviced this query
  totalMatches: number;           // Total items passing threshold
  candidateCount: number;         // Scored candidates returned from GPU/CPU
  hasOverflow: boolean;           // True if matches > candidate pool capacity (8192)
  results: SearchResultItem[];    // Top-K ranked results
  timings: SearchTimings;
}

export interface IndexOptions {
  threshold?: number;             // Item count cutoff for CPU vs GPU (Default: 30,000)
  preferGpu?: boolean;            // Force WebGPU if available regardless of size
  device?: GPUDevice;             // Custom injected GPUDevice (for testing/context sharing)
  powerPreference?: GPUPowerPreference; // 'high-performance' | 'low-power'
  slotBytes?: 64 | 128;           // Row slot width: 64 (59 chars) or 128 (123 chars)
}

export interface IndexStats {
  size: number;
  engine: EngineType;
  vramAllocatedBytes: number;
  adapterVendor?: string;
  adapterRenderer?: string;
}
```

### 5.2. Public API Surface (`src/index.ts`)

```ts
// High-Level Primary API
export { SearchIndex } from './hybrid-index';

// Low-Level Engines & Hardware Utilities for Power Users & Benchmarks
export { WebGPUEngine } from './webgpu-engine';
export { CPUEngine } from './cpu-engine';
export { WebGPUContextManager } from './context-manager';
export { packStringsToGPUBuffer, sanitizeStringForSlot } from './buffer';

// Unified Types
export type * from './types';
```

### 5.3. Consumer Usage Examples

#### Quickstart: Zero-Config Search Index
```ts
import { SearchIndex } from 'webgpu-fuzzy';

// 1. Create index with arbitrary string array
const index = await SearchIndex.create(items, {
  threshold: 30_000 // Auto-routes to CPU if items < 30k, WebGPU if >= 30k
});

// 2. Query with keystroke debouncing support
const controller = new AbortController();
const response = await index.search('AuthController', {
  mode: 'fuzzy',
  limit: 25,
  signal: controller.signal
});

console.log(`Found ${response.totalMatches} matches in ${response.timings.totalMs.toFixed(2)}ms via ${response.engine}`);
for (const item of response.results) {
  console.log(`[${item.score}] ${item.text}`);
}

// 3. Clean up GPU buffers when done
index.destroy();
```

#### Advanced: Web Worker Usage (Zero UI Stutter)
```ts
// worker.ts
import { SearchIndex } from 'webgpu-fuzzy';

let index: SearchIndex;

self.onmessage = async (e) => {
  const { type, payload, id } = e.data;
  if (type === 'INIT') {
    index = await SearchIndex.create(payload.items);
    self.postMessage({ id, type: 'READY' });
  } else if (type === 'SEARCH') {
    const res = await index.search(payload.query, payload.options);
    self.postMessage({ id, type: 'RESULTS', payload: res });
  }
};
```

---

## 6. Detailed Technical Architecture of Key Components

### 6.1. WebGPU Context Manager (`src/context-manager.ts`)
Solves **Gap 3 (GPU Context Limits)**:
- Maintains a reference-counted shared `GPUDevice` instance across all `SearchIndex` instances.
- Allows user-injected custom devices (used in `vgpu/mock` testing or apps already running Three.js/WebGPU).
- Listens to `device.lost` events to notify active indexes and gracefully degrade to CPU mode if the GPU crashes or sleeps.

### 6.2. Web Worker & SSR Safe Initialization (`src/webgpu-engine.ts`)
Solves **Gap 2 (Web Worker / Node Crash)**:
- Replaces direct `document.createElement('canvas')` with safe environment detection:
  ```ts
  function getUnmaskedRenderer(): string {
    if (typeof OffscreenCanvas !== 'undefined') {
      try {
        const canvas = new OffscreenCanvas(1, 1);
        const gl = canvas.getContext('webgl');
        // read debug info...
      } catch { /* ignore */ }
    } else if (typeof document !== 'undefined') {
      // DOM fallback
    }
    return 'WebGPU Hardware';
  }
  ```
- If `navigator?.gpu` is undefined (Node.js, SSR, unsupported browsers), `init()` returns `false` without throwing, triggering instant CPU fallback.

### 6.3. Buffer Packer & Memory Guard (`src/buffer.ts`)
Solves **Gap 5 (String Length & Memory Bounds)**:
- Validates string length: 64-byte slots store length (4 bytes) + 60 bytes of characters (59 usable ASCII chars + null termination).
- Sanitizes non-ASCII characters by stripping diacritics or replacing with closest ASCII match to prevent GPU byte mismatch.
- Checks total buffer size against `device.limits.maxStorageBufferBindingSize` (minimum 128 MB). If dataset requires 256 MB on a 128 MB device, it rejects GPU allocation and warns, routing to CPU engine.

### 6.4. Unified Scoring & Result Parity (`src/cpu-engine.ts`)
Solves **Gap 4 (Asymmetric Output & Missing Scores)**:
- Updates `CPUEngine.searchUFuzzy()` to extract matching order and assign normalized synthetic scores (`1000 - rank * 2`) to match the descending score contract of `WebGPUEngine`.
- Both engines output identical `SearchResultItem[]` arrays.

### 6.5. Keystroke Concurrency & Stale Query Dropping (`src/hybrid-index.ts`)
Solves **Gap 6 (Keystroke Concurrency)**:
- Tracks a monotonic `activeQueryToken: number`.
- When a search starts, it captures the current token.
- If an `AbortSignal` fires OR a newer query token has been issued while the GPU pass was in flight, the CPU readback and sorting steps are aborted immediately, preventing unnecessary main-thread contention.

---

## 7. Package Configuration & Build Setup

### 7.1. `packages/webgpu-fuzzy/package.json`

```json
{
  "name": "webgpu-fuzzy",
  "version": "0.1.0",
  "description": "Ultra-fast hybrid fuzzy and substring search library powered by WebGPU compute shaders with uFuzzy CPU fallback",
  "type": "module",
  "main": "./dist/index.cjs",
  "module": "./dist/index.js",
  "types": "./dist/index.d.ts",
  "exports": {
    ".": {
      "types": "./dist/index.d.ts",
      "import": "./dist/index.js",
      "require": "./dist/index.cjs"
    }
  },
  "files": [
    "dist",
    "README.md",
    "LICENSE"
  ],
  "sideEffects": false,
  "scripts": {
    "build": "tsup",
    "dev": "tsup --watch",
    "check:shaders": "vgpu check src/shaders/fuzzy.wgsl && vgpu check src/shaders/substring.wgsl",
    "typecheck": "tsc --noEmit"
  },
  "dependencies": {
    "@leeoniya/ufuzzy": "^1.0.19"
  },
  "devDependencies": {
    "@webgpu/types": "^0.1.72",
    "tsup": "^8.4.0",
    "typescript": "~6.0.2",
    "vgpu": "^0.5.0"
  },
  "peerDependencies": {
    "@webgpu/types": "^0.1.72"
  },
  "peerDependenciesMeta": {
    "@webgpu/types": {
      "optional": true
    }
  },
  "keywords": [
    "webgpu",
    "fuzzy-search",
    "substring-search",
    "search",
    "autocomplete",
    "wgsl",
    "compute-shader",
    "ufuzzy"
  ],
  "license": "MIT"
}
```

### 7.2. `packages/webgpu-fuzzy/tsup.config.ts`

```ts
import { defineConfig } from 'tsup';

export default defineConfig({
  entry: ['src/index.ts'],
  format: ['esm', 'cjs'],
  dts: true,
  clean: true,
  sourcemap: true,
  minify: false,
  treeshake: true,
  loader: {
    '.wgsl': 'text'
  },
  esbuildOptions(options) {
    options.banner = {
      js: '/* webgpu-fuzzy | MIT License */'
    };
  }
});
```

### 7.3. `apps/benchmark/vite.config.ts`
Solves **Gap 8 (HMR Dev Workflow)**:

```ts
import { defineConfig } from 'vite';
import path from 'path';

export default defineConfig({
  resolve: {
    alias: {
      // In development, resolve webgpu-fuzzy directly to TypeScript source for instant HMR
      'webgpu-fuzzy': path.resolve(__dirname, '../../packages/webgpu-fuzzy/src/index.ts')
    }
  },
  server: {
    port: 5173,
    open: true
  }
});
```

---

## 8. Implementation Roadmap & Phases

```
┌────────────────────────────────────────────────────────┐
│ Phase 1: Monorepo Foundation                           │
│ - Configure root package.json & workspaces             │
│ - Set up turbo.json v2 & tsconfig.base.json            │
│ - Verify Bun workspace linking                         │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Phase 2: Core Package Extraction (webgpu-fuzzy)        │
│ - Implement src/types.ts & src/context-manager.ts      │
│ - Implement src/buffer.ts with string sanitation       │
│ - Refactor webgpu-engine.ts (Worker safe, context reuse│
│ - Refactor cpu-engine.ts with unified score ranking    │
│ - Implement hybrid-index.ts (SearchIndex auto-routing) │
│ - Configure tsup.config.ts with WGSL raw loader        │
│ - Verify package build: bun run --filter webgpu-fuzzy  │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Phase 3: Benchmark App Migration (apps/benchmark)      │
│ - Scaffold apps/benchmark directory                    │
│ - Move UI files, dataset.ts, export.ts, benchmark.ts   │
│ - Update imports to use 'webgpu-fuzzy' workspace dep   │
│ - Configure vite.config.ts with direct src alias       │
│ - Verify UI & live benchmarking at localhost:5173      │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Phase 4: CI & Cross-Platform Verification              │
│ - Update scripts/test-vgpu-mock.ts to test SearchIndex │
│ - Parameterize scripts/test-regression.ts (CHROME_BIN) │
│ - Add .github/workflows/ci.yml (mock + typecheck)      │
│ - Run full turbo build, test, and typecheck            │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ Phase 5: NPM Release Polish                            │
│ - Audit bundle size (< 20 KB min+gzip)                 │
│ - Run npm pack --dry-run to verify clean tarball       │
│ - Complete packages/webgpu-fuzzy/README.md with demos   │
└────────────────────────────────────────────────────────┘
```

### Detailed Phase Tasks & Checklists

#### Phase 1: Monorepo Foundation
- [x] Update root `package.json` with `"workspaces": ["apps/*", "packages/*"]`.
- [x] Add `turbo.json` with dependency tasks (`build`, `check:shaders`, `typecheck`, `test`).
- [x] Create `tsconfig.base.json` for shared TS compiler settings.
- [x] Run `bun install` to generate unified root lockfile.

#### Phase 2: Package Extraction (`packages/webgpu-fuzzy`)
- [x] Create `packages/webgpu-fuzzy/` structure.
- [x] Implement `src/types.ts` with complete search contracts.
- [x] Implement `src/context-manager.ts` for shared `GPUDevice` management.
- [x] Implement `src/buffer.ts` with length validation and ASCII byte packing.
- [x] Update `src/webgpu-engine.ts` to remove DOM dependencies (`document.createElement`) and integrate with `context-manager`.
- [x] Update `src/cpu-engine.ts` to return normalized scores.
- [x] Implement `src/hybrid-index.ts` with `SearchIndex`, auto-routing, and `AbortSignal` cancellation.
- [x] Configure `tsup.config.ts` and verify build produces `dist/index.js`, `dist/index.cjs`, `dist/index.d.ts`.

#### Phase 3: Benchmark App Migration (`apps/benchmark`)
- [x] Create `apps/benchmark/package.json` with `"dependencies": { "webgpu-fuzzy": "workspace:*" }`.
- [x] Move `index.html`, `src/style.css`, `src/main.ts`, `src/benchmark.ts`, `src/dataset.ts`, `src/export.ts` to `apps/benchmark/`.
- [x] Configure `apps/benchmark/vite.config.ts` with dev alias to source files.
- [x] Verify benchmark dashboard loads and functions identically in browser.

#### Phase 4: CI & Test Hardening
- [x] Refactor `scripts/test-vgpu-mock.ts` to import from `webgpu-fuzzy` and test `SearchIndex` directly.
- [x] Refactor `scripts/test-regression.ts` to use `process.env.CHROME_BIN || puppeteer.executablePath()`.
- [x] Create `.github/workflows/ci.yml` running `bun run check:shaders`, `bun run build`, and `bun run test:mock`.

#### Phase 5: NPM Packaging & Documentation
- [x] Verify `npm pack --dry-run` in `packages/webgpu-fuzzy` contains only `dist/`, `README.md`, `LICENSE`.
- [x] Confirm bundle footprint meets budget (< 20 KB gzipped).
- [x] Write `packages/webgpu-fuzzy/README.md` with:
  - 3-line quickstart snippet.
  - Hybrid routing explanation.
  - Benchmark comparison table against `uFuzzy`.
  - Web Worker recipe.
- [x] Add MIT `LICENSE` file.

---

## 9. Verification & Acceptance Criteria

1. **Clean Workspace Isolation**: `bun run build` via Turbo builds `packages/webgpu-fuzzy` and `apps/benchmark` in parallel with zero circular dependencies.
2. **Deterministic Mock Tests (0ms GPU)**: `bun run test:mock` passes in headless CI with 0 failures using `vgpu/mock`.
3. **Web Worker Portability**: `SearchIndex.create()` can be called inside a standard Dedicated Web Worker without DOM errors.
4. **Scoring Consistency**: Results returned from CPU (`< 30k` items) and WebGPU (`>= 30k` items) adhere to the exact same `SearchResultItem` interface with valid descending scores.
5. **No Context Exhaustion**: Creating 10 `SearchIndex` instances sequentially reuses the underlying `GPUDevice` and does not trigger browser WebGPU context loss.
6. **Zero Stale UI Overwrites**: Triggering rapid queries with `AbortSignal` cancels in-flight candidate readback and returns the freshest result cleanly.

