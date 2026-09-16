# WebGPU vs uFuzzy Benchmark & Search Engine

[![Live Demo](https://img.shields.io/badge/demo-online-brightgreen.svg)](https://webgpu-fuzzy-search.vercel.app)
[![npm version](https://img.shields.io/npm/v/webgpu-search.svg)](https://www.npmjs.com/package/webgpu-search)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An ultra-fast hybrid fuzzy and substring search engine powered by parallel **WebGPU compute shaders (WGSL)** on retained VRAM, backed by **uFuzzy** CPU fallback, and featuring **zero-stutter Web Worker offloading** to guarantee **120 FPS** responsiveness across millions of records.

👉 **[Live Interactive Benchmark & Playground](https://webgpu-fuzzy-search.vercel.app)**

---

## ⚡ Monorepo Structure

- **[`packages/webgpu-search`](./packages/webgpu-search)**: Zero-dependency core library published to npm. Provides high-level `SearchIndex` with dynamic crossover routing, low-level `WebGPUEngine` and `CPUEngine`, and SSR/Worker-safe memory primitives.
- **[`apps/benchmark`](./apps/benchmark)**: Interactive evaluation dashboard and live test suite comparing WebGPU compute against CPU algorithms across 10,000 to 2,000,000+ records.

---

## 🚀 Quick Start

```bash
# Clone and install dependencies
git clone https://github.com/codewarnab/webgpu-fuzzy-search.git
cd webgpu-fuzzy-search
bun install

# Run the local benchmark dashboard
bun run dev
```

Open **`http://localhost:5173`** in any WebGPU-capable browser (Chrome, Edge, Safari 18+, or Firefox Nightly).

---

## 📊 Real-World Benchmark Results (Intel Iris Xe / D3D11 ANGLE)

Tested on **Intel(R) Iris(R) Xe Graphics (gen-12lp)** via Direct3D11 ANGLE on Windows 11 with query `"AuthController"`:

### 1. Exact Substring Search Matrix
| Dataset Size | WebGPU Retained | uFuzzy (CPU) | Native JS | Speedup vs uFuzzy | UI Frame Rate (Worker vs Main) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **10,000** | 4.43 ms | 1.47 ms | 1.27 ms | **0.33x** | ⚡ 120 FPS vs 60 FPS |
| **100,000** | 3.13 ms | 8.20 ms | 7.37 ms | **2.62x** | ⚡ 120 FPS vs 60 FPS |
| **500,000** | 4.43 ms | 36.47 ms | 42.37 ms | **8.23x** | ⚡ 120 FPS vs 27 FPS |
| **1,000,000** | 5.50 ms | 97.93 ms | 91.67 ms | **17.81x** | ⚡ 120 FPS vs 10 FPS |
| **2,000,000** | 11.10 ms | 173.53 ms | 158.40 ms | **15.63x** | ⚡ 120 FPS vs 6 FPS |

### 2. Fuzzy Subsequence Search Matrix
| Dataset Size | WebGPU Retained | uFuzzy (CPU) | Native JS | Speedup vs uFuzzy | UI Frame Rate (Worker vs Main) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **10,000** | 3.37 ms | 0.97 ms | 2.50 ms | **0.29x** | ⚡ 120 FPS vs 60 FPS |
| **100,000** | 3.20 ms | 9.60 ms | 10.03 ms | **3.00x** | ⚡ 120 FPS vs 60 FPS |
| **500,000** | 5.60 ms | 35.67 ms | 39.57 ms | **6.37x** | ⚡ 120 FPS vs 28 FPS |
| **1,000,000** | 59.10 ms* | 69.73 ms | 80.17 ms | **1.18x** | ⚡ 120 FPS vs 14 FPS |
| **2,000,000** | 17.27 ms | 173.63 ms | 167.33 ms | **10.06x** | ⚡ 120 FPS vs 6 FPS |

*\*Note: The one-off 59ms spike at 1M on integrated Intel Iris Xe is typical of dynamic UMA buffer paging under D3D11 before stabilizing at 17.2ms for 2M.*

---

## 🔬 Architectural Deep-Dive

### 1. The Retained VRAM vs Cold Upload Reality
- **Cold Upload**: Transferring 1,000,000 records (~64 MB packed) over the PCIe bus via `queue.writeBuffer` takes **~8–20 ms**. If an application re-uploads data on every keystroke, **uFuzzy on CPU will always win**.
- **Retained VRAM**: When dataset storage buffers are pre-loaded once into GPU VRAM, subsequent keystrokes only upload a **272-byte uniform buffer** (query characters, length, flags). The compute shader executes in parallel across thousands of GPU threads in **1.5ms – 5.5ms**.

### 2. The `mapAsync()` Fixed Latency Trap & Candidate Compaction
- WebGPU buffer readbacks have an inherent synchronization floor (~1ms to 3ms). Below 50,000 records, CPU uFuzzy completes in <1ms, making CPU faster at small scales.
- At 2,000,000 items, reading back a full result array across PCIe would introduce an extra 10–20ms transfer penalty.
- **Candidate Pool Compaction**: The WGSL compute shader writes up to **8,192 scored candidate matches** (`{ index: u32, score: i32 }`) into a compact 64 KB output buffer using atomic counters (`atomicAdd(&output.count, 1u)`). The CPU then sorts these candidates descending to yield the top 1,000 matches in <1ms, avoiding PCIe bus stalls.

### 3. Web Worker Offloading: The 120 FPS Guarantee
Searching 2,000,000 records directly on the browser UI thread freezes the render loop for 170ms+ (plunging frame rates to 6 FPS).
- By offloading the `WebGPUEngine`, buffers, and search execution to a dedicated **Web Worker**:
  1. The main UI thread spends ~0.05ms posting the query to the worker, maintaining a locked **120 FPS**.
  2. The worker dispatches compute pipelines and awaits buffer readbacks off-thread.
  3. Active `AbortController` cancellation discards in-flight passes during rapid user typing.
  4. Monotonic query IDs (`queryId`) ensure the UI only presents matches corresponding to the latest keystroke.

---

## 🛠️ Verification & Scripts

```bash
# Validate offline WGSL compute shader syntax
bun run check:shaders

# Run TypeScript typechecks across packages and apps
bun run typecheck

# Execute in-memory headless unit tests (vgpu/mock)
bun run test:mock

# Build packages and benchmark app
bun run build
```

---

## 📄 License

MIT © [codewarnab](https://github.com/codewarnab)
