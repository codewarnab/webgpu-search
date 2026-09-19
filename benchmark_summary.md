# WebGPU Fuzzy Search Benchmark Results

- **Date**: 2026-09-19T05:22:06.085Z
- **Hardware Qualification**: ⚠️ pending-hardware (Software Vulkan / Mock Render)
- **Adapter**: {"vendor":"google","architecture":"swiftshader","device":"ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (LLVM 10.0.0) (0x0000C0DE)), SwiftShader driver)","description":"ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (LLVM 10.0.0) (0x0000C0DE)), SwiftShader driver)","renderer":"ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (LLVM 10.0.0) (0x0000C0DE)), SwiftShader driver)","maxBufferSizeMB":1024,"maxStorageBindingSizeMB":1024,"maxComputeWorkgroupsPerDimension":65535,"maxComputeInvocationsPerWorkgroup":256,"hasTimestampQuery":true}

## Exact Substring Benchmark

| Dataset Size | Corpus | Packed VRAM | GPU Retained (med / p95) | CPU Parity (med / p95) | uFuzzy (med) | JS Native (med) | Speedup vs uFuzzy | Speedup vs Parity | Qualification |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 10,000 | ascii | 1.71 MB | 7.20 / 8.36 ms | 1.60 / 3.52 ms | 1.90 ms | 1.10 ms | 0.26x | 0.22x | pending-hardware |
| 100,000 | ascii | 16.88 MB | 50.50 / 55.40 ms | 20.10 / 35.64 ms | 21.30 ms | 21.10 ms | 0.42x | 0.4x | pending-hardware |

## Fuzzy Subsequence Benchmark

| Dataset Size | Corpus | Packed VRAM | GPU Retained (med / p95) | CPU Parity (med / p95) | uFuzzy (med) | JS Native (med) | Speedup vs uFuzzy | Speedup vs Parity | Qualification |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 10,000 | ascii | 1.71 MB | 10.50 / 12.84 ms | 2.70 / 5.52 ms | 3.20 ms | 2.00 ms | 0.3x | 0.26x | pending-hardware |
| 100,000 | ascii | 16.88 MB | 82.30 / 95.58 ms | 29.00 / 36.62 ms | 26.30 ms | 15.20 ms | 0.32x | 0.35x | pending-hardware |
