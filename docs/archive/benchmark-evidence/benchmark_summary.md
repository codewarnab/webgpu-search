# WebGPU Fuzzy Search Benchmark Results

- **Date**: 2026-09-19T05:41:04.089Z
- **Hardware Qualification**: ⚠️ pending-hardware (Software Vulkan / Mock Render)
- **Adapter**: {"vendor":"google","architecture":"swiftshader","device":"ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (LLVM 10.0.0) (0x0000C0DE)), SwiftShader driver)","description":"ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (LLVM 10.0.0) (0x0000C0DE)), SwiftShader driver)","renderer":"ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (LLVM 10.0.0) (0x0000C0DE)), SwiftShader driver)","maxBufferSizeMB":1024,"maxStorageBindingSizeMB":1024,"maxComputeWorkgroupsPerDimension":65535,"maxComputeInvocationsPerWorkgroup":256,"hasTimestampQuery":true}

## Exact Substring Benchmark

| Dataset Size | Corpus | Packed VRAM | GPU Retained (med / p95) | CPU Parity (med / p95) | uFuzzy (med) | JS Native (med) | Speedup vs uFuzzy | Speedup vs Parity | Qualification |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 10,000 | ascii | 1.71 MB | 5.60 / 8.76 ms | 1.60 / 3.90 ms | 1.70 ms | 1.00 ms | 0.3x | 0.29x | pending-hardware |
| 100,000 | ascii | 16.88 MB | 33.00 / 35.08 ms | 14.60 / 15.76 ms | 14.90 ms | 10.40 ms | 0.45x | 0.44x | pending-hardware |

## Fuzzy Subsequence Benchmark

| Dataset Size | Corpus | Packed VRAM | GPU Retained (med / p95) | CPU Parity (med / p95) | uFuzzy (med) | JS Native (med) | Speedup vs uFuzzy | Speedup vs Parity | Qualification |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 10,000 | ascii | 1.71 MB | 7.90 / 9.24 ms | 1.60 / 1.70 ms | 1.60 ms | 1.10 ms | 0.2x | 0.2x | pending-hardware |
| 100,000 | ascii | 16.88 MB | 63.80 / 67.64 ms | 14.70 / 15.18 ms | 15.80 ms | 10.60 ms | 0.25x | 0.23x | pending-hardware |
