# WebGPU Fuzzy Search Benchmark Results

- **Date**: 2026-09-18T19:13:10.365Z

## Exact Substring Benchmark

| Dataset Size | Corpus | Packed VRAM | GPU Retained (med / p95) | CPU Parity (med / p95) | uFuzzy (med) | JS Native (med) | Speedup vs uFuzzy | Speedup vs Parity | Qualification |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 10,000 | ascii | 1.71 MB | 7.20 / 8.82 ms | 3.30 / 5.22 ms | 2.90 ms | 2.40 ms | 0.4x | 0.46x | pending-hardware |
| 100,000 | ascii | 16.88 MB | 48.70 / 55.16 ms | 22.80 / 29.90 ms | 20.80 ms | 16.40 ms | 0.43x | 0.47x | pending-hardware |

## Fuzzy Subsequence Benchmark

| Dataset Size | Corpus | Packed VRAM | GPU Retained (med / p95) | CPU Parity (med / p95) | uFuzzy (med) | JS Native (med) | Speedup vs uFuzzy | Speedup vs Parity | Qualification |
|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 10,000 | ascii | 1.71 MB | 14.60 / 19.12 ms | 2.60 / 3.46 ms | 2.80 ms | 1.20 ms | 0.19x | 0.18x | pending-hardware |
| 100,000 | ascii | 16.88 MB | 98.50 / 220.08 ms | 25.20 / 33.36 ms | 21.50 ms | 13.10 ms | 0.22x | 0.26x | pending-hardware |
