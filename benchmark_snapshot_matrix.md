# Benchmark Snapshot Matrix (headless CPU baseline)

- Fixtures: `1.0.0` (see `scripts/benchmark-fixtures.ts`, `docs/benchmarks.md` §1)
- Generated: 2026-09-22T13:35:10.488Z
- Environment: engine=cpu, preferGpu=false, headless=true, runtime=bun/1.4.2, platform=linux-arm64

> Headless CPU numbers are NOT comparable to browser/WebGPU runs.
> Browser/WebGPU reports stay separate (`docs/benchmarks.md` §2, `bun run test:benchmark`).

| Scenario | Docs | Operation | Median (ms) | p95 (ms) | Extra |
|---|---|---|---:|---:|---|
| ide-symbols-5k | 5,000 | index-build | 35.032 | 35.032 | {"buildMs":35.032,"buildTimeMs":33.248981,"uploadMs":0,"vramBytes":0,"ramBytes":4184264} |
| ide-symbols-5k | 5,000 | prefix-search | 12.299 | 15.115 | {"buildMs":35.032,"uploadMs":0,"vramBytes":0,"ramBytes":4184264} |
| ide-symbols-5k | 5,000 | prefix-search+type-filter | 5.99 | 9.892 | {"buildMs":35.032,"uploadMs":0,"vramBytes":0,"ramBytes":4184264} |
| ide-symbols-5k | 5,000 | autocomplete | 14.48 | 16.25 | {"buildMs":35.032,"uploadMs":0,"vramBytes":0,"ramBytes":4184264,"suggestionCount":5} |
| ide-symbols-5k | 5,000 | fuzzy-search+type-facets | 7.502 | 10.716 | {"buildMs":35.032,"uploadMs":0,"vramBytes":0,"ramBytes":4184264} |
| log-grid-10k | 10,000 | index-build | 41.457 | 41.457 | {"buildMs":41.457,"buildTimeMs":41.19228499999997,"uploadMs":0,"vramBytes":0,"ramBytes":3815780} |
| log-grid-10k | 10,000 | fuzzy-search | 4.854 | 7.697 | {"buildMs":41.457,"uploadMs":0,"vramBytes":0,"ramBytes":3815780} |
| log-grid-10k | 10,000 | fuzzy-search+level+latency-filter | 2.537 | 3.499 | {"buildMs":41.457,"uploadMs":0,"vramBytes":0,"ramBytes":3815780,"totalMatches":150} |
| log-grid-10k | 10,000 | fuzzy-search+facets | 4.724 | 6.021 | {"buildMs":41.457,"uploadMs":0,"vramBytes":0,"ramBytes":3815780} |
| log-grid-10k | 10,000 | snapshot-serialize | 40.318 | 49.358 | {"buildMs":41.457,"uploadMs":0,"vramBytes":0,"ramBytes":3815780,"snapshotBytes":6671531,"schemaBytes":140329,"columnarBytes":548521,"docsBytes":2166845} |
| log-grid-10k | 10,000 | snapshot-restore | 48.96 | 66.213 | {"buildMs":41.457,"uploadMs":0,"vramBytes":0,"ramBytes":3815780,"snapshotBytes":6671531,"restoredMatches":150,"restoreTimeMs":14.314923999999792} |
| log-grid-50k | 50,000 | index-build | 368.329 | 368.329 | {"buildMs":368.329,"buildTimeMs":368.1782439999997,"uploadMs":0,"vramBytes":0,"ramBytes":19078864} |
| log-grid-50k | 50,000 | fuzzy-search | 19.24 | 21.693 | {"buildMs":368.329,"uploadMs":0,"vramBytes":0,"ramBytes":19078864} |
| log-grid-50k | 50,000 | fuzzy-search+level+latency-filter | 9.351 | 12.793 | {"buildMs":368.329,"uploadMs":0,"vramBytes":0,"ramBytes":19078864,"totalMatches":750} |
| log-grid-50k | 50,000 | fuzzy-search+facets | 25.014 | 27.154 | {"buildMs":368.329,"uploadMs":0,"vramBytes":0,"ramBytes":19078864} |
| log-grid-100k | 100,000 | index-build | 729.359 | 729.359 | {"buildMs":729.359,"buildTimeMs":729.2449860000002,"uploadMs":0,"vramBytes":0,"ramBytes":38157812} |
| log-grid-100k | 100,000 | fuzzy-search | 35.882 | 43.723 | {"buildMs":729.359,"uploadMs":0,"vramBytes":0,"ramBytes":38157812} |
| log-grid-100k | 100,000 | fuzzy-search+level+latency-filter | 16.461 | 17.223 | {"buildMs":729.359,"uploadMs":0,"vramBytes":0,"ramBytes":38157812,"totalMatches":1500} |
| log-grid-100k | 100,000 | fuzzy-search+facets | 34.188 | 35.185 | {"buildMs":729.359,"uploadMs":0,"vramBytes":0,"ramBytes":38157812} |
