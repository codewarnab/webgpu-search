# Reproducible Benchmarks (Phase 5)

> Plan ref: `docs/ISSUE-11-PLAN.md` Phase 5.
> Fixtures: `scripts/benchmark-fixtures.ts` (`BENCHMARK_FIXTURE_VERSION = 1.0.0`).
> Runner: `scripts/bench-snapshot-matrix.ts` (`bun run bench:snapshot-matrix`).
> Pin: `scripts/test-benchmarks.ts` (`bun run test:benchmarks`).
> Baseline: `benchmark_snapshot_matrix.json` + `benchmark_snapshot_matrix.md` (repo root).

Performance claims trace to named fixtures and environments. There are two
report families — headless-CPU and browser-WebGPU — and they are never
compared against each other.

---

## 1. Frozen fixtures (`1.0.0`)

Generators (deterministic; no `Math.random`):

| Corpus | Generator | Sizes | Anchor |
| --- | --- | --- | --- |
| IDE symbols (monaco-palette shape) | `generateMonacoRecords` (`apps/monaco-palette/src/sample-data.ts`) | 5,000 | fully deterministic (`f-001` …) |
| Data-grid rows (structured logs) | `generateStructuredLogs` (`apps/log-viewer/src/log-generator.ts`) | 10,000 / 50,000 / 100,000 | IDs, level/service/message/latency deterministic; timestamps anchored at `BENCHMARK_LOG_BASE_TIME_MS` (`2026-01-01T00:00:00.000Z`) so snapshot bytes reproduce run-to-run |

Index shapes:

| Scenario | `idField` | Fields (weights) | Filter fields |
| --- | --- | --- | --- |
| `ide-symbols-5k` | `id` | `filename` 3.0, `symbols` 2.0, `path` 1.0, `description` 0.5 | `type`, `language` |
| `log-grid-10k` / `50k` / `100k` | `id` | `message` 2.0, `service` 1.5, `level` 1.0, `traceId` 1.2 | `level`, `service`, `timestamp`, `latencyMs` (`number`) |

Frozen queries / modes / filters / facets (`BENCHMARK_QUERIES`):

| Operation | Query | Mode | Limit | Filter / facets |
| --- | --- | --- | --- | --- |
| `prefix-search` | `compute` | `prefix` | 20 | — |
| `prefix-search+type-filter` | `compute` | `prefix` | 20 | `filter: { type: 'shader' }` |
| `autocomplete` | `comp` | `prefix` | 5 | inline `autocomplete: { mode: 'prefix', limit: 5 }` |
| `fuzzy-search+type-facets` | `service` | `fuzzy` | 20 | `facets: { byType: terms(type, 10) }` |
| `fuzzy-search` | `timeout` | `fuzzy` | 50 | — |
| `fuzzy-search+level+latency-filter` | `timeout` | `fuzzy` | 50 | `filter: { level: 'ERROR', latencyMs: { gte: 300 } }` |
| `fuzzy-search+facets` | `timeout` | `fuzzy` | 50 | `byLevel: terms(level, 10)` + `byLatency: range(latencyMs, fast/normal/slow)` |
| `snapshot-serialize` / `snapshot-restore` | `timeout` probe (`fuzzy`, 5) | `fuzzy` | 5 | 10k grid only; bytes + wall-clock |

Statistics: 3 warmups + 10 samples per operation (`median` + `p95` with linear
interpolation, samples checked in). `index-build` rows are single-shot
(`median == p95 == buildMs`, `samples: [buildMs]`) alongside
`getStats().buildTimeMs`.

---

## 2. Environments (kept separate)

| Report | Command | Engine | Runtime / browser / GPU | Memory | Comparability |
| --- | --- | --- | --- | --- | --- |
| Headless CPU baseline (this doc, §§3–4) | `bun run bench:snapshot-matrix -- --out benchmark_snapshot_matrix.json` | CPU (`preferGpu: false`, exact scorer) | Bun/Node version + `platform` recorded in JSON `environment` | `vramBytes: 0` (no GPU upload), `ramBytes` from `getStats()`, `uploadMs: 0` | Headless CPU numbers are **NOT comparable to browser/WebGPU runs** |
| Browser WebGPU matrix | `bun run test:benchmark` / `test:benchmark:fuzzy` (`scripts/run-all-benchmarks.ts`, `apps/benchmark`) | WebGPU retained buffers vs CPU parity vs uFuzzy vs JS native | Chrome version, adapter (`vendor`/`device`/`architecture`), driver, OS recorded per row; software adapters report `pending-hardware` | Per-buffer VRAM (`records`/`offsets`/`query` 512 B/`output` 65,544 B) + packing pipeline (`normalizeMs`/`packMs`/`uploadMs`) | Browser numbers compare only against same-hardware runs; `pending-hardware` cells never qualify |

Browser reports (`benchmark_results.json` / `benchmark_summary.md` from the
browser runner) are versioned separately from the headless baseline and are
not checked in as the 1.0 frozen baseline.

---

## 3. Per-scenario baseline (headless CPU, fixtures `1.0.0`)

Generated `2026-09-22T13:35:10.488Z`, `runtime=bun/1.4.2`,
`platform=linux-arm64`, `engine=cpu`, `preferGpu=false`, `headless=true`.
Full samples + `environment` + `fixtures` in `benchmark_snapshot_matrix.json`;
same table in `benchmark_snapshot_matrix.md`.

| Scenario | Docs | Operation | Median (ms) | p95 (ms) | Costs |
|---|---|---|---:|---:|---|
| ide-symbols-5k | 5,000 | index-build | 35.032 | 35.032 | `buildTimeMs` 33.249, `ramBytes` 4184264, `vramBytes` 0, `uploadMs` 0 |
| ide-symbols-5k | 5,000 | prefix-search | 12.299 | 15.115 | same build/memory |
| ide-symbols-5k | 5,000 | prefix-search+type-filter | 5.99 | 9.892 | same build/memory |
| ide-symbols-5k | 5,000 | autocomplete | 14.48 | 16.25 | `suggestionCount` 5 |
| ide-symbols-5k | 5,000 | fuzzy-search+type-facets | 7.502 | 10.716 | same build/memory |
| log-grid-10k | 10,000 | index-build | 41.457 | 41.457 | `buildTimeMs` 41.192, `ramBytes` 3815780, `vramBytes` 0, `uploadMs` 0 |
| log-grid-10k | 10,000 | fuzzy-search | 4.854 | 7.697 | same build/memory |
| log-grid-10k | 10,000 | fuzzy-search+level+latency-filter | 2.537 | 3.499 | `totalMatches` 150 |
| log-grid-10k | 10,000 | fuzzy-search+facets | 4.724 | 6.021 | same build/memory |
| log-grid-10k | 10,000 | snapshot-serialize | 40.318 | 49.358 | `snapshotBytes` 6671531, `schemaBytes` 140329, `columnarBytes` 548521, `docsBytes` 2166845 |
| log-grid-10k | 10,000 | snapshot-restore | 48.96 | 66.213 | `snapshotBytes` 6671531, `restoredMatches` 150, `restoreTimeMs` 14.315 |
| log-grid-50k | 50,000 | index-build | 368.329 | 368.329 | `buildTimeMs` 368.178, `ramBytes` 19078864, `vramBytes` 0, `uploadMs` 0 |
| log-grid-50k | 50,000 | fuzzy-search | 19.24 | 21.693 | same build/memory |
| log-grid-50k | 50,000 | fuzzy-search+level+latency-filter | 9.351 | 12.793 | `totalMatches` 750 |
| log-grid-50k | 50,000 | fuzzy-search+facets | 25.014 | 27.154 | same build/memory |
| log-grid-100k | 100,000 | index-build | 729.359 | 729.359 | `buildTimeMs` 729.245, `ramBytes` 38157812, `vramBytes` 0, `uploadMs` 0 |
| log-grid-100k | 100,000 | fuzzy-search | 35.882 | 43.723 | same build/memory |
| log-grid-100k | 100,000 | fuzzy-search+level+latency-filter | 16.461 | 17.223 | `totalMatches` 1500 |
| log-grid-100k | 100,000 | fuzzy-search+facets | 34.188 | 35.185 | same build/memory |

Memory semantics: `ramBytes = tokenRamBytes + offsetRamBytes`
(`tokens × 4 + (rows + 1) × 4`) from `getStats().memory`; headless CPU always
reports `vramBytes: 0`, `uploadMs: 0` (nothing uploaded to a GPU). Browser
WebGPU reports carry real `vramBytes` per §2 and are not mixed into this table.

---

## 4. Reproduce

```bash
# Full headless matrix (several minutes: 100k build + 10 samples/op).
# Writes benchmark_snapshot_matrix.json + benchmark_snapshot_matrix.md.
bun run bench:snapshot-matrix -- --out benchmark_snapshot_matrix.json

# CI shape (throwaway copy, does not touch the checked-in baseline):
bun scripts/bench-snapshot-matrix.ts --out /tmp/benchmark_snapshot_matrix.json

# Fast pin (no 100k build; validates fixtures + baseline + this doc):
bun run test:benchmarks

# Browser WebGPU matrix (needs Chrome; separate, non-comparable report):
bun run test:benchmark
```

Re-baseline rule: changing any frozen generator, size, query, mode, filter,
facet, field weight, or statistic requires bumping
`BENCHMARK_FIXTURE_VERSION`, regenerating both checked-in artifacts, updating
the §3 table, and noting the break in the release notes. The pin
(`test:benchmarks` §4) fails on version drift or missing (scenario, operation)
pairs.

---

## 5. What this phase does not claim

- Browser/WebGPU absolute numbers for host hardware (run §4 browser matrix on
  the target device; `pending-hardware` cells from SwiftShader/LLVMpipe never
  qualify).
- Cross-engine score comparisons (`exact` vs `ufuzzy` vs GPU pool on
  overflow; see `docs/support-matrix.md` §6).
- Snapshot byte-exactness across package versions (wire compat lives in
  `docs/snapshot-format.md`; bytes reproduce only within the same fixtures +
  package version).
