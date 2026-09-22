# Issue #11 — 1.0 Lock Compatibility, Reliability, and the Embedding Ecosystem: Implementation Plan

> **Issue Reference**: [GitHub Issue #11: 1.0: lock compatibility, reliability, and the embedding ecosystem](https://github.com/codewarnab/webgpu-fuzzy-search/issues/11)
> **Target Milestone**: `1.0.0`
> **Status**: Phases 0–4 and 6 complete; Phase 5 open
> **Predecessors**: v0.1.1 (#8), v0.2 (#7), v0.3 (#9), v0.4 (#10)
> **Roadmap**: Issue #12 (embeddable search infrastructure through 1.0)

---

## 0. Executive Summary & Outcome

Ship a stable embeddable-search contract that other web products can depend on across browser, device, worker, and CPU fallback environments.

The package is an embeddable, private, offline-capable search engine for large datasets already inside a web app. It is not a hosted search platform, a UI product, a crawler, or a replacement for server-side search operations.

Success means a host can build or restore an index off the main thread, update it incrementally, query and cancel safely, highlight original text, observe fallback/recovery, and dispose resources without accessing private buffers.

---

## 1. Release Requirements (from Issue #11)

- [x] Freeze and document public TypeScript APIs, result ordering, versioning, and deprecation policy. (Done: `docs/public-api.md`, `scripts/check-public-api.ts` + `scripts/public-api-surface.txt` 187-export allowlist, `check:public-api` + `attw` CI gates.)
- [x] Publish a browser/OS/GPU support matrix and a tested CPU-only baseline. (Done: `docs/support-matrix.md`, `scripts/test-cpu-baseline.ts`, `test:compatibility`.)
- [x] Prove device-loss rebuild, resource cleanup, concurrent-query isolation, and deterministic fallback. (Done: `rebuildGpu()` on both index types, `docs/reliability.md`, `scripts/test-reliability.ts` 58 checks, parity Part 7, contracts §13, wired into `test:compatibility` + `test:all`.)
- [x] Define index-format compatibility and migration support across supported releases. (Done: `docs/snapshot-format.md` §§1–9 migration table + reject taxonomy, `snapshot-codec` schema/header agreement guard, `DocumentIndex` ctor/`fromSnapshotData`/`applySnapshotData` `ProfileMismatchError` guards, `scripts/test-snapshot.ts` §§14–17 hookIds/worker-version/all-caps/profile+live-format pins.)
- [x] Maintain integration examples for web IDE search, local logs/data grids, and offline docs. (Done: `apps/monaco-palette`, `apps/log-viewer`, `apps/docs-search`, `scripts/test-proof-apps.ts` §§1–12 incl. teardown + public-API-only + recipe + DOM-free audits.)
- [ ] Publish reproducible benchmark fixtures, environments, median/p95 results, memory, build/upload, and restore costs.
- [x] Provide package-size budgets, tree-shaking guidance, CSP/worker setup, SSR-safe imports, and accessibility guidance for example UIs. (Done: per-entry gzip budgets — 90 KB index / 76 KB worker — in `scripts/check-bundle-size.ts`, `"sideEffects": false`, `docs/packaging.md`.)
- [x] Document diagnostics, issue-report data, security/privacy boundaries, and unsupported configurations. (Done: `docs/diagnostics.md`, `.github/ISSUE_TEMPLATE/webgpu-search-issue.md`, `docs/security-privacy.md`, unsupported list in `docs/support-matrix.md` §6.)

### Acceptance boundary

- A documented compatibility suite passes on every supported browser path.
- GPU unavailability or loss does not change documented match semantics.
- A persisted index can be restored or rejected with a specific migration reason, never read under the wrong format/profile.
- Package examples consume only public APIs and include teardown and error handling.
- Major semantic or storage changes require a major version.

---

## 2. Current Baseline & Gaps

| Area | Current state | Gap to 1.0 |
| --- | --- | --- |
| Public API | `packages/webgpu-search/src/index.ts:1` exports `SearchIndex`, `DocumentIndex`, `SearchWorkerClient`, engines, packing, ranking, hooks, diagnostics, snapshots | Unfrozen; no `Public` vs `Internal` split; no versioning/deprecation doc |
| Result ordering | `src/ranking.ts` five-tier tie-breakers, `SearchResultItem` normalized integer scores in `src/types.ts:63` | Ordering contract not frozen in a public doc |
| Snapshot format | Canonical v4 + legacy v3 read-only, fail-closed `IncompatibleIndexError`; `docs/snapshot-format.md:1`, `src/text-profile.ts:14` | No cross-release migration table; reject reasons not standardized for hosts |
| Device loss | `GpuDevicePool.onDeviceLost` → CPU fallback + explicit `rebuildGpu()` on `SearchIndex` / `DocumentIndex` (`fallbackReason:'device-lost'`); headless simulation via `GpuDevicePool.simulateDeviceLoss`; proven in `docs/reliability.md` §1 | Done (Phase 2) |
| Lifecycle | Idempotent `destroy()` + `[Symbol.dispose]` on both index types, single device-loss subscription, `WebGPUEngine` unmap-before-destroy + shared-pool release, worker `DESTROY` lifecycle; multi-index leak gate in `scripts/test-reliability.ts` §2 | Done (Phase 2) |
| Proof apps | `apps/monaco-palette`, `apps/log-viewer`, `scripts/test-proof-apps.ts:1`, `workspace:*` linked | Missing third proof app: offline docs / API search |
| Framework recipes | `examples/react`, `examples/vue`, `examples/svelte`, `examples/vanilla` | No public-API-only + teardown lint gate |
| Benchmarks | Headless CPU matrix `scripts/bench-snapshot-matrix.ts:1`, browser runner `scripts/run-all-benchmarks.ts:1` | No frozen fixtures, env table, memory/build/upload/restore costs |
| Packaging | `packages/webgpu-search/tsup.config.ts:1` (esm+cjs, treeshake), `scripts/check-bundle-size.ts:1` (82KB gzip budget) | No published tree-shaking / CSP / worker / SSR / a11y guidance |
| Diagnostics | `src/diagnostics.ts`, `QueryDiagnostics` in `src/types.ts:791`, `FallbackReason` in `src/types.ts:268` | No diagnostics / issue-report / security-privacy docs |

---

## 3. Phases

### Phase 0 — Freeze Public API + Versioning Policy ✅ DONE

Scope: `src/index.ts`, `src/types.ts:12`, `src/errors.ts`, `src/diagnostics.ts`.

Shipped: `docs/public-api.md` (entry points, Stable/Power-user/Internal tiers, class contracts, frozen ordering + echoes, errors, version constants, breaking-change + deprecation rules), `scripts/check-public-api.ts` + `scripts/public-api-surface.txt` (187-export allowlist diff, `INTERNAL_*` guard, version/tie-breaker/echo/alias/exports-map pins) wired as `bun run check:public-api` and CI (`Check Public API Surface` + `attw --pack`).

Tasks (all complete):

1. Audit every export in `src/index.ts:1`; classify `Public` vs `Internal` (`INTERNAL_WORKER_ID_KEY`, packing internals, fold tables, ranking helpers).
2. Add `docs/public-api.md`: exported symbols, `SearchIndex` / `DocumentIndex` / `SearchWorkerClient` contracts, `SearchResultItem` ordering (score DESC + tie-breakers), `SearchResponse` echoes (`profileId`, `scoringVersion`, `cpuScorer`, `fallbackReason`).
3. Document breaking-change rule: major version for any scoring or storage semantic change; minor for additive APIs.
4. Document deprecation policy: removal window, alias handling (`limit` vs `maxResults` as template).
5. Gates: `attw` check, API surface diff check in CI, `bun run typecheck`, `bun run build`.

Exit: hosts can pin `1.x` and rely on frozen ordering + echoes.

### Phase 1 — Support Matrix + CPU Baseline ✅ DONE

Tasks:

1. [x] Add `docs/support-matrix.md`: Chrome / Edge / Safari / Firefox × Windows / macOS / Linux / Android, WebGPU vs CPU-only, Worker / Node.js / SSR. (Shipped; §5 maps each path to its named suite.)
2. [x] Define tested CPU-only baseline: `preferGpu:false` + `cpuScorer:'exact'` as the reference contract. (Pinned by `scripts/test-cpu-baseline.ts`, 39 checks, wired as `bun run test:cpu-baseline` and into `test:all` + CI.)
3. [x] Mark `cpuScorer:'ufuzzy'` as explicit opt-in, non-conforming, CPU-only. (`docs/support-matrix.md` §2; enforced fail-closed in `search-index.ts` / `document-index.ts`; pin covers echo `cpu-algorithm-requested` + all conflict throws.)
4. [x] Wire documented compatibility suite: `test:mock`, `test:contracts`, `test:parity`, `test:search-modes`, `test:browser` across supported paths. (Plus new `test:cpu-baseline`; one-command `bun run test:compatibility` for the headless set; `test:browser` stays the separate hardware gate; CI runs each suite incl. the pin as a named step.)

Exit: every supported path has a named passing suite; CPU baseline is reproducible without a GPU.

### Phase 2 — Reliability Proofs ✅ DONE

Scope: `src/document-index.ts:184`, `src/search-index.ts:72`, `src/gpu-device-pool.ts:31`, `src/webgpu-engine.ts:164`, `src/worker/worker-client.ts`.

Tasks:

1. [x] Device-loss rebuild: explicit `rebuildGpu()` on `SearchIndex` / `DocumentIndex` (re-acquire + re-upload retained corpora, identical-match-semantics; `false` for CPU-by-design/empty, throws only when destroyed) alongside existing `restore` / re-`init` paths. Headless simulation via `GpuDevicePool.simulateDeviceLoss()`. (Pinned by `scripts/test-reliability.ts` §1, 58 checks; worker indexes rebuild via `restore()` / re-`init()`; documented in `docs/reliability.md` §1.)
2. [x] Resource cleanup: audited idempotent `destroy()`, buffer disposal (unmap-before-destroy), `GpuDevicePool` release (injected no-op, shared refcount clamp), worker `DESTROY`; added 8+8 multi-index create/destroy leak gate (`getListenerCount()` returns to baseline) + `[Symbol.dispose]` on both index types. (Pinned by `scripts/test-reliability.ts` §2 + `test:contracts` §13; `docs/reliability.md` §2.)
3. [x] Concurrent-query isolation: direct paths proven independent (per-query echo + exact-oracle match, `AbortSignal` siblings isolated, mutation `generation` abort); `SearchWorkerClient` monotonic sequencing (rapid typing aborts superseded, last wins) + caller `signal`. (Pinned by `scripts/test-reliability.ts` §3 alongside existing `test:worker` §5 and `test:parity` Part 5; `docs/reliability.md` §3.)
4. [x] Deterministic fallback: GPU-unavailable and GPU-lost queries match the documented CPU contract (same matches/order/echoes; `prefer-cpu` / `unsupported-mode` / `device-lost` / `gpu-execution-error` reasons; `AbortError` never converts to fallback). Extended `scripts/test-parity-harness.ts` (Part 7) and `scripts/test-contracts.ts` (§13). (Pinned by `scripts/test-reliability.ts` §4; `docs/reliability.md` §4.)
5. [x] Gates: `bun run test:mock`, `bun run test:worker`, `bun run test:browser` all green; plus new `bun run test:reliability` wired into `test:compatibility` + `test:all` and `docs/support-matrix.md` §5.

Exit: loss, cleanup, and concurrency are proven, not just handled.

### Phase 3 — Index-Format Compatibility + Migration ✅ DONE

Scope: `src/snapshot-codec.ts`, `src/snapshot-idb.ts`, `src/persistence.ts`, `src/idb-storage.ts`, `docs/snapshot-format.md:1`.

Shipped: `docs/snapshot-format.md` §§1–9 (v3 → v4 → 1.0 table, reject taxonomy, `hasGetter`/`hookIds` rules, profile guarantee, live-format `4`, IDB caps, suites), codec agreement guard, `DocumentIndex` `ProfileMismatchError` guards, `test-snapshot` §§14–17. Gates: `test:snapshot` §§1–17, `test:records`, `bench:snapshot-matrix` all green; plus `check:shaders`, `typecheck`, `build`, `test:mock`, `check:public-api` (187 pinned).

Tasks (all complete):

1. [x] Extend `docs/snapshot-format.md` with a v3 → v4 → 1.0 compatibility + migration table.
2. [x] Standardize reject reasons: `IncompatibleIndexError` (version/profile/checksum/shape), `IncompatibleHookError` (missing hooks), `ProfileMismatchError` (case/profile).
3. [x] Cover in tests: legacy restore, corrupt CRC, oversize caps (`MAX_SNAPSHOT_*`), missing `getter` (`hasGetter`), missing `hookIds`, worker `serialize` / `restore` version guard.
4. [x] Guarantee: never read under wrong format/profile; `getStats().formatVersion` semantics stay live-format.
5. [x] Gates: `bun run test:snapshot`, `bun run test:records`, `bun run bench:snapshot-matrix`.

Exit: any persisted index restores or rejects with an actionable migration reason.

### Phase 4 — Integration Examples (Public-API Only) ✅ DONE

Shipped: `apps/docs-search` (offline documentation / API search: `DocsEngine`, versioned `generateDocsRecords` corpus, section/version filters + facets, IDB snapshot bundle, `rebuildGpu` recovery hook, `pagehide` teardown), `pagehide` teardown on all three proof-app UIs, `test-proof-apps` §§8–12 (docs-engine functional incl. cancel/recovery/snapshot/XSS pins, teardown lint, public-API-only import lint, framework-recipe audit, DOM-free engine/data audit). Gates: `test:proof-apps` §§1–12, `typecheck` (7 tasks), `build`, `test:mock` all green.

Tasks (all complete):

1. [x] Keep `apps/monaco-palette` — Monaco-style file / symbol / command search.
2. [x] Keep `apps/log-viewer` — large local log viewer / data-grid quick find.
3. [x] Add `apps/docs-search` — offline documentation / API search (the missing third proof app).
4. [x] Enforce per app: public-API-only imports from `webgpu-search`, off-main-thread build/restore via worker, incremental mutations, query cancel, original-text highlights, fallback/recovery observability, `destroy()` + error paths.
5. [x] Extend `scripts/test-proof-apps.ts` to cover all three + teardown lint + cross-platform safety audit (zero unguarded DOM globals).
6. [x] Check framework recipes (`examples/react`, `examples/vue`, `examples/svelte`, `examples/vanilla`) for worker-first + abort-safe + destroy-on-unmount patterns.

Exit: three proof apps pass using only public APIs.

### Phase 5 — Reproducible Benchmarks

Scope: `scripts/bench-snapshot-matrix.ts`, `scripts/run-all-benchmarks.ts`, `scripts/run-fuzzy-benchmark.ts`, `apps/benchmark`.

Tasks:

1. Freeze fixtures: version `generateMonacoRecords` and `generateStructuredLogs` corpora; record sizes, queries, modes, filters, facets.
2. Publish per-scenario table: environment (browser / GPU / driver / runtime), median/p95 with samples, memory (`vramBytes` / `ramBytes`), build/upload, serialize/restore bytes + time.
3. Keep headless-CPU and browser-WebGPU reports separate; label headless numbers non-comparable to GPU runs.
4. Check in `benchmark_snapshot_matrix.json` baselines + markdown summary.

Exit: performance claims trace to named fixtures and environments.

### Phase 6 — Packaging + Maintenance Docs ✅ DONE

Scope: `packages/webgpu-search/package.json`, `packages/webgpu-search/tsup.config.ts`, `scripts/check-bundle-size.ts`.

Shipped: per-entry gzip budgets (90 KB `index.js`/`index.cjs`, 76 KB `worker.js`/`worker.cjs`, 40 KB index delta cap + rationale log in `check-bundle-size.ts`), `"sideEffects": false`, `docs/packaging.md` (tree-shaking, CSP, worker setup, SSR-safe imports, example-UI a11y), `docs/diagnostics.md` (telemetry, warnings, budgets, issue-report data), `.github/ISSUE_TEMPLATE/webgpu-search-issue.md`, `docs/security-privacy.md` (IDB PII scope, snapshot trust boundary, CRC vs authenticity). Gates: `check:shaders`, `typecheck`, `build`, `test:mock`, `check:bundle-size`, `check:public-api` all green.

Tasks (all complete):

1. Keep per-file gzip budget gate; add per-entry (`index` vs `worker`) budget + rationale log.
2. Publish tree-shaking guidance: entry choice, side-effects, what pulls in WGSL / uFuzzy.
3. Publish setup guides: CSP (worker blob, no-WASM notes), worker setup (`SearchWorkerClient`, `startSearchWorker`), SSR-safe imports (no unguarded `window` / `document`).
4. Publish example-UI accessibility guidance: keyboard, focus, live regions, reduced motion.
5. Add `docs/diagnostics.md` (telemetry fields, warnings, budgets), issue-report template (snapshot header, `getStats()`, timings, `fallbackReason`), `docs/security-privacy.md` (IDB PII scope, snapshot trust boundary, CRC vs authenticity), unsupported configurations list.
6. Gates: `bun run check:shaders`, `bun run typecheck`, `bun run build`, `bun run test:mock`.

Exit: hosts can install, bundle, sandbox, debug, and report issues without reading source.

---

## 4. Suggested Execution Order

```text
0 API freeze → 3 format compat → 2 reliability → 1 matrix → 4 examples → 5 benchmarks → 6 packaging
```

Rationale: correctness and storage contracts (0, 3, 2) before ecosystem proof (1, 4, 5, 6), mirroring roadmap sequencing rules in Issue #12.

---

## 5. Verification Per Phase

- `bun run check:shaders`
- `bun run typecheck`
- `bun run build`
- `bun run test:mock`
- Phase-specific: `test:contracts`, `test:parity`, `test:snapshot`, `test:worker`, `test:proof-apps`, `test:browser`, `bench:snapshot-matrix`, `check:bundle-size`, `test:cpu-baseline`, `test:compatibility`
- Subagent verification before sign-off per `AGENTS.md`: contract/regression check, portability audit (no unguarded DOM), clean-diff audit.
