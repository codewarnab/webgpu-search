# Support Matrix + CPU Baseline

> Plan ref: `docs/ISSUE-11-PLAN.md` Phase 1.
> Scope: which browsers, operating systems, and runtimes are supported,
> what the tested CPU-only baseline is, where `cpuScorer: 'ufuzzy'`
> fits, and which named suite proves each path.

---

## 1. CPU-only baseline (reference contract)

The reference contract is reproducible **without any GPU** and is the
behavior every other path is compared against.

```ts
import { SearchIndex, DocumentIndex } from 'webgpu-search';

// Reference baseline: explicit CPU, exact scorer.
const index = await SearchIndex.create(records, { preferGpu: false });
const res = await index.search('auth', {
  mode: 'substring',
  cpuScorer: 'exact', // default; may be omitted
});
```

Contract pins:

| Item | Value |
| --- | --- |
| Index options | `preferGpu: false` (explicit CPU even on large corpora) |
| Scorer | `cpuScorer: 'exact'` (default; shared `exact-scorer.ts` pipeline) |
| `engine` | `'cpu'` |
| `fallbackReason` (flat `SearchIndex`) | `'prefer-cpu'` for `fuzzy` / `substring`; `'unsupported-mode'` for `token` / `prefix` / typo-tolerant queries (exact-only WGSL kernels are CPU-by-design) |
| `fallbackReason` (`DocumentIndex`) | same rule (`validateSearchOptions` + `routeEngine`, `document-index.ts`) |
| Echoes | `profileId: 'unicode-default'`, `scoringVersion: 'parity-v1'`, `cpuScorer: 'exact'` |
| Ordering | descending normalized integer scores; deterministic tie-breakers (`ranking.ts`); exact order iff `hasOverflow === false`, else `totalMatches` + `hasOverflow` + score multiset |
| Limits | `limit` default 50, clamped `1..8192` (`RESULT_LIMIT_MAX`); `maxResults` is a backwards-compatible alias (`limit` wins) |
| Query cap | 128 post-normalization tokens (`QUERY_TOKENS_MAX`); over-limit throws `QueryTooLongError` unless `onQueryTooLong: 'cpu-fallback'` |
| Reproducibility | deterministic: identical corpus + query + mode + options produce identical `index` / `score` order; integer scores only |

`preferGpu: false` forces CPU even above `threshold` (default 30,000);
callers must not rely on undocumented `threshold: Infinity` to stay on CPU.
Omitting `cpuScorer` is identical to `cpuScorer: 'exact'` — the pin suite
asserts default-equals-explicit.

Normative text pipeline and version caps live in
`docs/text-normalization.md`; snapshot wire compat lives in
`docs/snapshot-format.md`.

---

## 2. `cpuScorer: 'ufuzzy'` — explicit opt-in, non-conforming, CPU-only

```ts
// Opt-in only. CPU-only. Scores are explicitly non-conforming.
const res = await index.search('auth', {
  mode: 'fuzzy', // or 'substring'
  cpuScorer: 'ufuzzy',
});
```

Rules (enforced fail-closed in `search-index.ts` / `document-index.ts`):

- `cpuScorer: 'ufuzzy'` **never routes to WebGPU** and is **excluded from
  the differential matrix** (`scripts/test-parity-harness.ts`,
  `packages/webgpu-search/src/cpu-engine.ts` quarantine note).
- `preferGpu: true + cpuScorer: 'ufuzzy'` throws `IncompatibleOptionError`
  at `search()` — use `preferGpu: false` or `cpuScorer: 'exact'`.
- `cpuScorer: 'ufuzzy'` supports only exact `fuzzy` / `substring` without
  typo tolerance. `token` / `prefix` modes or `typoTolerance.enabled`
  throw `IncompatibleOptionError` (`Use cpuScorer:'exact'`).
- Successful ufuzzy queries return `engine: 'cpu'`,
  `cpuScorer: 'ufuzzy'`, `fallbackReason: 'cpu-algorithm-requested'`.
- uFuzzy scores are fabricated rank scores (`1000 - rank * 2` in
  `CPUEngine.searchWithUFuzzy`); substring-mode ufuzzy requests are served
  by the native scan, not uFuzzy ranking. Do not compare ufuzzy numbers
  against `exact` / GPU numbers.
- Unknown `cpuScorer` values throw `IncompatibleOptionError`
  (`normalizeCpuScorer`, `text-profile.ts`). Removed aliases
  (`cpuAlgorithm` / `'parity'`) throw with a migration message.

Hosts that need comparable, portable scores must use the baseline
(`cpuScorer: 'exact'`).

---

## 3. Browser / OS matrix

`GpuDevicePool.isSupported()` (`gpu-device-pool.ts`) reports WebGPU
availability as `typeof navigator !== 'undefined' && !!navigator.gpu`.
All library imports are SSR-safe (guarded `navigator` / `document` access);
absence of WebGPU always falls back to the Section 1 CPU contract with a
documented `fallbackReason` (`webgpu-unsupported`, `device-request-failed`,
`below-threshold`, `prefer-cpu`, `device-lost`, `gpu-execution-error`,
`unsupported-mode`, `cpu-algorithm-requested`, `query-too-long`,
`memory-budget-exceeded`).

| Browser | Windows | macOS | Linux | Android / ChromeOS | WebGPU path | CPU fallback |
| --- | --- | --- | --- | --- | --- | --- |
| Chrome (stable) | Supported (D3D12) | Supported (Metal) | Supported (Vulkan; headless via SwiftShader, see below) | Supported (Android); ChromeOS supported | `preferGpu: true` / large corpora route to WebGPU; device via `GpuDevicePool.acquireDevice` or injected `options.device` | Section 1 baseline, always available |
| Edge (Chromium) | Supported (same engine as Chrome) | Supported | n/a (no Edge Linux WebGPU target) | Supported (Android) | Same as Chrome | Section 1 baseline |
| Safari 18+ | n/a | Supported (Metal) | n/a | iOS 18+ supported | WebGPU where `navigator.gpu` is present; older Safari reports `webgpu-unsupported` and serves CPU | Section 1 baseline |
| Firefox (stable) | CPU-only baseline | CPU-only baseline | CPU-only baseline | CPU-only baseline | Not a supported WebGPU path (Nightly / flag-only builds are untested and treated as `webgpu-unsupported`) | Section 1 baseline (the supported Firefox path) |
| Headless Chromium (CI, SwiftShader / LLVMpipe) | Covered by `test:browser` gate | Covered by `test:browser` gate | Covered by `test:browser` gate (primary CI path) | n/a | API-present but software-rendered: numbers are **`pending-hardware`**, never compared against physical-GPU runs | Section 1 baseline, bit-comparable headless |

Notes:

- Device acquisition must go through `GpuDevicePool` or an injected
  `options.device`. Never call `navigator.gpu.requestDevice()` per index;
  the pool multiplexes one shared device and `powerPreference`
  (`'high-performance' | 'low-power'`, validated fail-closed even on
  CPU-only paths).
- `powerPreference` unknown values throw `IncompatibleOptionError` on every
  path, including CPU-only and injected-device paths.
- Oversize datasets fail closed to CPU (`memory-budget-exceeded`); there is
  no GPU paging. Per-buffer budgets are checked against
  `device.limits` (`records` / `offsets` / `query` / `output`).
- Device loss falls back to CPU (`fallbackReason: 'device-lost'`);
  explicit rebuild is proven in `docs/reliability.md` (`rebuildGpu()` on both
  index types; worker indexes rebuild via `restore()` / re-`init()`).

---

## 4. Runtime matrix (thread / server)

| Runtime | Status | Notes |
| --- | --- | --- |
| Browser main thread | Supported | Direct `SearchIndex` / `DocumentIndex`; prefer a worker for large corpora to avoid frame drops |
| Dedicated Worker (`SearchWorkerClient`, `startSearchWorker`) | Supported | Off-main-thread build / restore / search / mutate; `stringIsolated: true` (default) returns compact `{ index, score }` hits enriched on the main thread; function hooks cannot cross the boundary (`hooks` reject with `IncompatibleHookError`; empty `{}` is a no-op) |
| Node.js >= 18 / Bun >= 1.1 | Supported, CPU-only | No `navigator.gpu`; baseline (Section 1) is the path. Used by all headless suites |
| SSR (import-time) | Supported | No unguarded `window` / `document` at import; `GpuDevicePool.isSupported()` guards acquisition; `isDedicatedWorker === false` outside workers (`test:contracts` pins this) |
| IndexedDB persistence | Supported where IndexedDB exists | Raw snapshot bytes; PII scope and CRC-vs-authenticity notes in `docs/snapshot-format.md` |

Engines field (`package.json`): `node >= 18`, `bun >= 1.1`.

---

## 5. Compatibility suite (named suites per path)

Headless suites run anywhere Bun/Node runs (no GPU, no browser).
The browser suite is the executing-hardware release gate.

| Suite | Command | Proves | Paths covered |
| --- | --- | --- | --- |
| Mock device | `bun run test:mock` (`scripts/test-vgpu-mock.ts`) | WGSL buffer layout (32 B uniform, 512 B query, 65,544 B output), packing / serialization round-trip + fail-closed rejects, CPU auto-routing, overflow at 8191/8192/8193, abort / destroy / re-init, ufuzzy conflict | All OS headless; mock (non-executing) GPU + CPU baseline |
| Public contracts | `bun run test:contracts` (`scripts/test-contracts.ts`) | Snapshot constants, error hierarchy + worker serialization round-trip, `DocumentIndex` / `SearchWorkerClient` contracts, SSR worker guard, `package.json` exports, filter/facet/typo/ranking/autocomplete/hooks/diagnostics shapes | Node/Bun/SSR; worker-client protocol |
| Differential parity | `bun run test:parity` (`scripts/test-parity-harness.ts`) | Matrix corrections (a)-(n), CPU-wiring differential cells, echo contracts, failure-injection delta (compile/pipeline/OOM/`mapAsync`/device-loss), concurrency/abort, worker contract; exact GPU-order cells report `pending-hardware` on `vgpu/mock` (mock never executes WGSL) | Headless differential; browser subset is the release gate |
| Search modes | `bun run test:search-modes` (`packages/webgpu-search/test/search-modes.test.ts`) | Bounded typo tolerance, token AND/OR/quorum + proximity, prefix anchor + `prefixLength` / `exactCase` gates, 4-mode CPU wiring, `DocumentIndex` integration (filters/facets/highlights), GPU routing (token/prefix/typo rejected to CPU, WGSL exact-only), determinism, portability (no unguarded DOM) | CPU reference on all runtimes; GPU routing via mock device |
| CPU baseline pin | `bun run test:cpu-baseline` (`scripts/test-cpu-baseline.ts`) | Section 1 reproducibility without a GPU (default-equals-explicit `exact`, integer descending scores, 4-mode coverage, `prefer-cpu` vs `unsupported-mode` reasons, ufuzzy opt-in + conflicts, `DocumentIndex` echoes, SSR guard) | Node/Bun/SSR, no GPU required |
| Reliability proofs | `bun run test:reliability` (`scripts/test-reliability.ts`) | Device-loss rebuild (`rebuildGpu` identical semantics), multi-index create/destroy leak gate + worker `DESTROY`, direct + worker concurrent-query isolation under rapid typing, GPU-unavailable/lost deterministic fallback vs the CPU contract; see `docs/reliability.md` | Node/Bun/SSR headless (mock device); browser subset stays the hardware gate |
| Browser regression (hardware gate) | `bun run test:browser` (`scripts/test-regression.ts`) | Timings separation, broad-query top-K, overflow detection, unicode packing, **ordered `(index, score, text)` CPU/GPU parity on executing hardware** (Bun oracle vs Chrome subject, same-host) | Chrome/Chromium + benchmark dev server (`apps/benchmark`); requires a Chrome binary (`CHROME_BIN`) |
| Aggregator | `bun run test:compatibility` | Runs the headless compatibility set in one command: `test:mock` + `test:contracts` + `test:parity` + `test:search-modes` + `test:cpu-baseline` + `test:reliability`. The browser gate stays separate (`test:browser`) because it needs Chrome + WebGPU | CI + local |

Run:

```bash
# Headless compatibility set (no GPU, no browser needed)
bun run test:compatibility

# Executing-hardware release gate (needs Chrome; starts benchmark server)
bun run test:browser

# Full repo gate (includes compatibility suites plus records/highlight/
# mutations/worker/snapshot/observability/proof-apps/normalization/
# filtering/faceting/diagnostics)
bun run test:all
```

CI (`.github/workflows/ci.yml`) runs each compatibility suite as a named
step — including the CPU baseline pin (`test:cpu-baseline`) — plus the
browser regression gate on stable Chrome.

---

## 6. Unsupported configurations

- Firefox stable as a WebGPU path (use the CPU baseline).
- Safari < 18 as a WebGPU path (use the CPU baseline).
- Node.js / Bun / SSR as WebGPU paths (CPU-only by design).
- `cpuScorer: 'ufuzzy'` as a conforming / comparable scorer (opt-in only).
- Cross-scorer score comparisons (`exact` vs `ufuzzy` vs GPU pool on
  overflow): facet `isApproximate` is engine-relative, not cross-engine.
- Server-side GPU execution claims from headless SwiftShader numbers
  (always labeled `pending-hardware`).
- Function `filter` predicates + `facets` over `SearchWorkerClient`
  (facets drop fail-closed; closures cannot cross the worker boundary).

---

## 7. What this phase does not claim

Migration tables, the third proof app, and packaging guides are earlier
phases in `docs/ISSUE-11-PLAN.md` and are not claimed by
this matrix. Frozen benchmark fixtures remain Phase 5. Reliability proofs (rebuild, leak/concurrency gates) live in
`docs/reliability.md`.
