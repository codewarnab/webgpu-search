# Reliability Proofs (Phase 2)

> Plan ref: `docs/ISSUE-11-PLAN.md` Phase 2.
> Proves device-loss rebuild, resource cleanup, concurrent-query isolation,
> and deterministic fallback — not just handling.

---

## 1. Device-loss rebuild

GPU buffers are ephemeral; CPU corpora are retained across loss.

- Loss transitions to CPU with `fallbackReason: 'device-lost'`
  (`GpuDevicePool.onDeviceLost` fan-out; `WebGPUEngine` lost-handler nulls
  handles and bumps the generation so in-flight `mapAsync` continuations
  discard as `AbortError`).
- The explicit rebuild path re-acquires a device and re-uploads the retained
  corpus with identical match semantics:

```ts
import { SearchIndex, DocumentIndex } from 'webgpu-search';

// After fallbackReason === 'device-lost':
const ok: boolean = await index.rebuildGpu(); // or rebuildGpu({ device })
```

Rules:

| Case | Result |
| --- | --- |
| Already `webgpu` + ready | `true` (no-op, no listener leak) |
| `preferGpu: false` (CPU by design) | `false` (no-op, reason unchanged) |
| Empty corpus | `false` (nothing to upload) |
| GPU re-acquire succeeds | `true`, engine back to `'webgpu'`, single loss listener |
| GPU unavailable | `false`, stays CPU with `webgpu-unsupported` / `device-request-failed` |
| Index destroyed | throws fail-closed |

- `SearchIndex.rebuildGpu({ device?, powerPreference? })` reuses the
  creation-time device when no override is given; an override is persisted
  for subsequent rebuilds.
- `DocumentIndex.rebuildGpu({ device?, powerPreference? })` persists an
  override into `options.device` / `options.powerPreference` and reuses
  `tryInitializeGpuEngine` (same packing + headroom as initial build).
- Rebuild re-subscribes exactly one device-loss listener (cycles never leak;
  pinned by `GpuDevicePool.getListenerCount()`).
- Headless simulation: `GpuDevicePool.simulateDeviceLoss(reason)` clears the
  shared device and fans out to subscribers — the same path as the real
  `device.lost.then` handler — so rebuild semantics prove out on `vgpu/mock`
  without executing hardware. The mock never executes WGSL, so GPU-hit
  equality is `pending-hardware`; CPU-fallback equality and corpus retention
  (mutation-after-rebuild) are hard-gated.
- Worker indexes rebuild via `restore()` / re-`init()` (the worker owns its
  `DocumentIndex`; `DESTROY` tears it down).

Proven by: `bun run test:reliability` §1 + `test:parity` Part 7 +
`test:contracts` §13.

---

## 2. Resource cleanup

- `SearchIndex.destroy()` / `DocumentIndex.destroy()` are idempotent:
  unsubscribe exactly once, destroy the GPU engine (which unmaps before
  destroy and releases the shared-pool ref once via the `deviceReleased`
  guard), clear CPU residency so post-destroy `search()` throws fail-closed
  and `getStats()` reads `0`/empty, and park a barrier on the mutex so
  queued ops observe teardown.
- Both index types implement `[Symbol.dispose]` (symmetric teardown).
- `GpuDevicePool.releaseDevice(device, isShared)` is a no-op for injected
  (non-shared) devices; shared refcount clamps at `0`.
- `SearchWorkerClient.destroy()` rejects all pending queries/requests with
  `AbortError`, posts `DESTROY` (worker destroys its index + aborts the
  active controller), removes message/error listeners, terminates owned
  workers, and clears the doc map. Double-destroy is safe.
- Leak gate: `N` GPU indexes hold exactly `N` (flat) + `N` (document)
  listeners; after `destroy()` the count returns to baseline and shared
  refcount is untouched for injected devices.

Proven by: `bun run test:reliability` §2.

Introspection (portable, no DOM — safe in workers/Node/SSR):

- `GpuDevicePool.getListenerCount()`
- `GpuDevicePool.getRefCount()`
- `GpuDevicePool.hasSharedDevice()`

---

## 3. Concurrent-query isolation

| Path | Semantics |
| --- | --- |
| Direct `SearchIndex` / `DocumentIndex` | Concurrent queries are independent — none auto-aborts. Each response echoes its own `query`; results match the per-query exact-scorer oracle. Caller cancellation is via `AbortSignal` (`AbortError`, pre-abort and phase-boundary checks). Mutations bump `generation` so overlapping searches abort fail-closed instead of reading torn state. |
| `SearchWorkerClient` | Monotonic sequencing: each new `search()` immediately rejects superseded pendings with `AbortError` and posts `ABORT` (worker drops stale `queryId` and aborts its controller). Rapid typing resolves only the latest query. Caller `signal` also rejects with `AbortError`. |
| `WebGPUEngine` | Serialized via a single mutex (`queued()`); `loadDataset \|\| search` and `searchCold(A) x searchCold(B)` never cross query echoes; `generation` discards stale `mapAsync` reads. |

Proven by: `bun run test:reliability` §3 (direct independence + oracle
equality, pre-aborted sibling isolation, worker 5-rapid last-wins) plus the
existing `test:worker` §5 and `test:parity` Part 5 gates.

---

## 4. Deterministic fallback

GPU-unavailable and GPU-lost queries serve the Section 1 CPU contract
(`docs/support-matrix.md`): same `totalMatches` / `candidateCount` /
`hasOverflow`, same ordered `(index, score)` / `(id, score, matchedField)`,
same `profileId` / `scoringVersion` / `cpuScorer` echoes, descending integer
scores.

| Cause | `fallbackReason` |
| --- | --- |
| Explicit CPU (`preferGpu: false`), `fuzzy` / `substring` | `prefer-cpu` |
| `token` / `prefix` / typo-tolerant (exact-only WGSL) | `unsupported-mode` (even after loss — routing, not scorer) |
| No WebGPU / adapter reject | `webgpu-unsupported` / `device-request-failed` |
| Simulated/real device loss | `device-lost` |
| Dispatch failure (injected `gpuEngine.search` throw) | `gpu-execution-error` |
| Explicit `cpuScorer: 'ufuzzy'` | `cpu-algorithm-requested` (opt-in, non-conforming) |

`AbortError` never converts to fallback — it propagates (epoch-moved
`mapAsync` rejects map to `AbortError`, never to CPU results).

Proven by: `bun run test:reliability` §4 + `test:parity` Part 7
(unavailable-vs-oracle, lost-vs-baseline, dispatch-failure delta).

---

## 5. Suites

```bash
bun run test:reliability   # Phase 2 headless proofs (58 checks, no GPU/browser)
bun run test:mock          # buffer layout + routing + abort/destroy/re-init
bun run test:worker        # sequencing + AbortSignal + DESTROY lifecycle
bun run test:browser       # executing-hardware release gate (Chrome + WebGPU)
```
