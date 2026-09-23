# Packaging, Bundle Budgets, Setup & Example-UI Accessibility

> Plan ref: `docs/ISSUE-11-PLAN.md` Phase 6.
> Status: frozen guidance for `1.x`. Budgets are enforced by
> `bun run check:bundle-size` (`scripts/check-bundle-size.ts`).

---

## 1. Package-size budgets (enforced)

Build outputs per `packages/webgpu-search/tsup.config.ts:1`: `index` + `worker`
entries, each in ESM (`dist/*.js`) + CJS (`dist/*.cjs`) with matching
`.d.ts` / `.d.cts` (`tsup` `minify: false`, `treeshake: true`, sourcemaps on).
Sourcemaps are excluded from the gate but must not ship to npm
(`package.json` `files` covers `dist` + `README.md` + `LICENSE`).

| Entry file | Gzip budget (per file) | Current (Phase 6) |
| --- | --- | --- |
| `dist/index.js` / `dist/index.cjs` | 90 KB | ~87.6 / ~87.9 KB |
| `dist/worker.js` / `dist/worker.cjs` | 76 KB | ~72.7 / ~72.8 KB |
| Index delta vs 234,907 B raw / 49,246 B gzip baseline | +40 KB gzip cap | +38.4 KB |

Measurement is deterministic gzip (`node:zlib`, default compression,
`mtime: 0`) — NOT `gzip -c`, which embeds filename+mtime and differs by
~200–300 B. The gate is fail-closed: missing `dist` or `dist` older than
library sources fails. Rationale log for every bump lives in the header of
`scripts/check-bundle-size.ts`; the Phase 6 entry records the Phases 0–4
mainline growth (+~5.6 KB index vs the old 82 KB budget: naming hygiene,
engine-state extraction, `powerPreference` validation, `rebuildGpu()` /
`[Symbol.dispose]` / pool accounting, snapshot agreement + profile guards,
worker table-dispatch + error rehydration) and the new per-entry worker
budget (no historical baseline — total-only gate).

Run:

```bash
bun run build
bun run check:bundle-size
```

---

## 2. Tree-shaking guidance

- `package.json` sets `"sideEffects": false`, so bundlers may drop unused
  exports from both entries. Keep it `false`: the library has no
  import-time global side effects (no top-level DOM / worker / GPU calls).
- Entry choice:
  - `webgpu-search` (`src/index.ts:1`) — full API: `SearchIndex`,
    `DocumentIndex`, `SearchWorkerClient`, engines, packing, ranking, hooks,
    diagnostics, snapshots.
  - `webgpu-search/worker` (`src/worker/search-worker.ts:35`) — dedicated-worker
    entrypoint only: `startSearchWorker`, `isDedicatedWorker`. Import this
    **inside the worker file only** (see §4); the main thread imports the
    client from the root entry.
- What pulls in WGSL / uFuzzy: both entries bundle the inlined WGSL shader
  text (`tsup` `loader: { '.wgsl': 'text' }`) and `@leeoniya/ufuzzy`
  (statically imported by `src/cpu-engine.ts:1`, reachable from
  `DocumentIndex` → CPU engine). There is currently no ufuzzy-free entry:
  importing either entry includes both payloads. Do not work around this
  with deep `src/*` imports — they are unsupported and may break in any
  release (`docs/public-api.md` §1).
- Prefer the Stable API (`SearchIndex` / `DocumentIndex` /
  `SearchWorkerClient`) over power-user engine/packing imports; the latter
  widen the retained module graph for no host benefit in most apps.

---

## 3. CSP setup

- No WASM, no `eval`, no remote fetches. The library is zero-network:
  compute is WGSL shaders + CPU TypeScript; the single runtime dependency
  is `@leeoniya/ufuzzy`.
- Default worker construction uses a **same-origin module URL**, not a blob:
  `SearchWorkerClient` builds `new Worker(new URL('./worker.js', import.meta.url),
  { type: 'module' })` (`src/worker/worker-client.ts`), and hosts typically
  construct `new Worker(new URL('./search.worker.ts', import.meta.url), { type: 'module' })`
  under Vite (see `examples/react/README.md`). Keep `script-src 'self'`
  and matching `worker-src 'self'`.
- Only if the host itself wraps the worker file in a `Blob` URL must
  `script-src` / `worker-src` additionally allow `blob:`. The library never
  requires it on its own.

---

## 4. Worker setup

Dedicated worker file (one line — pattern used by `examples/react`,
`examples/vue`, and `examples/svelte`):

```ts
// src/worker.ts (bundled as a separate worker entry by Vite)
import { startSearchWorker } from 'webgpu-search/worker';

startSearchWorker();
```

Main-thread client (`examples/react/useDocumentSearch.ts` pattern):

```ts
import { SearchWorkerClient } from 'webgpu-search';

const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
const client = new SearchWorkerClient<DocPageRecord>({ worker });
await client.init(records, {
  idField: 'id',
  fields: [{ name: 'title', weight: 3.0 }, { name: 'content', weight: 1.0 }],
  filterFields: [{ name: 'section', type: 'string' }],
  preferGpu: true,
});

// Query with cancellation; destroy + terminate on teardown.
const res = await client.search(query, { mode: 'fuzzy', limit: 50, signal });
await client.destroy();
worker.terminate();
```

Rules:

- `startSearchWorker()` no-ops outside a dedicated worker scope
  (`isDedicatedWorker` guard, `src/worker/search-worker.ts`) — importing
  `webgpu-search/worker` on the main thread is safe but does nothing.
- Function `hooks` and function `filter` closures cannot cross the boundary:
  `init` / `search` / `restore` reject real hooks with `IncompatibleHookError`
  (empty `{}` is a no-op); a function `filter` is applied host-side
  post-clone with `facets` + `diagnostics` dropped fail-closed
  (`src/worker/worker-client.ts`).
- Worker indexes recover via `restore()` / re-`init()` (the worker owns its
  `DocumentIndex`); `DESTROY` tears it down. `SearchWorkerClient.destroy()`
  is idempotent — always call it plus `worker.terminate()` on
  `pagehide` / component unmount (framework-recipe pattern in
  `examples/react/useDocumentSearch.ts`).
- Node.js / SSR: there is no default `Worker` — pass a custom
  `options.worker` instance or factory, otherwise construction throws with
  an actionable message.

---

## 5. SSR-safe imports

- Importing `webgpu-search` (either entry) at SSR / Node.js import time is
  safe: zero unguarded `window` / `document` globals (pinned by the
  `test:search-modes` portability audit and the `test:contracts` SSR
  worker guard). GPU detection is guarded
  (`typeof navigator !== 'undefined' && !!navigator.gpu`,
  `docs/support-matrix.md` §3); absence of WebGPU serves the CPU baseline
  with a documented `fallbackReason`.
- `isDedicatedWorker` is `false` outside workers; `startSearchWorker` returns
  without touching `postMessage`. Never construct a `Worker` at module scope
  — construct lazily inside `init`/effect handlers with a `typeof Worker !==
  'undefined'` guard and a main-thread `DocumentIndex` fallback.

---

## 6. Example-UI accessibility guidance

Hosts shipping their own search UI (see the `examples/` framework recipes)
should meet at least this bar:

1. **Labels for every control.** Every `<select>` / `<input>` has a
   programmatic label (`<label for="mode-select">`). Placeholder text is never the only label.
2. **Full keyboard operation.** Search input + result list must work without
   a pointer: `ArrowDown` / `ArrowUp` move selection, `Enter` opens the
   selection, `Escape` clears the query. Keep focus in the input while navigating
   results; move focus into dialogs (e.g. `formTitle.focus()` on modal open)
   and return it on close.
3. **Live-region announcements.** Expose result-count + engine/fallback status
   in an `aria-live="polite"` region so screen-reader users hear
   "N matches, CPU fallback (device-lost)" without focus theft. Render these
   as plain telemetry spans wrapped in a live region.
4. **Highlight safety.** Render `highlightedText` only through an escaping
   sanitizer that preserves `<mark>`; never inject raw record text via
   `innerHTML`. Honor `escapeHtml: true` on untrusted corpora.
5. **Reduced motion.** Gate smooth scrolling / animations behind
   `matchMedia('(prefers-reduced-motion: reduce)')` — fall
   back to `behavior: 'auto'` when reduced motion is requested.
6. **Status + error visibility.** Surface `fallbackReason`, overflow, and
   restore rejects as text (not color alone), and announce them in the live
   region above.

---

## 7. Unsupported configurations

Authoritative list: `docs/support-matrix.md` §6 (Firefox/Safari-Nightly WebGPU
paths, Node/Bun/SSR as GPU paths, `cpuScorer: 'ufuzzy'` as conforming,
cross-scorer score comparisons, SwiftShader-as-hardware claims, function
`filter` + `facets` over the worker). Privacy/trust boundaries:
`docs/security-privacy.md`. Diagnostics + issue-report data:
`docs/diagnostics.md`.
