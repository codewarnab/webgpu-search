# Contributing to webgpu-search

Thanks for contributing. This repo is a Bun + Turborepo monorepo with a
zero-dependency core library (`packages/webgpu-search`), a product site
(`apps/site`), a benchmark harness (`apps/benchmark`), framework recipes
(`examples/`), and headless test/benchmark scripts (`scripts/`).

## Ground rules

- Be kind. The [Code of Conduct](./CODE_OF_CONDUCT.md) applies everywhere.
- Security issues: follow [SECURITY.md](./SECURITY.md), do not open public
  issues for vulnerabilities.
- Library code must stay portable across browser main thread, Web Workers,
  Node.js, and SSR: no unguarded `window` / `document` / `navigator` access.
- Route WebGPU device acquisition through `GpuDevicePool` /
  `WebGPUContextManager` or an injected `options.device`. Never call
  `navigator.gpu.requestDevice()` per index.
- Keep WebGPU and CPU scoring symmetric (`SearchResultItem` descending
  normalized integer scores, deterministic tie-breakers in `src/ranking.ts`).
- Every WGSL change must keep `.wgsl` sources and inlined export strings in
  sync and pass `bun run check:shaders`.

## Prerequisites

- Bun `>= 1.1` (repo pins `bun@1.4.2`; CI uses `oven-sh/setup-bun@v2`).
- Node `>= 18` (engines field; used for release publishing).
- Chrome/Chromium only if you run the browser gate (`bun run test:browser`).

## Setup

```bash
git clone https://github.com/codewarnab/webgpu-search.git
cd webgpu-search
bun install

# Workspace dev servers (site on :5178, benchmark on :5173)
bun run dev
```

Open the site at `http://localhost:5178`. The `#benchmark` section runs a
live in-browser WebGPU-vs-CPU comparison over generated data.

## Project layout

- `packages/webgpu-search`: published `webgpu-search` library (`index` +
  `worker` entries, ESM + CJS). `files` ships `dist` + `README.md` + `LICENSE`.
- `apps/site`: product homepage + embedded `#benchmark` quick comparison.
  Published output is `apps/site/dist` (see `vercel.json`).
- `apps/benchmark`: full local benchmark harness (dev server `:5173`).
- `examples/react|vue|svelte|vanilla`: framework integration recipes.
- `scripts/`: headless suites (`test-*.ts`), benchmark matrix
  (`bench-snapshot-matrix.ts`, `benchmark-fixtures.ts`,
  `bench-data/` generators), and repo gates (`check-*.ts`).
- `docs/`: frozen `1.x` contracts (`public-api.md`, `snapshot-format.md`,
  `support-matrix.md`, `text-normalization.md`, `benchmarks.md`,
  `packaging.md`, `diagnostics.md`, `reliability.md`, `security-privacy.md`).

## Common commands

```bash
# Validate WGSL shader syntax + uniform alignment
bun run check:shaders

# Typecheck all workspaces + framework examples
bun run typecheck

# Production build (Turborepo)
bun run build

# Headless WebGPU-mock tests (no GPU/browser needed)
bun run test:headless

# Targeted suites
bun run test:contracts
bun run test:parity
bun run test:search-modes
bun run test:cpu-baseline
bun run test:reliability
bun run test:benchmarks

# Headless compatibility set (no GPU/browser)
bun run test:compatibility

# Full headless gate (compatibility + records/highlight/mutations/worker/
# snapshot/observability/normalization/filtering/faceting/diagnostics)
bun run test:all

# Browser regression gate (needs Chrome; starts apps/benchmark dev server)
bun run test:browser

# Bundle + public-surface gates
bun run check:bundle-size
bun run check:public-api
bun run lint:naming
bun run lint:parity
```

The browser gate needs a Chrome binary (`CHROME_BIN`). CI installs stable
Chrome via `browser-actions/setup-chrome@v1` and serves `apps/benchmark`
on `127.0.0.1:5173` before running `scripts/test-regression.ts`.

## Making changes

1. Create a focused branch from `main`.
2. For public-surface changes, update `docs/public-api.md` **and** the
   allowlist in `scripts/check-public-api.ts` in the same PR (CI fails
   otherwise). Scoring, storage, ordering, or echo changes require a major
   per `docs/public-api.md` §7.
3. For snapshot changes, update `docs/snapshot-format.md` and cover legacy
   restore, CRC, caps, getters, hooks, and profile guards in
   `bun run test:snapshot`.
4. For benchmark fixture changes, bump `BENCHMARK_FIXTURE_VERSION`, regenerate
   `benchmark_snapshot_matrix.json` + `.md`, and update `docs/benchmarks.md` §3.
5. Run the relevant gates before opening a PR (at minimum `check:shaders`,
   `typecheck`, `build`, `test:mock`, plus any suite covering your area).

## Pull requests

- Fill out `.github/PULL_REQUEST_TEMPLATE.md` (what changed, tests run,
  contract impact).
- Keep diffs clean: no temporary test files, debug logs, or dead scripts.
- File bug reports with `.github/ISSUE_TEMPLATE/webgpu-search-issue.md`
  (header/stats/timings metadata, never PII record text).

## Releases (maintainers)

Changesets govern versioning (`@changesets/cli`):

```bash
bun run changeset          # add a changeset for your change
bun run version-packages   # apply versions (CI release job consumes this)
bun run release            # turbo build + changeset publish
```

Pre-publish verification:

```bash
bun run prepublish:check
```

`attw --pack ./packages/webgpu-search` must pass on every release (both
`webgpu-search` and `webgpu-search/worker` entries × ESM/CJS).
