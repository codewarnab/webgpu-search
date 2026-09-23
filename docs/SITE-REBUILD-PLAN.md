# webgpu-search Website Rebuild Plan

**Status:** Approved; implementation in progress on `feat/site-rebuild`.  
**Owner intent:** Replace the current benchmark dashboard UI with a polished product homepage, while keeping browser-based comparison and full benchmark testing available as clear, separate pages on the same site.  
**Primary deployment:** Vercel, currently configured to publish `apps/benchmark/dist`.  
**Plan date:** 2026-09-23

**Latest direction confirmation:** The current benchmark UI is to be discarded, not retained as the homepage. The full benchmark remains available as a redesigned page on this site. The user approved implementation after this plan was written.

## 1. Product goal

Turn the current deployed benchmark dashboard into an approachable product website for `webgpu-search`.

The first impression should explain what the library does and let a visitor take one obvious next step. Visitors who want to experiment can compare search in their own browser. Visitors who want rigorous numbers can open a dedicated full benchmark page on the same site. Example apps should be directly testable from the site.

The existing benchmark engine and data-generation work are useful; the current benchmark page's visual design and information layout are not. Retire that UI completely and redesign the benchmark experience rather than serving it unchanged under a different link.

## 2. Approved design direction

- **Visual style:** calm, bright, minimal, Vercel-inspired; white or warm-white canvas, near-black primary text, soft-gray supporting text and borders.
- **Color:** black/white/gray for the interface. No blue decorative text, blue primary buttons, or saturated metric tiles. Restrained syntax colors are acceptable inside code samples only.
- **Typography:** clean system sans-serif for reading; monospace only for code, queries, and compact technical values.
- **Density:** generous spacing, short explanations, a few clear metrics, and progressive disclosure for technical detail.
- **Language:** plain words first. Explain GPU, CPU fallback, cold upload, and benchmark qualification beside the relevant result rather than in a wall of documentation.
- **Responsive and accessible:** keyboard usable, named controls, visible focus, adequate contrast, semantic headings, and mobile layouts.

The Here.now prototype was an initial visual sketch. Its blue accents are superseded by this monochrome direction; the screenshot of the old dashboard is the anti-reference for density and visual noise.

## 3. Site map and visitor journeys

### `/` — Product homepage

1. **Header:** brand, Features, Examples, Docs/Source, and a clear “Try it in your browser” action.
2. **Hero:** one-sentence value proposition, concise explanation of WebGPU with CPU fallback, and two primary actions: “Try search” and “Compare performance.”
3. **Short code sample:** familiar `SearchIndex` usage, with syntax highlighting confined to the snippet.
4. **Interactive search demo:** small local sample dataset, immediate results, simple fuzzy/substring choice, and a clear note that it is a small demonstration—not a performance claim.
5. **Feature summary:** hybrid routing, fuzzy and substring search, Unicode handling, and Worker support; no exhaustive API inventory.
6. **Example app cards:** direct “Open test app” links for Docs Search, Code Palette, and Log Viewer.
7. **Closing install and navigation:** package install command and links to the comparison, full benchmark, source, and npm package.

### `/compare/` — Simple browser comparison

An approachable, real comparison of WebGPU and exact CPU search using the visitor’s own browser and the same generated data/query.

- Default to a safe, useful dataset size; offer a small and a larger option without exposing a long matrix of settings.
- Limit direct GPU/CPU comparison to the supported WebGPU modes (fuzzy and substring) so both sides use the same search contract.
- Run both engines against the same records and query, warm up, collect repeated samples, and report median search latency.
- Separate index/data preparation time from warm-search latency.
- Show each engine’s result count, elapsed time, and whether the top results agree. Report actual values; never promise the GPU will always win.
- Detect WebGPU availability and display CPU-only/fallback status honestly. If the GPU path did not run, do not label CPU timings as GPU timings or show a speedup.
- Keep results in the browser. The page must not upload a visitor’s queries or data.
- Explain in one sentence that device, browser, dataset size, and query affect results.
- Provide a visible link to `/benchmark/` for users who want the full test matrix.

### `/benchmark/` — Full benchmark, redesigned

Keep the full benchmark capability available on the same website, but replace the current dense dashboard UI with a new, calmer interface.

- Reuse the established `BenchmarkRunner`, dataset generator, parity/reference logic, and export functions where they remain correct.
- Present the run as a short flow: choose corpus and search mode, choose dataset sizes, run, then read the summary.
- Keep a sensible default run. Large sizes (up to the supported 2M rows) remain opt-in and display a note that runtime and memory use depend on the visitor’s device.
- Put the main answer first: which engine ran, typical search latency, result-count agreement, and hardware/fallback status.
- Show median and p95 clearly. Keep cold upload/index preparation distinct from retained, repeated searches.
- Put detailed per-engine timings, VRAM estimates, charts, and the full result table in collapsible “Details” sections.
- Preserve corpus options (ASCII, CJK, and emoji/mixed script) and both substring/fuzzy benchmark modes.
- Preserve useful export actions (CSV, chart image, and concise Markdown summary), but keep them secondary to reading the result.
- Clearly label software/mock or otherwise non-qualified hardware results; avoid presenting them as dedicated-GPU performance.

## 4. Example app test links

The repository already contains these working apps:

| Site route | Workspace app | Purpose |
| --- | --- | --- |
| `/examples/docs-search/` | `apps/docs-search` | Offline documentation search, filters/facets, suggestions, highlighting, and snapshots |
| `/examples/code-palette/` | `apps/monaco-palette` | Keyboard-first file/symbol search and preview |
| `/examples/log-viewer/` | `apps/log-viewer` | Large log search, filtering, live stream, and facets |

The deployed site should serve these apps on the same origin so the homepage links open runnable examples rather than source-code pages. Their Vite asset bases must match their mounted paths. The site build will stage each example build into the site’s static output under the corresponding route.

## 5. Implementation architecture

### Site shell

- Add a dedicated `apps/site` workspace for the new landing page and simple comparison page.
- Keep the site static and client-side; no backend service or remote search API is introduced.
- Use the `webgpu-search` workspace package for the interactive comparison, with CPU and WebGPU indexes over the same data.
- Keep the landing and comparison sources separate from the benchmark runner UI so each page remains focused.

### Full benchmark app

- Continue using `apps/benchmark` as the full benchmark execution workspace, but replace its current `index.html`, `src/style.css`, and old dashboard-oriented `src/main.ts` UI.
- Retain and reuse benchmark modules such as `src/benchmark.ts`, `src/dataset.ts`, `src/export.ts`, and the worker where practical.
- Build the redesigned full benchmark with its Vite base set to `/benchmark/`.
- The benchmark workspace may still be named `benchmark` internally; that name does not dictate the deployed home page.

### One Vercel deployment

- Change `vercel.json` so the deployed output is `apps/site/dist`.
- Make the site build depend on the benchmark and example-app builds, then stage their static output under `apps/site` before the site Vite build.
- Set each mounted app’s Vite base to its final same-origin route, so scripts, styles, workers, and assets resolve under the correct path.
- Verify Vercel serves `/`, `/compare/`, `/benchmark/`, and all three `/examples/.../` routes from this single deployment.

## 6. Existing benchmark UI retirement

The user explicitly wants the current dashboard design discarded. Implementation will:

1. Replace its current benchmark-console page as the deployed root homepage.
2. Remove obsolete console markup/styles/DOM wiring that are not needed by the redesigned `/benchmark/` page.
3. Reuse the benchmark logic, not the current visual treatment or information hierarchy.
4. Keep the full benchmark available as `/benchmark/` on the same site.
5. Keep source/data/benchmark modules only where they serve the redesigned full benchmark or tests; remove dead UI code after checking references.

## 7. Likely files and areas to change

- `docs/SITE-REBUILD-PLAN.md` — this durable plan and status.
- `apps/site/` — new homepage, comparison page, shared styles, comparison logic, build/staging script, and app metadata.
- `apps/benchmark/index.html`, `apps/benchmark/src/main.ts`, `apps/benchmark/src/style.css` — replace the old dashboard UI with the new full-benchmark UX.
- `apps/benchmark/vite.config.ts` — benchmark route asset base.
- `apps/docs-search/vite.config.ts`, `apps/monaco-palette/vite.config.ts`, `apps/log-viewer/vite.config.ts` — mounted asset bases for same-origin example routes.
- `apps/*/package.json` as needed — workspace build dependencies/order.
- `vercel.json` — publish the new site output.
- `README.md` — point the live demo and feature links at the new homepage and pages after the routes are implemented.

Exact file changes may be adjusted to fit existing Vite/Turbo conventions; the user-facing routes and requirements above are the contract.

## 8. Build sequence

1. **Record direction and inspect contracts:** keep this plan current; confirm index options, response timing fields, worker behavior, and benchmark runner inputs before wiring UI.
2. **Create the site workspace:** implement the monochrome responsive shell, homepage, feature summary, install block, and route navigation.
3. **Implement browser comparison:** build a reproducible generated corpus, run exact CPU and WebGPU-compatible fuzzy/substring queries, surface real engine/fallback status, and show simple readable results.
4. **Redesign the full benchmark page:** replace the old dashboard UI, reuse its benchmark core, make the result summary readable, and progressively disclose detailed tables/charts/exports.
5. **Mount examples on the same origin:** set their Vite bases, add build ordering/staging, and add working homepage cards for each deployed route.
6. **Switch the Vercel root:** point production output at `apps/site/dist`; confirm the old dashboard is no longer the home page and full benchmark is reachable under `/benchmark/`.
7. **Verify and refine:** run required repo checks, test GPU-available and CPU-fallback paths, inspect desktop/mobile layouts and all routes, and fix any issues before publishing a review preview.

## 9. Acceptance criteria

- [ ] `/` is a new clean product homepage, not the current benchmark dashboard.
- [ ] Primary UI is monochrome, light, calm, and readable; code highlighting may use restrained colors.
- [ ] The homepage has a compact working search demo and clear links to comparison, full benchmark, and example apps.
- [ ] `/compare/` runs a genuine in-browser WebGPU-versus-exact-CPU comparison over the same dataset and query.
- [ ] Comparison reports actual timings, result counts, and fallback/device state without fabricated values or unsupported speedup claims.
- [ ] `/benchmark/` runs the full matrix locally in the visitor’s browser and uses a redesigned, less overwhelming UI.
- [ ] Full benchmark retains useful detailed statistics, multi-corpus coverage, qualification notes, and exports.
- [ ] Docs Search, Code Palette, and Log Viewer work at their same-origin routes on the deployed site.
- [ ] Visitors’ searches/datasets remain local to their browser.
- [ ] All routes work after Vite build and Vercel static deployment, including workers and nested assets.
- [ ] `bun run check:shaders`, `bun run typecheck`, `bun run build`, and `bun run test:mock` pass.
- [ ] No old-dashboard-only DOM hooks, styles, debug logs, or dead route assets remain.

## 10. Verification and review

For completion, run the required repository gates:

```bash
bun run check:shaders
bun run typecheck
bun run build
bun run test:mock
```

Then verify the built site routes and the comparison/benchmark behaviors in a real browser where possible. Exercise WebGPU unavailable/fallback presentation as well as the WebGPU-ready path. Check that results are local, no benchmark claim is hard-coded as measured data, mobile layouts remain usable, and all visible test links point to working pages. Repository policy also requires a verification subagent review before sign-off.

## 11. Scope boundaries

- No change to the core library’s public search contract is planned.
- No WGSL shader edit is planned unless a defect is discovered; any such edit must keep source and inlined shader exports synchronized and pass shader checks.
- No server-side compute, account system, or analytics pipeline is needed for the website.
- The existing full benchmark capability is retained, but its current UI is not.
- A public Vercel deployment should happen only after local checks and review; a Here.now preview can be used for design review if requested/appropriate.
