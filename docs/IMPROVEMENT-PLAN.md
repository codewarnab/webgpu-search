# Post-1.0 Improvement Plan

> **Created**: 2026-10-11
> **Baseline**: `webgpu-search@1.0.1` (commit `bee5fda`)
> **Status**: Proposed

This plan lists the next improvements, most valuable first. It comes from a quick look at the repo layout, tests, open issues and recent commits, not a full audit.

---

## Phase 1: Quick cleanup (≈ half a day)

Low-risk housekeeping that clears the way for the bigger phases.

| # | Task | Done when |
|---|------|-----------|
| 1.1 | **Implemented:** Keep one shared dataset-size list in `scripts/bench-data`, consumed by `apps/site` and `apps/benchmark`. | Only one definition exists; issue #38 closes when the cleanup PR merges. |
| 1.2 | **Done:** Move the benchmark snapshot JSON and Markdown to `docs/benchmarks/` and update the runner, checks, and docs. | Both artifacts live under `docs/benchmarks/`. |
| 1.3 | **Done:** Delete the two finished implementation plans and remove stale references. | Remaining documentation references resolve. |

---

## Phase 2: Tests for the 1.0 promises (≈ 2–3 days)

There are about 300 tests, but several core modules have none of their own. This phase covers the guarantees that 1.0 commits to.

### 2.1 Tests that GPU and CPU return the same scores
- **Browser coverage extended:** the existing `scripts/test-regression.ts` differential gate now includes joined emoji and component queries, plus 128-token ASCII and case-fold-expanded queries in substring and fuzzy modes. Matching records and last-token near misses exercise the actual WGSL path against the CPU reference. CI already runs this gate on every PR.
- Add `test/scoring-parity.test.ts`. It runs a fixed set of queries through the mock WebGPU path and through `cpu-reference.ts` / `exact-scorer.ts`.
- Check that both return the same `SearchResultItem[]`: same order, same integer scores, scores in descending order.
- Cover ASCII, CJK, emoji/ZWJ, mixed case, and queries at the maximum length.
- **Why:** AGENTS.md §2.4 requires the two paths to score the same, and nothing enforces it today.

### 2.2 Snapshot codec tests
- **Historical compatibility covered:** `scripts/test-snapshot.ts` loads the frozen `test/fixtures/snapshot-1.0.1.bin` written by the published npm package, with provenance and fixed search expectations in its JSON manifest. Existing round-trip and typed corruption/profile error checks remain in the same suite, which CI runs through `test:snapshot`.
- Add `test/snapshot-codec.test.ts`:
  - Saving an index and loading it back gives identical search results.
  - Bad input is rejected with the right typed error: wrong magic bytes, wrong version, truncated data, a corrupted checksum, a mismatched text profile.
  - A checked-in snapshot from 1.0.x still loads (a "golden" fixture).
- **Why:** the format in `docs/snapshot-format.md` is a public promise to keep old snapshots loading.

### 2.3 Text pipeline tests
- **Existing coverage verified:** normalization and case folding pass 98 checks in both default and Turkish locales; the highlighting suite covers NFC/NFD, folding expansions, surrogate pairs, and emoji.
- **Fallback bug fixed:** without `Intl.Segmenter`, highlights could split joined emoji and flags. Shared fallback boundaries now preserve ZWJ sequences and regional-indicator pairs; regression cases run with segmentation disabled before the normal suite. See `docs/public-api.md` §2.5.
- Add `test/text-normalization.test.ts`, covering NFC, case folding with `CaseFolding-16.0.0` (ß→ss, the Turkish dotless i, Greek final sigma), surrogate pairs, and `u32` packing.
- Add `test/highlight.test.ts`, checking that highlight ranges land on whole characters in multi-byte text.

### 2.4 Run in CI
- Make sure `bun run test:mock` in `.github/workflows/ci.yml` picks up the new files.

---

## Phase 3: Public results page for the collected benchmarks (≈ 3–4 days)

Benchmark results already go to Upstash automatically (the opt-in `api/submit-benchmark.ts` intake), but nobody can see them.

| # | Task |
|---|------|
| 3.1 | Add a read-only, cached API route in `api/` that returns aggregated results (median speedup grouped by GPU type, browser and dataset size). Return only aggregates, never raw rows. |
| 3.2 | Add a results section or page to `apps/site`: a chart of WebGPU vs CPU vs uFuzzy vs Fuse.js, with integrated and discrete GPUs shown separately. |
| 3.3 | Use the data to set the default `threshold` (currently 30,000) from measured crossover points. Consider a separate default for integrated GPUs. |
| 3.4 | Show a minimum sample count on each group, and hide groups with too few samples. |

**Done when:** the page is live, and the README links to it in place of the old benchmark numbers.

---

## Phase 4: Break up the largest files (≈ 3–5 days)

`document-index.ts` has 3,331 lines, almost 30% of the library. `webgpu-engine.ts` has 1,093.

### 4.1 Split `document-index.ts`
Proposed modules under `src/document/`:
- `build.ts`: ingesting records and building the index
- `query.ts`: running searches and merging results
- `filter-facet.ts`: wiring in the code from `filtering/` and `faceting/`
- `snapshot.ts`: saving and loading with `snapshot-codec.ts`
- `document-index.ts`: a thin public class that delegates to the above

### 4.2 Split `webgpu-engine.ts`
- Move buffer allocation and resizing, pipeline/shader setup, and dispatch/readback into separate modules.

### Rules for this phase
- Behavior must not change. The Phase 2 tests must pass before and after.
- The public API in `src/index.ts` must not change. Compare `dist/index.d.ts` before and after.
- Run `bun run check:shaders`, `typecheck`, `build` and `test:mock`, then do the subagent review that AGENTS.md requires.

---

## Phase 5: Bundle size (≈ 1–2 days)

| # | Task |
|---|------|
| 5.1 | Add `size-limit` (or a small script) to CI with a budget for each entry point (`.` and `./worker`). |
| 5.2 | Measure how much `case-fold-table.ts` and `fold-table.ts` add to the bundle. |
| 5.3 | If they are large, load the full Unicode tables only for the `unicode-default` profile, so ASCII-only users don't pay for them. This must not change the snapshot format. |

---

## Phase 6: New features (later, one at a time)

Rank these again once Phases 1–5 are done.

1. **Add and remove records without a full rebuild.** This would need GPU buffers that can grow and a way to mark deleted records ("tombstones"). It needs its own design doc first.
2. **Official framework bindings** (`webgpu-search/react`, `/vue`), built from the recipes in `examples/`.
3. **A Node example on the `webgpu` npm package (Dawn)**, to back up the claim that search runs on the server too. Add it to `docs/support-matrix.md`.

---

## Order and dependencies

```
Phase 1 ──► Phase 2 ──► Phase 4 (refactor needs the tests first)
               │
               └──────► Phase 5
Phase 3 can run in parallel with everything else
Phase 6 starts after Phase 4
```

## Release notes
- Phases 1–3 and 5 are patch or minor releases (add a changeset).
- Phase 4 should produce a `patch` changeset with no public API changes.
- Phase 6 items are minor releases, each with its own changeset.
