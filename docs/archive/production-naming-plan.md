# Production-Ready Naming & Hygiene Plan — `webgpu-fuzzy-search`

## Implementation Status (updated 2026-09-21, uncommitted on `main`)

All phases implemented in working tree (114 changed files, nothing committed).
`bun run test:all` green, `lint:naming` clean, `check:bundle-size` passes,
`typecheck`/`check:shaders` green, forced `turbo build --force` dist fresh.

- [x] Phase A — `lint:naming` gate (`scripts/check-naming.ts`) clean.
- [x] Phase B — B1 constants + aliases; B2 files + old-path `export *` shims
  (incl. `modes/*`, `filter/*`, `facets/*`); B3 comment/error sweep,
  `folded`→`normalized` (private fields, exported params, JSDoc),
  `cpuScorer` canonical + `cpuAlgorithm` deprecated (options + response
  echo both), `hooks`/`autocomplete` canonical + deprecated aliases,
  `DocumentIndex.autocomplete()` + `getHooks()`, `GpuDevicePool`,
  `snapshot-codec`/`snapshot-idb`, `packDataset`, `compareExactResults`,
  `foldCaseScalar`/`CASE_FOLD_*` (generator emits `@deprecated FOLD_*`).
- [x] Phase C — scripts `git mv` per §3.3, `package.json` (`test:<domain>`,
  `check:bundle-size`, `bench:snapshot-matrix`, `lint:naming`,
  `test:headless` + `test:mock` alias), CI step names, `PARITY_SHARD_*`
  with `M4_SHARD_*` fallback + warning, locals renamed.
- [x] Phase D — `apps/benchmark`, `monaco-palette`, `log-viewer`,
  `examples/` on canonical names; WGSL untouched (logic freeze).
- [x] Phase E — root clutter → `docs/archive/benchmark-evidence/`,
  `PLAN`/`ISSUE-*` → `docs/archive/`, `GEMINI.md` removed (identical to
  `AGENTS.md`), `docs/text-normalization.md` + `docs/snapshot-format.md`
  rewritten, `docs/naming-conventions.md` added, README on canonical names.
- [x] Phase F — verification subagent PASS (2026-09-21): check:shaders,
  typecheck, build, test:mock green; contracts/aliases verified;
  portability clean (guarded document check only); clean-diff audit PASS.
  1 MUST-fix resolved: generator template now emits per-export
  `@deprecated` FOLD_* + `normalized(cp)` header; table regenerated
  (canonical CASE_FOLD_*, --check green). SHOULD-fix resolved:
  `encodeSnapshot` error string no longer leaks old name.
- Deviations from §3.2: kept `FORMAT_VERSION_4`/`DOC_FORMAT_VERSION_4` as
  `@deprecated` aliases (not deleted) so old imports compile for one minor
  per §4/§6; `SCORING_VERSION='parity-v1'` value frozen (wire).
- `test:diagnostics` wired into `test:all` + CI (was orphaned).
- Test updates: parity harness compares resolved `cpuScorer`; worker
  `IncompatibleHookError` key is `'hooks'` (alias path still throws).

## 0. Goal / Non-goals

**Goal:** remove all planning artifacts (`m1-m8`, `M1-M8`, `v0.x`, `v04`, `phase`, `milestone`, `install`, `U2D3/U2D4/U2F2`, `V2/V4`, `FORMAT_VERSION`) from **user-visible identifiers**: file names, exported functions/types/constants, npm script names, variables, error messages, docs headings.

**Non-goals:**

* No scoring, WGSL, binary layout, or behavioral change. Wire magics (`0x55324434`, `0x55324632`) stay byte-identical; only the *exported names* change.
* No feature work. Pure rename + relocate + deprecate.
* `U2D4` on-disk stays readable forever via migration path; we only stop exporting that acronym as public vocabulary.

**Principle:** names describe *domain concepts*, not *when they were built*.

* Bad: `test-m2-filter`, `PackedUnicodeBufferV2`, `U2D4_MAGIC`, `check-m2-bundle`, `bench-v04-matrix`, `// v0.4 M4: ...`
* Good: `test:filter`, `PackedDataset`, `DOCUMENT_SNAPSHOT_MAGIC`, `check:bundle-size`, `bench:snapshot-matrix`, `// Token quorum: ...`

---

## 1. Findings Inventory (verified by grep + read)

### A. File names leaking milestones

| Current | Location | Problem |
|---|---|---|
| `scripts/test-m1-contracts.ts` | `scripts/` | Milestone number in durable test |
| `scripts/test-m2-records.ts`, `test-m2-filter.ts`, `test-m2-preprocess.ts` | `scripts/` |  |
| `scripts/test-m3-highlight.ts`, `test-m3-facets.ts` | `scripts/` |  |
| `scripts/test-m4-mutations.ts`, `test-m4-parity.ts` | `scripts/` | `M4_SHARD_*` env vars inside parity harness |
| `scripts/test-m5-worker.ts` | `scripts/` |  |
| `scripts/test-m6-persistence.ts` | `scripts/` | Contains `runM6Tests()`, `❌ M6 Tests failed` |
| `scripts/test-m7-observability.ts` | `scripts/` |  |
| `scripts/test-m8-proof-apps.ts` | `scripts/` | Contains `runM8Tests()` |
| `scripts/check-m2-bundle.ts` | `scripts/` | Bundle gate, not M2-specific |
| `scripts/check-parity-lint.ts` header | `scripts/` | Comment says `M3 parity-path lint` |
| `scripts/bench-v04-matrix.ts` | `scripts/` | `v04` version tag in filename + `[bench-v04-matrix]` log prefix |
| `scripts/generate-fold-table.ts` | `scripts/` | `fold` jargon, otherwise keep |
| `PLAN.md`, `ISSUE-7-PLAN.md`, `ISSUE-9-PLAN.md`, `ISSUE-10-PLAN.md` | root | Planning artifacts, must not ship in npm / confuse contributors |
| `benchmark_*.json`, `fuzzy-benchmark-results.json`, `*.png`, `render_*.js`, `generate_*.py`, `teaching_*.tex` | root | Benchmark output clutter in repo root |
| `GEMINI.md` vs `AGENTS.md` | root | Duplicated agent instructions |

### B. Exported symbols leaking versions / cryptics

File: `packages/webgpu-search/src/text-profile.ts:9-24`, `buffer.ts:43`, `types.ts:121,314`, `persistence.ts`, `worker-client.ts`, `index.ts:31-247`

| Current export | Problem |
|---|---|
| `FORMAT_VERSION (=2)`, `DOC_FORMAT_VERSION (=3)`, `U2D4_FORMAT_VERSION (=4)`, `FORMAT_VERSION_4`, `DOC_FORMAT_VERSION_4` | Three names for `4`; generic `FORMAT_VERSION` collides; `U2D4` is internal codename |
| `SERIALIZED_MAGIC` (`U2F2`), `SERIALIZED_DOC_MAGIC` (`U2D3`), `U2D4_MAGIC` | `SERIALIZED_*` vague; `U2Dx` meaningless to users |
| `SERIALIZED_DOC_HEADER_BYTES (48)`, `U2D4_HEADER_BYTES (56)` | Same |
| `PackedUnicodeBufferV2`, `UnicodePackOptions`, `packUnicodeToGPUBuffer`, `serializeUnicodeDataset` | `V2` = "after M2"; `Unicode` redundant once profile system exists |
| `packStringsToGPUBuffer`, `sanitizeStringForSlot`, `sanitizeString` | Legacy v0.1 ASCII path still exported; names say `Slot` (removed fixed-slot design in v0.2) |
| `countUnicodeCodePoints` | Deprecated M1 pre-fold counter, kept for compat; name implies canonical counter |
| `foldCodePoint`, `FOLD_RANGES`, `FOLD_EXPANSIONS`, `FOLD_C_COUNT`, `FOLD_F_COUNT`, `FOLD_UNICODE_VERSION`, `folded: boolean` everywhere | `fold` is implementer jargon for Unicode Default Case Folding + NFC; users expect `normalize` / `caseInsensitive` |
| `CpuAlgorithm='parity'\|'ufuzzy'`, `searchCpuReference`, `CPUEngine.searchUFuzzy/searchNative` | `parity` = "M2 scorer", `ufuzzy` leaks dependency name into public union; `Cpu` casing inconsistent (`CPU` vs `Cpu`) |
| `hybrid-index.ts` → `SearchIndex` | `hybrid` is architecture history (CPU+GPU routing), not a user concept |
| `cpu-reference.ts` vs `cpu-engine.ts` | Two "CPU" modules; unclear which is public contract |
| `modes/token-search.ts`, `modes/prefix-search.ts`, `modes/typo-distance.ts` | Generic `modes/`; `typo-distance` undersells bounded Damerau-Levenshtein |
| `filter/bitset.ts`, `filter/columnar-store.ts`, `filter/filter-evaluator.ts` | Generic; `bitset` lowercase impl detail |
| `facets/facet-engine.ts` | Plural folder, singular concept |
| `extensions.ts` (`SearchExtensionHooks`, `hookId`, `hasAnyHook`) | `extensions` vs `hooks` used interchangeably |
| `diagnostics.ts` (`BROAD_QUERY_*`, `QueryDiagnostics`, `NormalizedCostBudget`) | `diagnostics` + `telemetry` + `budget` + `warnings` synonyms |
| `runtime-guards.ts` (`clampLimit`, `abortError`, `nowMs`) | Grab-bag name |
| `context-manager.ts` (`WebGPUContextManager`) | Manager suffix; hides singleton device pool |
| `persistence.ts` vs `idb-storage.ts` | Split snapshot vs IndexedDB confuses; both are persistence |
| `suggest.ts` (`SUGGEST_*`, `SuggestOptions`) | `suggest` ambiguous vs `autocomplete` used in docs |
| `worker/*` (`INTERNAL_WORKER_ID_KEY`, `SerializedWorkerError`, `Worker*Payload`) | `INTERNAL_*` exported from public entry; verbose payload union |

Local-variable leaks (low severity, fix opportunistically): `sm1/sm2/sm3` in `test-m3-highlight.ts:49-69`, `lim1/lim2` in `test-m2-preprocess.ts:260-270`, `mmc` in `modes/token-search.ts:105`, `cp`, `ok()` helpers in scripts.

### C. Comment / JSDoc / error-string leaks

* `src/index.ts:31,65,73,78,90,109,128,145,163,218,230,247` — section headers `// v0.4 ... (M2)`, `// v0.3 ... (M5)`. These ship in `.d.ts` tooltips.
* `src/types.ts:18,22,27,33,38,43,51,57,92-94,104-107,119,144,256,369,390,398,421,489,530,645,663,710,754` — `v0.2`, `v0.4 M4/M7/M8`, `M5 behavior change`, `reserved in M1`, `phase boundaries`.
* `src/text-profile.ts:2-4,29,40,50-53`, `src/buffer.ts:3,80-81,98`, `src/hybrid-index.ts:99,107,170,241,272,281,318,333,362,422,443,446,609`, `src/webgpu-engine.ts:56,494`, `src/runtime-guards.ts:2,78-79`, `src/persistence.ts:34,106,151,209-212,337,382-383,412,555` — same pattern.
* `apps/benchmark/src/main.ts:164,479-487` — `M4 string-isolated enrichment`, `U2F2 serialized buffer`, `M3 engine ignored them`.
* `docs/unicode-contract.md:8,50,153,164` — references `scripts/test-m4-parity.ts`, `M2 gate`.
* Error strings containing `v0.1`, `v0.2`, `v0.3`, `legacy-ascii-v0.1`, `slotBytes throw-on-use in v0.2` (`buffer.ts:87,104,170`, `webgpu-engine.ts:56`, `types.ts:105`).

### D. npm scripts + CI leaking milestones

Root `package.json:15-37` + `.github/workflows/ci.yml:29-96`:

`test:m1` … `test:m8`, `test:filter` → `test-m2-filter.ts`, `test:facets` → `test-m3-facets.ts`, `test:parity` → `test-m4-parity.ts`, `test:preprocess` → `test-m2-preprocess.ts`, `check:bundle` → `check-m2-bundle.ts`, `bench:v04-matrix`, `test:all` chaining all `m*`. Step names: `Run Milestone 1 …`, `Run M2 …`, `Check Bundle Size Budget (M2)`, `Lint Parity Invariants (M2)`, `Verify Fold Table Freshness (M2)`.

### E. Root clutter (not identifiers, but blocks production readiness)

Delete or archive: `benchmark_results.json`, `benchmark_summary.md`, `benchmark_v04_matrix.json`, `fuzzy-benchmark-results.json`, `*.png` (6 files), `render_chart.js`, `render_clean_minimal_chart.js`, `generate_chart_matplotlib.py`, `generate_minimal_*.py`, `teaching_gpu_performance_mechanisms.tex`, `full-benchmark-ui.png`. Keep only `docs/` + `examples/` + `apps/` + `packages/` + `scripts/` at root.

---

## 2. Naming Conventions to Adopt

1. **Domain-first, no temporal prefixes.** `document-snapshot`, `packed-dataset`, `text-normalization`, never `v2`, `m3`, `u2d4`.
2. **One concept, one term** (enforce via `check-parity-lint.ts` extension):
   * `normalize` (not `fold`/`sanitize`) for Unicode NFC + case folding.
   * `caseSensitive: boolean` at API boundary; internal `normalized: boolean` (not `folded`).
   * `cpuScorer: 'exact' | 'ufuzzy'` (not `parity`); `exact` describes behavior.
   * `autocomplete` for user-facing suggest feature; `suggest` stays as deprecated alias only if needed.
   * `snapshot` for versioned binary persistence (replaces `U2D*`/`serializeDocumentIndex` vocabulary in docs).
   * `devicePool` (not `context-manager`/`manager`).
3. **File name = exported concept, kebab-case.** `hybrid-index.ts` → `search-index.ts`, `cpu-reference.ts` → `exact-scorer.ts`, `modes/typo-distance.ts` → `search/typo-tolerance.ts`, etc.
4. **Constants:** `<DOMAIN>_<WHAT>` — e.g. `DATASET_MAGIC`, `SNAPSHOT_MAGIC`, `SNAPSHOT_HEADER_BYTES`.
5. **Scripts:** `<verb>-<domain>.ts` — e.g. `test-contracts.ts`, `check-bundle-size.ts`, `bench-snapshot-matrix.ts`. npm script `test:<domain>`.
6. **Comments:** explain *why*, never *when*. Ban regex `\b[Mm][1-8]\b|\bv0\.[1234]\b|\bU2[D F][234]\b|\bV2\b|\bphase [123]\b` in `packages/*/src` via lint.

---

## 3. Concrete Rename Map

### 3.1 Library files (`packages/webgpu-search/src/`)

| Old | New | Notes |
|---|---|---|
| `hybrid-index.ts` | `search-index.ts` | `SearchIndex` stays; re-export old path for 1 minor |
| `cpu-reference.ts` | `exact-scorer.ts` | Functions `searchCpuReference` → `scoreExactMatches`, `searchMultiFieldCpuReference` → `scoreExactMatchesMultiField` |
| `cpu-engine.ts` | `cpu-engine.ts` (keep) | Methods `searchUFuzzy` → `searchWithUFuzzy` (explicit vendor), `searchNative` → `searchNaiveScan`; keep aliases |
| `buffer.ts` | `dataset-packing.ts` | `PackedUnicodeBufferV2` → `PackedDataset`; `packUnicodeToGPUBuffer` → `packDataset`; `serializeUnicodeDataset` → `serializeDataset`, `deserializeUnicodeDataset` → `deserializeDataset` |
| `text-profile.ts` | `text-profile.ts` (keep, rewrite header) | Constants below; delete `FORMAT_VERSION_4` / `DOC_FORMAT_VERSION_4` dupes |
| `fold-table.ts` | `case-fold-table.ts` (generated, keep `generate-case-fold-table.ts` in sync) | `FOLD_*` → `CASE_FOLD_*`; `foldCodePoint` → `foldCaseScalar` (internal; keep alias) |
| `unicode-preprocess.ts` | `text-normalization.ts` | `normalizeText` stays (good name) |
| `modes/token-search.ts` | `search/token-search.ts` | Keep function names (`splitQueryTerms`, `scoreTokenTokens` good) |
| `modes/prefix-search.ts` | `search/prefix-search.ts` |  |
| `modes/typo-distance.ts` | `search/typo-tolerance.ts` | `damerauLevenshteinBounded` stays; file rename only |
| `filter/` | `filtering/` | `bitset.ts` → `doc-bitset.ts` (`DocumentBitset` stays), `columnar-store.ts` stays, `filter-evaluator.ts` → `compile-filter.ts` (matches exported `compileFilter`) |
| `facets/facet-engine.ts` | `faceting/facet-engine.ts` | Folder singular concept |
| `extensions.ts` | `hooks.ts` | `SearchExtensionHooks` → `SearchHooks` + `type SearchExtensionHooks = SearchHooks` alias; `normalizeSearchExtensionHooks` → `normalizeSearchHooks` |
| `diagnostics.ts` | `diagnostics.ts` (keep) | Rename `BROAD_QUERY_*` → `BROAD_SEARCH_*`; keep old as deprecated aliases |
| `runtime-guards.ts` | `guard.ts` | Small, cohesive |
| `context-manager.ts` | `gpu-device-pool.ts` | `WebGPUContextManager` → `GpuDevicePool` + alias |
| `persistence.ts` + `idb-storage.ts` | `snapshot-codec.ts` + `snapshot-idb.ts` | `serializeDocumentIndex` → `encodeSnapshot`, `deserializeDocumentSnapshot` → `decodeSnapshotHeader` family; keep aliases |
| `suggest.ts` | `autocomplete.ts` | `SuggestOptions` → `AutocompleteOptions` + alias; `SUGGEST_*` → `AUTOCOMPLETE_*` |
| `ranking.ts` | keep | Good name |
| `highlight.ts` | keep | Good name |
| `document-index.ts` | keep | Good name |
| `worker/protocol.ts`, `worker/search-worker.ts`, `worker/worker-client.ts` | keep paths | Remove `INTERNAL_WORKER_ID_KEY` from public `index.ts` (import from `worker/worker-client` deep path only) |

### 3.2 Version constants (byte values unchanged)

```ts
// text-profile.ts — new canonical names, old kept as deprecated aliases for 1 minor
export const DATASET_FORMAT_VERSION = 2;          // was FORMAT_VERSION
export const DATASET_MAGIC = 0x55324632;          // was SERIALIZED_MAGIC (U2F2 on wire)
export const SNAPSHOT_FORMAT_VERSION = 4;         // was U2D4_FORMAT_VERSION (canonical)
export const SNAPSHOT_MAGIC = 0x55324434;         // was U2D4_MAGIC
export const SNAPSHOT_HEADER_BYTES = 56;          // was U2D4_HEADER_BYTES
export const LEGACY_SNAPSHOT_MAGIC = 0x55324433;  // was SERIALIZED_DOC_MAGIC (U2D3, read-only)
export const LEGACY_SNAPSHOT_VERSION = 3;         // was DOC_FORMAT_VERSION
export const LEGACY_SNAPSHOT_HEADER_BYTES = 48;   // was SERIALIZED_DOC_HEADER_BYTES
// DELETE: FORMAT_VERSION_4, DOC_FORMAT_VERSION_4 (exact dupes of 4)
```

`types.ts:121,314` union becomes `typeof DATASET_FORMAT_VERSION | typeof LEGACY_SNAPSHOT_VERSION | typeof SNAPSHOT_FORMAT_VERSION`.

`CpuAlgorithm`: `type CpuScorer = 'exact' | 'ufuzzy'` canonical; `type CpuAlgorithm = CpuScorer` alias + runtime accepts `'parity'` mapped to `'exact'` with deprecation warning.

### 3.3 Scripts + npm scripts + CI

| Old script file | New script file | New npm script |
|---|---|---|
| `test-m1-contracts.ts` | `test-contracts.ts` | `test:contracts` |
| `test-m2-records.ts` | `test-document-records.ts` | `test:records` |
| `test-m2-preprocess.ts` | `test-text-normalization.ts` | `test:normalization` (+ `:tr` locale variant) |
| `test-m2-filter.ts` | `test-filtering.ts` | `test:filtering` |
| `test-m3-highlight.ts` | `test-highlight.ts` | `test:highlight` |
| `test-m3-facets.ts` | `test-faceting.ts` | `test:faceting` |
| `test-m4-mutations.ts` | `test-mutations.ts` | `test:mutations` |
| `test-m4-parity.ts` | `test-parity-harness.ts` | `test:parity` (keep name); env `M4_SHARD_*` → `PARITY_SHARD_*` |
| `test-m5-worker.ts` | `test-worker.ts` | `test:worker` |
| `test-m6-persistence.ts` | `test-snapshot.ts` | `test:snapshot` (`runM6Tests` → `runSnapshotTests`) |
| `test-m7-observability.ts` | `test-observability.ts` | `test:observability` |
| `test-m8-proof-apps.ts` | `test-proof-apps.ts` | `test:proof-apps` (`runM8Tests` → `runProofAppTests`) |
| `check-m2-bundle.ts` | `check-bundle-size.ts` | `check:bundle-size` (keep `check:bundle` alias) |
| `check-parity-lint.ts` | keep | `lint:parity` (keep); strip `M3` from header comment |
| `bench-v04-matrix.ts` | `bench-snapshot-matrix.ts` | `bench:snapshot-matrix` |
| `test-vgpu-mock.ts`, `test-regression.ts`, `run-all-benchmarks.ts`, `run-fuzzy-benchmark.ts` | keep | `test:mock` → `test:headless`, keep `test:mock` alias for 1 minor |

CI step names: `Run Milestone N …` → `Run <domain> tests`; `(M2)` suffixes removed. `test:all` rebuilt from new names.

### 3.4 Docs / root

* Move `PLAN.md`, `ISSUE-*-PLAN.md` → `docs/archive/` (or delete if git history suffices; do not link from README).
* Merge `GEMINI.md` into `AGENTS.md`; delete `GEMINI.md`.
* Move root `*.png`, `*.json` (benchmark outputs), `render_*.js`, `generate_*.py`, `*.tex` → `docs/archive/benchmark-evidence/` or delete; add root `.gitignore` entries for `benchmark_*.json`, `chart_*.png`.
* Rewrite `docs/unicode-contract.md`, `docs/migration-v0.2.md`, `docs/migration-v0.4-u2d4.md` → `docs/text-normalization.md`, `docs/snapshot-format.md` with concept names; keep old files as redirect stubs for 1 release or delete if internal-only.
* `src/index.ts` section comments: replace `// v0.4 Structured filtering (M2)` → `// Structured filtering`, etc. (11 headers).
* JSDoc: replace `v0.2 breaking:` → `Breaking:`, `M2 serves …` → `The exact scorer serves …`, `reserved in M1` → `Reserved: accepted but not forwarded`, `M5 behavior change` → `Ranking order:` with version-free description.

---

## 4. Backward-Compatibility Strategy

1. **Aliases, not forks.** Every renamed export keeps a `/** @deprecated Use X. Removal in next major. */ export const Old = New` in same module. No logic duplication.
2. **Deprecation warnings** only for runtime-behavioral aliases (`packStringsToGPUBuffer`, `sanitizeStringForSlot`, `countUnicodeCodePoints`, `cpuAlgorithm: 'parity'`). Pure type/constant renames are silent aliases to avoid log spam.
3. **Changeset:** `minor` (new names + aliases), then `major` removes aliases + legacy `U2D3` write path (read stays). Document in `docs/snapshot-format.md`.
4. **Codemod:** `scripts/codemod-rename.mjs` (jscodeshift-light via `grep -l` + explicit map) applied to `apps/`, `examples/`, `scripts/` in same PR so internal consumers use canonical names on day one.
5. **Bundle gate:** update `check-bundle-size.ts` baseline after rename (aliases add ~1–2 KB); assert delta, not absolute, so rename itself cannot silently grow bundle.

---

## 5. Execution Phases

### Phase A — Freeze + lint gate (0.5 day)

* Add temp CI check failing on new leaks: `grep -rnE '\b[Mm][1-8]\b|v0\.[1234]|U2D[34]|U2F2|FORMAT_VERSION_4' packages/webgpu-search/src apps/*/src examples/ || true` → turn into `lint:naming` script.
* Snapshot current `test:all` green as baseline.

### Phase B — Library renames (2–3 days)

* B1: `text-profile.ts` constants + `buffer.ts` → `dataset-packing.ts` + `types.ts` unions. Keep aliases. Update `index.ts` exports (both names).
* B2: `hybrid-index` → `search-index`, `cpu-reference` → `exact-scorer`, `modes/` → `search/`, `filter/` → `filtering/`, `facets/` → `faceting/`, `extensions` → `hooks`, `context-manager` → `gpu-device-pool`, `persistence/idb` → `snapshot-*`, `fold-table` → `case-fold-table`, `unicode-preprocess` → `text-normalization`, `runtime-guards` → `guard`, `suggest` → `autocomplete`.
* B3: Strip all `v0.x`/`Mx` comments + `folded` → `normalized` internal rename (keep public `folded?: boolean` as deprecated alias where it appears in options). Update error strings (`legacy-ascii-v0.1` → `legacy-ascii`).
* Gate: `bun run check:shaders && bun run typecheck && bun run build && bun run test:headless`.

### Phase C — Scripts / CI / package.json (1 day)

* `git mv` all scripts per §3.3; update root `package.json`, `turbo.json` inputs, `.github/workflows/ci.yml` steps, `.agents/skills/*/SKILL.md` references, `docs/*.md` references.
* Rename env `M4_SHARD_INDEX/TOTAL` → `PARITY_SHARD_INDEX/TOTAL` with fallback reading old vars + warning.
* Rename `runM6Tests/runM8Tests`, `sm1/sm2/lim1/mmc` locals, `[bench-v04-matrix]` prefixes.
* Gate: full `test:all` (new names) + `lint:parity` + `check:bundle-size`.

### Phase D — Apps / examples / shaders (1 day)

* `apps/benchmark/src/main.ts`, `search.worker.ts`, `dataset.ts`: `U2F2` → `DATASET_MAGIC`, `M4 string-isolated` comment rewrite, `PHASE 1/2` → `Stage 1/2: Substring/Fuzzy`.
* WGSL: no logic change; only variable-comment cleanup if `m*` appears; re-run `vgpu check`.
* Examples (`react/vue/svelte/vanilla`): switch imports to canonical names.

### Phase E — Docs + root hygiene (0.5 day)

* Archive/delete per §3.4; rewrite README quickstart to use `SearchIndex`, `PackedDataset`, `snapshot` vocabulary; no `M*`/`v0*` in user docs.
* Add `docs/naming-conventions.md` (this plan §2 condensed) so future milestones cannot reintroduce temporal names.

### Phase F — Verification (per repo rule)

* Spawn verification subagent: `check:shaders`, `typecheck`, `build`, `test:headless` + new `lint:naming` (zero hits for ban regex) + `test:all` + clean-diff audit (no `*.png`/`*.json` at root, no `PLAN.md`/`ISSUE-*` at root, no `console.log` in lib).
* Publish `changeset minor` describing renames + aliases.

---

## 6. Acceptance Criteria

* [ ] `grep -rnE '\b[Mm][1-8]\b' packages/webgpu-search/src apps/*/src examples/ scripts/ --include='*.ts'` returns only historical `docs/archive/` hits.
* [ ] `grep -rnE 'U2D4|U2F2|FORMAT_VERSION_4|V2\b' packages/webgpu-search/src --include='*.ts'` returns only deprecated-alias lines + wire-comment explaining magic bytes.
* [ ] `ls *.png *.json render_*.js generate_*.py` at root → no such files (except `package.json`, `tsconfig.*`, `turbo.json`, `vercel.json`).
* [ ] `bun run typecheck`, `bun run build`, `bun run check:shaders`, `bun run test:all` green on new script names.
* [ ] Public `dist/index.d.ts` contains zero occurrences of `M1-M8`, `v0.`, `U2D`, `V2`, `hybrid`, `folded` (except `@deprecated` aliases).
* [ ] Old imports still work for one minor: `import { U2D4_MAGIC, PackedUnicodeBufferV2 } from 'webgpu-search'` compiles with deprecation warning, resolves to new symbols.

---

## 7. Risks

* **Breaking downstream imports** — mitigated by aliases + minor-then-major.
* **Snapshot interop** — magic bytes unchanged; old snapshots decode via `LEGACY_*` path; add round-trip test `legacy-snapshot → decode → encode → decode`.
* **Bundle growth from aliases** — mitigated by type-only aliases (erased) + single runtime alias objects; gate enforces delta cap.
* **Rename churn obscuring blame** — use `git mv` + one commit per phase (B/C/D/E), no logic edits in rename commits.
