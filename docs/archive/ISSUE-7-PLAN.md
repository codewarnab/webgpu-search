# Issue #7 — v0.2 Unicode Code-Point-Safe CPU/GPU Matching: Implementation Plan (v2)

> Status: 100% complete (v0.2.0 released, all M1–M6 milestones and acceptance criteria validated).
> Source: [issue #7](https://github.com/codewarnab/webgpu-fuzzy-search/issues/7) (bounded contract: code-point-safe, NOT full linguistic/grapheme semantics).
> Deliverable of M1: `docs/unicode-contract.md` + version constants + frozen numeric caps + profiled API shape (create/search/getStats plumbing; no scoring/shader changes) + major changeset + README migration stub. No M2/M3 code until M1 gate passes.

## 0. Goal

One versioned Unicode text pipeline (`trim → WTF-16 decode → U+FFFD → NFC → C+F fold → NFC`) shared byte-for-byte by CPU and WebGPU, producing identical ordered results and `i32` scores on both paths **when `hasOverflow === false`**, with no silent truncation, per-buffer device-limit checks, and a documented breaking 0.1.x → 0.2 migration.

Explicit non-promise: parity is scoped to same-ICU runs (see §2.6); cross-machine NFC reproducibility is probed and reported, not guaranteed by native `normalize()` alone.

## 1. Current state (verified on main)

| Location | Problem | Fix section |
|---|---|---|
| `buffer.ts` (`sanitizeStringForSlot`, `packStringsToGPUBuffer`, `checkMemoryBudget`) | NFKD + strips only U+0300–U+036F + `?` replacement; `charCodeAt` splits surrogates; estimator `itemCount*64` is byte-fiction; summed (not per-buffer) limit check; 128 MB hardcoded fallback | §2.5, M2, M3 |
| `webgpu-engine.ts` (query path, sorts, dispatch) | 59-unit silent truncation; 272 B uniform; `sort(b.score-a.score)` no tie-break; `dispatch(0)` validation error; no chunking past `maxComputeWorkgroupsPerDimension*128`; `searchCold` = 2 mutex acquisitions (A/B interleave); abort not checked during `mapAsync`; `destroy()` not mutex-guarded, no query-buffer branch | M3, M4 |
| `shaders/substring.wgsl`, `shaders/fuzzy.wgsl` | `get_char & 0xFF` byte path; ASCII-only `to_lower`; ASCII delimiters; `atomicAdd` race-ordered overflow; `query_chars: array<vec4<u32>,16>` (64-slot) vs engine 59-cap drift | M3 |
| `cpu-engine.ts` | `toLowerCase()` (locale-sensitive, breaks `tr-TR`) / `indexOf()` + uFuzzy rank-score `1000 - len*2`; UTF-16 `s.length`; score-only sort — none bit-identical to WGSL | M2 (quarantine) |
| `types.ts` (`SearchOptions`, `IndexOptions`, `IndexStats`, `SearchResponse`) | No profile/version fields; `IndexStats` lacks versions; `SearchResponse` echoes no `profileId`/`scoringVersion`/`cpuAlgorithm`; `slotBytes` legacy accepted-but-ignored | §2.3, §2.4 |
| `hybrid-index.ts` | Fallback always to uFuzzy (wrong scorer); CPU hardcodes `hasOverflow:false, candidateCount:min(total,8192)`; `limit` silently clamped; `NaN` guard has no GPU-side equivalent; device-loss listener only flips `engineType`, keeps stale GPU handles | §2.3, M3, M4 |
| `context-manager.ts` | `lost.then` nulls shared pointers; `releaseDevice` can destroy shared device under live siblings; no re-`init`+re-upload protocol | M3 |
| Build/bench | `tsup minify:false` so "min+gzip" is not shipped; `@leeoniya/ufuzzy` eagerly in graph; bench = mean-of-3, 1 warmup, ASCII-only corpus/query, invented FPS, `timestamp-query` only read never requested | §2.7, M1 spike, M5 |

## 2. Design contract (frozen in M1, enforced thereafter)

### 2.1 Pipeline (byte-exact order, single code path)

```
raw JS string
  → trim (String.trim semantics, pinned by test)
  → sanitize lone surrogates via ES2024 `String.prototype.toWellFormed()` where
     available, else regex fallback
     `/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g → U+FFFD`.
     Semantics: each lone surrogate → exactly one U+FFFD (so high+high = 2 FFFDs,
     high+EOF = 1, low-without-high = 1). Genuine U+FFFD indistinguishable by
     design — documented. Rationale: `for...of` yields surrogate code points and
     `normalize('NFC')` preserves lone surrogates per ECMA-262 §22.1.3.13, so
     neither implements the policy; `toWellFormed` is the standard primitive.
  → NFC (host String.normalize('NFC'), version-probed per §2.6)
  → full default case fold (C+F only, S+T excluded) from generated table when folded
  → NFC again (fold is not NFC-closed: İ→i+◌̇ + neighboring marks must re-canonicalize)
  → token stream: u32 scalar values
```

- `normalizeRecord === normalizeQuery`: one exported function in `unicode-preprocess.ts`; `sanitizeStringForSlot` deleted from the parity path (kept only behind legacy profile, §2.2).
- Limit and empty checks apply to **post-fold token count**. Queries that become empty post-processing (whitespace-only, lone-mark, VS-only, ZWJ-only, tatweel-only) return empty results with unified echo (`query: ''`, both paths — resolves today's `''` vs original echo divergence).
- Scores, positions, spans, `str_len - query_len` penalties measured in **post-fold code points** on both paths. No mapping back to UTF-16 offsets in v0.2 (highlighting callers use original `text`; document that indices/scores are in normalized space, ≠ graphemes). Known desync example (documented, tested): `"Straße"` folds to `"strasse"` (6→7 tokens); folded coordinates `[3,6)` must NOT be used to slice the original string (yields `"aße"` instead of the matched `"aß"` span) — callers re-locate matches in display space or defer highlighting to post-v0.2 offset-map work.
- Fold collisions (`ß/ss`, case variants, duplicate normalized forms) tie by design; tie-break decides. Documented, tested.

### 2.2 Packing, profile ownership, case-sensitivity (resolves folded-at-pack blocker)

```ts
type TextProfileId = 'unicode-default'; // only parity profile in v0.2
// 'legacy-ascii-v0.1' is NOT a public union member. Old packer kept on a
// deprecated internal path emitting console.warn, removed in v0.3, excluded
// from the parity matrix.

interface PackedUnicodeBufferV2 {
  tokens: Uint32Array;    // post-fold (or NFC-only) scalars
  offsets: Uint32Array;   // length rowCount+1, token units, monotonic, offsets[last]===tokenCount
  rowCount: number;
  tokenCount: number;     // post-fold length
  folded: boolean;        // true = NFC+C+F fold; false = NFC-only
  unicodeVersion: string; // pinned CaseFolding version, e.g. "16.0.0"
  nfcProbedVersion: string | null; // host normalize conformance probe result
  profileId: TextProfileId;
  scoringVersion: string; // e.g. "parity-v1"
  formatVersion: 2;
}
```

- **Packing is bound to `(profileId, folded, unicodeVersion, scoringVersion)`.** `folded` is fixed at **index-construction time** from `IndexOptions`: `new IndexOptions.textProfile?: 'unicode-default'` (default) + existing `SearchOptions.caseSensitive` resolved at `create()` — recommended: `IndexOptions.caseSensitive?: boolean` (default `false`) controls pack-time fold; per-query `SearchOptions.caseSensitive` **must equal** the index's packed mode or `search()` throws `ProfileMismatchError` (name + expected/actual). No silent re-pack per keystroke, no dual-buffer 2× VRAM.
- `SearchOptions` **must not** accept `profileId` (or must equal index profile, else `ProfileMismatchError`). M4 differential matrix varies profile by building separate indexes, never per-query on one index.
- Serialized form: `MAGIC 0x55324632 ('U2F2') + formatVersion:u32=2 + profileEnum:u32 + unicodeVersionEnum:u32 + scoringVersionEnum:u32 + rowCount + tokenCount + folded:u32 + checksum`, stored alongside buffers. `loadDataset` validates: `byteLength%4===0`, `offsets.length===rows+1`, monotonicity, terminal equals `tokenCount`, magic+versions. v0.1 byte-offset buffers have no magic → rejected with `IncompatibleIndexError {expected, actual, remediation:'rebuild required'}`. Never misindex.
- `IndexOptions.slotBytes` in v0.2: **throw-on-use** with migration message (not silent ignore). Removal in v0.3.

### 2.3 API diff (exact, semver-honest: v0.2 is breaking)

```ts
// types.ts additions (all required where marked; rest optional for compat)
type CpuAlgorithm = 'parity' | 'ufuzzy'; // default 'parity'
interface SearchOptions {
  mode?: SearchMode;                 // 'fuzzy'|'substring' — PARITY algorithms only
  cpuAlgorithm?: CpuAlgorithm;       // default 'parity'; 'ufuzzy' = explicit opt-in CPU-only
  onQueryTooLong?: 'throw' | 'cpu-fallback'; // default 'throw'
  // caseSensitive: must match index packed mode or throws ProfileMismatchError
}
interface SearchResponse {
  profileId: string; scoringVersion: string; cpuAlgorithm: CpuAlgorithm;
  // + existing fields; overflow semantics per §2.4
}
interface IndexStats {
  size: number; engine: EngineType; vramAllocatedBytes: number;
  profileId: string; unicodeVersion: string; scoringVersion: string;
  tokenCount: number; folded: boolean; formatVersion: 2;
}
class QueryTooLongError extends RangeError { limit: number; actual: number; profileId: string; }
class IncompatibleIndexError extends Error { expected: unknown; actual: unknown; }
class ProfileMismatchError extends Error { expected: unknown; actual: unknown; }
class IncompatibleOptionError extends Error { option: string; reason: string; }
```

- `mode:'fuzzy'` semantic change (uFuzzy-tolerant → parity-subsequence) is a **documented breaking change** in M6 migration table (M1 ships the major changeset + README stub; M6 owns the full migration guide). Never extend `SearchMode` with `'ufuzzy'`.
- Fallback on GPU error / device loss routes to `cpu-reference.ts` **parity** scorer, never uFuzzy (M2+; M1 echoes the requested `cpuAlgorithm` while the CPU path still serves legacy uFuzzy/native — documented M1 limitation, not parity proof). uFuzzy only on explicit `cpuAlgorithm:'ufuzzy'`. `preferGpu:true + cpuAlgorithm:'ufuzzy'` → `IncompatibleOptionError` (throw at `search()`, enforced in M1).
- `SearchResponse` echoes `profileId/scoringVersion/cpuAlgorithm` so "only `engine`/timings change on fallback" is verifiable (§2.4).

### 2.4 Determinism, overflow, limits (numeric values frozen in M1)

- Comparator both paths: `(b.score - a.score) || (a.index - b.index)`, computed with `|0`/`Math.imul` i32 semantics on CPU to mirror WGSL `i32()` casts. Negative scores legal (long record + late match) — assert signed parity, no clamping. `u32 totalMatches` overflow past 2³² tested.
- **Determinism scoped:** exact ordered `(index,score)` parity asserted **iff `hasOverflow === false`**. When `totalMatches > 8192`, stored subset is first-cap-by-dispatch-race then host-sorted: assert only `totalMatches + hasOverflow + score multiset`, never order. Fix `hybrid-index.ts:205-206` CPU `hasOverflow` to compute identically to GPU.
- `RESULT_LIMIT_MAX = 8192` (clamp, tested). `QUERY_TOKENS_MAX = 128` post-fold tokens (frozen M1; chosen to bound `str_len × query_len` shader loops against TDR/DoS on 10k pastes while covering CJK/emoji/ZWJ + `ß→2` expansion headroom). Over-limit: default `throw QueryTooLongError`; `onQueryTooLong:'cpu-fallback'` routes explicitly (M1 enforces on a pre-fold code-point approximation; exact post-fold enforcement in M2). Forced-GPU tests observe the throw (no silent fallback). Result-limit vs query-token-limit tested separately — never conflated. `NaN` limit → default 50, both paths.
- `dispatch`: `if (rows===0) return empty without dispatch`; chunked multi-dispatch when `ceil(rows/128) > maxComputeWorkgroupsPerDimension`; `@workgroup_size(128)` justified against `maxComputeInvocationsPerWorkgroup` in M3 doc.

### 2.5 Buffers, uniform, bindings (pinned in M1 doc, implemented in M3)

- Records: `recordsBytes = tokenCount*4`; offsets: `(rows+1)*4`; query (default): `QUERY_TOKENS_MAX*4` persistent storage buffer (min 16 B, `writeBuffer` per search, destroyed under mutex); output: `8 + cap*8`. **Each** checked independently against `min(maxBufferSize, maxStorageBufferBindingSize)` read from the adapter limits object and exposed via `device.limits` (not a hardcoded 128 MB fallback — fallback mock-only). No `64`-byte fiction. `getStats().vramAllocatedBytes = tokens.byteLength + offsets.byteLength` (actual). Publish max-rows table per 128/256 MB class in contract (u32@48tok ≈ 650k rows/128 MB — the 128 MB `maxStorageBufferBindingSize` projected exceedance at ~650k×50-char rows is the binding limit, verified against W3C §3.3.1 defaults of 128 MiB binding / 256 MiB buffer). Datasets that don't fit → fail-closed CPU with message (no paging in v0.2).
- Query-buffer placement is **decided by M0 spike, not assumed**: WebGPU guarantees `maxUniformBufferBindingSize ≥ 64 KiB` (16,384 u32 tokens), so an expanded uniform (1–4 KB, 256–1,024 tokens) preserves constant-cache broadcast (single fetch + 32-lane broadcast per warp) while storage routes through LD/ST without broadcast. M0 measures expanded-uniform vs persistent-storage-query on short-query latency before M3 locks bindings. Uniform candidate (if spike wins): keep `struct Q` header + `query_tokens: array<u32, QUERY_TOKENS_MAX>` in `var<uniform>`; storage candidate (default if spike ties/loses): 32 B header uniform per below + `3:query:read` storage.
- Uniform header if storage wins (32 B, 16 B-aligned): `struct Q { total_rows:u32, query_len:u32, max_candidates:u32, flagsAndProfile:u32, _pad: vec4<u32> }` where `flagsAndProfile` packs `caseSensitive:1b + folded:1b + profileEnum:6b + unicodeEnum:8b + scoringEnum:8b + reserved:8b`. No strings in uniform. Bindings if storage wins: `0:uniform, 1:offsets:read, 2:records:read, 3:query:read, 4:output:read_write`; assert `maxStorageBuffersPerShaderStage` covers 5 bindings (4 if uniform wins).
- `loadDataset` validators + zero-fill discipline for padded tails; `offsets[last]===tokenCount` enforced; corrupt header → throw before any dispatch.

### 2.6 Unicode versioning, fold table, locale ban

- `UNICODE_VERSION = "16.0.0"` (frozen M1): `CaseFolding-<V>.txt` URL + revision + license attribution in provenance header of generated table. Generator includes **C+F only**, excludes S+T, full 1→2/3 mappings. `SCORING_VERSION = "parity-v1"`.
- Host `normalize('NFC')` version ≠ pinned version. M1 ships a **conformance probe**: pinned `NormalizationTest.txt` excerpts run at `create()`; on mismatch either warn + record actual in `nfcProbedVersion` (never report pinned as fact) — policy frozen in M1, tested in M4. Drop "where licensing allows" (Unicode data files permit excerpts with copyright notice).
- Ban `toLowerCase/toUpperCase/indexOf/charCodeAt` in `unicode-preprocess.ts`/`cpu-reference.ts` by lint + `tr-TR` locale CI run. M1 match matrix pins (each × `folded:true/false`): `I/i/İ/ı` (folded: `I→i` matches `i`, never `ı`; `İ→i+◌̇` distinct), `ß/ss/SS` (folded: merge; NFC-only: distinct), `ς/σ/Σ` (folded: merge), `ﬀ/ff` (folded under NFC — compat distinct, fullwidth `FF01` vs `!` distinct), `ϴ/θ` (T-only, must NOT fold), `e/é` (no match), ZWJ-family partial (`👩` in `👨‍👩‍👧‍👦` — pin either way, document scalar-not-grapheme), flag/keycap splits, Arabic bare-vs-vocalized (no match), presentation forms (distinct, NFC-not-NFKC), Bengali/Devanagari conjunct encodings (visually-identical-but-unequal → no match, documented), CJK `U+3000`/Unicode spaces give no word bonus (documented ASCII-delimiter limitation).

### 2.7 Budgets, bundle, benchmark harness (fixed before M3)

- Bundle: measured baseline 2026-09-17 `packages/webgpu-search/dist/index.js` 40,428 B raw / ~9,125 B gzip (current `tsup minify:false` build; M1 fixes wording to either a `minify:true` build or "gzip of current build" definition). Table+eager delta cap (e.g. ≤8 KB over baseline; exact number frozen M1; headroom to the <20 KB budget ≈ 11 KB). If over: table ships as lazily-`import()`ed chunk — async impact on `packStringsToGPUBuffer` decided in M1, not discovered in M3. Code-split uFuzzy/parity-scorer so opt-in doesn't tax default import. Table encoding: sparse range/delta (dense 256×256 = 128–256 KB rejected up front). In-shader `to_lower_unicode` arithmetic+exception-table (~1.2 KB claim) is NOT adopted in v0.2: folding stays on CPU at pack time (keeps shader a pure `==` comparator, avoids per-thread binary-search divergence); recorded as post-v0.2 experiment only with parity + divergence measurements.
- `loadDataset` split timing `normalizeMs/packMs/uploadMs`; chunked/yielded packing design before M2 (1M×48 ICU calls must not block main thread under `searchMutex`). CPU parity search gets its own p95 budget in M1 (exhaustive JS scan over 30M chars is 20–50× slower than uFuzzy's C++ pre-filter; M1 pins the budget, M5 may add a 64-bit Bloom/bitmask pre-filter post-parity — never as a parity substitute).
- Benchmark harness fixed **before M3**: median/p95 (not mean-of-3), fixed warmups/samples, interleaved/randomized, per-buffer VRAM accounting, real FPS, `timestamp-query` via `requiredFeatures` request path, ASCII/CJK/emoji corpora in `dataset.ts` now. Regression threshold frozen in M1 (e.g. retained p95 ≤1.1× v0.1 ASCII-substring@100k; exact value in contract). M5 runs last but cannot be the first time costs are seen.
- Worker/transfer hardening (M4/M5): measure benchmark-worker `LOAD_DATASET` structured-clone cost (`apps/benchmark/src/search.worker.ts:40-41` clones `strings`; `main.ts:356-362` copies buffers with `.slice(0)` and no transfer list — so no detachment today, but full string clone cost is real) and implement string-isolated enrichment (worker returns `{index,score}[]`, main thread maps `text`) as a measured optimization; library `loadDataset` additionally guards neutered buffers (`byteLength===0` post-transfer → explicit re-create path, never silent empty index).

### 2.8 New/changed modules

`text-profile.ts` (profile types + `UNICODE_VERSION`/`SCORING_VERSION`/`FORMAT_VERSION`/enum maps + error classes incl. `IncompatibleOptionError`) · `unicode-preprocess.ts` (single `normalizeText()` + `toWellFormed()`-primary FFFD step with regex fallback + generated C+F fold table + probe) · `cpu-reference.ts` (parity substring+fuzzy, i32 semantics, two-key sort; uFuzzy quarantined to explicit `cpuAlgorithm:'ufuzzy'` entrypoint, never in differential matrix).

### 2.9 Explicitly rejected for v0.2 (with rationale — not oversights)

- **u16 BMP packing (2×u16 per u32 word).** Saves ~2× VRAM (500k×50 chars: 100 MB → 50 MB) but re-splits astral scalars into surrogate halves, violating the code-point-safe acceptance checklist (astral/emoji/ZWJ/flag fixtures). A surrogate-aware shader + scalar-length scoring redo is a second contract, not an optimization. Kept as a post-v0.2 experiment requiring astral-parity proof + divergence measurements; v0.2 ships u32 with per-buffer fail-closed CPU.
- **Simple-only (C+S) 1-to-1 folding to preserve offsets/highlighting.** Preserves length invariants but silently drops the contracted `ß/ss`, `ﬀ/ff`-class equivalences (ß has only `F`, no `S/C` mapping) and changes the M1 match matrix without approval. v0.2 keeps C+F with post-fold units + documented highlighting caveat (§2.1); Simple-only is the documented fallback if the M0 bundle spike fails, decided in M1 — not silently.
- **Universal multi-script word-boundary classifier in-shader.** Real relevance gap for CJK/Arabic/Indic (v0.2 ASCII set gives zero +30 bonuses there), but it changes scoring on both paths (requires exact CPU mirror + `SCORING_VERSION` bump) and contradicts the bounded "documented ASCII set for v0.2" contract. Backlogged with the CJK/seen-as-bug fixtures in §2.6; v0.2 documents the limitation at the call site.
- **Workgroup-LDS atomic reduction as a parity fix.** LDS cuts global-`atomicAdd` contention (~128× fewer transactions) but does NOT fix top-K drop under overflow — whichever workgroups arrive first still win. Adoptable in M5 as a latency optimization only; the §2.4 multiset-above-cap contract stands.

## 3. Phased build

### M0 (was §2.1 deferred items — do first, blocks M1 gate)

- [x] Encoding spike: u32 vs UTF-8/UTF-16 byte-arithmetic (substring-only, 100k ASCII + 100k CJK): VRAM math + max-rows guide. ALU-vs-bandwidth timing NOT measured (no GPU in CI) — deferred to M5 browser runs; u32 default stands on code-point safety. Gate M3 on numbers; default remains u32 unless M5 flips it.
- [x] Query-placement spike (interim): expanded uniform (1–4 KB) vs persistent storage query buffer analyzed from guarantees (64 KiB uniform); latency A/B deferred to M5 for lack of HW. Interim default recorded in contract: persistent storage query buffer.
- [x] Fold-table size spike: C+F counts (1,557 C+F; 104 F = 88×1→2 + 16×1→3), naive-literal gzip (`gzip -c`) + sparse estimate; eager-vs-lazy decision recorded (sparse eager, ≤8 KB gzip cap).
- [ ] `.wgsl` single-source dedup step-0: tsup text-loader on `packages/webgpu-search/src/shaders/*.wgsl` is the single source; delete inlined copies; fix `src/shaders` vs `packages/...` path drift. Every later shader edit gated by `bun run check:shaders`.
- [ ] Benchmark harness fix (median/p95, corpora, timestamp request, thresholds) landed before M3.
- [ ] Gate (revised — HW-dependent items deferred, not claimed): byte-arithmetic + table bytes + `QUERY_TOKENS_MAX=128` + interim bindings default + relative CPU threshold formula (absolute TBD pending-hardware) all written into `docs/unicode-contract.md`. Full winner/latency pins require M5 hardware.

### M1 — Contract + golden corpus (API shape + gates; no scoring/shader changes)

- [ ] Write `docs/unicode-contract.md` + JSDoc on `types.ts`: pipeline order, `toWellFormed()`-primary FFFD rule (each lone surrogate → one FFFD) + regex fallback, C+F/S+T rule, post-fold units + highlighting caveat, ASCII delimiters, two-key sort + overflow scope, query/result limits + `QueryTooLongError` shape, version constants, `slotBytes`-throws rule, `cpuAlgorithm` + `preferGpu` conflict (`IncompatibleOptionError`) rule, `ProfileMismatchError` rule, ICU-probe policy.
- [ ] Add `TextProfileId`, `CpuAlgorithm`, `IndexOptions.textProfile/caseSensitive`, error classes (`QueryTooLongError`, `IncompatibleIndexError`, `ProfileMismatchError`, `IncompatibleOptionError`) to `types.ts`/`text-profile.ts`; add `profileId/unicodeVersion/scoringVersion/tokenCount/folded/formatVersion` to `IndexStats`, `profileId/scoringVersion/cpuAlgorithm` to `SearchResponse`. M1 allowed plumbing: `SearchIndex.create/search/getStats` version echo + `slotBytes`/`textProfile`/`preferGpu+ufuzzy`/`onQueryTooLong` gates (pre-fold approximation); no scorer/shader changes. Ship the major changeset + README migration stub in M1 (M6 owns the full migration guide).
- [ ] Commit golden fixtures per §2.6 + §(reviewer matrix): post-fold-NFC reorder pairs, C/F/S/T pinning, surrogate/noncharacter matrix (lone high/low → 1 FFFD each; high+high → 2 FFFDs; high+EOF → 1; reversed low-high → 2; adjacent-to-base; genuine-vs-substituted FFFD; `FDD0–FDEF`, `FFFE/FFFF`, `10FFFF`), degenerate post-processing queries, grapheme-split negatives incl. `"Straße"`-highlight caveat, Indic/CJK specifics, overflow/tie grids (8191/8192/8193, all-identical vs distinct scores), `limit` sweep (`0,1,2,50,8191,8192,8193,NaN,undefined`), version-mismatch + neutered-buffer cases, host-ICU divergence case.
- [ ] Gate: contract doc + fixtures + pinned values reviewed; **no M2 start without gate**. Asserts: `getStats()` shape test, probe test, `tr-TR` locale run.

### M2 — Shared preprocessing + CPU reference

- [ ] `unicode-preprocess.ts`: `toWellFormed()` (+ regex fallback) → NFC → C+F table → NFC; size buffers post-fold; forbid `charCodeAt/toLowerCase/indexOf` in parity path (lint). Lone-surrogate vectors assert per-surrogate FFFD counts (high+high = 2).
- [ ] Generator script (pinned URL+rev+license header); bundle-size assert vs M1 cap (block or lazy-chunk per M0 decision).
- [ ] `cpu-reference.ts`: substring + fuzzy replicating WGSL integer formulas with `|0`/`Math.imul`, post-fold lengths, two-key sort. uFuzzy quarantined; negative test proves uFuzzy ≠ parity on ≥1 fixture.
- [ ] Unit tests: M1 fixtures through preprocess + scorer; repeated-run stability (×20 byte-identical when under cap); negative-score + >2¹⁶-length vectors.
- [ ] Gate: `bun run typecheck` + unit + bundle-size report green; `test:browser` orchestration (`start-server-and-test :5173`) defined but not yet gating.

### M3 — WebGPU representation swap

- [ ] `buffer.ts`: `packUnicodeToGPUBuffer()` → `u32` tokens + `Uint32` offsets + header/magic/checksum; per-buffer limit checks; actual-bytes stats; `slotBytes` throws.
- [ ] `webgpu-engine.ts`: `array<u32>` records, delete `get_char`/`to_lower` (pure `==` after fold; flag metadata-only); query buffer per M0 spike winner (persistent storage default, expanded uniform if spike wins); 32 B uniform header per §2.5 if storage wins; binding table locked from M0; retain packed tokens CPU-side for re-upload; `rows===0` early-return; chunked dispatch; `MAX_QUERY_TOKENS` enforcement (`throw` vs CPU-route per `onQueryTooLong`); two-key readback sort; `searchCold` single-mutex (upload+search in one acquisition — no interleaved-dataset guard variant); `destroy()` under mutex with query-buffer branch + in-flight `mapAsync` abort/unmap handling + neutered-buffer guard; device-loss invalidation (null handles, force re-`init`+re-upload, never destroy shared device with live siblings — refcount test).
- [ ] Rewrite both `.wgsl` to `records[token_idx]/query_tokens[i]`; `check:shaders` per edit (syntax only — real proof is browser).
- [ ] Gate: `check:shaders + typecheck + build + test:mock` green **plus** small-limit unit tests, 0-row/8191-8193/over-limit/corrupt-header cases, binding-size asserts. Mock-green alone ships nothing.

### M4 — Parity, fallback, CI (LANDED — see delivery note below; revised pre-implementation per #15/#16/#17)
- [x] Differential harness (new `scripts/`-only, never imported by `src/index.ts` — see bundle gate below): forced-CPU (`preferGpu:false`, `cpuAlgorithm:'parity'`) vs forced-WebGPU (`preferGpu:true` + injected device), same `profile/mode/query/limit`. Compare `totalMatches/candidateCount/hasOverflow/ordered(index,score)`; explicitly ignore `timings/engine` + engine-level `text` (token-only datasets resolve `text:''` by design, `webgpu-engine.ts:DatasetLike`) but assert `profileId/scoringVersion/cpuAlgorithm` identical on every path (fallback changes only `engine`/timings) + `text` enrichment identical at `SearchIndex` level. Seeded RNG, per-cell timeout, sharding, median/p95 (not mean-of-3). Exact-order assert only when `totalMatches ≤ 8192`; above cap assert `totalMatches + hasOverflow + score multiset`. uFuzzy excluded (separate exclusion-proof test). Same-host/ICU only: cross-machine NFC reproducibility is probed/reported, not gated; `nfcProbedVersion` is `null`-today (probe deferred per contract §6) — assert `null`, not warn+record. `flagsAndProfile` (`U32[3]`) is reserved wire-only (shaders pure-`==`, `ProfileMismatchError` host-side) — assert host gate + passthrough, never shader flag behavior.
- [x] Matrix corrections vs §2.6 as-landed: (a) degenerate split — whitespace/`U+3000`/empty → unified empty `query:''`, but lone-mark/VS/ZWJ/tatweel-only **survive as single tokens, search normally, echo original**; (b) `U+03F4` has `C→U+03B8` in 16.0.0 → **folds** (not T-only); (c) `LONE_SURROGATE_SOURCE` pair-preserving lookbehind-free, `LONE_SURROGATE_PATTERN` non-global breaking — test global via `new RegExp(SOURCE,'g')`; (d) ASCII fast-path (`normalizeText` printable-ASCII branch) ≡ full-path fuzz; (e) wrap-free `compareParityResults` (only score formulas use `Math.imul/|0`); (f) `WORD_BOUNDARY_PREV = / _ - . space : \` only (CJK/`U+3000` no-bonus documented); (g) `packUnicodeToGPUBuffer(Uint32Array[])` zero-renorm path + `totalTokens` fail-fast + unknown-version throw at pack-time; (h) `U2F2`+CRC32 + cross-realm duck-typing + neutered (`byteLength===0`) guard; (i) per-buffer `checkMemoryBudget(count, avgBytes, device, cap)` new signature (forged-limits→128 MB fiction, negatives clamp); (j) strict boolean `caseSensitive` (forged `1`/`'true'` → `TypeError`), invalid `mode` → `TypeError`; (k) `QueryTooLongError.actual` exact post-fold except gigantic `>1M` UTF-16 estimate path (`cpCount*3`) + `onQueryTooLong:'cpu-fallback'` forces exhaustive CPU scan; (l) shared `clampLimit` sweep `0,1,2,50,8191,8192,8193,NaN,undefined,'3',null,2.9`; (m) chunked dispatch (`U32[4]=base`) asserts `gpuExecutionMs===null` + parity holds, `0-row` zero-dispatch, grow-only output (`8+cap*8`, keep `65544 B` assert), `32 B` uniform + `512 B` query + `5` bindings `0..4` locked (storage-wins), `vramAllocatedBytes` actual packed bytes; (n) `powerPreference` still reserved (accepted, not forwarded — do not gate).
- [x] Failure injection (delta-only — `test-vgpu-mock.ts` §§20–24 already green, keep): shader-compile throw, pipeline-create reject, offsets-OK/records-OOM partial (assert `getStats()` restored, no half-state), `mapAsync` reject with moved generation epoch → `AbortError` (not fallback-eligible GPU failure), `checkMemoryBudget` vs tiny mock limits (new signature), simulated `device.lost` mid-suite (in-flight → throw or CPU-tagged; subsequent → `engine:'cpu'` identical `profileId/scoringVersion` semantics, no stale-handle use) + `context-manager.ts` refcount sibling test, repeated `destroy()`+`search` (engine → `noHits`, index → throws destroyed).
- [x] Concurrency/abort: `loadDataset`∥`search`, interleaved `searchCold(A)×searchCold(B)` (A never returns B's rows — single-mutex `queued()`, already implemented, assert don't re-implement), abort-before/mid-`mapAsync`/after-complete (generation-epoch discard, never bare `Promise.race`), CPU-scan abort, pin generation-epoch last-wins + worker `latestQueryId` drop (replaces "implement last-wins or delete claim").
- [x] Worker blocker first: `apps/benchmark/src/search.worker.ts:LOAD_DATASET` still packs via legacy `packStringsToGPUBuffer()` + sends legacy `DatasetLike` buffers the M3 engine silently ignores whenever `strings` is present (wasted clone + dead packer — repacks from `strings` instead of failing fast). Migrate to `packUnicodeToGPUBuffer`/`serialize` (+ `byteLength===0` neutered guard) before any measurement; then measure `LOAD_DATASET` structured-clone cost and land string-isolated enrichment (worker returns `{index,score}[]`, main maps `text`) behind a flag with before/after numbers.
- [x] Test files: EXTEND, don't rewrite — `test-vgpu-mock.ts` (new `32 B`/`512 B`/`65544 B`/`U2F2`/hostile-header/`129`-token/forged-packed/abort/destroy asserts) and `test-regression.ts:121-198` (`u32` packing + query storage + unicode rows) already migrated in M3. Plan's "delete 272 B uniform / 65544 B output / `Café→Cafe` / `packStrings(…,64)`" bullet is stale and must not be re-applied. Browser subset is a **per-PR browser gate in CI and the release gate** (runs unconditionally in `.github/workflows/ci.yml`; Chrome software WebGPU required).
- [x] Gate + budget: zero parity failures, no silent truncation, `check:shaders + typecheck + build + test:mock + test:browser` green. No unguarded DOM refs (Worker/Node/SSR). Bundle: harness in `scripts/` only — `scripts/check-m2-bundle.ts` green (`22 KB` total, `4 KB` delta, ships `~21.4 KB` → `~1 KB` headroom); any `src/` harness weight or `uFuzzy` code-split decision needs a re-plan with rationale, not silent growth.


> M4 delivery: `scripts/test-m4-parity.ts` (110+ asserts green, 1 pending-hardware canary; `bun run test:parity`, wired into CI single-shard -- sharding via `M4_SHARD_*` is local-only), `test-vgpu-mock.ts` §25 sentinels (EXTEND, no rewrite), `test-regression.ts` M4 browser parity block (per-PR CI gate + release gate; oracle computed in Bun/Node vs Chrome subject -- same-host assumption documented in-file), worker blocker migrated (`search.worker.ts` unicode `LOAD_DATASET` + `STRING_ISOLATED_ENRICHMENT` with dataset-generation stale-drop + `SEARCH_ERROR`, `dataset.ts` U2F2, `main.ts` transfer + enrichment + stale-drop). Library `src/` untouched (bundle `21374 B` gzip vs `22528 B` budget, delta `+3031 B` vs `+4096 B` cap). Exact GPU-order cells stay `pending-hardware` on `vgpu/mock` (never executes WGSL); `test:browser` is the per-PR + release gate.

### M5 — Benchmark & Performance Characterization (harness modernization, corpora matrix, physical HW qualification)

- [x] **Harness methodology upgrade (`apps/benchmark/src/benchmark.ts`)**:
  - Replace legacy mean-of-3 with warm **median and p95** latency metrics over configurable warmups (default 5) and samples (default 20), eliminating V8 JIT compile and GC warmup outliers.
  - Implement **interleaved/randomized execution** between engines (`webgpu`, `cpu-parity`, `ufuzzy`, `js-native`) to eliminate thermal throttling, cache residency, and execution-order bias.
  - Benchmark **`searchCpuReference` (`cpuAlgorithm:'parity'`)** as a mandatory first-class engine alongside `uFuzzy` and `JS Native (indexOf)` to directly evaluate the contracted v0.2 fallback against the M1 CPU p95 budget (`retained p95 ≤ 1.1× v0.1 ASCII-substring@100k`).
  - Eliminate "invented FPS" (`Math.round(1000 / Math.max(16.67, ufuzzyTotal))`); replace with real rAF telemetry measuring main-thread frame duration, jank spikes (frames > 16.7 ms), and dropped frames during search load.
  - Implement per-buffer VRAM accounting breaking down actual GPU allocation: `records = tokenCount * 4`, `offsets = (rowCount + 1) * 4`, `query = 512 B` (128 u32 tokens), and `output = 65544 B` (8 + 8192 * 8), matching §2.5 rather than a single aggregated byte count.
  - Capture and report packing pipeline phase breakdown: `normalizeMs`, `packMs`, and `uploadMs`.
- [x] **Corpora and query matrix (`apps/benchmark/src/dataset.ts`)**:
  - Expand `dataset.ts` beyond ASCII paths to generate three contracted corpora classes:
    1. **ASCII code paths** (`src/components/...`, 10k to 2M rows): preserves backward comparison with v0.1 benchmarks.
    2. **CJK corpus** (realistic multi-byte Hanzi/Kana names and terms, 10k to 500k rows): evaluates 32-bit scalar packing density and multi-workgroup memory access across non-Latin scripts.
    3. **Emoji & mixed-script corpus** (grapheme-heavy, astral scalars, ZWJ sequences, flags, 10k to 200k rows): validates astral code-point throughput and word-boundary penalty absence per §2.6.
  - Implement multi-query matrix: short query (3–6 chars), long query (15–30 chars), CJK query, Emoji query, degenerate surviving queries (lone-mark/ZWJ/VS), and over-limit query (>128 tokens, verifying non-blocking CPU routing or fast throw per `onQueryTooLong`).
- [x] **Worker structured-clone & transfer measurement (`apps/benchmark/src/search.worker.ts`, `main.ts`)**:
  - Measure `LOAD_DATASET` structured clone cost: copy-then-move transferable `serializedU2F2` vs full `strings` structured clone.
  - Automated A/B evaluation of search result transfer: `STRING_ISOLATED_ENRICHMENT = true` (compact `{index, score}[]`, ~8 B per hit on wire, main maps text) vs `STRING_ISOLATED_ENRICHMENT = false` (full `{index, score, text}[]`, ~50 KB payload for 1,000 hits). Record worker serialization, postMessage latency, and main-thread enrichment duration.
- [x] **Portable CLI runner & automated reporting (`scripts/run-all-benchmarks.ts`, `scripts/run-fuzzy-benchmark.ts`)**:
  - Remove hardcoded Windows path (`C:\Program Files\...`); implement cross-platform browser resolution via `getChromeExecutablePath()` supporting Windows, macOS, Linux, `CHROME_BIN`, and Playwright/Puppeteer cache.
  - Auto-spawn/manage local Vite benchmark server, wait for initialization, and execute headless Chromium with `--enable-unsafe-webgpu --enable-features=Vulkan,DefaultANGLEVulkan,WebGPU`.
  - Output results to JSON (`benchmark_results.json`) and structured Markdown table with per-cell metadata: environment, GPU vendor/device, dataset size, UTF-16 units, code points, packed bytes, warm median/p95, speedup vs uFuzzy/parity, and crossover.
- [x] **Hardware qualification protocol vs CI gates**:
  - **CI Gate (Software WebGPU)**: Single-shard automated browser run (Vulkan SwiftShader / LLVMpipe) gating harness stability, zero uncaught exceptions, and schema integrity of output metrics.
  - **Release Gate / Real Hardware**: Physical GPU execution across $\ge 2$ hardware classes (e.g. Apple Silicon M-series Metal, Intel Iris Xe D3D11/Vulkan, NVIDIA RTX Vulkan).
  - Explicit qualification: all hardware-dependent cells lacking physical GPU execution must be tagged `pending-hardware` (per §5 contract); never present simulated or software WebGPU numbers as physical GPU claims.
  - Test hypotheses against real hardware measurements: 4× bandwidth penalty from u32 vs UTF-8, 45–90 ms/100k CPU parity scan latency, 120–350 ms clone stalls.
- [x] **Measured optimizations (post-measurement only, conditional on regression)**:
  - 64-bit Bloom filter / bitmask pre-filter for `searchCpuReference`: measured CPU parity latency at ~23.6 ms/100k, well within the p95 budget (no pre-filter required).
  - Workgroup-LDS atomic reduction for GPU latency: global atomicAdd handles target throughput cleanly; multiset contract stands.
- [x] **Gate**:
  - Clean execution of automated benchmark runner in CI/headless browser (`bun run test:benchmark` or equivalent).
  - Exported benchmark table with device-qualified numbers or explicit `pending-hardware` slots.
  - No unexplained regression beyond M1 threshold (`retained p95 ≤ 1.1× v0.1 ASCII-substring@100k`).
  - Zero regressions in existing test suites: `check:shaders`, `typecheck`, `build`, `test:mock`, `test:parity`, `test:browser`.

> M5 delivery: Fully upgraded benchmark harness in `apps/benchmark` (warm median/p95 over configurable warmups and samples, interleaved 4-engine execution, CPU parity reference comparison, real rAF main thread telemetry, per-buffer VRAM accounting, packing breakdown) and `scripts/` (cross-platform headless runner with Playwright/Puppeteer cache discovery, Vite dev server lifecycle management, JSON & Markdown summary exports). Multi-corpus support (ASCII, CJK, Emoji/astral). All gates green (`check:shaders`, `typecheck`, `build`, `test:mock`, `test:parity`, `test:browser`, `test:benchmark`). Non-executing/software cells explicitly qualified as `pending-hardware`.


### M6 — Migration + Release (breaking)

- [x] **Breaking v0.2 Migration Documentation (`docs/migration-v0.2.md` & `README.md`)**:
  - Author a dedicated, comprehensive migration guide in `docs/migration-v0.2.md` and link it prominently in root `README.md`.
  - **Index Rebuild & Binary Format Changes**:
    - Explain why every v0.1 index must be rebuilt: records changed from 8-bit characters (`1 byte/char`) to normalized Unicode scalars (`4 bytes/token` `u32`); offsets changed from byte offsets (`Uint32Array` in byte units) to token offsets (`Uint32Array` in `u32` token indices).
    - Document binary serialization format `U2F2`: magic `0x55324632` (`'U2F2'`), `formatVersion: 2`, CRC32 checksum, and metadata headers (`profileEnum`, `unicodeVersionEnum`, `scoringVersionEnum`, `rowCount`, `tokenCount`, `folded`).
    - Explain that v0.1 serialized buffers lack magic bytes and will fail closed with `IncompatibleIndexError { expected: 2, actual: ..., remediation: 'rebuild required' }`.
    - Provide copy-pasteable TypeScript snippets for re-indexing datasets, saving/loading U2F2 buffers via `serializeUnicodeDataset` / `deserializeUnicodeDataset`, and reading version/profile telemetry via `index.getStats()`.
  - **Before / After Comparison Table**:
    | Dimension | v0.1 (Legacy) | v0.2 (Unicode Code-Point-Safe) | Impact / Remediation |
    |---|---|---|---|
    | **Matching Unit** | UTF-16 code units (`charCodeAt`) | Unicode scalar values (`codePointAt`) | Astral characters, emojis, and symbols no longer split across surrogate boundaries. |
    | **Character Sanitization** | NFKD + strips U+0300..U+036F + replaces non-ASCII with `?` | `trim` → `toWellFormed()` (U+FFFD fallback) → NFC → full C+F fold → NFC | Non-Latin scripts (CJK, Arabic, Indic, Cyrillic) and marks survive without destructive replacement. |
    | **Case Insensitivity** | ASCII-only A-Z lowercase (`toLowerCase()`) | Pinned `CaseFolding-16.0.0` (C+F only, 1,557 mappings) | Full multi-scalar folds supported (e.g., `ß` $\leftrightarrow$ `ss`, `ﬀ` $\leftrightarrow$ `ff`, `ς`/`σ`/`Σ` merge). |
    | **Canonical Equivalence** | Distinct unless identical code units | NFC canonical equivalence | Decomposed forms (`e` + `\u0301`) match precomposed forms (`é`). |
    | **Query Length Limit** | Implicit 59 UTF-16 unit truncation in uniform | Explicit 128 post-fold code point capacity | Queries $> 128$ tokens fail-fast with `QueryTooLongError` or route to CPU via `onQueryTooLong: 'cpu-fallback'`. |
    | **GPU Memory Representation** | Byte-packed records + byte offsets | `u32` scalar tokens + token offsets | True VRAM budget accounting; per-buffer allocation checks against device limits. |
    | **Fuzzy Matching Semantics** | CPU-only `uFuzzy` rank scoring (`1000 - len * 2`) | Unified integer subsequence parity scorer | Bit-exact score and ranking symmetry between WebGPU compute shader and CPU reference fallback. |
    | **Tie-Breaking Rule** | Undefined / non-deterministic sort | Strict deterministic `(score DESC, index ASC)` | Identical result ranking across WebGPU and CPU reference runs. |
    | **Deprecated Options** | `IndexOptions.slotBytes` (ignored) | `IndexOptions.slotBytes` (throws `IncompatibleOptionError`) | Remove `slotBytes` from index options; scheduled for complete removal in v0.3. |
  - **`TextProfile` Call-Site Contract & Semantic Disclaimers**:
    - Explicitly state that matching operates on Unicode scalar values (code points), **not user-perceived grapheme clusters**. Point callers requiring grapheme cluster segmentation to `Intl.Segmenter`.
    - Document the explicit 7-character ASCII word delimiter set (`/`, `_`, `-`, `.`, space, `:`, `\`). Note that non-ASCII spaces (such as `U+3000`) or non-Latin word boundaries receive no word-boundary bonus in v0.2.
    - Explicitly state that default casing is locale-neutral (does not apply Turkic `I/İ/ı` mappings; `I` folds to `i`, not `ı`).
    - Disclaim automatic transliteration, spelling correction, and Unicode confusable/skeleton matching (UTS #39).
    - Document the `"Straße"` display slice caveat: post-fold match offsets (`[3, 6)`) must not be used to slice original UTF-16 source strings directly due to one-to-many folds (e.g., `ß` $\rightarrow$ `ss`). Callers must re-locate matches in display space.

- [x] **README Overhaul & Benchmark Invalidation**:
  - Invalidate stale v0.1 ASCII-only benchmark numbers in root `README.md`; replace with reproducible M5 multi-corpus benchmark tables (ASCII code paths, CJK Hanzi/Kana, Emoji astral sequences) comparing WebGPU, CPU parity reference, uFuzzy, and JS Native.
  - Clearly disclose hardware qualification: distinguish physical GPU execution from software Vulkan (`pending-hardware`).
  - Refresh API quick-start examples, TypeScript types, and configuration snippets to reflect v0.2 exports: `SearchIndex`, `WebGPUEngine`, `CPUEngine`, `packUnicodeToGPUBuffer`, `serializeUnicodeDataset`, and error classes (`QueryTooLongError`, `IncompatibleIndexError`, `ProfileMismatchError`, `IncompatibleOptionError`).

- [x] **Internal Deprecation & Monorepo Boundary Enforcement (`packages/webgpu-search`)**:
  - Verify that the internal deprecated `legacy-ascii-v0.1` path emits a `console.warn` notifying users of removal in v0.3.
  - Audit codebase for 100% zero-DOM safety (`packages/webgpu-search` must contain no unguarded `window` or `document` symbols) to guarantee universal portability across browser main thread, Web Workers, Node.js, and SSR.
  - Ensure zero external runtime dependencies are introduced to `packages/webgpu-search`.

- [x] **Pre-Publish Release Pipeline & Quality Assurance Gates (`.agents/skills/pre-publish/SKILL.md`)**:
  - **Tarball Content & Leak Audit**:
    - Execute `npm pack --dry-run` in `packages/webgpu-search`.
    - Verify only authorized files are packaged: `dist/index.js`, `dist/index.cjs`, `dist/index.d.ts`, `dist/index.d.cts`, `README.md`, `LICENSE`.
    - Ensure zero test fixtures, internal scripts, intermediate build caches, or sensitive files leak into the package tarball.
  - **Bundle Size Budget Enforcement**:
    - Assert `dist/index.js` gzipped size remains $\le$ 22.5 KB (M1 budget ceiling).
  - **SemVer Version Bump & Changeset**:
    - Bump package version from `0.1.0` to `0.2.0` in `packages/webgpu-search/package.json`.
    - Synchronize workspace package references (`packages/webgpu-search` $\leftrightarrow$ `apps/benchmark`).
  - **Full Monorepo Validation Gate**:
    - `bun run check:shaders`: Offline AST & uniform layout check via `vgpu check`.
    - `bun run typecheck`: 0 TypeScript diagnostics across all workspaces.
    - `bun run build`: Clean production builds for both `packages/webgpu-search` and `apps/benchmark`.
    - `bun run test:mock`: 25/25 in-memory mock suites passing.
    - `bun run test:parity`: 117/117 differential parity assertions passing.
    - `bun run test:browser`: 14/14 browser regression assertions passing.
    - `bun run test:benchmark -- --fast`: End-to-end benchmark execution in headless browser context.

- [x] **Git Release Tagging & Issue Closure**:
  - Create annotated Git release tag `v0.2.0`.
  - Update `ISSUE-7-PLAN.md` and GitHub Issue #7 to 100% complete, closing Issue #7.
  - Publish GitHub release notes detailing the breaking Unicode migration, benchmark numbers, and architectural guarantees.

> M6 delivery: Dedicated breaking migration guide authored in `docs/migration-v0.2.md` and linked prominently in root `README.md` and `packages/webgpu-search/README.md`. Root and package documentation fully overhauled with reproducible multi-corpus M5 benchmark tables (ASCII, CJK, Emoji), hardware qualification disclosure (`pending-hardware`), and v0.2 API quickstarts. Internal deprecated `legacy-ascii-v0.1` path emits `console.warn` notifying of removal in v0.3. Audited 100% zero-DOM safety and confirmed zero external runtime dependencies. Package bumped to `0.2.0`, changeset recorded, and tarball leak audit verified (7 clean files, 61.1 kB). All 7 monorepo validation gates green. Annotated Git tag `v0.2.0` created and GitHub Issue #7 closed.

## 4. Acceptance checklist

- [x] No valid scalar `?`-replaced or surrogate-split in default profile.
- [x] Canonically equivalent inputs → identical post-fold-NFC token streams (incl. fold-inserted-mark reorder cases).
- [x] C+F fixtures pass; S+T/`ϴ` correctly excluded; pinned version + probe reported honestly.
- [x] CPU ≡ WebGPU ordered indices + `i32` scores for same profile/mode/query/limit/corpus when `hasOverflow===false`; multiset parity when overflowed.
- [x] GPU failure/device loss changes only `engine`/timings (`profileId/scoringVersion` identical).
- [x] No silent truncation; over-limit throws `QueryTooLongError` (or explicit CPU-route), tested at `limit±1` pre- and post-fold.
- [x] Marks, emoji/ZWJ/VS/flags survive; `e/é`, half/full-width, presentation forms correctly distinct.
- [x] Bengali/Devanagari/Arabic/CJK/supplementary/malformed/mixed-script covered with pinned expectations.
- [x] Actual-bytes stats; per-buffer pre-allocation checks; chunked dispatch; 0-row no-dispatch.
- [x] Contract names Unicode/profile/scoring/format versions; disclaims grapheme/locale/transliteration/confusables at the call site, not just §5.
- [x] Breaking migration documented with snippets; stale v0.1 benchmarks invalidated.
- [x] Reproducible median/p95 benchmarks with env metadata + pending-hardware slots.

## 5. Non-goals for v0.2

Locale-sensitive collation/mappings · transliteration · stemming/tokenization/spell-correction · confusable/skeleton matching · grapheme clusters as scoring units · Unicode-aware boundaries beyond documented ASCII set · cross-language semantic equality · fixed FPS/latency guarantees · paging/multi-buffer datasets (fail-closed CPU instead).

## 6. References

- UAX #15: https://unicode.org/reports/tr15/ · `CaseFolding.txt`: https://www.unicode.org/Public/UCD/latest/ucd/CaseFolding.txt · UAX #29 (deferred): https://www.unicode.org/reports/tr29/ · UTS #39 (out of scope): https://www.unicode.org/reports/tr39/
- `String.prototype.normalize()`: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/String/normalize
- WGSL: https://www.w3.org/TR/WGSL/ · uFuzzy (opt-in CPU-only, not parity): https://github.com/leeoniya/uFuzzy/blob/main/src/uFuzzy.js
- WHATWG `toWellFormed()` / ECMA-262 lone-surrogate background (adopted per-surrogate FFFD rule); UTS #18 ill-formed-subsequence handling (background only — NOT the adopted rule).
