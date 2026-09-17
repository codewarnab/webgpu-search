# Issue #7 — v0.2 Unicode Code-Point-Safe CPU/GPU Matching: Implementation Plan (v2)

> Status: planned, revised after 5-way adversarial review. Assumes #8 (v0.1.1) is closed first.
> Source: [issue #7](https://github.com/codewarnab/webgpu-fuzzy-search/issues/7) (bounded contract: code-point-safe, NOT full linguistic/grapheme semantics).
> Deliverable of M1: `docs/unicode-contract.md` + version constants + frozen numeric caps. No M2/M3 code until M1 gate passes.

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

- `mode:'fuzzy'` semantic change (uFuzzy-tolerant → parity-subsequence) is a **documented breaking change** in M6 migration table. Never extend `SearchMode` with `'ufuzzy'`.
- Fallback on GPU error / device loss routes to `cpu-reference.ts` **parity** scorer, never uFuzzy. uFuzzy only on explicit `cpuAlgorithm:'ufuzzy'`. `preferGpu:true + cpuAlgorithm:'ufuzzy'` → `IncompatibleOptionError` (throw at `search()`).
- `SearchResponse` echoes `profileId/scoringVersion/cpuAlgorithm` so "only `engine`/timings change on fallback" is verifiable (§2.4).

### 2.4 Determinism, overflow, limits (numeric values frozen in M1)

- Comparator both paths: `(b.score - a.score) || (a.index - b.index)`, computed with `|0`/`Math.imul` i32 semantics on CPU to mirror WGSL `i32()` casts. Negative scores legal (long record + late match) — assert signed parity, no clamping. `u32 totalMatches` overflow past 2³² tested.
- **Determinism scoped:** exact ordered `(index,score)` parity asserted **iff `hasOverflow === false`**. When `totalMatches > 8192`, stored subset is first-cap-by-dispatch-race then host-sorted: assert only `totalMatches + hasOverflow + score multiset`, never order. Fix `hybrid-index.ts:205-206` CPU `hasOverflow` to compute identically to GPU.
- `RESULT_LIMIT_MAX = 8192` (clamp, tested). `QUERY_TOKENS_MAX = 128` post-fold tokens (frozen M1; chosen to bound `str_len × query_len` shader loops against TDR/DoS on 10k pastes while covering CJK/emoji/ZWJ + `ß→2` expansion headroom). Over-limit: default `throw QueryTooLongError`; `onQueryTooLong:'cpu-fallback'` routes explicitly. Forced-GPU tests observe the throw (no silent fallback). Result-limit vs query-token-limit tested separately — never conflated. `NaN` limit → default 50, both paths.
- `dispatch`: `if (rows===0) return empty without dispatch`; chunked multi-dispatch when `ceil(rows/128) > maxComputeWorkgroupsPerDimension`; `@workgroup_size(128)` justified against `maxComputeInvocationsPerWorkgroup` in M3 doc.

### 2.5 Buffers, uniform, bindings (pinned in M1 doc, implemented in M3)

- Records: `recordsBytes = tokenCount*4`; offsets: `(rows+1)*4`; query (default): `QUERY_TOKENS_MAX*4` persistent storage buffer (min 16 B, `writeBuffer` per search, destroyed under mutex); output: `8 + cap*8`. **Each** checked independently against `min(maxBufferSize, maxStorageBufferBindingSize)` read from the adapter limits object and exposed via `device.limits` (not a hardcoded 128 MB fallback — fallback mock-only). No `64`-byte fiction. `getStats().vramAllocatedBytes = tokens.byteLength + offsets.byteLength` (actual). Publish max-rows table per 128/256 MB class in contract (u32@48tok ≈ 650k rows/128 MB — the 128 MB `maxStorageBufferBindingSize` crash at ~650k×50-char rows is the binding limit, verified against W3C §3.3.1 defaults of 128 MiB binding / 256 MiB buffer). Datasets that don't fit → fail-closed CPU with message (no paging in v0.2).
- Query-buffer placement is **decided by M0 spike, not assumed**: WebGPU guarantees `maxUniformBufferBindingSize ≥ 64 KiB` (16,384 u32 tokens), so an expanded uniform (1–4 KB, 256–1,024 tokens) preserves constant-cache broadcast (single fetch + 32-lane broadcast per warp) while storage routes through LD/ST without broadcast. M0 measures expanded-uniform vs persistent-storage-query on short-query latency before M3 locks bindings. Uniform candidate (if spike wins): keep `struct Q` header + `query_tokens: array<u32, QUERY_TOKENS_MAX>` in `var<uniform>`; storage candidate (default if spike ties/loses): 32 B header uniform per below + `3:query:read` storage.
- Uniform header if storage wins (32 B, 16 B-aligned): `struct Q { total_rows:u32, query_len:u32, max_candidates:u32, flagsAndProfile:u32, _pad: vec4<u32> }` where `flagsAndProfile` packs `caseSensitive:1b + folded:1b + profileEnum:6b + unicodeEnum:8b + scoringEnum:8b + reserved:8b`. No strings in uniform. Bindings if storage wins: `0:uniform, 1:offsets:read, 2:records:read, 3:query:read, 4:output:read_write`; assert `maxStorageBuffersPerShaderStage` covers 5 bindings (4 if uniform wins).
- `loadDataset` validators + zero-fill discipline for padded tails; `offsets[last]===tokenCount` enforced; corrupt header → throw before any dispatch.

### 2.6 Unicode versioning, fold table, locale ban

- `UNICODE_VERSION = "16.0.0"` (example; frozen M1): `CaseFolding-<V>.txt` URL + revision + license attribution in provenance header of generated table. Generator includes **C+F only**, excludes S+T, full 1→2/3 mappings. `SCORING_VERSION = "parity-v1"`.
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

- [ ] Encoding spike: u32 vs UTF-8/UTF-16 microbenchmark (substring-only, 100k ASCII + 100k CJK): ALU vs bandwidth, VRAM math, arithmetic-intensity note (u32 ≈ 0.25–0.5 ops/byte, bandwidth-bound on ~45 GB/s iGPUs — treat external 1.1 ms→4.4 ms figures as estimates to confirm, not facts). Gate M3 on numbers; default remains u32 unless spike flips it.
- [ ] Query-placement spike: expanded uniform (1–4 KB) vs persistent storage query buffer on short-query p95; locks §2.5 bindings before M3.
- [ ] Fold-table size spike: build C+F table, report packed + `min+gzip` delta with `minify:true`; record eager-vs-lazy decision.
- [ ] `.wgsl` single-source dedup step-0: tsup text-loader on `packages/webgpu-search/src/shaders/*.wgsl` is the single source; delete inlined copies; fix `src/shaders` vs `packages/...` path drift. Every later shader edit gated by `bun run check:shaders`.
- [ ] Benchmark harness fix (median/p95, corpora, timestamp request, thresholds) landed before M3.
- [ ] Gate: spike numbers (encoding + query-placement winner) + table bytes + threshold + CPU p95 budget + `QUERY_TOKENS_MAX=128` + uniform/bindings pin all written into `docs/unicode-contract.md`.

### M1 — Contract + golden corpus (no engine changes)

- [ ] Write `docs/unicode-contract.md` + JSDoc on `types.ts`: pipeline order, `toWellFormed()`-primary FFFD rule (each lone surrogate → one FFFD) + regex fallback, C+F/S+T rule, post-fold units + highlighting caveat, ASCII delimiters, two-key sort + overflow scope, query/result limits + `QueryTooLongError` shape, version constants, `slotBytes`-throws rule, `cpuAlgorithm` + `preferGpu` conflict (`IncompatibleOptionError`) rule, `ProfileMismatchError` rule, ICU-probe policy.
- [ ] Add `TextProfileId`, `CpuAlgorithm`, `IndexOptions.textProfile/caseSensitive`, error classes (`QueryTooLongError`, `IncompatibleIndexError`, `ProfileMismatchError`, `IncompatibleOptionError`) to `types.ts`/`text-profile.ts`; add `profileId/unicodeVersion/scoringVersion/tokenCount/folded/formatVersion` to `IndexStats`, `profileId/scoringVersion/cpuAlgorithm` to `SearchResponse` (M6 owns semver note).
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

### M4 — Parity, fallback, CI

- [ ] Differential harness: forced-CPU (`preferGpu:false`, parity) vs forced-WebGPU, same profile/mode/query/limit; compare `totalMatches/candidateCount/hasOverflow/ordered(index,score)`; explicitly ignore `timings/engine/query` echo. Seeded RNG, per-cell timeout, sharding. Exact-order assert only when `totalMatches ≤ cap`; above cap assert multiset. uFuzzy excluded (separate exclusion-proof test).
- [ ] Failure injection: shader-compile throw, pipeline-create reject, offsets-OK/records-OOM partial (no leaked half-state), `mapAsync`/`onSubmittedWorkDone` reject, query-over-limit, `checkMemoryBudget` vs tiny mock limits, simulated `device.lost` mid-suite (in-flight → throw or CPU-tagged; subsequent → `engine:'cpu'` identical semantics), repeated `destroy()`+`search`.
- [ ] Concurrency/abort: `loadDataset`∥`search`, interleaved `searchCold(A)×searchCold(B)` (A never returns B's rows — guaranteed by single-mutex), abort-before/mid-`mapAsync`/after-complete, CPU-scan abort, latest-query epoch rule (implement last-wins or delete "latest-query" claim). Worker: measure `LOAD_DATASET` clone cost, land string-isolated enrichment behind a flag with before/after numbers.
- [ ] Rewrite `test-vgpu-mock.ts` (new layout, delete 4 v0.1 asserts: 272 B uniform, 65544 B output, `Café→Cafe ??` sanitizer, `packStrings(…,64)`) and `test-regression.ts:121-198` (u32 packing + query storage + `limit`); browser subset is **release gate**.
- [ ] Gate: zero parity failures, no silent truncation, `check:shaders + typecheck + build + test:mock + test:browser` green (resolve old contradiction). No unguarded DOM refs (Worker/Node/SSR).

### M5 — Benchmark (execute pre-fixed harness)

- [ ] Matrix §2.7 corpora × substring/fuzzy; per cell: env, counts, UTF-16 units, normalized code points, packed bytes, `normalizeMs/packMs/uploadMs`, VRAM, warm median/p95, timestamp execution where supported, CPU-parity + fallback latency (against M1 CPU p95 budget), overflow/query-limit rates, worker clone cost. Bloom pre-filter and LDS reduction allowed here as measured optimizations only — never as parity substitutes.
- [ ] v0.1 baseline vs v0.2 per row, Chrome/Edge × ≥2 GPU classes; missing hardware = `required/pending`. External latency/throughput figures (4× bandwidth penalty, 45–90 ms/100k CPU scans, 120–350 ms clone stalls) treated as hypotheses to confirm, never as shipped claims.
- [ ] Gate: no unexplained regression beyond M1 threshold; all claims device-qualified.

### M6 — Migration + release (breaking)

- [ ] Docs: 0.1.x limits note; **breaking** v0.2 migration (rebuild required, offsets bytes→tokens, `ß/ss` merges, canonical merges, astral/mark fixes, tie-break, `mode:'fuzzy'` semantic change, `slotBytes` throws, cache invalidation); before/after table; rebuild + `getStats()` version-check snippets; `TextProfile` doc block (code points ≠ graphemes, `Intl.Segmenter` pointer, ASCII delimiters, no locale mappings — preempts "Turkish bug"/"emoji ranking" issues); README benchmark invalidation note.
- [ ] `legacy-ascii-v0.1` internal-only `console.warn` + v0.3 removal target; prerelease → compat reports → stable.
- [ ] Final: no DOM refs, no temp files/logs, clean diff.

## 4. Acceptance checklist

- [ ] No valid scalar `?`-replaced or surrogate-split in default profile.
- [ ] Canonically equivalent inputs → identical post-fold-NFC token streams (incl. fold-inserted-mark reorder cases).
- [ ] C+F fixtures pass; S+T/`ϴ` correctly excluded; pinned version + probe reported honestly.
- [ ] CPU ≡ WebGPU ordered indices + `i32` scores for same profile/mode/query/limit/corpus when `hasOverflow===false`; multiset parity when overflowed.
- [ ] GPU failure/device loss changes only `engine`/timings (`profileId/scoringVersion` identical).
- [ ] No silent truncation; over-limit throws `QueryTooLongError` (or explicit CPU-route), tested at `limit±1` pre- and post-fold.
- [ ] Marks, emoji/ZWJ/VS/flags survive; `e/é`, half/full-width, presentation forms correctly distinct.
- [ ] Bengali/Devanagari/Arabic/CJK/supplementary/malformed/mixed-script covered with pinned expectations.
- [ ] Actual-bytes stats; per-buffer pre-allocation checks; chunked dispatch; 0-row no-dispatch.
- [ ] Contract names Unicode/profile/scoring/format versions; disclaims grapheme/locale/transliteration/confusables at the call site, not just §5.
- [ ] Breaking migration documented with snippets; stale v0.1 benchmarks invalidated.
- [ ] Reproducible median/p95 benchmarks with env metadata + pending-hardware slots.

## 5. Non-goals for v0.2

Locale-sensitive collation/mappings · transliteration · stemming/tokenization/spell-correction · confusable/skeleton matching · grapheme clusters as scoring units · Unicode-aware boundaries beyond documented ASCII set · cross-language semantic equality · fixed FPS/latency guarantees · paging/multi-buffer datasets (fail-closed CPU instead).

## 6. References

- UAX #15: https://unicode.org/reports/tr15/ · `CaseFolding.txt`: https://www.unicode.org/Public/UCD/latest/ucd/CaseFolding.txt · UAX #29 (deferred): https://www.unicode.org/reports/tr29/ · UTS #39 (out of scope): https://www.unicode.org/reports/tr39/
- `String.prototype.normalize()`: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/String/normalize
- WGSL: https://www.w3.org/TR/WGSL/ · uFuzzy (opt-in CPU-only, not parity): https://github.com/leeoniya/uFuzzy/blob/main/src/uFuzzy.js
- WHATWG `toWellFormed()` / ECMA-262 lone-surrogate background (adopted per-surrogate FFFD rule); UTS #18 ill-formed-subsequence handling (background only — NOT the adopted rule).
