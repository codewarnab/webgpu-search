# Text Normalization

## 1. Pipeline (byte-exact order)

```
raw JS string
 → String.trim
 → String.prototype.toWellFormed() where available,
 else pair-preserving fallback /([\uD800-\uDBFF][\uDC00-\uDFFF])|[\uD800-\uDFFF]/g
 (group 1 = valid pair preserved, else lone surrogate → U+FFFD; lookbehind-free
 for Safari <16.4 / old Hermes; each lone surrogate → exactly one U+FFFD:
 high+high = 2, high+EOF = 1, low-without-high = 1; genuine U+FFFD
 indistinguishable by design)
 → NFC (host String.normalize('NFC'), probe §6)
 → full default case fold (C+F only, S+T excluded) when normalized, else NFC-only
 → NFC again (fold is not NFC-closed: U+0130 → U+0069 U+0307 + neighbors)
 → u32 scalar token stream
```

Single function `normalizeText()` serves records and queries. `sanitizeStringForSlot`
is deleted from the exact path (kept behind the deprecated legacy export only;
removal in ). Limit/empty checks apply to **post-normalization token count**;
whitespace-only (incl. U+3000) / empty queries return unified empty results
(`query: ''`). Lone-mark / VS-only / ZWJ-only / tatweel-only inputs survive
NFC+C+F as single tokens per the survival rule, so they search normally
(usually 0 hits, echoing the original query) — pinned by test §8b. Scores,
spans, `str_len − query_len` are in **post-normalization code points**. Highlighting
caveat: `"Straße"` → `"strasse"` (6→7 tokens); normalized coordinates must NOT
slice the original string.

 representation swap (landed): GPU packs the same post-normalization u32 scalars via
`packUnicodeToGPUBuffer` (pre-tokenized fast path, no second normalization)
and searches with a pure-`==` scalar WGSL comparator (32 B uniform header +
512 B persistent storage query buffer, 5 bindings). `flagsAndProfile` (word 3)
is reserved wire format for the harness/debug — shaders do not read it;
profile enforcement lives host-side (`ProfileMismatchError`). All valid
queries up to 128 post-normalization tokens route to WebGPU when available; failures
fall back to the exact CPU scorer with identical semantics.

 differential position: `scripts/test-parity-harness.ts` pins the (a)-(n)
matrix corrections, echo contracts, failure-injection delta, and
concurrency/abort against the mock device (110+ asserts green; exact
GPU-order cells report `pending-hardware` because `vgpu/mock` never executes
WGSL -- mock-green alone ships nothing). The benchmark worker migrates with
it: `LOAD_DATASET` takes `{strings}` / dataset `{serialized}` (legacy byte
buffers rejected even in combo, neutered `byteLength===0` guarded,
fail-closed empty payload, state commits only on success) and SEARCH returns
compact `{index,score}[]` behind `STRING_ISOLATED_ENRICHMENT` for main-thread
text enrichment (~6x smaller worker->main clone per keystroke by byte math,
order-of-magnitude: per-object clone overhead and UTF-16-vs-bytes ignored;
browser-ms confirmation is pending-hardware). SEARCH failures post
`SEARCH_ERROR` (never silent stale); dataset switches carry a generation so
stale-dataset hits are dropped. True ordered `(index,score,text)` matching on
executing hardware is gated by the browser block in
`scripts/test-regression.ts` (per-PR CI gate and release gate; oracle in
Bun/Node vs subject in Chrome -- same-host assumption, see in-file note).
By type design that block is order/text-only: `WebGPUSearchResult` (engine
level) carries no version fields, so `profileId/scoringVersion/cpuAlgorithm`
identity is pinned at the `SearchIndex` level in the mock harness instead
(fallback changes only `engine`/timings by assertion).

## 2. Frozen versions and caps

| Constant | Value | Notes |
|---|---|---|
| `UNICODE_VERSION` | `16.0.0` | `CaseFolding-16.0.0.txt` (Unicode 16.0.0, Sept 2024), UCD terms |
| `SCORING_VERSION` | `parity-v1` | Bump on any scoring/boundary change |
| `DATASET_FORMAT_VERSION` | `2` | `MAGIC 0x55324632` + enums + counts + checksum |
| `QUERY_TOKENS_MAX` | `128` | Post-normalization tokens; bounds `str_len × query_len` shader loops |
| `RESULT_LIMIT_MAX` | `8192` | Clamp, tested |
| `TextProfileId` | `'unicode-default'` | Only text profile |
| `normalized` | index-construction-time | `IndexOptions.caseSensitive` (default `false`); per-query mismatch → `ProfileMismatchError` (breaking : build one index per mode) |
| `CpuScorer` | `'exact' \| 'ufuzzy'`, default `'exact'` | uFuzzy explicit opt-in only (CPU-only, explicitly non-conforming scores, skips GPU, excluded from differential matrix); default and GPU-failure fallback serve the exact `exact-scorer.ts` (legacy `'parity'` value maps to `'exact'`). `preferGpu:true + cpuScorer:'ufuzzy'` → `IncompatibleOptionError` (enforced in `SearchIndex.search()`). |
| `onQueryTooLong` | `'throw' \| 'cpu-fallback'`, default `'throw'` | Over-limit → `QueryTooLongError extends RangeError {limit, actual, profileId}`. : enforced on the exact post-normalization token count (`normalizeText(query, normalized)` vs `QUERY_TOKENS_MAX`), with a cheap raw-length pre-gate before NFC+folding. `'cpu-fallback'` forces the CPU path for that query. |

**Differential status (exact matching on both paths):** comparator, exact scorer,
post-normalization sizing, u32 packing, and scalar WGSL are enforced on both paths.
GPU/CPU differential matching is asserted by the harness (see limitation note
in §1).

Comparator both paths: score desc, index asc (wrap-free comparisons;
`Math.imul`/`|0` retained for the score formulas only). CPU
(`exact-scorer.ts`) and GPU readback sort share it; the WGSL scalar
rewrite landed in (pure-`==`, i32-arithmetic form). `LONE_SURROGATE_PATTERN` is non-global + pair-preserving
in (breaking; see `unicode-preprocess.ts` JSDoc). Determinism guaranteed iff
`hasOverflow === false`; above cap assert `totalMatches + hasOverflow + score
multiset` only.
`mode:'fuzzy'` semantic change (uFuzzy → exact-subsequence) is
a documented breaking change. `slotBytes` throws `IncompatibleOptionError`
with migration message (throw-on-use in, removal in ).

## 3. M0 spike results (measured 2026-09-17, Bun 1.4.2; method notes inline)

**Encoding (100k rows, byte-arithmetic projections — no GPU timing in CI):**
ASCII-heavy ~48 code points/row → u32 22.04 MiB vs utf8 5.51 MiB
(4.00×) vs utf16 11.02 MiB (2.00×); CJK ~30 code points/row → u32 11.40 MiB
vs utf8 8.00 MiB (1.43×) vs utf16 5.70 MiB (2.00×). Sizes are computed as
`tokenCount×4` (+ `(rows+1)×4` offsets where noted), not device-measured;
ratios are therefore exact by construction. Row-average derivation and corpus
seed are not pinned — treat absolute MiB as order-of-magnitude, ratios as
structural. Projections (ASCII records+offsets): 500k → ~112 MiB (fits 128 MiB
binding, marginal), 650k → ~146 MiB (**projected binding exceedance**, not an
observed crash), 1M → ~224 MiB (projected exceedance). CJK 1M → ~118 MiB
(marginal). Decision: **u32 default stands** on code-point safety (astral/emoji
survive without surrogate splitting), not on a measured latency win; ALU vs
bandwidth timing was not measured in M0 and is deferred to browser runs.
u16-BMP and UTF-8 remain post- experiments (u16 needs surrogate-aware
differential proof). Oversize datasets fail closed to CPU (no paging).

Max-rows guide (records+offsets vs binding limit; ~48 tokens/row ASCII,
~30 tokens/row CJK):

| Binding class | ASCII ~48 tok/row | CJK ~30 tok/row |
|---|---|---|
| 128 MiB (`maxStorageBufferBindingSize` default) | ~680k rows | ~1.05M rows |
| 256 MiB (`maxBufferSize` default) | ~1.37M rows | ~2.1M rows |

Limits are per-buffer (`records`, `offsets`, `query`, `output` checked
independently); the binding limit binds first. Verified against W3C WebGPU
§3.3.1 defaults (128 MiB binding / 256 MiB buffer) — defaults, not device
actuals; always read `device.limits` at runtime.

**Query placement (no HW in CI — interim default, A/B deferred):** 128 tokens
= 512 B + 32 B header ≪ 64 KiB `maxUniformBufferBindingSize` guarantee, so an
expanded uniform would preserve constant-cache broadcast while storage adds a
binding + LD/ST path. No GPU hardware in CI image → latency A/B deferred to 
browser runs; **interim default: persistent storage query buffer** (stable
bindings, avoids uniform churn). Revisit if short-query p95 on browser
hardware regresses beyond the measured baseline (not the §5 CPU-exact
threshold, which is a different metric).

**Fold table (CaseFolding-16.0.0, C+F only):** 1,557 entries with status C or F,
of which 104 have status F — all multi-char mappings (88 × 1→2, 16 × 1→3; so
1→2+ = 104 ⊃ 1→3 = 16). Naive literal table ≈ 15,628 chars → **7,387 B gzip**
(`gzip -c`, ~68% of ~11 KB headroom — marginal). Sparse range/delta ≈ 3,770 B
raw → est. ~1.8 KB gzip (comfortable; estimate, not a built artifact).
Generator script + packed bytes + `minify:true` delta land in.
Decision: **sparse-encoded eager table, delta cap ≤8 KB gzip over baseline**;
Simple-only (C+S) is fallback only (drops contracted `ß/ss`: ß is F-only).

**Bundle baseline (measured 2026-09-17 post--fix working tree):**
`packages/webgpu-search/dist/index.js` 44,581 B raw / 10,206 B gzip
(historical `gzip -c` numbers; the gate `scripts/check-m2-bundle.ts` uses
deterministic gzip level 6 mtime=0, which differs from `gzip -c` by ~200–300 B
filename/mtime bytes — do not compare across methods; re-baseline with the
gate method before enforcing byte-tight deltas. `tsup` with `minify:false` —
i.e. gzip of the current unminified build, not a shipped min+gzip). Headroom
to the 20 KB gzip budget ≈ 10 KB at. Pre-PR baseline was 40,428 B / ~9,125 B;
the delta is the contract code itself. asserted the case-fold-table delta
against the ≤8 KB **gzip** cap. re-baselined to the post- tree (`dist/index.js`
70,455 B raw / 18,343 B gzip, gate method): base +1,841 B, review hardening
(fail-closed trust boundaries) +~1.2 KB → ~21.4 KB gzip vs the 22 KB budget
(delta cap 4 KB) — see `scripts/check-m2-bundle.ts` (re-baseline + bump
rationale inline, not exemption). needs a budget re-plan before adding
harness weight.

## 4. Buffers and bindings

Per-buffer checks vs `min(maxBufferSize, maxStorageBufferBindingSize)` from
adapter limits (surfaced via `device.limits`; 128 MB fallback mock-only):
`records = tokenCount×4`, `offsets = (rows+1)×4`, `query = 128×4`,
`output = 8 + cap×8`. `vramAllocatedBytes` = actual packed bytes. No paging in
 — oversize fails closed to CPU. Storage-wins header (32 B, 16 B-aligned):
`struct Q { total_rows, query_len, max_candidates, flagsAndProfile, _pad }`;
bindings `0:uniform, 1:offsets:read, 2:records:read, 3:query:read,
4:output:read_write` (4 bindings if uniform-wins).

## 5. Budgets

CPU parity search budget (relative threshold, absolute TBD pending-hardware):
retained p95 ≤1.1× ASCII-substring@100k. The baseline (commit, corpus,
hardware, warmups/samples, median/p95) is not yet pinned — pins it on
browser hardware before enforcing; until then the threshold is a formula, not
a gate. Bloom 64-bit pre-filter allowed post-exact-match, never as the exact path
substitute. `loadDataset` reports `normalizeMs/packMs/uploadMs` with
chunked/yielded packing. Worker `LOAD_DATASET` clone cost measured in /
(`search.worker.ts` `LOAD_DATASET` clones `strings`; `main.ts`
`switchDataset` moves a `.slice(0)` copy of `serializeddataset` with a transfer
list -- copy-then-move, not zero-copy, since the original is retained);
string-isolated enrichment as measured optimization; `loadDataset` guards
neutered buffers (`byteLength===0`).

## 6. Versioning, probe, locale ban

`CaseFolding-<V>.txt` URL + revision + license in generator provenance header.
Host `normalize('NFC')` ≠ pinned version: conformance probe runs pinned
`NormalizationTest.txt` excerpts at `create()`; mismatch warns + records
`nfcProbedVersion`, never reports pinned as fact. NOTE: `nfcProbedVersion` is
currently always `null` (probe deferred — pack/deserialize hardcode null);
the warn+record behavior above is the contracted / target, not today's
runtime. `toLowerCase/toUpperCase/
indexOf/charCodeAt` banned in exact path (lint) + `tr-TR` locale CI run.

Match matrix (each × normalized true/false): `I/i/İ/ı` (I→i via C, never ı;
İ→i+dot via F, never bare i; ı identity), `ß/ss/SS` (F; S excluded),
`ς/σ/Σ` (C), `ﬀ/ff` (F), `ϴ/θ` (U+03F4 has C→U+03B8 in Unicode 16.0.0, so C+F
folds it), `e/é` (no match), ZWJ-family partial,
flag/keycap splits, Arabic bare-vs-vocalized (no match), presentation forms
(distinct, NFC≠NFKC), Bengali/Devanagari conjuncts (visually-equal-but-unequal
→ no match), CJK `U+3000`/Unicode spaces (no word bonus — documented ASCII
`\/ _ -. space : \` limitation). Code points ≠ graphemes (`Intl.Segmenter`
pointer at call site); no locale mappings, ever.

## 7. Rejected for (rationale)

u16-BMP packing (astral split) · Simple-only folding (drops `ß/ss`) ·
universal word boundaries (scoring-version change, backlogged)
(contention only, not top-K) · in-shader case-fold table (divergence; CPU-fold keeps
pure-`==` shader).
