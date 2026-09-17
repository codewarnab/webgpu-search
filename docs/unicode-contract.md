# Unicode Contract — v0.2 Code-Point-Safe Matching (M2: CPU parity landed)

> Source of truth for Issue #7. Frozen values in this file gate M2/M3.
> Plan: `ISSUE-7-PLAN.md`. Status: M0 spikes recorded (HW latency A/B deferred
> to M5 — see §3), M1 contract shape landed, M2 CPU parity landed (post-fold
> sizing + `cpu-reference.ts` + fold table). GPU engine swap stays M3.

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
  → full default case fold (C+F only, S+T excluded) when folded, else NFC-only
  → NFC again (fold is not NFC-closed: U+0130 → U+0069 U+0307 + neighbors)
  → u32 scalar token stream
```

Single function `normalizeText()` serves records and queries. `sanitizeStringForSlot`
is deleted from the parity path (kept behind the M2 legacy GPU path only;
removal in v0.3). Limit/empty checks apply to **post-fold token count**;
whitespace-only (incl. U+3000) / empty queries return unified empty results
(`query: ''`). Lone-mark / VS-only / ZWJ-only / tatweel-only inputs survive
NFC+C+F as single tokens per the survival rule, so they search normally
(usually 0 hits, echoing the original query) — pinned by test §8b. Scores,
spans, `str_len − query_len` are in **post-fold code points**. Highlighting
caveat: `"Straße"` → `"strasse"` (6→7 tokens); folded coordinates must NOT
slice the original string.

M2 GPU limitation (honest scaffolding, engine swap is M3): the GPU packer is
still legacy sanitized bytes (NFKD strip + `?`, 59-char query truncation) while
CPU is folded tokens. `SearchIndex` routes non-ASCII or >59-token queries to
CPU; ASCII-only short queries may serve GPU. GPU fallback can re-score vs
legacy GPU. Differential parity is asserted on CPU only until M3.

## 2. Frozen versions and caps

| Constant | Value | Notes |
|---|---|---|
| `UNICODE_VERSION` | `16.0.0` | `CaseFolding-16.0.0.txt` (Unicode 16.0.0, Sept 2024), UCD terms |
| `SCORING_VERSION` | `parity-v1` | Bump on any scoring/boundary change |
| `FORMAT_VERSION` | `2` | `MAGIC 0x55324632 ('U2F2')` + enums + counts + checksum |
| `QUERY_TOKENS_MAX` | `128` | Post-fold tokens; bounds `str_len × query_len` shader loops |
| `RESULT_LIMIT_MAX` | `8192` | Clamp, tested |
| `TextProfileId` | `'unicode-default'` | Only parity profile; legacy internal-only, removed v0.3 |
| `folded` | index-construction-time | `IndexOptions.caseSensitive` (default `false`); per-query mismatch → `ProfileMismatchError` (breaking v0.2: build one index per mode) |
| `CpuAlgorithm` | `'parity' \| 'ufuzzy'`, default `'parity'` | uFuzzy explicit opt-in only (CPU-only, explicitly non-conforming scores, skips GPU, excluded from parity matrix); default and GPU-failure fallback serve parity `cpu-reference.ts`. `preferGpu:true + cpuAlgorithm:'ufuzzy'` → `IncompatibleOptionError` (enforced in `SearchIndex.search()`). |
| `onQueryTooLong` | `'throw' \| 'cpu-fallback'`, default `'throw'` | Over-limit → `QueryTooLongError extends RangeError {limit, actual, profileId}`. M2: enforced on the exact post-fold token count (`normalizeText(query, folded)` vs `QUERY_TOKENS_MAX`), with a cheap raw-length pre-gate before NFC+fold. `'cpu-fallback'` forces the CPU path for that query. |

**M2 status (CPU parity landed, GPU legacy):** comparator, parity scorer, and
post-fold sizing are enforced on the CPU path. GPU stays on the legacy
sanitized-byte packer + 59-char truncation until M3, so GPU/CPU differential
parity is NOT asserted in M2 (see limitation note in §1).

Comparator both paths: score desc, index asc (wrap-free comparisons;
`Math.imul`/`|0` retained for the score formulas only). M2 CPU
(`cpu-reference.ts`) and M2 GPU readback sort share it; full WGSL scalar
rewrite stays M3. `LONE_SURROGATE_PATTERN` is non-global + pair-preserving
in v0.2 (breaking; see `unicode-preprocess.ts` JSDoc). Determinism guaranteed iff
`hasOverflow === false`; above cap assert `totalMatches + hasOverflow + score
multiset` only (M1 doc-only; CPU `hasOverflow` fix lands in M3/M4).
`mode:'fuzzy'` semantic change (uFuzzy → parity-subsequence) is
a documented breaking change. `slotBytes` throws `IncompatibleOptionError`
with migration message (throw-on-use in v0.2, removal in v0.3).

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
bandwidth timing was not measured in M0 and is deferred to M5 browser runs.
u16-BMP and UTF-8 remain post-v0.2 experiments (u16 needs surrogate-aware
parity proof). Oversize datasets fail closed to CPU (no paging in v0.2).

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
binding + LD/ST path. No GPU hardware in CI image → latency A/B deferred to M5
browser runs; **interim default: persistent storage query buffer** (stable
bindings, avoids uniform churn). Revisit if M5 short-query p95 on browser
hardware regresses beyond the M5-measured baseline (not the §5 CPU-parity
threshold, which is a different metric).

**Fold table (CaseFolding-16.0.0, C+F only):** 1,557 entries with status C or F,
of which 104 have status F — all multi-char mappings (88 × 1→2, 16 × 1→3; so
1→2+ = 104 ⊃ 1→3 = 16). Naive literal table ≈ 15,628 chars → **7,387 B gzip**
(`gzip -c`, ~68% of ~11 KB headroom — marginal). Sparse range/delta ≈ 3,770 B
raw → est. ~1.8 KB gzip (comfortable; estimate, not a built artifact).
Generator script + packed bytes + `minify:true` delta land in M2.
Decision: **sparse-encoded eager table, delta cap ≤8 KB gzip over baseline**;
Simple-only (C+S) is fallback only (drops contracted `ß/ss`: ß is F-only).

**Bundle baseline (measured 2026-09-17 post-M1-fix working tree):**
`packages/webgpu-search/dist/index.js` 44,581 B raw / 10,206 B gzip
(historical `gzip -c` numbers; the M2 gate `scripts/check-m2-bundle.ts` uses
deterministic gzip level 6 mtime=0, which differs from `gzip -c` by ~200–300 B
filename/mtime bytes — do not compare across methods; re-baseline with the
gate method before enforcing byte-tight deltas. `tsup` with `minify:false` —
i.e. gzip of the current unminified build, not a shipped min+gzip). Headroom
to the 20 KB gzip budget ≈ 10 KB at M1. Pre-PR baseline was 40,428 B / ~9,125 B;
the delta is the M1 contract code itself. M2 asserts the fold-table delta
against the ≤8 KB **gzip** cap (M2 measured +7,922 B deterministic — 270 B
headroom; do not grow the table further without the lazy-chunk plan).

## 4. Buffers and bindings

Per-buffer checks vs `min(maxBufferSize, maxStorageBufferBindingSize)` from
adapter limits (surfaced via `device.limits`; 128 MB fallback mock-only):
`records = tokenCount×4`, `offsets = (rows+1)×4`, `query = 128×4`,
`output = 8 + cap×8`. `vramAllocatedBytes` = actual packed bytes. No paging in
v0.2 — oversize fails closed to CPU. Storage-wins header (32 B, 16 B-aligned):
`struct Q { total_rows, query_len, max_candidates, flagsAndProfile, _pad }`;
bindings `0:uniform, 1:offsets:read, 2:records:read, 3:query:read,
4:output:read_write` (4 bindings if uniform-wins).

## 5. Budgets

CPU parity search budget (relative threshold, absolute TBD pending-hardware):
retained p95 ≤1.1× v0.1 ASCII-substring@100k. The v0.1 baseline (commit, corpus,
hardware, warmups/samples, median/p95) is not yet pinned — M5 pins it on
browser hardware before enforcing; until then the threshold is a formula, not
a gate. Bloom 64-bit pre-filter allowed post-parity in M5, never as parity
substitute. `loadDataset` reports `normalizeMs/packMs/uploadMs` with
chunked/yielded packing (M2 design: 1M×48 ICU calls must not block the main
thread). Worker `LOAD_DATASET` clone cost measured in M4/M5
(`search.worker.ts:40-41` clones `strings`; `main.ts:356-362` has no transfer
list → no detachment today); string-isolated enrichment as measured
optimization; `loadDataset` guards neutered buffers (`byteLength===0`).

## 6. Versioning, probe, locale ban

`CaseFolding-<V>.txt` URL + revision + license in generator provenance header.
Host `normalize('NFC')` ≠ pinned version: M1 conformance probe runs pinned
`NormalizationTest.txt` excerpts at `create()`; mismatch warns + records
`nfcProbedVersion`, never reports pinned as fact. `toLowerCase/toUpperCase/
indexOf/charCodeAt` banned in parity path (lint) + `tr-TR` locale CI run.

Match matrix (each × folded true/false): `I/i/İ/ı` (I→i via C, never ı;
İ→i+dot via F, never bare i; ı identity), `ß/ss/SS` (F; S excluded),
`ς/σ/Σ` (C), `ﬀ/ff` (F), `ϴ/θ` (U+03F4 has C→U+03B8 in Unicode 16.0.0, so C+F
folds it), `e/é` (no match), ZWJ-family partial,
flag/keycap splits, Arabic bare-vs-vocalized (no match), presentation forms
(distinct, NFC≠NFKC), Bengali/Devanagari conjuncts (visually-equal-but-unequal
→ no match), CJK `U+3000`/Unicode spaces (no word bonus — documented ASCII
`\/ _ - . space : \` limitation). Code points ≠ graphemes (`Intl.Segmenter`
pointer at call site); no locale mappings, ever.

## 7. Rejected for v0.2 (rationale)

u16-BMP packing (astral split) · Simple-only folding (drops `ß/ss`) ·
universal word boundaries (scoring-version change, backlogged) · LDS-as-parity-fix
(contention only, not top-K) · in-shader fold table (divergence; CPU-fold keeps
pure-`==` shader).
