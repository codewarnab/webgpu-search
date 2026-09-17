# Unicode Contract — v0.2 Code-Point-Safe Matching (M1 frozen)

> Source of truth for Issue #7. Frozen values in this file gate M2/M3.
> Plan: `ISSUE-7-PLAN.md`. Status: M0 spikes complete, M1 in progress.

## 1. Pipeline (byte-exact order)

```
raw JS string
  → String.trim
  → String.prototype.toWellFormed() where available,
    else /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g → U+FFFD
    (each lone surrogate → exactly one U+FFFD: high+high = 2, high+EOF = 1,
    low-without-high = 1; genuine U+FFFD indistinguishable by design)
  → NFC (host String.normalize('NFC'), probe §6)
  → full default case fold (C+F only, S+T excluded) when folded, else NFC-only
  → NFC again (fold is not NFC-closed: U+0130 → U+0069 U+0307 + neighbors)
  → u32 scalar token stream
```

Single function `normalizeText()` serves records and queries. `sanitizeStringForSlot`
is deleted from the parity path. Limit/empty checks apply to **post-fold token
count**; post-processing-empty queries (whitespace/mark/VS/ZWJ/tatweel-only)
return unified empty results (`query: ''`). Scores, spans, `str_len − query_len`
are in **post-fold code points**. Highlighting caveat: `"Straße"` → `"strasse"`
(6→7 tokens); folded coordinates must NOT slice the original string.

## 2. Frozen versions and caps

| Constant | Value | Notes |
|---|---|---|
| `UNICODE_VERSION` | `16.0.0` | `CaseFolding-16.0.0.txt` (2024-04-30), UCD terms |
| `SCORING_VERSION` | `parity-v1` | Bump on any scoring/boundary change |
| `FORMAT_VERSION` | `2` | `MAGIC 0x55324632 ('U2F2')` + enums + counts + checksum |
| `QUERY_TOKENS_MAX` | `128` | Post-fold tokens; bounds `str_len × query_len` shader loops |
| `RESULT_LIMIT_MAX` | `8192` | Clamp, tested |
| `TextProfileId` | `'unicode-default'` | Only parity profile; legacy internal-only, removed v0.3 |
| `folded` | index-construction-time | `IndexOptions.caseSensitive` (default `false`); per-query mismatch → `ProfileMismatchError` |
| `CpuAlgorithm` | `'parity' \| 'ufuzzy'`, default `'parity'` | uFuzzy explicit opt-in only; fallback is always parity; `preferGpu + ufuzzy` → `IncompatibleOptionError` |
| `onQueryTooLong` | `'throw' \| 'cpu-fallback'`, default `'throw'` | Over-limit → `QueryTooLongError extends RangeError {limit, actual, profileId}` |

Comparator both paths: `(b.score - a.score) || (a.index - b.index)` with
`|0`/`Math.imul` i32 semantics. Determinism guaranteed iff
`hasOverflow === false`; above cap assert `totalMatches + hasOverflow + score
multiset` only. `mode:'fuzzy'` semantic change (uFuzzy → parity-subsequence) is
a documented breaking change. `slotBytes` throws with migration message.

## 3. M0 spike results (measured 2026-09-17, Bun 1.4.2)

**Encoding (100k rows):** ASCII-heavy ~48ch → u32 22.04 MB vs utf8 5.51 MB
(4.00×) vs utf16 11.02 MB (2.00×); CJK ~30ch → u32 11.40 MB vs utf8 8.00 MB
(1.43×) vs utf16 5.70 MB (2.00×). Projections (ASCII): 500k → 112.1 MB
MARGINAL, 650k → 145.7 MB HARD CRASH (>128 MB binding), 1M → 224.2 MB CRASH.
CJK 1M → 117.8 MB MARGINAL. Decision: **u32 default stands** (code-point-safe
astral handling outweighs VRAM cost) with per-buffer fail-closed CPU; u16-BMP
and UTF-8 remain post-v0.2 experiments (u16 needs surrogate-aware proof).

**Query placement:** 128 tokens = 512 B + header ≪ 64 KiB uniform guarantee, so
an expanded uniform preserves constant-cache broadcast while storage adds a
binding + LD/ST path. No GPU hardware in CI image → latency A/B deferred to M5
browser runs; **default: persistent storage query buffer**, revisit if
short-query p95 regresses beyond §5 threshold.

**Fold table:** C+F = 1,557 entries (F = 104, 1→2+ = 104, 1→3 = 16).
Naive literal = 15,628 chars → **7,387 B gzip** (~65% of ~11 KB headroom —
marginal). Sparse range/delta ≈ 3,770 B raw → est ~1.8 KB gzip (comfortable).
Decision: **sparse-encoded eager table, delta cap ≤8 KB over baseline**;
Simple-only (C+S) is fallback only (drops contracted `ß/ss`: ß is F-only).

**Bundle baseline:** `packages/webgpu-search/dist/index.js` 40,428 B raw /
~9,125 B gzip (`minify:false` build); headroom to 20 KB ≈ 11 KB.

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

CPU parity search p95 budget pinned here (exact number: retained p95 ≤1.1× v0.1
ASCII-substring@100k; Bloom 64-bit pre-filter allowed post-parity in M5, never
as parity substitute). `loadDataset` reports `normalizeMs/packMs/uploadMs` with
chunked/yielded packing. Worker `LOAD_DATASET` clone cost measured in M4/M5
(`search.worker.ts:40-41` clones `strings`; `main.ts:356-362` has no transfer
list → no detachment today); string-isolated enrichment as measured
optimization; `loadDataset` guards neutered buffers (`byteLength===0`).

## 6. Versioning, probe, locale ban

`CaseFolding-<V>.txt` URL + revision + license in generator provenance header.
Host `normalize('NFC')` ≠ pinned version: M1 conformance probe runs pinned
`NormalizationTest.txt` excerpts at `create()`; mismatch warns + records
`nfcProbedVersion`, never reports pinned as fact. `toLowerCase/toUpperCase/
indexOf/charCodeAt` banned in parity path (lint) + `tr-TR` locale CI run.

Match matrix (each × folded true/false): `I/i/İ/ı`, `ß/ss/SS`, `ς/σ/Σ`,
`ﬀ/ff`, `ϴ/θ` (T-only, never folds), `e/é` (no match), ZWJ-family partial,
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
