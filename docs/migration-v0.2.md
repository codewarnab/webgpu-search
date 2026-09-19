# Migration Guide: v0.1 to v0.2 (Breaking Release)

> **Release**: `webgpu-search@0.2.0`  
> **Status**: Breaking Major Release  
> **Normative Contract**: [`docs/unicode-contract.md`](./unicode-contract.md)

---

## Overview

`webgpu-search` v0.2 introduces a unified, code-point-safe Unicode search architecture shared byte-for-byte across WebGPU compute shaders and CPU reference execution. 

In v0.1, the engine relied on UTF-16 byte slicing, destructive ASCII sanitization (`?` replacement for non-ASCII characters), and diverging CPU scoring models (`uFuzzy` rank scoring). In v0.2, every string is preprocessed through a deterministic, spec-compliant pipeline (`trim` → `toWellFormed()` surrogate sanitization → `NFC` → full `CaseFolding-16.0.0` `C+F` fold → `NFC`), packed as 32-bit Unicode scalar values (`u32`), and scored with bit-exact symmetry between WebGPU and CPU fallback paths.

Because record representation, buffer alignments, sorting rules, and index layouts have fundamentally changed, **all v0.1 indexes and serialized binary caches must be rebuilt**.

---

## Index Rebuild & Binary Format Changes

### Why Rebuilding Is Mandatory

Every index created or serialized in v0.1 is incompatible with v0.2 for two primary reasons:

1. **Storage Unit Expansion (`1 byte/char` $\to$ `4 bytes/token` `u32`)**:
   - **v0.1**: Records were packed as 8-bit ASCII characters (`1 byte/char`) in a raw byte array. Non-ASCII characters were stripped or replaced with `?`.
   - **v0.2**: Records are packed as normalized 32-bit Unicode scalar values (`Uint32Array`, `4 bytes/token`). This ensures astral plane characters, CJK ideographs, emojis, and combining marks survive without surrogate splitting or truncation.
2. **Offset Semantics (`byte offsets` $\to$ `token offsets`)**:
   - **v0.1**: The `offsets` array stored byte boundaries (`offsets[i]` was a byte offset into `Uint8Array`).
   - **v0.2**: The `offsets` array stores token indices (`offsets[i]` is a `u32` scalar index into `Uint32Array`).

### The `U2F2` Binary Serialization Format

v0.2 replaces raw binary buffers with a versioned, self-describing container format (`U2F2`):

```
+----------------------------------------------------------------------------------------------------+
| Bytes 0..3   | Bytes 4..7   | Bytes 8..11    | Bytes 12..15         | Bytes 16..19        | Bytes 20..23 |
| Magic 'U2F2' | formatVer: 2 | profileEnum: 1 | unicodeVerEnum: 1    | scoringVerEnum: 1   | rowCount:u32 |
| (0x55324632) |              |                | (16.0.0)             | (parity-v1)         |              |
+----------------------------------------------------------------------------------------------------+
| Bytes 24..27   | Bytes 28..31 | Bytes 32..35 | Bytes 36..(36+tokens*4) | ...                      |
| tokenCount:u32 | folded:u32   | CRC32 (u32)  | tokens (Uint32Array)    | offsets (Uint32Array)    |
+----------------------------------------------------------------------------------------------------+
```

- **Magic Bytes**: `0x55324632` (ASCII ASCII string `'U2F2'`).
- **Format Version**: `formatVersion = 2`.
- **Profile & Version Enums**: Identifies text profile (`1 = unicode-default`), Unicode version (`1 = 16.0.0`), and scoring version (`1 = parity-v1`).
- **Dimensions & State**: `rowCount`, `tokenCount`, and boolean `folded` flag (1 = case-insensitive fold, 0 = NFC-only).
- **Integrity**: CRC32 checksum over the payload prevents corrupted or partially written buffers from reaching GPU memory.

### Fail-Closed v0.1 Buffer Rejection

Legacy v0.1 serialized buffers lack the `U2F2` magic bytes and layout headers. When passed to `deserializeUnicodeDataset()` or `engine.loadDataset()`, the engine fails closed immediately by throwing an [`IncompatibleIndexError`](file:///home/ubuntu/code/webgpu-fuzzy-search/packages/webgpu-search/src/text-profile.ts#L64-L75):

```ts
IncompatibleIndexError: Incompatible index (expected 2, got legacy-v0.1). Rebuild.
```

No corrupted index data will ever be dispatched to WebGPU compute pipelines.

---

## TypeScript Migration Snippets

### 1. Re-indexing Datasets with `SearchIndex`

`SearchIndex.create()` now accepts Unicode options and immutable index profiles:

```ts
import { SearchIndex } from 'webgpu-search';

// Input dataset: strings containing international scripts, emojis, symbols
const documents: string[] = [
  'packages/core/src/AuthController.ts',
  'src/views/ユーザー設定/Profile.vue',
  'docs/api/Straße_v2.md',
  'assets/icons/🧑‍💻_developer.png'
];

// v0.2 Index Creation
const index = await SearchIndex.create(documents, {
  // Case sensitivity is fixed at pack-time (default: false -> NFC + C+F fold)
  caseSensitive: false,
  // Immutable profile (default: 'unicode-default')
  textProfile: 'unicode-default',
  // Dynamic crossover threshold (default: 30,000 items)
  threshold: 30_000,
  // Force WebGPU if available (default: false)
  preferGpu: false
});

// Search execution
const response = await index.search('ユーザー', {
  mode: 'fuzzy',
  limit: 20
});

console.log(`Matched ${response.totalMatches} records in ${response.timings.totalMs.toFixed(2)}ms via ${response.engine}`);
for (const hit of response.results) {
  console.log(`[Score: ${hit.score}] #${hit.index}: ${hit.text}`);
}
```

### 2. Binary Serialization (`serializeUnicodeDataset` / `deserializeUnicodeDataset`)

For caching pre-packed indexes in IndexedDB, LocalStorage, or disk storage:

```ts
import {
  packUnicodeToGPUBuffer,
  serializeUnicodeDataset,
  deserializeUnicodeDataset,
  WebGPUEngine
} from 'webgpu-search';

// 1. Pack dataset into normalized u32 tokens and offsets
const packed = packUnicodeToGPUBuffer(documents, {
  folded: true, // case-insensitive fold
  profileId: 'unicode-default'
});

// 2. Serialize to U2F2 ArrayBuffer (includes 36-byte header + CRC32)
const serializedBuffer: ArrayBuffer = serializeUnicodeDataset(packed);

// 3. Deserialize and validate (asserts magic, versions, monotonicity, CRC32)
const restored = deserializeUnicodeDataset(serializedBuffer);

// 4. Load directly into WebGPU VRAM without CPU re-normalization
const engine = new WebGPUEngine();
await engine.init();
await engine.loadDataset(restored);
```

### 3. Reading Version & Telemetry via `index.getStats()`

Inspect index metadata and VRAM consumption with zero guesswork:

```ts
const stats = index.getStats();

console.log('--- Index Telemetry ---');
console.log(`Dataset Size:       ${stats.size.toLocaleString()} items`);
console.log(`Post-Fold Tokens:   ${stats.tokenCount.toLocaleString()} code points`);
console.log(`VRAM Allocation:    ${(stats.vramAllocatedBytes / (1024 * 1024)).toFixed(2)} MB`);
console.log(`Serving Engine:     ${stats.engine}`);
console.log(`Text Profile:       ${stats.profileId}`);       // 'unicode-default'
console.log(`Unicode Version:    ${stats.unicodeVersion}`);    // '16.0.0'
console.log(`Scoring Version:    ${stats.scoringVersion}`);    // 'parity-v1'
console.log(`Format Version:     ${stats.formatVersion}`);     // 2
console.log(`Folded:             ${stats.folded}`);            // true | false
```

---

## Before / After Comparison Table

| Dimension | v0.1 (Legacy) | v0.2 (Unicode Code-Point-Safe) | Impact / Remediation |
|---|---|---|---|
| **Matching Unit** | UTF-16 code units (`charCodeAt`) | Unicode scalar values (`codePointAt`) | Astral characters, emojis, and symbols no longer split across surrogate boundaries. |
| **Character Sanitization** | NFKD + strips U+0300..U+036F + replaces non-ASCII with `?` | `trim` → `toWellFormed()` (U+FFFD fallback) → NFC → full C+F fold → NFC | Non-Latin scripts (CJK, Arabic, Indic, Cyrillic) and marks survive without destructive replacement. |
| **Case Insensitivity** | ASCII-only A-Z lowercase (`toLowerCase()`) | Pinned `CaseFolding-16.0.0` (C+F only, 1,557 mappings) | Full multi-scalar folds supported (e.g., `ß` ↔ `ss`, `ﬀ` ↔ `ff`, `ς`/`σ`/`Σ` merge). |
| **Canonical Equivalence** | Distinct unless identical code units | NFC canonical equivalence | Decomposed forms (`e` + `\u0301`) match precomposed forms (`é`). |
| **Query Length Limit** | Implicit 59 UTF-16 unit truncation in uniform | Explicit 128 post-fold code point capacity | Queries > 128 tokens fail-fast with `QueryTooLongError` or route to CPU via `onQueryTooLong: 'cpu-fallback'`. |
| **GPU Memory Representation** | Byte-packed records + byte offsets | `u32` scalar tokens + token offsets | True VRAM budget accounting; per-buffer allocation checks against device limits. |
| **Fuzzy Matching Semantics** | CPU-only `uFuzzy` rank scoring (`1000 - len * 2`) | Unified integer subsequence parity scorer | Bit-exact score and ranking symmetry between WebGPU compute shader and CPU reference fallback. |
| **Tie-Breaking Rule** | Undefined / non-deterministic sort | Strict deterministic `(score DESC, index ASC)` | Identical result ranking across WebGPU and CPU reference runs. |
| **Deprecated Options** | `IndexOptions.slotBytes` (ignored) | `IndexOptions.slotBytes` (throws `IncompatibleOptionError`) | Remove `slotBytes` from index options; scheduled for complete removal in v0.3. |

---

## `TextProfile` Call-Site Contract & Semantic Disclaimers

### 1. Code Points vs Grapheme Clusters
Matching operates strictly on **Unicode scalar values (code points)**, **not user-perceived grapheme clusters**. 
- A composite glyph such as `👨‍👩‍👧‍👦` (Family: Man, Woman, Girl, Boy) consists of 7 code points joined by Zero Width Joiners (`U+200D`).
- Searching for `👩` matches inside `👨‍👩‍👧‍👦` as a code-point subsequence.
- Callers requiring linguistic grapheme cluster boundaries should segment strings using JavaScript's native [`Intl.Segmenter`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Intl/Segmenter) prior to indexing or scoring.

### 2. Delimiter Set & Word-Boundary Bonuses
In fuzzy matching (`mode: 'fuzzy'`), a +30 bonus is awarded when a query character matches immediately following a delimiter. In v0.2, delimiters are strictly defined as the **7-character ASCII delimiter set**:
```
/   _   -   .   space   :   \
```
- Non-ASCII spaces (such as ideographic space `U+3000`) and non-Latin word boundaries (such as Japanese punctuation `、` or `。`) receive no word-boundary bonus in v0.2.
- Delimiter behavior is bit-identical between the WebGPU compute shader (`fuzzy.wgsl`) and the CPU parity scorer (`cpu-reference.ts`).

### 3. Locale-Neutral Case Folding
Default casing uses Unicode 16.0.0 Common and Full (`C+F`) case folding. It is strictly **locale-neutral**:
- Turkic dotted/dotless `I` rules (`tr-TR`, `az-AZ`) are **not applied**.
- Capital `I` (`U+0049`) always folds to lowercase `i` (`U+0069`), never dotless `ı` (`U+0131`).
- Dotted capital `İ` (`U+0130`) folds to `i` + combining dot above (`U+0069 U+0307`).

### 4. Non-Goals & Disclaimers
The v0.2 engine is intentionally bounded:
- **No Automatic Transliteration**: Cyrillic `а` (`U+0430`) does not match Latin `a` (`U+0061`).
- **No Typo / Levenshtein Distance Correction**: Matching is pure subsequence search (`needle` characters must appear in `haystack` in relative order).
- **No Skeleton / Confusable Matching**: UTS #39 confusables (e.g. Greek question mark `U+037E` vs semicolon `;`) remain distinct.

### 5. The `"Straße"` Display Slice Caveat
Because Unicode Case Folding supports one-to-many character expansions (e.g., German lowercase sharp S `ß` folds to `ss`), **post-fold token indices do not correspond 1:1 with original UTF-16 code unit offsets**:
- Input string: `"Straße"` (6 UTF-16 code units).
- Normalized post-fold tokens: `['s', 't', 'r', 'a', 's', 's', 'e']` (7 scalar tokens).
- A search for `"asse"` will locate matching tokens at post-fold range `[3, 7)`.
- **Caution**: Slicing the original source string `"Straße".slice(3, 7)` would produce `"aße"`, yielding invalid slice boundaries.
- **Remediation**: In v0.2, callers highlighting matches in UI components should re-locate match spans directly in display space against the original text, or use regex/substring matching on `item.text`. Exact bidirectional offset maps are planned for a subsequent release.

---

## Deprecation & Error Handling Reference

v0.2 exports explicit error classes to prevent silent runtime misbehavior:

```ts
import {
  QueryTooLongError,
  IncompatibleIndexError,
  ProfileMismatchError,
  IncompatibleOptionError
} from 'webgpu-search';
```

| Error Class | Trigger Condition | Remediation |
|---|---|---|
| [`QueryTooLongError`](file:///home/ubuntu/code/webgpu-fuzzy-search/packages/webgpu-search/src/text-profile.ts#L47-L58) | Query exceeds `QUERY_TOKENS_MAX` (128 post-fold code points). | Shorten query, or specify `onQueryTooLong: 'cpu-fallback'` in `SearchOptions`. |
| [`IncompatibleIndexError`](file:///home/ubuntu/code/webgpu-fuzzy-search/packages/webgpu-search/src/text-profile.ts#L64-L75) | Attempted to load v0.1 buffer, corrupt header, or mismatched version. | Rebuild index using `packUnicodeToGPUBuffer()` or `SearchIndex.create()`. |
| [`ProfileMismatchError`](file:///home/ubuntu/code/webgpu-fuzzy-search/packages/webgpu-search/src/text-profile.ts#L77-L91) | Query `caseSensitive` differs from index construction mode, or invalid `textProfile`. | Build separate indexes for case-sensitive vs case-insensitive search. |
| [`IncompatibleOptionError`](file:///home/ubuntu/code/webgpu-fuzzy-search/packages/webgpu-search/src/text-profile.ts#L93-L103) | `IndexOptions.slotBytes` provided, or `preferGpu: true` combined with `cpuAlgorithm: 'ufuzzy'`. | Remove `slotBytes` option; do not combine `preferGpu` with CPU-only uFuzzy. |

---

## Upgrading Checklist

- [ ] Update dependency: `npm install webgpu-search@^0.2.0` (or `bun add webgpu-search@^0.2.0`).
- [ ] Remove `slotBytes` from all `SearchIndex.create()` option objects.
- [ ] Purge any persisted v0.1 serialized binary index caches from IndexedDB / local disk.
- [ ] Ensure case-sensitivity requirements are configured at index creation time (`IndexOptions.caseSensitive`).
- [ ] If supporting queries > 128 characters, configure `onQueryTooLong: 'cpu-fallback'`.
- [ ] Replace any legacy calls to `packStringsToGPUBuffer` with `packUnicodeToGPUBuffer`.
- [ ] Re-test UI match highlighting against original strings rather than normalized offsets.
