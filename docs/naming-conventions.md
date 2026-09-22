# Naming Conventions

Domain-first names. No temporal prefixes (no `m1-m8`, `v0.x`, `phase`,
`milestone`), no internal codenames (`U2D3`/`U2D4`/`U2F2`, `V2`/`V4`,
`FORMAT_VERSION_4`), no architecture history (`hybrid`), no implementer
jargon (`fold`, `sanitize`, `parity`) in user-visible identifiers.

## One concept, one term

| Use | Not | Notes |
|---|---|---|
| `normalize` / `normalized` / `caseSensitive` | `fold` / `folded` / `sanitize` | Unicode NFC + default case folding; `folded` was removed from public fields — use `normalized` |
| `cpuScorer: 'exact' \| 'ufuzzy'` | `cpuAlgorithm` / `'parity'` | `exact` describes behavior; `cpuAlgorithm` / `'parity'` were removed (unknown values throw `IncompatibleOptionError`) |
| `autocomplete` | `suggest` | User-facing completion feature; `suggest` option / `SuggestOptions` / `SUGGEST_*` were removed — use `autocomplete` / `AutocompleteOptions` / `AUTOCOMPLETE_*` (`SuggestionItem` / `SuggestResponse` / `autocomplete()` stay canonical) |
| `snapshot` | `U2D*` / `serializeDocumentIndex` vocabulary | Versioned binary persistence; wire magics frozen (`SNAPSHOT_*` / `LEGACY_SNAPSHOT_*`; `U2D4_*` / `SERIALIZED_*` / `FORMAT_VERSION*` removed) |
| `dataset` / `packDataset` / `PackedDataset` | `U2F2` / `packUnicodeToGPUBuffer` / `PackedUnicodeBufferV2` | Packed token stream; wire bytes unchanged (`DATASET_*` canonical; `FORMAT_VERSION` / `SERIALIZED_*` removed) |
| `hooks` / `SearchHooks` | `extensions` / `SearchExtensionHooks` | `extensions` was removed — use `hooks` / `SearchHooks` / `normalizeSearchHooks` |
| `devicePool` / `GpuDevicePool` | `context-manager` / `WebGPUContextManager` | Singleton device pool; old name aliased |
| `guard` | `runtime-guards` | Small cohesive module |
| `broad-search` / `BROAD_SEARCH_*` | `broad-query` / `BROAD_QUERY_*` | Old constants aliased |

## Rules

1. **File name = exported concept, kebab-case.** `search-index.ts`,
   `exact-scorer.ts`, `dataset-packing.ts`, `text-normalization.ts`,
   `snapshot-codec.ts`, `snapshot-idb.ts`, `autocomplete.ts`,
   `gpu-device-pool.ts`, `guard.ts`, `hooks.ts`, `case-fold-table.ts`,
   `search/*`, `filtering/*`, `faceting/*`.
2. **Constants:** `<DOMAIN>_<WHAT>` — `DATASET_MAGIC`,
   `SNAPSHOT_MAGIC`, `SNAPSHOT_HEADER_BYTES`, `LEGACY_SNAPSHOT_*`.
3. **Scripts:** `<verb>-<domain>.ts` — `test-contracts.ts`,
   `check-bundle-size.ts`, `bench-snapshot-matrix.ts`; npm `test:<domain>`.
4. **Comments explain why, never when.** No milestone/version tags in
   `packages/*/src`, `apps/*/src`, `examples/`, `scripts/`.
5. **Breaking cleanup removed the rename aliases.** `cpuAlgorithm`,
   `extensions` / `SearchExtensionHooks`, `suggest` / `SuggestOptions` /
   `SUGGEST_*`, `folded` public fields, `FORMAT_VERSION*` / `U2D4_*` /
   `SERIALIZED_*`, `countUnicodeCodePoints`, `isAsciiTokens` /
   `isPrintableAsciiTokens`, and `CPUEngine.searchUFuzzy` / `searchNative`
   no longer exist — passing them throws fail-closed (`IncompatibleOptionError`
   / `TypeError`). Low-level CPU entry points are `searchWithUFuzzy` /
   `searchNaiveScan`. `SCORING_VERSION = 'parity-v1'` keeps its value
   (differential "parity harness" prose is unchanged).

## Lint

`bun run lint:naming` (`scripts/check-naming.ts`) bans temporal/codename
leaks in sources. It excludes `docs/archive/**`, `@deprecated` alias lines,
and wire-magic comments. Keep it clean.

## Observable renames (intentional, not zero-change)

Canonical renames change these observable surfaces; removed aliases throw
fail-closed instead of round-tripping (exact strings do not round-trip):

- `INTERNAL_WORKER_ID_KEY` stays exported from the package root for compat
  (internal use only).
- `IncompatibleOptionError.option` is `'cpuScorer'` (removed `'cpuAlgorithm'`;
  passing `cpuAlgorithm` throws "was removed; use `cpuScorer`");
  worker hook guard throws `IncompatibleHookError('hooks')` (removed
  `'extensions'`; passing `extensions` throws "was removed; use `hooks`").
  Branch on the canonical names.
- `Unknown autocomplete field` (removed `Unknown suggest field`);
  `slotBytes is not supported` (was versioned throw-on-use text).
- `QueryDiagnostics.timings` emits `autocompleteMs` (deprecated `suggestMs`
  mirror removed).
- `powerPreference` is forwarded to `GpuDevicePool.acquireDevice`
  (`navigator.gpu.requestAdapter({ powerPreference })`); unknown values throw
  `IncompatibleOptionError` fail-closed, even on CPU-only paths.
