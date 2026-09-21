# Naming Conventions

Domain-first names. No temporal prefixes (no `m1-m8`, `v0.x`, `phase`,
`milestone`), no internal codenames (`U2D3`/`U2D4`/`U2F2`, `V2`/`V4`,
`FORMAT_VERSION_4`), no architecture history (`hybrid`), no implementer
jargon (`fold`, `sanitize`, `parity`) in user-visible identifiers.

## One concept, one term

| Use | Not | Notes |
|---|---|---|
| `normalize` / `normalized` / `caseSensitive` | `fold` / `folded` / `sanitize` | Unicode NFC + default case folding; `folded` stays only as a `@deprecated` alias |
| `cpuScorer: 'exact' \| 'ufuzzy'` | `cpuAlgorithm` / `'parity'` | `exact` describes behavior; `'parity'` maps to `'exact'` with a warning |
| `autocomplete` | `suggest` | User-facing completion feature; `suggest` is a `@deprecated` alias |
| `snapshot` | `U2D*` / `serializeDocumentIndex` vocabulary | Versioned binary persistence; wire magics frozen |
| `dataset` / `packDataset` / `PackedDataset` | `U2F2` / `packUnicodeToGPUBuffer` / `PackedUnicodeBufferV2` | Packed token stream; wire bytes unchanged |
| `hooks` / `SearchHooks` | `extensions` / `SearchExtensionHooks` | `extensions` is a `@deprecated` alias |
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
5. **Aliases, not forks.** Every rename keeps a
   `/** @deprecated Use X. */ export const Old = New` for one minor. Only
   runtime-behavioral aliases log warnings (`packStringsToGPUBuffer`,
   `cpuScorer: 'parity'`); pure type/constant renames are silent.

## Lint

`bun run lint:naming` (`scripts/check-naming.ts`) bans temporal/codename
leaks in sources. It excludes `docs/archive/**`, `@deprecated` alias lines,
and wire-magic comments. Keep it clean.

## Observable renames (intentional, not zero-change)

Canonical renames change these observable surfaces; deprecated aliases cover
imports/options, but exact strings do not round-trip:

- `INTERNAL_WORKER_ID_KEY` stays exported from the package root for compat
  (internal use only).
- `IncompatibleOptionError.option` is `'cpuScorer'` (was `'cpuAlgorithm'`);
  worker hook guard throws `IncompatibleHookError('hooks')` (was `'extensions'`).
  Branch on the canonical names; `cpuAlgorithm`/`extensions` inputs still parse.
- `Unknown autocomplete field` (was `Unknown suggest field`);
  `slotBytes is not supported` (was versioned throw-on-use text).
- `QueryDiagnostics.timings` emits both `autocompleteMs` and deprecated
  `suggestMs` with identical values.
