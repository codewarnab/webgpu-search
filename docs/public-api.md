# Public API Freeze (1.0 Contract)

> Plan ref: `docs/ISSUE-11-PLAN.md` Phase 0.
> Status: frozen for `1.x`. Hosts can pin `^1.0.0` and rely on everything
> documented here: entry points, class contracts, result ordering, response
> echoes, errors, version constants, and the breaking-change / deprecation rules.
> Anything marked **Internal** may change in a minor release.

---

## 1. Entry points

Only these two import paths are public. Deep `src/*` imports are unsupported
and may break in any release.

| Path | Module | Contents |
| --- | --- | --- |
| `webgpu-search` | `packages/webgpu-search/src/index.ts:1` | Full API (this document) |
| `webgpu-search/worker` | `packages/webgpu-search/src/worker/search-worker.ts:35` | Dedicated-worker entrypoint: `startSearchWorker`, `isDedicatedWorker` |

Build outputs per `packages/webgpu-search/tsup.config.ts:1`: `index` + `worker`
entries, each in ESM (`dist/*.js`) + CJS (`dist/*.cjs`) with matching
`.d.ts` / `.d.cts`. `package.json` `exports` maps both entries for
`import` and `require` with types, plus a `typesVersions["*"]["worker"]`
fallback so legacy node10 resolution finds the worker types. `attw --pack`
must pass on every release (see §9).

SSR / Worker / Node.js safety: library code has zero unguarded DOM globals
(`window`, `document`). Worker detection is guarded (`typeof Worker`,
`typeof self`, `globalThis.Bun.isMainThread`) so importing `webgpu-search`
on the server or main thread never touches worker globals; only the
`webgpu-search/worker` entry calls `postMessage` and only inside a dedicated
worker scope.

---

## 2. Stable Public API (frozen for 1.x)

These symbols are covered by `scripts/check-public-api.ts` and must not be
renamed, removed, retyped, or behaviorally changed except in a major release.

### 2.1 Index classes

```ts
import { SearchIndex, DocumentIndex, SearchWorkerClient } from 'webgpu-search';
```

**`SearchIndex`** (`packages/webgpu-search/src/search-index.ts:65`) — flat
string corpus:

| Member | Signature | Notes |
| --- | --- | --- |
| `create` | `static create(items: string[], options?: IndexOptions): Promise<SearchIndex>` | `threshold` default 30,000; `preferGpu` default auto-route; `textProfile` default `'unicode-default'`; `slotBytes` throws `IncompatibleOptionError` |
| `search` | `search(query: string, options?: SearchOptions): Promise<SearchResponse>` | See §§4–5 |
| `getStats` | `getStats(): IndexStats` | `formatVersion` = live dataset format (2), not source version |
| `rebuildGpu` | `rebuildGpu(opts?: { device?: GPUDevice; powerPreference?: GPUPowerPreference }): Promise<boolean>` | Re-acquire + re-upload retained corpora; `false` for CPU-by-design / empty; throws only when destroyed |
| `destroy` | `destroy(): void` (idempotent) | Unmap-before-destroy, releases shared pool, unsubscribes device-loss |
| `[Symbol.dispose]` | `() => void` | `using` support; identical to `destroy()` |

**`DocumentIndex<TDoc>`** (`packages/webgpu-search/src/document-index.ts:177`) —
multi-field documents with filters, facets, mutations, snapshots:

| Member | Signature | Notes |
| --- | --- | --- |
| `create` | `static create<TDoc>(records: TDoc[], options: DocumentIndexOptions<TDoc>): Promise<DocumentIndex<TDoc>>` | `fields` non-empty; `idField` default `'id'`; weights `> 0` finite |
| `search` | `search(query: string, options?: DocumentSearchOptions<TDoc>): Promise<DocumentSearchResponse<TDoc>>` | See §§4–5 |
| `autocomplete` | `autocomplete(query: string, options?: AutocompleteOptions): Promise<SuggestResponse<TDoc>>` | `prefix` → `completion`, `fuzzy` → `did-you-mean`; index-wide by design |
| `add` / `update` / `remove` / `applyBatch` | `(docs / ids / batch) => Promise<MutationResult>` | `added/updated/removed/mutationEpoch/compacted/durationMs`; duplicate ids throw `DuplicateIdError` |
| `serialize` | `serialize(options?: SerializeDocumentIndexOptions): ArrayBuffer` | Canonical v4 only; `decoupled: true` omits docs |
| `restore` | `restore(buffer: ArrayBuffer, options?: RestoreDocumentIndexOptions<TDoc>): Promise<void>` | Accepts v3 legacy (read-only) + v4; fail-closed on version/profile/checksum/shape |
| `fromSnapshot` / `fromSnapshotData` | `static fromSnapshot(buffer, …) / fromSnapshotData(data, …)` | Same guards as `restore` |
| `rebuildGpu` | Same shape as `SearchIndex` | Same semantics |
| `getStats` | `getStats(): DocumentIndexStats` | `formatVersion` = live snapshot format (4) |
| `destroy` / `[Symbol.dispose]` | idempotent | Same semantics as `SearchIndex` |

**`SearchWorkerClient<TDoc>`**
(`packages/webgpu-search/src/worker/worker-client.ts:72`) — off-thread
`DocumentIndex` over `postMessage`:

| Member | Signature | Notes |
| --- | --- | --- |
| `constructor` | `new SearchWorkerClient(options?: WorkerClientOptions)` | `stringIsolated` default `true`; custom `worker` / factory for Node/SSR |
| `init` | `init(options: DocumentIndexOptions): Promise<void>` or `init(records, options): Promise<void>` | `hooks` rejected fail-closed (`IncompatibleHookError`); getters stripped pre-clone |
| `search` | `search(query: string, options?: DocumentSearchOptions): Promise<DocumentSearchResponse>` | Monotonic sequencing: rapid typing aborts superseded, last wins; caller `signal` honored; function `filter` applied host-side post-clone |
| `add` / `update` / `remove` / `applyBatch` | Same shapes as `DocumentIndex` | Routed via `MUTATE` |
| `serialize` / `restore` | `(options?) => Promise<ArrayBuffer>` / `(buffer, options?) => Promise<void>` | Same version guards; buffers transferable |
| `getStats` | `getStats(): Promise<DocumentIndexStats>` | Same `formatVersion` semantics |
| `destroy` | `destroy(): Promise<void>` (idempotent) | Sends `DESTROY`, terminates owned worker |
| `[Symbol.asyncDispose]` | `() => Promise<void>` | `await using` support |

Worker rejects `hooks` (any real hook) on `init` / `search` / `restore` with
`IncompatibleHookError` (closures cannot cross the boundary; empty `{}` is a
no-op). Function `filter` + `facets` over the worker drops `facets`
fail-closed (facets would reflect the unfiltered set).

### 2.2 Core types (frozen field sets; additive fields only in minor)

`packages/webgpu-search/src/types.ts:12`:

- `SearchMode` (`'fuzzy' | 'substring' | 'token' | 'prefix'`), `EngineType`,
  `CpuScorer` (`'exact' | 'ufuzzy'`), `FallbackReason` (§5).
- `SearchOptions` / `DocumentSearchOptions`: `mode` (default `'fuzzy'`),
  `limit` (default 50, clamped `1..8192`), `maxResults` alias (§7),
  `caseSensitive` (default `false`, must match pack-time mode),
  `signal`, `cpuScorer` (default `'exact'`), `onQueryTooLong` (default
  `'throw'`), `tokenMatch`, `prefixMatch`, `typoTolerance`, `budget`,
  `diagnostics` (boolean, default `false`).
- `SearchResultItem` (`types.ts:63`): `{ index, score, text }` — `score`
  descending normalized integer, higher is better.
- `SearchResponse` / `DocumentSearchResponse` (§5 echoes), `SearchTimings`,
  `QueryDiagnostics`.
- `IndexOptions` / `DocumentIndexOptions`, `IndexStats` /
  `DocumentIndexStats`, `WorkerClientOptions`.
- Filters / facets / ranking / autocomplete / hooks / budgets: `FilterExpression`,
  `FacetRequest` / `FacetResult`, `DeterministicRankingOptions`,
  `AutocompleteOptions` / `SuggestionItem`, `SearchHooks` /
  `ExtensionHookIds`, `CostBudgetOptions`.
- Snapshots / IDB: `DocumentIndexSchema`, `SerializeDocumentIndexOptions`,
  `RestoreDocumentIndexOptions`, `DocumentSnapshotHeader`, `IDBStorageOptions`.

### 2.3 Errors (frozen names + catch hierarchy)

`packages/webgpu-search/src/errors.ts:1` (+ re-exports in `text-profile.ts:68`):

| Class | When | Catch via |
| --- | --- | --- |
| `WebGPUSearchError` | base for all library errors | `instanceof WebGPUSearchError` |
| `IncompatibleIndexError` | snapshot version/profile/checksum/shape mismatch, oversize caps, missing getter/hookIds | base |
| `IncompatibleHookError` | missing snapshot hooks, worker hook rejection | base |
| `ProfileMismatchError` | `caseSensitive` / `textProfile` / `prefixMatch.exactCase` mismatch | base |
| `IncompatibleOptionError` | unknown `mode` / `cpuScorer`, `ufuzzy` conflicts, removed aliases (`slotBytes`, `cpuAlgorithm`) | base |
| `QueryTooLongError` | `> 128` post-normalization tokens (`extends RangeError`) | `RangeError` or base-independent |
| `DuplicateIdError` / `DocumentNotFoundError` | mutation id conflicts | base |
| `CostBudgetExceededError` | `budget.maxExecutionTimeMs` / `maxCandidates` exceeded (`budgetType` field) | base |
| `InvalidFilterError` | bad filter syntax / unindexed field | base |
| `AbortError` | `signal` / `budget.abortSignal` abort (`DOMException` when available, else `Error` with `name: 'AbortError'`) | `err.name === 'AbortError'`; never converts to fallback |

`AbortError` never becomes a `fallbackReason`. Budget overruns throw
fail-closed (work discarded, never partials).

### 2.4 Frozen version constants

`packages/webgpu-search/src/text-profile.ts:8`:

| Constant | Value | Meaning |
| --- | --- | --- |
| `UNICODE_VERSION` | `'16.0.0'` | normalization / fold-table version |
| `SCORING_VERSION` | `'parity-v1'` | echoed on every response; scoring change → major |
| `DATASET_FORMAT_VERSION` / `DATASET_MAGIC` | `2` / `0x55324632` | packed token-stream wire format |
| `SNAPSHOT_FORMAT_VERSION` / `SNAPSHOT_MAGIC` / `SNAPSHOT_HEADER_BYTES` | `4` / `0x55324434` / `56` | canonical write path |
| `LEGACY_SNAPSHOT_VERSION` / `LEGACY_SNAPSHOT_MAGIC` / `LEGACY_SNAPSHOT_HEADER_BYTES` | `3` / `0x55324433` / `48` | read-only migration path |
| `QUERY_TOKENS_MAX` | `128` | post-normalization code points incl. spaces |
| `RESULT_LIMIT_MAX` | `8192` | clamp ceiling; overflow flag, not truncation |

`getStats().formatVersion` reports the **live** format (dataset 2 /
snapshot 4), never the source snapshot version.

### 2.5 Frozen helpers hosts may rely on

Ranking (`src/ranking.ts:32`): `DEFAULT_TIE_BREAKERS`, `normalizeTieBreakers`,
`compareRanked`, `sortRanked`, `compareIdsAsc`, `isExactTokenMatch`.
Highlight (`src/highlight.ts`): `alignHighlights`, `normalizeWithSourceMap`,
`guardClusterBoundary`, `mergeHighlightRanges`, `renderHighlightedText`.
Hooks (`src/hooks.ts`): `defaultTokenizer`, `codeTokenizer`,
`normalizeSearchHooks`, `resolveEffectiveHooks`, `hasAnyHook`, `getHookId`,
`collectHookIds`, `assertHooksSatisfied`, `tokenizeWithHook`,
`getTokenTermsForQuery`, `applyScoringHook`, `applyPostProcess`.
Diagnostics (`src/diagnostics.ts`): `normalizeCostBudgetOptions`,
`assertTimeBudget`, `assertCandidateBudget`, `computeFilterSelectivity`,
`isBroadQueryHeuristic`, `isBroadSelectivity`, warning builders, `BROAD_SEARCH_*`
constants. Guards (`src/guard.ts`): `clampLimit`, `DEFAULT_LIMIT`,
`VALID_SEARCH_MODES`, `assertValidMode`, `abortError`, `nowMs`,
`throwIfAborted`. Text (`src/text-normalization.ts`): `normalizeText`,
`toWellFormedSafe`, `tokensEqual`. Snapshot high-level
(`src/snapshot-codec.ts`): `encodeSnapshot`, `decodeSnapshot`,
`decodeSnapshotHeader`, `restoreSnapshot`, `serializeDocumentIndex`,
`deserializeDocumentSnapshot`, `restoreDocumentIndex`, `MAX_SNAPSHOT_*` caps.
IDB (`src/snapshot-idb.ts`): `openSearchDatabase`, `saveIndexToIDB`,
`loadIndexFromIDB`, `deleteIndexFromIDB`, `restoreIndexFromIDB` + `DEFAULT_*`
names. Autocomplete (`src/autocomplete.ts`): `normalizeAutocompleteOptions`,
`AUTOCOMPLETE_*` constants. Worker (`src/worker/`): `startSearchWorker`,
`isDedicatedWorker`, `serializeError`, `deserializeError`.

---

## 3. Power-user Public (stable, low-level — prefer §2)

Engines, packing, columnar/facet internals, per-mode scorers, and the worker
protocol are exported for benchmarks and power users but are **not** covered
by the ordering/echo freeze beyond the shared `SearchResultItem` integer-score
contract. Prefer `SearchIndex` / `DocumentIndex` / `SearchWorkerClient`.

`WebGPUEngine`, `CPUEngine`, `GpuDevicePool`, `WebGPUEngine` context
(`WebGPUContextManager`, `assertValidPowerPreference`), `pack*` /
`serialize*` / `deserialize*` / `validatePackedOffsets` / `crc32Parts` /
`checkMemoryBudget` / `computeClampedHeadroomBytes`, `DocumentBitset`,
`ColumnarStore`, `compileFilter`, `FacetEngine` + helpers, `aggregateDocMatches`
/ `accumulateDocMatch`, exact/token/prefix/typo scorers
(`scoreExactMatches`, `scoreFuzzyTokens`, `scoreSubstringTokens`,
`scoreSubstringTypoTokens`, `scoreTokenTokens`, `scorePrefixTokens`, …),
`transitionEngineState` / `initialEngineState` / `isWebGpuState`, worker
protocol payloads (`Worker*Payload`, `SerializedWorkerError`).

Device acquisition must go through `WebGPUContextManager` / `GpuDevicePool`
or an injected `options.device` — never bare `navigator.gpu.requestDevice()`
per index.

---

## 4. Result ordering (frozen)

Primary key: `score` **DESC** (normalized integer, higher is better).
Ties break deterministically, identically on WebGPU readback, exact CPU,
worker, and snapshot-restore paths (`src/ranking.ts:1`):

1. `score` DESC — integer match score.
2. `weight` DESC — higher-weighted field wins (`DocumentIndex`; flat
   `SearchIndex` weight is uniform so this tier is a no-op there).
3. `exact` DESC — full post-normalization token equality beats partial/fuzzy.
4. `length` ASC — shorter matched-field token span wins.
5. `id` ASC — `compareIdsAsc`: numbers numerically, otherwise `String(id)`
   code-unit order (never locale collation). Numeric `2` vs string `"2"` are
   id-equal and fall through.
6. Implicit final: `docIndex` ASC (insertion order; defensive, ids unique).

Properties: wrap-free comparisons (no `|0` subtraction), code-unit string
order, fail-closed `TypeError` on non-finite score/weight/length/id keys
(never implementation-defined `Array.sort` order). Custom
`ranking.tieBreakers` subsets are honored; unlisted criteria are skipped and
`docIndex` ASC remains the total-order fallback. `DocumentIndex` default is
the full five-tier hierarchy; pass `ranking: { tieBreakers: ['score'] }` to
approximate legacy `(score DESC, docIndex ASC)` (exact legacy order is not
bit-reproduced when ids differ from insertion order). Autocomplete ranks per
`(doc, field)` row with the same comparator (a document may appear multiple
times in suggestions). `totalMatches` / `candidateCount` / `hasOverflow` are
snapshotted pre-pipeline (`scoringHook` / `postProcess` re-sort `results`
only).

---

## 5. Response echoes (frozen)

Every `SearchResponse` / `DocumentSearchResponse`
(`packages/webgpu-search/src/types.ts:83`) carries:

| Field | Value | Notes |
| --- | --- | --- |
| `query` | verbatim input | — |
| `mode` | requested mode | validated fail-closed |
| `engine` | `'webgpu' \| 'cpu'` | which engine served this query |
| `totalMatches` | count passing threshold | pre-limit, pre-pipeline |
| `candidateCount` | `min(totalMatches, pool capacity)` | **not** `results.length` |
| `hasOverflow` | `totalMatches > pool capacity` | limit truncation alone leaves `false` |
| `results` | Top-K per §4 | `length <= clampLimit(limit ?? maxResults ?? 50)` |
| `timings` | phase latencies | `gpuExecutionMs: null` when unsupported; `totalMs` excludes inline autocomplete |
| `profileId` | `'unicode-default'` | text profile that served the query |
| `scoringVersion` | `'parity-v1'` | scoring contract version |
| `cpuScorer` | requested scorer | `'exact'` default; echo even on GPU path |
| `fallbackReason?` | `FallbackReason` | present iff CPU served for a GPU-avoidance reason |
| `diagnostics?` | `QueryDiagnostics` | only when `options.diagnostics: true` |
| `facets?` / `suggestions?` | document responses only | when requested |

`FallbackReason` (`types.ts:268`): `'webgpu-unsupported'`,
`'device-request-failed'`, `'memory-budget-exceeded'`, `'device-lost'`,
`'below-threshold'`, `'prefer-cpu'`, `'query-too-long'`, `'unsupported-mode'`
(token/prefix/typo queries — WGSL kernels are exact-only for
fuzzy/substring), `'cpu-algorithm-requested'` (explicit `ufuzzy` opt-in),
`'gpu-execution-error'`. GPU-unavailable or GPU-lost queries match the
documented CPU contract (same matches/order/echoes); `AbortError` never
converts to fallback.

`cpuScorer: 'ufuzzy'` is explicit opt-in, CPU-only, non-conforming scores,
never routes to WebGPU, excluded from the differential matrix, conflicts with
`preferGpu: true` (`IncompatibleOptionError` at `search()`).

---

## 6. Internal (do not import — may change in minor)

- `INTERNAL_WORKER_ID_KEY` (`worker-client.ts:64`): worker-cloned `idField`
  marker. The `INTERNAL_` prefix is the rule: any `INTERNAL_*` export is
  internal by definition (`check-public-api` enforces the prefix allowlist).
- Case-fold tables: `CASE_FOLD_*` / `FOLD_*` ranges, counts, versions,
  `foldCaseScalar`, `foldCodePoint` (`src/case-fold-table.ts`) — regenerable
  via `generate-case-fold-table.ts`, versioned by `UNICODE_VERSION`.
- Deprecated aliases: `BROAD_QUERY_SELECTIVITY_THRESHOLD`,
  `BROAD_QUERY_MIN_DOCS`, `BROAD_QUERY_SHORT_QUERY_TOKENS` (use `BROAD_SEARCH_*`).
- Engine/packing/protocol internals beyond §2.5 when imported for debugging
  (pool refcounts, packed offsets, worker wire payloads).

If you need an internal for a real use case, file an issue — promotion to
§2.5 is a minor, removal of a public symbol is a major.

---

## 7. Breaking-change rule (frozen)

A **major** version is required for any of:

- scoring semantic change (`SCORING_VERSION` bump, score formula / normalization /
  threshold change, tie-breaker reorder or default change);
- storage semantic change (snapshot/dataset magic, version, header layout, CRC
  coverage, `MAX_SNAPSHOT_*` tightening that rejects previously valid snapshots,
  `hookIds` / `hasGetter` semantics);
- result ordering change (§4) or echo change (§5: rename/remove/retype an echo,
  change `fallbackReason` mapping);
- public symbol removal/rename/retype, `IndexOptions` / `SearchOptions` default
  change (`threshold`, `limit`, `cpuScorer`, `preferGpu`), error name/hierarchy
  change, worker protocol incompatibility.

A **minor** may add: new public symbols, new `SearchOptions` / index options
(fail-closed defaults, ignored by old readers), new `diagnostics` /
`timings` fields, new `FacetRequest` types, new error subclasses catchable via
the existing base, promotion of an Internal to §2.5. A **patch** fixes bugs
while preserving the contract (same matches/order/echoes on conforming inputs).

Removed-alias history (all required a major; template for future removals):
`cpuAlgorithm` → `cpuScorer`, `extensions` → `hooks`, `suggest` →
`autocomplete`, `slotBytes` → removed (throws), `U2D4_*` / `FORMAT_VERSION*` /
`SERIALIZED_*` → `DATASET_*` / `SNAPSHOT_*` / `LEGACY_SNAPSHOT_*`,
`BROAD_QUERY_*` → `BROAD_SEARCH_*` (alias shim, removal reserved for a major).

---

## 8. Deprecation policy (frozen)

1. Announce in the minor that introduces the replacement: `@deprecated`
   JSDoc (naming the replacement + removal target), a note in this doc, and a
   `CHANGELOG` entry. Runtime warnings only where cheap (no per-record spam).
2. Minimum removal window: **one full minor line** (deprecated in `1.N`, removed
   no earlier than `2.0`; on `0.x`, no earlier than the next minor). Security /
   correctness hazards may shorten the window with a documented migration.
3. Alias template (`limit` vs `maxResults`, `types.ts:20`): the new name wins
   when both are set (`limit ?? maxResults ?? 50`, then `clampLimit`:
   non-finite → 50, fractions floored, clamped `1..8192`, `null` → 1); the old
   alias keeps identical coercion until removal; removal throws
   fail-closed (`IncompatibleOptionError`) rather than silently changing limits.
   Every future rename follows this shape: both accepted → new wins →
   major-only removal.

---

## 9. Conformance gates

- `bun run check:public-api` (`scripts/check-public-api.ts`): frozen export
  allowlist diff (fails on added/removed public symbols), `INTERNAL_*` prefix
  allowlist, version-constant values, `DEFAULT_TIE_BREAKERS` value, echo-key
  presence in `types.ts`, `limit`/`maxResults` alias preservation, package
  `exports`-map shape. Runs in CI before the browser gate.
- `attw --pack packages/webgpu-search`: type-resolution gate for both entries
  × ESM/CJS. Must pass on every release (see CI "Resolve Package Types (attw)").
- `bun run typecheck`, `bun run build`, `bun run test:mock` (plus `test:contracts`
  §13 + parity Part 7 for fallback/ordering pins).

To propose a public-surface change: update this doc + the allowlist in
`scripts/check-public-api.ts` in the same PR; CI fails otherwise.
