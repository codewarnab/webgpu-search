# Diagnostics, Budgets & Issue-Report Data

> Plan ref: `docs/ISSUE-11-PLAN.md` Phase 6.
> API freeze: `docs/public-api.md` §§2.5/5. Field types: `src/types.ts`,
> policy: `src/diagnostics.ts`, guards: `src/guard.ts`.

Enable per-query telemetry with `options.diagnostics: true` (default `false`;
sub-microsecond overhead when disabled — clocks and warning strings are gated).

```ts
const res = await index.search('snapshot', { diagnostics: true });
console.log(res.diagnostics?.timings.totalMs, res.diagnostics?.warnings);
```

---

## 1. `QueryDiagnostics` (`types.ts:791`)

| Field | Meaning |
| --- | --- |
| `scannedCandidates` | Active documents evaluated (docs, not rows; rows = docs × fields) |
| `filterSelectivity` | `matched / total` in `[0, 1]`. No filter (or empty corpus, no filter) reports `1.0`; function-predicate filters report `1.0` (narrowing invisible to telemetry); empty corpus + structured filter reports `0` |
| `routedEngine` | Engine that processed the query (`'webgpu' \| 'cpu'`) |
| `hasOverflow` | Whether matches exceeded candidate-pool capacity |
| `timings` | Phase latencies (§2) |
| `warnings?` | Non-fatal advisories (§3); absent when nothing fired |

## 2. Timings

`SearchTimings` on every response (`types.ts:69`): `queryUploadMs`,
`encodeSubmitMs`, `gpuExecutionMs` (`null` when unsupported), `readbackMs`,
`totalMs` (scorer wall-clock **excluding** inline autocomplete), plus
`gpuDispatchMs` alias of `encodeSubmitMs`.

`QueryDiagnostics.timings` (`types.ts:775`, only with `diagnostics: true`):
`filteringMs` (columnar bitsets; function predicates cost ~0 here — their
evaluation lands in `scoringMs`), `scoringMs` (kernel/CPU scan incl.
post-match scoring hooks), `highlightMs`, `facetingMs?` (absent unless
facets requested; string index never emits), `autocompleteMs?` (absent
unless autocomplete requested), `totalMs` (end-to-end **including**
autocomplete — the one `SearchTimings.totalMs` excludes).

## 3. Warnings (non-fatal; results remain complete + ranked)

Built by `broadQueryRouteWarning` / `broadSelectivityWarning` /
`candidateOverflowWarning` (`src/diagnostics.ts`):

- **Broad-query pre-dispatch route** — `≤ 2`-token post-normalization query
  over `≥ 5000` docs routes to the CPU streaming scan pre-dispatch
  (`isBroadQueryHeuristic`; token count = Unicode code points incl. spaces,
  so this fires only on 1–2 character queries). Suppressed when the query is
  already CPU-by-design (token/prefix/typo modes, explicit ufuzzy).
- **Broad post-hoc selectivity** — match selectivity `> 80%` on `≥ 5000`
  docs (`isBroadSelectivity`, `BROAD_SEARCH_SELECTIVITY_THRESHOLD = 0.8`,
  `BROAD_SEARCH_MIN_DOCS = 5000`, `BROAD_SEARCH_SHORT_QUERY_TOKENS = 2`).
- **Candidate overflow** — matches exceed pool capacity; message names the
  remediation knob (`candidateCapacity`). Facets-clause appended only when
  facets were requested and remain approximate.

Worker caveat: with a **function** `filter` over `SearchWorkerClient`, the
predicate runs host-side post-clone, so `facets` **and** `diagnostics` are
dropped fail-closed (worker-computed values would reflect the unfiltered
set). Callers needing predicate + diagnostics run a local `DocumentIndex`.

## 4. Cost budgets (`CostBudgetOptions`, `types.ts:758`)

```ts
await index.search(q, { budget: { maxExecutionTimeMs: 50, maxCandidates: 10000 } });
```

Validated fail-closed by `normalizeCostBudgetOptions`: non-object budgets
throw `TypeError`; `maxExecutionTimeMs` must be finite `> 0` (`RangeError`
otherwise); `maxCandidates` must be an integer `≥ 1`; unknown `max*` keys
throw `TypeError` (typo'd limits must not silently disable the ceiling);
`abortSignal` passes through (`budget.abortSignal` aborts with `AbortError`).

Enforcement is best-effort at **phase boundaries** (post-filter, post-score,
post-highlight, post-facet): over-budget scans run to completion then throw
`CostBudgetExceededError` (`budgetType: 'time' | 'candidates'`) — work is
discarded, never returned as partials. There is no intra-scan preemption;
callers needing hard deadlines also pass `signal` / `abortSignal` for
cooperative cancellation. Structured filters enforce `maxCandidates`
pre-scan on the exact post-filter population; function predicates enforce on
the pre-predicate population (conservative — post-predicate unknowable).
`AbortError` never converts to a `fallbackReason`.

---

## 5. Issue-report data (copy-paste template)

File reports with `.github/ISSUE_TEMPLATE/webgpu-search-issue.md`. Minimum
actionable set:

1. Snapshot header: `decodeSnapshotHeader(buf)` JSON (magic, formatVersion,
   profile/unicode/scoring, docCount/rowCount/tokenCount, byte lengths, CRC).
2. `getStats()` JSON (engine, `formatVersion`, `fallbackReason`,
   `memory.vramBytes/ramBytes`, `mutationEpoch`).
3. Failing query: verbatim `query`, `mode`, full search options
   (`limit`, `cpuScorer`, `tokenMatch`/`prefixMatch`/`typoTolerance`,
   `budget`, `filter`/`facets` shape — never paste PII record text).
4. Response echoes: `engine`, `fallbackReason`, `profileId`,
   `scoringVersion`, `cpuScorer`, `totalMatches`/`candidateCount`/`hasOverflow`.
5. Timings: `timings` + (`diagnostics: true`) `diagnostics` JSON incl.
   `warnings`.
6. Environment: browser + version, OS, GPU/driver (or `webgpu-unsupported`
   path), `apps/benchmark` URL or Node/Bun version for headless repros,
   package version + entry (`webgpu-search` vs `webgpu-search/worker`).
7. Error: full `name`/`message`/`details` for throws (`IncompatibleIndexError`,
   `ProfileMismatchError`, `CostBudgetExceededError`, …).
