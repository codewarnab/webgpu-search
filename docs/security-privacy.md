# Security & Privacy Boundaries

> Plan ref: `docs/ISSUE-11-PLAN.md` Phase 6.
> Wire compat: `docs/snapshot-format.md`; runtime matrix:
> `docs/support-matrix.md` §§3–4; packaging: `docs/packaging.md`.

---

## 1. What the library is (and is not)

Embeddable, private, offline-capable search over datasets already inside the
host app. No network calls, no telemetry exfiltration, no hosted platform.
The single runtime dependency is `@leeoniya/ufuzzy` (CPU fallback scorer);
there is no WASM, no `eval`, no remote fetch.

## 2. IndexedDB PII scope

- Snapshots persist **raw record bytes unencrypted**, origin-scoped
  (`docs/snapshot-format.md` §7). Production corpora (logs, docs, messages)
  may carry real PII — treat IDB database contents (`openSearchDatabase` /
  `saveIndexToIDB` / `loadIndexFromIDB`, `src/snapshot-idb.ts`) as
  sensitive as the source records.
- `serialize({ decoupled: true })` omits docs (`docsLen 0`); restore then
  requires the external `documents` array or embedded `schema.docIds`
  coverage, else `IncompatibleIndexError`. Use decoupled storage when the
  host already guards docs elsewhere.
- `saveIndexToIDB` serializes fully before opening the transaction;
  connections close in `finally`. Removal is explicit:
  `deleteIndexFromIDB({ dbName, key })` (proof-app "Clear" pattern).
  Uninstall / origin-eviction is the only other erasure path — document it
  in host privacy copy.

## 3. Snapshot trust boundary

- Never read under the wrong format/profile: unknown magic/version/enum,
  non-zero reserved, truncated bytes, CRC mismatch, shape violations, and
  oversize `MAX_SNAPSHOT_*` caps (schema 16 MiB, columnar 64 MiB, docs
  256 MiB, total 512 MiB, docCount 10 M, tokenCount 256 M) all throw
  fail-closed (`IncompatibleIndexError` / `IncompatibleHookError` /
  `ProfileMismatchError`, taxonomy in `docs/snapshot-format.md` §3) with no
  partial index. Restore-supplied hooks replace live hooks wholesale and
  must satisfy persisted `hookIds`; custom filter `getter` closures are
  never serialized (`hasGetter` guard).
- **CRC32 is corruption detection, not authenticity** (`docs/snapshot-format.md`
  §8): a forged snapshot with recomputed CRC is rejected only by
  shape/schema/offset validation. Treat snapshots from untrusted origins as
  untrusted input — restore inside the existing fail-closed path, never via
  a custom decoder that skips validation. Oversize checks run before any
  `TextDecoder` / `JSON.parse` / `slice` allocation.
- IDB and `postMessage` transports move raw bytes; `SearchWorkerClient`
  strips function `hooks` pre-clone and sanitizes restore options so
  closures never cross the boundary.

## 4. Host responsibilities

- Escape/sanitize `highlightedText` before `innerHTML` (proof-app
  `sanitizeHighlighted` pattern); pass `escapeHtml: true` for untrusted
  corpora.
- Surface `fallbackReason` + restore rejects as user-visible status, not
  color alone (see `docs/packaging.md` §6).
- Report issues with header/stats/timings metadata only
  (`docs/diagnostics.md` §5) — never paste PII record text into tickets.

## 5. Unsupported configurations

Authoritative list: `docs/support-matrix.md` §6. Security-relevant
restatements: cross-scorer score comparisons are meaningless (facet
`isApproximate` is engine-relative); SwiftShader/headless numbers are
`pending-hardware`, never hardware claims; function `filter` + `facets`
over the worker drops `facets` fail-closed.
