# Snapshot Format + Index-Format Compatibility (1.0 Contract)

> Plan ref: `docs/ISSUE-11-PLAN.md` Phase 3.
> Status: frozen for `1.x`. The write path is canonical v4 only; the read
> path accepts v3 (legacy, read-only migration) + v4. Any persisted index
> restores or rejects with an actionable migration reason — never reads
> under the wrong format/profile.

Versioned binary persistence for `DocumentIndex` (`encodeSnapshot` /
`index.serialize()`, worker `serialize()`).

---

## 1. Snapshot format versions

- **Legacy v3 (read-only)**: 48-byte LE header, magic `0x55324433`
  (`LEGACY_SNAPSHOT_MAGIC`), version 3 (`LEGACY_SNAPSHOT_VERSION`).
  Layout: `header[48] + schema + tokens + offsets + docs`. No columnar
  segment (`columnarByteLength` reports `0`).
- **Canonical v4 (write path, 1.0)**: 56-byte LE header, magic `0x55324434`
  (`SNAPSHOT_MAGIC`), version 4 (`SNAPSHOT_FORMAT_VERSION`, 8-byte aligned).
  Layout: `header[56] + schema + tokens + offsets + columnar + docs`.
  Header words: magic@0, version@4, profile@8, unicode@12, scoring@16,
  docCount@20, rowCount@24, tokenCount@28, normalized@32, schemaLen@36,
  docsLen@40, columnarLen@44, reserved(0)@48, CRC32@52 covering
  `header[0..52) + schema + tokens + offsets + columnar + docs`.

Magic bytes are frozen: old snapshots stay readable forever via the
migration read path. Only the exported constant names changed
(`SNAPSHOT_MAGIC`, `SNAPSHOT_FORMAT_VERSION`, `SNAPSHOT_HEADER_BYTES`;
legacy: `LEGACY_SNAPSHOT_MAGIC`, `LEGACY_SNAPSHOT_VERSION`,
`LEGACY_SNAPSHOT_HEADER_BYTES`). The old names (`FORMAT_VERSION*`,
`U2D4_*`, `SERIALIZED_*`) were removed — use `DATASET_*` /
`SNAPSHOT_*` / `LEGACY_SNAPSHOT_*`.

`1.0` ships v4 as the canonical wire format. A storage-semantic change
(new magic, version, header layout, CRC coverage, `MAX_SNAPSHOT_*`
tightening that rejects previously valid snapshots, `hookIds` /
`hasGetter` semantics) requires a **major** version per
`docs/public-api.md` §7.

---

## 2. Compatibility + migration table (v3 → v4 → 1.0)

| Source | Writer | Reader (this release) | Result | Migration |
| --- | --- | --- | --- | --- |
| v3 legacy (48 B, `0x55324433`) | pre-v4 releases | `decodeSnapshot` / `restoreSnapshot` / `DocumentIndex.fromSnapshot` / `fromSnapshotData` / instance `restore()` / `SearchWorkerClient.restore` | ✅ Restores (migration read path). `columnarByteLength: 0`; authoritative filtering rebuilds via `ColumnarStore.init(records)` | Re-`serialize()` to upgrade to v4; no host action unless filters/hooks were added (see §§4–5) |
| v4 canonical (56 B, `0x55324434`) | `encodeSnapshot` / `index.serialize()` / worker `serialize()` (this release) | Same read set | ✅ Restores; byte-length must equal `header + schema + tokens + offsets + columnar + docs`, CRC must match, shapes must validate | None — round-trips identically |
| v4 with `decoupled: true` (`docsLen 0`) | `serialize({ decoupled: true })` | `fromSnapshot(buf, { documents })` / IDB decoupled stores | ✅ Restores iff `options.documents` length matches `header.docCount` or embedded `schema.docIds` cover the docs; else `IncompatibleIndexError` | Supply the external docs array or re-serialize embedded |
| v3/v4 with custom filter getters | Writer records `hasGetter: true` per field | Reader without a matching `getter` override | ❌ `IncompatibleIndexError` (`filter-getter:<name>`) | Re-restore with `options.options.filterFields` carrying the same getters |
| v4 with `hookIds` | Writer records declarative `hookIds` (never closures) | Reader without matching `options.options.hooks` handlers (or mismatched `hookId`) | ❌ `IncompatibleHookError` | Supply matching handlers (`myFn.hookId = 'stable-v1'`) and re-restore |
| v4 with mismatched `caseSensitive` / `textProfile` override | Any writer | `fromSnapshot(buf, { options: { caseSensitive } })` disagreeing with `!header.normalized`, or `textProfile !== 'unicode-default'`; instance `restore()` where live `!normalized` disagrees with snapshot | ❌ `ProfileMismatchError` (`caseSensitive` / `textProfile`) | Rebuild the index under the snapshot polarity or re-restore without the override — never reinterpret tokens |
| Unknown magic / version / enum / non-zero reserved / truncated / CRC / shape / oversize | Forged, corrupt, or future-major bytes | `decodeSnapshotHeader` / `decodeSnapshot` | ❌ `IncompatibleIndexError` (fail-closed, no partial index) | Rebuild from source records; future-major snapshots require a reader that knows the new version |
| v4 read by an older v3-only reader | This release | Pre-v4 host pinned to the legacy magic | ❌ `IncompatibleIndexError` on the old reader | Upgrade the reader; v4 is the 1.0 write path |

Write path is **canonical-only** (`encodeSnapshot`, `index.serialize()`,
worker `serialize()`). Older readers accepting only the legacy magic throw
`IncompatibleIndexError` fail-closed.

---

## 3. Reject-reason taxonomy (standardized)

| Category | Error | When | Actionable message carries |
| --- | --- | --- | --- |
| Version / magic / header width | `IncompatibleIndexError` | Unknown magic, `formatVersion !== 3\|4`, truncated `< 48 B`, v4 `< 56 B`, non-zero `reserved`, `schema.caseSensitive !== !header.normalized` | `expected` = known magic/version/`0`/polarity, `actual` = offending word; hosts rebuild or upgrade the reader |
| Profile enums | `IncompatibleIndexError` | Unknown `profile` / `unicode` / `scoring` enum in the header | `expected 'valid-enum'`, `actual` = bad enum; future profiles require a new major reader |
| Checksum | `IncompatibleIndexError` | `crc32Parts(...)` mismatch (single-bit schema/columnar/offset/docs tamper) | `expected` = header CRC, `actual` = recomputed; rebuild from source |
| Shape | `IncompatibleIndexError` | Length mismatch, bad `schema.fields` / `hookIds` envelope, columnar field/row mismatch, stripped columnar on filtered non-empty index, non-monotonic offsets, `docIds`/`records` length mismatch, decoupled docs missing, oversize `MAX_SNAPSHOT_*` caps | `expected` = schema/row/field contract, `actual` = offending value; fix the producer or drop the tampered bytes |
| Missing filter getter | `IncompatibleIndexError` | Snapshot `filterFields[].hasGetter: true` without a matching `getter` override | `expected 'filter-getter:<name>'`; supply `options.options.filterFields` with the getter |
| Missing / mismatched hooks | `IncompatibleHookError` | Snapshot `hookIds` without matching `options.options.hooks` handlers, or `hookId` mismatch | `hookId` + `reason`; supply the recorded handler (`hookId`, `function.name`, or `'anonymous'`) |
| Case / text profile | `ProfileMismatchError` | `caseSensitive` override disagreeing with the snapshot (`override === header.normalized`), `textProfile !== 'unicode-default'`, instance `restore()` where live polarity disagrees with snapshot polarity, `prefixMatch.exactCase` disagreement at query time | `property` = `'caseSensitive'` / `'textProfile'` / `'prefixMatch.exactCase'`; rebuild under one polarity instead of varying per query/restore |

`AbortError` never surfaces from decode/restore. Budget overruns throw
`CostBudgetExceededError` at query time, never as partial restores.

---

## 4. Filter getters

- Custom `filterFields[].getter` closures are never serialized. The snapshot
  persists `hasGetter: true` per field; restore requires a matching getter
  override via `options.options.filterFields` or throws
  `IncompatibleIndexError` fail-closed (mirrors the `hookIds` idiom).
- Plain `doc[name]` fields (`hasGetter` absent/false) restore without overrides.
- Columnar segment is advisory + integrity-checked; authoritative filtering
  rebuilds via `ColumnarStore.init(records)`. A stripped empty segment on a
  filtered non-empty index throws `IncompatibleIndexError`.

---

## 5. Hooks (`hookIds`)

- `serialize()` records only declarative `ExtensionHookIds` (stable `hookId`
  property when set and non-blank, else `function.name`, else `'anonymous'`);
  closures never cross the snapshot boundary.
- `restore` / `fromSnapshot` / `fromSnapshotData` / instance `restore()`
  require matching handlers via `options.options.hooks` or throw
  `IncompatibleHookError` (see `assertHooksSatisfied` in `hooks.ts`).
  Restore-supplied handlers replace live hooks wholesale (not per-key merged).
- Name-derived IDs can collide; assign explicit `hookId`
  (`myScorer.hookId = 'recency-v1'`) for any persisted hook.
- Worker boundary: `SearchWorkerClient` `init` / `search` / `restore` reject
  real hooks with `IncompatibleHookError` (closures cannot be cloned; empty
  `{}` is a no-op).

---

## 6. Profile guarantee (never read under the wrong format/profile)

- `decodeSnapshot` rejects unknown magics/versions/enums, non-zero reserved,
  and `schema.caseSensitive` / `header.normalized` disagreement with
  `IncompatibleIndexError` — tampered or future-major bytes never decode.
- `fromSnapshot` / `fromSnapshotData` reject a `caseSensitive` override that
  disagrees with the snapshot polarity and any
  `textProfile !== 'unicode-default'` with `ProfileMismatchError`.
- Instance `restore()` additionally rejects when the live index polarity
  (`!this.normalized`) disagrees with the snapshot polarity
  (`!header.normalized`) — the retained token stream is never reinterpreted.
- `getStats().formatVersion` reports the **live format** (4) — what the
  index will emit on serialize — not the source snapshot version. Immediately
  after a legacy v3 restore it still reads 4 (pinned by `test:snapshot` §10).

---

## 7. IndexedDB

- Snapshots persist raw log docs unencrypted (origin-scoped). Logs may carry
  real PII in production — treat IDB contents accordingly.
- `MAX_SNAPSHOT_*` caps enforce fail-closed bounds before decode
  (schema 16 MiB, columnar 64 MiB, docs 256 MiB, total 512 MiB,
  docCount 10M, tokenCount 256M). Oversize snapshots throw
  `IncompatibleIndexError` before any `TextDecoder`/`JSON.parse`/`slice`
  allocation.
- `saveIndexToIDB` serializes fully before opening the transaction;
  connections close in `finally` for future version migrations.

---

## 8. Benchmarks

- `bun run bench:snapshot-matrix` runs headless CPU only (`preferGpu: false`).
  Numbers are deterministic medians of 10 (serialize/restore also 10× sampled)
  with `samples` preserved in JSON. They are **not comparable** to
  browser/WebGPU runs. `environment` in the JSON records runtime/platform.
- CRC32 is corruption detection, not authenticity — forged snapshots with
  recomputed CRC are rejected only by shape/schema/offset validation.

---

## 9. Suites

```bash
bun run test:snapshot   # §§1–17: canonical + legacy + columnar + getters + caps + hooks + worker + profile guards
bun run test:records    # DocumentIndex engine + mutations + ProfileMismatch at search time
bun run bench:snapshot-matrix  # headless CPU matrix (serialize/restore bytes + time)
```
