# Snapshot Format

Versioned binary persistence for `DocumentIndex` (`encodeSnapshot` /
`index.serialize()`, worker `serialize()`).

## Snapshot format versions

- **Legacy (read-only)**: 48-byte header, magic `0x55324433`, version 3.
  Layout: header + schema + tokens + offsets + docs. No columnar segment.
- **Canonical (write path)**: 56-byte LE header, magic `0x55324434`,
  version 4 (8-byte aligned). Layout:
  `header[56] + schema + tokens + offsets + columnar + docs`.
  Header words: magic@0, version@4, profile@8, unicode@12, scoring@16,
  docCount@20, rowCount@24, tokenCount@28, normalized@32, schemaLen@36,
  docsLen@40, columnarLen@44, reserved(0)@48, CRC32@52 covering
  `header[0..52) + schema + tokens + offsets + columnar + docs`.

Magic bytes are frozen: old snapshots stay readable forever via the
migration read path. Only the exported constant names changed
(`SNAPSHOT_MAGIC`, `SNAPSHOT_FORMAT_VERSION`, `SNAPSHOT_HEADER_BYTES`;
legacy: `LEGACY_SNAPSHOT_MAGIC`, `LEGACY_SNAPSHOT_VERSION`,
`LEGACY_SNAPSHOT_HEADER_BYTES`).

## Compatibility

- Write path is **canonical-only** (`encodeSnapshot`, `index.serialize()`,
  worker `serialize()`). Older readers accepting only the legacy magic throw
  `IncompatibleIndexError` fail-closed.
- Read path accepts **legacy + canonical** (`decodeSnapshot`,
  `restoreSnapshot`, `DocumentIndex.fromSnapshot`,
  `SearchWorkerClient.restore`). Legacy headers report `columnarByteLength: 0`.
- `getStats().formatVersion` reports the **live format** (4) — what the
  index will emit on serialize — not the source snapshot version. Immediately
  after a legacy restore it still reads 4.

## Filter getters

- Custom `filterFields[].getter` closures are never serialized. The snapshot
  persists `hasGetter: true` per field; restore requires a matching getter
  override via `options.options.filterFields` or throws
  `IncompatibleIndexError` fail-closed (mirrors the `hookIds` idiom).
- Plain `doc[name]` fields (`hasGetter` absent/false) restore without overrides.
- Columnar segment is advisory + integrity-checked; authoritative filtering
  rebuilds via `ColumnarStore.init(records)`. A stripped empty segment on a
  filtered non-empty index throws `IncompatibleIndexError`.

## IndexedDB

- Snapshots persist raw log docs unencrypted (origin-scoped). Logs may carry
  real PII in production — treat IDB contents accordingly.
- `MAX_SNAPSHOT_*` caps enforce fail-closed bounds before decode
  (schema 16 MiB, columnar 64 MiB, docs 256 MiB, total 512 MiB,
  docCount 10M, tokenCount 256M). Oversize snapshots throw
  `IncompatibleIndexError`.

## Benchmarks

- `bun run bench:snapshot-matrix` runs headless CPU only (`preferGpu: false`).
  Numbers are deterministic medians of 10 (serialize/restore also 10× sampled)
  with `samples` preserved in JSON. They are **not comparable** to
  browser/WebGPU runs. `environment` in the JSON records runtime/platform.
- CRC32 is corruption detection, not authenticity — forged snapshots with
  recomputed CRC are rejected only by shape/schema/offset validation.
