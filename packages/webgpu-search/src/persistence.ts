import { crc32Parts, validatePackedOffsets } from './buffer';
import { collectHookIds } from './extensions';
import {
  DOC_FORMAT_VERSION,
  ENUM_TO_PROFILE,
  ENUM_TO_SCORING,
  ENUM_TO_UNICODE_VERSION,
  PROFILE_TO_ENUM,
  SCORING_TO_ENUM,
  SERIALIZED_DOC_HEADER_BYTES,
  SERIALIZED_DOC_MAGIC,
  U2D4_FORMAT_VERSION,
  U2D4_HEADER_BYTES,
  U2D4_MAGIC,
  UNICODE_VERSION_TO_ENUM,
  IncompatibleIndexError
} from './text-profile';
import type { DocumentIndex, InternalField } from './document-index';
import type {
  DocumentId,
  DocumentIndexSchema,
  DocumentSnapshotHeader,
  RestoreDocumentIndexOptions,
  SerializeDocumentIndexOptions
} from './types';

/** Endianness detection for cross-architecture safety */
const isLittleEndian = (() => {
  const u16 = new Uint16Array([0x1234]);
  const u8 = new Uint8Array(u16.buffer);
  return u8[0] === 0x34;
})();

/** U2D4 reserved word must be zero (forward-extension guard). */
const U2D4_RESERVED_EXPECTED = 0;

/**
 * Fail-closed snapshot size caps (untrusted-input hardening). Length words
 * are attacker-controlled u32s; the total-length equality check alone does
 * not bound memory because the attacker supplies the bytes. Caps are
 * enforced *before* any `TextDecoder`/`JSON.parse`/`slice` allocation.
 */
export const MAX_SNAPSHOT_SCHEMA_BYTES = 16 << 20; // 16 MiB
export const MAX_SNAPSHOT_COLUMNAR_BYTES = 64 << 20; // 64 MiB
export const MAX_SNAPSHOT_DOCS_BYTES = 256 << 20; // 256 MiB
export const MAX_SNAPSHOT_BYTES = 512 << 20; // 512 MiB total
export const MAX_SNAPSHOT_DOC_COUNT = 10_000_000;
export const MAX_SNAPSHOT_TOKEN_COUNT = 256_000_000; // ~1 GiB tokens

function assertSnapshotSizeCaps(
  schemaByteLength: number,
  columnarByteLength: number,
  docsByteLength: number,
  docCount: number,
  tokenCount: number,
  rowCount: number
): void {
  if (
    !Number.isFinite(schemaByteLength) ||
    !Number.isFinite(columnarByteLength) ||
    !Number.isFinite(docsByteLength) ||
    !Number.isFinite(docCount) ||
    !Number.isFinite(tokenCount) ||
    !Number.isFinite(rowCount)
  ) {
    throw new IncompatibleIndexError('finite-snapshot-lengths', 'non-finite');
  }
  if (schemaByteLength > MAX_SNAPSHOT_SCHEMA_BYTES) {
    throw new IncompatibleIndexError(`schema-bytes<=${MAX_SNAPSHOT_SCHEMA_BYTES}`, schemaByteLength);
  }
  if (columnarByteLength > MAX_SNAPSHOT_COLUMNAR_BYTES) {
    throw new IncompatibleIndexError(`columnar-bytes<=${MAX_SNAPSHOT_COLUMNAR_BYTES}`, columnarByteLength);
  }
  if (docsByteLength > MAX_SNAPSHOT_DOCS_BYTES) {
    throw new IncompatibleIndexError(`docs-bytes<=${MAX_SNAPSHOT_DOCS_BYTES}`, docsByteLength);
  }
  if (docCount > MAX_SNAPSHOT_DOC_COUNT) {
    throw new IncompatibleIndexError(`docCount<=${MAX_SNAPSHOT_DOC_COUNT}`, docCount);
  }
  if (tokenCount > MAX_SNAPSHOT_TOKEN_COUNT) {
    throw new IncompatibleIndexError(`tokenCount<=${MAX_SNAPSHOT_TOKEN_COUNT}`, tokenCount);
  }
}

/** JSON-safe normalization: BigInt/functions/symbols cannot cross `JSON.stringify`. */
function toJsonSafeValue(v: unknown): unknown {
  if (typeof v === 'bigint') return String(v);
  if (typeof v === 'function' || typeof v === 'symbol') return null;
  if (v === undefined) return null;
  return v;
}

function stringifyColumnarPayload(payload: { v: number; fields: unknown; rows: unknown }): string {
  return JSON.stringify(payload, (_key, value) =>
    typeof value === 'bigint'
      ? String(value)
      : typeof value === 'function' || typeof value === 'symbol'
        ? null
        : value === undefined
          ? null
          : (value as unknown)
  );
}

/**
 * Encodes the columnar filter-attribute segment for a U2D4 snapshot.
 *
 * Layout: UTF-8 JSON bytes of `{ v: 1, fields, rows }` where `fields` mirrors
 * `schema.filterFields` (name + type) and `rows[d][f]` holds the raw getter
 * value for doc `d` and filter field `f` (`undefined` normalizes to `null`,
 * matching columnar presence semantics where null/undefined clears presence).
 *
 * The segment is advisory + integrity-checked: restore rebuilds the
 * authoritative `ColumnarStore` via `init(records)` and validates shape only
 * (field names/types, row counts). Corrupt payloads throw
 * `IncompatibleIndexError` via CRC or shape validation. Empty (0 bytes) when
 * the index declares no `filterFields` or holds no docs.
 */
export function encodeColumnarPayload<TDoc>(
  records: TDoc[],
  filterDefs: Array<{ name: string; type?: string; getter?: (doc: TDoc) => unknown }>
): Uint8Array {
  if (filterDefs.length === 0 || records.length === 0) {
    return new Uint8Array(0);
  }
  const fields = filterDefs.map((ff) => ({
    name: ff.name,
    ...(ff.type !== undefined ? { type: ff.type } : {})
  }));
  const rows: unknown[][] = new Array(records.length);
  for (let d = 0; d < records.length; d++) {
    const doc = records[d];
    const row: unknown[] = new Array(filterDefs.length);
    for (let f = 0; f < filterDefs.length; f++) {
      const getter = filterDefs[f]!.getter;
      let v: unknown = null;
      try {
        v = getter ? getter(doc) : (doc as any)?.[filterDefs[f]!.name];
      } catch {
        v = null;
      }
      row[f] = toJsonSafeValue(v);
    }
    rows[d] = row;
  }
  const json = stringifyColumnarPayload({ v: 1, fields, rows });
  return new TextEncoder().encode(json);
}

/**
 * Validates a decoded U2D4 columnar segment against the snapshot schema.
 * Throws `IncompatibleIndexError` fail-closed on version, field, or shape
 * mismatch. Empty segments are only valid when no filter fields are declared
 * or the snapshot holds no docs; a stripped segment on a filtered non-empty
 * index throws (integrity signal — authoritative rebuild would otherwise
 * silently succeed with default accessors).
 */
export function validateColumnarPayload(
  columnarBytes: Uint8Array,
  schemaFilterFields: Array<{ name: string; type?: string }> | undefined,
  docCount: number
): void {
  const expectedFields = schemaFilterFields ?? [];
  if (columnarBytes.length === 0) {
    if (expectedFields.length > 0 && docCount > 0) {
      throw new IncompatibleIndexError('columnar-present', 'stripped-empty-segment');
    }
    return;
  }
  if (columnarBytes.length > MAX_SNAPSHOT_COLUMNAR_BYTES) {
    throw new IncompatibleIndexError(`columnar-bytes<=${MAX_SNAPSHOT_COLUMNAR_BYTES}`, columnarBytes.length);
  }
  let parsed: any;
  try {
    parsed = JSON.parse(new TextDecoder().decode(columnarBytes));
  } catch {
    throw new IncompatibleIndexError('valid-columnar-json', 'parse-failure');
  }
  if (!parsed || typeof parsed !== 'object' || parsed.v !== 1 || !Array.isArray(parsed.fields) || !Array.isArray(parsed.rows)) {
    throw new IncompatibleIndexError('valid-columnar-envelope', typeof parsed);
  }
  if (parsed.fields.length !== expectedFields.length) {
    throw new IncompatibleIndexError(`columnar fields length ${expectedFields.length}`, parsed.fields.length);
  }
  for (let i = 0; i < expectedFields.length; i++) {
    const exp = expectedFields[i]!;
    const got = parsed.fields[i];
    if (!got || typeof got !== 'object' || got.name !== exp.name) {
      throw new IncompatibleIndexError(`columnar field[${i}].name ${exp.name}`, got?.name);
    }
    const expType = exp.type ?? 'string';
    const gotType = got.type ?? 'string';
    if (gotType !== expType) {
      throw new IncompatibleIndexError(`columnar field[${i}].type ${expType}`, gotType);
    }
  }
  if (parsed.rows.length !== docCount) {
    throw new IncompatibleIndexError(`columnar rows length ${docCount}`, parsed.rows.length);
  }
  for (let d = 0; d < parsed.rows.length; d++) {
    const row = parsed.rows[d];
    if (!Array.isArray(row) || row.length !== expectedFields.length) {
      throw new IncompatibleIndexError(`columnar row[${d}] width ${expectedFields.length}`, Array.isArray(row) ? row.length : typeof row);
    }
  }
}

/**
 * Parses a snapshot header, accepting both legacy U2D3 (48 bytes,
 * magic `0x55324433`, version 3) and canonical U2D4 (56 bytes, magic
 * `0x55324434`, version 4) fail-closed. Unknown magics, versions, enums, or
 * non-zero U2D4 reserved words throw `IncompatibleIndexError`.
 *
 * U2D3 headers report `columnarByteLength: 0` (absent segment).
 */
export function deserializeDocumentSnapshotHeader(buffer: ArrayBuffer): DocumentSnapshotHeader {
  const buf = buffer as unknown as { byteLength?: unknown; slice?: unknown };
  const byteLen = typeof buf?.byteLength === 'number' ? (buf.byteLength as number) : NaN;
  const canSlice = typeof (buf as any)?.slice === 'function';
  if (!Number.isFinite(byteLen) || byteLen < SERIALIZED_DOC_HEADER_BYTES || !canSlice) {
    throw new IncompatibleIndexError(SERIALIZED_DOC_MAGIC, 'neutered/short');
  }

  const magic = new DataView(buffer, 0, 4).getUint32(0, true);

  if (magic === SERIALIZED_DOC_MAGIC) {
    const dv = new DataView(buffer, 0, SERIALIZED_DOC_HEADER_BYTES);
    const formatVersion = dv.getUint32(4, true);
    if (formatVersion !== DOC_FORMAT_VERSION) {
      throw new IncompatibleIndexError(DOC_FORMAT_VERSION, formatVersion);
    }
    const profileEnum = dv.getUint32(8, true);
    const unicodeVersionEnum = dv.getUint32(12, true);
    const scoringVersionEnum = dv.getUint32(16, true);

    const profileId = ENUM_TO_PROFILE[profileEnum];
    const unicodeVersion = ENUM_TO_UNICODE_VERSION[unicodeVersionEnum];
    const scoringVersion = ENUM_TO_SCORING[scoringVersionEnum];

    if (profileId === undefined || unicodeVersion === undefined || scoringVersion === undefined) {
      const badEnum = profileId === undefined ? profileEnum : unicodeVersion === undefined ? unicodeVersionEnum : scoringVersionEnum;
      throw new IncompatibleIndexError('valid-enum', badEnum);
    }

    const docCount = dv.getUint32(20, true);
    const rowCount = dv.getUint32(24, true);
    const tokenCount = dv.getUint32(28, true);
    const foldedVal = dv.getUint32(32, true);

    if (foldedVal !== 0 && foldedVal !== 1) {
      throw new IncompatibleIndexError('folded 0|1', foldedVal);
    }

    const schemaByteLength = dv.getUint32(36, true);
    const docsByteLength = dv.getUint32(40, true);
    const checksum = dv.getUint32(44, true);

    return {
      magic,
      formatVersion,
      profileId,
      unicodeVersion,
      scoringVersion,
      docCount,
      rowCount,
      tokenCount,
      folded: foldedVal === 1,
      schemaByteLength,
      docsByteLength,
      columnarByteLength: 0,
      checksum
    };
  }

  if (magic === U2D4_MAGIC) {
    if (byteLen < U2D4_HEADER_BYTES) {
      throw new IncompatibleIndexError(U2D4_MAGIC, 'neutered/short-u2d4');
    }
    const dv = new DataView(buffer, 0, U2D4_HEADER_BYTES);
    const formatVersion = dv.getUint32(4, true);
    if (formatVersion !== U2D4_FORMAT_VERSION) {
      throw new IncompatibleIndexError(U2D4_FORMAT_VERSION, formatVersion);
    }
    const profileEnum = dv.getUint32(8, true);
    const unicodeVersionEnum = dv.getUint32(12, true);
    const scoringVersionEnum = dv.getUint32(16, true);

    const profileId = ENUM_TO_PROFILE[profileEnum];
    const unicodeVersion = ENUM_TO_UNICODE_VERSION[unicodeVersionEnum];
    const scoringVersion = ENUM_TO_SCORING[scoringVersionEnum];

    if (profileId === undefined || unicodeVersion === undefined || scoringVersion === undefined) {
      const badEnum = profileId === undefined ? profileEnum : unicodeVersion === undefined ? unicodeVersionEnum : scoringVersionEnum;
      throw new IncompatibleIndexError('valid-enum', badEnum);
    }

    const docCount = dv.getUint32(20, true);
    const rowCount = dv.getUint32(24, true);
    const tokenCount = dv.getUint32(28, true);
    const foldedVal = dv.getUint32(32, true);

    if (foldedVal !== 0 && foldedVal !== 1) {
      throw new IncompatibleIndexError('folded 0|1', foldedVal);
    }

    const schemaByteLength = dv.getUint32(36, true);
    const docsByteLength = dv.getUint32(40, true);
    const columnarByteLength = dv.getUint32(44, true);
    const reserved = dv.getUint32(48, true);
    const checksum = dv.getUint32(52, true);

    if (reserved !== U2D4_RESERVED_EXPECTED) {
      throw new IncompatibleIndexError('reserved 0', reserved);
    }

    return {
      magic,
      formatVersion,
      profileId,
      unicodeVersion,
      scoringVersion,
      docCount,
      rowCount,
      tokenCount,
      folded: foldedVal === 1,
      schemaByteLength,
      docsByteLength,
      columnarByteLength,
      checksum
    };
  }

  throw new IncompatibleIndexError(`${SERIALIZED_DOC_MAGIC}|${U2D4_MAGIC}`, magic);
}

/**
 * Serializes a DocumentIndex into a versioned U2D4 Little-Endian binary ArrayBuffer.
 *
 * Header layout: 56 bytes (14 x u32 words, LE), evenly divisible by 8 so
 * subsequent 64-bit columnar typed arrays stay aligned without padding.
 * Word 13 [0x34..0x37] is the checksum destination covering header `[0..52)`
 * plus payload segments in order: schema + tokens + offsets + columnar + docs.
 *
 * Invariants:
 * 1. Automatic compaction pre-condition: compacts tombstones prior to snapshot creation.
 * 2. Circular-dependency-free CRC32 covers header [0..52) + all payload segments.
 * 3. Decoupled document storage support: docsByteLength = 0 avoids 100MB+ JSON string allocations.
 * 4. Columnar filter metadata serializes alongside attribute schemas
 *    (empty segment when no filterFields); closures are never serialized.
 */
export function serializeDocumentIndex<TDoc = Record<string, unknown>>(
  index: DocumentIndex<TDoc>,
  options?: SerializeDocumentIndexOptions
): ArrayBuffer {
  if (!index) {
    throw new TypeError('[webgpu-search] serializeDocumentIndex requires a DocumentIndex instance.');
  }

  // Ensure 0 tombstones before serializing
  index.compactSync();

  const records = index.getRecords();
  const docIds = index.getDocIds();
  const sortedFields = index.getSortedFields();
  const rowTokens = index.getRowTokens();
  const tokenCount = index.getTotalTokens();
  const docCount = docIds.length;
  const rowCount = rowTokens.length;
  const folded = index.isFolded();
  const profileId = index.getProfileId();
  const unicodeVersion = index.getUnicodeVersion();
  const scoringVersion = index.getScoringVersion();

  const pe = PROFILE_TO_ENUM[profileId];
  const ue = UNICODE_VERSION_TO_ENUM[unicodeVersion];
  const se = SCORING_TO_ENUM[scoringVersion];

  if (pe === undefined || ue === undefined || se === undefined) {
    throw new IncompatibleIndexError(1, 'unknown-version');
  }

  // Schema segment encoding (v0.4 M6: declarative hookIds only — closures
  // are never serialized; v0.4 M8: filter `hasGetter` persisted so restore
  // can fail closed when a custom getter cannot be revived).
  const hookIds = collectHookIds(index.getExtensions?.() as any);
  const schema: DocumentIndexSchema = {
    fields: sortedFields.map((f: InternalField<TDoc>) => ({
      name: f.name,
      weight: f.weight
    })),
    idField: typeof index.options.idField === 'string' ? index.options.idField : undefined,
    docIds: docIds.slice(),
    caseSensitive: index.options.caseSensitive ?? !folded,
    preferGpu: index.options.preferGpu,
    threshold: index.options.threshold,
    candidateCapacity: index.options.candidateCapacity,
    initialCapacity: index.options.initialCapacity,
    growthFactor: index.options.growthFactor,
    mutationEpoch: index.getStats().mutationEpoch,
    filterFields: index.getFilterFieldDefinitions().map((ff) => ({
      name: ff.name,
      type: ff.type,
      ...((ff as { hasGetter?: boolean }).hasGetter === true ? { hasGetter: true as const } : {})
    })),
    ...(hookIds !== undefined ? { hookIds } : {})
  };

  const schemaJson = JSON.stringify(schema);
  const schemaBytes = new TextEncoder().encode(schemaJson);
  const schemaByteLength = schemaBytes.length;

  // Columnar filter-attribute segment (U2D4).
  const filterDefs = index.getFilterFieldDefinitions() as Array<{ name: string; type?: string; getter?: (doc: TDoc) => unknown }>;
  const columnarBytes = encodeColumnarPayload(records, filterDefs);
  const columnarByteLength = columnarBytes.length;

  // Decoupled vs embedded document storage
  const decoupled = options?.decoupled === true;
  let docsBytes: Uint8Array;
  let docsByteLength = 0;

  if (decoupled) {
    docsBytes = new Uint8Array(0);
    docsByteLength = 0;
  } else {
    const docsJson = JSON.stringify(records);
    docsBytes = new TextEncoder().encode(docsJson);
    docsByteLength = docsBytes.length;
  }

  // Pack contiguous offsets
  const offsets = new Uint32Array(rowCount + 1);
  offsets[0] = 0;
  let currentTokenPos = 0;

  for (let r = 0; r < rowCount; r++) {
    const row = rowTokens[r];
    currentTokenPos += row.length;
    offsets[r + 1] = currentTokenPos;
  }

  const tokensByteLength = tokenCount * 4;
  const offsetsByteLength = (rowCount + 1) * 4;

  const totalBytes =
    U2D4_HEADER_BYTES +
    schemaByteLength +
    tokensByteLength +
    offsetsByteLength +
    columnarByteLength +
    docsByteLength;

  const out = new ArrayBuffer(totalBytes);
  const dv = new DataView(out);

  // Write header [0..52) in Little-Endian
  dv.setUint32(0, U2D4_MAGIC, true);
  dv.setUint32(4, U2D4_FORMAT_VERSION, true);
  dv.setUint32(8, pe, true);
  dv.setUint32(12, ue, true);
  dv.setUint32(16, se, true);
  dv.setUint32(20, docCount, true);
  dv.setUint32(24, rowCount, true);
  dv.setUint32(28, tokenCount, true);
  dv.setUint32(32, folded ? 1 : 0, true);
  dv.setUint32(36, schemaByteLength, true);
  dv.setUint32(40, docsByteLength, true);
  dv.setUint32(44, columnarByteLength, true);
  dv.setUint32(48, U2D4_RESERVED_EXPECTED, true);

  // Copy Schema JSON
  let cursor = U2D4_HEADER_BYTES;
  new Uint8Array(out, cursor, schemaByteLength).set(schemaBytes);
  cursor += schemaByteLength;

  // Copy Tokens directly from rowTokens (eliminates intermediate tokens Uint32Array allocation)
  let tokenCursor = cursor;
  for (let r = 0; r < rowCount; r++) {
    const row = rowTokens[r];
    const rowBytes = row.byteLength;
    if (isLittleEndian) {
      new Uint8Array(out, tokenCursor, rowBytes).set(
        new Uint8Array(row.buffer, row.byteOffset, rowBytes)
      );
    } else {
      for (let i = 0; i < row.length; i++) {
        dv.setUint32(tokenCursor + i * 4, row[i], true);
      }
    }
    tokenCursor += rowBytes;
  }
  cursor += tokensByteLength;

  // Copy Offsets
  const offsetsDst = new Uint8Array(out, cursor, offsetsByteLength);
  if (isLittleEndian) {
    offsetsDst.set(new Uint8Array(offsets.buffer, offsets.byteOffset, offsetsByteLength));
  } else {
    for (let i = 0; i <= rowCount; i++) {
      dv.setUint32(cursor + i * 4, offsets[i], true);
    }
  }
  cursor += offsetsByteLength;

  // Copy Columnar segment
  if (columnarByteLength > 0) {
    new Uint8Array(out, cursor, columnarByteLength).set(columnarBytes);
    cursor += columnarByteLength;
  }

  // Copy Document Records (if embedded)
  if (docsByteLength > 0) {
    new Uint8Array(out, cursor, docsByteLength).set(docsBytes);
    cursor += docsByteLength;
  }

  // Calculate CRC32 over header [0..52) + payload segments
  const header52 = new Uint8Array(out, 0, 52);
  const payloadSchema = new Uint8Array(out, U2D4_HEADER_BYTES, schemaByteLength);
  const payloadTokens = new Uint8Array(out, U2D4_HEADER_BYTES + schemaByteLength, tokensByteLength);
  const payloadOffsets = new Uint8Array(out, U2D4_HEADER_BYTES + schemaByteLength + tokensByteLength, offsetsByteLength);
  const payloadColumnar = columnarByteLength > 0
    ? new Uint8Array(out, U2D4_HEADER_BYTES + schemaByteLength + tokensByteLength + offsetsByteLength, columnarByteLength)
    : new Uint8Array(0);
  const payloadDocs = docsByteLength > 0
    ? new Uint8Array(out, U2D4_HEADER_BYTES + schemaByteLength + tokensByteLength + offsetsByteLength + columnarByteLength, docsByteLength)
    : new Uint8Array(0);

  const crc = crc32Parts([
    header52,
    payloadSchema,
    payloadTokens,
    payloadOffsets,
    payloadColumnar,
    payloadDocs
  ]);

  // Destination Word 13 [0x34..0x37]
  dv.setUint32(52, crc, true);

  return out;
}

export interface RestoredDocumentSnapshot<TDoc = Record<string, unknown>> {
  header: DocumentSnapshotHeader;
  schema: DocumentIndexSchema;
  tokens: Uint32Array;
  offsets: Uint32Array;
  records: TDoc[];
  docIds: DocumentId[];
}

/**
 * Deserializes and validates a versioned binary ArrayBuffer.
 * Accepts canonical U2D4 and legacy U2D3 (migration read path).
 * Validates MAGIC, version, CRC32, schema, columnar shape, monotonic
 * offsets, and document records.
 */
export function deserializeDocumentSnapshot<TDoc = Record<string, unknown>>(
  buffer: ArrayBuffer,
  options?: RestoreDocumentIndexOptions<TDoc>
): RestoredDocumentSnapshot<TDoc> {
  if (!buffer || typeof (buffer as ArrayBuffer).byteLength !== 'number') {
    throw new IncompatibleIndexError('arraybuffer', typeof buffer);
  }
  if ((buffer as ArrayBuffer).byteLength > MAX_SNAPSHOT_BYTES) {
    throw new IncompatibleIndexError(`snapshot-bytes<=${MAX_SNAPSHOT_BYTES}`, (buffer as ArrayBuffer).byteLength);
  }
  const header = deserializeDocumentSnapshotHeader(buffer);
  const byteLen = buffer.byteLength;
  const isU2D4 = header.magic === U2D4_MAGIC;
  const headerBytes = isU2D4 ? U2D4_HEADER_BYTES : SERIALIZED_DOC_HEADER_BYTES;
  const headerPrefixLen = isU2D4 ? 52 : 44;
  const columnarLen = header.columnarByteLength ?? 0;

  // Fail-closed size caps before any decode/slice allocation.
  assertSnapshotSizeCaps(
    header.schemaByteLength,
    columnarLen,
    header.docsByteLength,
    header.docCount,
    header.tokenCount,
    header.rowCount
  );

  const wantTokensBytes = header.tokenCount * 4;
  const wantOffsetsBytes = (header.rowCount + 1) * 4;
  const expectedTotal =
    headerBytes +
    header.schemaByteLength +
    wantTokensBytes +
    wantOffsetsBytes +
    columnarLen +
    header.docsByteLength;

  if (byteLen !== expectedTotal) {
    throw new IncompatibleIndexError(expectedTotal, byteLen);
  }

  // Verify Checksum
  const headerPrefix = new Uint8Array(buffer, 0, headerPrefixLen);
  const payloadSchema = new Uint8Array(buffer, headerBytes, header.schemaByteLength);
  const payloadTokens = new Uint8Array(buffer, headerBytes + header.schemaByteLength, wantTokensBytes);
  const payloadOffsets = new Uint8Array(buffer, headerBytes + header.schemaByteLength + wantTokensBytes, wantOffsetsBytes);
  const payloadColumnar = columnarLen > 0
    ? new Uint8Array(buffer, headerBytes + header.schemaByteLength + wantTokensBytes + wantOffsetsBytes, columnarLen)
    : new Uint8Array(0);
  const payloadDocs = header.docsByteLength > 0
    ? new Uint8Array(buffer, headerBytes + header.schemaByteLength + wantTokensBytes + wantOffsetsBytes + columnarLen, header.docsByteLength)
    : new Uint8Array(0);

  const computedCrc = isU2D4
    ? crc32Parts([headerPrefix, payloadSchema, payloadTokens, payloadOffsets, payloadColumnar, payloadDocs])
    : crc32Parts([headerPrefix, payloadSchema, payloadTokens, payloadOffsets, payloadDocs]);

  if (computedCrc !== header.checksum) {
    throw new IncompatibleIndexError(header.checksum, computedCrc);
  }

  // Parse Schema JSON
  let schema: DocumentIndexSchema;
  try {
    const schemaStr = new TextDecoder().decode(payloadSchema);
    schema = JSON.parse(schemaStr) as DocumentIndexSchema;
  } catch {
    throw new IncompatibleIndexError('valid-schema-json', 'parse-failure');
  }

  if (!schema || !Array.isArray(schema.fields) || schema.fields.length === 0) {
    throw new IncompatibleIndexError('schema.fields non-empty array', typeof schema?.fields);
  }

  for (let i = 0; i < schema.fields.length; i++) {
    const f = schema.fields[i];
    if (!f || typeof f !== 'object' || typeof f.name !== 'string' || f.name.trim().length === 0) {
      throw new IncompatibleIndexError('valid-schema-fields', typeof f);
    }
  }

  // v0.4 M6: fail-closed hookIds validation (declarative IDs only).
  if (schema.hookIds !== undefined) {
    if (typeof schema.hookIds !== 'object' || schema.hookIds === null || Array.isArray(schema.hookIds)) {
      throw new IncompatibleIndexError('valid-hookIds-object', typeof schema.hookIds);
    }
    const allowedHookKeys = ['tokenizer', 'scoringHook', 'filterPredicate', 'postProcess'];
    for (const key of Object.keys(schema.hookIds)) {
      if (allowedHookKeys.indexOf(key) < 0) {
        throw new IncompatibleIndexError('valid-hookIds-keys', key);
      }
      const val = (schema.hookIds as Record<string, unknown>)[key];
      if (typeof val !== 'string' || (val as string).trim().length === 0) {
        throw new IncompatibleIndexError(`valid-hookIds.${key}-string`, typeof val);
      }
    }
  }

  // U2D4: fail-closed columnar shape validation (authoritative rebuild via
  // ColumnarStore.init(records) happens in applySnapshotData).
  if (isU2D4) {
    validateColumnarPayload(
      payloadColumnar,
      (schema.filterFields as Array<{ name: string; type?: string }>) ?? undefined,
      header.docCount
    );
  }

  // Validate row count matches field count * doc count
  const expectedRowCount = header.docCount * schema.fields.length;
  if (header.rowCount !== expectedRowCount) {
    throw new IncompatibleIndexError(expectedRowCount, header.rowCount);
  }

  // Unpack Tokens & Offsets (independent sliced allocations to guarantee 4-byte alignment)
  const tokensStart = headerBytes + header.schemaByteLength;
  const tokensBuf = buffer.slice(tokensStart, tokensStart + wantTokensBytes);
  const tokens = new Uint32Array(tokensBuf);

  const offsetsStart = tokensStart + wantTokensBytes;
  const offsetsBuf = buffer.slice(offsetsStart, offsetsStart + wantOffsetsBytes);
  const offsets = new Uint32Array(offsetsBuf);

  if (!isLittleEndian) {
    const dv = new DataView(buffer);
    for (let i = 0; i < header.tokenCount; i++) {
      tokens[i] = dv.getUint32(tokensStart + i * 4, true);
    }
    for (let i = 0; i <= header.rowCount; i++) {
      offsets[i] = dv.getUint32(offsetsStart + i * 4, true);
    }
  }

  validatePackedOffsets(offsets, header.rowCount, header.tokenCount);

  // Unpack Document Records & IDs
  let docIds: DocumentId[] = [];

  if (schema.docIds !== undefined) {
    if (!Array.isArray(schema.docIds) || schema.docIds.length !== header.docCount) {
      throw new IncompatibleIndexError(
        `schema.docIds array length ${header.docCount}`,
        Array.isArray(schema.docIds) ? schema.docIds.length : typeof schema.docIds
      );
    }
    docIds = schema.docIds;
  }

  let records: TDoc[] = [];

  if (header.docsByteLength > 0) {
    try {
      const docsStr = new TextDecoder().decode(payloadDocs);
      records = JSON.parse(docsStr) as TDoc[];
    } catch {
      throw new IncompatibleIndexError('valid-docs-json', 'parse-failure');
    }
    if (!Array.isArray(records) || records.length !== header.docCount) {
      throw new IncompatibleIndexError(`array of ${header.docCount} records`, records?.length);
    }
  } else {
    // Decoupled document storage
    if (options?.documents && Array.isArray(options.documents)) {
      if (options.documents.length !== header.docCount) {
        throw new IncompatibleIndexError(`documents array length ${header.docCount}`, options.documents.length);
      }
      records = options.documents;
    } else {
      records = [];
    }

    if (header.docCount > 0 && docIds.length === 0 && records.length === 0) {
      throw new IncompatibleIndexError('docIds or options.documents required for decoupled restore', 'missing');
    }
  }

  return {
    header,
    schema,
    tokens,
    offsets,
    records,
    docIds
  };
}

/**
 * Restores a DocumentIndex instance from a versioned binary ArrayBuffer
 * (U2D4 canonical, U2D3 legacy migration path).
 */
export async function restoreDocumentIndex<TDoc = Record<string, unknown>>(
  buffer: ArrayBuffer,
  options?: RestoreDocumentIndexOptions<TDoc>
): Promise<DocumentIndex<TDoc>> {
  const snapshot = deserializeDocumentSnapshot<TDoc>(buffer, options);

  // Import dynamically or construct via DocumentIndex.restoreFromSnapshot
  const { DocumentIndex } = await import('./document-index');
  return DocumentIndex.fromSnapshotData<TDoc>(snapshot, options);
}
