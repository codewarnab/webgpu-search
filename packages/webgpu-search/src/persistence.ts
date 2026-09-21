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

/**
 * Parses and returns the 48-byte Little-Endian U2D3 binary header.
 * Validates MAGIC and formatVersion fail-closed.
 */
export function deserializeDocumentSnapshotHeader(buffer: ArrayBuffer): DocumentSnapshotHeader {
  const buf = buffer as unknown as { byteLength?: unknown; slice?: unknown };
  const byteLen = typeof buf?.byteLength === 'number' ? (buf.byteLength as number) : NaN;
  const canSlice = typeof (buf as any)?.slice === 'function';
  if (!Number.isFinite(byteLen) || byteLen < SERIALIZED_DOC_HEADER_BYTES || !canSlice) {
    throw new IncompatibleIndexError(SERIALIZED_DOC_MAGIC, 'neutered/short');
  }

  const dv = new DataView(buffer, 0, SERIALIZED_DOC_HEADER_BYTES);
  const magic = dv.getUint32(0, true);
  if (magic !== SERIALIZED_DOC_MAGIC) {
    throw new IncompatibleIndexError(SERIALIZED_DOC_MAGIC, magic);
  }

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
    checksum
  };
}

/**
 * Serializes a DocumentIndex into a versioned U2D3 Little-Endian binary ArrayBuffer.
 *
 * Invariants:
 * 1. Automatic compaction pre-condition: compacts tombstones prior to snapshot creation.
 * 2. Header layout: 48 bytes (12 x u32 words, LE). Word 11 [0x2C..0x2F] is checksum destination.
 * 3. Circular-dependency-free CRC32 covers header [0..44) + schema + tokens + offsets + docs.
 * 4. Decoupled document storage support: docsByteLength = 0 avoids 100MB+ JSON string allocations.
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
  // are never serialized).
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
      type: ff.type
    })),
    ...(hookIds !== undefined ? { hookIds } : {})
  };

  const schemaJson = JSON.stringify(schema);
  const schemaBytes = new TextEncoder().encode(schemaJson);
  const schemaByteLength = schemaBytes.length;

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
    SERIALIZED_DOC_HEADER_BYTES +
    schemaByteLength +
    tokensByteLength +
    offsetsByteLength +
    docsByteLength;

  const out = new ArrayBuffer(totalBytes);
  const dv = new DataView(out);

  // Write header [0..44) in Little-Endian
  dv.setUint32(0, SERIALIZED_DOC_MAGIC, true);
  dv.setUint32(4, DOC_FORMAT_VERSION, true);
  dv.setUint32(8, pe, true);
  dv.setUint32(12, ue, true);
  dv.setUint32(16, se, true);
  dv.setUint32(20, docCount, true);
  dv.setUint32(24, rowCount, true);
  dv.setUint32(28, tokenCount, true);
  dv.setUint32(32, folded ? 1 : 0, true);
  dv.setUint32(36, schemaByteLength, true);
  dv.setUint32(40, docsByteLength, true);

  // Copy Schema JSON
  let cursor = SERIALIZED_DOC_HEADER_BYTES;
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

  // Copy Document Records (if embedded)
  if (docsByteLength > 0) {
    new Uint8Array(out, cursor, docsByteLength).set(docsBytes);
    cursor += docsByteLength;
  }

  // Calculate CRC32 over header [0..44) + payload segments
  const header44 = new Uint8Array(out, 0, 44);
  const payloadSchema = new Uint8Array(out, SERIALIZED_DOC_HEADER_BYTES, schemaByteLength);
  const payloadTokens = new Uint8Array(out, SERIALIZED_DOC_HEADER_BYTES + schemaByteLength, tokensByteLength);
  const payloadOffsets = new Uint8Array(out, SERIALIZED_DOC_HEADER_BYTES + schemaByteLength + tokensByteLength, offsetsByteLength);
  const payloadDocs = docsByteLength > 0
    ? new Uint8Array(out, SERIALIZED_DOC_HEADER_BYTES + schemaByteLength + tokensByteLength + offsetsByteLength, docsByteLength)
    : new Uint8Array(0);

  const crc = crc32Parts([
    header44,
    payloadSchema,
    payloadTokens,
    payloadOffsets,
    payloadDocs
  ]);

  // Destination Word 11 [0x2C..0x2F]
  dv.setUint32(44, crc, true);

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
 * Deserializes and validates a U2D3 binary ArrayBuffer.
 * Validates MAGIC, version, CRC32, schema, monotonic offsets, and document records.
 */
export function deserializeDocumentSnapshot<TDoc = Record<string, unknown>>(
  buffer: ArrayBuffer,
  options?: RestoreDocumentIndexOptions<TDoc>
): RestoredDocumentSnapshot<TDoc> {
  const header = deserializeDocumentSnapshotHeader(buffer);
  const byteLen = buffer.byteLength;

  const wantTokensBytes = header.tokenCount * 4;
  const wantOffsetsBytes = (header.rowCount + 1) * 4;
  const expectedTotal =
    SERIALIZED_DOC_HEADER_BYTES +
    header.schemaByteLength +
    wantTokensBytes +
    wantOffsetsBytes +
    header.docsByteLength;

  if (byteLen !== expectedTotal) {
    throw new IncompatibleIndexError(expectedTotal, byteLen);
  }

  // Verify Checksum
  const header44 = new Uint8Array(buffer, 0, 44);
  const payloadSchema = new Uint8Array(buffer, SERIALIZED_DOC_HEADER_BYTES, header.schemaByteLength);
  const payloadTokens = new Uint8Array(buffer, SERIALIZED_DOC_HEADER_BYTES + header.schemaByteLength, wantTokensBytes);
  const payloadOffsets = new Uint8Array(buffer, SERIALIZED_DOC_HEADER_BYTES + header.schemaByteLength + wantTokensBytes, wantOffsetsBytes);
  const payloadDocs = header.docsByteLength > 0
    ? new Uint8Array(buffer, SERIALIZED_DOC_HEADER_BYTES + header.schemaByteLength + wantTokensBytes + wantOffsetsBytes, header.docsByteLength)
    : new Uint8Array(0);

  const computedCrc = crc32Parts([
    header44,
    payloadSchema,
    payloadTokens,
    payloadOffsets,
    payloadDocs
  ]);

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
      if (typeof val !== 'string' || (val as string).length === 0) {
        throw new IncompatibleIndexError(`valid-hookIds.${key}-string`, typeof val);
      }
    }
  }

  // Validate row count matches field count * doc count
  const expectedRowCount = header.docCount * schema.fields.length;
  if (header.rowCount !== expectedRowCount) {
    throw new IncompatibleIndexError(expectedRowCount, header.rowCount);
  }

  // Unpack Tokens & Offsets (independent sliced allocations to guarantee 4-byte alignment)
  const tokensStart = SERIALIZED_DOC_HEADER_BYTES + header.schemaByteLength;
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
 * Restores a DocumentIndex instance from a U2D3 binary ArrayBuffer.
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
