/**
 * Buffer packing, string sanitation, and WebGPU memory limit validation.
 * v0.2 parity path packs post-fold u32 scalars (no locale/UTF-16 helpers).
 * Portable: no DOM refs.
 */

import { normalizeText } from './unicode-preprocess';
import {
  ENUM_TO_PROFILE,
  ENUM_TO_SCORING,
  ENUM_TO_UNICODE_VERSION,
  FORMAT_VERSION,
  PROFILE_TO_ENUM,
  QUERY_TOKENS_MAX,
  RESULT_LIMIT_MAX,
  SCORING_TO_ENUM,
  SCORING_VERSION,
  SERIALIZED_MAGIC,
  UNICODE_VERSION,
  UNICODE_VERSION_TO_ENUM,
  IncompatibleIndexError,
  IncompatibleOptionError,
  type TextProfileId,
} from './text-profile';

export interface PackedGPUBuffer {
  /** Contiguous character byte buffer (padded to 4-byte boundary for u32 storage) */
  recordsBufferData: ArrayBuffer;
  recordsByteLength: number;
  /** Uint32 offsets array (length stringCount + 1) defining [start, end) byte slices */
  offsetsBufferData: ArrayBuffer;
  offsetsByteLength: number;
  stringCount: number;
  totalChars: number;
  /** Backwards-compatible alias for recordsBufferData */
  bufferData: ArrayBuffer;
  /** Total combined byte allocation (records + offsets) */
  byteLength: number;
  slotBytes?: number;
  maxCharsPerString?: number;
}

export interface PackedUnicodeBufferV2 {
  tokens: Uint32Array;
  offsets: Uint32Array;
  rowCount: number;
  tokenCount: number;
  folded: boolean;
  profileId: TextProfileId;
  unicodeVersion: string;
  nfcProbedVersion: string | null;
  scoringVersion: string;
  formatVersion: 2;
  recordsBufferData: ArrayBuffer;
  offsetsBufferData: ArrayBuffer;
  recordsByteLength: number;
  offsetsByteLength: number;
  combinedByteLength: number;
}

export interface UnicodePackOptions {
  folded?: boolean;
  profileId?: TextProfileId;
  unicodeVersion?: string;
  scoringVersion?: string;
  totalTokens?: number;
  slotBytes?: number;
}

export interface MemoryBudgetCheck {
  allowed: boolean;
  requiredBytes: number;
  maxBytes: number;
  reason?: string;
}

/**
 * Legacy v0.1 sanitizer for the GPU byte path (NOT parity).
 * @deprecated Parity uses `normalizeText()` folded tokens. Kept for M6 migration only. Removal in v0.3.
 */
export function sanitizeStringForSlot(str: string, maxChars?: number): string {
  if (!str) return '';
  const normalized = str.normalize('NFKD').replace(/[\u0300-\u036f]/g, '');
  const asciiOnly = normalized.replace(/[^\x20-\x7E]/g, '?');
  return typeof maxChars === 'number' && maxChars > 0 ? asciiOnly.slice(0, maxChars) : asciiOnly;
}

export const sanitizeString = sanitizeStringForSlot;

/** Legacy v0.1 byte packer (NOT parity). @deprecated See sanitizeStringForSlot. Removal in v0.3. */
export function packStringsToGPUBuffer(
  strings: string[],
  _legacySlotBytes?: number
): PackedGPUBuffer {
  const count = strings.length;
  const offsets = new Uint32Array(count + 1);
  offsets[0] = 0;

  const cleanStrings = new Array<string>(count);
  let totalChars = 0;

  for (let i = 0; i < count; i++) {
    const raw = strings[i] ?? '';
    const clean = sanitizeStringForSlot(raw);
    cleanStrings[i] = clean;
    totalChars += clean.length;
    offsets[i + 1] = totalChars;
  }

  const recordsByteLength = Math.max(16, Math.ceil(totalChars / 4) * 4);
  const recordsBufferData = new ArrayBuffer(recordsByteLength);
  const recordsU8 = new Uint8Array(recordsBufferData);

  let currentByte = 0;
  for (let i = 0; i < count; i++) {
    const s = cleanStrings[i];
    const len = s.length;
    for (let c = 0; c < len; c++) {
      // Legacy byte path only ever stores ASCII (`?`-mapped); codePointAt
      // equals charCodeAt here and keeps the parity lint clean.
      recordsU8[currentByte + c] = (s.codePointAt(c) as number) & 0xff;
    }
    currentByte += len;
  }

  const offsetsByteLength = Math.max(16, (count + 1) * 4);
  const offsetsBufferData = offsets.buffer as ArrayBuffer;
  const combinedByteLength = recordsByteLength + offsetsByteLength;

  return {
    recordsBufferData,
    recordsByteLength,
    offsetsBufferData,
    offsetsByteLength,
    stringCount: count,
    totalChars,
    bufferData: recordsBufferData,
    byteLength: combinedByteLength,
    slotBytes: _legacySlotBytes,
    maxCharsPerString: Infinity
  };
}

/**
 * Pack post-fold u32 scalars (parity path). Accepts pre-tokenized
 * `Uint32Array[]` (zero re-normalization) or `string[]` (normalized once).
 */
export function packUnicodeToGPUBuffer(
  input: readonly Uint32Array[] | readonly string[],
  options: UnicodePackOptions = {}
): PackedUnicodeBufferV2 {
  if (options.slotBytes !== undefined) {
    throw new IncompatibleOptionError(
      'slotBytes',
      '[webgpu-search] slotBytes throw-on-use in v0.2 (fixed slots removed; removal in v0.3).'
    );
  }
  const folded = options.folded ?? true;
  const profileId: TextProfileId = options.profileId ?? 'unicode-default';
  const unicodeVersion: string = options.unicodeVersion ?? UNICODE_VERSION;
  const scoringVersion: string = options.scoringVersion ?? SCORING_VERSION;
  const rowCount = input.length;
  if (rowCount === 0) {
    const offsets = new Uint32Array(1);
    offsets[0] = 0;
    const tokens = new Uint32Array(0);
    return {
      tokens, offsets, rowCount: 0, tokenCount: 0, folded, profileId,
      unicodeVersion, nfcProbedVersion: null, scoringVersion,
      formatVersion: FORMAT_VERSION,
      recordsBufferData: tokens.buffer as ArrayBuffer,
      offsetsBufferData: offsets.buffer as ArrayBuffer,
      recordsByteLength: 0, offsetsByteLength: 4, combinedByteLength: 4,
    };
  }
  const first: unknown = (input as readonly unknown[])[0];
  const tokenRows = new Array<Uint32Array>(rowCount);
  let tokenCount = 0;
  if (typeof first === 'string') {
    const strs = input as readonly string[];
    for (let i = 0; i < rowCount; i++) {
      const el: unknown = strs[i];
      if (typeof el !== 'string') {
        throw new TypeError(`[webgpu-search] packUnicode: bad item ${i}.`);
      }
      const norm = normalizeText(el as string, folded);
      tokenRows[i] = norm.tokens;
      tokenCount += norm.tokenCount;
    }
  } else {
    const arrs = input as readonly Uint32Array[];
    for (let i = 0; i < rowCount; i++) {
      const el: unknown = arrs[i];
      if (!(el instanceof Uint32Array)) {
        throw new TypeError(`[webgpu-search] packUnicode: bad item ${i}.`);
      }
      const t = el as Uint32Array;
      tokenRows[i] = t;
      tokenCount += t.length;
    }
  }
  const tokens = new Uint32Array(tokenCount);
  const offsets = new Uint32Array(rowCount + 1);
  offsets[0] = 0;
  let pos = 0;
  for (let i = 0; i < rowCount; i++) {
    const t = tokenRows[i] as Uint32Array;
    tokens.set(t, pos);
    pos += t.length;
    offsets[i + 1] = pos;
  }
  const recordsByteLength = tokenCount * 4;
  const offsetsByteLength = (rowCount + 1) * 4;
  return {
    tokens, offsets, rowCount, tokenCount, folded, profileId,
    unicodeVersion, nfcProbedVersion: null, scoringVersion,
    formatVersion: FORMAT_VERSION,
    recordsBufferData: tokens.buffer as ArrayBuffer,
    offsetsBufferData: offsets.buffer as ArrayBuffer,
    recordsByteLength, offsetsByteLength,
    combinedByteLength: recordsByteLength + offsetsByteLength,
  };
}

// Serialized header: 9 u32 words = 36 bytes (MAGIC, versions, counts, checksum).

let crcTable: Uint32Array | null = null;
function crcTableLazy(): Uint32Array {
  if (crcTable !== null) return crcTable;
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c: number = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  crcTable = t;
  return t;
}

function crc32Parts(parts: readonly Uint8Array[]): number {
  const t = crcTableLazy();
  let crc: number = 0xffffffff;
  for (let p = 0; p < parts.length; p++) {
    const d = parts[p] as Uint8Array;
    for (let i = 0; i < d.length; i++) crc = t[(crc ^ (d[i])) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

/** Serialize to `header(36) + records + offsets` (little-endian, Uint32Array-native). */
export function serializeUnicodeDataset(packed: PackedUnicodeBufferV2): ArrayBuffer {
  const pe = PROFILE_TO_ENUM[packed.profileId];
  const ue = UNICODE_VERSION_TO_ENUM[packed.unicodeVersion];
  const se = SCORING_TO_ENUM[packed.scoringVersion];
  if (pe === undefined || ue === undefined || se === undefined) {
    throw new IncompatibleIndexError(1, 'unknown-version');
  }
  const total = 36 + packed.recordsByteLength + packed.offsetsByteLength;
  const out = new ArrayBuffer(total);
  const h = new Uint32Array(out, 0, 9);
  h.set([SERIALIZED_MAGIC, FORMAT_VERSION, pe, ue, se, packed.rowCount, packed.tokenCount, packed.folded ? 1 : 0, 0]);
  new Uint8Array(out, 36, packed.recordsByteLength).set(new Uint8Array(packed.recordsBufferData, 0, packed.recordsByteLength));
  new Uint8Array(out, 36 + packed.recordsByteLength, packed.offsetsByteLength).set(new Uint8Array(packed.offsetsBufferData, 0, packed.offsetsByteLength));
  const crc = crc32Parts([
    new Uint8Array(out, 0, 32),
    new Uint8Array(out, 36, packed.recordsByteLength),
    new Uint8Array(out, 36 + packed.recordsByteLength, packed.offsetsByteLength),
  ]);
  h[8] = crc;
  return out;
}

/** Deserialize + validate (magic, versions, sizes, monotonicity, checksum). */
export function deserializeUnicodeDataset(buffer: ArrayBuffer): PackedUnicodeBufferV2 {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength === 0) {
    throw new IncompatibleIndexError(SERIALIZED_MAGIC, 'neutered/empty');
  }
  if (buffer.byteLength < 36 || buffer.byteLength % 4 !== 0) {
    throw new IncompatibleIndexError(SERIALIZED_MAGIC, 'bad-length');
  }
  const h = new Uint32Array(buffer, 0, 9);
  const magic = h[0];
  if (magic !== SERIALIZED_MAGIC) throw new IncompatibleIndexError(SERIALIZED_MAGIC, magic);
  if ((h[1]) !== FORMAT_VERSION) throw new IncompatibleIndexError(FORMAT_VERSION, h[1]);
  const profileId = ENUM_TO_PROFILE[h[2]];
  const unicodeVersion = ENUM_TO_UNICODE_VERSION[h[3]];
  const scoringVersion = ENUM_TO_SCORING[h[4]];
  if (profileId === undefined || unicodeVersion === undefined || scoringVersion === undefined) {
    throw new IncompatibleIndexError(1, h[2]);
  }
  const rowCount = h[5];
  const tokenCount = h[6];
  const folded = (h[7]) === 1;
  const want = 36 + tokenCount * 4 + (rowCount + 1) * 4;
  if (buffer.byteLength !== want) throw new IncompatibleIndexError(want, buffer.byteLength);
  const crc = crc32Parts([
    new Uint8Array(buffer, 0, 32),
    new Uint8Array(buffer, 36, tokenCount * 4),
    new Uint8Array(buffer, 36 + tokenCount * 4, (rowCount + 1) * 4),
  ]);
  if (crc !== (h[8])) throw new IncompatibleIndexError(h[8], crc);
  const tokens = new Uint32Array(buffer.slice(36, 36 + tokenCount * 4));
  const offsets = new Uint32Array(buffer.slice(36 + tokenCount * 4));
  if ((offsets[0]) !== 0) throw new IncompatibleIndexError(0, offsets[0]);
  for (let i = 0; i < rowCount; i++) {
    if ((offsets[i + 1]) < (offsets[i])) {
      throw new IncompatibleIndexError('monotonic-offsets', `${i}`);
    }
  }
  if ((offsets[rowCount]) !== tokenCount) {
    throw new IncompatibleIndexError(tokenCount, offsets[rowCount]);
  }
  const recordsByteLength = tokenCount * 4;
  const offsetsByteLength = (rowCount + 1) * 4;
  return {
    tokens, offsets, rowCount, tokenCount, folded, profileId,
    unicodeVersion, nfcProbedVersion: null, scoringVersion,
    formatVersion: FORMAT_VERSION,
    recordsBufferData: tokens.buffer as ArrayBuffer,
    offsetsBufferData: offsets.buffer as ArrayBuffer,
    recordsByteLength, offsetsByteLength,
    combinedByteLength: recordsByteLength + offsetsByteLength,
  };
}

/**
 * Per-buffer limit check vs `min(maxBufferSize, maxStorageBufferBindingSize)`.
 * Each of records / offsets / query (512 B) / output (`8+cap*8`) must fit.
 */
export function checkMemoryBudget(
  itemCount: number,
  estimatedAvgBytes: number = 64,
  device?: GPUDevice | null,
  cap: number = RESULT_LIMIT_MAX
): MemoryBudgetCheck {
  const l = device?.limits;
  const maxBytes = l === undefined ? 134217728 : Math.min(l.maxBufferSize, l.maxStorageBufferBindingSize);
  const recordsBytes = itemCount * estimatedAvgBytes;
  const offsetsBytes = (itemCount + 1) * 4;
  const queryBytes = QUERY_TOKENS_MAX * 4;
  const outputBytes = 8 + (cap > 0 ? cap : RESULT_LIMIT_MAX) * 8;
  const requiredBytes = recordsBytes + offsetsBytes + queryBytes + outputBytes;
  const over =
    recordsBytes > maxBytes ? 'records' :
    offsetsBytes > maxBytes ? 'offsets' :
    queryBytes > maxBytes ? 'query' :
    outputBytes > maxBytes ? 'output' : null;
  if (over !== null) {
    return {
      allowed: false, requiredBytes, maxBytes,
      reason: `Dataset ${over} buffer too large.`,
    };
  }
  return { allowed: true, requiredBytes, maxBytes };
}
