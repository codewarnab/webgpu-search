/**
 * Buffer packing, string sanitation, and WebGPU memory limit validation.
 * Supports dynamic, variable-length strings via contiguous character storage and offset tables.
 */

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

export interface MemoryBudgetCheck {
  allowed: boolean;
  requiredBytes: number;
  maxBytes: number;
  reason?: string;
}

/**
 * Sanitizes an arbitrary string for GPU storage:
 * 1. Strips diacritics via unicode normalization (NFKD).
 * 2. Replaces remaining non-ASCII characters with '?' to preserve position and prevent multi-byte misalignment.
 * 3. Truncates to maxChars only if maxChars is explicitly specified.
 */
export function sanitizeStringForSlot(str: string, maxChars?: number): string {
  if (!str) return '';
  // Normalize and remove diacritics
  const normalized = str.normalize('NFKD').replace(/[\u0300-\u036f]/g, '');
  // Replace non-ASCII characters (keep printable ASCII 0x20-0x7E)
  const asciiOnly = normalized.replace(/[^\x20-\x7E]/g, '?');
  return typeof maxChars === 'number' && maxChars > 0 ? asciiOnly.slice(0, maxChars) : asciiOnly;
}

export const sanitizeString = sanitizeStringForSlot;

/**
 * Packs an array of arbitrary-length JS strings into WebGPU contiguous byte storage
 * with an Arrow-style Uint32 offsets table (offsets[i] to offsets[i+1]).
 * Removes the fixed 59-character restriction while eliminating slot padding waste.
 */
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

  // Ensure character records buffer is aligned to a 4-byte boundary for WGSL u32 reading
  const recordsByteLength = Math.max(16, Math.ceil(totalChars / 4) * 4);
  const recordsBufferData = new ArrayBuffer(recordsByteLength);
  const recordsU8 = new Uint8Array(recordsBufferData);

  let currentByte = 0;
  for (let i = 0; i < count; i++) {
    const s = cleanStrings[i];
    const len = s.length;
    for (let c = 0; c < len; c++) {
      recordsU8[currentByte + c] = s.charCodeAt(c);
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
 * Checks whether the dataset fits within device storage buffer limits.
 */
export function checkMemoryBudget(
  itemCount: number,
  estimatedAvgBytes: number = 64,
  device?: GPUDevice | null
): MemoryBudgetCheck {
  // Offsets buffer + character records buffer
  const offsetsBytes = (itemCount + 1) * 4;
  const recordsBytes = itemCount * estimatedAvgBytes;
  const requiredBytes = offsetsBytes + recordsBytes;
  const maxBytes = device?.limits?.maxStorageBufferBindingSize ?? (128 * 1024 * 1024);

  if (requiredBytes > maxBytes || recordsBytes > maxBytes || offsetsBytes > maxBytes) {
    const reqMB = (requiredBytes / (1024 * 1024)).toFixed(1);
    const maxMB = (maxBytes / (1024 * 1024)).toFixed(1);
    return {
      allowed: false,
      requiredBytes,
      maxBytes,
      reason: `Dataset size (${reqMB} MB) exceeds WebGPU maxStorageBufferBindingSize limit (${maxMB} MB).`
    };
  }

  return {
    allowed: true,
    requiredBytes,
    maxBytes
  };
}
