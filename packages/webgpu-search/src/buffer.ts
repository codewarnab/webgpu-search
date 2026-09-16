/**
 * Buffer packing, string sanitation, and WebGPU memory limit validation.
 */

export interface PackedGPUBuffer {
  bufferData: ArrayBuffer;
  byteLength: number;
  stringCount: number;
  slotBytes: 64 | 128;
  maxCharsPerString: number;
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
 * 2. Replaces remaining non-ASCII characters with '?' to preserve position and prevent UTF-8 multi-byte misalignment.
 * 3. Truncates to maxChars.
 */
export function sanitizeStringForSlot(str: string, maxChars: number = 59): string {
  if (!str) return '';
  // Normalize and remove diacritics
  const normalized = str.normalize('NFKD').replace(/[\u0300-\u036f]/g, '');
  // Replace non-ASCII printable characters (keep printable ASCII 0x20-0x7E)
  const asciiOnly = normalized.replace(/[^\x20-\x7E]/g, '?');
  return asciiOnly.slice(0, maxChars);
}

/**
 * Packs an array of JS strings into WebGPU aligned memory buffer:
 * - 64-byte slots: 1 u32 length (max 59 chars) + 60 bytes ASCII
 * - 128-byte slots: 1 u32 length (max 123 chars) + 124 bytes ASCII
 */
export function packStringsToGPUBuffer(
  strings: string[],
  slotBytes: 64 | 128 = 64
): PackedGPUBuffer {
  const count = strings.length;
  const maxChars = slotBytes === 128 ? 123 : 59;
  const wordsPerSlot = slotBytes / 4;
  const byteLength = count * slotBytes;

  const bufferData = new ArrayBuffer(byteLength);
  const u32View = new Uint32Array(bufferData);
  const u8View = new Uint8Array(bufferData);

  for (let i = 0; i < count; i++) {
    const raw = strings[i] ?? '';
    const clean = sanitizeStringForSlot(raw, maxChars);
    const len = clean.length;

    // Length header
    u32View[i * wordsPerSlot] = len;

    // Byte characters
    const baseByte = i * slotBytes + 4;
    for (let c = 0; c < len; c++) {
      u8View[baseByte + c] = clean.charCodeAt(c);
    }
  }

  return {
    bufferData,
    byteLength,
    stringCount: count,
    slotBytes,
    maxCharsPerString: maxChars
  };
}

/**
 * Checks whether the dataset fits within device storage buffer limits.
 */
export function checkMemoryBudget(
  itemCount: number,
  slotBytes: 64 | 128 = 64,
  device?: GPUDevice | null
): MemoryBudgetCheck {
  const requiredBytes = itemCount * slotBytes;
  const maxBytes = device?.limits?.maxStorageBufferBindingSize ?? (128 * 1024 * 1024);

  if (requiredBytes > maxBytes) {
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
