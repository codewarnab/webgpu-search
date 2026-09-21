/**
 * High-speed document bitset for columnar candidate filtering and evaluation.
 * Backed by contiguous Uint32Array words with O(1) bitwise operations.
 * 100% portable across Web Workers, Node.js, and SSR (zero DOM references).
 */

export class DocumentBitset {
  capacity: number;
  words: Uint32Array;
  private cachedPopcount: number = -1;
  private isDirty: boolean = true;

  constructor(capacity: number = 0) {
    const safe = Number.isFinite(capacity) ? Math.max(0, Math.floor(capacity)) : 0;
    this.capacity = safe;
    const wordCount = (this.capacity + 31) >>> 5;
    this.words = new Uint32Array(Math.max(wordCount, 1));
  }

  static all(capacity: number): DocumentBitset {
    const bs = new DocumentBitset(capacity);
    bs.fill(true, capacity);
    return bs;
  }

  static none(capacity: number): DocumentBitset {
    return new DocumentBitset(capacity);
  }

  static fromIndices(indices: number[], capacity?: number): DocumentBitset {
    let maxIdx = -1;
    for (let i = 0; i < indices.length; i++) {
      const idx = indices[i];
      if (Number.isInteger(idx) && idx > maxIdx) maxIdx = idx;
    }
    const cap = capacity !== undefined ? Math.max(capacity, maxIdx + 1) : maxIdx + 1;
    const bs = new DocumentBitset(cap);
    for (let i = 0; i < indices.length; i++) {
      const idx = indices[i];
      if (Number.isInteger(idx) && idx >= 0) {
        bs.set(idx);
      }
    }
    return bs;
  }

  get size(): number {
    return this.capacity;
  }

  ensureCapacity(minCapacity: number): void {
    if (!Number.isFinite(minCapacity)) return;
    minCapacity = Math.floor(minCapacity);
    if (minCapacity <= this.capacity) return;
    const newWordCount = (minCapacity + 31) >>> 5;
    if (newWordCount > this.words.length) {
      const allocWordCount = Math.max(this.words.length * 2, newWordCount, 4);
      const newWords = new Uint32Array(allocWordCount);
      newWords.set(this.words);
      this.words = newWords;
    }
    this.capacity = minCapacity;
  }

  set(docIndex: number): void {
    if (!Number.isInteger(docIndex) || docIndex < 0) return;
    if (docIndex >= this.capacity) {
      this.ensureCapacity(docIndex + 1);
    }
    const wordIdx = docIndex >>> 5;
    const bitMask = 1 << (docIndex & 31);
    if ((this.words[wordIdx] & bitMask) === 0) {
      this.words[wordIdx] |= bitMask;
      this.isDirty = true;
    }
  }

  clear(docIndex: number): void {
    if (!Number.isInteger(docIndex) || docIndex < 0 || docIndex >= this.capacity) return;
    const wordIdx = docIndex >>> 5;
    const bitMask = 1 << (docIndex & 31);
    if ((this.words[wordIdx] & bitMask) !== 0) {
      this.words[wordIdx] &= ~bitMask;
      this.isDirty = true;
    }
  }

  has(docIndex: number): boolean {
    if (!Number.isInteger(docIndex) || docIndex < 0 || docIndex >= this.capacity) return false;
    const wordIdx = docIndex >>> 5;
    return (this.words[wordIdx] & (1 << (docIndex & 31))) !== 0;
  }

  get(docIndex: number): boolean {
    return this.has(docIndex);
  }

  toggle(docIndex: number): void {
    if (!Number.isInteger(docIndex) || docIndex < 0) return;
    if (docIndex >= this.capacity) {
      this.ensureCapacity(docIndex + 1);
    }
    const wordIdx = docIndex >>> 5;
    this.words[wordIdx] ^= (1 << (docIndex & 31));
    this.isDirty = true;
  }

  fill(value: boolean, count?: number): void {
    const rawLimit = count !== undefined ? count : this.capacity;
    const limit = Number.isFinite(rawLimit) ? Math.min(Math.floor(rawLimit), this.capacity) : 0;
    if (limit <= 0) {
      this.words.fill(0);
      this.cachedPopcount = 0;
      this.isDirty = false;
      return;
    }

    if (value) {
      const fullWords = limit >>> 5;
      this.words.fill(0xffffffff, 0, fullWords);
      const remainder = limit & 31;
      if (remainder > 0) {
        const mask = ~0 >>> (32 - remainder);
        this.words[fullWords] = mask;
        this.words.fill(0, fullWords + 1);
      } else {
        this.words.fill(0, fullWords);
      }
    } else {
      this.words.fill(0);
    }
    this.isDirty = true;
  }

  popcount(): number {
    if (!this.isDirty && this.cachedPopcount >= 0) {
      return this.cachedPopcount;
    }

    let count = 0;
    const words = this.words;
    const numWords = (this.capacity + 31) >>> 5;

    for (let i = 0; i < numWords; i++) {
      let v = words[i];
      if (v !== 0) {
        // Fast 32-bit Hamming weight algorithm
        v = v - ((v >>> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
        count += (((v + (v >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
      }
    }

    this.cachedPopcount = count;
    this.isDirty = false;
    return count;
  }

  isEmpty(): boolean {
    if (!this.isDirty && this.cachedPopcount >= 0) {
      return this.cachedPopcount === 0;
    }
    const numWords = (this.capacity + 31) >>> 5;
    for (let i = 0; i < numWords; i++) {
      if (this.words[i] !== 0) return false;
    }
    this.cachedPopcount = 0;
    this.isDirty = false;
    return true;
  }

  getMatchingIndices(): number[] {
    const result: number[] = [];
    const words = this.words;
    const numWords = (this.capacity + 31) >>> 5;
    const capacity = this.capacity;

    for (let w = 0; w < numWords; w++) {
      let word = words[w];
      if (word === 0) continue;
      const base = w << 5;
      while (word !== 0) {
        const t = word & -word;
        const bit = 31 - Math.clz32(t);
        const idx = base + bit;
        if (idx < capacity) {
          result.push(idx);
        }
        word ^= t;
      }
    }

    return result;
  }

  clone(): DocumentBitset {
    const copy = new DocumentBitset(this.capacity);
    const wordCount = (this.capacity + 31) >>> 5;
    copy.words = new Uint32Array(Math.max(wordCount, 1));
    copy.words.set(this.words.subarray(0, wordCount));
    copy.cachedPopcount = this.cachedPopcount;
    copy.isDirty = this.isDirty;
    return copy;
  }

  and(other: DocumentBitset): DocumentBitset {
    const cap = Math.min(this.capacity, other.capacity);
    const result = new DocumentBitset(cap);
    const wordCount = (cap + 31) >>> 5;
    for (let w = 0; w < wordCount; w++) {
      result.words[w] = this.words[w] & other.words[w];
    }
    maskRemainder(result.words, cap);
    return result;
  }

  andInPlace(other: DocumentBitset): this {
    const cap = Math.min(this.capacity, other.capacity);
    this.capacity = cap;
    const wordCount = (cap + 31) >>> 5;
    for (let w = 0; w < wordCount; w++) {
      this.words[w] &= other.words[w];
    }
    // Clear any extra words beyond cap
    if (wordCount < this.words.length) {
      this.words.fill(0, wordCount);
    }
    this.isDirty = true;
    return this;
  }

  or(other: DocumentBitset): DocumentBitset {
    const cap = Math.max(this.capacity, other.capacity);
    const result = new DocumentBitset(cap);
    const wordCount = (cap + 31) >>> 5;
    const thisNumWords = (this.capacity + 31) >>> 5;
    const otherNumWords = (other.capacity + 31) >>> 5;
    for (let w = 0; w < wordCount; w++) {
      const w1 = w < thisNumWords ? this.words[w] : 0;
      const w2 = w < otherNumWords ? other.words[w] : 0;
      result.words[w] = w1 | w2;
    }
    maskRemainder(result.words, cap);
    return result;
  }

  orInPlace(other: DocumentBitset): this {
    if (other.capacity > this.capacity) {
      this.ensureCapacity(other.capacity);
    }
    const otherWordCount = (other.capacity + 31) >>> 5;
    for (let w = 0; w < otherWordCount; w++) {
      this.words[w] |= other.words[w];
    }
    this.isDirty = true;
    return this;
  }

  andNot(other: DocumentBitset): DocumentBitset {
    const cap = this.capacity;
    const result = new DocumentBitset(cap);
    const wordCount = (cap + 31) >>> 5;
    const otherNumWords = (other.capacity + 31) >>> 5;
    for (let w = 0; w < wordCount; w++) {
      const otherWord = w < otherNumWords ? other.words[w] : 0;
      result.words[w] = this.words[w] & ~otherWord;
    }
    maskRemainder(result.words, cap);
    return result;
  }

  andNotInPlace(other: DocumentBitset): this {
    const wordCount = (this.capacity + 31) >>> 5;
    const otherWordCount = (other.capacity + 31) >>> 5;
    const minWords = Math.min(wordCount, otherWordCount);
    for (let w = 0; w < minWords; w++) {
      this.words[w] &= ~other.words[w];
    }
    this.isDirty = true;
    return this;
  }

  not(capacity?: number): DocumentBitset {
    const cap = capacity !== undefined
      ? (Number.isFinite(capacity) ? Math.max(0, Math.floor(capacity)) : 0)
      : this.capacity;
    const result = new DocumentBitset(cap);
    const wordCount = (cap + 31) >>> 5;
    const thisNumWords = (this.capacity + 31) >>> 5;
    for (let w = 0; w < wordCount; w++) {
      const srcWord = w < thisNumWords ? this.words[w] : 0;
      result.words[w] = ~srcWord;
    }
    const remainder = cap & 31;
    if (remainder > 0 && wordCount > 0) {
      const mask = ~0 >>> (32 - remainder);
      result.words[wordCount - 1] &= mask;
    }
    return result;
  }

  notInPlace(capacity?: number): this {
    const cap = capacity !== undefined
      ? (Number.isFinite(capacity) ? Math.max(0, Math.floor(capacity)) : 0)
      : this.capacity;
    this.ensureCapacity(cap);
    this.capacity = cap;
    const wordCount = (cap + 31) >>> 5;
    for (let w = 0; w < wordCount; w++) {
      this.words[w] = ~this.words[w];
    }
    const remainder = cap & 31;
    if (remainder > 0 && wordCount > 0) {
      const mask = ~0 >>> (32 - remainder);
      this.words[wordCount - 1] &= mask;
    }
    if (wordCount < this.words.length) {
      this.words.fill(0, wordCount);
    }
    this.isDirty = true;
    return this;
  }
}

/** Masks off trailing bits above the logical capacity in the last word. */
function maskRemainder(words: Uint32Array, cap: number): void {
  const remainder = cap & 31;
  const wordCount = (cap + 31) >>> 5;
  if (remainder > 0 && wordCount > 0 && wordCount <= words.length) {
    const mask = ~0 >>> (32 - remainder);
    words[wordCount - 1] &= mask;
  }
}
