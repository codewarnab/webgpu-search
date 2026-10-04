/**
 * Adaptive per-value posting list for columnar filter indexes.
 *
 * Low-frequency values are kept as a sparse `Set<number>` of row indices
 * (memory proportional to the value's document count); a value is promoted
 * to a capacity-sized `DocumentBitset` only once it is dense enough that the
 * bitset is the cheaper representation. Dense lists are demoted back to
 * sparse when they become sparse relative to the store capacity (after
 * removals or capacity growth), so total memory stays proportional to the
 * number of (row, value) entries rather than `distinctValues x capacity`.
 * 100% portable (zero DOM references).
 */

import { DocumentBitset } from './doc-bitset';

/** Minimum doc count before a value may be promoted to a dense bitset. */
const MIN_DENSE_COUNT = 64;

/** Promote when count exceeds capacity / PROMOTE_DIVISOR (a Set costs ~16+ B/entry vs 1 bit/row). */
const PROMOTE_DIVISOR = 128;

/** Demote when count drops below capacity / DEMOTE_DIVISOR (hysteresis). */
const DEMOTE_DIVISOR = 256;

function promoteThreshold(capacity: number): number {
  return Math.max(MIN_DENSE_COUNT, Math.floor(capacity / PROMOTE_DIVISOR));
}

function demoteThreshold(capacity: number): number {
  return Math.max(MIN_DENSE_COUNT >>> 1, Math.floor(capacity / DEMOTE_DIVISOR));
}

export class PostingList {
  private sparse: Set<number> | null = new Set();
  private dense: DocumentBitset | null = null;
  private count = 0;

  get size(): number {
    return this.count;
  }

  /** True when backed by a capacity-sized bitset (exposed for tests/diagnostics). */
  get isDense(): boolean {
    return this.dense !== null;
  }

  has(docIndex: number): boolean {
    return this.dense ? this.dense.has(docIndex) : this.sparse!.has(docIndex);
  }

  add(docIndex: number, capacity: number): void {
    if (this.dense) {
      if (this.dense.has(docIndex)) return;
      this.dense.set(docIndex);
      this.count++;
      return;
    }
    const sparse = this.sparse!;
    if (sparse.has(docIndex)) return;
    sparse.add(docIndex);
    this.count++;
    if (this.count > promoteThreshold(capacity)) {
      const bs = new DocumentBitset(capacity);
      for (const d of sparse) bs.set(d);
      this.dense = bs;
      this.sparse = null;
    }
  }

  delete(docIndex: number, capacity: number): void {
    if (this.dense) {
      if (!this.dense.has(docIndex)) return;
      this.dense.clear(docIndex);
      this.count--;
      if (this.count < demoteThreshold(capacity)) this.demote();
      return;
    }
    if (this.sparse!.delete(docIndex)) this.count--;
  }

  /** Called on store capacity growth: grow dense bitsets or demote them. */
  ensureCapacity(capacity: number): void {
    if (!this.dense) return;
    if (this.count < demoteThreshold(capacity)) {
      this.demote();
    } else {
      this.dense.ensureCapacity(capacity);
    }
  }

  /** ORs this posting list into `target` (sized to the store capacity). */
  orInto(target: DocumentBitset): void {
    if (this.dense) {
      target.orInPlace(this.dense);
      return;
    }
    for (const d of this.sparse!) target.set(d);
  }

  /** Materializes this list as a fresh bitset of `capacity`. */
  toBitset(capacity: number): DocumentBitset {
    if (this.dense) {
      const out = new DocumentBitset(capacity);
      out.orInPlace(this.dense);
      return out;
    }
    const out = new DocumentBitset(capacity);
    for (const d of this.sparse!) out.set(d);
    return out;
  }

  private demote(): void {
    const dense = this.dense!;
    this.sparse = new Set(dense.getMatchingIndices());
    this.dense = null;
  }
}
