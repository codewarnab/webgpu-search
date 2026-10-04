/**
 * Columnar attribute storage for structured pre-filtering and facet aggregations.
 * Backed by typed arrays (Float64Array, Uint32Array), DocumentBitsets and
 * adaptive per-value posting lists (sparse sets promoted to bitsets only when
 * dense), so memory stays proportional to the stored (row, value) entries
 * rather than `distinctValues x capacity`.
 *
 * Value coercion (applied identically at index and query time):
 * - `number`: finite numbers, or non-blank numeric strings; anything else
 *   (blank strings, NaN, +/-Infinity, booleans, objects) is treated as missing.
 * - `boolean`: only `true`/`false` and the exact strings `"true"`/`"false"`;
 *   any other value is treated as missing at index time and rejected with
 *   `InvalidFilterError` at query time.
 * - Default getters read own properties only (plus non-`Object.prototype`
 *   inherited ones such as class accessors), never `constructor`/`toString`.
 *
 * 100% portable across browser main thread, Web Workers, Node.js, and SSR (zero DOM references).
 */

import { DocumentBitset } from './doc-bitset';
import { PostingList } from './posting-list';
import type {
  FilterFieldDefinition,
  FilterFieldType,
  FilterValue,
  FieldComparison,
  TermsFacetBucket
} from '../types';
import { InvalidFilterError } from '../errors';

interface BaseColumn {
  name: string;
  type: FilterFieldType;
  getter: (doc: any) => any;
  presence: DocumentBitset;
}

interface StringColumn extends BaseColumn {
  type: 'string';
  stringTable: string[];
  stringToCode: Map<string, number>;
  codes: Uint32Array;
  /** code -> rows holding it; entries with zero rows are removed eagerly. */
  invertedIndex: Map<number, PostingList>;
}

interface NumberColumn extends BaseColumn {
  type: 'number';
  values: Float64Array;
}

interface BooleanColumn extends BaseColumn {
  type: 'boolean';
  trueBitset: DocumentBitset;
}

interface StringArrayColumn extends BaseColumn {
  type: 'string[]';
  /** tag -> rows holding it; entries with zero rows are removed eagerly. */
  tagInverted: Map<string, PostingList>;
  docTags: Map<number, string[]>;
}

type Column = StringColumn | NumberColumn | BooleanColumn | StringArrayColumn;

/**
 * Default field accessor: reads `doc[name]` only when it is an own property
 * or an inherited one not provided by `Object.prototype` (so class accessors
 * still work, but `constructor`, `toString`, `__proto__`, ... never leak in as
 * field values).
 */
export function readOwnField(doc: any, name: string): any {
  if (doc === null || doc === undefined) return undefined;
  if (Object.prototype.hasOwnProperty.call(doc, name)) return doc[name];
  if (name in Object.prototype) return undefined;
  return doc[name];
}

/**
 * Strict numeric coercion shared by index and query paths: finite numbers
 * and non-blank numeric strings only. Returns NaN for anything else
 * (`Number("")`/`Number("  ")` would otherwise silently yield 0).
 */
function coerceNumber(raw: unknown): number {
  if (typeof raw === 'number') return Number.isFinite(raw) ? raw : NaN;
  if (typeof raw === 'string') {
    if (raw.trim().length === 0) return NaN;
    const n = Number(raw);
    return Number.isFinite(n) ? n : NaN;
  }
  return NaN;
}

/**
 * Strict boolean coercion shared by index and query paths: `true`/`false`
 * and the exact strings `"true"`/`"false"`; `undefined` for anything else.
 */
function coerceBoolean(raw: unknown): boolean | undefined {
  if (raw === true || raw === 'true') return true;
  if (raw === false || raw === 'false') return false;
  return undefined;
}

/** Query-time boolean operand: rejects non-boolean values fail-closed. */
function toBooleanOperand(raw: unknown, field: string): boolean {
  const b = coerceBoolean(raw);
  if (b === undefined) {
    throw new InvalidFilterError(
      `Boolean field "${field}" expects true/false (or "true"/"false"), got ${JSON.stringify(raw)}.`,
      field
    );
  }
  return b;
}

/** Validates a numeric range bound, throwing InvalidFilterError on non-numeric input. */
function toNumberBound(raw: unknown, op: string, field: string): number {
  if (typeof raw !== 'number' && typeof raw !== 'string') {
    throw new InvalidFilterError(
      `Range operator "${op}" on field "${field}" expects a numeric bound, got ${raw === null ? 'null' : typeof raw}.`,
      field
    );
  }
  const n = coerceNumber(raw);
  if (Number.isNaN(n)) {
    throw new InvalidFilterError(
      `Range operator "${op}" on field "${field}" expects a numeric bound, got ${JSON.stringify(String(raw))}.`,
      field
    );
  }
  return n;
}

/**
 * Normalized column value produced by `ColumnarStore.prepare`:
 * `undefined` = missing; otherwise string / finite number / boolean /
 * non-empty deduped string[] matching the column type.
 */
export type PreparedColumnValue = string | number | boolean | string[] | undefined;

/** One normalized value per registered column, in column order. */
export type PreparedColumnarRow = PreparedColumnValue[];

function normalizeColumnValue(type: FilterFieldType, rawVal: unknown): PreparedColumnValue {
  if (rawVal === null || rawVal === undefined) return undefined;
  switch (type) {
    case 'string':
      return String(rawVal);
    case 'number': {
      const n = coerceNumber(rawVal);
      return Number.isFinite(n) ? n : undefined;
    }
    case 'boolean':
      return coerceBoolean(rawVal);
    case 'string[]': {
      if (!Array.isArray(rawVal) || rawVal.length === 0) return undefined;
      const seen = new Set<string>();
      const tags: string[] = [];
      for (let i = 0; i < rawVal.length; i++) {
        const item = rawVal[i];
        if (item === null || item === undefined) continue;
        const t = String(item);
        if (seen.has(t)) continue;
        seen.add(t);
        tags.push(t);
      }
      return tags.length > 0 ? tags : undefined;
    }
  }
  return undefined;
}

function assertDocIndex(docIndex: number, op: string): void {
  if (!Number.isInteger(docIndex) || docIndex < 0) {
    throw new RangeError(
      `[webgpu-search] ColumnarStore.${op} requires a non-negative integer docIndex, got ${String(docIndex)}.`
    );
  }
}

export interface ColumnarStoreOptions {
  initialCapacity?: number;
  growthFactor?: number;
}

export class ColumnarStore<TDoc = Record<string, unknown>> {
  private columns: Map<string, Column> = new Map();
  private activeDocs: DocumentBitset;
  capacity: number;
  private readonly growthFactor: number;
  private readonly fieldDefs: FilterFieldDefinition<TDoc>[];

  constructor(
    fieldDefs: FilterFieldDefinition<TDoc>[] = [],
    options: ColumnarStoreOptions = {}
  ) {
    this.fieldDefs = fieldDefs;
    this.capacity = (Math.max(
      typeof options.initialCapacity === 'number' && options.initialCapacity > 0
        ? Math.floor(options.initialCapacity)
        : 16,
      16
    ) + 31) & ~31;
    this.growthFactor = typeof options.growthFactor === 'number' && options.growthFactor >= 1.0
      ? options.growthFactor
      : 1.5;
    this.activeDocs = new DocumentBitset(this.capacity);

    for (let i = 0; i < fieldDefs.length; i++) {
      this.registerField(fieldDefs[i]);
    }
  }

  private registerField(fieldDef: FilterFieldDefinition<TDoc>): void {
    const name = fieldDef.name;
    const getter = fieldDef.getter ?? ((doc: any) => readOwnField(doc, name));
    const type: FilterFieldType = fieldDef.type ?? 'string';
    const cap = this.capacity;

    switch (type) {
      case 'string': {
        const col: StringColumn = {
          name,
          type: 'string',
          getter,
          presence: new DocumentBitset(cap),
          stringTable: [],
          stringToCode: new Map(),
          codes: new Uint32Array(cap),
          invertedIndex: new Map()
        };
        this.columns.set(name, col);
        break;
      }
      case 'number': {
        const values = new Float64Array(cap);
        values.fill(NaN);
        const col: NumberColumn = {
          name,
          type: 'number',
          getter,
          presence: new DocumentBitset(cap),
          values
        };
        this.columns.set(name, col);
        break;
      }
      case 'boolean': {
        const col: BooleanColumn = {
          name,
          type: 'boolean',
          getter,
          presence: new DocumentBitset(cap),
          trueBitset: new DocumentBitset(cap)
        };
        this.columns.set(name, col);
        break;
      }
      case 'string[]': {
        const col: StringArrayColumn = {
          name,
          type: 'string[]',
          getter,
          presence: new DocumentBitset(cap),
          tagInverted: new Map(),
          docTags: new Map()
        };
        this.columns.set(name, col);
        break;
      }
      default:
        throw new InvalidFilterError(`Unsupported filter field type "${String(type)}"`, name);
    }
  }

  hasField(name: string): boolean {
    return this.columns.has(name);
  }

  getFieldType(name: string): FilterFieldType | undefined {
    return this.columns.get(name)?.type;
  }

  getFieldNames(): string[] {
    return Array.from(this.columns.keys());
  }

  getFieldDefs(): FilterFieldDefinition<TDoc>[] {
    return this.fieldDefs.slice();
  }

  getActiveDocs(): DocumentBitset {
    return this.activeDocs;
  }

  ensureCapacity(minCapacity: number): void {
    if (minCapacity <= this.capacity) return;
    const newCap = (Math.max(Math.floor(this.capacity * this.growthFactor), minCapacity, 16) + 31) & ~31;
    this.capacity = newCap;
    this.activeDocs.ensureCapacity(newCap);

    for (const col of this.columns.values()) {
      col.presence.ensureCapacity(newCap);
      switch (col.type) {
        case 'string': {
          const newCodes = new Uint32Array(newCap);
          newCodes.set(col.codes);
          col.codes = newCodes;
          for (const pl of col.invertedIndex.values()) {
            pl.ensureCapacity(newCap);
          }
          break;
        }
        case 'number': {
          const newValues = new Float64Array(newCap);
          newValues.fill(NaN);
          newValues.set(col.values);
          col.values = newValues;
          break;
        }
        case 'boolean': {
          col.trueBitset.ensureCapacity(newCap);
          break;
        }
        case 'string[]': {
          for (const pl of col.tagInverted.values()) {
            pl.ensureCapacity(newCap);
          }
          break;
        }
      }
    }
  }

  init(records: TDoc[]): void {
    const count = records.length;
    if (count > this.capacity) {
      this.ensureCapacity(count);
    }
    // Full reset so re-init never leaks stale column state.
    this.activeDocs = new DocumentBitset(this.capacity);
    for (const col of this.columns.values()) {
      col.presence = new DocumentBitset(this.capacity);
      switch (col.type) {
        case 'string': {
          col.stringTable = [];
          col.stringToCode.clear();
          col.codes = new Uint32Array(this.capacity);
          col.invertedIndex.clear();
          break;
        }
        case 'number': {
          col.values = new Float64Array(this.capacity);
          col.values.fill(NaN);
          break;
        }
        case 'boolean': {
          col.trueBitset = new DocumentBitset(this.capacity);
          break;
        }
        case 'string[]': {
          col.tagInverted.clear();
          col.docTags.clear();
          break;
        }
      }
    }

    for (let d = 0; d < count; d++) {
      const doc = records[d];
      if (doc !== null && doc !== undefined) {
        this.add(d, doc);
      }
    }
  }

  /**
   * Runs every column getter for `doc` and normalizes the results into a
   * plain per-column value vector. This is the only step that executes user
   * code (getters, `String()` conversions) and may throw; it performs no
   * mutation, so callers needing all-or-nothing semantics (e.g.
   * `DocumentIndex.applyBatch`) can prepare every doc before mutating
   * anything and then apply with `addPrepared`, which cannot throw on value
   * conversion.
   */
  prepare(doc: TDoc): PreparedColumnarRow {
    const out: PreparedColumnValue[] = new Array(this.columns.size);
    let i = 0;
    for (const col of this.columns.values()) {
      out[i++] = normalizeColumnValue(col.type, col.getter(doc));
    }
    return out;
  }

  add(docIndex: number, doc: TDoc): void {
    assertDocIndex(docIndex, 'add');
    this.addPrepared(docIndex, this.prepare(doc));
  }

  /** Applies a row produced by `prepare` (no user code runs here). */
  addPrepared(docIndex: number, row: PreparedColumnarRow): void {
    assertDocIndex(docIndex, 'add');
    if (!Array.isArray(row) || row.length !== this.columns.size) {
      throw new TypeError(
        `[webgpu-search] ColumnarStore.addPrepared expects a prepared row of ${this.columns.size} values.`
      );
    }
    if (docIndex >= this.capacity) {
      this.ensureCapacity(docIndex + 1);
    }
    this.activeDocs.set(docIndex);

    let i = 0;
    for (const col of this.columns.values()) {
      // Clear-then-set so re-add on an occupied slot never leaks stale
      // string codes or tag bits.
      this.clearDocColumnValue(col, docIndex);
      this.setDocColumnValue(col, docIndex, row[i++]);
    }
  }

  update(docIndex: number, doc: TDoc): void {
    assertDocIndex(docIndex, 'update');
    this.addPrepared(docIndex, this.prepare(doc));
  }

  remove(docIndex: number): void {
    if (docIndex < 0 || docIndex >= this.capacity) return;
    this.activeDocs.clear(docIndex);

    for (const col of this.columns.values()) {
      this.clearDocColumnValue(col, docIndex);
    }
  }

  private setDocColumnValue(col: Column, docIndex: number, val: PreparedColumnValue): void {
    if (val === undefined) {
      col.presence.clear(docIndex);
      return;
    }

    switch (col.type) {
      case 'string': {
        const str = val as string;
        col.presence.set(docIndex);
        let code = col.stringToCode.get(str);
        if (code === undefined) {
          code = col.stringTable.length;
          col.stringTable.push(str);
          col.stringToCode.set(str, code);
        }
        let pl = col.invertedIndex.get(code);
        if (!pl) {
          pl = new PostingList();
          col.invertedIndex.set(code, pl);
        }
        col.codes[docIndex] = code;
        pl.add(docIndex, this.capacity);
        break;
      }
      case 'number': {
        col.presence.set(docIndex);
        col.values[docIndex] = val as number;
        break;
      }
      case 'boolean': {
        col.presence.set(docIndex);
        if (val === true) {
          col.trueBitset.set(docIndex);
        } else {
          col.trueBitset.clear(docIndex);
        }
        break;
      }
      case 'string[]': {
        // Tags are already per-doc deduped by normalizeColumnValue: facet
        // counts are per-doc (not per-occurrence).
        const tags = val as string[];
        for (let i = 0; i < tags.length; i++) {
          let pl = col.tagInverted.get(tags[i]);
          if (!pl) {
            pl = new PostingList();
            col.tagInverted.set(tags[i], pl);
          }
          pl.add(docIndex, this.capacity);
        }
        col.presence.set(docIndex);
        col.docTags.set(docIndex, tags);
        break;
      }
    }
  }

  private clearDocColumnValue(col: Column, docIndex: number): void {
    if (!col.presence.has(docIndex)) return;
    col.presence.clear(docIndex);

    switch (col.type) {
      case 'string': {
        const code = col.codes[docIndex];
        const pl = col.invertedIndex.get(code);
        if (pl) {
          pl.delete(docIndex, this.capacity);
          if (pl.size === 0) col.invertedIndex.delete(code);
        }
        col.codes[docIndex] = 0;
        break;
      }
      case 'number': {
        col.values[docIndex] = NaN;
        break;
      }
      case 'boolean': {
        col.trueBitset.clear(docIndex);
        break;
      }
      case 'string[]': {
        const prevTags = col.docTags.get(docIndex);
        if (prevTags) {
          for (let i = 0; i < prevTags.length; i++) {
            const pl = col.tagInverted.get(prevTags[i]);
            if (pl) {
              pl.delete(docIndex, this.capacity);
              if (pl.size === 0) col.tagInverted.delete(prevTags[i]);
            }
          }
          col.docTags.delete(docIndex);
        }
        break;
      }
    }
  }

  compact(oldToNewDocIndexMap: Map<number, number>, newDocCount: number): void {
    const newCap = (Math.max(newDocCount, 16) + 31) & ~31;
    this.capacity = newCap;
    this.activeDocs = new DocumentBitset(newCap);
    for (let i = 0; i < newDocCount; i++) {
      this.activeDocs.set(i);
    }

    for (const [name, col] of this.columns.entries()) {
      switch (col.type) {
        case 'string': {
          // Re-dictionary live values only: codes with zero surviving rows
          // are dropped. Codes are assigned in old-code order so the table
          // stays deterministic.
          const newCol: StringColumn = {
            name,
            type: 'string',
            getter: col.getter,
            presence: new DocumentBitset(newCap),
            stringTable: [],
            stringToCode: new Map(),
            codes: new Uint32Array(newCap),
            invertedIndex: new Map()
          };
          const liveOld: number[] = [];
          for (const [oldD] of oldToNewDocIndexMap.entries()) {
            if (col.presence.has(oldD)) liveOld.push(col.codes[oldD]);
          }
          liveOld.sort((a, b) => a - b);
          const oldToNewCode = new Map<number, number>();
          for (let i = 0; i < liveOld.length; i++) {
            const oc = liveOld[i];
            if (oldToNewCode.has(oc)) continue;
            const nc = newCol.stringTable.length;
            const str = col.stringTable[oc];
            oldToNewCode.set(oc, nc);
            newCol.stringTable.push(str);
            newCol.stringToCode.set(str, nc);
          }
          for (const [oldD, newD] of oldToNewDocIndexMap.entries()) {
            if (col.presence.has(oldD)) {
              newCol.presence.set(newD);
              const code = oldToNewCode.get(col.codes[oldD])!;
              newCol.codes[newD] = code;
              let pl = newCol.invertedIndex.get(code);
              if (!pl) {
                pl = new PostingList();
                newCol.invertedIndex.set(code, pl);
              }
              pl.add(newD, newCap);
            }
          }
          this.columns.set(name, newCol);
          break;
        }
        case 'number': {
          const newValues = new Float64Array(newCap);
          newValues.fill(NaN);
          const newCol: NumberColumn = {
            name,
            type: 'number',
            getter: col.getter,
            presence: new DocumentBitset(newCap),
            values: newValues
          };
          for (const [oldD, newD] of oldToNewDocIndexMap.entries()) {
            if (col.presence.has(oldD)) {
              newCol.presence.set(newD);
              newCol.values[newD] = col.values[oldD];
            }
          }
          this.columns.set(name, newCol);
          break;
        }
        case 'boolean': {
          const newCol: BooleanColumn = {
            name,
            type: 'boolean',
            getter: col.getter,
            presence: new DocumentBitset(newCap),
            trueBitset: new DocumentBitset(newCap)
          };
          for (const [oldD, newD] of oldToNewDocIndexMap.entries()) {
            if (col.presence.has(oldD)) {
              newCol.presence.set(newD);
              if (col.trueBitset.has(oldD)) {
                newCol.trueBitset.set(newD);
              }
            }
          }
          this.columns.set(name, newCol);
          break;
        }
        case 'string[]': {
          const newCol: StringArrayColumn = {
            name,
            type: 'string[]',
            getter: col.getter,
            presence: new DocumentBitset(newCap),
            tagInverted: new Map(),
            docTags: new Map()
          };
          for (const [oldD, newD] of oldToNewDocIndexMap.entries()) {
            if (col.presence.has(oldD)) {
              newCol.presence.set(newD);
              const tags = col.docTags.get(oldD);
              if (tags) {
                newCol.docTags.set(newD, tags.slice());
                for (let i = 0; i < tags.length; i++) {
                  let pl = newCol.tagInverted.get(tags[i]);
                  if (!pl) {
                    pl = new PostingList();
                    newCol.tagInverted.set(tags[i], pl);
                  }
                  pl.add(newD, newCap);
                }
              }
            }
          }
          this.columns.set(name, newCol);
          break;
        }
      }
    }
  }

  evaluateEquality(name: string, value: FilterValue, activeMask?: DocumentBitset): DocumentBitset {
    const col = this.columns.get(name);
    if (!col) {
      throw new InvalidFilterError(`Unknown filter field "${name}"`, name);
    }

    const active = activeMask ?? this.activeDocs;

    if (value === null) {
      return active.andNot(col.presence);
    }

    switch (col.type) {
      case 'string': {
        const str = String(value);
        const code = col.stringToCode.get(str);
        if (code === undefined) {
          return DocumentBitset.none(this.capacity);
        }
        const pl = col.invertedIndex.get(code);
        return pl ? pl.toBitset(this.capacity).andInPlace(active) : DocumentBitset.none(this.capacity);
      }
      case 'number': {
        const target = coerceNumber(value);
        if (!Number.isFinite(target)) {
          return DocumentBitset.none(this.capacity);
        }
        const result = new DocumentBitset(this.capacity);
        const activeWords = active.words;
        const presWords = col.presence.words;
        const numWords = Math.min(activeWords.length, presWords.length, result.words.length);
        const vals = col.values;

        for (let w = 0; w < numWords; w++) {
          let mask = activeWords[w] & presWords[w];
          if (mask === 0) continue;
          const base = w << 5;
          let outWord = 0;
          while (mask !== 0) {
            const t = mask & -mask;
            const bit = 31 - Math.clz32(t);
            const d = base + bit;
            if (vals[d] === target) {
              outWord |= t;
            }
            mask ^= t;
          }
          result.words[w] = outWord;
        }
        return result;
      }
      case 'boolean': {
        const target = toBooleanOperand(value, name);
        if (target) {
          return col.trueBitset.and(active);
        } else {
          return col.presence.andNot(col.trueBitset).and(active);
        }
      }
      case 'string[]': {
        const tag = String(value);
        const pl = col.tagInverted.get(tag);
        return pl ? pl.toBitset(this.capacity).andInPlace(active) : DocumentBitset.none(this.capacity);
      }
    }
  }

  evaluateIn(name: string, values: FilterValue[], activeMask?: DocumentBitset): DocumentBitset {
    const col = this.columns.get(name);
    if (!col) {
      throw new InvalidFilterError(`Unknown filter field "${name}"`, name);
    }

    if (!Array.isArray(values) || values.length === 0) {
      return DocumentBitset.none(this.capacity);
    }

    const active = activeMask ?? this.activeDocs;
    const result = new DocumentBitset(this.capacity);

    switch (col.type) {
      case 'string': {
        for (let i = 0; i < values.length; i++) {
          const v = values[i];
          if (v === null) {
            result.orInPlace(active.andNot(col.presence));
          } else {
            const code = col.stringToCode.get(String(v));
            if (code !== undefined) {
              col.invertedIndex.get(code)?.orInto(result);
            }
          }
        }
        return result.andInPlace(active);
      }
      case 'string[]': {
        let includeNull = false;
        for (let i = 0; i < values.length; i++) {
          const v = values[i];
          if (v === null) {
            includeNull = true;
          } else {
            col.tagInverted.get(String(v))?.orInto(result);
          }
        }
        if (includeNull) {
          result.orInPlace(active.andNot(col.presence));
        }
        return result.andInPlace(active);
      }
      case 'number': {
        const numSet = new Set<number>();
        let includeNull = false;
        for (let i = 0; i < values.length; i++) {
          const v = values[i];
          if (v === null) {
            includeNull = true;
          } else {
            const n = coerceNumber(v);
            if (Number.isFinite(n)) numSet.add(n);
          }
        }
        const activeWords = active.words;
        const presWords = col.presence.words;
        const numWords = Math.min(activeWords.length, presWords.length, result.words.length);
        const vals = col.values;

        for (let w = 0; w < numWords; w++) {
          let mask = activeWords[w] & presWords[w];
          if (mask === 0) continue;
          const base = w << 5;
          let outWord = 0;
          while (mask !== 0) {
            const t = mask & -mask;
            const bit = 31 - Math.clz32(t);
            const d = base + bit;
            if (numSet.has(vals[d])) {
              outWord |= t;
            }
            mask ^= t;
          }
          result.words[w] = outWord;
        }
        if (includeNull) {
          result.orInPlace(active.andNot(col.presence));
        }
        return result;
      }
      case 'boolean': {
        for (let i = 0; i < values.length; i++) {
          const v = values[i];
          if (v === null) {
            result.orInPlace(active.andNot(col.presence));
          } else if (toBooleanOperand(v, name)) {
            result.orInPlace(col.trueBitset);
          } else {
            result.orInPlace(col.presence.andNot(col.trueBitset));
          }
        }
        return result.andInPlace(active);
      }
    }
  }

  evaluateRange(name: string, comparison: FieldComparison, activeMask?: DocumentBitset): DocumentBitset {
    const col = this.columns.get(name);
    if (!col) {
      throw new InvalidFilterError(`Unknown filter field "${name}"`, name);
    }

    const { gt, gte, lt, lte } = comparison;
    const active = activeMask ?? this.activeDocs;
    const result = new DocumentBitset(this.capacity);

    if (col.type === 'number') {
      const hasGt = gt !== undefined;
      const hasGte = gte !== undefined;
      const hasLt = lt !== undefined;
      const hasLte = lte !== undefined;
      const gtVal = hasGt ? toNumberBound(gt, 'gt', name) : -Infinity;
      const gteVal = hasGte ? toNumberBound(gte, 'gte', name) : -Infinity;
      const ltVal = hasLt ? toNumberBound(lt, 'lt', name) : Infinity;
      const lteVal = hasLte ? toNumberBound(lte, 'lte', name) : Infinity;

      const activeWords = active.words;
      const presWords = col.presence.words;
      const numWords = Math.min(activeWords.length, presWords.length, result.words.length);
      const vals = col.values;

      if (hasGte && hasLte && !hasGt && !hasLt) {
        for (let w = 0; w < numWords; w++) {
          const act = activeWords[w] & presWords[w];
          if (act === 0) continue;
          const b = w << 5;
          if (act === 0xFFFFFFFF) {
            let word = 0;
            if (vals[b] >= gteVal && vals[b] <= lteVal) word |= (1 << 0);
            if (vals[b+1] >= gteVal && vals[b+1] <= lteVal) word |= (1 << 1);
            if (vals[b+2] >= gteVal && vals[b+2] <= lteVal) word |= (1 << 2);
            if (vals[b+3] >= gteVal && vals[b+3] <= lteVal) word |= (1 << 3);
            if (vals[b+4] >= gteVal && vals[b+4] <= lteVal) word |= (1 << 4);
            if (vals[b+5] >= gteVal && vals[b+5] <= lteVal) word |= (1 << 5);
            if (vals[b+6] >= gteVal && vals[b+6] <= lteVal) word |= (1 << 6);
            if (vals[b+7] >= gteVal && vals[b+7] <= lteVal) word |= (1 << 7);
            if (vals[b+8] >= gteVal && vals[b+8] <= lteVal) word |= (1 << 8);
            if (vals[b+9] >= gteVal && vals[b+9] <= lteVal) word |= (1 << 9);
            if (vals[b+10] >= gteVal && vals[b+10] <= lteVal) word |= (1 << 10);
            if (vals[b+11] >= gteVal && vals[b+11] <= lteVal) word |= (1 << 11);
            if (vals[b+12] >= gteVal && vals[b+12] <= lteVal) word |= (1 << 12);
            if (vals[b+13] >= gteVal && vals[b+13] <= lteVal) word |= (1 << 13);
            if (vals[b+14] >= gteVal && vals[b+14] <= lteVal) word |= (1 << 14);
            if (vals[b+15] >= gteVal && vals[b+15] <= lteVal) word |= (1 << 15);
            if (vals[b+16] >= gteVal && vals[b+16] <= lteVal) word |= (1 << 16);
            if (vals[b+17] >= gteVal && vals[b+17] <= lteVal) word |= (1 << 17);
            if (vals[b+18] >= gteVal && vals[b+18] <= lteVal) word |= (1 << 18);
            if (vals[b+19] >= gteVal && vals[b+19] <= lteVal) word |= (1 << 19);
            if (vals[b+20] >= gteVal && vals[b+20] <= lteVal) word |= (1 << 20);
            if (vals[b+21] >= gteVal && vals[b+21] <= lteVal) word |= (1 << 21);
            if (vals[b+22] >= gteVal && vals[b+22] <= lteVal) word |= (1 << 22);
            if (vals[b+23] >= gteVal && vals[b+23] <= lteVal) word |= (1 << 23);
            if (vals[b+24] >= gteVal && vals[b+24] <= lteVal) word |= (1 << 24);
            if (vals[b+25] >= gteVal && vals[b+25] <= lteVal) word |= (1 << 25);
            if (vals[b+26] >= gteVal && vals[b+26] <= lteVal) word |= (1 << 26);
            if (vals[b+27] >= gteVal && vals[b+27] <= lteVal) word |= (1 << 27);
            if (vals[b+28] >= gteVal && vals[b+28] <= lteVal) word |= (1 << 28);
            if (vals[b+29] >= gteVal && vals[b+29] <= lteVal) word |= (1 << 29);
            if (vals[b+30] >= gteVal && vals[b+30] <= lteVal) word |= (1 << 30);
            if (vals[b+31] >= gteVal && vals[b+31] <= lteVal) word |= (1 << 31);
            result.words[w] = word;
          } else {
            let mask = act;
            let word = 0;
            while (mask !== 0) {
              const t = mask & -mask;
              const bit = 31 - Math.clz32(t);
              const v = vals[b + bit];
              if (v >= gteVal && v <= lteVal) word |= t;
              mask ^= t;
            }
            result.words[w] = word;
          }
        }
      } else if (hasGte && !hasGt && !hasLt && !hasLte) {
        for (let w = 0; w < numWords; w++) {
          const act = activeWords[w] & presWords[w];
          if (act === 0) continue;
          const b = w << 5;
          if (act === 0xFFFFFFFF) {
            let word = 0;
            for (let bit = 0; bit < 32; bit++) {
              if (vals[b + bit] >= gteVal) word |= (1 << bit);
            }
            result.words[w] = word;
          } else {
            let mask = act;
            let word = 0;
            while (mask !== 0) {
              const t = mask & -mask;
              const bit = 31 - Math.clz32(t);
              if (vals[b + bit] >= gteVal) word |= t;
              mask ^= t;
            }
            result.words[w] = word;
          }
        }
      } else if (hasGt && !hasGte && !hasLt && !hasLte) {
        for (let w = 0; w < numWords; w++) {
          const act = activeWords[w] & presWords[w];
          if (act === 0) continue;
          const b = w << 5;
          if (act === 0xFFFFFFFF) {
            let word = 0;
            for (let bit = 0; bit < 32; bit++) {
              if (vals[b + bit] > gtVal) word |= (1 << bit);
            }
            result.words[w] = word;
          } else {
            let mask = act;
            let word = 0;
            while (mask !== 0) {
              const t = mask & -mask;
              const bit = 31 - Math.clz32(t);
              if (vals[b + bit] > gtVal) word |= t;
              mask ^= t;
            }
            result.words[w] = word;
          }
        }
      } else if (hasLte && !hasGt && !hasGte && !hasLt) {
        for (let w = 0; w < numWords; w++) {
          const act = activeWords[w] & presWords[w];
          if (act === 0) continue;
          const b = w << 5;
          if (act === 0xFFFFFFFF) {
            let word = 0;
            for (let bit = 0; bit < 32; bit++) {
              if (vals[b + bit] <= lteVal) word |= (1 << bit);
            }
            result.words[w] = word;
          } else {
            let mask = act;
            let word = 0;
            while (mask !== 0) {
              const t = mask & -mask;
              const bit = 31 - Math.clz32(t);
              if (vals[b + bit] <= lteVal) word |= t;
              mask ^= t;
            }
            result.words[w] = word;
          }
        }
      } else if (hasLt && !hasGt && !hasGte && !hasLte) {
        for (let w = 0; w < numWords; w++) {
          const act = activeWords[w] & presWords[w];
          if (act === 0) continue;
          const b = w << 5;
          if (act === 0xFFFFFFFF) {
            let word = 0;
            for (let bit = 0; bit < 32; bit++) {
              if (vals[b + bit] < ltVal) word |= (1 << bit);
            }
            result.words[w] = word;
          } else {
            let mask = act;
            let word = 0;
            while (mask !== 0) {
              const t = mask & -mask;
              const bit = 31 - Math.clz32(t);
              if (vals[b + bit] < ltVal) word |= t;
              mask ^= t;
            }
            result.words[w] = word;
          }
        }
      } else {
        for (let w = 0; w < numWords; w++) {
          let mask = activeWords[w] & presWords[w];
          if (mask === 0) continue;
          const base = w << 5;
          let outWord = 0;
          while (mask !== 0) {
            const t = mask & -mask;
            const bit = 31 - Math.clz32(t);
            const d = base + bit;
            const val = vals[d];
            if (
              (!hasGt || val > gtVal) &&
              (!hasGte || val >= gteVal) &&
              (!hasLt || val < ltVal) &&
              (!hasLte || val <= lteVal)
            ) {
              outWord |= t;
            }
            mask ^= t;
          }
          result.words[w] = outWord;
        }
      }
      return result;
    }

    if (col.type === 'string') {
      const gtStr = gt !== undefined ? String(gt) : undefined;
      const gteStr = gte !== undefined ? String(gte) : undefined;
      const ltStr = lt !== undefined ? String(lt) : undefined;
      const lteStr = lte !== undefined ? String(lte) : undefined;

      // Filter matching string codes
      for (let code = 0; code < col.stringTable.length; code++) {
        const str = col.stringTable[code];
        if (gtStr !== undefined && !(str > gtStr)) continue;
        if (gteStr !== undefined && !(str >= gteStr)) continue;
        if (ltStr !== undefined && !(str < ltStr)) continue;
        if (lteStr !== undefined && !(str <= lteStr)) continue;
        col.invertedIndex.get(code)?.orInto(result);
      }
      return result.andInPlace(active);
    }

    throw new InvalidFilterError(
      `Range comparisons (gt, gte, lt, lte) are only supported on number and string fields, got "${col.type}"`,
      name
    );
  }

  evaluateExists(name: string, exists: boolean, activeMask?: DocumentBitset): DocumentBitset {
    const col = this.columns.get(name);
    if (!col) {
      throw new InvalidFilterError(`Unknown filter field "${name}"`, name);
    }

    const active = activeMask ?? this.activeDocs;

    if (exists) {
      return col.presence.and(active);
    } else {
      return active.andNot(col.presence);
    }
  }

  getColumn(name: string): Column | undefined {
    return this.columns.get(name);
  }

  /**
   * Exact value distribution over candidate doc rows, specialized per column
   * type for hot-loop speed (string columns count by dictionary code with
   * zero per-doc allocation in the inner loop; numeric/boolean use primitive
   * counters; string[] counts per-doc, deduped within each doc).
   * Returns `undefined` for unknown fields. Missing values and inactive
   * (tombstoned) docs are skipped. Counts are per matching doc.
   */
  termsDistribution(
    name: string,
    candidates: Iterable<number>
  ): TermsFacetBucket[] | undefined {
    const col = this.columns.get(name);
    if (!col) return undefined;

    switch (col.type) {
      case 'string': {
        const tableLen = col.stringTable.length;
        const counts = new Uint32Array(Math.max(tableLen, 1));
        for (const d of candidates) {
          if (!Number.isInteger(d) || d < 0 || d >= this.capacity) continue;
          if (!this.activeDocs.has(d)) continue;
          if (!col.presence.has(d)) continue;
          const code = col.codes[d];
          if (code < counts.length) counts[code]++;
        }
        const out: TermsFacetBucket[] = [];
        for (let c = 0; c < tableLen; c++) {
          if (counts[c] > 0) out.push({ value: col.stringTable[c], count: counts[c] });
        }
        return out;
      }
      case 'number': {
        const map = new Map<number, number>();
        for (const d of candidates) {
          if (!Number.isInteger(d) || d < 0 || d >= this.capacity) continue;
          if (!this.activeDocs.has(d)) continue;
          if (!col.presence.has(d)) continue;
          const v = col.values[d];
          if (!Number.isFinite(v)) continue;
          map.set(v, (map.get(v) ?? 0) + 1);
        }
        const out: TermsFacetBucket[] = [];
        for (const [value, count] of map) out.push({ value, count });
        return out;
      }
      case 'boolean': {
        let trueCount = 0;
        let falseCount = 0;
        for (const d of candidates) {
          if (!Number.isInteger(d) || d < 0 || d >= this.capacity) continue;
          if (!this.activeDocs.has(d)) continue;
          if (!col.presence.has(d)) continue;
          if (col.trueBitset.has(d)) trueCount++;
          else falseCount++;
        }
        const out: TermsFacetBucket[] = [];
        if (trueCount > 0) out.push({ value: true, count: trueCount });
        if (falseCount > 0) out.push({ value: false, count: falseCount });
        return out;
      }
      case 'string[]': {
        const map = new Map<string, number>();
        for (const d of candidates) {
          if (!Number.isInteger(d) || d < 0 || d >= this.capacity) continue;
          if (!this.activeDocs.has(d)) continue;
          const tags = col.docTags.get(d);
          if (!tags) continue;
          if (tags.length <= 1) {
            const t = tags[0];
            if (t !== undefined) map.set(t, (map.get(t) ?? 0) + 1);
          } else {
            const seen = new Set<string>();
            for (let i = 0; i < tags.length; i++) {
              const t = tags[i];
              if (seen.has(t)) continue;
              seen.add(t);
              map.set(t, (map.get(t) ?? 0) + 1);
            }
          }
        }
        const out: TermsFacetBucket[] = [];
        for (const [value, count] of map) out.push({ value, count });
        return out;
      }
    }
  }

  /**
   * Numeric bucket counts over candidate doc rows for half-open intervals
   * `[from, to)`. O(candidates x ranges); overlapping ranges double-count by
   * design (independent buckets; sum may exceed total matches).
   * Returns `undefined` for unknown or non-number fields. Inactive
   * (tombstoned) docs are skipped.
   */
  rangeDistribution(
    name: string,
    candidates: Iterable<number>,
    ranges: Array<{ from?: number; to?: number }>
  ): number[] | undefined {
    const col = this.columns.get(name);
    if (!col || col.type !== 'number') return undefined;
    const counts = new Array<number>(ranges.length).fill(0);
    const froms = ranges.map((r) => r.from);
    const tos = ranges.map((r) => r.to);
    for (const d of candidates) {
      if (!Number.isInteger(d) || d < 0 || d >= this.capacity) continue;
      if (!this.activeDocs.has(d)) continue;
      if (!col.presence.has(d)) continue;
      const v = col.values[d];
      if (!Number.isFinite(v)) continue;
      for (let b = 0; b < ranges.length; b++) {
        const from = froms[b];
        const to = tos[b];
        if ((from === undefined || v >= from) && (to === undefined || v < to)) {
          counts[b]++;
        }
      }
    }
    return counts;
  }
}
