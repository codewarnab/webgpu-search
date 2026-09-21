/**
 * Facet aggregation engine for categorical terms and numeric range buckets.
 * Aggregates over explicit query-matched candidate doc indices with exact vs.
 * approximate precision semantics flagged per response.
 * 100% portable across browser main thread, Web Workers, Node.js, and SSR
 * (zero DOM references, zero runtime dependencies).
 */

import type { ColumnarStore } from '../filter/columnar-store';
import type {
  FacetRequest,
  FacetResult,
  FilterExpression,
  RangeFacetBucket,
  RangeFacetBucketResult,
  RangeFacetRequest,
  RangeFacetResult,
  TermsFacetBucket,
  TermsFacetRequest,
  TermsFacetResult,
} from '../types';
import { InvalidFilterError } from '../errors';

/** A validated facet request paired with its response key. */
export interface NormalizedFacet {
  /** Response key: record key for object form, field name for array form. */
  name: string;
  request: FacetRequest;
}

/** Default terms bucket limit when `limit` is omitted. */
export const DEFAULT_TERMS_LIMIT = 10;

/** Fail-closed DoS caps: facet count and range buckets per request. */
export const MAX_FACET_REQUESTS = 32;
export const MAX_RANGE_BUCKETS = 100;

/** Keys that would mutate object prototype via plain assignment. */
function isUnsafeKey(key: string): boolean {
  return key === '__proto__' || key === 'constructor' || key === 'prototype';
}

/** Prototype-safe assignment for attacker-influenced keys. */
function safeSet(out: Record<string, unknown>, key: string, value: unknown): void {
  if (key === '__proto__') {
    Object.defineProperty(out, key, { value, enumerable: true, configurable: true, writable: true });
  } else {
    out[key] = value;
  }
}

function assertRecord(value: unknown, what: string): asserts value is Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`[webgpu-search] Invalid facets: expected ${what}, got ${Array.isArray(value) ? 'array' : value === null ? 'null' : typeof value}.`);
  }
}

function validateTermsRequest(raw: unknown): TermsFacetRequest {
  assertRecord(raw, 'TermsFacetRequest object');
  const fieldRaw = (raw as { field?: unknown }).field;
  if (typeof fieldRaw !== 'string' || fieldRaw.trim().length === 0) {
    throw new TypeError('[webgpu-search] Invalid terms facet: "field" must be a non-empty string.');
  }
  // Trim whitespace-padded fields so " category " fails closed at lookup
  // with a clear unknown-field error instead of silent mismatch.
  const field = fieldRaw.trim();
  const out: TermsFacetRequest = { type: 'terms', field };
  if ((raw as { limit?: unknown }).limit !== undefined) {
    const limit = (raw as { limit: unknown }).limit;
    if (typeof limit !== 'number' || !Number.isFinite(limit)) {
      throw new TypeError(`[webgpu-search] Invalid terms facet on field "${field}": "limit" must be a finite number.`);
    }
    // Fractional limits are floored (2.9 -> 2); documented on TermsFacetRequest.
    const floored = Math.floor(limit);
    if (floored < 1) {
      throw new RangeError(`[webgpu-search] Invalid terms facet on field "${field}": "limit" must be >= 1, got ${String(limit)}.`);
    }
    out.limit = floored;
  }
  if ((raw as { sortBy?: unknown }).sortBy !== undefined) {
    const sortBy = (raw as { sortBy: unknown }).sortBy;
    if (sortBy !== 'count' && sortBy !== 'value') {
      throw new TypeError(`[webgpu-search] Invalid terms facet on field "${field}": "sortBy" must be 'count' or 'value'.`);
    }
    out.sortBy = sortBy;
  }
  return out;
}

function validateRangeBucket(raw: unknown, field: string, index: number): RangeFacetBucket {
  assertRecord(raw, `range bucket at index ${index} for field "${field}"`);
  const rec = raw as { from?: unknown; to?: unknown; key?: unknown };
  const out: RangeFacetBucket = {};
  if (rec.from !== undefined) {
    if (typeof rec.from !== 'number' || !Number.isFinite(rec.from)) {
      throw new TypeError(`[webgpu-search] Invalid range facet on field "${field}": bucket ${index} "from" must be a finite number.`);
    }
    out.from = rec.from;
  }
  if (rec.to !== undefined) {
    if (typeof rec.to !== 'number' || !Number.isFinite(rec.to)) {
      throw new TypeError(`[webgpu-search] Invalid range facet on field "${field}": bucket ${index} "to" must be a finite number.`);
    }
    out.to = rec.to;
  }
  if (rec.key !== undefined) {
    if (typeof rec.key !== 'string' || rec.key.length === 0) {
      throw new TypeError(`[webgpu-search] Invalid range facet on field "${field}": bucket ${index} "key" must be a non-empty string.`);
    }
    out.key = rec.key;
  }
  if (out.from === undefined && out.to === undefined) {
    throw new TypeError(`[webgpu-search] Invalid range facet on field "${field}": bucket ${index} must define at least one of "from" or "to".`);
  }
  if (out.from !== undefined && out.to !== undefined && out.from >= out.to) {
    throw new RangeError(`[webgpu-search] Invalid range facet on field "${field}": bucket ${index} has inverted bounds (from ${out.from} >= to ${out.to}). Half-open [from, to) requires from < to.`);
  }
  return out;
}

function validateRangeRequest(raw: unknown): RangeFacetRequest {
  assertRecord(raw, 'RangeFacetRequest object');
  const fieldRaw = (raw as { field?: unknown }).field;
  if (typeof fieldRaw !== 'string' || fieldRaw.trim().length === 0) {
    throw new TypeError('[webgpu-search] Invalid range facet: "field" must be a non-empty string.');
  }
  const field = fieldRaw.trim();
  const ranges = (raw as { ranges?: unknown }).ranges;
  if (!Array.isArray(ranges) || ranges.length === 0) {
    throw new TypeError(`[webgpu-search] Invalid range facet on field "${field}": "ranges" must be a non-empty array.`);
  }
  if (ranges.length > MAX_RANGE_BUCKETS) {
    throw new RangeError(`[webgpu-search] Invalid range facet on field "${field}": "ranges" length ${ranges.length} exceeds cap ${MAX_RANGE_BUCKETS}.`);
  }
  return { type: 'range', field, ranges: ranges.map((r, i) => validateRangeBucket(r, field, i)) };
}

function validateFacetRequest(raw: unknown): FacetRequest {
  assertRecord(raw, 'FacetRequest object');
  const type = (raw as { type?: unknown }).type;
  if (type === 'terms') return validateTermsRequest(raw);
  if (type === 'range') return validateRangeRequest(raw);
  throw new TypeError(`[webgpu-search] Invalid facet request: "type" must be 'terms' or 'range', got ${String(type)}.`);
}

/**
 * Normalizes the `facets` search option into an ordered list of named requests.
 * - Object form: response keys are the record keys.
 * - Array form: response keys are the facet field names (later duplicates win,
 *   winner moves to end so both value and order reflect last occurrence).
 * Returns `undefined` when no facets were requested. Throws TypeError on
 * malformed shapes and RangeError when facet count exceeds cap.
 * Unknown extra keys on facet requests are stripped (allowlist rebuild).
 * Field names are trimmed. Facet names `__proto__`/`constructor`/`prototype`
 * are rejected fail-closed to prevent prototype pollution.
 * Unknown-field errors surface as InvalidFilterError up front in
 * `validateRequests` (fail-closed before empty-match early exits).
 */
export function normalizeFacetRequests(
  facets: unknown
): NormalizedFacet[] | undefined {
  if (facets === undefined) return undefined;
  if (Array.isArray(facets)) {
    if (facets.length > MAX_FACET_REQUESTS) {
      throw new RangeError(`[webgpu-search] Invalid facets: count ${facets.length} exceeds cap ${MAX_FACET_REQUESTS}.`);
    }
    // Map preserves insertion order; delete+set moves later duplicates to end.
    const map = new Map<string, NormalizedFacet>();
    for (let i = 0; i < facets.length; i++) {
      const request = validateFacetRequest(facets[i]);
      const name = request.field;
      if (isUnsafeKey(name)) {
        throw new TypeError(`[webgpu-search] Invalid facet field "${name}": reserved key.`);
      }
      map.delete(name);
      map.set(name, { name, request });
    }
    return Array.from(map.values());
  }
  assertRecord(facets, 'Record<string, FacetRequest> or FacetRequest[]');
  const keys = Object.keys(facets);
  if (keys.length > MAX_FACET_REQUESTS) {
    throw new RangeError(`[webgpu-search] Invalid facets: count ${keys.length} exceeds cap ${MAX_FACET_REQUESTS}.`);
  }
  const out: NormalizedFacet[] = [];
  for (const name of keys) {
    if (isUnsafeKey(name)) {
      throw new TypeError(`[webgpu-search] Invalid facet name "${name}": reserved key.`);
    }
    if (name.trim().length === 0) {
      throw new TypeError('[webgpu-search] Invalid facet name: must be a non-empty string.');
    }
    out.push({ name, request: validateFacetRequest((facets as Record<string, unknown>)[name]) });
  }
  return out;
}

/**
 * Deep check whether a filter expression references a field anywhere
 * (including inside `or` / `not` branches).
 * Note: fields literally named "and"/"or"/"not" are ambiguous with the DSL
 * operators and cannot be reliably detected (pre-existing DSL limitation
 * shared with filter-evaluator; such field names are unsupported for facets).
 */
export function filterReferencesField(expression: FilterExpression, field: string): boolean {
  if (expression === null || typeof expression !== 'object' || Array.isArray(expression)) return false;
  const keys = Object.keys(expression);
  for (let k = 0; k < keys.length; k++) {
    const key = keys[k];
    const val = (expression as unknown as Record<string, unknown>)[key];
    if (key === 'and' || key === 'or') {
      if (!Array.isArray(val)) continue;
      for (let i = 0; i < val.length; i++) {
        if (filterReferencesField(val[i] as FilterExpression, field)) return true;
      }
    } else if (key === 'not') {
      if (val !== null && typeof val === 'object' && !Array.isArray(val)) {
        if (filterReferencesField(val as FilterExpression, field)) return true;
      }
    } else if (key === field) {
      return true;
    }
  }
  return false;
}

/**
 * Returns a shallow copy of the filter expression with all conjunction-level
 * conditions on `field` removed (disjunctive facet exclusion). Returns
 * `undefined` when nothing remains (match-all). Conservative inside
 * disjunctions and negations: an `or`/`not` branch that references the
 * field is kept verbatim (aliased, not cloned, for speed) so exclusion can
 * only widen conjunctions and never unsoundly expand an `or` or invert a
 * `not`. Overlapping range buckets double-count by design (independent
 * buckets; sum may exceed total matches).
 */
export function excludeFieldFromFilter(
  expression: unknown,
  field: string
): FilterExpression | undefined {
  if (expression === null || typeof expression !== 'object' || Array.isArray(expression)) {
    throw new TypeError('[webgpu-search] Invalid filter expression for facet exclusion: expected object.');
  }
  const keys = Object.keys(expression);
  const out: Record<string, unknown> = {};
  for (let k = 0; k < keys.length; k++) {
    const key = keys[k];
    const val = (expression as unknown as Record<string, unknown>)[key];
    if (key === field) {
      continue;
    }
    if (key === 'and' && Array.isArray(val)) {
      const kept: FilterExpression[] = [];
      for (let i = 0; i < val.length; i++) {
        const sub = excludeFieldFromFilter(val[i] as FilterExpression, field);
        if (sub !== undefined) kept.push(sub);
      }
      if (kept.length === 0) continue;
      safeSet(out, key, kept);
    } else if (key === 'or' || key === 'not') {
      safeSet(out, key, val);
    } else {
      safeSet(out, key, val);
    }
  }
  return Object.keys(out).length === 0 ? undefined : (out as unknown as FilterExpression);
}

/**
 * Categorical + numeric facet aggregator over explicit candidate doc indices.
 * All methods are synchronous typed-array scans; missing (null/absent)
 * values are skipped rather than bucketed.
 */
export class FacetEngine<TDoc = Record<string, unknown>> {
  constructor(private readonly store: ColumnarStore<TDoc>) {}

  private requireField(field: string): void {
    if (!this.store.hasField(field)) {
      const available = this.store.getFieldNames();
      const desc = available.length > 0
        ? `Configured fields: ${available.map((f) => `"${f}"`).join(', ')}.`
        : 'No filterFields were configured on this index.';
      throw new InvalidFilterError(`Unknown facet field "${field}". ${desc}`, field);
    }
  }

  private requireNumberField(field: string): void {
    const fieldType = this.store.getFieldType(field);
    if (fieldType !== 'number') {
      throw new InvalidFilterError(
        `Range facets require a number field, got "${String(fieldType)}" on field "${field}"`,
        field
      );
    }
  }

  /**
   * Fail-closed field validation for a normalized facet list: unknown fields
   * and range-on-non-number requests throw `InvalidFilterError` up front, so
   * even empty-match early exits cannot swallow malformed requests.
   */
  validateRequests(facets: NormalizedFacet[]): void {
    for (const facet of facets) {
      this.requireField(facet.request.field);
      if (facet.request.type === 'range') {
        this.requireNumberField(facet.request.field);
      }
    }
  }

  /** Terms aggregation over candidate doc indices. */
  aggregateTerms(
    field: string,
    candidates: Iterable<number>,
    options: { limit?: number; sortBy?: 'count' | 'value'; isApproximate?: boolean } = {}
  ): TermsFacetResult {
    this.requireField(field);
    let limit = options.limit ?? DEFAULT_TERMS_LIMIT;
    if (typeof limit !== 'number' || !Number.isFinite(limit)) {
      throw new TypeError(`[webgpu-search] Invalid terms facet on field "${field}": "limit" must be a finite number.`);
    }
    limit = Math.floor(limit);
    if (limit < 1) {
      throw new RangeError(`[webgpu-search] Invalid terms facet on field "${field}": "limit" must be >= 1.`);
    }
    const sortBy = options.sortBy ?? 'count';
    if (sortBy !== 'count' && sortBy !== 'value') {
      throw new TypeError(`[webgpu-search] Invalid terms facet on field "${field}": "sortBy" must be 'count' or 'value'.`);
    }
    // Type-specialized counting lives in the store (dictionary-code fast path).
    const buckets: TermsFacetBucket[] = this.store.termsDistribution(field, candidates) ?? [];
    // Precompute string keys once so high-cardinality sorts avoid per-compare
    // String() allocations (O(U) instead of O(U log U) coercions).
    if (sortBy === 'value') {
      const decorated = buckets.map((b) => ({
        b,
        isNum: typeof b.value === 'number',
        num: typeof b.value === 'number' ? (b.value as number) : 0,
        s: typeof b.value === 'number' ? '' : String(b.value),
        t: typeof b.value,
      }));
      decorated.sort((x, y) => {
        if (x.isNum && y.isNum) {
          if (x.num < y.num) return -1;
          if (x.num > y.num) return 1;
          return 0;
        }
        if (x.isNum !== y.isNum) {
          // Numbers vs non-numbers: fall back to stringified compare for
          // parity with compareFacetValues (numeric-aware only for num-num).
          const sx = String(x.b.value);
          const sy = String(y.b.value);
          if (sx < sy) return -1;
          if (sx > sy) return 1;
        } else if (!x.isNum) {
          if (x.s < y.s) return -1;
          if (x.s > y.s) return 1;
        }
        if (x.t < y.t) return -1;
        if (x.t > y.t) return 1;
        return 0;
      });
      for (let i = 0; i < buckets.length; i++) buckets[i] = decorated[i].b;
    } else {
      const decorated = buckets.map((b) => ({
        b,
        s: typeof b.value === 'number' ? '' : String(b.value),
        t: typeof b.value,
      }));
      decorated.sort((x, y) => {
        if (y.b.count !== x.b.count) return y.b.count - x.b.count;
        // Tiebreak identical to compareFacetValues but with cached strings.
        if (typeof x.b.value === 'number' && typeof y.b.value === 'number') {
          const a = x.b.value as number;
          const c = y.b.value as number;
          if (a < c) return -1;
          if (a > c) return 1;
          return 0;
        }
        const sx = typeof x.b.value === 'number' ? String(x.b.value) : x.s;
        const sy = typeof y.b.value === 'number' ? String(y.b.value) : y.s;
        if (sx < sy) return -1;
        if (sx > sy) return 1;
        if (x.t < y.t) return -1;
        if (x.t > y.t) return 1;
        return 0;
      });
      for (let i = 0; i < buckets.length; i++) buckets[i] = decorated[i].b;
    }
    return {
      type: 'terms',
      field,
      isApproximate: options.isApproximate ?? false,
      buckets: buckets.length > limit ? buckets.slice(0, limit) : buckets,
    };
  }

  /** Numeric range aggregation over candidate doc indices (half-open [from, to)). */
  aggregateRange(
    field: string,
    candidates: Iterable<number>,
    ranges: RangeFacetBucket[],
    options: { isApproximate?: boolean } = {}
  ): RangeFacetResult {
    this.requireField(field);
    this.requireNumberField(field);
    if (!Array.isArray(ranges) || ranges.length === 0) {
      throw new TypeError(`[webgpu-search] Invalid range facet on field "${field}": "ranges" must be a non-empty array.`);
    }
    if (ranges.length > MAX_RANGE_BUCKETS) {
      throw new RangeError(`[webgpu-search] Invalid range facet on field "${field}": "ranges" length ${ranges.length} exceeds cap ${MAX_RANGE_BUCKETS}.`);
    }
    for (let i = 0; i < ranges.length; i++) {
      const r = ranges[i];
      if (r === null || typeof r !== 'object' || Array.isArray(r)) {
        throw new TypeError(`[webgpu-search] Invalid range facet on field "${field}": bucket ${i} must be an object.`);
      }
      if (r.from !== undefined && (typeof r.from !== 'number' || !Number.isFinite(r.from))) {
        throw new TypeError(`[webgpu-search] Invalid range facet on field "${field}": bucket ${i} "from" must be a finite number.`);
      }
      if (r.to !== undefined && (typeof r.to !== 'number' || !Number.isFinite(r.to))) {
        throw new TypeError(`[webgpu-search] Invalid range facet on field "${field}": bucket ${i} "to" must be a finite number.`);
      }
      if (r.from === undefined && r.to === undefined) {
        throw new TypeError(`[webgpu-search] Invalid range facet on field "${field}": bucket ${i} must define at least one of "from" or "to".`);
      }
      if (r.from !== undefined && r.to !== undefined && r.from >= r.to) {
        throw new RangeError(`[webgpu-search] Invalid range facet on field "${field}": bucket ${i} has inverted bounds (from ${r.from} >= to ${r.to}).`);
      }
    }
    const counts = this.store.rangeDistribution(field, candidates, ranges) ?? ranges.map(() => 0);
    const buckets: RangeFacetBucketResult[] = ranges.map((r, i) => ({
      key: r.key ?? `${r.from ?? '*'}-${r.to ?? '*'}`,
      ...(r.from !== undefined ? { from: r.from } : {}),
      ...(r.to !== undefined ? { to: r.to } : {}),
      count: counts[i] ?? 0,
    }));
    return { type: 'range', field, isApproximate: options.isApproximate ?? false, buckets };
  }

  /** Aggregates one normalized facet over candidate doc indices. */
  aggregateOne(
    facet: NormalizedFacet,
    candidates: Iterable<number>,
    isApproximate: boolean
  ): FacetResult {
    if (facet.request.type === 'terms') {
      return this.aggregateTerms(facet.request.field, candidates, {
        limit: facet.request.limit,
        sortBy: facet.request.sortBy,
        isApproximate,
      });
    }
    return this.aggregateRange(facet.request.field, candidates, facet.request.ranges, { isApproximate });
  }

  /**
   * Empty (zero-count) results for a facet list. Range buckets are emitted
   * with count 0 so response shape stays stable on empty match sets.
   */
  emptyResults(facets: NormalizedFacet[]): Record<string, FacetResult> {
    const out: Record<string, FacetResult> = {};
    for (const facet of facets) {
      let result: FacetResult;
      if (facet.request.type === 'terms') {
        result = { type: 'terms', field: facet.request.field, isApproximate: false, buckets: [] };
      } else {
        result = {
          type: 'range',
          field: facet.request.field,
          isApproximate: false,
          buckets: facet.request.ranges.map((r) => ({
            key: r.key ?? `${r.from ?? '*'}-${r.to ?? '*'}`,
            ...(r.from !== undefined ? { from: r.from } : {}),
            ...(r.to !== undefined ? { to: r.to } : {}),
            count: 0,
          })),
        };
      }
      safeSet(out as Record<string, unknown>, facet.name, result);
    }
    return out;
  }
}
