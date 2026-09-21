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
  FilterValue,
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

function assertRecord(value: unknown, what: string): asserts value is Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    throw new TypeError(`[webgpu-search] Invalid facets: expected ${what}, got ${Array.isArray(value) ? 'array' : value === null ? 'null' : typeof value}.`);
  }
}

function validateTermsRequest(raw: unknown): TermsFacetRequest {
  assertRecord(raw, 'TermsFacetRequest object');
  const field = (raw as { field?: unknown }).field;
  if (typeof field !== 'string' || field.trim().length === 0) {
    throw new TypeError('[webgpu-search] Invalid terms facet: "field" must be a non-empty string.');
  }
  const out: TermsFacetRequest = { type: 'terms', field };
  if ((raw as { limit?: unknown }).limit !== undefined) {
    const limit = (raw as { limit: unknown }).limit;
    if (typeof limit !== 'number' || !Number.isFinite(limit)) {
      throw new TypeError(`[webgpu-search] Invalid terms facet on field "${field}": "limit" must be a finite number.`);
    }
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
  return out;
}

function validateRangeRequest(raw: unknown): RangeFacetRequest {
  assertRecord(raw, 'RangeFacetRequest object');
  const field = (raw as { field?: unknown }).field;
  if (typeof field !== 'string' || field.trim().length === 0) {
    throw new TypeError('[webgpu-search] Invalid range facet: "field" must be a non-empty string.');
  }
  const ranges = (raw as { ranges?: unknown }).ranges;
  if (!Array.isArray(ranges) || ranges.length === 0) {
    throw new TypeError(`[webgpu-search] Invalid range facet on field "${field}": "ranges" must be a non-empty array.`);
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
 * - Array form: response keys are the facet field names (later duplicates win).
 * Returns `undefined` when no facets were requested. Throws TypeError on
 * malformed shapes; unknown-field errors surface as InvalidFilterError during
 * aggregation (fail-closed at evaluation time).
 */
export function normalizeFacetRequests(
  facets: Record<string, FacetRequest> | FacetRequest[] | undefined
): NormalizedFacet[] | undefined {
  if (facets === undefined) return undefined;
  if (Array.isArray(facets)) {
    const out: NormalizedFacet[] = [];
    const seen = new Map<string, number>();
    for (let i = 0; i < facets.length; i++) {
      const request = validateFacetRequest(facets[i]);
      const name = request.field;
      const prev = seen.get(name);
      if (prev !== undefined) {
        out[prev] = { name, request };
      } else {
        seen.set(name, out.length);
        out.push({ name, request });
      }
    }
    return out;
  }
  assertRecord(facets, 'Record<string, FacetRequest> or FacetRequest[]');
  const out: NormalizedFacet[] = [];
  for (const name of Object.keys(facets)) {
    out.push({ name, request: validateFacetRequest((facets as Record<string, FacetRequest>)[name]) });
  }
  return out;
}

/**
 * Deep check whether a filter expression references a field anywhere
 * (including inside `or` / `not` branches).
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
 * Returns a copy of the filter expression with all conjunction-level
 * conditions on `field` removed (disjunctive facet exclusion). Returns
 * `undefined` when nothing remains (match-all). Conservative inside
 * disjunctions and negations: an `or`/`not` branch that references the
 * field is kept verbatim so exclusion can only widen conjunctions and
 * never unsoundly expand an `or` or invert a `not`.
 */
export function excludeFieldFromFilter(
  expression: FilterExpression,
  field: string
): FilterExpression | undefined {
  if (expression === null || typeof expression !== 'object' || Array.isArray(expression)) return expression;
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
      out[key] = kept;
    } else if (key === 'or' || key === 'not') {
      out[key] = val;
    } else {
      out[key] = val;
    }
  }
  return Object.keys(out).length === 0 ? undefined : (out as unknown as FilterExpression);
}

/** Sort key for deterministic bucket ordering. Numbers compare numerically;
 * all other pairs compare by stringified value (with a typeof tiebreak so
 * `1` vs `'1'` vs `true` stay distinct and stable). Locale-free, so the
 * order is bit-identical across browsers, workers, and Node.js. */
function compareFacetValues(a: FilterValue, b: FilterValue): number {
  if (typeof a === 'number' && typeof b === 'number') {
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
  }
  const sa = String(a);
  const sb = String(b);
  if (sa < sb) return -1;
  if (sa > sb) return 1;
  // Distinguish 1 vs '1' vs true deterministically when stringified equal.
  const ta = typeof a;
  const tb = typeof b;
  if (ta < tb) return -1;
  if (ta > tb) return 1;
  return 0;
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
    const limit = options.limit ?? DEFAULT_TERMS_LIMIT;
    const sortBy = options.sortBy ?? 'count';
    // Type-specialized counting lives in the store (dictionary-code fast path).
    const buckets: TermsFacetBucket[] = this.store.termsDistribution(field, candidates) ?? [];
    if (sortBy === 'value') {
      buckets.sort((a, b) => compareFacetValues(a.value, b.value));
    } else {
      buckets.sort((a, b) => {
        if (b.count !== a.count) return b.count - a.count;
        return compareFacetValues(a.value, b.value);
      });
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
      if (facet.request.type === 'terms') {
        out[facet.name] = { type: 'terms', field: facet.request.field, isApproximate: false, buckets: [] };
      } else {
        out[facet.name] = {
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
    }
    return out;
  }
}
