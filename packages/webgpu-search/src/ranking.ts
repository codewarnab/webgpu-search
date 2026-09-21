/**
 * Deterministic ranking.
 *
 * Multi-tier tie-breaking applied identically on every execution path
 * (WebGPU readback, exact CPU, legacy ufuzzy CPU, autocomplete candidates):
 * 1. `score` DESC — primary integer match score.
 * 2. `weight` DESC — matches in higher-weighted fields take precedence.
 * 3. `exact` DESC — full post-normalization token equality precedes partial/fuzzy.
 * 4. `length` ASC — shorter matched-field token spans precede longer ones.
 * 5. `id` ASC — stable string/numeric document ID breaks remaining ties.
 *
 * Search paths rank per document (best-field-wins); autocomplete ranks per
 * (doc, field) row with the same comparator, so a document may appear
 * multiple times in suggestions. Custom hierarchies are honored on both
 * surfaces (see `AutocompleteOptions.tieBreakers`).
 *
 * All comparisons are wrap-free (`>` / `<`, never `|0` subtraction) and use
 * code-unit order for strings (never locale collation) so ordering is
 * bit-for-bit identical across browsers, workers, and Node.js.
 *
 * Fail-closed on non-finite keys: NaN/Infinity scores, weights, lengths, or
 * numeric IDs throw `TypeError` instead of producing implementation-defined
 * `Array.sort` order. Field weights are validated positive-finite at index
 * construction, snapshot restore, and deterministic exact entry points.
 *
 * Portable: no DOM refs. Operates on plain numbers/strings only.
 */

import type { TieBreakerCriterion } from './types';

/** Default hierarchy: score > weight > exact > length > id. */
export const DEFAULT_TIE_BREAKERS: readonly TieBreakerCriterion[] = Object.freeze([
  'score',
  'weight',
  'exact',
  'length',
  'id',
] as const) as readonly TieBreakerCriterion[];

/**
 * Rankable candidate keys for the deterministic comparator.
 * - `fieldWeight`: weight of the matched (primary) field.
 * - `isExactMatch`: post-normalization full-string equality with the query.
 * - `matchedLength`: post-normalization token count of the matched field/record.
 * - `id`: unique document ID (string or finite number).
 * - `docIndex`: insertion-order fallback when IDs compare equal
 * (defensive; IDs are unique by construction).
 */
export interface RankableCandidate {
  score: number;
  fieldWeight: number;
  isExactMatch: boolean;
  matchedLength: number;
  id: string | number;
  docIndex: number;
}

/**
 * Validate and normalize a custom tie-breaker hierarchy (fail-closed).
 * - `undefined` resolves to the default hierarchy.
 * - Must be a non-empty array of known criteria with no duplicates.
 */
export function normalizeTieBreakers(
  raw: TieBreakerCriterion[] | undefined
): TieBreakerCriterion[] {
  if (raw === undefined) return [...DEFAULT_TIE_BREAKERS];
  if (!Array.isArray(raw)) {
    throw new TypeError('[webgpu-search] ranking.tieBreakers must be an array.');
  }
  if (raw.length === 0) {
    throw new RangeError('[webgpu-search] ranking.tieBreakers must not be empty.');
  }
  const seen = new Set<string>();
  for (let i = 0; i < raw.length; i++) {
    const c = raw[i];
    if (c !== 'score' && c !== 'weight' && c !== 'exact' && c !== 'length' && c !== 'id') {
      throw new TypeError(
        `[webgpu-search] Unknown tie-breaker criterion '${String(c)}'. Expected one of 'score', 'weight', 'exact', 'length', 'id'.`
      );
    }
    if (seen.has(c)) {
      throw new RangeError(`[webgpu-search] Duplicate tie-breaker criterion '${c}'.`);
    }
    seen.add(c);
  }
  return raw.slice();
}

/** Stable code-unit string comparison (never locale-aware). */
function compareStringsAsc(a: string, b: string): number {
  if (a === b) return 0;
  return a < b ? -1 : 1;
}

/**
 * Deterministic ID comparison:
 * - number vs number: numeric ascending (both must be finite; NaN/Infinity
 * throw fail-closed to preserve the total-order contract).
 * - otherwise: String(id) code-unit ascending (covers string/string and
 * mixed string/number pairs deterministically; note numeric `2` and string
 * `"2"` compare id-equal and fall through to `docIndex`).
 */
export function compareIdsAsc(a: string | number, b: string | number): number {
  if (typeof a === 'number' && typeof b === 'number') {
    if (!Number.isFinite(a) || !Number.isFinite(b)) {
      throw new TypeError(
        `[webgpu-search] compareIdsAsc requires finite numeric ids, got ${String(a)} vs ${String(b)}.`
      );
    }
    if (a !== b) return a > b ? 1 : -1;
    return 0;
  }
  if (typeof a === 'number' && !Number.isFinite(a)) {
    throw new TypeError(`[webgpu-search] compareIdsAsc requires finite numeric id, got ${String(a)}.`);
  }
  if (typeof b === 'number' && !Number.isFinite(b)) {
    throw new TypeError(`[webgpu-search] compareIdsAsc requires finite numeric id, got ${String(b)}.`);
  }
  return compareStringsAsc(String(a), String(b));
}

/** Throw fail-closed on non-finite ranking keys (preserves total order). */
function assertFiniteRankKey(value: number, key: string): void {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(
      `[webgpu-search] compareRanked requires a finite number for '${key}', got ${String(value)}.`
    );
  }
}

/**
 * Total-order comparator over the requested tie-breaker hierarchy.
 * Criteria not listed are ignored; `docIndex` ascending is the implicit
 * final fallback so the order is total even for duplicate IDs.
 * Non-finite `score`/`fieldWeight`/`matchedLength` throw fail-closed.
 */
export function compareRanked(
  a: RankableCandidate,
  b: RankableCandidate,
  tieBreakers: readonly TieBreakerCriterion[]
): number {
  for (let i = 0; i < tieBreakers.length; i++) {
    const c = tieBreakers[i];
    if (c === 'score') {
      assertFiniteRankKey(a.score, 'score');
      assertFiniteRankKey(b.score, 'score');
      if (b.score !== a.score) return b.score > a.score ? 1 : -1;
    } else if (c === 'weight') {
      assertFiniteRankKey(a.fieldWeight, 'fieldWeight');
      assertFiniteRankKey(b.fieldWeight, 'fieldWeight');
      if (b.fieldWeight !== a.fieldWeight) return b.fieldWeight > a.fieldWeight ? 1 : -1;
    } else if (c === 'exact') {
      const ea = a.isExactMatch ? 1 : 0;
      const eb = b.isExactMatch ? 1 : 0;
      if (eb !== ea) return eb > ea ? 1 : -1;
    } else if (c === 'length') {
      assertFiniteRankKey(a.matchedLength, 'matchedLength');
      assertFiniteRankKey(b.matchedLength, 'matchedLength');
      if (a.matchedLength !== b.matchedLength) return a.matchedLength > b.matchedLength ? 1 : -1;
    } else if (c === 'id') {
      const idCmp = compareIdsAsc(a.id, b.id);
      if (idCmp !== 0) return idCmp;
    }
  }
  if (a.docIndex !== b.docIndex) return a.docIndex > b.docIndex ? 1 : -1;
  return 0;
}

/**
 * Sort candidates in place with the deterministic comparator.
 * Returns the same array for chaining.
 */
export function sortRanked<T extends RankableCandidate>(
  items: T[],
  tieBreakers: readonly TieBreakerCriterion[]
): T[] {
  items.sort((a, b) => compareRanked(a, b, tieBreakers));
  return items;
}

/** True when two post-normalization token streams are exactly equal. */
export function isExactTokenMatch(
  fieldTokens: Uint32Array | readonly number[],
  queryTokens: Uint32Array | readonly number[]
): boolean {
  if (fieldTokens.length !== queryTokens.length) return false;
  for (let i = 0; i < fieldTokens.length; i++) {
    if ((fieldTokens[i] as number) !== (queryTokens[i] as number)) return false;
  }
  return true;
}
