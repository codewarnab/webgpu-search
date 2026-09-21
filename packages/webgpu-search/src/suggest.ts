/**
 * v0.4 autocomplete & suggestion primitives (Issue #10 M5).
 *
 * Framework-agnostic completion and did-you-mean primitives over the same
 * post-fold token streams and integer scorers as search:
 * - `mode: 'prefix'` (default) — prefix-anchored symbol completion via
 *   `scorePrefixTokens` (exact, or typo-tolerant when `fuzzyDistance > 0`);
 *   results are typed `'completion'`.
 * - `mode: 'fuzzy'` — subsequence matching via `scoreFuzzyTokens` when
 *   `fuzzyDistance === 0`, or bounded typo-tolerant substring alignment via
 *   `scoreSubstringTypoTokens` when `fuzzyDistance > 0`; results are typed
 *   `'did-you-mean'`.
 *
 * Ranking reuses the deterministic M5 comparator
 * (`score DESC, weight DESC, exact DESC, length ASC, id ASC`) so suggestion
 * order is bit-for-bit stable across engines and runs.
 *
 * Portable: no DOM refs. Pure option validation + scoring helpers; index
 * enumeration lives in `document-index.ts` (avoids import cycles).
 */

import type { SuggestOptions } from './types';
import { RESULT_LIMIT_MAX } from './text-profile';
import { IncompatibleOptionError } from './errors';

export const SUGGEST_DEFAULT_LIMIT = 5 as const;
export const SUGGEST_DEFAULT_MODE = 'prefix' as const;
export const SUGGEST_MAX_FUZZY_DISTANCE = 2 as const;

/** Suggest options with defaults resolved. */
export interface NormalizedSuggestOptions {
  limit: number;
  mode: 'prefix' | 'fuzzy';
  fuzzyDistance: number;
  field?: string;
}

/**
 * Validate and normalize suggest options (fail-closed).
 * - `undefined`/`true` resolve to defaults.
 * - `limit` coerces like search limits (non-finite → default 5, floor,
 *   clamp 1..RESULT_LIMIT_MAX).
 * - `mode` must be 'prefix' or 'fuzzy'.
 * - `fuzzyDistance` must be an integer 0..2.
 * - `field` must be a non-empty string when provided (existence is checked
 *   against the index at suggest() time).
 */
export function normalizeSuggestOptions(
  raw: SuggestOptions | boolean | undefined
): NormalizedSuggestOptions {
  if (raw === undefined || raw === true) {
    return { limit: SUGGEST_DEFAULT_LIMIT, mode: SUGGEST_DEFAULT_MODE, fuzzyDistance: 0 };
  }
  if (raw === false) {
    throw new TypeError('[webgpu-search] suggest:false disables suggestions; omit the option instead.');
  }
  if (typeof raw !== 'object' || raw === null) {
    throw new TypeError('[webgpu-search] suggest options must be an object, true, or undefined.');
  }
  const opts = raw as SuggestOptions;

  let limit: number = SUGGEST_DEFAULT_LIMIT;
  if (opts.limit !== undefined) {
    const n: number = typeof opts.limit === 'number' ? opts.limit : Number(opts.limit);
    if (!Number.isFinite(n)) {
      limit = SUGGEST_DEFAULT_LIMIT;
    } else {
      const floored = Math.floor(n);
      if (floored < 1) limit = 1;
      else if (floored > RESULT_LIMIT_MAX) limit = RESULT_LIMIT_MAX;
      else limit = floored;
    }
  }

  const mode = opts.mode ?? SUGGEST_DEFAULT_MODE;
  if (mode !== 'prefix' && mode !== 'fuzzy') {
    throw new IncompatibleOptionError(
      'mode',
      `Unknown suggest mode '${String(mode)}'. Expected 'prefix' or 'fuzzy'.`
    );
  }

  let fuzzyDistance = 0;
  if (opts.fuzzyDistance !== undefined) {
    const fd = opts.fuzzyDistance;
    if (typeof fd !== 'number' || !Number.isInteger(fd) || fd < 0 || fd > SUGGEST_MAX_FUZZY_DISTANCE) {
      throw new RangeError(
        `[webgpu-search] suggest.fuzzyDistance must be an integer 0..${SUGGEST_MAX_FUZZY_DISTANCE}, got ${String(fd)}.`
      );
    }
    fuzzyDistance = fd;
  }

  let field: string | undefined = undefined;
  if (opts.field !== undefined) {
    if (typeof opts.field !== 'string' || opts.field.length === 0) {
      throw new TypeError('[webgpu-search] suggest.field must be a non-empty string.');
    }
    field = opts.field;
  }

  return field === undefined ? { limit, mode, fuzzyDistance } : { limit, mode, fuzzyDistance, field };
}

/**
 * Totally-ordered suggestion candidate keys (mirrors RankableCandidate).
 * Kept local to avoid a ranking.ts value import cycle in either direction;
 * `document-index.ts` maps these onto `RankableCandidate` for sorting.
 */
export interface SuggestCandidateKeys {
  score: number;
  fieldWeight: number;
  isExactMatch: boolean;
  matchedLength: number;
  id: string | number;
  docIndex: number;
}
