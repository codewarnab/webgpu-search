/**
 * Autocomplete primitives.
 *
 * Framework-agnostic completion and did-you-mean primitives over the same
 * post-normalization token streams and integer scorers as search:
 * - `mode: 'prefix'` (default) — prefix-anchored symbol completion via
 * `scorePrefixTokens` (exact, or typo-tolerant when `fuzzyDistance > 0`);
 * results are typed `'completion'`.
 * - `mode: 'fuzzy'` — subsequence matching via `scoreFuzzyTokens` when
 * `fuzzyDistance === 0`, or bounded typo-tolerant substring alignment via
 * `scoreSubstringTypoTokens` when `fuzzyDistance > 0`; results are typed
 * `'did-you-mean'`.
 *
 * Ranking reuses the deterministic comparator
 * (`score DESC, weight DESC, exact DESC, length ASC, id ASC`) so suggestion
 * order is bit-for-bit stable across engines and runs.
 *
 * Portable: no DOM refs. Pure option validation + scoring helpers; index
 * enumeration lives in `document-index.ts` (avoids import cycles).
 */

import type { AutocompleteOptions, TieBreakerCriterion } from './types';
import { RESULT_LIMIT_MAX } from './text-profile';
import { IncompatibleOptionError } from './errors';
import { normalizeTieBreakers, DEFAULT_TIE_BREAKERS } from './ranking';
import type { RankableCandidate } from './ranking';

export const AUTOCOMPLETE_DEFAULT_LIMIT = 5 as const;
export const AUTOCOMPLETE_DEFAULT_MODE = 'prefix' as const;
export const AUTOCOMPLETE_MAX_FUZZY_DISTANCE = 2 as const;

/** Suggest options with defaults resolved. */
export interface NormalizedAutocompleteOptions {
  limit: number;
  mode: 'prefix' | 'fuzzy';
  fuzzyDistance: number;
  field?: string;
  tieBreakers: TieBreakerCriterion[];
}

/**
 * Validate and normalize autocomplete options (fail-closed).
 * - `undefined`/`true` resolve to defaults.
 * - `false` is only meaningful as inline `search({ autocomplete: false })`
 * (disabled); standalone `autocomplete()` rejects it — omit the option instead.
 * - `limit` coerces like search limits (string numerics via `Number()`,
 * non-finite → default 5, fractions floored, clamp 1..RESULT_LIMIT_MAX;
 * note `fuzzyDistance` is strict number-only by contrast).
 * - `mode` must be 'prefix' or 'fuzzy'.
 * - `fuzzyDistance` must be an integer 0..2 (strict `typeof number`; `'1'` throws).
 * - `field` must be a non-empty string when provided (existence is checked
 * against the index at autocomplete() time).
 * - `tieBreakers` defaults to the 5-tier order; validated fail-closed.
 */
export function normalizeAutocompleteOptions(
  raw: AutocompleteOptions | boolean | undefined
): NormalizedAutocompleteOptions {
  if (raw === undefined || raw === true) {
    return { limit: AUTOCOMPLETE_DEFAULT_LIMIT, mode: AUTOCOMPLETE_DEFAULT_MODE, fuzzyDistance: 0, tieBreakers: [...DEFAULT_TIE_BREAKERS] };
  }
  if (raw === false) {
    throw new TypeError('[webgpu-search] autocomplete:false disables inline search suggestions; omit the option (or standalone autocomplete() call) instead.');
  }
  if (typeof raw !== 'object' || raw === null) {
    throw new TypeError('[webgpu-search] autocomplete options must be an object, true, or undefined.');
  }
  const opts = raw as AutocompleteOptions;

  let limit: number = AUTOCOMPLETE_DEFAULT_LIMIT;
  if (opts.limit !== undefined) {
    const n: number = typeof opts.limit === 'number' ? opts.limit : Number(opts.limit);
    if (!Number.isFinite(n)) {
      limit = AUTOCOMPLETE_DEFAULT_LIMIT;
    } else {
      const floored = Math.floor(n);
      if (floored < 1) limit = 1;
      else if (floored > RESULT_LIMIT_MAX) limit = RESULT_LIMIT_MAX;
      else limit = floored;
    }
  }

  const mode = opts.mode ?? AUTOCOMPLETE_DEFAULT_MODE;
  if (mode !== 'prefix' && mode !== 'fuzzy') {
    throw new IncompatibleOptionError(
      'mode',
      `Unknown autocomplete mode '${String(mode)}'. Expected 'prefix' or 'fuzzy'.`
    );
  }

  let fuzzyDistance = 0;
  if (opts.fuzzyDistance !== undefined) {
    const fd = opts.fuzzyDistance;
    if (typeof fd !== 'number' || !Number.isInteger(fd) || fd < 0 || fd > AUTOCOMPLETE_MAX_FUZZY_DISTANCE) {
      throw new RangeError(
        `[webgpu-search] autocomplete.fuzzyDistance must be an integer 0..${AUTOCOMPLETE_MAX_FUZZY_DISTANCE}, got ${String(fd)}.`
      );
    }
    fuzzyDistance = fd;
  }

  let field: string | undefined = undefined;
  if (opts.field !== undefined) {
    if (typeof opts.field !== 'string' || opts.field.length === 0) {
      throw new TypeError('[webgpu-search] autocomplete.field must be a non-empty string.');
    }
    field = opts.field;
  }

  const tieBreakers = normalizeTieBreakers(opts.tieBreakers);

  const base = { limit, mode, fuzzyDistance, tieBreakers };
  return field === undefined ? base : { ...base, field };
}

/**
 * @deprecated Use RankableCandidate. Identical alias kept so existing imports
 * keep working without a duplicate interface to drift.
 */
export type SuggestCandidateKeys = RankableCandidate;

/** @deprecated Use AUTOCOMPLETE_DEFAULT_LIMIT. */
export const SUGGEST_DEFAULT_LIMIT = AUTOCOMPLETE_DEFAULT_LIMIT;
/** @deprecated Use AUTOCOMPLETE_DEFAULT_MODE. */
export const SUGGEST_DEFAULT_MODE = AUTOCOMPLETE_DEFAULT_MODE;
/** @deprecated Use AUTOCOMPLETE_MAX_FUZZY_DISTANCE. */
export const SUGGEST_MAX_FUZZY_DISTANCE = AUTOCOMPLETE_MAX_FUZZY_DISTANCE;
/** @deprecated Use NormalizedAutocompleteOptions. */
export type NormalizedSuggestOptions = NormalizedAutocompleteOptions;
/** @deprecated Use normalizeAutocompleteOptions. */
export const normalizeSuggestOptions = normalizeAutocompleteOptions;
