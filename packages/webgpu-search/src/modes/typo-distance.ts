/**
 * v0.4 bounded typo tolerance (Issue #10 M4).
 *
 * Bounded Damerau-Levenshtein (optimal-string-alignment variant) with
 * early exit, applied per query term against candidate record spans.
 *
 * Gates (per plan section 3.3):
 * - Terms shorter than `minWordLengthForOneTypo` allow zero edits.
 * - Terms shorter than `minWordLengthForTwoTypos` allow at most one edit.
 * - The first `prefixExactLength` code points must match exactly.
 * - `maxDistance` is 1 or 2 (fail-closed otherwise).
 *
 * Portable: no DOM refs. Operates on post-fold u32 token streams so GPU
 * (exact-only) and CPU (exact + typo) share one normalization pipeline;
 * typo queries route to the CPU reference engine (see cpu-reference.ts).
 */

import type { TypoToleranceOptions } from '../types';

/** Typo options with all defaults resolved. */
export interface NormalizedTypoOptions {
  enabled: boolean;
  maxDistance: 1 | 2;
  minWordLengthForOneTypo: number;
  minWordLengthForTwoTypos: number;
  prefixExactLength: number;
}

export const DEFAULT_MAX_DISTANCE = 1 as const;
export const DEFAULT_MIN_WORD_LENGTH_FOR_ONE_TYPO = 4 as const;
export const DEFAULT_MIN_WORD_LENGTH_FOR_TWO_TYPOS = 8 as const;
export const DEFAULT_PREFIX_EXACT_LENGTH = 1 as const;

/** Per-distance score penalty subtracted from the exact-match base score. */
export const TYPO_DISTANCE_PENALTY = 100 as const;

/**
 * Validate and normalize caller-supplied typo tolerance options.
 * Accepts `true` (defaults, enabled), `false`/`undefined` (disabled),
 * or a partial options object (`enabled` defaults to false per contract).
 * Throws TypeError/RangeError on malformed values (fail-closed).
 */
export function normalizeTypoTolerance(
  raw: TypoToleranceOptions | boolean | undefined
): NormalizedTypoOptions {
  if (raw === undefined || raw === false) {
    return {
      enabled: false,
      maxDistance: DEFAULT_MAX_DISTANCE,
      minWordLengthForOneTypo: DEFAULT_MIN_WORD_LENGTH_FOR_ONE_TYPO,
      minWordLengthForTwoTypos: DEFAULT_MIN_WORD_LENGTH_FOR_TWO_TYPOS,
      prefixExactLength: DEFAULT_PREFIX_EXACT_LENGTH
    };
  }
  if (raw === true) {
    return {
      enabled: true,
      maxDistance: DEFAULT_MAX_DISTANCE,
      minWordLengthForOneTypo: DEFAULT_MIN_WORD_LENGTH_FOR_ONE_TYPO,
      minWordLengthForTwoTypos: DEFAULT_MIN_WORD_LENGTH_FOR_TWO_TYPOS,
      prefixExactLength: DEFAULT_PREFIX_EXACT_LENGTH
    };
  }
  if (typeof raw !== 'object' || raw === null) {
    throw new TypeError(
      '[webgpu-search] typoTolerance must be a boolean or TypoToleranceOptions object.'
    );
  }
  const enabled = raw.enabled ?? false;
  if (typeof enabled !== 'boolean') {
    throw new TypeError('[webgpu-search] typoTolerance.enabled must be a boolean.');
  }
  const maxDistance = raw.maxDistance ?? DEFAULT_MAX_DISTANCE;
  if (maxDistance !== 1 && maxDistance !== 2) {
    throw new RangeError(
      `[webgpu-search] typoTolerance.maxDistance must be 1 or 2, got ${String(maxDistance)}.`
    );
  }
  const minOne = raw.minWordLengthForOneTypo ?? DEFAULT_MIN_WORD_LENGTH_FOR_ONE_TYPO;
  if (!Number.isInteger(minOne) || minOne < 1) {
    throw new RangeError(
      `[webgpu-search] typoTolerance.minWordLengthForOneTypo must be an integer >= 1, got ${String(minOne)}.`
    );
  }
  const minTwo = raw.minWordLengthForTwoTypos ?? DEFAULT_MIN_WORD_LENGTH_FOR_TWO_TYPOS;
  if (!Number.isInteger(minTwo) || minTwo < 1) {
    throw new RangeError(
      `[webgpu-search] typoTolerance.minWordLengthForTwoTypos must be an integer >= 1, got ${String(minTwo)}.`
    );
  }
  if (minTwo < minOne) {
    throw new RangeError(
      `[webgpu-search] typoTolerance.minWordLengthForTwoTypos (${minTwo}) must be >= minWordLengthForOneTypo (${minOne}).`
    );
  }
  const prefixExact = raw.prefixExactLength ?? DEFAULT_PREFIX_EXACT_LENGTH;
  if (!Number.isInteger(prefixExact) || prefixExact < 0) {
    throw new RangeError(
      `[webgpu-search] typoTolerance.prefixExactLength must be an integer >= 0, got ${String(prefixExact)}.`
    );
  }
  return {
    enabled,
    maxDistance,
    minWordLengthForOneTypo: minOne,
    minWordLengthForTwoTypos: minTwo,
    prefixExactLength: prefixExact
  };
}

/**
 * Maximum edit distance allowed for a term of the given post-fold length.
 * Returns 0 when typo is disabled or the term trips a length lock.
 */
export function allowedDistanceForTerm(
  termLen: number,
  opts: NormalizedTypoOptions
): 0 | 1 | 2 {
  if (!opts.enabled) return 0;
  if (termLen < opts.minWordLengthForOneTypo) return 0;
  if (termLen < opts.minWordLengthForTwoTypos) {
    return opts.maxDistance > 1 ? 1 : opts.maxDistance;
  }
  return opts.maxDistance;
}

/**
 * Bounded Damerau-Levenshtein distance (optimal-string-alignment: one
 * adjacent transposition counts as a single edit) between two token
 * streams. Returns `maxDist + 1` when the true distance exceeds the bound
 * (early exit keeps long-record scans cheap). i32-safe integer arithmetic.
 */
export function damerauLevenshteinBounded(
  a: Uint32Array | readonly number[],
  b: Uint32Array | readonly number[],
  maxDist: number
): number {
  const n: number = a.length;
  const m: number = b.length;
  if (maxDist < 0) return 0;
  const lenDiff: number = n > m ? n - m : m - n;
  if (lenDiff > maxDist) return maxDist + 1;
  if (n === 0) return m <= maxDist ? m : maxDist + 1;
  if (m === 0) return n <= maxDist ? n : maxDist + 1;

  // Three-row OSA DP: prev2 (i-2), prev (i-1), cur (i).
  let prev2: number[] = new Array<number>(m + 1);
  let prev: number[] = new Array<number>(m + 1);
  let cur: number[] = new Array<number>(m + 1);
  for (let j = 0; j <= m; j++) prev[j] = j;

  for (let i = 1; i <= n; i++) {
    cur[0] = i;
    let rowMin: number = cur[0];
    const ai: number = a[i - 1] as number;
    for (let j = 1; j <= m; j++) {
      const bj: number = b[j - 1] as number;
      const cost: number = ai === bj ? 0 : 1;
      let v: number = prev[j] + 1; // deletion
      const ins: number = cur[j - 1] + 1; // insertion
      if (ins < v) v = ins;
      const sub: number = prev[j - 1] + cost; // substitution
      if (sub < v) v = sub;
      if (
        i > 1 &&
        j > 1 &&
        ai === (b[j - 2] as number) &&
        (a[i - 2] as number) === bj
      ) {
        const transp: number = prev2[j - 2] + 1; // adjacent transposition
        if (transp < v) v = transp;
      }
      // Band clamp: values beyond the bound carry no signal downstream.
      if (v > maxDist + 1) v = maxDist + 1;
      cur[j] = v;
      if (v < rowMin) rowMin = v;
    }
    if (rowMin > maxDist) return maxDist + 1;
    const tmp: number[] = prev2;
    prev2 = prev;
    prev = cur;
    cur = tmp;
  }
  const dist: number = prev[m] as number;
  return dist <= maxDist ? dist : maxDist + 1;
}

export interface TypoWindowMatch {
  matched: boolean;
  distance: number;
  start: number;
  /** Window length in the record that produced the match. */
  windowLength: number;
}

/**
 * Slide a query term over a record and return the best (lowest-distance,
 * earliest-start) span with `distance <= allowedMax`. Span lengths range over
 * `[termLen - allowedMax, termLen + allowedMax]` clamped to >= 1.
 * The first `prefixExact` code points of term and span must agree exactly.
 * With `allowedMax === 0` this degrades to an exact earliest-substring scan
 * (no DP), keeping the common path fast.
 */
export function findBestTypoWindow(
  record: Uint32Array | readonly number[],
  term: Uint32Array | readonly number[],
  allowedMax: number,
  prefixExact: number
): TypoWindowMatch {
  const noMatch: TypoWindowMatch = { matched: false, distance: allowedMax + 1, start: -1, windowLength: 0 };
  const n: number = record.length;
  const t: number = term.length;
  if (t === 0 || n === 0 || t > n + allowedMax) return noMatch;

  if (allowedMax === 0) {
    const maxStart: number = n - t;
    for (let start = 0; start <= maxStart; start++) {
      let ok = true;
      for (let j = 0; j < t; j++) {
        if ((record[start + j] as number) !== (term[j] as number)) {
          ok = false;
          break;
        }
      }
      if (ok) return { matched: true, distance: 0, start, windowLength: t };
    }
    return noMatch;
  }

  const gateLen: number = prefixExact < t ? prefixExact : t;
  const minWin: number = t - allowedMax >= 1 ? t - allowedMax : 1;
  const maxWin: number = t + allowedMax;
  let bestDist: number = allowedMax + 1;
  let bestStart = -1;
  let bestWin = 0;

  const maxStart: number = n - minWin;
  for (let start = 0; start <= maxStart; start++) {
    // Prefix-exact gate before any DP work.
    let gateOk = true;
    for (let k = 0; k < gateLen; k++) {
      if (start + k >= n || (record[start + k] as number) !== (term[k] as number)) {
        gateOk = false;
        break;
      }
    }
    if (!gateOk) continue;
    for (let w = minWin; w <= maxWin; w++) {
      if (start + w > n) break;
      // Compare via index accessors instead of slice allocation.
      const dist: number = damerauWindow(record, start, w, term, bestDist - 1);
      if (dist < bestDist) {
        bestDist = dist;
        bestStart = start;
        bestWin = w;
        if (dist === 0) {
          // Earliest start wins: starts ascend, so the first distance-0
          // span is optimal overall.
          return { matched: true, distance: 0, start: bestStart, windowLength: bestWin };
        }
      }
    }
  }
  if (bestStart < 0) return noMatch;
  return { matched: true, distance: bestDist, start: bestStart, windowLength: bestWin };
}

/** Bounded DL between record[start..start+winLen) and term without slicing. */
function damerauWindow(
  record: Uint32Array | readonly number[],
  start: number,
  winLen: number,
  term: Uint32Array | readonly number[],
  maxDist: number
): number {
  const m: number = term.length;
  const lenDiff: number = winLen > m ? winLen - m : m - winLen;
  if (lenDiff > maxDist) return maxDist + 1;
  if (winLen === 0) return m <= maxDist ? m : maxDist + 1;
  if (m === 0) return winLen <= maxDist ? winLen : maxDist + 1;

  let prev2: number[] = new Array<number>(m + 1);
  let prev: number[] = new Array<number>(m + 1);
  let cur: number[] = new Array<number>(m + 1);
  for (let j = 0; j <= m; j++) prev[j] = j;

  for (let i = 1; i <= winLen; i++) {
    cur[0] = i;
    let rowMin: number = cur[0];
    const ai: number = record[start + i - 1] as number;
    for (let j = 1; j <= m; j++) {
      const bj: number = term[j - 1] as number;
      const cost: number = ai === bj ? 0 : 1;
      let v: number = prev[j] + 1;
      const ins: number = cur[j - 1] + 1;
      if (ins < v) v = ins;
      const sub: number = prev[j - 1] + cost;
      if (sub < v) v = sub;
      if (i > 1 && j > 1 && ai === (term[j - 2] as number) && (record[start + i - 2] as number) === bj) {
        const transp: number = prev2[j - 2] + 1;
        if (transp < v) v = transp;
      }
      if (v > maxDist + 1) v = maxDist + 1;
      cur[j] = v;
      if (v < rowMin) rowMin = v;
    }
    if (rowMin > maxDist) return maxDist + 1;
    const tmp: number[] = prev2;
    prev2 = prev;
    prev = cur;
    cur = tmp;
  }
  const dist: number = prev[m] as number;
  return dist <= maxDist ? dist : maxDist + 1;
}
