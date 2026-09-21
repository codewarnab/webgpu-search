/**
 * bounded typo tolerance .
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
 * Portable: no DOM refs. Operates on post-normalization u32 token streams so GPU
 * (exact-only) and CPU (exact + typo) share one normalization pipeline;
 * typo queries route to the CPU reference engine (see exact-scorer.ts).
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
 *
 * Note: `maxDistance` alone does NOT enable tolerance — `enabled: true`
 * is required. `{ maxDistance: 2 }` without `enabled` validates fine but
 * stays disabled (contract-consistent, sharp edge for callers).
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
 * Maximum edit distance allowed for a term of the given post-normalization length.
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

// Module-level DP scratch rows shared across bounded-DL calls.
// JS is single-threaded and every DP call is synchronous, so reuse is safe
// (no awaits inside the inner loops). `ensureSharedCapacity` grows the
// buffers; rotation swaps roles without allocating.
let sharedPrev2: number[] = [];
let sharedPrev: number[] = [];
let sharedCur: number[] = [];
function ensureSharedCapacity(m: number): void {
  const need: number = m + 1;
  if (sharedPrev.length < need) {
    sharedPrev2 = new Array<number>(need);
    sharedPrev = new Array<number>(need);
    sharedCur = new Array<number>(need);
  }
}

/**
 * Bounded DL between `record[recordStart..recordStart+recordLen)` and
 * `query[queryStart..queryStart+queryLen)` without slicing or copying.
 * Shared by typo substring spans and prefix anchored spans so neither
 * path allocates per-candidate spans. Returns `maxDist + 1` past the bound.
 */
export function damerauLevenshteinBoundedRange(
  record: Uint32Array | readonly number[],
  recordStart: number,
  recordLen: number,
  query: Uint32Array | readonly number[],
  queryStart: number,
  queryLen: number,
  maxDist: number
): number {
  const lenDiff: number = recordLen > queryLen ? recordLen - queryLen : queryLen - recordLen;
  if (lenDiff > maxDist) return maxDist + 1;
  if (recordLen === 0) return queryLen <= maxDist ? queryLen : maxDist + 1;
  if (queryLen === 0) return recordLen <= maxDist ? recordLen : maxDist + 1;

  ensureSharedCapacity(queryLen);
  let prev2: number[] = sharedPrev2;
  let prev: number[] = sharedPrev;
  let cur: number[] = sharedCur;
  for (let j = 0; j <= queryLen; j++) prev[j] = j;

  for (let i = 1; i <= recordLen; i++) {
    cur[0] = i;
    let rowMin: number = cur[0];
    const ai: number = record[recordStart + i - 1] as number;
    for (let j = 1; j <= queryLen; j++) {
      const bj: number = query[queryStart + j - 1] as number;
      const cost: number = ai === bj ? 0 : 1;
      let v: number = prev[j] + 1;
      const ins: number = cur[j - 1] + 1;
      if (ins < v) v = ins;
      const sub: number = prev[j - 1] + cost;
      if (sub < v) v = sub;
      if (
        i > 1 &&
        j > 1 &&
        ai === (query[queryStart + j - 2] as number) &&
        (record[recordStart + i - 2] as number) === bj
      ) {
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
  const dist: number = prev[queryLen] as number;
  return dist <= maxDist ? dist : maxDist + 1;
}

/**
 * Bounded Damerau-Levenshtein distance (optimal-string-alignment: one
 * adjacent transposition counts as a single edit) between two token
 * streams. Returns `maxDist + 1` when the true distance exceeds the bound
 * (early exit keeps long-record scans cheap). i32-safe integer arithmetic.
 *
 * Reuses module-level scratch rows (see `ensureSharedCapacity`) instead of
 * allocating 3×(m+1) arrays per call — `findBestTypoWindow` invokes this
 * up to n×(2·allowed+1) times per record, so per-call allocation is a
 * GC DoS on long repetitive corpora.
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

  ensureSharedCapacity(m);
  // Three-row OSA DP: prev2 (i-2), prev (i-1), cur (i).
  let prev2: number[] = sharedPrev2;
  let prev: number[] = sharedPrev;
  let cur: number[] = sharedCur;
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
    // Prefix-exact gate before any DP work. Clamped per span below so the
    // gate never reads past the candidate span end.
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
      // Per-span gate clamp: a span shorter than the gate cannot satisfy
      // the prefix-exact contract, so skip it instead of reading past w.
      if (w < gateLen) continue;
      // Compare via shared scratch without slice allocation.
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
  return damerauLevenshteinBoundedRange(record, start, winLen, term, 0, term.length, maxDist);
}
