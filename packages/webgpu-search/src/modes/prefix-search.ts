/**
 * v0.4 prefix search mode (Issue #10 M4).
 *
 * Prefix-anchored matching for symbol navigation and autocomplete: a record
 * matches when the query is an exact (or typo-tolerant) prefix of any
 * whitespace/punctuation-delimited token in the record, including position
 * 0. `camelCase` boundaries are NOT token starts (code-aware splitting
 * lands with M6 tokenizer hooks).
 *
 * Integer scoring (i32 semantics):
 *   score = 1000 - tokenStart·10 - (strLen - queryLen) + (tokenStart == 0 ? 40 : 0) - distance·100
 * An exact full-string match at 0 scores 1040; mid-string token prefixes
 * decay with the token offset so earlier symbols outrank later ones.
 *
 * Portable: no DOM refs. Operates on post-fold u32 token streams; shares
 * the token delimiter set with token-search.ts (single source of truth).
 */

import type { PrefixSearchOptions } from '../types';
import {
  allowedDistanceForTerm,
  damerauLevenshteinBoundedRange,
  TYPO_DISTANCE_PENALTY,
  type NormalizedTypoOptions
} from './typo-distance';
import { isTokenDelimiter } from './token-search';

/** Prefix options with defaults resolved. */
export interface NormalizedPrefixOptions {
  /** Leading query code points used for matching (undefined = full query). */
  prefixLength?: number;
  exactCase: boolean;
}

/**
 * Validate prefix options (fail-closed). `prefixLength` must be an integer
 * >= 1; the range-vs-query check happens per query via
 * `assertPrefixLengthForQuery` (fail-closed even on empty corpora).
 * `exactCase` must be a boolean; callers enforce index-polarity agreement
 * (see cpu-reference.ts). Note: a fixed `prefixLength` with a shorter
 * autocomplete keystroke throws `RangeError` (fail-closed) — autocomplete
 * callers should clamp or catch and treat as no-match.
 */
export function normalizePrefixOptions(
  raw: PrefixSearchOptions | undefined
): NormalizedPrefixOptions {
  if (raw === undefined) return { exactCase: false };
  if (typeof raw !== 'object' || raw === null) {
    throw new TypeError('[webgpu-search] prefixMatch must be an object.');
  }
  let prefixLength: number | undefined = undefined;
  if (raw.prefixLength !== undefined) {
    if (!Number.isInteger(raw.prefixLength) || (raw.prefixLength as number) < 1) {
      throw new RangeError(
        `[webgpu-search] prefixMatch.prefixLength must be an integer >= 1, got ${String(raw.prefixLength)}.`
      );
    }
    prefixLength = raw.prefixLength;
  }
  const exactCase = raw.exactCase ?? false;
  if (typeof exactCase !== 'boolean') {
    throw new TypeError('[webgpu-search] prefixMatch.exactCase must be a boolean.');
  }
  return prefixLength === undefined ? { exactCase } : { prefixLength, exactCase };
}

export interface PrefixMatchResult {
  matched: boolean;
  score: number;
  /** Record offset of the winning token start (-1 on no match). */
  matchStart: number;
  /** Edit distance of the winning alignment (0 = exact). */
  distance: number;
  /** Record span length covered (effective query length for exact). */
  windowLength: number;
}

/** Record offsets where tokens start (after-delimiter positions + 0 when non-delimiter). */
function tokenStarts(record: Uint32Array | readonly number[]): number[] {
  if (record.length === 0) return [];
  const starts: number[] = [];
  // Offset 0 is a token start only when the record does not lead with a
  // delimiter; otherwise the first real start comes after the delimiters
  // (prevents a delimiter alignment from earning the +40 anchor bonus).
  if (!isTokenDelimiter(record[0] as number)) starts.push(0);
  for (let i = 1; i < record.length; i++) {
    if (isTokenDelimiter(record[i - 1] as number) && !isTokenDelimiter(record[i] as number)) {
      starts.push(i);
    }
  }
  return starts;
}

/**
 * Fail-closed `prefixLength` vs query check, hoisted before scan loops so
 * empty corpora throw identically to non-empty ones. Direct
 * `scorePrefixTokens` callers also hit the per-record check; index and
 * CPU-reference paths must call this upfront (before early noHits exits).
 */
export function assertPrefixLengthForQuery(
  prefixOpts: NormalizedPrefixOptions,
  queryLen: number
): number {
  if (prefixOpts.prefixLength !== undefined) {
    if ((prefixOpts.prefixLength as number) > queryLen) {
      throw new RangeError(
        `[webgpu-search] prefixMatch.prefixLength (${prefixOpts.prefixLength}) exceeds query length (${queryLen}).`
      );
    }
    return prefixOpts.prefixLength as number;
  }
  return queryLen;
}

/**
 * Score one record against a normalized query. `effectiveLen` truncates the
 * query head when `prefixLength` is set (validated here against the query).
 * i32 arithmetic via `|0`/`Math.imul`.
 */
export function scorePrefixTokens(
  record: Uint32Array | readonly number[],
  query: Uint32Array | readonly number[],
  prefixOpts: NormalizedPrefixOptions,
  typo: NormalizedTypoOptions
): PrefixMatchResult {
  const noMatch: PrefixMatchResult = { matched: false, score: 0, matchStart: -1, distance: 0, windowLength: 0 };
  const strLen: number = record.length;
  if (strLen === 0 || query.length === 0) return noMatch;

  const effLen: number = assertPrefixLengthForQuery(prefixOpts, query.length);

  const allowed: 0 | 1 | 2 = allowedDistanceForTerm(effLen, typo);
  const starts: number[] = tokenStarts(record);

  let bestStart = -1;
  let bestDist = 0;
  let bestWin = effLen;
  if (allowed === 0) {
    for (let s = 0; s < starts.length; s++) {
      const st: number = starts[s] as number;
      if (st + effLen > strLen) continue;
      // Compare against the query head without slicing.
      let ok = true;
      for (let j = 0; j < effLen; j++) {
        if ((record[st + j] as number) !== (query[j] as number)) {
          ok = false;
          break;
        }
      }
      if (ok) {
        bestStart = st;
        break;
      }
    }
  } else {
    // Typo-tolerant: span lengths [effLen - allowed, effLen + allowed]
    // anchored at each token start; lowest distance wins, earliest on ties.
    // The prefix-exact gate applies to the query head.
    const gateLen: number = typo.prefixExactLength < effLen ? typo.prefixExactLength : effLen;
    bestDist = allowed + 1;
    const minWin: number = effLen - allowed >= 1 ? effLen - allowed : 1;
    const maxWin: number = effLen + allowed;
    for (let s = 0; s < starts.length; s++) {
      const st: number = starts[s] as number;
      if (st >= strLen) continue;
      let gateOk = true;
      for (let k = 0; k < gateLen; k++) {
        if (st + k >= strLen || (record[st + k] as number) !== (query[k] as number)) {
          gateOk = false;
          break;
        }
      }
      if (!gateOk) continue;
      for (let w = minWin; w <= maxWin; w++) {
        if (st + w > strLen) break;
        // Windows shorter than the prefix-exact gate cannot satisfy it.
        if (w < gateLen) continue;
        const dist: number = prefixWindowDistance(record, st, w, query, effLen, bestDist - 1);
        if (dist < bestDist) {
          bestDist = dist;
          bestStart = st;
          bestWin = w;
          if (dist === 0) break;
        }
      }
      if (bestDist === 0) break;
    }
    if (bestStart < 0) return noMatch;
  }

  if (bestStart < 0) return noMatch;
  const anchorBonus: number = bestStart === 0 ? 40 : 0;
  const score: number =
    ((1000 - Math.imul(bestStart, 10) - (strLen - effLen) + anchorBonus - Math.imul(bestDist, TYPO_DISTANCE_PENALTY)) | 0);
  return { matched: true, score, matchStart: bestStart, distance: bestDist, windowLength: bestWin };
}

/** Bounded DL between record[st..st+winLen) and query[0..effLen) sliceless. */
function prefixWindowDistance(
  record: Uint32Array | readonly number[],
  st: number,
  winLen: number,
  query: Uint32Array | readonly number[],
  effLen: number,
  maxDist: number
): number {
  const lenDiff: number = winLen > effLen ? winLen - effLen : effLen - winLen;
  if (lenDiff > maxDist) return maxDist + 1;
  // Sliceless offset comparison via shared scratch (no per-span copies).
  return damerauLevenshteinBoundedRange(record, st, winLen, query, 0, effLen, maxDist);
}
