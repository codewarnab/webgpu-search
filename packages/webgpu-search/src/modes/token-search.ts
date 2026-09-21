/**
 * v0.4 multi-term token search (Issue #10 M4).
 *
 * Splits the normalized query into terms on ASCII whitespace/punctuation
 * and matches each term as a contiguous (or typo-tolerant) substring of
 * the record. `operator: 'and'` (default) requires every term;
 * `'or'` requires any term; `minMatchCount` overrides both with an exact
 * quorum. Per-term hits aggregate with a proximity penalty so clustered
 * terms outrank scattered ones.
 *
 * Integer scoring (i32 semantics, single-term reduction invariant):
 *   score = Σ(1000 - start_i·10) - (strLen - Σlen_i) - proximity·2 - typoPenalty
 * where proximity = max(0, maxEnd - minStart - Σlen_i) over matched terms
 * (span ends use aligned typo span lengths, not raw query-term lengths)
 * and typoPenalty = Σ(distance_i · 100). A single exact term reduces to the
 * substring formula `1000 - start·10 - (strLen - termLen)`.
 *
 * Duplicate query terms (e.g. "hello hello") match the same span twice and
 * double-count in the score while highlights merge to one range —
 * score↔highlight cardinality differs by design; callers wanting set
 * semantics should dedupe terms before scoring.
 *
 * Portable: no DOM refs. Operates on post-fold u32 token streams.
 */

import type { TokenMatchOptions } from '../types';
import {
  allowedDistanceForTerm,
  findBestTypoWindow,
  TYPO_DISTANCE_PENALTY,
  type NormalizedTypoOptions
} from './typo-distance';

/**
 * ASCII delimiter code points splitting query terms and record tokens.
 * Whitespace (TAB/LF/CR/space) plus ASCII punctuation so `camelCase`
 * stays intact (code-aware splitting lands with M6 tokenizer hooks).
 */
const TOKEN_DELIMITERS: ReadonlySet<number> = new Set<number>([
  9, 10, 13, 32, // \t \n \r space
  33, 34, 39, 40, 41, 44, 46, 58, 59, 63, // ! " ' ( ) , . : ; ?
  91, 93, 123, 125, // [ ] { }
  47, 92, 124, // / \ |
  45, 95, // - _
  43, 42, 61, // + * =
  60, 62, // < >
  38, 94, 37, 36, 35, 64, 126, 96 // & ^ % $ # @ ~ `
]);

/** True for code points that split query terms and record tokens. */
export function isTokenDelimiter(cp: number): boolean {
  return TOKEN_DELIMITERS.has(cp);
}

/**
 * Split normalized query tokens into non-empty terms on delimiters.
 * An all-delimiter query yields zero terms (callers treat as no-match).
 */
export function splitQueryTerms(queryTokens: Uint32Array | readonly number[]): Uint32Array[] {
  const terms: Uint32Array[] = [];
  let start = -1;
  const n: number = queryTokens.length;
  for (let i = 0; i <= n; i++) {
    const cp: number = i < n ? (queryTokens[i] as number) : -1;
    if (i === n || isTokenDelimiter(cp)) {
      if (start >= 0) {
        const len: number = i - start;
        const term = new Uint32Array(len);
        for (let k = 0; k < len; k++) term[k] = queryTokens[start + k] as number;
        terms.push(term);
        start = -1;
      }
    } else if (start < 0) {
      start = i;
    }
  }
  return terms;
}

/** Token match options with defaults resolved. */
export interface NormalizedTokenMatchOptions {
  operator: 'and' | 'or';
  minMatchCount?: number;
}

/**
 * Validate and normalize token match options (fail-closed).
 * `minMatchCount` must be an integer >= 1 when provided; the quorum-vs-term
 * check happens per query (minMatchCount > termCount matches nothing).
 */
export function normalizeTokenMatchOptions(
  raw: TokenMatchOptions | undefined
): NormalizedTokenMatchOptions {
  if (raw === undefined) return { operator: 'and' };
  if (typeof raw !== 'object' || raw === null) {
    throw new TypeError('[webgpu-search] tokenMatch must be an object.');
  }
  const operator = raw.operator ?? 'and';
  if (operator !== 'and' && operator !== 'or') {
    throw new TypeError(
      `[webgpu-search] tokenMatch.operator must be 'and' or 'or', got ${String(operator)}.`
    );
  }
  if (raw.minMatchCount === undefined) return { operator };
  const mmc = raw.minMatchCount;
  if (typeof mmc !== 'number' || !Number.isInteger(mmc) || mmc < 1) {
    throw new RangeError(
      `[webgpu-search] tokenMatch.minMatchCount must be an integer >= 1, got ${String(mmc)}.`
    );
  }
  return { operator, minMatchCount: mmc };
}

export interface TokenMatchResult {
  matched: boolean;
  score: number;
  /** Per-term record start offsets (-1 for unmatched terms). */
  matchStarts: number[];
  /** Per-term edit distances (0 for exact terms). */
  distances: number[];
  /** Per-term record span lengths (term length for exact, aligned span for typo; 0 when unmatched). */
  matchLengths: number[];
  matchedCount: number;
}

/** Earliest exact substring occurrence of a term, or -1. */
function earliestSubstring(
  record: Uint32Array | readonly number[],
  term: Uint32Array | readonly number[]
): number {
  const n: number = record.length;
  const t: number = term.length;
  if (t === 0 || n < t) return -1;
  const maxStart: number = n - t;
  for (let start = 0; start <= maxStart; start++) {
    let ok = true;
    for (let j = 0; j < t; j++) {
      if ((record[start + j] as number) !== (term[j] as number)) {
        ok = false;
        break;
      }
    }
    if (ok) return start;
  }
  return -1;
}

/**
 * Score one record against pre-split query terms. Terms array must be
 * non-empty (empty yields no match). i32 arithmetic via `|0`/`Math.imul`.
 */
export function scoreTokenTokens(
  record: Uint32Array | readonly number[],
  terms: readonly (Uint32Array | readonly number[])[],
  tokenOpts: NormalizedTokenMatchOptions,
  typo: NormalizedTypoOptions
): TokenMatchResult {
  const empty: TokenMatchResult = { matched: false, score: 0, matchStarts: [], distances: [], matchLengths: [], matchedCount: 0 };
  if (terms.length === 0) return empty;
  const strLen: number = record.length;
  if (strLen === 0) return empty;

  const required: number =
    tokenOpts.minMatchCount ?? (tokenOpts.operator === 'or' ? 1 : terms.length);

  const matchStarts: number[] = new Array<number>(terms.length);
  const distances: number[] = new Array<number>(terms.length);
  const matchLengths: number[] = new Array<number>(terms.length);
  for (let i = 0; i < terms.length; i++) {
    matchStarts[i] = -1;
    distances[i] = 0;
    matchLengths[i] = 0;
  }
  let matchedCount = 0;

  for (let i = 0; i < terms.length; i++) {
    const term = terms[i];
    const termLen: number = term.length;
    // Upper-bound skip uses the configured max (not a hardcoded +2) so a
    // future maxDistance bump cannot silently miss matches.
    const maxAllowed: number = typo.enabled ? typo.maxDistance : 0;
    if (termLen === 0 || termLen > strLen + maxAllowed) continue;
    const allowed: 0 | 1 | 2 = allowedDistanceForTerm(termLen, typo);
    if (allowed === 0) {
      const start: number = earliestSubstring(record, term);
      if (start >= 0) {
        matchStarts[i] = start;
        distances[i] = 0;
        matchLengths[i] = termLen;
        matchedCount++;
      }
    } else {
      const win = findBestTypoWindow(record, term, allowed, typo.prefixExactLength);
      if (win.matched && win.distance <= allowed) {
        matchStarts[i] = win.start;
        distances[i] = win.distance;
        matchLengths[i] = win.windowLength;
        matchedCount++;
      }
    }
  }

  if (matchedCount < required) return { ...empty, matchStarts, distances, matchLengths, matchedCount };

  let sumBase = 0;
  let sumLens = 0;
  let typoPenalty = 0;
  let minStart = -1;
  let maxEnd = -1;
  for (let i = 0; i < terms.length; i++) {
    const s: number = matchStarts[i] as number;
    if (s < 0) continue;
    const len: number = terms[i].length;
    const alignedLen: number = matchLengths[i] as number;
    sumBase = (sumBase + 1000 - Math.imul(s, 10)) | 0;
    sumLens = (sumLens + len) | 0;
    typoPenalty = (typoPenalty + Math.imul(distances[i] as number, TYPO_DISTANCE_PENALTY)) | 0;
    // Proximity spans the aligned record windows (typo indel windows differ
    // from raw term lengths by up to `allowed`), matching highlight spans.
    const end: number = (s + (alignedLen > 0 ? alignedLen : len)) | 0;
    if (minStart < 0 || s < minStart) minStart = s;
    if (maxEnd < 0 || end > maxEnd) maxEnd = end;
  }
  let proximity: number = ((maxEnd - minStart - sumLens) | 0) as number;
  if (proximity < 0) proximity = 0;
  const score: number = (sumBase - (strLen - sumLens) - Math.imul(proximity, 2) - typoPenalty) | 0;
  return { matched: true, score, matchStarts, distances, matchLengths, matchedCount };
}
