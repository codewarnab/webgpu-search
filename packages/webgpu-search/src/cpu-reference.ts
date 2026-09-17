/**
 * v0.2 CPU parity reference scorer (Issue #7 M2).
 *
 * Mirrors the WGSL integer formulas on post-fold u32 token streams:
 * - substring: 1000 - matchStart*10 - (strLen - queryLen)
 * - fuzzy: subsequence scan with word-boundary/consecutive bonuses,
 *   span + length penalties (see shaders/fuzzy.wgsl).
 *
 * All arithmetic uses i32 semantics (`|0` / `Math.imul`) and the shared
 * two-key order `(b.score - a.score) || (a.index - b.index)`.
 * Negative scores are legal (long record + late match); no clamping.
 * Units are post-fold code points on both paths.
 *
 * Portable: no DOM refs. Parity path uses scalar `===` only.
 */

import type { SearchResultItem } from './types';

/** ASCII delimiter set from shaders/fuzzy.wgsl (documented v0.2 limit). */
function isWordBoundaryPrev(prev: number): boolean {
  return (
    prev === 47 ||
    prev === 95 ||
    prev === 45 ||
    prev === 46 ||
    prev === 32 ||
    prev === 58 ||
    prev === 92
  );
}

/**
 * Shared two-key comparator with i32 semantics.
 * Mirrors the host readback sort on both paths.
 */
export function compareParityResults(
  a: { score: number; index: number },
  b: { score: number; index: number }
): number {
  const byScore: number = ((b.score - a.score) | 0) as number;
  if (byScore !== 0) return byScore;
  return ((a.index - b.index) | 0) as number;
}

/** Earliest-occurrence substring score on token streams. */
export function scoreSubstringTokens(
  record: Uint32Array,
  query: Uint32Array
): { matched: boolean; score: number; matchStart: number } {
  const strLen: number = record.length;
  const queryLen: number = query.length;
  if (queryLen === 0 || strLen < queryLen) {
    return { matched: false, score: 0, matchStart: -1 };
  }
  const maxStart: number = strLen - queryLen;
  for (let start = 0; start <= maxStart; start++) {
    let ok = true;
    for (let j = 0; j < queryLen; j++) {
      if (record[start + j] !== query[j]) {
        ok = false;
        break;
      }
    }
    if (ok) {
      const score: number = (1000 - Math.imul(start, 10) - (strLen - queryLen)) | 0;
      return { matched: true, score, matchStart: start };
    }
  }
  return { matched: false, score: 0, matchStart: -1 };
}

/** Ordered-subsequence fuzzy score on token streams. */
export function scoreFuzzyTokens(
  record: Uint32Array,
  query: Uint32Array
): { matched: boolean; score: number } {
  const strLen: number = record.length;
  const queryLen: number = query.length;
  if (queryLen === 0 || strLen < queryLen) {
    return { matched: false, score: 0 };
  }
  let score: number = 100 | 0;
  let consecutive: number = 0 | 0;
  let first: number = -1;
  let last: number = 0;
  let q = 0;
  for (let i = 0; i < strLen; i++) {
    const sc: number = record[i] as number;
    const qc: number = query[q] as number;
    if (sc === qc) {
      if (first < 0) {
        first = i;
        if (i === 0) {
          score = (score + 40) | 0;
        }
      }
      last = i;
      if (i > 0) {
        const prev: number = record[i - 1] as number;
        if (isWordBoundaryPrev(prev)) {
          score = (score + 30) | 0;
        }
      }
      score = (score + 15 + Math.imul(consecutive, 10)) | 0;
      consecutive = (consecutive + 1) | 0;
      q++;
      if (q === queryLen) break;
    } else {
      consecutive = 0;
    }
  }
  if (q !== queryLen) {
    return { matched: false, score: 0 };
  }
  const span: number = ((last - first + 1) | 0) as number;
  score = (score - Math.imul(span - queryLen, 2)) | 0;
  score = (score - (strLen - queryLen)) | 0;
  return { matched: true, score };
}

export interface CpuReferenceOutput {
  totalMatches: number;
  results: SearchResultItem[];
  durationMs: number;
}

function nowMs(): number {
  return performance.now();
}

/**
 * Exhaustive parity scan over pre-normalized record token streams.
 *
 * @param recordTokens post-fold tokens per record (same order as texts).
 * @param queryTokens post-fold query tokens (non-empty; empty yields no hits).
 * @param mode parity algorithm to run.
 * @param limit max results to return (caller clamps to 1..8192).
 * @param texts original display strings for the `text` field (scores stay
 *   in normalized space; callers must not slice originals by folded spans).
 */
export function searchCpuReference(
  recordTokens: readonly Uint32Array[],
  queryTokens: Uint32Array,
  mode: 'fuzzy' | 'substring',
  limit: number,
  texts: readonly string[]
): CpuReferenceOutput {
  const t0: number = nowMs();
  const hits: SearchResultItem[] = [];
  if (queryTokens.length !== 0) {
    for (let idx = 0; idx < recordTokens.length; idx++) {
      const rec: Uint32Array = recordTokens[idx] as Uint32Array;
      if (mode === 'substring') {
        const r = scoreSubstringTokens(rec, queryTokens);
        if (r.matched) {
          hits.push({ index: idx, score: r.score, text: texts[idx] ?? '' });
        }
      } else {
        const r = scoreFuzzyTokens(rec, queryTokens);
        if (r.matched) {
          hits.push({ index: idx, score: r.score, text: texts[idx] ?? '' });
        }
      }
    }
  }
  const totalMatches: number = hits.length;
  hits.sort(compareParityResults);
  // Mirror hybrid-index clamping: NaN → default 50, then 1..8192 (caller
  // clamps; this keeps direct callers symmetric).
  const saneLimit: number = Number.isNaN(limit) ? 50 : limit;
  const capped: number = saneLimit < 1 ? 1 : saneLimit;
  const results: SearchResultItem[] =
    hits.length > capped ? hits.slice(0, capped) : hits;
  const durationMs: number = nowMs() - t0;
  return { totalMatches, results, durationMs };
}
