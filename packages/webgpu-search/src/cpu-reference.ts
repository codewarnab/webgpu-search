/**
 * v0.2 CPU parity reference scorer (Issue #7 M2).
 *
 * Mirrors the WGSL integer formulas on post-fold u32 token streams:
 * - substring: 1000 - matchStart*10 - (strLen - queryLen)
 * - fuzzy: subsequence scan with word-boundary/consecutive bonuses,
 *   span + length penalties (see shaders/fuzzy.wgsl).
 *
 * All arithmetic uses i32 semantics (`|0` / `Math.imul`) and the shared
 * two-key order (score desc, index asc; wrap-free).
 * Negative scores are legal (long record + late match); no clamping.
 * Units are post-fold code points on both paths.
 *
 * Portable: no DOM refs. Parity path uses scalar `===` only.
 */

import type { SearchResultItem, SearchMode, TokenMatchOptions, PrefixSearchOptions, TypoToleranceOptions } from './types';
import { clampLimit, nowMs } from './runtime-guards';
import { IncompatibleOptionError } from './errors';
import {
  normalizeTypoTolerance,
  allowedDistanceForTerm,
  findBestTypoWindow,
  TYPO_DISTANCE_PENALTY,
  type NormalizedTypoOptions
} from './modes/typo-distance';
import {
  normalizeTokenMatchOptions,
  splitQueryTerms,
  scoreTokenTokens,
  type NormalizedTokenMatchOptions
} from './modes/token-search';
import {
  normalizePrefixOptions,
  scorePrefixTokens,
  type NormalizedPrefixOptions
} from './modes/prefix-search';

/**
 * Per-query mode options threading token/prefix/typo configuration into
 * the parity scorers. All fields optional; defaults resolve per module.
 * `exactCase` polarity (prefixMatch.exactCase vs index folded mode) is
 * enforced by index callers (document-index.ts, hybrid-index.ts), which
 * own the pack-time polarity — the scorer validates shape only.
 */
export interface CpuModeOptions {
  tokenMatch?: TokenMatchOptions;
  prefixMatch?: PrefixSearchOptions;
  typoTolerance?: TypoToleranceOptions | boolean;
}

interface NormalizedModeOptions {
  tokenOpts: NormalizedTokenMatchOptions;
  prefixOpts: NormalizedPrefixOptions;
  typo: NormalizedTypoOptions;
}

function normalizeModeOptions(raw: CpuModeOptions | undefined): NormalizedModeOptions {
  return {
    tokenOpts: normalizeTokenMatchOptions(raw?.tokenMatch),
    prefixOpts: normalizePrefixOptions(raw?.prefixMatch),
    typo: normalizeTypoTolerance(raw?.typoTolerance)
  };
}

/**
 * Typo-tolerant substring score for one record. With zero allowed edits
 * this delegates to the exact path (bit-identical scores); otherwise the
 * best bounded-DL alignment scores as
 * `1000 - start·10 - (strLen - queryLen) - distance·100` (i32).
 */
export function scoreSubstringTypoTokens(
  record: Uint32Array | readonly number[],
  query: Uint32Array | readonly number[],
  typo: NormalizedTypoOptions
): { matched: boolean; score: number; matchStart: number; distance: number; windowLength: number } {
  const strLen: number = record.length;
  const queryLen: number = query.length;
  if (queryLen === 0 || strLen === 0) {
    return { matched: false, score: 0, matchStart: -1, distance: 0, windowLength: 0 };
  }
  const allowed: 0 | 1 | 2 = allowedDistanceForTerm(queryLen, typo);
  if (allowed === 0) {
    const r = scoreSubstringTokens(record as Uint32Array, query as Uint32Array);
    return { matched: r.matched, score: r.score, matchStart: r.matchStart, distance: 0, windowLength: r.matched ? queryLen : 0 };
  }
  const win = findBestTypoWindow(record, query, allowed, typo.prefixExactLength);
  if (!win.matched || win.distance > allowed) {
    return { matched: false, score: 0, matchStart: -1, distance: 0, windowLength: 0 };
  }
  const score: number =
    ((1000 - Math.imul(win.start, 10) - (strLen - queryLen) - Math.imul(win.distance, TYPO_DISTANCE_PENALTY)) | 0);
  return { matched: true, score, matchStart: win.start, distance: win.distance, windowLength: win.windowLength };
}

/**
 * ASCII delimiter set shared with shaders/fuzzy.wgsl (documented v0.2
 * limitation: tab/newline/NBSP/U+3000 get no word bonus; changing the set
 * is a scoring-version break requiring lockstep WGSL+CPU update).
 */
export const WORD_BOUNDARY_PREV: readonly number[] = Object.freeze([
  47, 95, 45, 46, 32, 58, 92,
] as const) as readonly number[];

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
 * Shared two-key comparator (score desc, index asc).
 * Wrap-free total order; identical in-range results to the old `|0` form.
 * Mirrors the host readback sort on both paths.
 */
export function compareParityResults(
  a: { score: number; index: number },
  b: { score: number; index: number }
): number {
  if (b.score !== a.score) return b.score > a.score ? 1 : -1;
  if (a.index !== b.index) return a.index > b.index ? 1 : -1;
  return 0;
}

/**
 * Earliest-occurrence substring score on token streams.
 * Note: `Math.imul(start, 10)` wraps only past ~214.7M-token start offsets
 * (~859 MB single record) — unreachable in practice; corpus OOMs first.
 * Kept bit-identical with WGSL by design.
 */
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

/**
 * Exhaustive parity scan over pre-normalized record token streams.
 *
 * @param recordTokens post-fold tokens per record (same order as texts).
 * @param queryTokens post-fold query tokens (non-empty; empty yields no hits).
 * @param mode parity algorithm to run.
 * @param limit max results to return (shared clamp 1..8192).
 * @param texts original display strings for the `text` field (scores stay
 *   in normalized space; callers must not slice originals by folded spans).
 * Precondition: `recordTokens`/`texts` are parallel arrays of valid
 * `Uint32Array`/string entries (sparse/undefined entries throw TypeError).
 */
export function searchCpuReference(
  recordTokens: readonly Uint32Array[],
  queryTokens: Uint32Array,
  mode: SearchMode,
  limit: number,
  texts: readonly string[],
  modeOptions?: CpuModeOptions
): CpuReferenceOutput {
  const t0: number = nowMs();
  if (mode !== 'substring' && mode !== 'fuzzy' && mode !== 'token' && mode !== 'prefix') {
    throw new IncompatibleOptionError(
      'mode',
      `CPU reference encountered unknown mode '${String(mode)}'. Expected 'fuzzy', 'substring', 'token', or 'prefix'.`
    );
  }
  // Fail-closed option validation before scanning (invalid token/prefix/
  // typo shapes throw even when the corpus is empty).
  const opts: NormalizedModeOptions = normalizeModeOptions(modeOptions);
  // 'fuzzy' is inherently typo-tolerant via subsequence matching; typo
  // options are validated but do not alter fuzzy scoring (documented).
  const queryTerms: Uint32Array[] | null =
    mode === 'token' ? splitQueryTerms(queryTokens) : null;
  const hits: SearchResultItem[] = [];
  if (queryTokens.length !== 0 && (queryTerms === null || queryTerms.length !== 0)) {
    for (let idx = 0; idx < recordTokens.length; idx++) {
      const rec: Uint32Array = recordTokens[idx] as Uint32Array;
      if (mode === 'substring') {
        const r = scoreSubstringTypoTokens(rec, queryTokens, opts.typo);
        if (r.matched) {
          hits.push({ index: idx, score: r.score, text: texts[idx] ?? '' });
        }
      } else if (mode === 'fuzzy') {
        const r = scoreFuzzyTokens(rec, queryTokens);
        if (r.matched) {
          hits.push({ index: idx, score: r.score, text: texts[idx] ?? '' });
        }
      } else if (mode === 'token') {
        const r = scoreTokenTokens(rec, queryTerms as Uint32Array[], opts.tokenOpts, opts.typo);
        if (r.matched) {
          hits.push({ index: idx, score: r.score, text: texts[idx] ?? '' });
        }
      } else {
        const r = scorePrefixTokens(rec, queryTokens, opts.prefixOpts, opts.typo);
        if (r.matched) {
          hits.push({ index: idx, score: r.score, text: texts[idx] ?? '' });
        }
      }
    }
  }
  const totalMatches: number = hits.length;
  hits.sort(compareParityResults);
  // Shared clamp (1..8192, NaN/non-finite → 50, fractions floored) — same
  // helper as hybrid-index so direct callers cannot bypass the cap.
  const capped: number = clampLimit(limit);
  const results: SearchResultItem[] =
    hits.length > capped ? hits.slice(0, capped) : hits;
  const durationMs: number = nowMs() - t0;
  return { totalMatches, results, durationMs };
}

export interface MultiFieldMatch {
  field: string;
  score: number;
}

export interface MultiFieldHit {
  docIndex: number;
  score: number;
  matchedField: string;
  matches?: MultiFieldMatch[];
}

export interface MultiFieldCpuReferenceOutput {
  totalMatches: number;
  candidateCount: number;
  hasOverflow: boolean;
  results: MultiFieldHit[];
  durationMs: number;
  /**
   * Full ranked query-matched doc indices (post-filter when `filterDoc` is
   * set, unfiltered otherwise), unaffected by `limit` truncation. Used by
   * facet aggregation for exact bucket counts over the entire match set.
   * Optional for backward compat: hand-constructed mocks may omit it;
   * consumers must fall back to `results.map(r => r.docIndex)`. Omitted
   * (or empty) when `collectAllMatched` is false to avoid taxing non-facet
   * searches with an extra O(n) pass.
   */
  allMatchedDocIndices?: number[];
}

export interface FieldScoreDefinition {
  name: string;
  weight: number;
}

/**
 * Multi-field parity reference scorer.
 * Evaluates records across multiple fields, applies fixed-point integer weighting,
 * aggregates matching fields per document, and returns ranked document hits.
 * v0.4 M4: token `minMatchCount` quorums evaluate per field-row (best-row-wins
 * per document); cross-row term coverage does not combine.
 */
export function searchMultiFieldCpuReference(
  docCount: number,
  fields: readonly FieldScoreDefinition[],
  rowTokens: readonly Uint32Array[],
  rowToDocIndex: readonly number[],
  rowToFieldIndex: readonly number[],
  queryTokens: Uint32Array,
  mode: SearchMode,
  limit: number,
  candidateCapacity: number = 8192,
  allowedFieldIndices?: ReadonlySet<number>,
  tombstonedRows?: ReadonlySet<number>,
  filterDoc?: (docIndex: number) => boolean,
  collectAllMatched: boolean = true,
  modeOptions?: CpuModeOptions
): MultiFieldCpuReferenceOutput {
  const t0: number = nowMs();
  if (mode !== 'substring' && mode !== 'fuzzy' && mode !== 'token' && mode !== 'prefix') {
    throw new IncompatibleOptionError(
      'mode',
      `CPU reference encountered unknown mode '${String(mode)}'. Expected 'fuzzy', 'substring', 'token', or 'prefix'.`
    );
  }
  const opts: NormalizedModeOptions = normalizeModeOptions(modeOptions);
  const queryTerms: Uint32Array[] | null =
    mode === 'token' ? splitQueryTerms(queryTokens) : null;
  if (queryTokens.length === 0 || docCount === 0 || rowTokens.length === 0 ||
    (queryTerms !== null && queryTerms.length === 0)) {
    return {
      totalMatches: 0,
      candidateCount: 0,
      hasOverflow: false,
      results: [],
      durationMs: nowMs() - t0,
      allMatchedDocIndices: []
    };
  }

  // Aggregate field matches per document:
  // docIndex -> { bestScore, bestFieldIdx, fieldScores: Map<fieldIdx, weightedScore> }
  const docMatches = new Map<number, {
    bestScore: number;
    bestFieldIdx: number;
    fieldScores: Map<number, number>;
  }>();

  for (let r = 0; r < rowTokens.length; r++) {
    if (tombstonedRows && tombstonedRows.size > 0 && tombstonedRows.has(r)) continue;
    const fIdx = rowToFieldIndex[r];
    if (allowedFieldIndices && !allowedFieldIndices.has(fIdx)) continue;
    const dIdx = rowToDocIndex[r];
    if (filterDoc && !filterDoc(dIdx)) continue;

    const rec = rowTokens[r];
    let matched = false;
    let rawScore = 0;
    if (mode === 'substring') {
      const scored = scoreSubstringTypoTokens(rec, queryTokens, opts.typo);
      matched = scored.matched;
      rawScore = scored.score;
    } else if (mode === 'fuzzy') {
      const scored = scoreFuzzyTokens(rec, queryTokens);
      matched = scored.matched;
      rawScore = scored.score;
    } else if (mode === 'token') {
      const scored = scoreTokenTokens(rec, queryTerms as Uint32Array[], opts.tokenOpts, opts.typo);
      matched = scored.matched;
      rawScore = scored.score;
    } else {
      const scored = scorePrefixTokens(rec, queryTokens, opts.prefixOpts, opts.typo);
      matched = scored.matched;
      rawScore = scored.score;
    }

    if (matched) {
      const fDef = fields[fIdx];
      const weightedScore = Math.round(rawScore * fDef.weight);
      let entry = docMatches.get(dIdx);
      if (!entry) {
        entry = {
          bestScore: weightedScore,
          bestFieldIdx: fIdx,
          fieldScores: new Map<number, number>()
        };
        entry.fieldScores.set(fIdx, weightedScore);
        docMatches.set(dIdx, entry);
      } else {
        entry.fieldScores.set(fIdx, weightedScore);
        if (
          weightedScore > entry.bestScore ||
          (weightedScore === entry.bestScore && fIdx < entry.bestFieldIdx)
        ) {
          entry.bestScore = weightedScore;
          entry.bestFieldIdx = fIdx;
        }
      }
    }
  }

  const hits: MultiFieldHit[] = [];
  for (const [dIdx, entry] of docMatches.entries()) {
    const primaryField = fields[entry.bestFieldIdx];
    const auxMatches: MultiFieldMatch[] = [];
    for (const [fIdx, score] of entry.fieldScores.entries()) {
      if (fIdx !== entry.bestFieldIdx) {
        auxMatches.push({ field: fields[fIdx].name, score });
      }
    }
    if (auxMatches.length > 1) {
      auxMatches.sort((a, b) => {
        if (b.score !== a.score) return b.score - a.score;
        return a.field < b.field ? -1 : (a.field > b.field ? 1 : 0);
      });
    }

    hits.push({
      docIndex: dIdx,
      score: entry.bestScore,
      matchedField: primaryField.name,
      matches: auxMatches.length > 0 ? auxMatches : undefined
    });
  }

  const totalMatches = hits.length;
  // Two-key sort: score descending, docIndex ascending
  hits.sort((a, b) => {
    if (b.score !== a.score) return b.score > a.score ? 1 : -1;
    if (a.docIndex !== b.docIndex) return a.docIndex > b.docIndex ? 1 : -1;
    return 0;
  });

  const capped = clampLimit(limit);
  const results = hits.length > capped ? hits.slice(0, capped) : hits;
  const durationMs = nowMs() - t0;
  const candidateCount = Math.min(totalMatches, candidateCapacity);
  const hasOverflow = totalMatches > candidateCapacity;

  return {
    totalMatches,
    candidateCount,
    hasOverflow,
    results,
    durationMs,
    // Gated: non-facet searches pass collectAllMatched=false to skip O(n) map.
    ...(collectAllMatched ? { allMatchedDocIndices: hits.map((h) => h.docIndex) } : {}),
  };
}

