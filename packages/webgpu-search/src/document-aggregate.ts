/**
 * Pure document aggregation helpers.
 *
 * Extracted from `DocumentIndex.aggregateDocMatches` so ranking can be
 * unit-tested without a GPU device. No DOM / WebGPU / Node references —
 * safe in browser main thread, Web Workers, Node.js, and SSR.
 */

import { compareRanked, isExactTokenMatch, type RankableCandidate } from './ranking';
import type { DocumentId, DocumentSearchResultItem, TieBreakerCriterion } from './types';

/** Per-document best-field accumulator (same shape as the index readback map). */
export interface DocMatchEntry {
  bestScore: number;
  bestFieldIdx: number;
  fieldScores: Map<number, number>;
}

/** Minimal field view needed for ranking (name + weight only). */
export interface AggregateField {
  name: string;
  weight: number;
}

/** Ranked per-document candidate produced by aggregation. */
export interface RankedDocCandidate<TDoc> {
  dIdx: number;
  item: DocumentSearchResultItem<TDoc>;
  rank: RankableCandidate;
}

/** Context required to rank accumulated matches (all plain data, no `this`). */
export interface AggregateDocMatchesContext<TDoc> {
  sortedFields: ReadonlyArray<AggregateField>;
  docToRowIndices: ReadonlyArray<ReadonlyArray<number> | undefined>;
  rowTokens: ReadonlyArray<Uint32Array | undefined>;
  docIds: ReadonlyArray<DocumentId>;
  records: ReadonlyArray<TDoc>;
  normalizedQueryTokens: Uint32Array;
  tieBreakers: readonly TieBreakerCriterion[];
}

/**
 * Fold one weighted field score into the per-document accumulator.
 * Best-field-wins: higher score wins, ties prefer the lower field index
 * (fields are pre-sorted by weight desc, so this is deterministic).
 * Pure apart from mutating the caller-owned map. No scoring change.
 */
export function accumulateDocMatch(
  docMatches: Map<number, DocMatchEntry>,
  dIdx: number,
  fIdx: number,
  weightedScore: number
): void {
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

/**
 * Aggregate per-row matches into per-document best-field hits.
 * Deterministic sort via `compareRanked` + `tieBreakers`.
 * Byte-for-byte port of the former `DocumentIndex.aggregateDocMatches`
 * private method. No behavior change.
 */
export function aggregateDocMatches<TDoc>(
  docMatches: ReadonlyMap<number, DocMatchEntry>,
  ctx: AggregateDocMatchesContext<TDoc>
): RankedDocCandidate<TDoc>[] {
  const {
    sortedFields,
    docToRowIndices,
    rowTokens,
    docIds,
    records,
    normalizedQueryTokens,
    tieBreakers
  } = ctx;
  const hits: RankedDocCandidate<TDoc>[] = [];
  for (const [dIdx, entry] of docMatches.entries()) {
    const primaryField = sortedFields[entry.bestFieldIdx];
    const auxMatches: Array<{ field: string; score: number }> = [];
    for (const [fIdx, score] of entry.fieldScores.entries()) {
      if (fIdx !== entry.bestFieldIdx) {
        auxMatches.push({ field: sortedFields[fIdx].name, score });
      }
    }
    if (auxMatches.length > 1) {
      auxMatches.sort((a, b) => {
        if (b.score !== a.score) return b.score > a.score ? 1 : -1;
        return a.field < b.field ? -1 : (a.field > b.field ? 1 : 0);
      });
    }

    const bestRowIdx = docToRowIndices[dIdx]?.[entry.bestFieldIdx];
    const bestRowTokens = bestRowIdx !== undefined ? rowTokens[bestRowIdx] : undefined;
    hits.push({
      dIdx,
      item: {
        id: docIds[dIdx],
        doc: records[dIdx],
        score: entry.bestScore,
        matchedField: primaryField.name,
        matches: auxMatches.length > 0 ? auxMatches : undefined
      },
      rank: {
        score: entry.bestScore,
        fieldWeight: primaryField.weight,
        isExactMatch: bestRowTokens !== undefined
          ? isExactTokenMatch(bestRowTokens, normalizedQueryTokens)
          : false,
        matchedLength: bestRowTokens !== undefined ? bestRowTokens.length : 0,
        id: docIds[dIdx],
        docIndex: dIdx
      }
    });
  }

  // deterministic ranking: score DESC, weight DESC, exact DESC,
  // length ASC, id ASC (docIndex ASC implicit fallback).
  hits.sort((a, b) => compareRanked(a.rank, b.rank, tieBreakers));
  return hits;
}
