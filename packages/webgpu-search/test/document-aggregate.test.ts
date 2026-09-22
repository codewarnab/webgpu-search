/**
 * Pure aggregation unit tests (no GPU / no device required).
 *
 * Covers `accumulateDocMatch` + `aggregateDocMatches` extracted from
 * DocumentIndex: best-field-wins, aux-match sorting, exact-match keys,
 * and the deterministic five-tier ranking.
 *
 * Run: bun test packages/webgpu-search/test/document-aggregate.test.ts
 */
import { describe, test, expect } from 'bun:test';
import {
  aggregateDocMatches,
  accumulateDocMatch,
  type DocMatchEntry,
  type AggregateDocMatchesContext,
} from '../src/document-aggregate';

type Doc = { title: string };

function ctx(over: Partial<AggregateDocMatchesContext<Doc>> = {}): AggregateDocMatchesContext<Doc> {
  return {
    sortedFields: [
      { name: 'title', weight: 2 },
      { name: 'body', weight: 1 },
    ],
    docToRowIndices: [[0, 1], [2, 3]],
    rowTokens: [new Uint32Array([1, 2]), new Uint32Array([3]), new Uint32Array([1, 2]), new Uint32Array([9])],
    docIds: ['a', 'b'],
    records: [{ title: 'a' }, { title: 'b' }],
    normalizedQueryTokens: new Uint32Array([1, 2]),
    tieBreakers: ['score', 'weight', 'exact', 'length', 'id'],
    ...over,
  };
}

describe('accumulateDocMatch', () => {
  test('creates entries and keeps highest score', () => {
    const m = new Map<number, DocMatchEntry>();
    accumulateDocMatch(m, 0, 1, 10);
    accumulateDocMatch(m, 0, 0, 20);
    expect(m.get(0)?.bestScore).toBe(20);
    expect(m.get(0)?.bestFieldIdx).toBe(0);
    expect(m.get(0)?.fieldScores.get(1)).toBe(10);
    // lower score must not dethrone the best field
    accumulateDocMatch(m, 0, 1, 5);
    expect(m.get(0)?.bestFieldIdx).toBe(0);
    expect(m.get(0)?.fieldScores.get(1)).toBe(5);
  });

  test('ties prefer the lower field index', () => {
    const m = new Map<number, DocMatchEntry>();
    accumulateDocMatch(m, 0, 1, 10);
    accumulateDocMatch(m, 0, 0, 10);
    expect(m.get(0)?.bestFieldIdx).toBe(0);
  });
});

describe('aggregateDocMatches', () => {
  test('empty map yields no hits', () => {
    expect(aggregateDocMatches(new Map(), ctx())).toEqual([]);
  });

  test('selects best field and exposes aux matches', () => {
    const m = new Map<number, DocMatchEntry>([
      [0, { bestScore: 20, bestFieldIdx: 0, fieldScores: new Map([[0, 20], [1, 5]]) }],
    ]);
    const hits = aggregateDocMatches(m, ctx());
    expect(hits).toHaveLength(1);
    expect(hits[0].dIdx).toBe(0);
    expect(hits[0].item.matchedField).toBe('title');
    expect(hits[0].item.score).toBe(20);
    expect(hits[0].item.matches).toEqual([{ field: 'body', score: 5 }]);
    expect(hits[0].rank.fieldWeight).toBe(2);
    expect(hits[0].rank.isExactMatch).toBe(true);
    expect(hits[0].rank.matchedLength).toBe(2);
  });

  test('omits matches when only one field scored', () => {
    const m = new Map<number, DocMatchEntry>([
      [1, { bestScore: 7, bestFieldIdx: 1, fieldScores: new Map([[1, 7]]) }],
    ]);
    const hits = aggregateDocMatches(m, ctx());
    expect(hits[0].item.matches).toBeUndefined();
    expect(hits[0].rank.isExactMatch).toBe(false);
  });

  test('sorts aux matches by score desc then field asc', () => {
    const c = ctx({
      sortedFields: [
        { name: 'a', weight: 3 },
        { name: 'b', weight: 2 },
        { name: 'c', weight: 1 },
      ],
      docToRowIndices: [[0, 1, 2]],
      rowTokens: [new Uint32Array([1]), new Uint32Array([1]), new Uint32Array([1])],
      docIds: ['x'],
      records: [{ title: 'x' }],
    });
    const m = new Map<number, DocMatchEntry>([
      [0, { bestScore: 10, bestFieldIdx: 0, fieldScores: new Map([[0, 10], [1, 10], [2, 10]]) }],
    ]);
    const hits = aggregateDocMatches(m, c);
    // b before c on field-name tiebreak (equal scores, best excluded)
    expect(hits[0].item.matches).toEqual([
      { field: 'b', score: 10 },
      { field: 'c', score: 10 },
    ]);
  });

  test('deterministic ranking: score desc, then weight, then id', () => {
    const c = ctx({
      docToRowIndices: [[0, 1], [0, 1]],
      rowTokens: [new Uint32Array([9]), new Uint32Array([9])],
      docIds: ['b', 'a'],
      records: [{ title: 'b' }, { title: 'a' }],
      normalizedQueryTokens: new Uint32Array([1]),
    });
    const m = new Map<number, DocMatchEntry>([
      [0, { bestScore: 5, bestFieldIdx: 0, fieldScores: new Map([[0, 5]]) }],
      [1, { bestScore: 5, bestFieldIdx: 0, fieldScores: new Map([[0, 5]]) }],
    ]);
    const hits = aggregateDocMatches(m, c);
    // equal score/weight/exact/length -> id asc ('a' = dIdx 1 first)
    expect(hits.map((h) => h.dIdx)).toEqual([1, 0]);
  });

  test('weight breaks score ties', () => {
    const m = new Map<number, DocMatchEntry>([
      [0, { bestScore: 5, bestFieldIdx: 1, fieldScores: new Map([[1, 5]]) }],
      [1, { bestScore: 5, bestFieldIdx: 0, fieldScores: new Map([[0, 5]]) }],
    ]);
    const hits = aggregateDocMatches(m, ctx());
    expect(hits[0].dIdx).toBe(1);
  });
});
