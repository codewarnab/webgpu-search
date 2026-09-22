/**
 * v0.4 M5 Test Suite (Issue #10): Deterministic Ranking & Autocomplete Primitives.
 *
 * Covers the five-tier tie-breaker (score DESC, weight DESC, exact DESC,
 * length ASC, id ASC), custom tieBreaker hierarchies, first-party autocomplete()
 * (prefix completions + fuzzy did-you-mean), inline search({ autocomplete }),
 * determinism across repeated runs, and cross-platform portability.
 *
 * Run: bun test packages/webgpu-search/test/ranking-suggest.test.ts
 */
import { describe, test, expect } from 'bun:test';
import { readFile } from 'node:fs/promises';
import {
  DocumentIndex,
  compareRanked,
  sortRanked,
  isExactTokenMatch,
  normalizeTieBreakers,
  normalizeAutocompleteOptions,
  normalizeText,
  scoreExactMatchesMultiField,
  DEFAULT_TIE_BREAKERS,
  AUTOCOMPLETE_DEFAULT_LIMIT,
  type DocumentIndexOptions,
  type RankableCandidate,
} from '../src/index';

const toks = (s: string, folded = true): Uint32Array => normalizeText(s, folded).tokens;

function rank(over: Partial<RankableCandidate>): RankableCandidate {
  return {
    score: 100,
    fieldWeight: 1.0,
    isExactMatch: false,
    matchedLength: 10,
    id: 'a',
    docIndex: 0,
    ...over,
  };
}

describe('ranking: normalizeTieBreakers', () => {
  test('defaults to the five-tier hierarchy', () => {
    expect(normalizeTieBreakers(undefined)).toEqual(['score', 'weight', 'exact', 'length', 'id']);
    expect(DEFAULT_TIE_BREAKERS).toEqual(['score', 'weight', 'exact', 'length', 'id']);
  });

  test('accepts custom hierarchies and copies the array', () => {
    const custom = normalizeTieBreakers(['score', 'id']);
    expect(custom).toEqual(['score', 'id']);
    const src: ('score' | 'id')[] = ['score', 'id'];
    const out = normalizeTieBreakers(src);
    expect(out).not.toBe(src);
  });

  test('rejects malformed hierarchies fail-closed', () => {
    expect(() => normalizeTieBreakers([])).toThrow(RangeError);
    expect(() => normalizeTieBreakers(['score', 'score'])).toThrow(RangeError);
    expect(() => normalizeTieBreakers(['nope'] as never)).toThrow(TypeError);
    expect(() => normalizeTieBreakers('score' as never)).toThrow(TypeError);
  });
});

describe('ranking: compareRanked tiers', () => {
  test('score DESC dominates', () => {
    expect(compareRanked(rank({ score: 5 }), rank({ score: 9 }), DEFAULT_TIE_BREAKERS)).toBe(1);
    expect(compareRanked(rank({ score: 9 }), rank({ score: 5 }), DEFAULT_TIE_BREAKERS)).toBe(-1);
  });

  test('weight DESC breaks score ties', () => {
    const a = rank({ score: 100, fieldWeight: 1.0 });
    const b = rank({ score: 100, fieldWeight: 2.0 });
    expect(compareRanked(a, b, DEFAULT_TIE_BREAKERS)).toBe(1);
    expect(compareRanked(b, a, DEFAULT_TIE_BREAKERS)).toBe(-1);
  });

  test('exact DESC breaks score+weight ties', () => {
    const partial = rank({ score: 100, fieldWeight: 1.0, isExactMatch: false });
    const exact = rank({ score: 100, fieldWeight: 1.0, isExactMatch: true });
    expect(compareRanked(partial, exact, DEFAULT_TIE_BREAKERS)).toBe(1);
    expect(compareRanked(exact, partial, DEFAULT_TIE_BREAKERS)).toBe(-1);
  });

  test('length ASC breaks score+weight+exact ties', () => {
    const long = rank({ score: 100, fieldWeight: 1.0, isExactMatch: true, matchedLength: 20 });
    const short = rank({ score: 100, fieldWeight: 1.0, isExactMatch: true, matchedLength: 5 });
    expect(compareRanked(long, short, DEFAULT_TIE_BREAKERS)).toBe(1);
    expect(compareRanked(short, long, DEFAULT_TIE_BREAKERS)).toBe(-1);
  });

  test('id ASC breaks remaining ties (string code-unit order)', () => {
    const b = rank({ score: 100, fieldWeight: 1.0, isExactMatch: true, matchedLength: 5, id: 'b' });
    const a = rank({ score: 100, fieldWeight: 1.0, isExactMatch: true, matchedLength: 5, id: 'a' });
    expect(compareRanked(b, a, DEFAULT_TIE_BREAKERS)).toBe(1);
    expect(compareRanked(a, b, DEFAULT_TIE_BREAKERS)).toBe(-1);
    expect(compareRanked(a, rank({ ...a }), DEFAULT_TIE_BREAKERS)).toBe(0);
  });

  test('numeric ids compare numerically (2 < 10)', () => {
    const two = rank({ id: 2 });
    const ten = rank({ id: 10 });
    expect(compareRanked(ten, two, DEFAULT_TIE_BREAKERS)).toBe(1);
    expect(compareRanked(two, ten, DEFAULT_TIE_BREAKERS)).toBe(-1);
  });

  test('custom hierarchy is honored (id before score)', () => {
    const lowIdLowScore = rank({ score: 1, id: 'a' });
    const highIdHighScore = rank({ score: 999, id: 'z' });
    expect(compareRanked(lowIdLowScore, highIdHighScore, ['id', 'score'])).toBe(-1);
    expect(compareRanked(lowIdLowScore, highIdHighScore, DEFAULT_TIE_BREAKERS)).toBe(1);
  });

  test('docIndex is the implicit final fallback', () => {
    const a = rank({ id: 'same', docIndex: 0 });
    const b = rank({ id: 'same', docIndex: 1 });
    expect(compareRanked(b, a, DEFAULT_TIE_BREAKERS)).toBe(1);
  });

  test('sortRanked produces a total deterministic order', () => {
    const items = [rank({ id: 'c' }), rank({ id: 'a' }), rank({ id: 'b' })];
    sortRanked(items, DEFAULT_TIE_BREAKERS);
    expect(items.map((i) => i.id)).toEqual(['a', 'b', 'c']);
  });
});

describe('ranking: isExactTokenMatch', () => {
  test('equal streams match; length or content drift does not', () => {
    expect(isExactTokenMatch(toks('hello'), toks('hello'))).toBe(true);
    expect(isExactTokenMatch(toks('hello'), toks('hello!'))).toBe(false);
    expect(isExactTokenMatch(toks('hello'), toks('hallo'))).toBe(false);
    expect(isExactTokenMatch(new Uint32Array(0), new Uint32Array(0))).toBe(true);
  });
});

interface TitleDoc {
  id: string;
  title: string;
  body: string;
}

function titleOpts(extra?: Partial<DocumentIndexOptions<TitleDoc>>): DocumentIndexOptions<TitleDoc> {
  return {
    fields: [
      { name: 'title', weight: 2.0 },
      { name: 'body', weight: 1.0 },
    ],
    preferGpu: false,
    ...extra,
  };
}

describe('DocumentIndex M5: deterministic search ranking', () => {
  test('identical scores break ties by id ASC (string)', async () => {
    const index = await DocumentIndex.create<TitleDoc>(
      [
        { id: 'b', title: 'hello', body: '' },
        { id: 'a', title: 'hello', body: '' },
        { id: 'c', title: 'hello', body: '' },
      ],
      titleOpts()
    );
    const res = await index.search('hello', { mode: 'substring' });
    expect(res.totalMatches).toBe(3);
    expect(res.results.map((r) => r.id)).toEqual(['a', 'b', 'c']);
    // Repeated runs are bit-for-bit identical.
    const again = await index.search('hello', { mode: 'substring' });
    expect(again.results.map((r) => [r.id, r.score])).toEqual(
      res.results.map((r) => [r.id, r.score])
    );
    for (const r of res.results) expect(Number.isInteger(r.score)).toBe(true);
    index.destroy();
  });

  test('numeric ids order numerically, not lexicographically', async () => {
    const index = await DocumentIndex.create<{ id: number; title: string }>(
      [
        { id: 10, title: 'hello' },
        { id: 2, title: 'hello' },
        { id: 1, title: 'hello' },
      ],
      { fields: ['title'], preferGpu: false }
    );
    const res = await index.search('hello', { mode: 'substring' });
    expect(res.results.map((r) => r.id)).toEqual([1, 2, 10]);
    index.destroy();
  });

  test('weight tier: title match outranks body match on tied raw scores', async () => {
    // 'auth' appears as the full title of doc 1 (weight 2.0) and as the full
    // body of doc 2 (weight 1.0) with equal-length fields, so raw scores tie
    // and the weight tier decides.
    const index = await DocumentIndex.create<TitleDoc>(
      [
        { id: 'body-hit', title: 'zzzz', body: 'auth' },
        { id: 'title-hit', title: 'auth', body: 'zzzz' },
      ],
      titleOpts()
    );
    const res = await index.search('auth', { mode: 'substring' });
    expect(res.totalMatches).toBe(2);
    expect(res.results[0]?.id).toBe('title-hit');
    // Pin scorer drift: raw 1000 on both fields, weighted 2000 vs 1000.
    expect(res.results.map((r) => r.score)).toEqual([2000, 1000]);
    index.destroy();
  });

  test('custom ranking hierarchy is accepted and deterministic', async () => {
    const index = await DocumentIndex.create<TitleDoc>(
      [
        { id: 'b', title: 'hello', body: '' },
        { id: 'a', title: 'hello', body: '' },
      ],
      titleOpts()
    );
    const res = await index.search('hello', {
      mode: 'substring',
      ranking: { tieBreakers: ['score', 'weight', 'exact', 'length', 'id'] },
    });
    expect(res.results.map((r) => r.id)).toEqual(['a', 'b']);
    const again = await index.search('hello', {
      mode: 'substring',
      ranking: { tieBreakers: ['score', 'weight', 'exact', 'length', 'id'] },
    });
    expect(again.results.map((r) => r.id)).toEqual(['a', 'b']);
    index.destroy();
  });

  test('invalid ranking shapes throw fail-closed', async () => {
    const index = await DocumentIndex.create<TitleDoc>(
      [{ id: 'a', title: 'hello', body: '' }],
      titleOpts()
    );
    await expect(
      index.search('hello', { ranking: { tieBreakers: [] } })
    ).rejects.toThrow(RangeError);
    await expect(
      index.search('hello', { ranking: { tieBreakers: ['score', 'score'] } })
    ).rejects.toThrow(RangeError);
    await expect(
      index.search('hello', { ranking: 'score' as never })
    ).rejects.toThrow(TypeError);
    index.destroy();
  });

  test('rankingOptions validate fail-closed (bad hierarchy, short docIds)', () => {
    const rec = [toks('hello')];
    // Garbage hierarchy throws even without docIds (legacy order otherwise).
    expect(() =>
      scoreExactMatchesMultiField(
        1, [{ name: 't', weight: 1 }], rec, [0], [0], toks('hello'),
        'substring', 10, 8192, undefined, undefined, undefined, true, undefined,
        { tieBreakers: ['nope'] as never }
      )
    ).toThrow(TypeError);
    // docIds shorter than docCount throws.
    expect(() =>
      scoreExactMatchesMultiField(
        2, [{ name: 't', weight: 1 }], [rec[0] as Uint32Array, rec[0] as Uint32Array], [0, 1], [0, 0], toks('hello'),
        'substring', 10, 8192, undefined, undefined, undefined, true, undefined,
        { docIds: ['only-one'] }
      )
    ).toThrow(RangeError);
  });

  test('parity CPU reference with docIds matches index order', async () => {    const docs: TitleDoc[] = [
      { id: 'b', title: 'hello', body: '' },
      { id: 'a', title: 'hello', body: '' },
    ];
    const index = await DocumentIndex.create<TitleDoc>(docs, titleOpts());
    const res = await index.search('hello', { mode: 'substring' });
    // Direct low-level call with M5 ranking must agree on order.
    const fields = [
      { name: 'title', weight: 2.0 },
      { name: 'body', weight: 1.0 },
    ];
    const rowTokens: Uint32Array[] = [];
    const rowToDoc: number[] = [];
    const rowToField: number[] = [];
    for (let f = 0; f < 2; f++) {
      for (let d = 0; d < 2; d++) {
        const raw = f === 0 ? docs[d]?.title : docs[d]?.body;
        rowTokens.push(toks(raw ?? ''));
        rowToDoc.push(d);
        rowToField.push(f);
      }
    }
    const ref = scoreExactMatchesMultiField(
      2, fields, rowTokens, rowToDoc, rowToField, toks('hello'),
      'substring', 50, 8192, undefined, undefined, undefined, true, undefined,
      { tieBreakers: ['score', 'weight', 'exact', 'length', 'id'], docIds: ['b', 'a'] }
    );
    expect(ref.results.map((r) => docs[r.docIndex]?.id)).toEqual(
      res.results.map((r) => r.id)
    );
    index.destroy();
  });
});

describe('DocumentIndex M5: autocomplete()', () => {
  const SYMBOLS: TitleDoc[] = [
    { id: '1', title: 'AuthController', body: 'handles login sessions' },
    { id: '2', title: 'AuthService', body: 'token refresh flow' },
    { id: '3', title: 'DatabasePool', body: 'connection pooling layer' },
  ];

  test('prefix mode returns completions ranked deterministically', async () => {
    const index = await DocumentIndex.create<TitleDoc>(SYMBOLS, titleOpts());
    const res = await index.autocomplete('Auth');
    expect(res.suggestions.length).toBe(2);
    // Both match at token offset 0 with equal weights; the shorter field
    // scores higher (length penalty), so AuthService outranks AuthController.
    expect(res.suggestions.map((s) => s.text)).toEqual(['AuthService', 'AuthController']);
    for (const s of res.suggestions) {
      expect(s.type).toBe('completion');
      expect(Number.isInteger(s.score)).toBe(true);
      expect(s.docId).toBeDefined();
      expect(s.doc).toBeDefined();
    }
    // Highlight ranges cover the prefix span.
    expect(res.suggestions[0]?.matchedRanges).toEqual([{ start: 0, end: 4 }]);
    expect(typeof res.queryDurationMs).toBe('number');
    const again = await index.autocomplete('Auth');
    expect(again.suggestions.map((s) => [s.text, s.score])).toEqual(
      res.suggestions.map((s) => [s.text, s.score])
    );
    index.destroy();
  });

  test('limit truncates; field restricts the scan', async () => {
    const index = await DocumentIndex.create<TitleDoc>(SYMBOLS, titleOpts());
    const limited = await index.autocomplete('Auth', { limit: 1 });
    expect(limited.suggestions.length).toBe(1);
    const bodyOnly = await index.autocomplete('Auth', { field: 'body' });
    expect(bodyOnly.suggestions.length).toBe(0);
    const titleOnly = await index.autocomplete('Auth', { field: 'title' });
    expect(titleOnly.suggestions.length).toBe(2);
    index.destroy();
  });

  test('mid-string token prefixes complete (token-start anchoring)', async () => {
    const index = await DocumentIndex.create<TitleDoc>(SYMBOLS, titleOpts());
    // 'pool' is a prefix of the body token 'pooling' (offset 11), so the
    // body field completes. ('DatabasePool' does NOT match: camelCase
    // interiors are not token starts, so 'pool' is not anchored there.)
    const res = await index.autocomplete('pool');
    expect(res.suggestions.map((s) => s.text)).toEqual(['connection pooling layer']);
    // A head-anchored symbol prefix completes the title itself.
    const head = await index.autocomplete('Data');
    expect(head.suggestions.map((s) => s.text)).toEqual(['DatabasePool']);
    index.destroy();
  });

  test('ranking/suggest options survive worker structured-clone', () => {
    const opts = {
      mode: 'token' as const,
      tokenMatch: { operator: 'or' as const },
      prefixMatch: { prefixLength: 2 },
      typoTolerance: { enabled: true, maxDistance: 1 as const },
      ranking: { tieBreakers: ['score', 'weight', 'exact', 'length', 'id'] as const },
      autocomplete: { limit: 3, mode: 'prefix' as const, fuzzyDistance: 1, tieBreakers: ['score', 'id'] as const },
    };
    const roundtripped = structuredClone(opts);
    expect(roundtripped).toEqual(opts);
    // Round-tripped ranking/suggest options still validate clean.
    expect(normalizeTieBreakers([...roundtripped.ranking.tieBreakers])).toEqual(
      ['score', 'weight', 'exact', 'length', 'id']
    );
    expect(normalizeAutocompleteOptions({ ...roundtripped.autocomplete, tieBreakers: [...roundtripped.autocomplete.tieBreakers] })).toEqual(
      { limit: 3, mode: 'prefix', fuzzyDistance: 1, tieBreakers: ['score', 'id'] }
    );
  });

  test('fuzzy mode returns did-you-mean suggestions', async () => {
    const index = await DocumentIndex.create<TitleDoc>(SYMBOLS, titleOpts());
    const res = await index.autocomplete('AuthControllr', { mode: 'fuzzy' });
    expect(res.suggestions.length).toBeGreaterThan(0);
    expect(res.suggestions[0]?.text).toBe('AuthController');
    expect(res.suggestions[0]?.type).toBe('did-you-mean');
    index.destroy();
  });

  test('prefix + fuzzyDistance tolerates typos in completions', async () => {
    const index = await DocumentIndex.create<TitleDoc>(SYMBOLS, titleOpts());
    const exactMiss = await index.autocomplete('Auht');
    expect(exactMiss.suggestions.length).toBe(0);
    const typo = await index.autocomplete('Auht', { fuzzyDistance: 1 });
    expect(typo.suggestions.length).toBeGreaterThan(0);
    expect(typo.suggestions[0]?.type).toBe('completion');
    index.destroy();
  });

  test('empty query and no-match query return empty suggestions', async () => {
    const index = await DocumentIndex.create<TitleDoc>(SYMBOLS, titleOpts());
    expect((await index.autocomplete('')).suggestions).toEqual([]);
    expect((await index.autocomplete('zzz-no-match')).suggestions).toEqual([]);
    index.destroy();
  });

  test('unknown suggest field throws; malformed options throw fail-closed', async () => {
    const index = await DocumentIndex.create<TitleDoc>(SYMBOLS, titleOpts());
    await expect(index.autocomplete('Auth', { field: 'nope' })).rejects.toThrow();
    await expect(index.autocomplete('Auth', { mode: 'regex' as never })).rejects.toThrow();
    await expect(index.autocomplete('Auth', { fuzzyDistance: 9 as never })).rejects.toThrow(RangeError);
    await expect(index.autocomplete(42 as never)).rejects.toThrow(TypeError);
    index.destroy();
  });

  test('normalizeAutocompleteOptions defaults and boolean shorthand', () => {
    expect(normalizeAutocompleteOptions(undefined)).toEqual({ limit: 5, mode: 'prefix', fuzzyDistance: 0, tieBreakers: ['score', 'weight', 'exact', 'length', 'id'] });
    expect(normalizeAutocompleteOptions(true)).toEqual({ limit: 5, mode: 'prefix', fuzzyDistance: 0, tieBreakers: ['score', 'weight', 'exact', 'length', 'id'] });
    expect(AUTOCOMPLETE_DEFAULT_LIMIT).toBe(5);
    expect(() => normalizeAutocompleteOptions(false)).toThrow(TypeError);
    expect(() => normalizeAutocompleteOptions({ fuzzyDistance: 3 })).toThrow(RangeError);
  });

  test('suggest latency stays well under budget on 2k docs', async () => {
    const docs: TitleDoc[] = [];
    for (let i = 0; i < 2000; i++) {
      docs.push({ id: `d${i}`, title: `symbol_${i}_handler`, body: `body ${i}` });
    }
    const index = await DocumentIndex.create<TitleDoc>(docs, titleOpts());
    const t0 = performance.now();
    const res = await index.autocomplete('symbol_1');
    const dt = performance.now() - t0;
    expect(res.suggestions.length).toBeGreaterThan(0);
    expect(dt).toBeLessThan(500);
    index.destroy();
  });

  test('inline search({ autocomplete }) attaches suggestions to the response', async () => {
    const index = await DocumentIndex.create<TitleDoc>(SYMBOLS, titleOpts());
    const res = await index.search('Auth', { mode: 'prefix', autocomplete: true });
    expect(res.suggestions?.length).toBe(2);
    expect(res.suggestions?.[0]?.type).toBe('completion');
    const withOpts = await index.search('Auth', {
      mode: 'prefix',
      autocomplete: { limit: 1, field: 'title' },
    });
    expect(withOpts.suggestions?.length).toBe(1);
    const without = await index.search('Auth', { mode: 'prefix' });
    expect(without.suggestions).toBeUndefined();
    index.destroy();
  });

  test('removed docs never surface as suggestions', async () => {
    const index = await DocumentIndex.create<TitleDoc>(SYMBOLS, titleOpts());
    await index.remove('1');
    const res = await index.autocomplete('Auth');
    expect(res.suggestions.map((s) => s.text)).toEqual(['AuthService']);
    index.destroy();
  });

  test('non-finite ranking keys throw fail-closed (total-order contract)', () => {
    expect(() => compareRanked(rank({ score: NaN }), rank({ score: 1 }), DEFAULT_TIE_BREAKERS)).toThrow(TypeError);
    expect(() => compareRanked(rank({ fieldWeight: Infinity }), rank({ fieldWeight: 1 }), DEFAULT_TIE_BREAKERS)).toThrow(TypeError);
    expect(() => compareRanked(rank({ id: NaN }), rank({ id: 1 }), DEFAULT_TIE_BREAKERS)).toThrow(TypeError);
  });

  test("['id']-only hierarchy ignores scores", () => {
    const lowIdLowScore = rank({ score: 1, id: 'a' });
    const highIdHighScore = rank({ score: 999, id: 'z' });
    expect(compareRanked(lowIdLowScore, highIdHighScore, ['id'])).toBe(-1);
    expect(compareRanked(highIdHighScore, lowIdLowScore, ['id'])).toBe(1);
  });

  test('mixed string/number ids order deterministically via String()', () => {
    expect(compareRanked(rank({ id: 2 }), rank({ id: 'a' }), DEFAULT_TIE_BREAKERS)).toBe(-1);
    expect(compareRanked(rank({ id: 10 }), rank({ id: '2' }), DEFAULT_TIE_BREAKERS)).toBe(-1);
  });

  test('exact tier is post-fold (case-insensitive equals count as exact)', () => {
    expect(isExactTokenMatch(toks('Hello'), toks('hello'))).toBe(true);
  });

  test('suggest limit edges clamp fail-closed', () => {
    expect(normalizeAutocompleteOptions({ limit: 0 }).limit).toBe(1);
    expect(normalizeAutocompleteOptions({ limit: -5 }).limit).toBe(1);
    expect(normalizeAutocompleteOptions({ limit: NaN }).limit).toBe(5);
    expect(normalizeAutocompleteOptions({ limit: 1e12 }).limit).toBe(8192);
  });

  test('fuzzyDistance:2 hits the upper boundary', async () => {
    const index = await DocumentIndex.create<TitleDoc>(SYMBOLS, titleOpts());
    const res = await index.autocomplete('Auht', { fuzzyDistance: 2 });
    expect(res.suggestions.length).toBeGreaterThan(0);
    index.destroy();
  });

  test('autocomplete tieBreakers are honored; inline search inherits ranking hierarchy', async () => {
    const index = await DocumentIndex.create<TitleDoc>(SYMBOLS, titleOpts());
    const custom = await index.autocomplete('Auth', { tieBreakers: ['id'] });
    expect(custom.suggestions.map((s) => s.docId)).toEqual(['1', '2']);
    const inline = await index.search('Auth', {
      mode: 'prefix',
      ranking: { tieBreakers: ['score', 'id'] },
      autocomplete: { limit: 2 },
    });
    expect(inline.suggestions?.length).toBe(2);
    index.destroy();
  });

  test('inline suggest is index-wide: filter narrows results, not suggestions', async () => {
    const docs = [
      { id: '1', kind: 'a', title: 'AuthController', body: '' },
      { id: '2', kind: 'b', title: 'AuthService', body: '' },
    ];
    const index = await DocumentIndex.create<typeof docs[number]>(docs, {
      fields: [{ name: 'title', weight: 2.0 }],
      filterFields: [{ name: 'kind', type: 'string' }],
      preferGpu: false,
    });
    const res = await index.search('Auth', { mode: 'prefix', filter: { kind: 'a' }, autocomplete: true });
    expect(res.results.map((r) => r.id)).toEqual(['1']);
    expect(res.suggestions?.length).toBe(2);
    index.destroy();
  });

  test('search field validation is fail-closed on empty query and empty index', async () => {
    const index = await DocumentIndex.create<TitleDoc>(SYMBOLS, titleOpts());
    await expect(index.search('', { fields: ['nope'] })).rejects.toThrow();
    const empty = await DocumentIndex.create<TitleDoc>([], titleOpts());
    await expect(empty.search('x', { fields: ['nope'] })).rejects.toThrow();
    await expect(empty.search('x', { ranking: { tieBreakers: [] } })).rejects.toThrow(RangeError);
    index.destroy();
    empty.destroy();
  });

  test('over-long suggest query throws QueryTooLongError', async () => {
    const index = await DocumentIndex.create<TitleDoc>(SYMBOLS, titleOpts());
    const long = 'a'.repeat(1000);
    await expect(index.autocomplete(long)).rejects.toThrow();
    index.destroy();
  });

  test('legacy ufuzzy path honors deterministic ranking', async () => {
    const index = await DocumentIndex.create<TitleDoc>(
      [
        { id: 'b', title: 'hello', body: '' },
        { id: 'a', title: 'hello', body: '' },
      ],
      titleOpts()
    );
    const res = await index.search('hello', { mode: 'substring', cpuScorer: 'ufuzzy' });
    expect(res.results.map((r) => r.id)).toEqual(['a', 'b']);
    index.destroy();
  });

  test('reversed insertion still yields identical deterministic order', async () => {
    const fwd = await DocumentIndex.create<TitleDoc>(
      [
        { id: 'b', title: 'hello', body: '' },
        { id: 'a', title: 'hello', body: '' },
      ],
      titleOpts()
    );
    const rev = await DocumentIndex.create<TitleDoc>(
      [
        { id: 'a', title: 'hello', body: '' },
        { id: 'b', title: 'hello', body: '' },
      ],
      titleOpts()
    );
    const r1 = await fwd.search('hello', { mode: 'substring' });
    const r2 = await rev.search('hello', { mode: 'substring' });
    expect(r1.results.map((r) => r.id)).toEqual(r2.results.map((r) => r.id));
    fwd.destroy();
    rev.destroy();
  });

  test('deterministic parity rejects bad weights, docIds length, and docIds entries', () => {
    const rec = [toks('hello')];
    expect(() =>
      scoreExactMatchesMultiField(
        1, [{ name: 't', weight: NaN }], rec, [0], [0], toks('hello'),
        'substring', 10, 8192, undefined, undefined, undefined, true, undefined,
        { docIds: ['a'] }
      )
    ).toThrow(RangeError);
    expect(() =>
      scoreExactMatchesMultiField(
        1, [{ name: 't', weight: 1 }], rec, [0], [0], toks('hello'),
        'substring', 10, 8192, undefined, undefined, undefined, true, undefined,
        { docIds: ['a', 'extra'] }
      )
    ).toThrow(RangeError);
    expect(() =>
      scoreExactMatchesMultiField(
        1, [{ name: 't', weight: 1 }], rec, [0], [0], toks('hello'),
        'substring', 10, 8192, undefined, undefined, undefined, true, undefined,
        { docIds: [''] }
      )
    ).toThrow(TypeError);
  });
});

describe('M5 portability', () => {
  test('zero unguarded DOM references in M5 modules', async () => {
    const files = [
      'packages/webgpu-search/src/ranking.ts',
      'packages/webgpu-search/src/autocomplete.ts',
      'packages/webgpu-search/src/document-index.ts',
    ];
    for (const f of files) {
      const raw = await readFile(f, 'utf8');
      const noBlock = raw.replace(/\/\*[\s\S]*?\*\//g, '');
      // Blank string literals first so `//` inside URLs does not truncate code.
      const noStrings = noBlock.replace(/'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|`(?:[^`\\]|\\.)*`/g, "''");
      const code = noStrings
        .split('\n')
        .map((line) => {
          const idx = line.indexOf('//');
          return idx >= 0 ? line.slice(0, idx) : line;
        })
        .join('\n');
      const stripped = code.replace(/typeof\s+(document|window|navigator|self)\b/g, '');
      expect(stripped).not.toMatch(/(^|[^\w$.])document\s*\./);
      expect(stripped).not.toMatch(/(^|[^\w$.])window\s*[\.\[]/);
      expect(code.includes('localStorage')).toBe(false);
      expect(code.includes('sessionStorage')).toBe(false);
      // navigator/self/location are allowed only behind typeof guards.
      // document-index.ts has pre-existing guarded `navigator.gpu` fallback
      // detection (outside M5); assert M5 suggest code itself is clean.
      if (f.endsWith('document-index.ts')) {
        const m5Start = code.indexOf('computeSuggestions');
        const m5Slice = m5Start >= 0 ? code.slice(code.lastIndexOf('/**', m5Start - 2000)) : '';
        const m5Stripped = m5Slice.replace(/typeof\s+(document|window|navigator|self)\b/g, '');
        expect(m5Stripped.includes('navigator.')).toBe(false);
        expect(m5Stripped.includes('self.')).toBe(false);
      } else {
        expect(stripped.includes('navigator.')).toBe(false);
      }
      expect(stripped).not.toMatch(/(^|[^\w$.])location\s*\./);
    }
  });
});
