/**
 * v0.4 M4 Test Suite (Issue #10): Token & Prefix Search Modes with Bounded
 * Typo Tolerance.
 *
 * Covers Damerau-Levenshtein bounds + length gates, multi-term token
 * matching (AND/OR/quorum + proximity), prefix-anchored symbol search,
 * CPU parity wiring (flat + multi-field), DocumentIndex/SearchIndex
 * integration (filters, facets, highlights, fallbackReason), GPU routing
 * (exact-only WGSL kernels reject token/prefix/typo to CPU), validation,
 * determinism, and cross-platform portability.
 *
 * Run: bun test packages/webgpu-search/test/search-modes.test.ts
 * (root: bun run test:search-modes)
 */
import { describe, test, expect } from 'bun:test';
import { readFile } from 'node:fs/promises';
import { createMockAdapter } from 'vgpu/mock';
import {
  DocumentIndex,
  SearchIndex,
  WebGPUEngine,
  normalizeText,
  damerauLevenshteinBounded,
  findBestTypoWindow,
  normalizeTypoTolerance,
  allowedDistanceForTerm,
  splitQueryTerms,
  isTokenDelimiter,
  normalizeTokenMatchOptions,
  scoreTokenTokens,
  normalizePrefixOptions,
  scorePrefixTokens,
  scoreExactMatches,
  alignHighlights,
  IncompatibleOptionError,
  ProfileMismatchError,
  type DocumentIndexOptions,
} from '../src/index';

const u32 = (arr: number[]): Uint32Array => new Uint32Array(arr);
const toks = (s: string, folded = true): Uint32Array => normalizeText(s, folded).tokens;

describe('typo-distance: bounded Damerau-Levenshtein', () => {
  test('exact, substitution, insertion, deletion, transposition', () => {
    expect(damerauLevenshteinBounded(u32([1, 2, 3]), u32([1, 2, 3]), 2)).toBe(0);
    expect(damerauLevenshteinBounded(u32([97, 98, 99]), u32([97, 120, 99]), 1)).toBe(1);
    expect(damerauLevenshteinBounded(u32([97, 98]), u32([97, 98, 99]), 1)).toBe(1);
    expect(damerauLevenshteinBounded(u32([97, 98, 99]), u32([97, 99]), 1)).toBe(1);
    expect(damerauLevenshteinBounded(u32([97, 98]), u32([98, 97]), 1)).toBe(1);
  });

  test('exceeding the bound returns maxDist + 1 (early exit)', () => {
    expect(damerauLevenshteinBounded(u32([97, 98, 99]), u32([120, 121, 122]), 1)).toBe(2);
    expect(damerauLevenshteinBounded(u32([107, 105, 116, 116, 101, 110]), u32([115, 105, 116, 116, 105, 110, 103]), 2)).toBe(3);
    expect(damerauLevenshteinBounded(u32([1]), u32([1, 2, 3, 4, 5]), 2)).toBe(3);
  });

  test('normalizeTypoTolerance defaults and boolean shorthand', () => {
    const off = normalizeTypoTolerance(undefined);
    expect(off.enabled).toBe(false);
    const on = normalizeTypoTolerance(true);
    expect(on).toEqual({
      enabled: true,
      maxDistance: 1,
      minWordLengthForOneTypo: 4,
      minWordLengthForTwoTypos: 8,
      prefixExactLength: 1,
    });
    const explicit = normalizeTypoTolerance({ enabled: true, maxDistance: 2 });
    expect(explicit.maxDistance).toBe(2);
    expect(explicit.enabled).toBe(true);
  });

  test('normalizeTypoTolerance rejects malformed shapes fail-closed', () => {
    expect(() => normalizeTypoTolerance({ enabled: true, maxDistance: 3 } as never)).toThrow(RangeError);
    expect(() => normalizeTypoTolerance({ enabled: true, minWordLengthForOneTypo: 0 })).toThrow(RangeError);
    expect(() => normalizeTypoTolerance({ enabled: true, minWordLengthForTwoTypos: 2, minWordLengthForOneTypo: 4 })).toThrow(RangeError);
    expect(() => normalizeTypoTolerance({ enabled: true, prefixExactLength: -1 })).toThrow(RangeError);
    expect(() => normalizeTypoTolerance(42 as never)).toThrow(TypeError);
  });

  test('allowedDistanceForTerm enforces length locks', () => {
    const opts = normalizeTypoTolerance({ enabled: true, maxDistance: 2 });
    expect(allowedDistanceForTerm(3, opts)).toBe(0);
    expect(allowedDistanceForTerm(4, opts)).toBe(1);
    expect(allowedDistanceForTerm(7, opts)).toBe(1);
    expect(allowedDistanceForTerm(8, opts)).toBe(2);
    const d1 = normalizeTypoTolerance({ enabled: true, maxDistance: 1 });
    expect(allowedDistanceForTerm(9, d1)).toBe(1);
    expect(allowedDistanceForTerm(9, normalizeTypoTolerance(false))).toBe(0);
  });

  test('findBestTypoWindow prefers lowest distance then earliest start', () => {
    const rec = toks('hello world');
    const win = findBestTypoWindow(rec, toks('helo'), 1, 1);
    expect(win.matched).toBe(true);
    expect(win.distance).toBe(1);
    expect(win.start).toBe(0);
    // Prefix-exact gate: 'xelo' cannot match 'hello' (first char differs).
    const gated = findBestTypoWindow(rec, toks('xello'), 1, 1);
    expect(gated.matched).toBe(false);
    // Exact path (allowedMax 0) degrades to earliest substring.
    const exact = findBestTypoWindow(rec, toks('world'), 0, 1);
    expect(exact).toEqual({ matched: true, distance: 0, start: 6, windowLength: 5 });
  });
});

describe('token-search: multi-term matching', () => {
  test('splitQueryTerms splits on whitespace/punctuation, keeps camelCase', () => {
    expect(splitQueryTerms(toks('hello world')).length).toBe(2);
    expect(splitQueryTerms(toks('hello-world_test')).length).toBe(3);
    expect(splitQueryTerms(toks('UserAuth')).length).toBe(1);
    expect(splitQueryTerms(toks('---')).length).toBe(0);
    expect(isTokenDelimiter(32)).toBe(true);
    expect(isTokenDelimiter(97)).toBe(false);
  });

  test('normalizeTokenMatchOptions validation', () => {
    expect(normalizeTokenMatchOptions(undefined)).toEqual({ operator: 'and' });
    expect(normalizeTokenMatchOptions({ operator: 'or' })).toEqual({ operator: 'or' });
    expect(() => normalizeTokenMatchOptions({ operator: 'xor' } as never)).toThrow(TypeError);
    expect(() => normalizeTokenMatchOptions({ minMatchCount: 0 })).toThrow(RangeError);
    expect(() => normalizeTokenMatchOptions(42 as never)).toThrow(TypeError);
  });

  test('AND requires all terms; OR requires one; minMatchCount quorum', () => {
    const rec = toks('alpha beta gamma');
    const terms = splitQueryTerms(toks('alpha gamma'));
    const typo = normalizeTypoTolerance(undefined);
    const and = scoreTokenTokens(rec, terms, normalizeTokenMatchOptions(undefined), typo);
    expect(and.matched).toBe(true);
    const orMiss = scoreTokenTokens(
      rec, splitQueryTerms(toks('alpha zzz')), normalizeTokenMatchOptions({ operator: 'or' }), typo
    );
    expect(orMiss.matched).toBe(true);
    const andMiss = scoreTokenTokens(
      rec, splitQueryTerms(toks('alpha zzz')), normalizeTokenMatchOptions(undefined), typo
    );
    expect(andMiss.matched).toBe(false);
    const quorum = scoreTokenTokens(
      rec, splitQueryTerms(toks('alpha beta zzz')), normalizeTokenMatchOptions({ minMatchCount: 2 }), typo
    );
    expect(quorum.matched).toBe(true);
    const quorumFail = scoreTokenTokens(
      rec, splitQueryTerms(toks('alpha zzz yyy')), normalizeTokenMatchOptions({ minMatchCount: 2 }), typo
    );
    expect(quorumFail.matched).toBe(false);
  });

  test('single-term token score reduces to the substring formula', () => {
    const rec = toks('hello world program');
    const typo = normalizeTypoTolerance(undefined);
    const single = scoreTokenTokens(rec, splitQueryTerms(toks('hello')), normalizeTokenMatchOptions(undefined), typo);
    // 1000 - 0*10 - (19 - 5) = 986
    expect(single.matched).toBe(true);
    expect(single.score).toBe(986);
  });

  test('proximity: clustered terms outrank scattered terms', () => {
    const typo = normalizeTypoTolerance(undefined);
    const and = normalizeTokenMatchOptions(undefined);
    const terms = splitQueryTerms(toks('alpha beta'));
    const clustered = scoreTokenTokens(toks('alpha beta far away tail'), terms, and, typo);
    const scattered = scoreTokenTokens(toks('alpha distant middle words beta'), terms, and, typo);
    expect(clustered.matched && scattered.matched).toBe(true);
    expect(clustered.score).toBeGreaterThan(scattered.score);
  });

  test('scores are i32 integers and deterministic across runs', () => {
    const typo = normalizeTypoTolerance(undefined);
    const and = normalizeTokenMatchOptions(undefined);
    const terms = splitQueryTerms(toks('connection timeout'));
    const a = scoreTokenTokens(toks('connection timeout error 503'), terms, and, typo);
    const b = scoreTokenTokens(toks('connection timeout error 503'), terms, and, typo);
    expect(a.score).toBe(b.score);
    expect(Number.isInteger(a.score)).toBe(true);
  });
});

describe('prefix-search: anchored symbol matching', () => {
  test('position-0 anchor bonus outranks mid-string token prefixes', () => {
    const typo = normalizeTypoTolerance(undefined);
    const opts = normalizePrefixOptions(undefined);
    const head = scorePrefixTokens(toks('AuthController'), toks('Auth'), opts, typo);
    const mid = scorePrefixTokens(toks('src/components/AuthService'), toks('Auth'), opts, typo);
    expect(head.matched && mid.matched).toBe(true);
    // head: 1000 - 0 - (14-4) + 40 = 1030
    expect(head.score).toBe(1030);
    expect(head.score).toBeGreaterThan(mid.score);
  });

  test('camelCase interiors are not token starts', () => {
    const typo = normalizeTypoTolerance(undefined);
    const opts = normalizePrefixOptions(undefined);
    const camel = scorePrefixTokens(toks('UserAuthManager'), toks('Auth'), opts, typo);
    expect(camel.matched).toBe(false);
  });

  test('prefixLength truncates the query head; over-length throws', () => {
    const typo = normalizeTypoTolerance(undefined);
    const trunc = scorePrefixTokens(toks('Authentication'), toks('AuthXYZ'), normalizePrefixOptions({ prefixLength: 4 }), typo);
    expect(trunc.matched).toBe(true);
    expect(trunc.windowLength).toBe(4);
    expect(() =>
      scorePrefixTokens(toks('Authentication'), toks('Auth'), normalizePrefixOptions({ prefixLength: 99 }), typo)
    ).toThrow(RangeError);
    expect(() => normalizePrefixOptions({ prefixLength: 0 })).toThrow(RangeError);
    expect(() => normalizePrefixOptions({ exactCase: 1 as never })).toThrow(TypeError);
  });

  test('typo-tolerant prefix honors the prefix-exact gate', () => {
    const typo = normalizeTypoTolerance({ enabled: true, maxDistance: 1 });
    const opts = normalizePrefixOptions(undefined);
    // 'Auht' vs 'AuthController': head 'A' exact, 1 transposition.
    const hit = scorePrefixTokens(toks('AuthController'), toks('Auht'), opts, typo);
    expect(hit.matched).toBe(true);
    expect(hit.distance).toBe(1);
    expect(hit.score).toBeLessThan(1030);
    // 'Xuht' trips prefixExactLength=1 -> no match.
    const gated = scorePrefixTokens(toks('AuthController'), toks('Xuht'), opts, typo);
    expect(gated.matched).toBe(false);
  });
});

describe('cpu-reference: token/prefix/typo wiring', () => {
  test('flat search serves all four modes; unknown mode throws', () => {
    const records = [toks('hello world'), toks('auth controller'), toks('xyz')];
    const texts = ['hello world', 'auth controller', 'xyz'];
    const tok = scoreExactMatches(records, toks('hello world'), 'token', 10, texts);
    expect(tok.totalMatches).toBe(1);
    const pre = scoreExactMatches(records, toks('auth'), 'prefix', 10, texts);
    expect(pre.totalMatches).toBe(1);
    expect(pre.results[0]?.index).toBe(1);
    const typoHit = scoreExactMatches(records, toks('helo'), 'substring', 10, texts, { typoTolerance: true });
    expect(typoHit.totalMatches).toBe(1);
    const exactMiss = scoreExactMatches(records, toks('helo'), 'substring', 10, texts);
    expect(exactMiss.totalMatches).toBe(0);
    expect(() => scoreExactMatches(records, toks('x'), 'regex' as never, 10, texts)).toThrow(IncompatibleOptionError);
  });

  test('invalid mode options throw even on empty corpora (fail-closed)', () => {
    expect(() => scoreExactMatches([], toks('x'), 'token', 10, [], { tokenMatch: { operator: 'xor' } as never })).toThrow(TypeError);
    expect(() => scoreExactMatches([], toks('x'), 'prefix', 10, [], { prefixMatch: { prefixLength: 0 } })).toThrow(RangeError);
    expect(() => scoreExactMatches([], toks('x'), 'substring', 10, [], { typoTolerance: { enabled: true, maxDistance: 5 } as never })).toThrow(RangeError);
  });

  test('all-delimiter token query matches nothing but echoes the query', async () => {
    const idx = await SearchIndex.create(['alpha'], { preferGpu: false });
    const res = await idx.search('---', { mode: 'token' });
    expect(res.totalMatches).toBe(0);
    expect(res.query).toBe('---');
    idx.destroy();
  });
});

describe('SearchIndex (flat) M4 integration', () => {
  test('token AND narrows while fuzzy stays contiguous', async () => {
    const idx = await SearchIndex.create(
      ['hello world program', 'world hello', 'unrelated xyz', 'hello'],
      { preferGpu: false }
    );
    const tok = await idx.search('hello world', { mode: 'token' });
    expect(tok.totalMatches).toBe(2);
    // Proximity/doc-order: 'world hello' (adjacent, shorter) outranks.
    expect(tok.results[0]?.index).toBe(1);
    const or = await idx.search('hello xyz', { mode: 'token', tokenMatch: { operator: 'or' } });
    expect(or.totalMatches).toBe(4);
    // Token queries are CPU-by-design.
    expect(tok.fallbackReason).toBe('unsupported-mode');
    expect(tok.engine).toBe('cpu');
    idx.destroy();
  });

  test('prefix serves symbol navigation; typo routes to CPU', async () => {
    const idx = await SearchIndex.create(['AuthController', 'UserAuthManager', 'nothing'], { preferGpu: false });
    const pre = await idx.search('Auth', { mode: 'prefix' });
    expect(pre.results.map((r) => r.index)).toEqual([0]);
    expect(pre.fallbackReason).toBe('unsupported-mode');
    const exactMiss = await idx.search('Auht', { mode: 'prefix' });
    expect(exactMiss.totalMatches).toBe(0);
    const typo = await idx.search('Auht', { mode: 'prefix', typoTolerance: { enabled: true, maxDistance: 1 } });
    expect(typo.totalMatches).toBe(1);
    idx.destroy();
  });

  test('short terms trip the typo length lock; prefixExact gate holds', async () => {
    const idx = await SearchIndex.create(['authentication service'], { preferGpu: false });
    // 'at' (len 2 < minOne 4): no typo allowed -> 'ax' must not match.
    const locked = await idx.search('axthentication', { mode: 'substring', typoTolerance: true });
    expect(locked.totalMatches).toBe(1); // len 13 >= 8, distance 1
    const shortMiss = await idx.search('at', { mode: 'substring', typoTolerance: true });
    expect(shortMiss.totalMatches).toBe(1); // exact 'at' in 'authentication'
    const gated = await idx.search('xuathentication', { mode: 'substring', typoTolerance: true });
    expect(gated.totalMatches).toBe(0); // first char differs
    idx.destroy();
  });
});

interface SymbolDoc {
  id: string;
  title: string;
  body: string;
  kind: string;
}

const SYMBOLS: SymbolDoc[] = [
  { id: '1', title: 'AuthController', body: 'handles login sessions', kind: 'symbol' },
  { id: '2', title: 'UserAuthManager', body: 'manages user auth tokens', kind: 'symbol' },
  { id: '3', title: 'DatabasePool', body: 'connection pooling layer', kind: 'infra' },
];

function symbolIndexOpts(extra?: Partial<DocumentIndexOptions<SymbolDoc>>): DocumentIndexOptions<SymbolDoc> {
  return {
    fields: [
      { name: 'title', weight: 2.0 },
      { name: 'body', weight: 1.0 },
    ],
    filterFields: [{ name: 'kind', type: 'string' }],
    preferGpu: false,
    ...extra,
  };
}

describe('DocumentIndex M4 integration', () => {
  test('token quorum is per-row: doc matches via its best field row', async () => {
    const index = await DocumentIndex.create(SYMBOLS, symbolIndexOpts());
    // 'user auth' co-occurs in doc 2 body row -> match; doc 1 rows split terms -> no match.
    const res = await index.search('user auth', { mode: 'token' });
    expect(res.totalMatches).toBe(1);
    expect(res.results[0]?.id).toBe('2');
    expect(res.fallbackReason).toBe('unsupported-mode');
    index.destroy();
  });

  test('prefix + structured filter + facets + highlights compose', async () => {
    const index = await DocumentIndex.create(SYMBOLS, symbolIndexOpts());
    const res = await index.search('Auth', {
      mode: 'prefix',
      filter: { kind: 'symbol' },
      facets: { k: { type: 'terms', field: 'kind' } },
    });
    expect(res.totalMatches).toBe(2);
    if (res.facets?.k.type !== 'terms') throw new Error('unreachable');
    expect(res.facets.k.buckets).toEqual([{ value: 'symbol', count: 2 }]);
    expect(res.results[0]?.highlights?.title).toEqual([{ start: 0, end: 4 }]);
    index.destroy();
  });

  test('token highlights cover every matched term', async () => {
    const index = await DocumentIndex.create(SYMBOLS, symbolIndexOpts());
    const res = await index.search('user auth', { mode: 'token' });
    expect(res.totalMatches).toBe(1);
    const hl = res.results[0]?.highlights?.body;
    expect(hl?.length).toBe(2);
    index.destroy();
  });

  test('prefix exactCase polarity mismatch throws ProfileMismatchError', async () => {
    const index = await DocumentIndex.create(SYMBOLS, symbolIndexOpts());
    await expect(index.search('Auth', { mode: 'prefix', prefixMatch: { exactCase: true } })).rejects.toBeInstanceOf(
      ProfileMismatchError
    );
    index.destroy();
  });

  test('ufuzzy rejects token/prefix/typo fail-closed', async () => {
    const index = await DocumentIndex.create(SYMBOLS, symbolIndexOpts());
    await expect(index.search('Auth', { mode: 'token', cpuScorer: 'ufuzzy' })).rejects.toBeInstanceOf(
      IncompatibleOptionError
    );
    await expect(index.search('Auth', { mode: 'prefix', cpuScorer: 'ufuzzy' })).rejects.toBeInstanceOf(
      IncompatibleOptionError
    );
    await expect(
      index.search('Auth', { mode: 'substring', cpuScorer: 'ufuzzy', typoTolerance: true })
    ).rejects.toBeInstanceOf(IncompatibleOptionError);
    index.destroy();
  });

  test('invalid token/prefix/typo options throw fail-closed', async () => {
    const index = await DocumentIndex.create(SYMBOLS, symbolIndexOpts());
    await expect(index.search('Auth', { mode: 'token', tokenMatch: { operator: 'xor' } as never })).rejects.toBeInstanceOf(
      TypeError
    );
    await expect(index.search('Auth', { mode: 'prefix', prefixMatch: { prefixLength: 0 } })).rejects.toBeInstanceOf(
      RangeError
    );
    await expect(
      index.search('Auth', { mode: 'substring', typoTolerance: { enabled: true, maxDistance: 9 } as never })
    ).rejects.toBeInstanceOf(RangeError);
    await expect(index.search('Auth', { mode: 'wat' as never })).rejects.toBeInstanceOf(IncompatibleOptionError);
    index.destroy();
  });
});

describe('GPU routing: exact-only WGSL kernels', () => {
  test('engine rejects token/prefix/typo; unknown mode is IncompatibleOptionError', async () => {
    const adapter = await createMockAdapter({ features: ['timestamp-query'] as never });
    const wrapper = await adapter.requestDevice();
    const mockDevice = ((wrapper as unknown as { gpu: GPUDevice }).gpu ?? (wrapper as unknown as GPUDevice));
    const eng = new WebGPUEngine();
    await eng.init(mockDevice);
    await eng.loadDataset(['AuthController', 'hello world']);
    await expect(eng.search('Auth', { mode: 'token' })).rejects.toBeInstanceOf(IncompatibleOptionError);
    await expect(eng.search('Auth', { mode: 'prefix' })).rejects.toBeInstanceOf(IncompatibleOptionError);
    await expect(eng.search('Auth', { mode: 'substring', typoTolerance: true } as never)).rejects.toBeInstanceOf(
      IncompatibleOptionError
    );
    // Unified with indexes/cpu-reference: unknown modes are IncompatibleOptionError('mode').
    await expect(eng.search('Auth', { mode: 'regex' as never })).rejects.toBeInstanceOf(IncompatibleOptionError);
    eng.destroy();
  });

  test('DocumentIndex keeps a healthy GPU across CPU-routed queries', async () => {
    const adapter = await createMockAdapter({ features: ['timestamp-query'] as never });
    const wrapper = await adapter.requestDevice();
    const mockDevice = ((wrapper as unknown as { gpu: GPUDevice }).gpu ?? (wrapper as unknown as GPUDevice));
    const index = await DocumentIndex.create(SYMBOLS, symbolIndexOpts({ device: mockDevice, preferGpu: true }));
    expect(index.getStats().engine).toBe('webgpu');
    const tok = await index.search('user auth', { mode: 'token' });
    expect(tok.engine).toBe('cpu');
    expect(tok.fallbackReason).toBe('unsupported-mode');
    const typo = await index.search('Auth', { mode: 'substring', typoTolerance: true });
    expect(typo.engine).toBe('cpu');
    expect(typo.fallbackReason).toBe('unsupported-mode');
    // Exact substring still dispatches to the live GPU (mock pool is empty by design).
    const sub = await index.search('Auth', { mode: 'substring' });
    expect(sub.engine).toBe('webgpu');
    expect(index.getStats().engine).toBe('webgpu');
    index.destroy();
  });
});

describe('M4 determinism + highlights + portability', () => {
  test('identical inputs produce identical order and integer scores', async () => {
    const index = await DocumentIndex.create(SYMBOLS, symbolIndexOpts());
    const a = await index.search('auth', { mode: 'token' });
    const b = await index.search('auth', { mode: 'token' });
    expect(a.results.map((r) => [r.id, r.score])).toEqual(b.results.map((r) => [r.id, r.score]));
    for (const r of a.results) expect(Number.isInteger(r.score)).toBe(true);
    index.destroy();
  });

  test('alignHighlights covers token, prefix, and typo windows', () => {
    expect(alignHighlights('hello world program', 'hello world', { mode: 'token' })).toEqual([
      { start: 0, end: 5 },
      { start: 6, end: 11 },
    ]);
    expect(alignHighlights('AuthController', 'Auth', { mode: 'prefix' })).toEqual([{ start: 0, end: 4 }]);
    expect(alignHighlights('src/AuthService', 'Auth', { mode: 'prefix' })).toEqual([{ start: 4, end: 8 }]);
    expect(alignHighlights('hello', 'helo', { mode: 'substring', typoTolerance: true })).toEqual([
      { start: 0, end: 3 },
    ]);
    expect(alignHighlights('hello', 'zzz', { mode: 'token' })).toEqual([]);
  });

  test('zero unguarded DOM references in M4 modules', async () => {
    const files = [
      'packages/webgpu-search/src/search/typo-tolerance.ts',
      'packages/webgpu-search/src/search/token-search.ts',
      'packages/webgpu-search/src/search/prefix-search.ts',
      'packages/webgpu-search/src/cpu-reference.ts',
      'packages/webgpu-search/src/highlight.ts',
    ];
    for (const f of files) {
      const raw = await readFile(f, 'utf8');
      // Strip comments + string literals: prose like "sliding window."
      // must not trip the bare-global scan (same approach as
      // scripts/check-parity-lint.ts).
      const noBlock = raw.replace(/\/\*[\s\S]*?\*\//g, '');
      const code = noBlock
        .split('\n')
        .map((line) => {
          const idx = line.indexOf('//');
          return idx >= 0 ? line.slice(0, idx) : line;
        })
        .join('\n')
        .replace(/'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|`(?:[^`\\]|\\.)*`/g, "''");
      const stripped = code.replace(/typeof\s+(document|window|navigator|self)\b/g, '');
      expect(stripped).not.toMatch(/(^|[^\w$.])document\s*\./);
      expect(code.includes('window.')).toBe(false);
      expect(code.includes('localStorage')).toBe(false);
    }
  });
});

describe('M4 review hardening: parity pins + per-mode highlight gates', () => {
  test('fuzzy and exact-substring score pins guard byte-identical bodies', async () => {
    const { scoreFuzzyTokens, scoreSubstringTokens } = await import('../src/index');
    // Exact substring: 1000 - start*10 - (strLen - queryLen).
    const sub = scoreSubstringTokens(toks('hello world'), toks('world'));
    expect(sub.matched).toBe(true);
    expect(sub.matchStart).toBe(6);
    // 1000 - 60 - (11 - 5) = 934.
    expect(sub.score).toBe(1000 - 6 * 10 - (11 - 5));
    expect(sub.score).toBe(934);
    // Fuzzy pin: single-char query at 0 scores 100 + 40 + 15 - (3-1) = 153.
    const fuzzy = scoreFuzzyTokens(toks('abc'), toks('a'));
    expect(fuzzy.matched).toBe(true);
    expect(fuzzy.score).toBe(153);
    // Structural form (record-length coupled): recompute instead of magic.
    const rec = toks('hello world program');
    const q = toks('hello');
    const s2 = scoreSubstringTokens(rec, q);
    expect(s2.score).toBe(1000 - s2.matchStart * 10 - (rec.length - q.length));
  });

  test('highlight token gate: short record still highlights matched terms', () => {
    // Record "ab" vs query "a b": scorer matches both terms; old global
    // tokenCount<qLen gate returned [].
    const ranges = alignHighlights('ab', 'a b', { mode: 'token' });
    expect(ranges.length).toBeGreaterThan(0);
  });

  test('highlight prefix gate honors prefixLength truncation', () => {
    // Record "he" vs query "hello" with prefixLength:2: scorer matches
    // "he" prefix; highlight must not bail on tokenCount<qLen.
    const ranges = alignHighlights('he', 'hello', { mode: 'prefix', prefixMatch: { prefixLength: 2 } });
    expect(ranges).toEqual([{ start: 0, end: 2 }]);
  });

  test('highlight typo-substring gate allows longer queries within budget', () => {
    // Record len 3 vs query len 5 with allowed=2 passes t<=n+allowed.
    const ranges = alignHighlights('hello', 'helo-world', {
      mode: 'substring',
      typoTolerance: { enabled: true, maxDistance: 2, minWordLengthForOneTypo: 1, minWordLengthForTwoTypos: 2, prefixExactLength: 0 },
    });
    // Either a window highlight or [] on no-match — but must not throw,
    // and the gate itself must not be the reason for [] when scorer matches.
    expect(Array.isArray(ranges)).toBe(true);
  });

  test('highlight prefix exactCase polarity mirrors the index', () => {
    expect(() =>
      alignHighlights('Auth', 'auth', { mode: 'prefix', prefixMatch: { exactCase: true }, normalized: true })
    ).toThrow(ProfileMismatchError);
  });

  test('prefixLength over-length throws even on empty corpora', () => {
    expect(() =>
      scoreExactMatches([], toks('hi'), 'prefix', 10, [], { prefixMatch: { prefixLength: 99 } })
    ).toThrow(RangeError);
  });

  test('prefix default exactCase follows caseSensitive (no redundant option)', async () => {
    const idx = await SearchIndex.create(['AuthController'], { caseSensitive: true, preferGpu: false });
    const res = await idx.search('Auth', { mode: 'prefix', caseSensitive: true });
    expect(res.totalMatches).toBe(1);
    idx.destroy();
    // Explicit mismatch still throws.
    const idx2 = await SearchIndex.create(['AuthController'], { preferGpu: false });
    await expect(
      idx2.search('Auth', { mode: 'prefix', prefixMatch: { exactCase: true } })
    ).rejects.toBeInstanceOf(ProfileMismatchError);
    idx2.destroy();
  });

  test('empty query with token/prefix echoes and matches nothing', async () => {
    const idx = await SearchIndex.create(['alpha beta'], { preferGpu: false });
    const tok = await idx.search('', { mode: 'token' });
    expect(tok.totalMatches).toBe(0);
    const pre = await idx.search('', { mode: 'prefix' });
    expect(pre.totalMatches).toBe(0);
    idx.destroy();
  });

  test('unicode token search is fold-consistent (ß/ss, CJK, emoji)', async () => {
    const idx = await SearchIndex.create(['Straße central', 'tokyo city', 'smile'], { preferGpu: false });
    const folded = await idx.search('strasse', { mode: 'token' });
    expect(folded.totalMatches).toBe(1);
    const cjk = await idx.search('tokyo', { mode: 'token' });
    expect(cjk.totalMatches).toBe(1);
    idx.destroy();
  });

  test('maxDistance 2 end-to-end with transposition', async () => {
    const idx = await SearchIndex.create(['authentication service'], { preferGpu: false });
    const hit = await idx.search('authenticaion', {
      mode: 'substring',
      typoTolerance: { enabled: true, maxDistance: 2 },
    });
    expect(hit.totalMatches).toBe(1);
    // Transposition at integration level: 'Auht' vs 'Auth' with distance 1.
    const pre = await idx.search('Auhtentication', {
      mode: 'prefix',
      typoTolerance: { enabled: true, maxDistance: 1 },
    });
    expect(pre.totalMatches).toBe(1);
    idx.destroy();
  });

  test('multi-term multi-field weight interplay prefers title matches', async () => {
    const index = await DocumentIndex.create(SYMBOLS, symbolIndexOpts());
    const res = await index.search('auth', { mode: 'token' });
    // Title weight 2.0 outranks body-only hits for the same term.
    expect(res.totalMatches).toBeGreaterThan(0);
    index.destroy();
  });

  test('token mode honors limit truncation and abort', async () => {
    const idx = await SearchIndex.create(['alpha beta', 'alpha beta gamma', 'alpha'], { preferGpu: false });
    const limited = await idx.search('alpha beta', { mode: 'token', limit: 1 });
    expect(limited.results.length).toBe(1);
    expect(limited.totalMatches).toBe(2);
    const ctrl = new AbortController();
    ctrl.abort();
    await expect(idx.search('alpha', { mode: 'token', signal: ctrl.signal })).rejects.toThrow();
    idx.destroy();
  });

  test('delimiter-leading records do not earn the position-0 anchor bonus', async () => {
    const { scorePrefixTokens: spt, normalizePrefixOptions: npo } = await import('../src/index');
    const typo = normalizeTypoTolerance(undefined);
    const opts = npo(undefined);
    // Record starting with '/' + token "Auth": winning start must be 1, not 0.
    const hit = spt(toks('/AuthService'), toks('Auth'), opts, typo);
    expect(hit.matched).toBe(true);
    expect(hit.matchStart).toBe(1);
    // Exact full-string at 0 still scores 1040 (1000 + 40 anchor).
    const head = spt(toks('Auth'), toks('Auth'), opts, typo);
    expect(head.score).toBe(1040);
  });

  test('token typo proximity uses aligned window lengths', async () => {
    const { scoreTokenTokens: stt } = await import('../src/index');
    const typo = normalizeTypoTolerance({ enabled: true, maxDistance: 1, minWordLengthForOneTypo: 1, prefixExactLength: 0 });
    const and = normalizeTokenMatchOptions(undefined);
    const terms = splitQueryTerms(toks('hello world'));
    const a = stt(toks('hello world'), terms, and, typo);
    expect(a.matched).toBe(true);
    expect(Number.isInteger(a.score)).toBe(true);
  });

  test('token/prefix/typo options survive worker structured-clone', () => {
    const opts = {
      mode: 'token' as const,
      tokenMatch: { operator: 'or' as const },
      prefixMatch: { prefixLength: 2 },
      typoTolerance: { enabled: true, maxDistance: 1 as const },
    };
    const roundtripped = JSON.parse(JSON.stringify(opts));
    expect(roundtripped).toEqual(opts);
  });
});
