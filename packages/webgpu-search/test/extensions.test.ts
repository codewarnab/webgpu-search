/**
 * v0.4 M6 Test Suite (Issue #10): Extensibility Pipeline & Safe Hook Architecture.
 *
 * Covers custom tokenization (default + code-aware camelCase/snake/kebab),
 * scoring hooks (post-match Top-K only + deterministic re-sort), filter
 * predicates (conjunctive), post-processing, persistence guards
 * (IncompatibleHookError fail-closed), worker boundary rejection, and
 * cross-platform portability.
 *
 * Run: bun test packages/webgpu-search/test/extensions.test.ts
 */
import { describe, test, expect } from 'bun:test';
import { readFile } from 'node:fs/promises';
import {
  DocumentIndex,
  defaultTokenizer,
  codeTokenizer,
  normalizeSearchExtensionHooks,
  resolveEffectiveHooks,
  hasAnyHook,
  getHookId,
  collectHookIds,
  assertHooksSatisfied,
  tokenizeWithHook,
  getTokenTermsForQuery,
  applyScoringHook,
  applyPostProcess,
  normalizeText,
  deserializeDocumentSnapshot,
  IncompatibleHookError,
  type DocumentIndexOptions,
} from '../src/index';

interface Doc {
  id: string;
  title: string;
  body: string;
  kind: string;
  year: number;
}

const DOCS: Doc[] = [
  { id: '1', title: 'AuthController', body: 'handles login sessions', kind: 'symbol', year: 2023 },
  { id: '2', title: 'UserAuthManager', body: 'manages user auth tokens', kind: 'symbol', year: 2024 },
  { id: '3', title: 'DatabasePool', body: 'connection pooling layer', kind: 'infra', year: 2022 },
  { id: '4', title: 'Auth Service', body: 'authentication service endpoint', kind: 'symbol', year: 2025 },
];

function baseOpts(extra?: Partial<DocumentIndexOptions<Doc>>): DocumentIndexOptions<Doc> {
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

describe('tokenizers: default + code-aware', () => {
  test('defaultTokenizer splits on shared delimiters, keeps camelCase', () => {
    expect(defaultTokenizer('hello world')).toEqual(['hello', 'world']);
    expect(defaultTokenizer('hello-world_test')).toEqual(['hello', 'world', 'test']);
    expect(defaultTokenizer('UserAuth')).toEqual(['UserAuth']);
    expect(defaultTokenizer('---')).toEqual([]);
    expect(defaultTokenizer('')).toEqual([]);
    expect(() => defaultTokenizer(42 as never)).toThrow(TypeError);
  });

  test('codeTokenizer splits camelCase, snake, kebab, acronyms', () => {
    expect(codeTokenizer('UserAuthManager')).toEqual(['User', 'Auth', 'Manager']);
    expect(codeTokenizer('auth_controller')).toEqual(['auth', 'controller']);
    expect(codeTokenizer('my-component')).toEqual(['my', 'component']);
    expect(codeTokenizer('getHTTPResponse')).toEqual(['get', 'HTTP', 'Response']);
    expect(codeTokenizer('HTTPResponse')).toEqual(['HTTP', 'Response']);
    expect(codeTokenizer('getHTTP')).toEqual(['get', 'HTTP']);
    expect(codeTokenizer('AuthController')).toEqual(['Auth', 'Controller']);
    expect(codeTokenizer('')).toEqual([]);
    expect(() => codeTokenizer(42 as never)).toThrow(TypeError);
    expect(() => codeTokenizer('hi', { minTokenLength: -1 })).toThrow(RangeError);
    expect(() => codeTokenizer('hi', { splitOnDigitBoundaries: 1 as never })).toThrow(TypeError);
  });

  test('codeTokenizer splits letter<->digit by default, opt-out preserves', () => {
    expect(codeTokenizer('auth2Login')).toEqual(['auth', '2', 'Login']);
    expect(codeTokenizer('abc123')).toEqual(['abc', '123']);
    expect(codeTokenizer('abc123', { splitOnDigitBoundaries: false })).toEqual(['abc123']);
    expect(codeTokenizer('abc123', { minTokenLength: 4 })).toEqual([]);
  });

  test('tokenizeWithHook validates hook returns fail-closed', () => {
    expect(tokenizeWithHook('hi', () => ['a', 'b'])).toEqual(['a', 'b']);
    expect(tokenizeWithHook('hi', () => ['', 'a', ''])).toEqual(['a']);
    expect(() => tokenizeWithHook('hi', (() => 'nope') as never)).toThrow(TypeError);
    expect(() => tokenizeWithHook('hi', (() => [42]) as never)).toThrow(TypeError);
    expect(() => tokenizeWithHook('hi', 42 as never)).toThrow(TypeError);
  });

  test('getTokenTermsForQuery falls back to splitQueryTerms without hook', async () => {
    const { splitQueryTerms } = await import('../src/index');
    const q = normalizeText('hello world', true).tokens;
    const fallback = getTokenTermsForQuery('hello world', q, true, undefined);
    expect(fallback.map((t) => Array.from(t))).toEqual(
      splitQueryTerms(q).map((t) => Array.from(t))
    );
    const custom = getTokenTermsForQuery('UserAuth', normalizeText('UserAuth', true).tokens, true, codeTokenizer);
    expect(custom.length).toBe(2);
  });
});

describe('hook validation + resolution', () => {
  test('normalizeSearchExtensionHooks validates shapes fail-closed', () => {
    expect(normalizeSearchExtensionHooks(undefined)).toBeUndefined();
    expect(normalizeSearchExtensionHooks({})).toBeUndefined();
    const tok = (s: string) => [s];
    expect(normalizeSearchExtensionHooks({ tokenizer: tok })?.tokenizer).toBe(tok);
    expect(() => normalizeSearchExtensionHooks(42 as never)).toThrow(TypeError);
    expect(() => normalizeSearchExtensionHooks({ tokenizer: 'x' } as never)).toThrow(TypeError);
    expect(() => normalizeSearchExtensionHooks({ bogus: () => {} } as never)).toThrow(TypeError);
  });

  test('resolveEffectiveHooks merges index + query (query wins per key)', () => {
    const a = () => 1;
    const b = () => 2;
    const idx = { scoringHook: a as never, filterPredicate: a as never };
    const qry = { scoringHook: b as never };
    const merged = resolveEffectiveHooks(idx as never, qry as never);
    expect(merged?.scoringHook).toBe(b as never);
    expect(merged?.filterPredicate).toBe(a as never);
    expect(resolveEffectiveHooks(undefined, undefined)).toBeUndefined();
    expect(() => resolveEffectiveHooks({ tokenizer: 1 as never }, undefined)).toThrow(TypeError);
  });

  test('hasAnyHook detects presence', () => {
    expect(hasAnyHook(undefined)).toBe(false);
    expect(hasAnyHook({})).toBe(false);
    expect(hasAnyHook({ tokenizer: (() => []) as never })).toBe(true);
  });

  test('getHookId prefers hookId, then name, then anonymous', () => {
    function named() { return []; }
    expect(getHookId(named)).toBe('named');
    const anon = (() => []) as { hookId?: string };
    anon.hookId = 'my-tok-v1';
    expect(getHookId(anon)).toBe('my-tok-v1');
    expect(getHookId(() => {})).toBe('anonymous');
    expect(() => getHookId(42 as never)).toThrow(TypeError);
  });

  test('collectHookIds + assertHooksSatisfied fail-closed restore guard', () => {
    function myScorer() { return 1; }
    const ids = collectHookIds({ scoringHook: myScorer as never });
    expect(ids).toEqual({ scoringHook: 'myScorer' });
    expect(collectHookIds(undefined)).toBeUndefined();
    // Satisfied when matching handler supplied.
    expect(() => assertHooksSatisfied(ids, { scoringHook: myScorer as never })).not.toThrow();
    // Missing handler throws IncompatibleHookError.
    expect(() => assertHooksSatisfied(ids, undefined)).toThrow(IncompatibleHookError);
    expect(() => assertHooksSatisfied(ids, {})).toThrow(IncompatibleHookError);
    // Mismatched ID throws.
    function other() { return 2; }
    expect(() => assertHooksSatisfied(ids, { scoringHook: other as never })).toThrow(IncompatibleHookError);
    // No required hooks is a no-op (pre-M6 backward compat).
    expect(() => assertHooksSatisfied(undefined, undefined)).not.toThrow();
  });

  test('applyScoringHook validates finite returns, propagates throws', () => {
    const results = [
      { id: '1', doc: { id: '1' }, score: 100, matchedField: 'title' },
    ] as never[];
    applyScoringHook(results as never, (() => 200) as never, 'q');
    expect((results[0] as { score: number }).score).toBe(200);
    expect(() => applyScoringHook(results as never, (() => Number.NaN) as never, 'q')).toThrow(TypeError);
    expect(() => applyScoringHook(results as never, (() => { throw new Error('boom'); }) as never, 'q')).toThrow('boom');
  });

  test('applyPostProcess validates array returns', () => {
    const results = [{ id: '1' }] as never[];
    expect(applyPostProcess(results as never, undefined)).toBe(results as never);
    expect(applyPostProcess(results as never, ((r: never[]) => r.slice(0, 0)) as never)).toEqual([]);
    expect(() => applyPostProcess(results as never, (() => 42) as never)).toThrow(TypeError);
    expect(() => applyPostProcess(results as never, 42 as never)).toThrow(TypeError);
  });
});

describe('DocumentIndex extension integration', () => {
  test('constructor rejects malformed extensions fail-closed', async () => {
    await expect(
      DocumentIndex.create(DOCS, baseOpts({ extensions: { tokenizer: 'x' } as never }))
    ).rejects.toBeInstanceOf(TypeError);
    await expect(
      DocumentIndex.create(DOCS, baseOpts({ extensions: { bogus: () => {} } as never }))
    ).rejects.toBeInstanceOf(TypeError);
  });

  test('custom tokenizer splits camelCase for token mode', async () => {
    const index = await DocumentIndex.create(DOCS, baseOpts({
      extensions: { tokenizer: codeTokenizer },
    }));
    // 'UserAuth' -> ['User','Auth']: doc 2 contains both in title/body row.
    const res = await index.search('UserAuth', { mode: 'token' });
    expect(res.totalMatches).toBeGreaterThan(0);
    expect(res.results.map((r) => r.id)).toContain('2');
    // Default tokenizer keeps 'UserAuth' whole: only exact substring rows match.
    const plain = await DocumentIndex.create(DOCS, baseOpts());
    const resPlain = await plain.search('UserAuth', { mode: 'token' });
    // Plain single-term token search matches docs containing 'UserAuth' contiguously.
    expect(resPlain.totalMatches).toBeLessThanOrEqual(res.totalMatches);
    index.destroy();
    plain.destroy();
  });

  test('per-query tokenizer overrides index tokenizer', async () => {
    const index = await DocumentIndex.create(DOCS, baseOpts({
      extensions: { tokenizer: defaultTokenizer },
    }));
    const res = await index.search('UserAuth', {
      mode: 'token',
      extensions: { tokenizer: codeTokenizer },
    });
    expect(res.results.map((r) => r.id)).toContain('2');
    index.destroy();
  });

  test('tokenizer ignored for non-token modes (fuzzy/substring)', async () => {
    const index = await DocumentIndex.create(DOCS, baseOpts({
      extensions: { tokenizer: codeTokenizer },
    }));
    const fuzzy = await index.search('Auth', { mode: 'fuzzy' });
    expect(fuzzy.totalMatches).toBeGreaterThan(0);
    const sub = await index.search('Auth', { mode: 'substring' });
    expect(sub.totalMatches).toBeGreaterThan(0);
    index.destroy();
  });

  test('scoringHook runs only on surviving Top-K + deterministic re-sort', async () => {
    const index = await DocumentIndex.create(DOCS, baseOpts());
    let calls = 0;
    const scoringHook = (doc: Doc, baseScore: number) => {
      calls++;
      // Recency boost: year 2025 docs get +5000.
      return doc.year === 2025 ? baseScore + 5000 : baseScore;
    };
    const res = await index.search('Auth', {
      mode: 'substring',
      limit: 2,
      extensions: { scoringHook },
    });
    expect(res.results.length).toBeLessThanOrEqual(2);
    expect(calls).toBe(res.results.length);
    // Boosted doc 4 must rank first despite lower base score.
    expect(res.results[0]?.id).toBe('4');
    // Determinism: identical inputs produce identical order.
    calls = 0;
    const res2 = await index.search('Auth', {
      mode: 'substring',
      limit: 2,
      extensions: { scoringHook },
    });
    expect(res2.results.map((r) => [r.id, r.score])).toEqual(res.results.map((r) => [r.id, r.score]));
    index.destroy();
  });

  test('scoringHook returning non-finite throws fail-closed', async () => {
    const index = await DocumentIndex.create(DOCS, baseOpts());
    await expect(
      index.search('Auth', { extensions: { scoringHook: (() => Number.NaN) as never } })
    ).rejects.toBeInstanceOf(TypeError);
    index.destroy();
  });

  test('filterPredicate composes conjunctively with options.filter', async () => {
    const index = await DocumentIndex.create(DOCS, baseOpts({
      extensions: { filterPredicate: (d: Doc) => d.kind === 'symbol' },
    }));
    // Query 'pool' matches infra doc 3, but extension predicate excludes infra.
    const res = await index.search('pool', { mode: 'substring' });
    expect(res.totalMatches).toBe(0);
    // Query 'Auth' with structured filter + extension predicate (AND).
    const res2 = await index.search('Auth', {
      filter: { kind: 'symbol' },
    });
    expect(res2.totalMatches).toBeGreaterThan(0);
    // Per-query predicate ANDs with index predicate.
    const res3 = await index.search('Auth', {
      filter: (d: Doc) => d.year >= 2024,
    });
    // Must satisfy both: kind symbol AND year>=2024 -> docs 2,4.
    expect(res3.results.every((r) => (r.doc as Doc).kind === 'symbol' && (r.doc as Doc).year >= 2024)).toBe(true);
    index.destroy();
  });

  test('postProcess transforms final results', async () => {
    const index = await DocumentIndex.create(DOCS, baseOpts());
    const res = await index.search('Auth', {
      mode: 'substring',
      limit: 10,
      extensions: {
        postProcess: (results) => results.filter((r) => r.id !== '1').slice(0, 1),
      },
    });
    expect(res.results.length).toBe(1);
    expect(res.results[0]?.id).not.toBe('1');
    // postProcess returning non-array throws.
    await expect(
      index.search('Auth', { extensions: { postProcess: (() => 42) as never } })
    ).rejects.toBeInstanceOf(TypeError);
    index.destroy();
  });

  test('index-level + per-query hooks merge (query wins per key)', async () => {
    const index = await DocumentIndex.create(DOCS, baseOpts({
      extensions: {
        scoringHook: ((doc: Doc, s: number) => s + 1) as never,
        filterPredicate: ((d: Doc) => d.kind === 'symbol') as never,
      },
    }));
    // Per-query scoringHook replaces index scoringHook; filterPredicate stays.
    const res = await index.search('Auth', {
      extensions: { scoringHook: ((doc: Doc, s: number) => s + 100) as never },
    });
    expect(res.results.length).toBeGreaterThan(0);
    expect(res.results.every((r) => (r.doc as Doc).kind === 'symbol')).toBe(true);
    index.destroy();
  });

  test('extensions validation fail-closed on empty query/corpus', async () => {
    const index = await DocumentIndex.create(DOCS, baseOpts());
    await expect(
      index.search('Auth', { extensions: { tokenizer: 1 as never } })
    ).rejects.toBeInstanceOf(TypeError);
    const empty = await DocumentIndex.create([], baseOpts());
    await expect(
      empty.search('Auth', { extensions: { scoringHook: 1 as never } })
    ).rejects.toBeInstanceOf(TypeError);
    index.destroy();
    empty.destroy();
  });
});

describe('persistence safety: hookIds + fail-closed restore', () => {
  test('serialize records hookIds, restore requires handlers', async () => {
    function recencyBoost(doc: Doc, s: number) { return doc.year >= 2024 ? s + 100 : s; }
    const index = await DocumentIndex.create(DOCS, baseOpts({
      extensions: { scoringHook: recencyBoost as never },
    }));
    const buf = index.serialize();
    const snap = deserializeDocumentSnapshot(buf);
    expect(snap.schema.hookIds?.scoringHook).toBe('recencyBoost');

    // Restore without handlers throws IncompatibleHookError.
    const { restoreDocumentIndex } = await import('../src/index');
    await expect(restoreDocumentIndex(buf)).rejects.toBeInstanceOf(IncompatibleHookError);
    // Restore with matching handler succeeds and preserves behavior.
    const restored = await restoreDocumentIndex<Doc>(buf, {
      options: { extensions: { scoringHook: recencyBoost as never } },
    });
    const r1 = await index.search('Auth', { mode: 'substring' });
    const r2 = await restored.search('Auth', { mode: 'substring' });
    expect(r2.results.map((r) => [r.id, r.score])).toEqual(r1.results.map((r) => [r.id, r.score]));
    index.destroy();
    restored.destroy();
  });

  test('mismatched hookId throws IncompatibleHookError', async () => {
    function scorerA(doc: Doc, s: number) { return s + 1; }
    function scorerB(doc: Doc, s: number) { return s + 2; }
    const index = await DocumentIndex.create(DOCS, baseOpts({
      extensions: { scoringHook: scorerA as never },
    }));
    const buf = index.serialize();
    const { restoreDocumentIndex } = await import('../src/index');
    await expect(
      restoreDocumentIndex<Doc>(buf, { options: { extensions: { scoringHook: scorerB as never } } })
    ).rejects.toBeInstanceOf(IncompatibleHookError);
    index.destroy();
  });

  test('pre-M6 snapshots without hookIds restore without handlers', async () => {
    const index = await DocumentIndex.create(DOCS, baseOpts());
    const buf = index.serialize();
    const snap = deserializeDocumentSnapshot(buf);
    expect(snap.schema.hookIds).toBeUndefined();
    const { restoreDocumentIndex } = await import('../src/index');
    const restored = await restoreDocumentIndex<Doc>(buf);
    expect(restored.getStats().docCount).toBe(4);
    index.destroy();
    restored.destroy();
  });

  test('instance restore() enforces hook guard', async () => {
    function tok(s: string) { return [s]; }
    const src = await DocumentIndex.create(DOCS, baseOpts({
      extensions: { tokenizer: tok },
    }));
    const buf = src.serialize();
    const dst = await DocumentIndex.create(DOCS, baseOpts());
    await expect(dst.restore(buf)).rejects.toBeInstanceOf(IncompatibleHookError);
    await dst.restore(buf, { options: { extensions: { tokenizer: tok } } });
    expect(dst.getExtensions()?.tokenizer).toBe(tok as never);
    src.destroy();
    dst.destroy();
  });

  test('closures never serialized (schema contains IDs, not functions)', async () => {
    const index = await DocumentIndex.create(DOCS, baseOpts({
      extensions: {
        tokenizer: codeTokenizer,
        scoringHook: ((d: Doc, s: number) => s) as never,
        filterPredicate: ((d: Doc) => true) as never,
        postProcess: ((r) => r) as never,
      },
    }));
    const buf = index.serialize();
    const headerBytes = 48;
    const dv = new DataView(buf, 0, headerBytes);
    const schemaLen = dv.getUint32(36, true);
    const schemaStr = new TextDecoder().decode(new Uint8Array(buf, headerBytes, schemaLen));
    expect(schemaStr).toContain('hookIds');
    expect(schemaStr).not.toContain('=>');
    expect(schemaStr).not.toContain('function');
    const snap = deserializeDocumentSnapshot(buf);
    expect(snap.schema.hookIds?.tokenizer).toBe('codeTokenizer');
    index.destroy();
  });
});

describe('worker boundary + portability', () => {
  test('SearchWorkerClient rejects extensions fail-closed', async () => {
    const { SearchWorkerClient } = await import('../src/index');
    const client = new SearchWorkerClient<Doc>({ worker: (() => { throw new Error('no worker'); }) as never });
    // init path validates before touching the worker.
    await expect(
      client.init(DOCS, baseOpts({ extensions: { tokenizer: codeTokenizer } as never }) as never)
    ).rejects.toBeInstanceOf(IncompatibleHookError);
    await client.destroy();
  });

  test('zero unguarded DOM references in M6 modules', async () => {
    const files = [
      'packages/webgpu-search/src/extensions.ts',
      'packages/webgpu-search/src/document-index.ts',
      'packages/webgpu-search/src/persistence.ts',
      'packages/webgpu-search/src/cpu-reference.ts',
      'packages/webgpu-search/src/highlight.ts',
    ];
    for (const f of files) {
      const raw = await readFile(f, 'utf8');
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
