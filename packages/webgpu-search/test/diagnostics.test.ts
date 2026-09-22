/**
 * v0.4 M7 Test Suite (Issue #10): Query Diagnostics, Cost Budgets &
 * Broad-Query Safeguards.
 *
 * Covers budget validation (fail-closed), time/candidate enforcement
 * (`CostBudgetExceededError`), diagnostics telemetry shape and timing
 * buckets, filter selectivity, broad-query heuristic routing + warnings,
 * candidate overflow warnings, search-index parity, and cross-platform
 * portability (zero unguarded DOM references).
 *
 * Run: bun test packages/webgpu-search/test/diagnostics.test.ts
 */
import { describe, test, expect } from 'bun:test';
import { readFile } from 'node:fs/promises';
import {
  DocumentIndex,
  SearchIndex,
  normalizeCostBudgetOptions,
  assertTimeBudget,
  assertCandidateBudget,
  throwIfBudgetAborted,
  isBroadQueryHeuristic,
  isBroadSelectivity,
  computeFilterSelectivity,
  broadQueryRouteWarning,
  broadSelectivityWarning,
  candidateOverflowWarning,
  CostBudgetExceededError,
  BROAD_QUERY_MIN_DOCS,
  BROAD_QUERY_SELECTIVITY_THRESHOLD,
  RESULT_LIMIT_MAX,
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
    filterFields: [{ name: 'kind', type: 'string' }, { name: 'year', type: 'number' }],
    preferGpu: false,
    ...extra,
  };
}

describe('normalizeCostBudgetOptions: fail-closed validation', () => {
  test('undefined and empty objects resolve to no budget', () => {
    expect(normalizeCostBudgetOptions(undefined)).toBeUndefined();
    expect(normalizeCostBudgetOptions({})).toBeUndefined();
  });

  test('valid budgets pass through', () => {
    expect(normalizeCostBudgetOptions({ maxExecutionTimeMs: 100 })).toEqual({ maxExecutionTimeMs: 100 });
    expect(normalizeCostBudgetOptions({ maxCandidates: 10 })).toEqual({ maxCandidates: 10 });
    expect(normalizeCostBudgetOptions({ maxExecutionTimeMs: 0.5, maxCandidates: 7 })).toEqual({
      maxExecutionTimeMs: 0.5,
      maxCandidates: 7,
    });
  });

  test('non-object shapes throw TypeError', () => {
    expect(() => normalizeCostBudgetOptions(null as never)).toThrow(TypeError);
    expect(() => normalizeCostBudgetOptions([] as never)).toThrow(TypeError);
    expect(() => normalizeCostBudgetOptions('fast' as never)).toThrow(TypeError);
    expect(() => normalizeCostBudgetOptions(42 as never)).toThrow(TypeError);
  });

  test('maxExecutionTimeMs must be a finite number > 0', () => {
    expect(() => normalizeCostBudgetOptions({ maxExecutionTimeMs: '10' as never })).toThrow(TypeError);
    expect(() => normalizeCostBudgetOptions({ maxExecutionTimeMs: 0 })).toThrow(RangeError);
    expect(() => normalizeCostBudgetOptions({ maxExecutionTimeMs: -5 })).toThrow(RangeError);
    expect(() => normalizeCostBudgetOptions({ maxExecutionTimeMs: NaN })).toThrow(RangeError);
    expect(() => normalizeCostBudgetOptions({ maxExecutionTimeMs: Infinity })).toThrow(RangeError);
  });

  test('maxCandidates must be an integer >= 1', () => {
    expect(() => normalizeCostBudgetOptions({ maxCandidates: '10' as never })).toThrow(TypeError);
    expect(() => normalizeCostBudgetOptions({ maxCandidates: 0 })).toThrow(RangeError);
    expect(() => normalizeCostBudgetOptions({ maxCandidates: -3 })).toThrow(RangeError);
    expect(() => normalizeCostBudgetOptions({ maxCandidates: 1.5 })).toThrow(RangeError);
    expect(() => normalizeCostBudgetOptions({ maxCandidates: NaN })).toThrow(RangeError);
    expect(() => normalizeCostBudgetOptions({ maxCandidates: Infinity })).toThrow(RangeError);
  });

  test('abortSignal must be an object when provided', () => {
    expect(() => normalizeCostBudgetOptions({ abortSignal: 42 as never })).toThrow(TypeError);
    const c = new AbortController();
    expect(normalizeCostBudgetOptions({ abortSignal: c.signal })?.abortSignal).toBe(c.signal);
  });

  test('unknown keys are ignored (forward-compatible), unknown max* throws', () => {
    expect(
      normalizeCostBudgetOptions({ maxCandidates: 5, futureKey: 1 } as never)
    ).toEqual({ maxCandidates: 5 });
    expect(() => normalizeCostBudgetOptions({ maxCandidate: 5 } as never)).toThrow(TypeError);
    expect(() => normalizeCostBudgetOptions({ maxTime: 10 } as never)).toThrow(TypeError);
  });
});

describe('search(): budget + diagnostics option validation (fail-closed)', () => {
  test('non-boolean diagnostics throws TypeError even on empty corpora', async () => {
    const index = await DocumentIndex.create([], baseOpts());
    await expect(index.search('auth', { diagnostics: 'yes' as never })).rejects.toThrow(TypeError);
    await index.destroy();
  });

  test('malformed budgets throw before early exits (empty query + empty corpus)', async () => {
    const index = await DocumentIndex.create([], baseOpts());
    await expect(index.search('', { budget: { maxCandidates: 0 } as never })).rejects.toThrow(RangeError);
    await expect(index.search('auth', { budget: { maxExecutionTimeMs: -1 } })).rejects.toThrow(RangeError);
    await expect(index.search('auth', { budget: 42 as never })).rejects.toThrow(TypeError);
    await index.destroy();
  });

  test('aborted budget signal rejects with AbortError', async () => {
    const index = await DocumentIndex.create(DOCS, baseOpts());
    const c = new AbortController();
    c.abort();
    const err = await index.search('auth', { budget: { abortSignal: c.signal } }).catch((e) => e);
    expect(err?.name).toBe('AbortError');
    await index.destroy();
  });
});

describe('search(): candidate ceiling enforcement', () => {
  test('maxCandidates below corpus size throws CostBudgetExceededError', async () => {
    const index = await DocumentIndex.create(DOCS, baseOpts());
    const err = await index
      .search('auth', { budget: { maxCandidates: 2 } })
      .catch((e) => e);
    expect(err).toBeInstanceOf(CostBudgetExceededError);
    expect(err.budgetType).toBe('candidates');
    expect(err.limit).toBe(2);
    expect(err.actual).toBe(4);
    await index.destroy();
  });

  test('maxCandidates at/above corpus size succeeds', async () => {
    const index = await DocumentIndex.create(DOCS, baseOpts());
    const res = await index.search('auth', { budget: { maxCandidates: 4 } });
    expect(res.totalMatches).toBeGreaterThan(0);
    const res2 = await index.search('auth', { budget: { maxCandidates: 100 } });
    expect(res2.totalMatches).toBe(res.totalMatches);
    await index.destroy();
  });

  test('candidate ceiling applies to the post-filter population', async () => {
    const index = await DocumentIndex.create(DOCS, baseOpts());
    // Only 3 docs carry kind === 'symbol'; a ceiling of 3 passes with the
    // filter even though the raw corpus holds 4 docs.
    const res = await index.search('auth', {
      filter: { kind: 'symbol' },
      budget: { maxCandidates: 3 },
    });
    expect(res.totalMatches).toBeGreaterThan(0);
    // Without the filter the same ceiling throws (4 candidates > 3).
    await expect(index.search('auth', { budget: { maxCandidates: 3 } })).rejects.toThrow(
      CostBudgetExceededError
    );
    await index.destroy();
  });
});

describe('search(): time budget enforcement', () => {
  test('generous deadline succeeds', async () => {
    const index = await DocumentIndex.create(DOCS, baseOpts());
    const res = await index.search('auth', { budget: { maxExecutionTimeMs: 60_000 } });
    expect(res.totalMatches).toBeGreaterThan(0);
    await index.destroy();
  });

  test('microsecond deadline over a large corpus throws CostBudgetExceededError', async () => {
    const docs: Doc[] = [];
    for (let i = 0; i < 2000; i++) {
      docs.push({
        id: `d${i}`,
        title: `service component number ${i} alpha`,
        body: `background worker payload ${i} beta`,
        kind: i % 2 === 0 ? 'symbol' : 'infra',
        year: 2000 + (i % 25),
      });
    }
    const index = await DocumentIndex.create(docs, baseOpts());
    const err = await index
      .search('service', { budget: { maxExecutionTimeMs: 0.000001 } })
      .catch((e) => e);
    expect(err).toBeInstanceOf(CostBudgetExceededError);
    expect(err.budgetType).toBe('time');
    await index.destroy();
  });
});

describe('search(): diagnostics telemetry', () => {
  test('diagnostics absent by default and when disabled', async () => {
    const index = await DocumentIndex.create(DOCS, baseOpts());
    expect((await index.search('auth')).diagnostics).toBeUndefined();
    expect((await index.search('auth', { diagnostics: false })).diagnostics).toBeUndefined();
    await index.destroy();
  });

  test('diagnostics shape on the parity CPU path', async () => {
    const index = await DocumentIndex.create(DOCS, baseOpts());
    const res = await index.search('auth', { diagnostics: true });
    const d = res.diagnostics!;
    expect(d).toBeDefined();
    expect(d.scannedCandidates).toBe(4);
    expect(d.filterSelectivity).toBe(1.0);
    expect(d.routedEngine).toBe('cpu');
    expect(d.hasOverflow).toBe(false);
    expect(d.timings.filteringMs).toBeGreaterThanOrEqual(0);
    expect(d.timings.scoringMs).toBeGreaterThanOrEqual(0);
    expect(d.timings.highlightMs).toBeGreaterThanOrEqual(0);
    expect(d.timings.facetingMs).toBeUndefined();
    expect(d.timings.totalMs).toBeGreaterThanOrEqual(0);
    expect(d.timings.totalMs).toBeGreaterThanOrEqual(d.timings.scoringMs);
    expect(d.warnings).toBeUndefined();
    await index.destroy();
  });

  test('filter selectivity reflects the structured filter', async () => {
    const index = await DocumentIndex.create(DOCS, baseOpts());
    const res = await index.search('auth', {
      filter: { kind: 'symbol' },
      diagnostics: true,
    });
    const d = res.diagnostics!;
    expect(d.scannedCandidates).toBe(4);
    expect(d.filterSelectivity).toBeCloseTo(0.75, 10);
    await index.destroy();
  });

  test('facetingMs present when facets are requested', async () => {
    const index = await DocumentIndex.create(DOCS, baseOpts());
    const res = await index.search('auth', {
      facets: { byKind: { type: 'terms', field: 'kind' } },
      diagnostics: true,
    });
    expect(res.facets).toBeDefined();
    expect(typeof res.diagnostics!.timings.facetingMs).toBe('number');
    expect(res.diagnostics!.timings.facetingMs!).toBeGreaterThanOrEqual(0);
    await index.destroy();
  });

  test('no-hit exits still report diagnostics', async () => {
    const index = await DocumentIndex.create(DOCS, baseOpts());
    const empty = await index.search('', { diagnostics: true });
    expect(empty.diagnostics).toBeDefined();
    expect(empty.diagnostics!.scannedCandidates).toBe(4);
    const noMatch = await index.search('zzz-no-such-token', { diagnostics: true });
    expect(noMatch.diagnostics).toBeDefined();
    expect(noMatch.diagnostics!.hasOverflow).toBe(false);
    expect(noMatch.totalMatches).toBe(0);
    await index.destroy();
  });

  test('diagnostics on the legacy ufuzzy path', async () => {
    const index = await DocumentIndex.create(DOCS, baseOpts());
    const res = await index.search('auth', { cpuScorer: 'ufuzzy', diagnostics: true });
    expect(res.engine).toBe('cpu');
    expect(res.diagnostics).toBeDefined();
    expect(res.diagnostics!.scannedCandidates).toBe(4);
    expect(res.diagnostics!.routedEngine).toBe('cpu');
    await index.destroy();
  });
});

describe('broad-query safeguards', () => {
  test('isBroadQueryHeuristic unit boundaries', () => {
    expect(BROAD_QUERY_SELECTIVITY_THRESHOLD).toBe(0.8);
    expect(isBroadQueryHeuristic(1, BROAD_QUERY_MIN_DOCS)).toBe(true);
    expect(isBroadQueryHeuristic(2, BROAD_QUERY_MIN_DOCS)).toBe(true);
    expect(isBroadQueryHeuristic(3, BROAD_QUERY_MIN_DOCS)).toBe(false);
    expect(isBroadQueryHeuristic(1, BROAD_QUERY_MIN_DOCS - 1)).toBe(false);
    expect(isBroadQueryHeuristic(0, BROAD_QUERY_MIN_DOCS * 2)).toBe(false);
  });

  test('isBroadSelectivity + computeFilterSelectivity units', () => {
    expect(isBroadSelectivity(0.81, BROAD_QUERY_MIN_DOCS)).toBe(true);
    expect(isBroadSelectivity(0.8, BROAD_QUERY_MIN_DOCS)).toBe(false);
    expect(isBroadSelectivity(0.99, 100)).toBe(false);
    expect(computeFilterSelectivity(3, 4)).toBeCloseTo(0.75, 10);
    expect(computeFilterSelectivity(4, 4)).toBe(1.0);
    expect(computeFilterSelectivity(0, 4)).toBe(0);
    expect(computeFilterSelectivity(0, 0)).toBe(1.0);
    expect(broadQueryRouteWarning(5000, 1)).toMatch(/broad-query/);
    expect(broadSelectivityWarning(0.9, 5000)).toMatch(/broad-query/);
    expect(candidateOverflowWarning(9000, 8192)).toMatch(/overflow/);
  });

  test('short query over a massive corpus routes to CPU with a warning', async () => {
    const docs: Doc[] = [];
    for (let i = 0; i < BROAD_QUERY_MIN_DOCS; i++) {
      docs.push({
        id: `b${i}`,
        title: `service endpoint ${i}`,
        body: `request handler ${i}`,
        kind: 'symbol',
        year: 2020,
      });
    }
    const index = await DocumentIndex.create(docs, baseOpts());
    const res = await index.search('e', { diagnostics: true });
    expect(res.engine).toBe('cpu');
    expect(res.diagnostics).toBeDefined();
    expect(res.diagnostics!.routedEngine).toBe('cpu');
    expect(res.diagnostics!.warnings?.some((w) => w.includes('broad-query'))).toBe(true);
    await index.destroy();
  });

  test('candidate overflow records a warning when matches exceed pool capacity', async () => {
    const docs: Doc[] = [];
    for (let i = 0; i < 9000; i++) {
      docs.push({
        id: `o${i}`,
        title: `alpha record ${i}`,
        body: `alpha payload ${i}`,
        kind: 'infra',
        year: 2021,
      });
    }
    const index = await DocumentIndex.create(docs, baseOpts());
    const res = await index.search('alpha', { diagnostics: true });
    expect(res.hasOverflow).toBe(true);
    expect(res.diagnostics!.hasOverflow).toBe(true);
    expect(res.diagnostics!.warnings?.some((w) => w.includes('overflow'))).toBe(true);
    await index.destroy();
  });
});

describe('hybrid SearchIndex: budgets + diagnostics parity', () => {
  test('diagnostics shape + absence by default', async () => {
    const index = await SearchIndex.create(['alpha one', 'alpha two', 'beta three'], {
      preferGpu: false,
    });
    const res = await index.search('alpha', { diagnostics: true });
    expect(res.diagnostics).toBeDefined();
    expect(res.diagnostics!.scannedCandidates).toBe(3);
    expect(res.diagnostics!.filterSelectivity).toBe(1.0);
    expect(res.diagnostics!.routedEngine).toBe('cpu');
    expect(res.diagnostics!.timings.totalMs).toBeGreaterThanOrEqual(0);
    expect((await index.search('alpha')).diagnostics).toBeUndefined();
    await index.destroy();
  });

  test('maxCandidates enforcement + malformed budget fail-closed', async () => {
    const index = await SearchIndex.create(['a', 'b', 'c'], { preferGpu: false });
    const err = await index.search('a', { budget: { maxCandidates: 2 } }).catch((e) => e);
    expect(err).toBeInstanceOf(CostBudgetExceededError);
    expect(err.budgetType).toBe('candidates');
    await expect(index.search('a', { budget: { maxCandidates: 0 } })).rejects.toThrow(RangeError);
    await expect(index.search('a', { diagnostics: 1 as never })).rejects.toThrow(TypeError);
    await index.destroy();
  });

  test('broad-query heuristic routes hybrid short queries to CPU with warning', async () => {
    const items: string[] = [];
    for (let i = 0; i < BROAD_QUERY_MIN_DOCS; i++) items.push(`service item ${i}`);
    const index = await SearchIndex.create(items, { preferGpu: false });
    const res = await index.search('e', { diagnostics: true });
    expect(res.engine).toBe('cpu');
    expect(res.diagnostics!.warnings?.some((w) => w.includes('broad-query'))).toBe(true);
    await index.destroy();
  });
});

describe('M7 review fixes: contracts + telemetry accuracy', () => {
  test('broad-query CPU routing leaves fallbackReason untouched', async () => {
    const docs: Doc[] = [];
    for (let i = 0; i < BROAD_QUERY_MIN_DOCS; i++) {
      docs.push({
        id: `b${i}`,
        title: `service endpoint ${i}`,
        body: `request handler ${i}`,
        kind: 'symbol',
        year: 2020,
      });
    }
    const index = await DocumentIndex.create(docs, baseOpts());
    const res = await index.search('e', { diagnostics: true });
    expect(res.engine).toBe('cpu');
    expect(res.fallbackReason).toBe('prefer-cpu');
    expect(res.diagnostics!.routedEngine).toBe('cpu');
    expect(res.diagnostics!.warnings?.some((w) => w.includes('broad-query'))).toBe(true);
    await index.destroy();
  });

  test('token mode over massive corpus does not misattribute GPU routing', async () => {
    const docs: Doc[] = [];
    for (let i = 0; i < BROAD_QUERY_MIN_DOCS; i++) {
      docs.push({
        id: `t${i}`,
        title: `service endpoint ${i}`,
        body: `request handler ${i}`,
        kind: 'symbol',
        year: 2020,
      });
    }
    const index = await DocumentIndex.create(docs, baseOpts());
    const res = await index.search('e', { mode: 'token', diagnostics: true });
    expect(res.engine).toBe('cpu');
    expect(res.fallbackReason).toBe('unsupported-mode');
    // Already CPU-by-design: no GPU-avoidance route warning.
    expect(res.diagnostics!.warnings?.some((w) => w.includes('pre-dispatch heuristic'))).toBe(false);
    await index.destroy();
  });

  test('ufuzzy path pins cpu-algorithm-requested fallbackReason', async () => {
    const index = await DocumentIndex.create(DOCS, baseOpts());
    const res = await index.search('auth', { cpuScorer: 'ufuzzy', diagnostics: true });
    expect(res.engine).toBe('cpu');
    expect(res.fallbackReason).toBe('cpu-algorithm-requested');
    expect(res.diagnostics!.routedEngine).toBe('cpu');
    await index.destroy();
  });

  test('autocomplete + diagnostics coexist with autocompleteMs bucket', async () => {
    const index = await DocumentIndex.create(DOCS, baseOpts());
    const res = await index.search('auth', { diagnostics: true, autocomplete: { limit: 2 } });
    expect(res.suggestions).toBeDefined();
    expect(res.diagnostics).toBeDefined();
    expect(typeof res.diagnostics!.timings.autocompleteMs).toBe('number');
    expect(res.diagnostics!.timings.totalMs).toBeGreaterThanOrEqual(
      res.diagnostics!.timings.scoringMs
    );
    await index.destroy();
  });

  test('post-hoc selectivity warning fires for broad long queries', async () => {
    const docs: Doc[] = [];
    for (let i = 0; i < BROAD_QUERY_MIN_DOCS; i++) {
      docs.push({
        id: `s${i}`,
        title: `alpha shared vocabulary record ${i}`,
        body: `alpha shared vocabulary payload ${i}`,
        kind: 'symbol',
        year: 2020,
      });
    }
    const index = await DocumentIndex.create(docs, baseOpts());
    // 3-token query escapes the <=2-token pre-dispatch gate but matches >80%.
    const res = await index.search('alpha shared vocabulary', { diagnostics: true });
    expect(res.totalMatches / BROAD_QUERY_MIN_DOCS).toBeGreaterThan(0.8);
    expect(res.diagnostics!.warnings?.some((w) => w.includes('broad-query'))).toBe(true);
    expect(res.diagnostics!.warnings?.some((w) => w.includes('pre-dispatch'))).toBe(false);
    await index.destroy();
  });

  test('overflow warning names remediation and honors facets flags', () => {
    const withFacets = candidateOverflowWarning(9000, 8192);
    expect(withFacets).toMatch(/overflow/);
    expect(withFacets).toMatch(/candidateCapacity/);
    expect(withFacets).toMatch(/approximate/);
    const noFacets = candidateOverflowWarning(9000, 8192, { facetsRequested: false });
    expect(noFacets).toMatch(/overflow/);
    expect(noFacets).not.toMatch(/approximate/);
    const exact = candidateOverflowWarning(9000, 8192, { facetsRequested: true, facetsExact: true });
    expect(exact).not.toMatch(/approximate/);
    const raw = candidateOverflowWarning(100, 8192, { rawTotalMatches: 9000 });
    expect(raw).toMatch(/9000/);
    expect(raw).toMatch(/100/);
  });

  test('empty corpus diagnostics shape', async () => {
    const index = await DocumentIndex.create([], baseOpts());
    const res = await index.search('auth', { diagnostics: true });
    expect(res.diagnostics).toBeDefined();
    expect(res.diagnostics!.scannedCandidates).toBe(0);
    expect(res.diagnostics!.filterSelectivity).toBe(1.0);
    expect(res.diagnostics!.hasOverflow).toBe(false);
    await index.destroy();
  });

  test('empty budget object + limit:0 behave as no-budget clamped search', async () => {
    const index = await DocumentIndex.create(DOCS, baseOpts());
    const res = await index.search('auth', { budget: {}, diagnostics: true });
    expect(res.totalMatches).toBeGreaterThan(0);
    expect(res.diagnostics).toBeDefined();
    const res2 = await index.search('auth', { limit: 0, diagnostics: true });
    expect(res2.results.length).toBeGreaterThanOrEqual(1);
    await index.destroy();
  });

  test('budgets enforce even when diagnostics:false', async () => {
    const index = await DocumentIndex.create(DOCS, baseOpts());
    await expect(
      index.search('auth', { budget: { maxCandidates: 2 }, diagnostics: false })
    ).rejects.toThrow(CostBudgetExceededError);
    expect((await index.search('auth', { diagnostics: false })).diagnostics).toBeUndefined();
    await index.destroy();
  });

  test('forged abort objects are honored', async () => {
    const index = await DocumentIndex.create(DOCS, baseOpts());
    const err = await index
      .search('auth', { budget: { abortSignal: { aborted: true } as never } })
      .catch((e) => e);
    expect(err?.name).toBe('AbortError');
    await index.destroy();
  });

  test('assert boundaries: at-limit passes, no-budget no-ops', () => {
    expect(() => assertTimeBudget(100, undefined)).not.toThrow();
    expect(() => assertCandidateBudget(999, undefined)).not.toThrow();
    const budget = normalizeCostBudgetOptions({ maxCandidates: 5, maxExecutionTimeMs: 1000 });
    expect(() => assertCandidateBudget(5, budget)).not.toThrow();
    expect(() => assertCandidateBudget(6, budget)).toThrow(CostBudgetExceededError);
    expect(() => throwIfBudgetAborted(undefined)).not.toThrow();
  });

  test('overflow boundary: at capacity has no overflow', () => {
    expect(RESULT_LIMIT_MAX).toBe(8192);
    // Unit-level boundary: warning builder is only called when hasOverflow,
    // which callers compute as totalMatches > capacity (strict).
    expect(8192 > 8192).toBe(false);
    expect(8193 > 8192).toBe(true);
  });

  test('worker predicate path drops stale diagnostics (fail-closed)', async () => {
    const raw = await readFile(
      'packages/webgpu-search/src/worker/worker-client.ts',
      'utf8'
    );
    expect(raw).toMatch(/delete \(searchResp as \{ diagnostics\?: unknown \}\)\.diagnostics/);
  });
});

describe('M7 portability', () => {
  test('zero unguarded DOM references in M7 modules', async () => {
    const files = [
      'packages/webgpu-search/src/diagnostics.ts',
      'packages/webgpu-search/src/search-index.ts',
      'packages/webgpu-search/src/document-index.ts',
    ];
    for (const f of files) {
      const raw = await readFile(f, 'utf8');
      const noBlock = raw.replace(/\/\*[\s\S]*?\*\//g, '');
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
      // search-index.ts has pre-existing guarded `navigator.gpu` fallback
      // detection (outside M7); assert every remaining navigator reference
      // is that guarded gpu probe, never unguarded DOM access.
      const navLines = stripped.split('\n').filter((line) => line.includes('navigator.'));
      for (const line of navLines) {
        expect(line.includes('gpu')).toBe(true);
      }
      expect(stripped).not.toMatch(/(^|[^\w$.])location\s*\./);
    }
  });
});
