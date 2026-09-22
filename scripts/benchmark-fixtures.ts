/**
 * Frozen benchmark fixtures (Issue #11 Phase 5).
 *
 * Single source of truth for the reproducible benchmark matrix. Both
 * `scripts/bench-snapshot-matrix.ts` (full headless matrix) and
 * `scripts/test-benchmarks.ts` (fast pin) import this module so the
 * versioned corpora, record sizes, queries, modes, filters, and facets
 * cannot drift between the runner, the checked-in baseline, and the docs.
 *
 * Plan ref: `docs/ISSUE-11-PLAN.md` Phase 5. Normative doc:
 * `docs/benchmarks.md`.
 *
 * Portability: DOM-free (no `window` / `document`); safe in workers,
 * Node.js, Bun, and SSR.
 */

/** Frozen fixture revision. Bump only with a deliberate re-baseline. */
export const BENCHMARK_FIXTURE_VERSION = '1.0.0';

/**
 * Deterministic anchor for structured-log timestamps.
 * `generateStructuredLogs` defaults to `Date.now()`; the matrix passes this
 * fixed base so snapshot bytes are reproducible run-to-run. Recording the
 * anchor here pins it for hosts re-running the matrix.
 */
export const BENCHMARK_LOG_BASE_TIME_MS = Date.UTC(2026, 0, 1, 0, 0, 0);

export interface BenchmarkFixtureField {
  name: string;
  weight: number;
}

export interface BenchmarkFixtureFilterField {
  name: string;
  type?: 'number';
}

export interface BenchmarkFixtureScenario {
  scenario: string;
  corpus: 'monaco' | 'logs';
  docs: number;
  idField: string;
  fields: BenchmarkFixtureField[];
  filterFields: BenchmarkFixtureFilterField[];
  operations: string[];
}

export interface BenchmarkFixtureQuery {
  operation: string;
  query: string;
  mode: 'prefix' | 'fuzzy';
  limit: number;
  filter?: Record<string, unknown>;
  facets?: Record<string, unknown>;
  autocomplete?: { mode: 'prefix'; limit: number };
}

/** Frozen corpora + index shapes (must match the matrix runner). */
export const BENCHMARK_FIXTURES: BenchmarkFixtureScenario[] = [
  {
    scenario: 'ide-symbols-5k',
    corpus: 'monaco',
    docs: 5000,
    idField: 'id',
    fields: [
      { name: 'filename', weight: 3.0 },
      { name: 'symbols', weight: 2.0 },
      { name: 'path', weight: 1.0 },
      { name: 'description', weight: 0.5 },
    ],
    filterFields: [{ name: 'type' }, { name: 'language' }],
    operations: [
      'index-build',
      'prefix-search',
      'prefix-search+type-filter',
      'autocomplete',
      'fuzzy-search+type-facets',
    ],
  },
  {
    scenario: 'log-grid-10k',
    corpus: 'logs',
    docs: 10_000,
    idField: 'id',
    fields: [
      { name: 'message', weight: 2.0 },
      { name: 'service', weight: 1.5 },
      { name: 'level', weight: 1.0 },
      { name: 'traceId', weight: 1.2 },
    ],
    filterFields: [
      { name: 'level' },
      { name: 'service' },
      { name: 'timestamp' },
      { name: 'latencyMs', type: 'number' },
    ],
    operations: [
      'index-build',
      'fuzzy-search',
      'fuzzy-search+level+latency-filter',
      'fuzzy-search+facets',
      'snapshot-serialize',
      'snapshot-restore',
    ],
  },
  {
    scenario: 'log-grid-50k',
    corpus: 'logs',
    docs: 50_000,
    idField: 'id',
    fields: [
      { name: 'message', weight: 2.0 },
      { name: 'service', weight: 1.5 },
      { name: 'level', weight: 1.0 },
      { name: 'traceId', weight: 1.2 },
    ],
    filterFields: [
      { name: 'level' },
      { name: 'service' },
      { name: 'timestamp' },
      { name: 'latencyMs', type: 'number' },
    ],
    operations: [
      'index-build',
      'fuzzy-search',
      'fuzzy-search+level+latency-filter',
      'fuzzy-search+facets',
    ],
  },
  {
    scenario: 'log-grid-100k',
    corpus: 'logs',
    docs: 100_000,
    idField: 'id',
    fields: [
      { name: 'message', weight: 2.0 },
      { name: 'service', weight: 1.5 },
      { name: 'level', weight: 1.0 },
      { name: 'traceId', weight: 1.2 },
    ],
    filterFields: [
      { name: 'level' },
      { name: 'service' },
      { name: 'timestamp' },
      { name: 'latencyMs', type: 'number' },
    ],
    operations: [
      'index-build',
      'fuzzy-search',
      'fuzzy-search+level+latency-filter',
      'fuzzy-search+facets',
    ],
  },
];

/** Frozen queries / modes / limits / filters / facets per operation. */
export const BENCHMARK_QUERIES: BenchmarkFixtureQuery[] = [
  { operation: 'prefix-search', query: 'compute', mode: 'prefix', limit: 20 },
  {
    operation: 'prefix-search+type-filter',
    query: 'compute',
    mode: 'prefix',
    limit: 20,
    filter: { type: 'shader' },
  },
  {
    operation: 'autocomplete',
    query: 'comp',
    mode: 'prefix',
    limit: 5,
    autocomplete: { mode: 'prefix', limit: 5 },
  },
  {
    operation: 'fuzzy-search+type-facets',
    query: 'service',
    mode: 'fuzzy',
    limit: 20,
    facets: { byType: { type: 'terms', field: 'type', limit: 10 } },
  },
  { operation: 'fuzzy-search', query: 'timeout', mode: 'fuzzy', limit: 50 },
  {
    operation: 'fuzzy-search+level+latency-filter',
    query: 'timeout',
    mode: 'fuzzy',
    limit: 50,
    filter: { level: 'ERROR', latencyMs: { gte: 300 } },
  },
  {
    operation: 'fuzzy-search+facets',
    query: 'timeout',
    mode: 'fuzzy',
    limit: 50,
    facets: {
      byLevel: { type: 'terms', field: 'level', limit: 10 },
      byLatency: {
        type: 'range',
        field: 'latencyMs',
        ranges: [
          { to: 50, key: 'fast' },
          { from: 50, to: 300, key: 'normal' },
          { from: 300, key: 'slow' },
        ],
      },
    },
  },
];

/** Every frozen (scenario, operation) pair the baseline must contain. */
export const BENCHMARK_EXPECTED_ROWS: Array<{ scenario: string; operation: string }> =
  BENCHMARK_FIXTURES.flatMap((f) =>
    f.operations.map((operation) => ({ scenario: f.scenario, operation })),
  );
