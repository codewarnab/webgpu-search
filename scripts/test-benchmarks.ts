/**
 * Benchmark fixtures pin (Issue #11 Phase 5).
 *
 * Verifies the frozen reproducible-benchmark contract without running the
 * full 100k matrix:
 * - `scripts/benchmark-fixtures.ts` version + frozen scenario/query spec.
 * - Generator determinism (`generateMonacoRecords`, `generateStructuredLogs`
 *   with the frozen log-time anchor).
 * - Frozen query shapes serve live on small smoke indexes (ordering,
 *   filters, facets, autocomplete, snapshot serialize/restore with
 *   bytes + time, memory `vramBytes` / `ramBytes`, build costs).
 * - Checked-in baseline `benchmark_snapshot_matrix.json` exists, matches the
 *   frozen version, covers every (scenario, operation) pair, and keeps
 *   headless-CPU / browser-WebGPU reports separate (non-comparable label).
 * - `docs/benchmarks.md` publishes the per-scenario table + environments.
 *
 * Run: bun scripts/test-benchmarks.ts (root: bun run test:benchmarks)
 * Requires: no GPU, no browser, no network.
 */
import fs from 'node:fs';
import path from 'node:path';
import {
  DocumentIndex,
  SNAPSHOT_FORMAT_VERSION,
} from '../packages/webgpu-search/src/index';
import { generateMonacoRecords } from '../apps/monaco-palette/src/sample-data';
import { generateStructuredLogs } from '../apps/log-viewer/src/log-generator';
import {
  BENCHMARK_FIXTURE_VERSION,
  BENCHMARK_LOG_BASE_TIME_MS,
  BENCHMARK_FIXTURES,
  BENCHMARK_QUERIES,
  BENCHMARK_EXPECTED_ROWS,
} from './benchmark-fixtures';

let passed = 0;
function check(name: string, cond: boolean, extra = ''): void {
  if (cond) {
    passed++;
    console.log(`   ok - ${name}`);
  } else {
    console.error(`   FAIL - ${name}${extra ? ` -- ${extra}` : ''}`);
    process.exitCode = 1;
  }
}

function isDescendingIntegers(scores: number[]): boolean {
  for (let i = 0; i < scores.length; i++) {
    if (!Number.isInteger(scores[i] as number)) return false;
    if (i > 0 && (scores[i] as number) > (scores[i - 1] as number)) return false;
  }
  return true;
}

function isFiniteNonNegative(n: unknown): n is number {
  return typeof n === 'number' && Number.isFinite(n) && n >= 0;
}

async function main(): Promise<void> {
  console.log('--- Benchmark fixtures pin (docs/benchmarks.md Phase 5) ---');
  const rootDir = path.resolve(__dirname, '..');

  // ------------------------------------------------ 1. Frozen spec
  console.log('1. Frozen fixture spec');
  check('fixture version is 1.0.0', BENCHMARK_FIXTURE_VERSION === '1.0.0', BENCHMARK_FIXTURE_VERSION);
  check('four frozen scenarios', BENCHMARK_FIXTURES.length === 4, String(BENCHMARK_FIXTURES.length));
  check(
    'scenario ids frozen',
    JSON.stringify(BENCHMARK_FIXTURES.map((f) => f.scenario)) ===
      JSON.stringify(['ide-symbols-5k', 'log-grid-10k', 'log-grid-50k', 'log-grid-100k']),
  );
  check(
    'scenario sizes frozen',
    JSON.stringify(BENCHMARK_FIXTURES.map((f) => f.docs)) === JSON.stringify([5000, 10_000, 50_000, 100_000]),
  );
  check('seven frozen queries', BENCHMARK_QUERIES.length === 7, String(BENCHMARK_QUERIES.length));
  const ops = new Set(BENCHMARK_QUERIES.map((q) => q.operation));
  for (const op of [
    'prefix-search',
    'prefix-search+type-filter',
    'autocomplete',
    'fuzzy-search+type-facets',
    'fuzzy-search',
    'fuzzy-search+level+latency-filter',
    'fuzzy-search+facets',
  ]) {
    check(`frozen query covers ${op}`, ops.has(op));
  }
  check('expected rows cover 19 pairs', BENCHMARK_EXPECTED_ROWS.length === 19, String(BENCHMARK_EXPECTED_ROWS.length));
  const fixturesRaw = fs.readFileSync(path.join(rootDir, 'scripts/benchmark-fixtures.ts'), 'utf8');
  // Strip comments + string literals before matching so prose mentioning
  // `window` / `document` does not trip the bare-global scan (same approach
  // as scripts/test-proof-apps.ts §7 and the search-modes DOM test).
  const fixturesSrc = fixturesRaw
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .split('\n')
    .map((line) => {
      const idx = line.indexOf('//');
      return idx >= 0 ? line.slice(0, idx) : line;
    })
    .join('\n')
    .replace(/'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|`(?:[^`\\]|\\.)*`/g, "''");
  check('fixtures module is DOM-free', !/\bwindow\b/.test(fixturesSrc) && !/\bdocument\./.test(fixturesSrc));

  // ------------------------------------------------ 2. Generator determinism
  console.log('2. Generator determinism');
  const monaco600 = generateMonacoRecords(600);
  check('monaco smoke length', monaco600.length === 600, String(monaco600.length));
  check('monaco first id stable', monaco600[0]!.id === 'f-001', monaco600[0]!.id);
  const monacoAgain = generateMonacoRecords(600);
  check(
    'monaco deterministic across calls',
    JSON.stringify(monaco600.map((r) => r.id)) === JSON.stringify(monacoAgain.map((r) => r.id)),
  );
  const logsA = generateStructuredLogs(100, 1, BENCHMARK_LOG_BASE_TIME_MS);
  const logsB = generateStructuredLogs(100, 1, BENCHMARK_LOG_BASE_TIME_MS);
  check('logs smoke length', logsA.length === 100, String(logsA.length));
  check('logs first id stable', logsA[0]!.id === 'log-0000001', logsA[0]!.id);
  check(
    'logs deterministic with frozen anchor',
    JSON.stringify(logsA) === JSON.stringify(logsB),
  );
  check(
    'logs anchor timestamp pinned',
    logsA[99]!.timestamp === new Date(BENCHMARK_LOG_BASE_TIME_MS - 80).toISOString(),
    logsA[99]!.timestamp,
  );
  const levels = new Set(logsA.map((l) => l.level));
  check('logs contain all levels', levels.has('ERROR') && levels.has('WARN') && levels.has('INFO') && levels.has('DEBUG'));

  // ------------------------------------------------ 3. Live smoke (small, fast)
  console.log('3. Live frozen-query smoke');
  const ideDocs = generateMonacoRecords(600);
  const tIde0 = performance.now();
  const ide = await DocumentIndex.create(ideDocs, {
    idField: 'id',
    fields: [
      { name: 'filename', weight: 3.0 },
      { name: 'symbols', weight: 2.0 },
      { name: 'path', weight: 1.0 },
      { name: 'description', weight: 0.5 },
    ],
    filterFields: [{ name: 'type' }, { name: 'language' }],
    preferGpu: false,
  });
  const ideBuildMs = performance.now() - tIde0;
  const ideStats = ide.getStats();
  check('ide build wall-clock finite', isFiniteNonNegative(ideBuildMs));
  check('ide buildTimeMs finite', isFiniteNonNegative(ideStats.buildTimeMs));
  check('ide headless vramBytes is 0', ideStats.memory.vramBytes === 0, String(ideStats.memory.vramBytes));
  check('ide ramBytes positive', ideStats.memory.ramBytes > 0, String(ideStats.memory.ramBytes));

  const idePrefix = await ide.search('compute', { mode: 'prefix', limit: 20 });
  check('ide prefix serves', idePrefix.totalMatches >= 1, String(idePrefix.totalMatches));
  check('ide prefix scores descending integers', isDescendingIntegers(idePrefix.results.map((r) => r.score)));
  const ideFiltered = await ide.search('compute', { mode: 'prefix', limit: 20, filter: { type: 'shader' } });
  check('ide type filter narrows', ideFiltered.totalMatches <= idePrefix.totalMatches);
  check('ide type filter exact', ideFiltered.results.every((r) => (r.doc as unknown as { type: string }).type === 'shader'));
  const ideAuto = await ide.autocomplete('comp', { mode: 'prefix', limit: 5 });
  check('ide autocomplete suggestions', ideAuto.suggestions.length >= 1, String(ideAuto.suggestions.length));
  const ideFacets = await ide.search('service', {
    mode: 'fuzzy',
    limit: 20,
    facets: { byType: { type: 'terms', field: 'type', limit: 10 } },
  });
  check('ide type facets present', (ideFacets.facets as unknown as { byType?: { type: string } })?.byType?.type === 'terms');
  ide.destroy();

  const logDocs = generateStructuredLogs(1000, 1, BENCHMARK_LOG_BASE_TIME_MS);
  const tLog0 = performance.now();
  const logs = await DocumentIndex.create(logDocs as never[], {
    idField: 'id',
    fields: [
      { name: 'message', weight: 2.0 },
      { name: 'service', weight: 1.5 },
      { name: 'level', weight: 1.0 },
      { name: 'traceId', weight: 1.2 },
    ],
    filterFields: [{ name: 'level' }, { name: 'service' }, { name: 'timestamp' }, { name: 'latencyMs', type: 'number' }],
    preferGpu: false,
  });
  const logBuildMs = performance.now() - tLog0;
  const logStats = logs.getStats();
  check('log build wall-clock finite', isFiniteNonNegative(logBuildMs));
  check('log headless vramBytes is 0', logStats.memory.vramBytes === 0, String(logStats.memory.vramBytes));
  check('log ramBytes positive', logStats.memory.ramBytes > 0, String(logStats.memory.ramBytes));

  const fuzzy = await logs.search('timeout', { mode: 'fuzzy', limit: 50 });
  check('log fuzzy serves', fuzzy.totalMatches >= 1, String(fuzzy.totalMatches));
  check('log fuzzy scores descending integers', isDescendingIntegers(fuzzy.results.map((r) => r.score)));
  const logFiltered = await logs.search('timeout', {
    mode: 'fuzzy',
    limit: 50,
    filter: { level: 'ERROR', latencyMs: { gte: 300 } },
  });
  check('log level+latency filter narrows', logFiltered.totalMatches <= fuzzy.totalMatches);
  const logFaceted = await logs.search('timeout', {
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
  });
  check('log facets present', (logFaceted.facets as unknown as { byLevel?: { type: string } })?.byLevel?.type === 'terms');

  // snapshot bytes + time + restore costs on the smoke index.
  const tSer0 = performance.now();
  const snap = logs.serialize();
  const serMs = performance.now() - tSer0;
  check('snapshot bytes exceed header', snap.byteLength > 56, String(snap.byteLength));
  check('snapshot serialize wall-clock finite', isFiniteNonNegative(serMs));
  const tRes0 = performance.now();
  const restored = await DocumentIndex.fromSnapshot(snap);
  const resMs = performance.now() - tRes0;
  check('snapshot restore wall-clock finite', isFiniteNonNegative(resMs));
  check('snapshot live format version', restored.getStats().formatVersion === SNAPSHOT_FORMAT_VERSION);
  check('restoreTimeMs reported', isFiniteNonNegative(restored.getStats().restoreTimeMs ?? -1));
  const probeBefore = await logs.search('timeout', { mode: 'fuzzy', limit: 5 });
  const probeAfter = await restored.search('timeout', { mode: 'fuzzy', limit: 5 });
  check('restore preserves matches', probeAfter.totalMatches === probeBefore.totalMatches);
  restored.destroy();
  logs.destroy();

  // ------------------------------------------------ 4. Checked-in baseline
  console.log('4. Checked-in baseline artifacts');
  const jsonPath = path.join(rootDir, 'benchmark_snapshot_matrix.json');
  const mdPath = path.join(rootDir, 'benchmark_snapshot_matrix.md');
  check('baseline JSON checked in', fs.existsSync(jsonPath));
  check('baseline markdown checked in', fs.existsSync(mdPath));
  if (fs.existsSync(jsonPath)) {
    const baseline = JSON.parse(fs.readFileSync(jsonPath, 'utf8')) as {
      fixturesVersion?: string;
      generatedAt?: string;
      environment?: Record<string, unknown>;
      rows?: Array<{
        scenario: string;
        docs: number;
        operation: string;
        medianMs: number;
        p95Ms: number;
        samples?: number[];
        extra?: Record<string, number | string | boolean>;
      }>;
    };
    check('baseline fixtures version pinned', baseline.fixturesVersion === BENCHMARK_FIXTURE_VERSION, String(baseline.fixturesVersion));
    check('baseline generatedAt present', typeof baseline.generatedAt === 'string' && baseline.generatedAt.length > 0);
    const env = baseline.environment ?? {};
    check('baseline engine is cpu', env.engine === 'cpu', String(env.engine));
    check('baseline headless true', env.headless === true, String(env.headless));
    check('baseline preferGpu false', env.preferGpu === false, String(env.preferGpu));
    check(
      'baseline labeled non-comparable',
      typeof env.note === 'string' && env.note.includes('not comparable to browser/WebGPU runs'),
      String(env.note),
    );
    const rows = baseline.rows ?? [];
    for (const expected of BENCHMARK_EXPECTED_ROWS) {
      const found = rows.find((r) => r.scenario === expected.scenario && r.operation === expected.operation);
      check(`baseline covers ${expected.scenario}/${expected.operation}`, found !== undefined);
    }
    for (const r of rows) {
      const label = `${r.scenario}/${r.operation}`;
      if (!isFiniteNonNegative(r.medianMs) || !isFiniteNonNegative(r.p95Ms)) {
        check(`row latencies finite (${label})`, false, `${r.medianMs}/${r.p95Ms}`);
      }
      if (!Array.isArray(r.samples) || r.samples.length < 1 || !r.samples.every(isFiniteNonNegative)) {
        check(`row samples present (${label})`, false);
      }
    }
    if (process.exitCode !== 1) {
      // Count these bulk row-shape gates once (19 rows × shape already asserted
      // per-row above via early-exit only on failure).
      passed++;
      console.log('   ok - all baseline rows carry median/p95/samples');
    }
    const searchRow = rows.find((r) => r.operation === 'fuzzy-search');
    check(
      'search rows carry memory + build costs',
      typeof searchRow?.extra?.ramBytes === 'number' &&
        typeof searchRow?.extra?.vramBytes === 'number' &&
        typeof searchRow?.extra?.buildMs === 'number' &&
        typeof searchRow?.extra?.uploadMs === 'number',
      JSON.stringify(searchRow?.extra ?? null),
    );
    const serRow = rows.find((r) => r.operation === 'snapshot-serialize');
    check(
      'serialize row carries bytes',
      typeof serRow?.extra?.snapshotBytes === 'number' && (serRow?.extra?.snapshotBytes as number) > 56,
      JSON.stringify(serRow?.extra ?? null),
    );
    const resRow = rows.find((r) => r.operation === 'snapshot-restore');
    check(
      'restore row carries restore costs',
      typeof resRow?.extra?.snapshotBytes === 'number' &&
        typeof resRow?.extra?.restoredMatches === 'number',
      JSON.stringify(resRow?.extra ?? null),
    );
  }

  // ------------------------------------------------ 5. Published doc
  console.log('5. Published benchmark doc');
  const docPath = path.join(rootDir, 'docs/benchmarks.md');
  check('docs/benchmarks.md exists', fs.existsSync(docPath));
  if (fs.existsSync(docPath)) {
    const doc = fs.readFileSync(docPath, 'utf8');
    check('doc pins fixture version', doc.includes(BENCHMARK_FIXTURE_VERSION));
    check('doc labels headless non-comparable', doc.includes('NOT comparable to browser/WebGPU runs'));
    check('doc has per-scenario table', doc.includes('| ide-symbols-5k |') && doc.includes('| log-grid-100k |'));
    check('doc documents memory costs', doc.includes('vramBytes') && doc.includes('ramBytes'));
    check('doc documents serialize/restore costs', doc.includes('snapshot-serialize') && doc.includes('snapshot-restore'));
  }

  if (process.exitCode === 1) {
    console.error('\nBenchmark fixtures pin FAILED');
    process.exit(1);
  }
  console.log(`\n--- Benchmark fixtures pin passed (${passed} checks) ---`);
}

main().catch((err) => {
  console.error('test failed:', err);
  process.exit(1);
});
