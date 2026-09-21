/**
 * v0.4 (Issue #10 M8) headless benchmark matrix.
 *
 * Covers the M8 benchmark invariant without requiring a browser/GPU:
 *  - IDE symbols (monaco-palette records): prefix search, type-filtered
 *    search, and suggest() autocomplete latency.
 *  - Data-grid rows (structured logs at 10k / 50k / 100k): fuzzy search,
 *    structured level/service/latency filtering, and facet aggregation.
 *  - U2D4 persistence: serialize/restore wall-clock, snapshot bytes, and
 *    columnar segment overhead.
 *
 * Run: `bun scripts/bench-v04-matrix.ts [--out <path>]`
 * All engines run CPU (`preferGpu: false`) for headless determinism.
 */
import { DocumentIndex } from '../packages/webgpu-search/src/index';
import { generateMonacoRecords } from '../apps/monaco-palette/src/sample-data';
import { generateStructuredLogs } from '../apps/log-viewer/src/log-generator';

interface Timed {
  medianMs: number;
  p95Ms: number;
  samples: number[];
}

function stats(samples: number[]): Timed {
  const sorted = [...samples].sort((a, b) => a - b);
  const q = (p: number): number => {
    if (sorted.length === 0) return 0;
    if (sorted.length === 1) return sorted[0]!;
    const pos = (sorted.length - 1) * p;
    const base = Math.floor(pos);
    const rest = pos - base;
    return sorted[base + 1] !== undefined
      ? sorted[base]! + rest * (sorted[base + 1]! - sorted[base]!)
      : sorted[base]!;
  };
  return {
    medianMs: Number(q(0.5).toFixed(3)),
    p95Ms: Number(q(0.95).toFixed(3)),
    samples: sorted.map((s) => Number(s.toFixed(3)))
  };
}

async function timeSamples(
  fn: () => Promise<unknown>,
  warmups = 3,
  samples = 10
): Promise<Timed> {
  for (let i = 0; i < warmups; i++) await fn();
  const out: number[] = [];
  for (let i = 0; i < samples; i++) {
    const t0 = performance.now();
    await fn();
    out.push(performance.now() - t0);
  }
  return stats(out);
}

interface MatrixRow {
  scenario: string;
  docs: number;
  operation: string;
  medianMs: number;
  p95Ms: number;
  extra?: Record<string, number | string | boolean>;
}

async function main(): Promise<void> {
  const rows: MatrixRow[] = [];
  const outIdx = process.argv.indexOf('--out');
  const outPath = outIdx >= 0 ? process.argv[outIdx + 1] : 'benchmark_v04_matrix.json';

  // ------------------------------------------------------------------
  // 1. IDE symbols (monaco-palette shape)
  // ------------------------------------------------------------------
  {
    const records = generateMonacoRecords(5000);
    const index = await DocumentIndex.create(records, {
      idField: 'id',
      fields: [
        { name: 'filename', weight: 3.0 },
        { name: 'symbols', weight: 2.0 },
        { name: 'path', weight: 1.0 },
        { name: 'description', weight: 0.5 }
      ],
      filterFields: [{ name: 'type' }, { name: 'language' }],
      preferGpu: false
    });

    const prefix = await timeSamples(() =>
      index.search('compute', { mode: 'prefix', limit: 20 }).then(() => {})
    );
    rows.push({ scenario: 'ide-symbols-5k', docs: 5000, operation: 'prefix-search', medianMs: prefix.medianMs, p95Ms: prefix.p95Ms });

    const filtered = await timeSamples(() =>
      index.search('compute', { mode: 'prefix', limit: 20, filter: { type: 'shader' } }).then(() => {})
    );
    rows.push({ scenario: 'ide-symbols-5k', docs: 5000, operation: 'prefix-search+type-filter', medianMs: filtered.medianMs, p95Ms: filtered.p95Ms });

    const suggest = await timeSamples(() =>
      index.suggest('comp', { mode: 'prefix', limit: 5 }).then(() => {})
    );
    const suggestRes = await index.suggest('comp', { mode: 'prefix', limit: 5 });
    rows.push({
      scenario: 'ide-symbols-5k', docs: 5000, operation: 'suggest-autocomplete',
      medianMs: suggest.medianMs, p95Ms: suggest.p95Ms,
      extra: { suggestionCount: suggestRes.suggestions.length }
    });

    const facets = await timeSamples(() =>
      index.search('service', {
        mode: 'fuzzy', limit: 20,
        facets: { byType: { type: 'terms', field: 'type', limit: 10 } }
      }).then(() => {})
    );
    rows.push({ scenario: 'ide-symbols-5k', docs: 5000, operation: 'fuzzy-search+type-facets', medianMs: facets.medianMs, p95Ms: facets.p95Ms });

    index.destroy();
  }

  // ------------------------------------------------------------------
  // 2. Data-grid rows (structured logs at 10k / 50k / 100k)
  // ------------------------------------------------------------------
  for (const count of [10_000, 50_000, 100_000]) {
    const logs = generateStructuredLogs(count);
    const index = await DocumentIndex.create(logs as any, {
      idField: 'id',
      fields: [
        { name: 'message', weight: 2.0 },
        { name: 'service', weight: 1.5 },
        { name: 'level', weight: 1.0 },
        { name: 'traceId', weight: 1.2 }
      ],
      filterFields: [{ name: 'level' }, { name: 'service' }, { name: 'timestamp' }, { name: 'latencyMs', type: 'number' }],
      preferGpu: false
    });

    const fuzzy = await timeSamples(() =>
      index.search('timeout', { mode: 'fuzzy', limit: 50 }).then(() => {})
    );
    rows.push({ scenario: `log-grid-${count / 1000}k`, docs: count, operation: 'fuzzy-search', medianMs: fuzzy.medianMs, p95Ms: fuzzy.p95Ms });

    const filtered = await timeSamples(() =>
      index.search('timeout', {
        mode: 'fuzzy', limit: 50,
        filter: { level: 'ERROR', latencyMs: { gte: 300 } }
      }).then(() => {})
    );
    const filteredRes = await index.search('timeout', {
      mode: 'fuzzy', limit: 50,
      filter: { level: 'ERROR', latencyMs: { gte: 300 } }
    });
    rows.push({
      scenario: `log-grid-${count / 1000}k`, docs: count, operation: 'fuzzy-search+level+latency-filter',
      medianMs: filtered.medianMs, p95Ms: filtered.p95Ms,
      extra: { totalMatches: filteredRes.totalMatches }
    });

    const faceted = await timeSamples(() =>
      index.search('timeout', {
        mode: 'fuzzy', limit: 50,
        facets: {
          byLevel: { type: 'terms', field: 'level', limit: 10 },
          byLatency: {
            type: 'range', field: 'latencyMs',
            ranges: [
              { to: 50, key: 'fast' },
              { from: 50, to: 300, key: 'normal' },
              { from: 300, key: 'slow' }
            ]
          }
        }
      }).then(() => {})
    );
    rows.push({ scenario: `log-grid-${count / 1000}k`, docs: count, operation: 'fuzzy-search+facets', medianMs: faceted.medianMs, p95Ms: faceted.p95Ms });

    // U2D4 persistence profile only on the 10k grid to bound runtime.
    if (count === 10_000) {
      const tSer0 = performance.now();
      const snapshot = index.serialize();
      const serMs = performance.now() - tSer0;
      const header = new DataView(snapshot, 0, 56);
      const columnarLen = header.getUint32(44, true);
      const schemaLen = header.getUint32(36, true);
      const docsLen = header.getUint32(40, true);
      const tRes0 = performance.now();
      const restored = await DocumentIndex.fromSnapshot(snapshot);
      const resMs = performance.now() - tRes0;
      const probe = await restored.search('timeout', { mode: 'fuzzy', limit: 5 });
      rows.push({
        scenario: 'log-grid-10k', docs: count, operation: 'u2d4-serialize',
        medianMs: Number(serMs.toFixed(3)), p95Ms: Number(serMs.toFixed(3)),
        extra: { snapshotBytes: snapshot.byteLength, schemaBytes: schemaLen, columnarBytes: columnarLen, docsBytes: docsLen }
      });
      rows.push({
        scenario: 'log-grid-10k', docs: count, operation: 'u2d4-restore',
        medianMs: Number(resMs.toFixed(3)), p95Ms: Number(resMs.toFixed(3)),
        extra: { restoredMatches: probe.totalMatches }
      });
      restored.destroy();
    }

    index.destroy();
  }

  // ------------------------------------------------------------------
  // Report
  // ------------------------------------------------------------------
  console.log('\n## v0.4 Benchmark Matrix (headless CPU, median of 10)\n');
  console.log('| Scenario | Docs | Operation | Median (ms) | p95 (ms) | Extra |');
  console.log('|---|---|---|---:|---:|---|');
  for (const r of rows) {
    console.log(
      `| ${r.scenario} | ${r.docs.toLocaleString()} | ${r.operation} | ${r.medianMs} | ${r.p95Ms} | ${r.extra ? JSON.stringify(r.extra) : ''} |`
    );
  }
  console.log('');

  if (outPath) {
    const { writeFileSync } = await import('node:fs');
    writeFileSync(outPath!, JSON.stringify({ generatedAt: new Date().toISOString(), rows }, null, 2));
    console.log(`Matrix written to ${outPath}`);
  }
}

main().catch((err) => {
  console.error('[bench-v04-matrix] fatal:', err);
  process.exit(1);
});
