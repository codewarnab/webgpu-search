/**
 * headless benchmark matrix.
 *
 * Covers the benchmark invariant without requiring a browser/GPU:
 * - IDE symbols (monaco-palette records): prefix search, type-filtered
 * search, and autocomplete() latency.
 * - Data-grid rows (structured logs at 10k / 50k / 100k): fuzzy search,
 * structured level/service/latency filtering, and facet aggregation.
 * - snapshot persistence: serialize/restore wall-clock, snapshot bytes, and
 * columnar segment overhead.
 *
 * Environment: headless CPU only (`preferGpu: false`) for determinism.
 * Numbers are NOT comparable to browser/WebGPU runs. The JSON report records
 * `environment` (runtime, cpu, headless) alongside `generatedAt` + rows.
 *
 * Run: `bun scripts/bench-snapshot-matrix.ts [--out <path>]`
 * `--out` is constrained to the repo working directory (basename sanitized)
 * to avoid arbitrary file writes; absolute paths under cwd or /tmp are allowed.
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
  samples?: number[];
  extra?: Record<string, number | string | boolean>;
}

function resolveOutPath(raw: string | undefined): string {
  const fallback = 'benchmark_snapshot_matrix.json';
  if (!raw) return fallback;
  // Allow explicit absolute paths under cwd or /tmp (CI uses /tmp), otherwise
  // sanitize to basename inside cwd to avoid arbitrary writes (e.g. /etc/passwd).
  if (raw.startsWith('/tmp/') || raw.startsWith(`${process.cwd()}/`)) return raw;
  const base = raw.split('/').pop()!.split('\\').pop()!;
  const safe = base.replace(/[^a-zA-Z0-9._-]/g, '_').slice(0, 128) || fallback;
  return safe;
}

async function main(): Promise<void> {
  const rows: MatrixRow[] = [];
  const outIdx = process.argv.indexOf('--out');
  const outPath = resolveOutPath(outIdx >= 0 ? process.argv[outIdx + 1] : undefined);

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
    rows.push({ scenario: 'ide-symbols-5k', docs: 5000, operation: 'prefix-search', medianMs: prefix.medianMs, p95Ms: prefix.p95Ms, samples: prefix.samples });

    const filtered = await timeSamples(() =>
      index.search('compute', { mode: 'prefix', limit: 20, filter: { type: 'shader' } }).then(() => {})
    );
    rows.push({ scenario: 'ide-symbols-5k', docs: 5000, operation: 'prefix-search+type-filter', medianMs: filtered.medianMs, p95Ms: filtered.p95Ms, samples: filtered.samples });

    const autocomplete = await timeSamples(() =>
      index.autocomplete('comp', { mode: 'prefix', limit: 5 }).then(() => {})
    );
    const autocompleteRes = await index.autocomplete('comp', { mode: 'prefix', limit: 5 });
    rows.push({
      scenario: 'ide-symbols-5k', docs: 5000, operation: 'autocomplete',
      medianMs: autocomplete.medianMs, p95Ms: autocomplete.p95Ms, samples: autocomplete.samples,
      extra: { suggestionCount: autocompleteRes.suggestions.length }
    });

    const facets = await timeSamples(() =>
      index.search('service', {
        mode: 'fuzzy', limit: 20,
        facets: { byType: { type: 'terms', field: 'type', limit: 10 } }
      }).then(() => {})
    );
    rows.push({ scenario: 'ide-symbols-5k', docs: 5000, operation: 'fuzzy-search+type-facets', medianMs: facets.medianMs, p95Ms: facets.p95Ms, samples: facets.samples });

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
    rows.push({ scenario: `log-grid-${count / 1000}k`, docs: count, operation: 'fuzzy-search', medianMs: fuzzy.medianMs, p95Ms: fuzzy.p95Ms, samples: fuzzy.samples });

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
      medianMs: filtered.medianMs, p95Ms: filtered.p95Ms, samples: filtered.samples,
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
    rows.push({ scenario: `log-grid-${count / 1000}k`, docs: count, operation: 'fuzzy-search+facets', medianMs: faceted.medianMs, p95Ms: faceted.p95Ms, samples: faceted.samples });

    // snapshot persistence profile only on the 10k grid to bound runtime.
    // Sampled 10× with warmup like every other row (previously n=1).
    if (count === 10_000) {
      const serSamples = await timeSamples(async () => {
        index.serialize();
      }, 1, 10);
      const snapshot = index.serialize();
      if (snapshot.byteLength < 56) {
        throw new Error('[bench-snapshot-matrix] snapshot shorter than snapshot header');
      }
      const header = new DataView(snapshot, 0, 56);
      const columnarLen = header.getUint32(44, true);
      const schemaLen = header.getUint32(36, true);
      const docsLen = header.getUint32(40, true);
      const resSamples = await timeSamples(async () => {
        const r = await DocumentIndex.fromSnapshot(snapshot);
        r.destroy();
      }, 1, 10);
      const tRes0 = performance.now();
      const restored = await DocumentIndex.fromSnapshot(snapshot);
      const resSingleMs = performance.now() - tRes0;
      void resSingleMs;
      const probe = await restored.search('timeout', { mode: 'fuzzy', limit: 5 });
      rows.push({
        scenario: 'log-grid-10k', docs: count, operation: 'snapshot-serialize',
        medianMs: serSamples.medianMs, p95Ms: serSamples.p95Ms, samples: serSamples.samples,
        extra: { snapshotBytes: snapshot.byteLength, schemaBytes: schemaLen, columnarBytes: columnarLen, docsBytes: docsLen }
      });
      rows.push({
        scenario: 'log-grid-10k', docs: count, operation: 'snapshot-restore',
        medianMs: resSamples.medianMs, p95Ms: resSamples.p95Ms, samples: resSamples.samples,
        extra: { restoredMatches: probe.totalMatches }
      });
      restored.destroy();
    }

    index.destroy();
  }

  // ------------------------------------------------------------------
  // Report
  // ------------------------------------------------------------------
  console.log('\n## Benchmark Matrix (headless CPU, median of 10)\n');
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
    const environment = {
      engine: 'cpu',
      preferGpu: false,
      headless: true,
      runtime: typeof (globalThis as any).Bun !== 'undefined' ? `bun/${(globalThis as any).Bun.version}` : `node/${process.version}`,
      platform: `${process.platform}-${process.arch}`,
      note: 'Headless CPU numbers; not comparable to browser/WebGPU runs.'
    };
    writeFileSync(outPath!, JSON.stringify({ generatedAt: new Date().toISOString(), environment, rows }, null, 2));
    console.log(`Matrix written to ${outPath}`);
  }
}

main().catch((err) => {
  console.error('[bench-snapshot-matrix] fatal:', err);
  process.exit(1);
});
