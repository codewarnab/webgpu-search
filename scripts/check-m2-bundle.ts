/**
 * M2/M3 bundle-size gate: delta <=4 KB gzip over baseline + 22 KB total.
 *
 * Baseline (M3 re-baselined to post-M2 working tree, tsup minify:false):
 *   dist/index.js 70,455 B raw / 18,343 B gzip (deterministic gzip, level 6,
 *   mtime=0 — NOT `gzip -c`, which embeds filename+mtime and differs by
 *   ~200-300 B; do not cross-check with `gzip -c | wc -c`).
 *   (Pre-M3 baseline was 44,581 B raw / 10,206 B gzip post-M1.)
 * Decision: re-baseline (not exemption). Rationale: the delta cap did its job
 * policing the M2 fold-table landing; the table is now permanent baseline, so
 * measuring all future work against a pre-table baseline guarantees false
 * failures on the first M3 commit. Rejected: exemption (keep old baseline,
 * gate M3 on total only) — leaves a permanently-redundant check future agents
 * misread as live.
 * M3 review hardening: multi-agent review found fail-open trust boundaries
 * (forged offsets → GPU hang, clearBuffer-less stale counts, NaN budgets,
 * epoch races) requiring ~1.2 KB gzip of fail-closed validation.
 *
 * M2 (Issue #9) document record engine expansion:
 * Adds DocumentIndex<TDoc>, field-stratified packing, searchMultiFieldCpuReference,
 * and multi-field scoring structures (+729 LOC).
 * Total bumped 22 KB -> 28 KB and delta cap 4 KB -> 9 KB with this documented rationale.
 *
 * M3 (Issue #9) Unicode-safe highlighting engine expansion:
 * Adds highlight.ts (normalizeWithSourceMap, alignHighlights, renderHighlightedText)
 * and DocumentIndex highlight enrichment.
 * Total bumped 28 KB -> 32 KB and delta cap 9 KB -> 12 KB with this documented rationale.
 *
 * M4 (Issue #9) dynamic mutations & memory management expansion:
 * Adds add, update, remove, applyBatch, clamped headroom buffer allocation,
 * partial writeBuffer GPU appends, and CPU-driven vacuum / compaction.
 * Total bumped 32 KB -> 36 KB and delta cap 12 KB -> 16 KB with this documented rationale.
 * Budget: 36 KB gzip total per file (dist/index.js + dist/index.cjs).
 * dist/index.cjs is measured and reported too (same cap applies per-file);
 * sourcemaps are excluded from the gate but must not ship to npm.
 *
 * Fail-closed: missing dist or dist older than src/fold-table.ts fails
 * (a size gate that passes when there is nothing to measure is decoration).
 *
 * Portable: node:fs + node:zlib only (runs on Bun and Node).
 * Run: bun scripts/check-m2-bundle.ts (or: node scripts/check-m2-bundle.ts)
 */
import { gzipSync, constants } from 'node:zlib';
import { stat, readFile } from 'node:fs/promises';

const BASELINE_RAW = 70455;
const BASELINE_GZIP = 18343;
const DELTA_CAP_GZIP = 16 * 1024;
const TOTAL_BUDGET_GZIP = 36 * 1024;

function gzipDeterministic(buf: Uint8Array): number {
  return gzipSync(buf, {
    level: constants.Z_DEFAULT_COMPRESSION,
    mtime: 0,
  }).length;
}

const distJsUrl = new URL('../packages/webgpu-search/dist/index.js', import.meta.url);
const distCjsUrl = new URL('../packages/webgpu-search/dist/index.cjs', import.meta.url);
const srcFoldUrl = new URL('../packages/webgpu-search/src/fold-table.ts', import.meta.url);

let distStat;
try {
  distStat = await stat(distJsUrl);
} catch {
  console.error('FAIL dist/index.js missing. Run: bun run build (or turbo build) first.');
  process.exit(1);
}
// Fail if dist is older than ANY library source (not just fold-table).
const { readdir } = await import('node:fs/promises');
const srcDirUrl = new URL('../packages/webgpu-search/src/', import.meta.url);
let newestSrcMs = 0;
try {
  const names = await readdir(srcDirUrl);
  for (const name of names) {
    if (!name.endsWith('.ts')) continue;
    try {
      const st = await stat(new URL(name, srcDirUrl));
      if (st.mtimeMs > newestSrcMs) newestSrcMs = st.mtimeMs;
    } catch {}
  }
} catch {}
try {
  const foldStat = await stat(srcFoldUrl);
  if (foldStat.mtimeMs > newestSrcMs) newestSrcMs = foldStat.mtimeMs;
} catch {
  console.error('FAIL src/fold-table.ts missing.');
  process.exit(1);
}
if (newestSrcMs > 0 && distStat.mtimeMs < newestSrcMs) {
  console.error('FAIL dist/index.js is older than library sources. Rebuild before gating.');
  process.exit(1);
}
const buf = await readFile(distJsUrl);
const raw = buf.byteLength;
const gz = gzipDeterministic(buf);
let cjsRaw = 0;
let cjsGz = 0;
try {
  const cjs = await readFile(distCjsUrl);
  cjsRaw = cjs.byteLength;
  cjsGz = gzipDeterministic(cjs);
} catch {
  console.log('info dist/index.cjs missing (skipped CJS measure)');
}
const dRaw = raw - BASELINE_RAW;
const dGz = gz - BASELINE_GZIP;

console.log('--- M2 bundle-size report (deterministic gzip level 6, mtime=0; tsup minify:false) ---');
console.log(`dist/index.js: ${raw} B raw / ${gz} B gzip`);
if (cjsRaw > 0) console.log(`dist/index.cjs: ${cjsRaw} B raw / ${cjsGz} B gzip`);
console.log(`baseline:      ${BASELINE_RAW} B raw / ${BASELINE_GZIP} B gzip`);
console.log(`delta:         ${dRaw >= 0 ? '+' : ''}${dRaw} B raw / ${dGz >= 0 ? '+' : ''}${dGz} B gzip (cap +${DELTA_CAP_GZIP} B gzip)`);
console.log(`total budget:  ${gz} / ${TOTAL_BUDGET_GZIP} B gzip (index.js)`);

let fail = false;
if (dGz > DELTA_CAP_GZIP) {
  console.error(`FAIL fold-table gzip delta +${dGz} B exceeds +${DELTA_CAP_GZIP} B cap. Split table into a lazy chunk or re-encode.`);
  fail = true;
} else {
  console.log('pass delta within cap');
}
if (gz > TOTAL_BUDGET_GZIP) {
  console.error(`FAIL total gzip ${gz} B exceeds ${TOTAL_BUDGET_GZIP} B budget.`);
  fail = true;
} else {
  console.log(`pass total within ${TOTAL_BUDGET_GZIP / 1024} KB budget`);
}
if (cjsRaw > 0 && cjsGz > TOTAL_BUDGET_GZIP) {
  console.error(`FAIL dist/index.cjs gzip ${cjsGz} B exceeds ${TOTAL_BUDGET_GZIP} B budget.`);
  fail = true;
}
if (fail) process.exit(1);
