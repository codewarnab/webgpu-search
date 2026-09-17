/**
 * M2 bundle-size gate: fold-table delta ≤8 KB gzip over baseline.
 *
 * Baseline (contract §3, post-M1 working tree, tsup minify:false):
 *   dist/index.js 44,581 B raw / 10,206 B gzip (`gzip -c`, default level 6).
 * Budget: 20 KB gzip total → ~10 KB headroom; table delta cap 8 KB gzip.
 *
 * Method: node:zlib gzipSync at level 6, equivalent to `gzip -c` default.
 * Cross-check: `gzip -c dist/index.js | wc -c` should match within ~tens
 * of bytes (header OS byte differs; stream is identical).
 *
 * Run: bun scripts/check-m2-bundle.ts
 */
import { gzipSync, constants } from 'node:zlib';

const BASELINE_RAW = 44581;
const BASELINE_GZIP = 10206;
const DELTA_CAP_GZIP = 8 * 1024;
const TOTAL_BUDGET_GZIP = 20 * 1024;

const distUrl = new URL('../packages/webgpu-search/dist/index.js', import.meta.url);
const buf = await Bun.file(distUrl).arrayBuffer();
const raw = buf.byteLength;
const gz = gzipSync(Buffer.from(buf), { level: constants.Z_DEFAULT_COMPRESSION }).length;
const dRaw = raw - BASELINE_RAW;
const dGz = gz - BASELINE_GZIP;

console.log('--- M2 bundle-size report (`gzip -c`, tsup minify:false) ---');
console.log(`dist/index.js: ${raw} B raw / ${gz} B gzip`);
console.log(`baseline:      ${BASELINE_RAW} B raw / ${BASELINE_GZIP} B gzip`);
console.log(`delta:         ${dRaw >= 0 ? '+' : ''}${dRaw} B raw / ${dGz >= 0 ? '+' : ''}${dGz} B gzip (cap +${DELTA_CAP_GZIP} B gzip)`);
console.log(`total budget:  ${gz} / ${TOTAL_BUDGET_GZIP} B gzip`);

let fail = false;
if (dGz > DELTA_CAP_GZIP) {
  console.error(`❌ fold-table gzip delta +${dGz} B exceeds +${DELTA_CAP_GZIP} B cap. Split table into a lazy chunk or re-encode.`);
  fail = true;
} else {
  console.log(`✅ delta within cap`);
}
if (gz > TOTAL_BUDGET_GZIP) {
  console.error(`❌ total gzip ${gz} B exceeds ${TOTAL_BUDGET_GZIP} B budget.`);
  fail = true;
} else {
  console.log(`✅ total within 20 KB budget`);
}
if (fail) process.exit(1);
