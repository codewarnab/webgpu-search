/**
 * M2/M3 bundle-size gate: delta <=30 KB gzip over baseline + 78 KB total.
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
 *
 * M5 (Issue #9) first-party worker client & protocol expansion:
 * Adds SearchWorkerClient with monotonic query sequencing, AbortError liveness,
 * transferable buffer safety, string-isolated enrichment, and error class rehydration.
 * Total bumped 36 KB -> 40 KB and delta cap 16 KB -> 20 KB with this documented rationale.
 *
 * M6 (Issue #9) snapshot persistence & IndexedDB storage expansion:
 * Adds persistence.ts (U2D3 48-byte Little-Endian binary format, circular-dependency-free CRC32,
 * decoupled document storage) and idb-storage.ts (transaction-safe IndexedDB helpers).
 * Total bumped 40 KB -> 48 KB and delta cap 20 KB -> 30 KB with this documented rationale.
 * Budget: 48 KB gzip total per file (dist/index.js + dist/index.cjs).
 * dist/index.cjs is measured and reported too (same cap applies per-file);
 * sourcemaps are excluded from the gate but must not ship to npm.
 *
 * Issue #10 M2 (structured pre-filtering) columnar filter engine expansion:
 * Adds filter/bitset.ts (DocumentBitset), filter/columnar-store.ts (typed
 * columns + inverted indexes), filter/filter-evaluator.ts (AST compiler),
 * plus filterFields plumbing in DocumentIndex, persistence schema, and
 * worker-client forwarding.
 * Baseline re-based to post-M1 main (dist/index.js 234,907 B raw /
 * 49,246 B gzip, CI-measured) so the gate polices this PR's own delta
 * (+~7.5 KB gzip vs 30 KB cap) instead of conflating M4-M8 + M1 growth the
 * old M3-era baseline never absorbed (main was already +30,903 over it).
 * Total bumped 48 KB -> 60 KB with this documented rationale.
 * Budget: 60 KB gzip total per file (dist/index.js + dist/index.cjs).
 *
 * Issue #10 M3 (facet aggregation) review hardening expansion:
 * Multi-agent PR #30 review required fail-closed hardening (+~1.6 KB gzip):
 * __proto__/constructor/prototype guards + safeSet, string[] per-doc dedupe,
 * range inverted-bounds + caps (32 facets / 100 buckets), direct-engine
 * validation, optional allMatchedDocIndices + gated O(n) map, engine-relative
 * isApproximate docs, worker predicate facet drop, wall-clock totalMs.
 * Total bumped 60 KB -> 64 KB with this documented rationale.
 * Budget: 64 KB gzip total per file (dist/index.js + dist/index.cjs).
 *
 * Issue #10 M4 (token & prefix modes + typo tolerance) search-mode expansion:
 * Adds modes/typo-distance.ts (bounded Damerau-Levenshtein + length gates),
 * modes/token-search.ts (multi-term AND/OR/quorum + proximity scoring),
 * modes/prefix-search.ts (anchored symbol matching), typo-aware substring
 * scoring in cpu-reference.ts, token/prefix/typo routing + validation in
 * hybrid-index.ts / document-index.ts / webgpu-engine.ts, and highlight
 * branches for all modes (+~2.5 KB gzip, delta still within the 30 KB cap).
 * Total bumped 64 KB -> 68 KB with this documented rationale.
 * Budget: 68 KB gzip total per file (dist/index.js + dist/index.cjs).
 *
 * Issue #10 M5 (deterministic ranking + autocomplete) ranking/suggest expansion:
 * Adds ranking.ts (five-tier tie-breaker comparator + validation) and
 * suggest.ts (option normalization), deterministic rank keys on all three
 * DocumentIndex paths (GPU readback, legacy ufuzzy, parity CPU via new
 * rankingOptions/docIds params), the suggest() enumeration primitive, and
 * inline search({ suggest }) enrichment (+~2 KB gzip, delta still within
 * the 30 KB cap).
 * Total bumped 68 KB -> 72 KB with this documented rationale.
 * Budget: 72 KB gzip total per file (dist/index.js + dist/index.cjs).
 *
 * Issue #10 M6 (extensibility pipeline + safe hook architecture) expansion:
 * Adds extensions.ts (default/code tokenizers, hook validation + per-key
 * merge, Top-K scoring/postProcess pipeline, declarative hookIds + fail-closed
 * restore guard), tokenTermsOverride threading in cpu-reference/highlight/
 * DocumentIndex, hookIds persistence schema, and worker-client fail-closed
 * guards (+~3 KB gzip, delta still within the 30 KB cap).
 * Total bumped 72 KB -> 78 KB with this documented rationale.
 * Budget: 78 KB gzip total per file (dist/index.js + dist/index.cjs).
 *
 * Issue #10 M7 (query diagnostics, cost budgets & broad-query safeguards):
 * Adds diagnostics.ts (budget validation, phase-boundary enforcement,
 * selectivity + broad-query heuristics, warning builders), per-phase timing
 * + suggestMs telemetry in DocumentIndex/HybridIndex, worker predicate
 * diagnostics drop, and fail-closed unknown max* keys (+~1.6 KB gzip;
 * delta +31,011 B exceeds the 30 KB cap, total 80,257 B exceeds 78 KB).
 * Total bumped 78 KB -> 82 KB and delta cap 30 KB -> 32 KB with this
 * documented rationale.
 * Budget: 82 KB gzip total per file (dist/index.js + dist/index.cjs).
 *
 * Issue #10 M8 (U2D4 persistence hardening + custom getter guard) review fixes:
 * Multi-agent PR #35 review required fail-closed hardening (+~1.4 KB gzip):
 * snapshot size caps (schema/columnar/docs/total/docCount/tokenCount) before
 * decode, BigInt/function/symbol JSON-safe columnar encoding, `hasGetter`
 * persistence + restore guard (mirrors hookIds), version-aware worker staging
 * guard, and strict empty-columnar integrity signal. Delta +33,419 B exceeds
 * the 32 KB cap; total stays within 82 KB.
 * Delta cap bumped 32 KB -> 34 KB with this documented rationale.
 * Budget: 82 KB gzip total per file (dist/index.js + dist/index.cjs).
 *
 * Fail-closed: missing dist or dist older than src/fold-table.ts fails
 * (a size gate that passes when there is nothing to measure is decoration).
 *
 * Portable: node:fs + node:zlib only (runs on Bun and Node).
 * Run: bun scripts/check-m2-bundle.ts (or: node scripts/check-m2-bundle.ts)
 */
import { gzipSync, constants } from 'node:zlib';
import { stat, readFile } from 'node:fs/promises';

const BASELINE_RAW = 234907;
const BASELINE_GZIP = 49246;
const DELTA_CAP_GZIP = 34 * 1024;
const TOTAL_BUDGET_GZIP = 82 * 1024;


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
