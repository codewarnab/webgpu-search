#!/usr/bin/env bun
/**
 * check-public-api.ts — Phase 0 API-surface diff gate.
 *
 * Fails closed when the frozen 1.x public surface drifts:
 * 1. Runtime export allowlist of `packages/webgpu-search/src/index.ts`
 *    (added/removed symbols fail; update docs/public-api.md + this list together).
 * 2. INTERNAL_* prefix allowlist (only INTERNAL_WORKER_ID_KEY may leak).
 * 3. Frozen version constants + DEFAULT_TIE_BREAKERS value.
 * 4. SearchResponse echo keys present in src/types.ts.
 * 5. limit/maxResults alias preserved across index paths.
 * 6. Package exports map exposes "." + "./worker" with import/require/types.
 *
 * Portable: no DOM refs. Runs under bun (also node-compatible, no bun-only APIs).
 */
import { readFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const pkgDir = join(root, 'packages', 'webgpu-search');
const srcIndex = join(pkgDir, 'src', 'index.ts');
const typesPath = join(pkgDir, 'src', 'types.ts');
const guardPath = join(pkgDir, 'src', 'guard.ts');
const rankingPath = join(pkgDir, 'src', 'ranking.ts');
const textProfilePath = join(pkgDir, 'src', 'text-profile.ts');
const pkgJsonPath = join(pkgDir, 'package.json');

let failures: string[] = [];
const fail = (msg: string) => failures.push(msg);
const read = (p: string): string => readFileSync(p, 'utf8');

// --- 1. Runtime export surface -------------------------------------------
const mod = (await import(srcIndex)) as Record<string, unknown>;
const actual = Object.keys(mod).sort();

// Frozen surface lives in scripts/public-api-surface.txt (checked in).
// Update it + docs/public-api.md in the same PR when adding public symbols;
// removals require a major.
const frozenPath = join(root, 'scripts', 'public-api-surface.txt');
if (!existsSync(frozenPath)) {
  fail('check-public-api: scripts/public-api-surface.txt missing (surface baseline).');
}
let expected: string[] = [];
if (existsSync(frozenPath)) {
  expected = read(frozenPath).split('\n').map((s) => s.trim()).filter(Boolean);
}
if (expected.length === 0) {
  fail('check-public-api: frozen surface allowlist is empty (bootstrap error).');
}

const actualSet = new Set(actual);
const expectedSet = new Set(expected);
for (const sym of actual) {
  if (!expectedSet.has(sym)) fail(`unexpected public export added: ${sym} (update docs/public-api.md + scripts/public-api-surface.txt in the same PR)`);
}
for (const sym of expected) {
  if (!actualSet.has(sym)) fail(`frozen public export removed: ${sym} (removal requires a major + docs/public-api.md update)`);
}

// --- 2. INTERNAL_* prefix rule --------------------------------------------
const internalLeaks = actual.filter((s) => s.startsWith('INTERNAL_'));
const ALLOWED_INTERNAL = new Set(['INTERNAL_WORKER_ID_KEY']);
for (const sym of internalLeaks) {
  if (!ALLOWED_INTERNAL.has(sym)) fail(`internal symbol leaking via public entry: ${sym} (prefix INTERNAL_ is reserved; document in docs/public-api.md §6 or remove)`);
}
if (!actualSet.has('INTERNAL_WORKER_ID_KEY')) fail('INTERNAL_WORKER_ID_KEY missing from public entry (worker string-isolation contract).');

// --- 3. Frozen constants ----------------------------------------------------
const textProfile = read(textProfilePath);
const expectPairs: Array<[string, string]> = [
  [`UNICODE_VERSION = '16.0.0'`, 'UNICODE_VERSION must stay 16.0.0'],
  [`SCORING_VERSION = 'parity-v1'`, 'SCORING_VERSION must stay parity-v1 (scoring change = major)'],
  ['DATASET_FORMAT_VERSION = 2', 'DATASET_FORMAT_VERSION must stay 2'],
  ['DATASET_MAGIC = 0x55324632', 'DATASET_MAGIC frozen'],
  ['SNAPSHOT_FORMAT_VERSION = 4', 'SNAPSHOT_FORMAT_VERSION must stay 4'],
  ['SNAPSHOT_MAGIC = 0x55324434', 'SNAPSHOT_MAGIC frozen'],
  ['SNAPSHOT_HEADER_BYTES = 56', 'SNAPSHOT_HEADER_BYTES frozen'],
  ['LEGACY_SNAPSHOT_VERSION = 3', 'LEGACY_SNAPSHOT_VERSION frozen'],
  ['LEGACY_SNAPSHOT_MAGIC = 0x55324433', 'LEGACY_SNAPSHOT_MAGIC frozen'],
  ['LEGACY_SNAPSHOT_HEADER_BYTES = 48', 'LEGACY_SNAPSHOT_HEADER_BYTES frozen'],
  ['QUERY_TOKENS_MAX = 128', 'QUERY_TOKENS_MAX frozen'],
  ['RESULT_LIMIT_MAX = 8192', 'RESULT_LIMIT_MAX frozen'],
];
for (const [needle, msg] of expectPairs) {
  if (!textProfile.includes(needle)) fail(`${msg} (drift in src/text-profile.ts)`);
}
const ranking = read(rankingPath);
for (const tier of [`'score'`, `'weight'`, `'exact'`, `'length'`, `'id'`]) {
  if (!ranking.includes(tier)) fail(`DEFAULT_TIE_BREAKERS drift: missing tier ${tier} in src/ranking.ts`);
}
// Order check: score < weight < exact < length < id in DEFAULT_TIE_BREAKERS block.
const tieBlock = ranking.slice(ranking.indexOf('DEFAULT_TIE_BREAKERS'), ranking.indexOf('DEFAULT_TIE_BREAKERS') + 400);
const order = [`'score'`, `'weight'`, `'exact'`, `'length'`, `'id'`].map((t) => tieBlock.indexOf(t));
if (order.some((i) => i < 0) || !(order[0] < order[1] && order[1] < order[2] && order[2] < order[3] && order[3] < order[4])) {
  fail('DEFAULT_TIE_BREAKERS order must stay [score, weight, exact, length, id] (ordering change = major).');
}

// --- 4. Response echo keys ---------------------------------------------------
const types = read(typesPath);
for (const key of ['profileId', 'scoringVersion', 'cpuScorer', 'fallbackReason', 'candidateCount', 'hasOverflow']) {
  if (!types.includes(key)) fail(`SearchResponse echo key missing in src/types.ts: ${key}`);
}

// --- 5. limit/maxResults alias (public behavioral check) ---------------------
// Frozen contract: default 50, maxResults honored, limit honored, and limit
// wins when both are set. Checked at RUNTIME through both public gates
// (SearchIndex + DocumentIndex) instead of grepping source files, so the alias
// expression may move between internal modules (e.g. document/query.ts) with
// the modular refactor without breaking the gate; breaking the behavior does.
type AliasSearchable = {
  search(query: string, options?: Record<string, unknown>): Promise<{ results: unknown[] }>;
  destroy?(): void;
};
{
  const aliasVecs: Array<[Record<string, unknown>, number, string]> = [
    [{}, 50, 'default clamps to 50'],
    [{ maxResults: 7 }, 7, 'maxResults alias honored'],
    [{ limit: 3 }, 3, 'limit honored'],
    [{ limit: 2, maxResults: 9 }, 2, 'limit precedes maxResults'],
  ];
  const gates: Array<[string, AliasSearchable]> = [];
  const si = await ((mod as Record<string, unknown>)['SearchIndex'] as {
    create(items: unknown[], options?: Record<string, unknown>): Promise<AliasSearchable>;
  }).create(
    Array.from({ length: 60 }, (_, i) => `aliasdoc-${i}`),
    { preferGpu: false }
  );
  gates.push(['SearchIndex', si]);
  const di = await ((mod as Record<string, unknown>)['DocumentIndex'] as {
    create<T>(records: T[], options: Record<string, unknown>): Promise<AliasSearchable>;
  }).create(
    Array.from({ length: 60 }, (_, i) => ({ id: i, title: `aliasdoc-${i} note` })),
    { fields: [{ name: 'title' }], preferGpu: false }
  );
  gates.push(['DocumentIndex', di]);
  for (const [gate, searcher] of gates) {
    for (const [opts, want, label] of aliasVecs) {
      const res = await searcher.search('aliasdoc', { ...opts, mode: 'substring' });
      if (res.results.length !== want) {
        fail(`${gate} limit/maxResults alias broken: ${label} (got ${res.results.length}, want ${want})`);
      }
    }
  }
  for (const [, searcher] of gates) {
    if (typeof searcher.destroy === 'function') searcher.destroy();
  }
}
if (!types.includes('maxResults?: number')) fail('SearchOptions.maxResults alias missing in src/types.ts.');

// --- 6. Package exports map ---------------------------------------------------
const pkg = JSON.parse(read(pkgJsonPath)) as {
  exports?: Record<string, { import?: Record<string, string>; require?: Record<string, string> }>;
  typesVersions?: Record<string, Record<string, string[]>>;
};
for (const entry of ['.', './worker']) {
  const e = pkg.exports?.[entry];
  if (!e) { fail(`package.json exports missing entry "${entry}".`); continue; }
  if (!e.import?.types || !e.import?.default) fail(`exports["${entry}"].import must define types + default.`);
  if (!e.require?.types || !e.require?.default) fail(`exports["${entry}"].require must define types + default.`);
}
// Legacy node10 fallback for the worker subpath (attw gate): without
// typesVersions, `webgpu-search/worker` fails node10 resolution.
const workerTypes = pkg.typesVersions?.['*']?.['worker'] ?? [];
if (!workerTypes.includes('./dist/worker.d.ts')) {
  fail('package.json typesVersions["*"]["worker"] must include ./dist/worker.d.ts (attw node10 gate).');
}
void guardPath;

if (failures.length > 0) {
  console.error(`check-public-api: ${failures.length} failure(s):`);
  for (const f of failures) console.error(`  - ${f}`);
  process.exit(1);
}
console.log(`check-public-api: ok (${actual.length} public exports, echoes + aliases + exports-map pinned).`);
