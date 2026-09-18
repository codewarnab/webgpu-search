/**
 * M4 differential harness (Issue #7): CPU/GPU parity, fallback, CI.
 *
 * Scripts-only: this file must NEVER be imported by `src/index.ts` (bundle
 * gate `scripts/check-m2-bundle.ts` — harness weight stays out of the
 * shipped library). Run: `bun scripts/test-m4-parity.ts`.
 * Sharding: `M4_SHARD_INDEX` / `M4_SHARD_TOTAL` (defaults 0/1).
 * Per-cell timeout 15 s. Durations reported as median/p95 (never mean-of-3).
 *
 * Execution policy (same-host/ICU only):
 * - `vgpu/mock` never executes WGSL (canary: known-hit corpus returns 0 on
 *   the GPU path), so exact GPU-parity cells are recorded as
 *   `pending-hardware`, NOT failures. Under plain bun even a real device
 *   could not execute: unbundled `import x from '*.wgsl'` resolves to an
 *   asset path, not WGSL text (bundler text-loader only), so shader compile
 *   fails before dispatch. The browser subset in
 *   `scripts/test-regression.ts` is the release gate for true parity.
 * - Everything else here is hard-gated: matrix corrections (a)-(n),
 *   harness self-consistency, echo contracts, failure injection delta,
 *   concurrency/abort, worker contract.
 *
 * ASCII-only source: non-ASCII strings built via String.fromCodePoint.
 */
import { createMockAdapter } from 'vgpu/mock';
import {
  WebGPUEngine,
  SearchIndex,
  packUnicodeToGPUBuffer,
  serializeUnicodeDataset,
  deserializeUnicodeDataset,
  checkMemoryBudget,
  normalizeText,
  toWellFormedSafe,
  tokensEqual,
  compareParityResults,
  searchCpuReference,
  scoreSubstringTokens,
  clampLimit,
  foldCodePoint,
  WORD_BOUNDARY_PREV,
  LONE_SURROGATE_SOURCE,
  LONE_SURROGATE_PATTERN,
  QUERY_TOKENS_MAX,
  RESULT_LIMIT_MAX,
  IncompatibleIndexError,
  IncompatibleOptionError,
  ProfileMismatchError,
  QueryTooLongError,
} from '../packages/webgpu-search/src/index';

const FCP = String.fromCodePoint;
const SHARP_S = FCP(0xdf);
const DOT_I_CAP = FCP(0x130);
const THETA_SYM = FCP(0x3f4);
const THETA = FCP(0x3b8);
const GRIN = FCP(0x1f600);
const WOMAN = FCP(0x1f469);
const FAMILY = FCP(0x1f468, 0x200d, 0x1f469, 0x200d, 0x1f467, 0x200d, 0x1f466);
const CJK_A = FCP(0x5317);
const IDEO_SPACE = FCP(0x3000);
const MARK = FCP(0x301);
const VS = FCP(0xfe0f);
const ZWJ = FCP(0x200d);
const TATWEEL = FCP(0x640);
const HIGH = FCP(0xd800);
const LOW = FCP(0xdc00);

// ---------------------------------------------------------------- counters
let passed = 0;
let failed = 0;
let pending = 0;
const pendingNotes: string[] = [];
function ok(name: string, cond: boolean, extra = ''): void {
  if (cond) {
    passed++;
    console.log(`   ok - ${name}`);
  } else {
    failed++;
    console.error(`   FAIL - ${name}${extra ? ` -- ${extra}` : ''}`);
  }
}
function pend(name: string, reason: string): void {
  pending++;
  pendingNotes.push(`${name}: ${reason}`);
  console.log(`   pending-hardware - ${name} (${reason})`);
}

// ---------------------------------------------------------------- utils
/** Deterministic seeded RNG (mulberry32). Same seed => same corpora. */
function mulberry32(seed: number): () => number {
  let a: number = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t: number = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function median(xs: number[]): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? (s[m] as number) : (((s[m - 1] as number) + (s[m] as number)) / 2);
}
function p95(xs: number[]): number {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.ceil(0.95 * s.length) - 1)] as number;
}
class CellTimeoutError extends Error {
  constructor(label: string) {
    super(`cell timeout: ${label}`);
    this.name = 'CellTimeoutError';
  }
}
async function withTimeout<T>(p: Promise<T>, ms: number, label: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | null = null;
  try {
    return await Promise.race([
      p,
      new Promise<never>((_, rej) => {
        timer = setTimeout(() => rej(new CellTimeoutError(label)), ms);
      }),
    ]);
  } finally {
    if (timer !== null) clearTimeout(timer);
  }
}
function scoreMultiset(scores: readonly number[]): Map<number, number> {
  const m = new Map<number, number>();
  for (const s of scores) m.set(s, (m.get(s) ?? 0) + 1);
  return m;
}
function multisetsEqual(a: Map<number, number>, b: Map<number, number>): boolean {
  if (a.size !== b.size) return false;
  for (const [k, v] of a) if (b.get(k) !== v) return false;
  return true;
}

const CELL_TIMEOUT_MS = 15000;
const shardIndex = Number(process.env.M4_SHARD_INDEX ?? 0);
const shardTotal = Math.max(1, Number(process.env.M4_SHARD_TOTAL ?? 1));

// ---------------------------------------------------------------- corpora
function asciiCorpus(n: number, seed: number): string[] {
  const rnd = mulberry32(seed);
  const pre = ['src/components', 'src/views', 'src/utils', 'src/services'];
  const nouns = ['User', 'Account', 'Session', 'Auth', 'Order', 'Product'];
  const suf = ['Controller', 'Service', 'Handler', 'Manager', 'Provider'];
  const out = new Array<string>(n);
  for (let i = 0; i < n; i++) {
    const p = pre[Math.floor(rnd() * pre.length)];
    const w = nouns[Math.floor(rnd() * nouns.length)];
    const s = suf[Math.floor(rnd() * suf.length)];
    out[i] = `${p}/${w}${s}_${i}.ts`;
  }
  return out;
}
function unicodeCorpus(): string[] {
  return [
    'stra' + SHARP_S + 'e',
    'STRASSE',
    'strasse',
    'Istanbul',
    'i' + DOT_I_CAP + 'i',
    'final' + FCP(0x3c2) + 'vs' + FCP(0x3c3) + 'vs' + FCP(0x3a3),
    'ligature' + FCP(0xfb00),
    THETA_SYM + 'heory',
    'caf' + FCP(0xe9),
    'cafe' + FCP(0x301),
    CJK_A + FCP(0x4eac) + 'city',
    GRIN + 'smile' + GRIN,
    WOMAN + 'alone',
    FAMILY,
    'plain',
    '',
    '   ',
    'a'.repeat(2000) + 'tail Fin',
    'mixed' + SHARP_S + GRIN + CJK_A,
  ];
}

interface DiffCell {
  name: string;
  corpus: string[];
  query: string;
  mode: 'substring' | 'fuzzy';
  limit: number;
  caseSensitive?: boolean;
}

function buildCells(): DiffCell[] {
  const ascii = asciiCorpus(300, 0xC10C);
  const uni = unicodeCorpus();
  const edge = ['', '   ', 'a'.repeat(2000) + 'xyz', GRIN + GRIN, 'abc'];
  const straQueries = ['strasse', 'SS', SHARP_S, 'stra'];
  const cells: DiffCell[] = [];
  for (const q of ['auth', 'Controller_7', 'zzz-no-match', 'Auth']) {
    for (const mode of ['substring', 'fuzzy'] as const) {
      cells.push({ name: `ascii/${mode}/${q}`, corpus: ascii, query: q, mode, limit: 50 });
    }
  }
  cells.push({ name: 'ascii/substring/limit-1', corpus: ascii, query: 'src', mode: 'substring', limit: 1 });
  for (const q of straQueries) {
    cells.push({ name: `unicode/substring/${q.length}ch`, corpus: uni, query: q, mode: 'substring', limit: 50 });
  }
  cells.push({ name: 'unicode/fuzzy/strasse', corpus: uni, query: 'strasse', mode: 'fuzzy', limit: 50 });
  cells.push({ name: 'unicode/substring/cjk', corpus: uni, query: CJK_A, mode: 'substring', limit: 10 });
  cells.push({ name: 'unicode/substring/emoji', corpus: uni, query: GRIN, mode: 'substring', limit: 10 });
  cells.push({ name: 'unicode/substring/woman-in-family', corpus: uni, query: WOMAN, mode: 'substring', limit: 10 });
  cells.push({ name: 'unicode/substring/theta-fold', corpus: uni, query: THETA, mode: 'substring', limit: 10 });
  cells.push({ name: 'unicode/substring/e-acute', corpus: uni, query: FCP(0xe9), mode: 'substring', limit: 10 });
  cells.push({ name: 'edge/substring/tail', corpus: edge, query: 'xyz', mode: 'substring', limit: 10 });
  cells.push({ name: 'edge/fuzzy/abc', corpus: edge, query: 'abc', mode: 'fuzzy', limit: 10 });
  cells.push({ name: 'edge/substring/at-cap', corpus: ascii, query: 'a'.repeat(QUERY_TOKENS_MAX), mode: 'substring', limit: 5 });
  cells.push({ name: 'cs/substring/exact', corpus: ['Auth', 'auth', 'AUTH'], query: 'Auth', mode: 'substring', limit: 10, caseSensitive: true });
  return cells;
}

// ---------------------------------------------------------------- compare
interface ComparedResponse {
  totalMatches: number;
  candidateCount: number;
  hasOverflow: boolean;
  results: Array<{ index: number; score: number; text: string }>;
  profileId: string;
  scoringVersion: string;
  cpuAlgorithm: string;
}

/** Exact-order when under cap, else totalMatches + hasOverflow + score multiset. */
function compareResponses(a: ComparedResponse, b: ComparedResponse): string | null {
  if (a.totalMatches !== b.totalMatches) return `totalMatches ${a.totalMatches} vs ${b.totalMatches}`;
  if (a.candidateCount !== b.candidateCount) return `candidateCount ${a.candidateCount} vs ${b.candidateCount}`;
  if (a.hasOverflow !== b.hasOverflow) return `hasOverflow ${a.hasOverflow} vs ${b.hasOverflow}`;
  if (a.profileId !== b.profileId) return `profileId ${a.profileId} vs ${b.profileId}`;
  if (a.scoringVersion !== b.scoringVersion) return `scoringVersion ${a.scoringVersion} vs ${b.scoringVersion}`;
  if (a.cpuAlgorithm !== b.cpuAlgorithm) return `cpuAlgorithm ${a.cpuAlgorithm} vs ${b.cpuAlgorithm}`;
  if (!a.hasOverflow) {
    if (a.results.length !== b.results.length) return `results.length ${a.results.length} vs ${b.results.length}`;
    for (let i = 0; i < a.results.length; i++) {
      const x = a.results[i] as { index: number; score: number };
      const y = b.results[i] as { index: number; score: number };
      if (x.index !== y.index || x.score !== y.score) {
        return `ordered[${i}] (${x.index},${x.score}) vs (${y.index},${y.score})`;
      }
    }
    return null;
  }
  const ma = scoreMultiset(a.results.map((r) => r.score));
  const mb = scoreMultiset(b.results.map((r) => r.score));
  if (!multisetsEqual(ma, mb)) return 'score multiset drift above cap';
  return null;
}

// ---------------------------------------------------------------- main
async function main(): Promise<void> {
  console.log('--- M4 differential harness (scripts-only; mock device, browser is release gate) ---');
  console.log(`shard ${shardIndex}/${shardTotal}, cell timeout ${CELL_TIMEOUT_MS}ms`);

  const mockAdapter = createMockAdapter({ features: ['timestamp-query'] as never });
  const mockDeviceWrapper = await mockAdapter.requestDevice();
  const mockDevice = mockDeviceWrapper.gpu as unknown as GPUDevice;

  // ============================================================ Part 1: matrix (a)-(n)
  console.log('Part 1. Matrix corrections vs S2.6 as-landed');

  // (a) degenerate split: whitespace/U+3000/empty unify; survivors echo original.
  {
    const idx = await SearchIndex.create(['alpha'], { preferGpu: false });
    const ws = await idx.search('   ', { mode: 'substring' });
    ok('(a) whitespace unifies to empty', ws.query === '' && ws.totalMatches === 0);
    const ide = await idx.search(IDEO_SPACE, { mode: 'substring' });
    ok('(a) U+3000-only unifies to empty', ide.query === '' && ide.totalMatches === 0);
    const empty = await idx.search('', { mode: 'substring' });
    ok('(a) empty unifies to empty', empty.query === '' && empty.totalMatches === 0);
    for (const [label, ch] of [['lone-mark', MARK], ['VS', VS], ['ZWJ', ZWJ], ['tatweel', TATWEEL]] as const) {
      const n = normalizeText(ch, true);
      const r = await idx.search(ch, { mode: 'substring' });
      ok(`(a) ${label} survives + echoes original`, n.tokenCount === 1 && r.query === ch);
    }
    // Engine-level echo parity for both paths.
    const eng = new WebGPUEngine();
    await eng.init(mockDevice);
    await eng.loadDataset(['alpha']);
    const eWs = await eng.search('   ', { mode: 'substring' });
    const eMark = await eng.search(MARK, { mode: 'substring' });
    ok('(a) engine echoes match index contract', eWs.query === '' && eMark.query === MARK);
    eng.destroy();
    idx.destroy();
  }

  // (b) U+03F4 folds via C (not T-only).
  {
    ok('(b) fold table maps U+03F4->U+03B8', JSON.stringify(foldCodePoint(0x3f4)) === JSON.stringify([0x3b8]));
    ok('(b) folded U+03F4 == theta tokens', tokensEqual(normalizeText(THETA_SYM, true).tokens, normalizeText(THETA, true).tokens));
    ok('(b) NFC-only U+03F4 distinct', !tokensEqual(normalizeText(THETA_SYM, false).tokens, normalizeText(THETA, false).tokens));
  }

  // (c) lone-surrogate source: pair-preserving, lookbehind-free; global via new RegExp.
  {
    ok('(c) SOURCE has no lookbehind', !LONE_SURROGATE_SOURCE.includes('?<'));
    ok('(c) PATTERN non-global (breaking)', LONE_SURROGATE_PATTERN.global === false);
    const FFFD = FCP(0xfffd);
    const pair = HIGH.replace(HIGH, FCP(0xd800, 0xdc00));
    const globalRe = new RegExp(LONE_SURROGATE_SOURCE, 'g');
    const scrub = (s: string): string =>
      s.replace(globalRe, (m: string, g1: string | undefined) => (g1 !== undefined ? m : FFFD));
    ok('(c) global preserves valid pair', scrub(pair) === pair);
    ok('(c) global high+high -> 2 FFFDs', scrub(HIGH + HIGH) === FFFD + FFFD);
    ok('(c) global lone low -> 1 FFFD', scrub('a' + LOW) === 'a' + FFFD);
    ok('(c) toWellFormedSafe agrees', toWellFormedSafe(HIGH + HIGH) === FFFD + FFFD);
  }

  // (d) ASCII fast-path == full-path (fuzz).
  {
    const rnd = mulberry32(0xA5C11);
    const refFull = (s: string, folded: boolean): Uint32Array => {
      const trimmed = s.trim();
      if (trimmed.length === 0) return new Uint32Array(0);
      const wf = toWellFormedSafe(trimmed);
      const nfc1 = wf.normalize('NFC');
      if (!folded) {
        const buf: number[] = [];
        for (const ch of nfc1) buf.push(ch.codePointAt(0) as number);
        return new Uint32Array(buf);
      }
      const scalars: number[] = [];
      for (const ch of nfc1) {
        const cp = ch.codePointAt(0) as number;
        const m = foldCodePoint(cp);
        if (m === null) scalars.push(cp);
        else for (const v of m) scalars.push(v);
      }
      const nfc2 = String.fromCodePoint(...scalars);
      const buf: number[] = [];
      for (const ch of nfc2.normalize('NFC')) buf.push(ch.codePointAt(0) as number);
      return new Uint32Array(buf);
    };
    let drift = '';
    for (let t = 0; t < 200; t++) {
      const len = 1 + Math.floor(rnd() * 24);
      let s = '';
      for (let i = 0; i < len; i++) s += FCP(0x20 + Math.floor(rnd() * 0x5f));
      for (const folded of [true, false]) {
        if (!tokensEqual(normalizeText(s, folded).tokens, refFull(s, folded))) {
          drift = `t=${t} folded=${folded} s=${JSON.stringify(s)}`;
          break;
        }
      }
      if (drift) break;
    }
    ok('(d) ASCII fast-path == full-path x200x2', drift === '', drift);
  }

  // (e) wrap-free comparator; only score formulas use Math.imul/|0.
  {
    const wrap: Array<{ score: number; index: number }> = [
      { score: 2147483647, index: 1 },
      { score: -2147483648, index: 0 },
      { score: 2147483647, index: 0 },
    ];
    const sorted = [...wrap].sort(compareParityResults);
    ok(
      '(e) comparator total order at i32 extremes',
      sorted[0]?.index === 0 && sorted[1]?.index === 1 && sorted[2]?.index === 0,
      JSON.stringify(sorted)
    );
    const fs = await import('node:fs/promises');
    const cpuRef = await fs.readFile(new URL('../packages/webgpu-search/src/cpu-reference.ts', import.meta.url), 'utf8');
    const codeOnly = cpuRef.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|\s)\/\/.*$/gm, '$1');
    const imulHits = (codeOnly.match(/Math\.imul/g) ?? []).length;
    const cmpBlock = codeOnly.slice(codeOnly.indexOf('export function compareParityResults'));
    const cmpBody = cmpBlock.slice(cmpBlock.indexOf('{'), cmpBlock.indexOf('\n}') + 1);
    ok('(e) Math.imul confined to score formulas', imulHits === 3 && !cmpBody.includes('Math.imul'), `imul x${imulHits}`);
  }

  // (f) WORD_BOUNDARY_PREV set pinned; CJK/U+3000 no-bonus documented.
  {
    ok('(f) boundary set == / _ - . space : \\', JSON.stringify([...WORD_BOUNDARY_PREV]) === JSON.stringify([47, 95, 45, 46, 32, 58, 92]));
    const fs2 = await import('node:fs/promises');
    const fuzzyWgsl = await fs2.readFile(new URL('../packages/webgpu-search/src/shaders/fuzzy.wgsl', import.meta.url), 'utf8');
    const wbHits = [47, 95, 45, 46, 32, 58, 92].map((n) => `${n}u`);
    ok('(f) WGSL mirrors the 7 delimiters', wbHits.every((h) => fuzzyWgsl.includes(h)));
  }

  // (g) zero-renorm pack + totalTokens fail-fast + unknown-version throw.
  {
    const pre = [new Uint32Array([5, 6]), new Uint32Array([7])];
    const p = packUnicodeToGPUBuffer(pre, { folded: true });
    ok('(g) Uint32Array[] zero-renorm', p.tokenCount === 3 && p.tokens[2] === 7);
    let tt = false;
    try {
      packUnicodeToGPUBuffer(pre, { folded: true, totalTokens: 99 });
    } catch (e) {
      tt = e instanceof IncompatibleIndexError;
    }
    ok('(g) totalTokens mismatch fail-fast', tt);
    let uv = false;
    try {
      packUnicodeToGPUBuffer(['a'], { unicodeVersion: 'nope' });
    } catch (e) {
      uv = e instanceof IncompatibleIndexError;
    }
    ok('(g) unknown version throws at pack-time', uv);
  }

  // (h) U2F2 + CRC32 + cross-realm duck-typing + neutered guard.
  {
    const rt = packUnicodeToGPUBuffer(['hello'], { folded: true });
    const bytes = serializeUnicodeDataset(rt);
    ok('(h) roundtrip', deserializeUnicodeDataset(bytes).tokenCount === 5);
    let neut = false;
    try {
      deserializeUnicodeDataset(new ArrayBuffer(0));
    } catch (e) {
      neut = e instanceof IncompatibleIndexError;
    }
    ok('(h) neutered byteLength 0 throws', neut);
    // Cross-realm duck-typing via node:vm (genuine foreign-realm buffers:
    // `instanceof` fails but typed-array construction still works — exactly
    // the iframe/worker case the duck-type guards target).
    const vm = await import('node:vm');
    const xRealmU8 = vm.runInNewContext('new Uint8Array(8)');
    ok('(h) fixture is truly cross-realm', (xRealmU8.buffer as ArrayBuffer) instanceof ArrayBuffer === false);
    const xRealmBuf: ArrayBuffer = vm.runInNewContext(`new ArrayBuffer(${bytes.byteLength})`);
    new Uint8Array(xRealmBuf).set(new Uint8Array(bytes));
    let xRealmOk = false;
    try {
      xRealmOk = deserializeUnicodeDataset(xRealmBuf).tokenCount === 5;
    } catch {
      xRealmOk = false;
    }
    ok('(h) cross-realm ArrayBuffer deserializes', xRealmOk);
    const xRealmRow = vm.runInNewContext('new Uint32Array([9, 8, 7])') as Uint32Array;
    let xPackOk = false;
    try {
      xPackOk = packUnicodeToGPUBuffer([xRealmRow], { folded: true }).tokenCount === 3;
    } catch {
      xPackOk = false;
    }
    ok('(h) cross-realm Uint32Array packs (zero-renorm)', xPackOk);
  }

  // (i) per-buffer budget, new signature.
  {
    const tiny = { limits: { maxBufferSize: 1024, maxStorageBufferBindingSize: 1024, maxComputeWorkgroupsPerDimension: 1 } } as unknown as GPUDevice;
    ok('(i) records over-budget', checkMemoryBudget(10_000, 64, tiny).allowed === false);
    const qTiny = { limits: { maxBufferSize: 100, maxStorageBufferBindingSize: 100, maxComputeWorkgroupsPerDimension: 1 } } as unknown as GPUDevice;
    const q = checkMemoryBudget(1, 1, qTiny);
    ok('(i) query/output over tiny limits', q.allowed === false, q.reason ?? '');
    const forged = checkMemoryBudget(30_000_000, 64, { limits: {} } as unknown as GPUDevice);
    ok('(i) forged limits fall back finite', forged.allowed === false && Number.isFinite(forged.maxBytes));
    ok('(i) negatives clamp to allowed', checkMemoryBudget(-5, -10).allowed === true);
  }

  // (j) strict boolean caseSensitive + invalid mode.
  {
    const eng = new WebGPUEngine();
    await eng.init(mockDevice);
    await eng.loadDataset(['alpha']);
    let cs = false;
    try {
      await eng.search('alpha', { mode: 'substring', caseSensitive: 1 as unknown as boolean });
    } catch (e) {
      cs = e instanceof TypeError;
    }
    ok('(j) forged caseSensitive:1 TypeError', cs);
    let md = false;
    try {
      await eng.search('alpha', { mode: 'regex' as unknown as 'substring' });
    } catch (e) {
      md = e instanceof TypeError;
    }
    ok('(j) invalid mode TypeError', md);
    // Index-level mismatch still ProfileMismatchError.
    const idx = await SearchIndex.create(['alpha'], { preferGpu: false });
    let pm = false;
    try {
      await idx.search('alpha', { mode: 'substring', caseSensitive: true });
    } catch (e) {
      pm = e instanceof ProfileMismatchError;
    }
    ok('(j) index ProfileMismatchError', pm);
    eng.destroy();
    idx.destroy();
  }

  // (k) QueryTooLongError exact actual + gigantic estimate + cpu-fallback scan.
  {
    const idx = await SearchIndex.create(['a', 'b'], { preferGpu: false });
    let threw = false;
    try {
      await idx.search('a'.repeat(QUERY_TOKENS_MAX + 1), { mode: 'substring' });
    } catch (e) {
      threw = e instanceof QueryTooLongError && (e as QueryTooLongError).actual === QUERY_TOKENS_MAX + 1;
    }
    ok('(k) exact post-fold actual=129', threw);
    const gigantic = 'a'.repeat(1_000_001);
    let gest = false;
    try {
      await idx.search(gigantic, { mode: 'substring' });
    } catch (e) {
      gest = e instanceof QueryTooLongError && (e as QueryTooLongError).actual === 1_000_001 * 3;
    }
    ok('(k) gigantic >1M estimate actual=cp*3', gest);
    const fb = await withTimeout(
      idx.search(gigantic, { mode: 'substring', onQueryTooLong: 'cpu-fallback' }),
      CELL_TIMEOUT_MS,
      'gigantic cpu-fallback'
    );
    ok('(k) cpu-fallback forces exhaustive CPU scan', fb.engine === 'cpu' && fb.totalMatches === 0);
    idx.destroy();
  }

  // (l) shared clampLimit sweep.
  {
    const cases: Array<[unknown, number]> = [
      [0, 1], [1, 1], [2, 2], [50, 50], [8191, 8191], [8192, 8192], [8193, 8192],
      [NaN, 50], [undefined, 50], ['3', 3], [null, 1], [2.9, 2],
    ];
    const bad = cases.filter(([raw, want]) => clampLimit(raw) !== want);
    ok('(l) clampLimit sweep 12-vector', bad.length === 0, JSON.stringify(bad));
  }

  // (m) chunked dispatch layout + zero-dispatch + grow-only output + sizes.
  {
    const eng = new WebGPUEngine();
    await eng.init(mockDevice);
    const rows = Array.from({ length: 300 }, (_, i) => `chunk-row-${i}`);
    await eng.loadDataset(rows);
    const calls = (mockDevice as unknown as { __vgpuMockInstrumentation: { calls: Record<string, number> } }).__vgpuMockInstrumentation.calls;
    const e0 = calls.createCommandEncoder;
    const fast = await eng.search('chunk-row-1', { mode: 'substring' });
    ok('(m) fast path single encoder', calls.createCommandEncoder - e0 === 1, `delta=${calls.createCommandEncoder - e0}`);
    // Timestamp key present; when the mock reports a number it must be finite
    // (the real pin is the chunked `=== null` below, not this union check).
    ok(
      '(m) fast-path timestamp key well-formed',
      'gpuExecutionMs' in fast.timings && (fast.timings.gpuExecutionMs === null || Number.isFinite(fast.timings.gpuExecutionMs))
    );
    // Force chunked: maxDim=1 -> step=128 rows, 300 rows -> 3 chunks + clear + copy = 5 encoders.
    const realDevice = (eng as unknown as { device: GPUDevice }).device;
    (eng as unknown as { device: GPUDevice }).device = {
      ...realDevice,
      limits: { ...(realDevice as unknown as { limits: Record<string, number> }).limits, maxComputeWorkgroupsPerDimension: 1 },
    } as unknown as GPUDevice;
    const e1 = calls.createCommandEncoder;
    const chunked = await eng.search('chunk-row-1', { mode: 'substring' });
    const delta = calls.createCommandEncoder - e1;
    ok('(m) chunked multi-dispatch (5 encoders)', delta === 5, `delta=${delta}`);
    ok('(m) chunked skips timestamps (null)', chunked.timings.gpuExecutionMs === null);
    (eng as unknown as { device: GPUDevice }).device = realDevice;
    ok('(m) grow-only output 65544 B', (eng as unknown as { outputByteLength: number }).outputByteLength === 65544);
    await eng.loadDataset([]);
    const e2 = calls.createCommandEncoder;
    const zr = await eng.search('chunk-row-1', { mode: 'substring' });
    ok('(m) 0-row zero-dispatch', zr.totalMatches === 0 && calls.createCommandEncoder - e2 === 0);
    eng.destroy();
  }

  // (n) powerPreference reserved (accepted, not forwarded).
  {
    const cpuIdx = await SearchIndex.create(['a'], { preferGpu: false, powerPreference: 'low-power' });
    ok('(n) powerPreference accepted on CPU path', cpuIdx.getStats().engine === 'cpu');
    cpuIdx.destroy();
    const gpuIdx = await SearchIndex.create(['a'], { device: mockDevice, preferGpu: true, powerPreference: 'low-power' });
    ok('(n) powerPreference accepted on GPU path (not gated)', gpuIdx.getStats().engine === 'webgpu');
    gpuIdx.destroy();
  }

  // Echo/version pins: nfcProbedVersion null-today; flagsAndProfile wire-only.
  {
    const idx = await SearchIndex.create(['hello'], { preferGpu: false });
    const st = idx.getStats();
    // Probe deferred: pack-level is exactly null; stats type carries no probe
    // field at all (a future warn+record change must add the field AND flip
    // this assert — no `||` escape).
    ok('echo stats carry no nfcProbedVersion field', !('nfcProbedVersion' in (st as unknown as Record<string, unknown>)));
    const p = packUnicodeToGPUBuffer(['hello'], { folded: true });
    ok('echo pack nfcProbedVersion null', p.nfcProbedVersion === null);
    ok('echo scoring/profile versions', st.profileId === 'unicode-default' && st.scoringVersion === 'parity-v1' && st.formatVersion === 2);
    const fs3 = await import('node:fs/promises');
    for (const f of ['substring.wgsl', 'fuzzy.wgsl']) {
      const src = await fs3.readFile(new URL(`../packages/webgpu-search/src/shaders/${f}`, import.meta.url), 'utf8');
      ok(`echo ${f} never reads flagsAndProfile`, !/uni\.flagsAndProfile/.test(src));
    }
    idx.destroy();
  }

  // uFuzzy exclusion proof (M4 gate: never in the differential matrix).
  {
    const idx = await SearchIndex.create(['hello', 'hallo'], { preferGpu: false });
    let conflict = false;
    const gpuIdx = await SearchIndex.create(['hello'], { device: mockDevice, preferGpu: true });
    try {
      await gpuIdx.search('hello', { mode: 'fuzzy', cpuAlgorithm: 'ufuzzy' });
    } catch (e) {
      conflict = e instanceof IncompatibleOptionError;
    }
    ok('ufuzzy preferGpu:true + ufuzzy throws IncompatibleOptionError', conflict);
    const u = await idx.search('hello', { mode: 'fuzzy', cpuAlgorithm: 'ufuzzy' });
    ok('ufuzzy explicit stays CPU-only', u.engine === 'cpu' && u.cpuAlgorithm === 'ufuzzy');
    gpuIdx.destroy();
    idx.destroy();
  }

  // ============================================================ Part 2: CPU wiring differential
  console.log('Part 2. Harness self-check + CPU wiring differential (cells)');
  const cells = buildCells().filter((_, i) => i % shardTotal === shardIndex);
  console.log(`   ${cells.length} cells on this shard`);
  const cpuDurations: number[] = [];
  for (const cell of cells) {
    const run = async (): Promise<void> => {
      const t0 = performance.now();
      const cs = cell.caseSensitive ?? false;
      const idx = await SearchIndex.create(cell.corpus, { preferGpu: false, caseSensitive: cs });
      try {
        const res = await idx.search(cell.query, { mode: cell.mode, limit: cell.limit, caseSensitive: cs, cpuAlgorithm: 'parity' });
        const recordTokens = (idx as unknown as { recordTokens: Uint32Array[] }).recordTokens;
        const folded = (idx as unknown as { folded: boolean }).folded;
        const qT = normalizeText(cell.query, folded).tokens;
        const direct = searchCpuReference(recordTokens, qT, cell.mode, cell.limit, cell.corpus);
        const asCompared: ComparedResponse = {
          totalMatches: res.totalMatches,
          candidateCount: res.candidateCount,
          hasOverflow: res.hasOverflow,
          results: res.results,
          profileId: res.profileId,
          scoringVersion: res.scoringVersion,
          cpuAlgorithm: res.cpuAlgorithm,
        };
        const directCompared: ComparedResponse = {
          totalMatches: direct.totalMatches,
          candidateCount: Math.min(direct.totalMatches, RESULT_LIMIT_MAX),
          hasOverflow: direct.totalMatches > RESULT_LIMIT_MAX,
          results: direct.results,
          profileId: 'unicode-default',
          scoringVersion: 'parity-v1',
          cpuAlgorithm: 'parity',
        };
        const drift = compareResponses(asCompared, directCompared);
        // Text enrichment identical at SearchIndex level.
        let enrich = '';
        for (const r of res.results) {
          if (r.text !== (cell.corpus[r.index] ?? '')) {
            enrich = `enrichment drift at index ${r.index}`;
            break;
          }
        }
        // Degenerate echo rule.
        let echo = '';
        const postEmpty = normalizeText(cell.query, folded).isEmpty;
        if (postEmpty && res.query !== '') echo = `degenerate must echo '', got ${JSON.stringify(res.query)}`;
        if (!postEmpty && cell.query === 'a'.repeat(QUERY_TOKENS_MAX) && res.query !== cell.query) {
          echo = 'at-cap must echo original';
        }
        const problem = drift ?? enrich ?? echo;
        if (problem) {
          ok(`cell ${cell.name}`, false, problem);
        } else {
          passed++;
        }
      } finally {
        idx.destroy();
      }
      cpuDurations.push(performance.now() - t0);
    };
    try {
      await withTimeout(run(), CELL_TIMEOUT_MS, cell.name);
    } catch (e) {
      ok(`cell ${cell.name}`, false, String(e));
    }
  }
  console.log(`   CPU wiring cells done: median ${median(cpuDurations).toFixed(2)}ms p95 ${p95(cpuDurations).toFixed(2)}ms`);

  // Overflow multiset-scope cell (above cap: totalMatches + hasOverflow + multiset only).
  {
    const items = Array.from({ length: 9000 }, (_, i) => `ov-item-${i}`);
    const idx = await SearchIndex.create(items, { preferGpu: false });
    const res = await withTimeout(idx.search('ov-item-', { mode: 'substring', limit: 9000 }), CELL_TIMEOUT_MS, 'overflow');
    const recordTokens = (idx as unknown as { recordTokens: Uint32Array[] }).recordTokens;
    const direct = searchCpuReference(recordTokens, normalizeText('ov-item-', true).tokens, 'substring', 9000, items);
    const drift = compareResponses(
      { totalMatches: res.totalMatches, candidateCount: res.candidateCount, hasOverflow: res.hasOverflow, results: res.results, profileId: res.profileId, scoringVersion: res.scoringVersion, cpuAlgorithm: res.cpuAlgorithm },
      { totalMatches: direct.totalMatches, candidateCount: Math.min(direct.totalMatches, RESULT_LIMIT_MAX), hasOverflow: direct.totalMatches > RESULT_LIMIT_MAX, results: direct.results, profileId: 'unicode-default', scoringVersion: 'parity-v1', cpuAlgorithm: 'parity' }
    );
    ok('overflow multiset scope (9000 > 8192)', drift === null && res.hasOverflow && res.candidateCount === 8192, drift ?? '');
    idx.destroy();
  }

  // ============================================================ Part 3: GPU parity phase
  console.log('Part 3. GPU parity phase (forced-WebGPU vs forced-CPU, same cells)');
  {
    // Canary: executing device returns hits for a known-hit corpus.
    const canaryCpu = await SearchIndex.create(['hello', 'world'], { preferGpu: false });
    const canaryGpu = await SearchIndex.create(['hello', 'world'], { device: mockDevice, preferGpu: true });
    const cCpu = await canaryCpu.search('hello', { mode: 'substring' });
    const cGpu = await canaryGpu.search('hello', { mode: 'substring' });
    const executing = cGpu.totalMatches === cCpu.totalMatches && cCpu.totalMatches > 0;
    // Echo asserts hold on every path (fallback changes only engine/timings).
    ok(
      'echo profileId/scoringVersion/cpuAlgorithm identical on GPU path',
      cGpu.profileId === cCpu.profileId && cGpu.scoringVersion === cCpu.scoringVersion && cGpu.cpuAlgorithm === cCpu.cpuAlgorithm
    );
    if (!executing) {
      pend('gpu-parity cells', `non-executing device (mock GPU total=${cGpu.totalMatches} vs CPU total=${cCpu.totalMatches}); browser subset is the release gate`);
    } else {
      for (const cell of cells) {
        try {
          await withTimeout((async () => {
            const cs = cell.caseSensitive ?? false;
            const cpuIdx = await SearchIndex.create(cell.corpus, { preferGpu: false, caseSensitive: cs });
            const gpuIdx = await SearchIndex.create(cell.corpus, { device: mockDevice, preferGpu: true, caseSensitive: cs });
            try {
              const a = await cpuIdx.search(cell.query, { mode: cell.mode, limit: cell.limit, caseSensitive: cs, cpuAlgorithm: 'parity' });
              const b = await gpuIdx.search(cell.query, { mode: cell.mode, limit: cell.limit, caseSensitive: cs, cpuAlgorithm: 'parity' });
              const drift = compareResponses(
                { totalMatches: b.totalMatches, candidateCount: b.candidateCount, hasOverflow: b.hasOverflow, results: b.results, profileId: b.profileId, scoringVersion: b.scoringVersion, cpuAlgorithm: b.cpuAlgorithm },
                { totalMatches: a.totalMatches, candidateCount: a.candidateCount, hasOverflow: a.hasOverflow, results: a.results, profileId: a.profileId, scoringVersion: a.scoringVersion, cpuAlgorithm: a.cpuAlgorithm }
              );
              ok(`gpu-cell ${cell.name}`, drift === null, drift ?? '');
            } finally {
              cpuIdx.destroy();
              gpuIdx.destroy();
            }
          })(), CELL_TIMEOUT_MS, `gpu-cell ${cell.name}`);
        } catch (e) {
          ok(`gpu-cell ${cell.name}`, false, String(e));
        }
      }
    }
    // Engine-level token-only text:'' contract (holds vacuously on mock).
    const eng = new WebGPUEngine();
    await eng.init(mockDevice);
    const packed = packUnicodeToGPUBuffer(['hello', 'world'], { folded: true });
    await eng.loadDataset(serializeUnicodeDataset(packed));
    const er = await eng.search('hello', { mode: 'substring' });
    ok('engine token-only resolves text to empty', er.results.every((r) => r.text === ''));
    eng.destroy();
    canaryCpu.destroy();
    canaryGpu.destroy();
  }

  // ============================================================ Part 4: failure injection (delta)
  console.log('Part 4. Failure injection (delta-only; M3 SS20-24 stay green)');
  {
    // Shader-compile throw -> hybrid falls back to CPU with identical semantics.
    const brokenCompile = { ...mockDevice, createShaderModule: () => { throw new Error('injected compile failure'); } } as unknown as GPUDevice;
    const fbIdx = await SearchIndex.create(['hello', 'world'], { device: brokenCompile, preferGpu: true });
    ok('shader-compile throw falls back to CPU', fbIdx.getStats().engine === 'cpu');
    const fbRes = await fbIdx.search('hello', { mode: 'substring' });
    ok('fallback keeps profileId/scoringVersion', fbRes.engine === 'cpu' && fbRes.profileId === 'unicode-default' && fbRes.scoringVersion === 'parity-v1');
    fbIdx.destroy();

    // Pipeline-create reject -> same fail-closed fallback.
    const brokenPipe = { ...mockDevice, createComputePipelineAsync: () => Promise.reject(new Error('injected pipeline failure')) } as unknown as GPUDevice;
    const pipeIdx = await SearchIndex.create(['hello'], { device: brokenPipe, preferGpu: true });
    ok('pipeline-create reject falls back to CPU', pipeIdx.getStats().engine === 'cpu');
    pipeIdx.destroy();

    // Offsets-OK / records-OOM partial: no half-state (stats restored).
    {
      const eng = new WebGPUEngine();
      await eng.init(mockDevice);
      await eng.loadDataset(['stable']);
      const sizeBefore = eng.currentSize;
      const realCreate = (mockDevice as unknown as { createBuffer: (d: unknown) => GPUBuffer }).createBuffer.bind(mockDevice);
      let n = 0;
      (mockDevice as unknown as { createBuffer: (d: unknown) => GPUBuffer }).createBuffer = ((d: unknown): GPUBuffer => {
        n++;
        if (n % 2 === 0) throw new Error('injected records OOM');
        return realCreate(d);
      }) as unknown as (d: unknown) => GPUBuffer;
      let oom = false;
      try {
        await eng.loadDataset(['replacement-a', 'replacement-b']);
      } catch {
        oom = true;
      }
      (mockDevice as unknown as { createBuffer: (d: unknown) => GPUBuffer }).createBuffer = realCreate as unknown as (d: unknown) => GPUBuffer;
      ok('records-OOM throws (no silent half-load)', oom);
      ok('stats restored after partial OOM', eng.currentSize === sizeBefore);
      await eng.loadDataset(['recovered']);
      ok('engine recovers after OOM', eng.currentSize === 1);
      eng.destroy();
    }

    // mapAsync reject: epoch intact -> raw error (fallback-eligible); epoch moved -> AbortError.
    {
      const eng = new WebGPUEngine();
      await eng.init(mockDevice);
      await eng.loadDataset(['apple']);
      const sb = (eng as unknown as { stagingBuffer: { mapAsync: () => Promise<void> } }).stagingBuffer;
      const orig = sb.mapAsync.bind(sb);
      sb.mapAsync = () => Promise.reject(new Error('injected mapAsync failure'));
      let raw = false;
      try {
        await eng.search('apple', { mode: 'substring' });
      } catch (e) {
        raw = (e as Error).message === 'injected mapAsync failure';
      }
      ok('mapAsync reject propagates when epoch intact', raw);
      // Move the epoch *inside* the in-flight await (destroy-race shape):
      // gen is captured before mapAsync, so a bump during the await reads
      // as moved and must map to AbortError (never fallback-eligible).
      sb.mapAsync = () => {
        (eng as unknown as { generation: number }).generation += 1;
        return Promise.reject(new Error('injected mapAsync failure'));
      };
      let aborted = false;
      try {
        await eng.search('apple', { mode: 'substring' });
      } catch (e) {
        aborted = (e as { name?: string }).name === 'AbortError';
      }
      ok('mapAsync reject + moved epoch -> AbortError', aborted);
      sb.mapAsync = orig;
      eng.destroy();
    }

    // AbortError through SearchIndex never converts to CPU fallback.
    {
      const idx = await SearchIndex.create(['apple'], { device: mockDevice, preferGpu: true });
      const engHandle = (idx as unknown as { gpuEngine: WebGPUEngine }).gpuEngine;
      const sb = (engHandle as unknown as { stagingBuffer: { mapAsync: () => Promise<void> } }).stagingBuffer;
      const orig = sb.mapAsync.bind(sb);
      // Epoch moves mid-flight (destroy-race shape) -> AbortError, which the
      // hybrid must rethrow instead of converting to a CPU fallback.
      sb.mapAsync = () => {
        (engHandle as unknown as { generation: number }).generation += 1;
        return Promise.reject(new Error('injected mid-flight failure'));
      };
      let aborted = false;
      try {
        await idx.search('apple', { mode: 'substring' });
      } catch (e) {
        aborted = (e as { name?: string }).name === 'AbortError';
      }
      ok('AbortError propagates through SearchIndex (no fallback)', aborted);
      sb.mapAsync = orig;
      idx.destroy();
    }

    // Simulated device loss: destroy mid-suite -> subsequent index is CPU-tagged.
    {
      const idx = await SearchIndex.create(['hello'], { device: mockDevice, preferGpu: true });
      const h = (idx as unknown as { gpuEngine: WebGPUEngine }).gpuEngine;
      h.destroy();
      const after = await idx.search('hello', { mode: 'substring' });
      ok('post-destroy GPU handle serves CPU semantics', after.engine === 'cpu' && after.profileId === 'unicode-default');
      idx.destroy();
      let destroyedThrows = false;
      try {
        await idx.search('hello', { mode: 'substring' });
      } catch {
        destroyedThrows = true;
      }
      ok('destroyed index throws (fail-closed)', destroyedThrows);
    }

    // Context-manager refcount: injected devices never disturb shared state.
    {
      const { WebGPUContextManager } = await import('../packages/webgpu-search/src/index');
      const d1 = await WebGPUContextManager.acquireDevice({ device: mockDevice });
      const d2 = await WebGPUContextManager.acquireDevice({ device: mockDevice });
      ok('injected acquire is non-shared', d1 !== null && d2 !== null && d1.isShared === false && d2.isShared === false);
      WebGPUContextManager.releaseDevice(mockDevice, false);
      const d3 = await WebGPUContextManager.acquireDevice({ device: mockDevice });
      ok('non-shared release is a no-op (device still acquirable)', d3 !== null && d3.device === mockDevice);
      const unsub = WebGPUContextManager.onDeviceLost(() => {});
      unsub();
      ok('device-loss subscribe/unsubscribe round-trip', true);
    }
  }

  // ============================================================ Part 5: concurrency / abort
  console.log('Part 5. Concurrency / abort');
  {
    // loadDataset || search: single mutex, no torn state.
    {
      const eng = new WebGPUEngine();
      await eng.init(mockDevice);
      await eng.loadDataset(['base']);
      const [ld, sr] = await Promise.all([
        eng.loadDataset(['aaa', 'aab', 'aac']),
        eng.search('aaa', { mode: 'substring' }),
      ]);
      ok('loadDataset||search both settle', typeof ld.uploadTimeMs === 'number' && typeof sr.totalMs === 'undefined');
      ok('post-race dataset is the loaded one', eng.currentSize === 3);
      eng.destroy();
    }
    // Interleaved searchCold(A) x searchCold(B): query echo never crosses.
    {
      const eng = new WebGPUEngine();
      await eng.init(mockDevice);
      const corpusA = asciiCorpus(50, 1);
      const corpusB = asciiCorpus(50, 2);
      const [ra, rb] = await Promise.all([
        eng.searchCold(corpusA, 'query-alpha', { mode: 'substring' }),
        eng.searchCold(corpusB, 'query-beta', { mode: 'substring' }),
      ]);
      ok('searchCold interleave keeps query echo', ra.query === 'query-alpha' && rb.query === 'query-beta');
      ok('searchCold interleave keeps mode/timings shape', typeof ra.coldTotalMs === 'number' && typeof rb.datasetUploadMs === 'number');
      eng.destroy();
    }
    // Abort mid-mapAsync via controller (deterministic: abort inside mapAsync).
    {
      const eng = new WebGPUEngine();
      await eng.init(mockDevice);
      await eng.loadDataset(['apple']);
      const sb = (eng as unknown as { stagingBuffer: { mapAsync: () => Promise<void> } }).stagingBuffer;
      const orig = sb.mapAsync.bind(sb);
      const ac = new AbortController();
      sb.mapAsync = async (): Promise<void> => {
        ac.abort();
        return orig();
      };
      let aborted = false;
      try {
        await eng.search('apple', { mode: 'substring', signal: ac.signal });
      } catch (e) {
        aborted = (e as { name?: string }).name === 'AbortError';
      }
      ok('abort during mapAsync -> AbortError (epoch discard)', aborted);
      sb.mapAsync = orig;
      eng.destroy();
    }
    // CPU-scan abort (pre-aborted fast path on the parity scorer).
    {
      const idx = await SearchIndex.create(['apple', 'application'], { preferGpu: false });
      const ac = new AbortController();
      ac.abort();
      let aborted = false;
      try {
        await idx.search('app', { mode: 'substring', signal: ac.signal });
      } catch (e) {
        aborted = (e as { name?: string }).name === 'AbortError';
      }
      ok('CPU parity scan honors pre-aborted signal', aborted);
      idx.destroy();
    }
    // Generation-epoch last-wins: destroy parks a barrier; queued ops observe teardown.
    {
      const eng = new WebGPUEngine();
      await eng.init(mockDevice);
      await eng.loadDataset(['x']);
      eng.destroy();
      eng.destroy();
      const r = await eng.search('x', { mode: 'substring' });
      ok('double-destroy safe; post-destroy search noHits', r.totalMatches === 0);
    }
  }

  // ============================================================ Part 6: worker contract
  console.log('Part 6. Benchmark worker contract (LOAD_DATASET / SEARCH, fake self)');
  {
    const posted: unknown[] = [];
    const fakeSelf = {
      postMessage: (msg: unknown): void => {
        posted.push(msg);
      },
      onmessage: null as unknown as ((e: MessageEvent) => Promise<void>) | null,
    };
    (globalThis as unknown as { self: unknown }).self = fakeSelf;
    await import('../apps/benchmark/src/search.worker.ts');
    const send = async (data: unknown): Promise<void> => {
      if (fakeSelf.onmessage === null) throw new Error('worker onmessage not installed');
      await fakeSelf.onmessage({ data } as MessageEvent);
    };
    const take = (type: string): unknown[] => {
      const out = posted.filter((m) => (m as { type?: string }).type === type);
      posted.length = 0;
      return out;
    };

    await send({ type: 'INIT' });
    const initDone = take('INIT_DONE');
    ok('worker INIT_DONE (CPU software, no GPU here)', initDone.length === 1);

    // Legacy buffers without strings/serialized fail fast (no silent repack).
    await send({ type: 'LOAD_DATASET', payload: { recordsBufferData: new ArrayBuffer(16), offsetsBufferData: new ArrayBuffer(16) } });
    const legacy = take('DATASET_LOADED') as Array<{ payload: { error?: string } }>;
    ok('worker rejects legacy v0.1 buffers explicitly', legacy.length === 1 && typeof legacy[0]?.payload.error === 'string');

    // Neutered serialized buffer -> explicit re-create error.
    await send({ type: 'LOAD_DATASET', payload: { serialized: new ArrayBuffer(0) } });
    const neut = take('DATASET_LOADED') as Array<{ payload: { error?: string } }>;
    ok('worker neutered guard (byteLength 0)', neut.length === 1 && typeof neut[0]?.payload.error === 'string');

    // Strings path: unicode pack, metrics reported.
    await send({ type: 'LOAD_DATASET', payload: { strings: ['hello', 'world'] } });
    const loaded = take('DATASET_LOADED') as Array<{ payload: { size: number; error?: string; stringsChars: number; tokenCount: number } }>;
    ok('worker strings LOAD_DATASET (unicode, no legacy packer)', loaded.length === 1 && loaded[0]?.payload.size === 2 && loaded[0]?.payload.error === undefined);

    // Serialized path.
    const packed = packUnicodeToGPUBuffer(['alpha', 'beta'], { folded: true });
    const serialized = serializeUnicodeDataset(packed);
    await send({ type: 'LOAD_DATASET', payload: { strings: ['alpha', 'beta'], serialized } });
    const sloaded = take('DATASET_LOADED') as Array<{ payload: { size: number; error?: string; serializedBytes: number } }>;
    ok('worker serialized LOAD_DATASET (U2F2)', sloaded.length === 1 && sloaded[0]?.payload.size === 2 && sloaded[0]?.payload.serializedBytes === serialized.byteLength);

    // SEARCH: CPU comparison runs (no GPU here); latestQueryId drops stale.
    await send({ type: 'SEARCH', payload: { queryId: 10, query: 'alpha', mode: 'substring', limit: 5, runCpuComparison: true } });
    const first = take('SEARCH_RESULTS') as Array<{ payload: { queryId: number; ufuzzyResult: unknown; gpuCompact: unknown } }>;
    ok('worker SEARCH runs CPU comparison', first.length === 1 && first[0]?.payload.ufuzzyResult !== null);
    await send({ type: 'SEARCH', payload: { queryId: 20, query: 'alpha', mode: 'substring', limit: 5 } });
    await send({ type: 'SEARCH', payload: { queryId: 15, query: 'alpha', mode: 'substring', limit: 5 } });
    const dropped = take('SEARCH_RESULTS') as Array<{ payload: { queryId: number } }>;
    ok('worker latestQueryId drops superseded query', dropped.length === 1 && dropped[0]?.payload.queryId === 20);
    delete (globalThis as unknown as { self?: unknown }).self;
  }

  // ---------------------------------------------------------------- report
  console.log(`\n--- M4 harness: ${passed} passed, ${failed} failed, ${pending} pending-hardware ---`);
  if (pendingNotes.length > 0) {
    console.log('pending-hardware slots (require browser GPU, release-gated by test-regression.ts):');
    for (const n of pendingNotes) console.log(`   - ${n}`);
  }
  if (failed > 0) process.exit(1);
}

await main();
