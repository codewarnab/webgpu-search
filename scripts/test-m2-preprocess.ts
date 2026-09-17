/**
 * M2 unit tests: shared preprocessing + CPU reference (Issue #7).
 * ASCII-only source: all non-ASCII strings built via String.fromCodePoint.
 * Run: bun scripts/test-m2-preprocess.ts
 */
import {
  SearchIndex,
  normalizeText,
  tokensEqual,
  searchCpuReference,
  scoreSubstringTokens,
  scoreFuzzyTokens,
  compareParityResults,
  foldCodePoint,
  FOLD_C_COUNT,
  FOLD_F_COUNT,
  QUERY_TOKENS_MAX,
  RESULT_LIMIT_MAX,
  QueryTooLongError,
} from '../packages/webgpu-search/src/index';

const FCP = String.fromCodePoint;
const HIGH = FCP(0xd800);
const LOW = FCP(0xdc00);
const FFFD_STR = FCP(0xfffd);
const E_ACUTE = FCP(0xe9);
const E_PLUS_ACUTE = FCP(0x65, 0x301);
const A_RING = FCP(0xc5);
const A_PLUS_RING = FCP(0x41, 0x30a);
const STRASSE_CAP = 'Stra' + FCP(0xdf) + 'e';
const DOT_I_CAP = FCP(0x130);
const DOTLESS_I = FCP(0x131);
const SHARP_S = FCP(0xdf);
const FINAL_SIGMA = FCP(0x3c2);
const SIGMA = FCP(0x3c3);
const SIGMA_CAP = FCP(0x3a3);
const FF_LIG = FCP(0xfb00);
const THETA_SYM_CAP = FCP(0x3f4);
const FULLWIDTH_BANG = FCP(0xff01);
const FULLWIDTH_A_CAP = FCP(0xff21);
const WOMAN = FCP(0x1f469);
const FAMILY = FCP(0x1f468, 0x200d, 0x1f469, 0x200d, 0x1f467, 0x200d, 0x1f466);
const FLAG_US = FCP(0x1f1fa, 0x1f1f8);
const RI_U = FCP(0x1f1fa);
const AR_BARE = FCP(0x643, 0x62a, 0x627, 0x628);
const AR_VOCAL = FCP(0x643, 0x650, 0x62a, 0x64e, 0x627, 0x628);
const DEVA_PLAIN = FCP(0x915, 0x94d, 0x937);
const DEVA_ZWJ = FCP(0x915, 0x94d, 0x200d, 0x937);
const IDEO_SPACE = FCP(0x3000);
const CJK_UNIV = FCP(0x5317, 0x4eac, 0x5927, 0x5b66);
const CJK_BEIJING = FCP(0x5317, 0x4eac);
const NON_FDD0 = FCP(0xfdd0);
const NON_FFFE = FCP(0xfffe);
const NON_FFFF = FCP(0xffff);
const NON_MAX = FCP(0x10ffff);

let passed = 0;
let failed = 0;
function ok(name: string, cond: boolean, extra = ''): void {
  if (cond) {
    passed++;
    console.log(`   ok - ${name}`);
  } else {
    failed++;
    console.error(`   FAIL - ${name}${extra ? ` -- ${extra}` : ''}`);
  }
}
function arr(t: Uint32Array): number[] {
  return Array.from(t);
}

async function main(): Promise<void> {
  console.log('--- M2 preprocessing + CPU reference tests ---');

  console.log('1. Fold-table counts + spot fixtures');
  ok('C count = 1453', FOLD_C_COUNT === 1453, `got ${FOLD_C_COUNT}`);
  ok('F count = 104', FOLD_F_COUNT === 104, `got ${FOLD_F_COUNT}`);
  ok('A->a', JSON.stringify(foldCodePoint(0x41)) === JSON.stringify([0x61]));
  ok('sharp-s -> ss', JSON.stringify(foldCodePoint(0xdf)) === JSON.stringify([0x73, 0x73]));
  ok('dotted-I -> i+dot', JSON.stringify(foldCodePoint(0x130)) === JSON.stringify([0x69, 0x307]));
  ok('Sigma -> sigma', JSON.stringify(foldCodePoint(0x3a3)) === JSON.stringify([0x3c3]));
  ok('final-sigma -> sigma', JSON.stringify(foldCodePoint(0x3c2)) === JSON.stringify([0x3c3]));
  ok('ff-ligature -> ff', JSON.stringify(foldCodePoint(0xfb00)) === JSON.stringify([0x66, 0x66]));
  ok('I -> i (never dotless)', JSON.stringify(foldCodePoint(0x49)) === JSON.stringify([0x69]));
  ok('dotless-i identity', foldCodePoint(0x131) === null);
  ok('1E9E -> ss (S excluded)', JSON.stringify(foldCodePoint(0x1e9e)) === JSON.stringify([0x73, 0x73]));
  ok('theta-symbol -> theta via C', JSON.stringify(foldCodePoint(0x3f4)) === JSON.stringify([0x3b8]));
  ok('FF01 distinct (no fold)', foldCodePoint(0xff01) === null);

  console.log('2. Surrogate / noncharacter matrix');
  const FFFD = 0xfffd;
  ok('lone high -> 1 FFFD', JSON.stringify(arr(normalizeText(HIGH, true).tokens)) === JSON.stringify([FFFD]));
  ok('lone low -> 1 FFFD', JSON.stringify(arr(normalizeText(LOW, true).tokens)) === JSON.stringify([FFFD]));
  ok('high+high = 2 FFFDs', JSON.stringify(arr(normalizeText(HIGH + HIGH, true).tokens)) === JSON.stringify([FFFD, FFFD]));
  ok('high+EOF = 1 FFFD', JSON.stringify(arr(normalizeText('a' + HIGH, true).tokens)) === JSON.stringify([0x61, FFFD]));
  ok('reversed low-high = 2 FFFDs', JSON.stringify(arr(normalizeText(LOW + HIGH, true).tokens)) === JSON.stringify([FFFD, FFFD]));
  ok('genuine FFFD indistinguishable', tokensEqual(normalizeText(FFFD_STR, true).tokens, normalizeText(HIGH, true).tokens));
  ok('adjacent-to-base survives', JSON.stringify(arr(normalizeText('a' + HIGH + 'b', true).tokens)) === JSON.stringify([0x61, FFFD, 0x62]));
  ok('FDD0 survives', arr(normalizeText(NON_FDD0, true).tokens)[0] === 0xfdd0);
  ok('FFFE survives', arr(normalizeText(NON_FFFE, true).tokens)[0] === 0xfffe);
  ok('FFFF survives', arr(normalizeText(NON_FFFF, true).tokens)[0] === 0xffff);
  ok('10FFFF survives', arr(normalizeText(NON_MAX, true).tokens)[0] === 0x10ffff);
  {
    const s = String(HIGH);
    const proto = String.prototype as unknown as Record<string, unknown>;
    const saved = proto.toWellFormed;
    try {
      (proto as Record<string, unknown>).toWellFormed = undefined;
      const { toWellFormedSafe } = await import('../packages/webgpu-search/src/unicode-preprocess');
      ok('regex fallback high->FFFD', toWellFormedSafe(s) === FFFD_STR);
      ok('regex fallback high+high->2', toWellFormedSafe(HIGH + HIGH) === FFFD_STR + FFFD_STR);
    } finally {
      if (saved !== undefined) proto.toWellFormed = saved;
      else delete (proto as Record<string, unknown>).toWellFormed;
    }
  }

  console.log('3. NFC + canonical equivalence');
  ok('e+acute -> single e-acute', JSON.stringify(arr(normalizeText(E_ACUTE, true).tokens)) === JSON.stringify([0xe9]));
  ok('A+ring folded -> single a-ring (0xe5)', JSON.stringify(arr(normalizeText(A_RING, true).tokens)) === JSON.stringify([0xe5]));
  ok('A+ring NFC-only stays 0xc5', JSON.stringify(arr(normalizeText(A_RING, false).tokens)) === JSON.stringify([0xc5]));
  ok('canonically equivalent -> identical streams', tokensEqual(normalizeText(E_ACUTE, true).tokens, normalizeText(E_PLUS_ACUTE, true).tokens));
  ok('A-ring decomposed identical', tokensEqual(normalizeText(A_RING, true).tokens, normalizeText(A_PLUS_RING, true).tokens));
  ok('e vs e-acute distinct', !tokensEqual(normalizeText('e', true).tokens, normalizeText(E_ACUTE, true).tokens));

  console.log('4. Strasse highlight caveat (6->7 tokens)');
  {
    const t = normalizeText(STRASSE_CAP, true);
    ok('Strasse folds to 7 tokens', t.tokenCount === 7, `got ${t.tokenCount}`);
    ok('strasse literal identical', tokensEqual(t.tokens, normalizeText('strasse', true).tokens));
    const wrongSlice = [...STRASSE_CAP].slice(3, 6).join('');
    const expectWrong = 'a' + SHARP_S + 'e';
    ok('folded slice of original is wrong span', wrongSlice === expectWrong, `got ${wrongSlice}`);
  }

  console.log('5. Case matrix x folded true/false');
  ok('folded I->i matches i', tokensEqual(normalizeText('I', true).tokens, normalizeText('i', true).tokens));
  ok('folded I never dotless', !tokensEqual(normalizeText('I', true).tokens, normalizeText(DOTLESS_I, true).tokens));
  ok('NFC-only I distinct from i', !tokensEqual(normalizeText('I', false).tokens, normalizeText('i', false).tokens));
  ok('folded dotted-I distinct (i+dot)', JSON.stringify(arr(normalizeText(DOT_I_CAP, true).tokens)) === JSON.stringify([0x69, 0x307]));
  ok('folded sharp/ss/SS merge', tokensEqual(normalizeText(SHARP_S, true).tokens, normalizeText('ss', true).tokens) && tokensEqual(normalizeText('SS', true).tokens, normalizeText('ss', true).tokens));
  ok('NFC-only sharp distinct', !tokensEqual(normalizeText(SHARP_S, false).tokens, normalizeText('ss', false).tokens));
  ok('folded sigmas merge', tokensEqual(normalizeText(FINAL_SIGMA, true).tokens, normalizeText(SIGMA, true).tokens) && tokensEqual(normalizeText(SIGMA_CAP, true).tokens, normalizeText(SIGMA, true).tokens));
  ok('folded ff-ligature/ff merge', tokensEqual(normalizeText(FF_LIG, true).tokens, normalizeText('ff', true).tokens));
  ok('fullwidth bang vs ascii distinct', !tokensEqual(normalizeText(FULLWIDTH_BANG, true).tokens, normalizeText('!', true).tokens));
  ok('fullwidth A vs A distinct', !tokensEqual(normalizeText(FULLWIDTH_A_CAP, true).tokens, normalizeText('A', true).tokens));
  ok('tr-TR: I->i stable', arr(normalizeText('I', true).tokens)[0] === 0x69);

  console.log('6. ZWJ / flags / Arabic / Indic / CJK');
  {
    const famT = normalizeText(FAMILY, true).tokens;
    const wT = normalizeText(WOMAN, true).tokens;
    const r = scoreSubstringTokens(famT, wT);
    ok('ZWJ-family partial (scalar woman found)', r.matched && r.matchStart === 2, `start=${r.matchStart}`);
    const flag = normalizeText(FLAG_US, true).tokens;
    ok('flag = 2 RIs', flag.length === 2 && flag[0] === 0x1f1fa && flag[1] === 0x1f1f8);
    const singleRI = normalizeText(RI_U, true).tokens;
    ok('flag split (single RI found)', scoreSubstringTokens(flag, singleRI).matched);
    ok('Arabic bare vs vocalized distinct', !tokensEqual(normalizeText(AR_BARE, true).tokens, normalizeText(AR_VOCAL, true).tokens));
    const c1 = normalizeText(DEVA_PLAIN, true).tokens;
    const c2 = normalizeText(DEVA_ZWJ, true).tokens;
    ok('Devanagari ZWJ variant distinct', !tokensEqual(c1, c2));
    ok('U+3000 trims', normalizeText(IDEO_SPACE, true).isEmpty);
    ok('CJK substring', scoreSubstringTokens(normalizeText(CJK_UNIV, true).tokens, normalizeText(CJK_BEIJING, true).tokens).matched);
    const recAscii = normalizeText('a b', true).tokens;
    const recIdeo = normalizeText('a' + IDEO_SPACE + 'b', true).tokens;
    const qB = normalizeText('b', true).tokens;
    const sAscii = scoreFuzzyTokens(recAscii, qB).score;
    const sIdeo = scoreFuzzyTokens(recIdeo, qB).score;
    ok('U+3000 gives no word bonus (ASCII +30 only)', sAscii === sIdeo + 30, `ascii=${sAscii} ideo=${sIdeo}`);
  }

  console.log('7. Scorer semantics (i32, two-key sort, penalties)');
  {
    const rec = normalizeText('hello world', true).tokens;
    const q = normalizeText('world', true).tokens;
    const r = scoreSubstringTokens(rec, q);
    const expect = (1000 - 6 * 10 - (11 - 5)) | 0;
    ok('substring formula', r.matched && r.score === expect, `got ${r.score} want ${expect}`);
    const rf = scoreFuzzyTokens(normalizeText('abc', true).tokens, normalizeText('abc', true).tokens);
    ok('fuzzy exact abc=215', rf.matched && rf.score === 215, `got ${rf.score}`);
    const a = { index: 5, score: 100 };
    const b = { index: 2, score: 100 };
    ok('two-key tie-break', compareParityResults(a, b) > 0 && compareParityResults(b, a) < 0);
    const longRec = normalizeText(`${'a'.repeat(2000)}xyz`, true).tokens;
    const qx = normalizeText('xyz', true).tokens;
    const rn = scoreSubstringTokens(longRec, qx);
    ok('negative score (long+late)', rn.matched && rn.score < 0, `got ${rn.score}`);
    const huge = normalizeText(`${'a'.repeat(70000)}xyz`, true).tokens;
    ok('>2^16 length preserved', huge.length === 70003, `got ${huge.length}`);
    const rh = scoreSubstringTokens(huge, qx);
    const expectH = (1000 - Math.imul(70000, 10) - (70003 - 3)) | 0;
    ok('huge-vector score exact', rh.matched && rh.score === expectH, `got ${rh.score} want ${expectH}`);
  }

  console.log('8. SearchIndex wiring (post-fold gate, tokenCount, limits)');
  {
    const idx = await SearchIndex.create([STRASSE_CAP, 'STRASSE', 'other'], { preferGpu: false });
    const stats = idx.getStats();
    ok('tokenCount post-fold (7+7+5=19)', stats.tokenCount === 19, `got ${stats.tokenCount}`);
    const res = await idx.search('strasse', { mode: 'substring' });
    ok('sharp/ss folded search finds both', res.totalMatches === 2 && res.results[0].index === 0 && res.results[1].index === 1);
    ok('tie-break by index', res.results[0].score === res.results[1].score);
    const atCap = SHARP_S.repeat(64);
    const overCap = SHARP_S.repeat(65);
    ok('post-fold at-cap count=128', normalizeText(atCap, true).tokenCount === 128);
    ok('post-fold over-cap count=130', normalizeText(overCap, true).tokenCount === 130);
    const idx2 = await SearchIndex.create(['a'], { preferGpu: false });
    let threw = false;
    try {
      await idx2.search(overCap, { mode: 'substring' });
    } catch (e) {
      threw = e instanceof QueryTooLongError;
    }
    ok('over-limit throws QueryTooLongError (post-fold)', threw);
    const fb = await idx2.search(overCap, { mode: 'substring', onQueryTooLong: 'cpu-fallback' });
    ok('cpu-fallback forces CPU', fb.engine === 'cpu');
    const emptyRes = await idx2.search('   ', { mode: 'substring' });
    ok('whitespace-only -> query empty', emptyRes.query === '' && emptyRes.totalMatches === 0);
    console.log('8b. Degenerate survival (NOT empty; echoes original)');
    {
      const MARK = FCP(0x301);
      const VS = FCP(0xfe0f);
      const ZWJ = FCP(0x200d);
      const TATWEEL = FCP(0x640);
      ok('lone-mark survives (1 token)', normalizeText(MARK, true).tokenCount === 1);
      ok('VS-only survives (1 token)', normalizeText(VS, true).tokenCount === 1);
      ok('ZWJ-only survives (1 token)', normalizeText(ZWJ, true).tokenCount === 1);
      ok('tatweel-only survives (1 token)', normalizeText(TATWEEL, true).tokenCount === 1);
      const rMark = await idx2.search(MARK, { mode: 'substring' });
      const rVS = await idx2.search(VS, { mode: 'substring' });
      const rZWJ = await idx2.search(ZWJ, { mode: 'substring' });
      const rTat = await idx2.search(TATWEEL, { mode: 'substring' });
      ok('lone-mark echoes original', rMark.query === MARK && rMark.totalMatches === 0);
      ok('VS-only echoes original', rVS.query === VS && rVS.totalMatches === 0);
      ok('ZWJ-only echoes original', rZWJ.query === ZWJ && rZWJ.totalMatches === 0);
      ok('tatweel-only echoes original', rTat.query === TATWEEL && rTat.totalMatches === 0);
      const rIde = await idx2.search(IDEO_SPACE, { mode: 'substring' });
      ok('U+3000-only -> query empty', rIde.query === '' && rIde.totalMatches === 0);
    }
    idx.destroy();
    idx2.destroy();
  }

  console.log('9. Overflow (8191/8192/8193) + limit sweep');
  {
    const mk = (n: number): string[] => Array.from({ length: n }, (_v, i) => `match-${i}`);
    const items = mk(8193);
    const idx = await SearchIndex.create(items, { preferGpu: false });
    const q = 'match-';
    const r8191 = await idx.search(q, { mode: 'substring', limit: 8191 });
    ok('8193 corpus limit=8191 -> 8191 results', r8191.results.length === 8191);
    const rMax = await idx.search(q, { mode: 'substring', limit: 8192 });
    ok('limit=8192 -> 8192 results, overflow true', rMax.results.length === 8192 && rMax.hasOverflow === true && rMax.totalMatches === 8193);
    ok('candidateCount capped', rMax.candidateCount === 8192);
    const idx2 = await SearchIndex.create(mk(8192), { preferGpu: false });
    const rExact = await idx2.search(q, { mode: 'substring', limit: 8192 });
    ok('8192 total -> hasOverflow false', rExact.hasOverflow === false && rExact.totalMatches === 8192);
    const lim0 = await idx2.search(q, { mode: 'substring', limit: 0 });
    const lim1 = await idx2.search(q, { mode: 'substring', limit: 1 });
    const lim2 = await idx2.search(q, { mode: 'substring', limit: 2 });
    const limNaN = await idx2.search(q, { mode: 'substring', limit: NaN });
    const limUndef = await idx2.search(q, { mode: 'substring' });
    const limHuge = await idx2.search(q, { mode: 'substring', limit: 9000 });
    const limAlias = await idx2.search(q, { mode: 'substring', maxResults: 3 });
    ok('limit=0 clamps to 1', lim0.results.length === 1);
    ok('limit=1 -> 1', lim1.results.length === 1);
    ok('limit=2 -> 2', lim2.results.length === 2);
    ok('limit=NaN -> 50', limNaN.results.length === 50);
    ok('limit=undef -> 50', limUndef.results.length === 50);
    ok('limit=9000 clamps 8192', limHuge.results.length === 8192);
    ok('maxResults alias=3', limAlias.results.length === 3);
    idx.destroy();
    idx2.destroy();
  }

  console.log('10. x20 stability (byte-identical under cap)');
  {
    const idx = await SearchIndex.create(['apple', 'application', 'banana', 'app'], { preferGpu: false });
    const first = await idx.search('app', { mode: 'fuzzy', limit: 10 });
    let stable = true;
    for (let i = 0; i < 20; i++) {
      const r = await idx.search('app', { mode: 'fuzzy', limit: 10 });
      if (r.totalMatches !== first.totalMatches || r.results.length !== first.results.length) {
        stable = false;
        break;
      }
      for (let k = 0; k < r.results.length; k++) {
        if (r.results[k]?.index !== first.results[k]?.index || r.results[k]?.score !== first.results[k]?.score) {
          stable = false;
          break;
        }
      }
      if (!stable) break;
    }
    ok('20x identical ordered (index,score)', stable);
    idx.destroy();
  }

  console.log('11. uFuzzy != parity (negative proof) + quarantine');
  {
    const items = ['hello', 'hallo', 'hollow', 'help'];
    const idx = await SearchIndex.create(items, { preferGpu: false });
    const parity = await idx.search('hello', { mode: 'substring', cpuAlgorithm: 'parity' });
    const ufuzzy = await idx.search('hello', { mode: 'fuzzy', cpuAlgorithm: 'ufuzzy' });
    const differ =
      parity.totalMatches !== ufuzzy.totalMatches ||
      parity.results.length !== ufuzzy.results.length ||
      parity.results.some((r, i) => r.index !== ufuzzy.results[i]?.index || r.score !== ufuzzy.results[i]?.score);
    ok('uFuzzy != parity on >=1 fixture', differ, `parity=${parity.totalMatches} ufuzzy=${ufuzzy.totalMatches}`);
    ok('parity echo', parity.cpuAlgorithm === 'parity');
    ok('ufuzzy echo', ufuzzy.cpuAlgorithm === 'ufuzzy');
    const recT = items.map((s) => normalizeText(s, true).tokens);
    const qT = normalizeText('hello', true).tokens;
    const ref = searchCpuReference(recT, qT, 'substring', 10, items);
    ok('parity substring finds exactly hello', ref.totalMatches === 1 && ref.results[0]?.index === 0);
    idx.destroy();
  }

  console.log(`\n--- M2 tests: ${passed} passed, ${failed} failed ---`);
  if (failed > 0) process.exit(1);
  if (QUERY_TOKENS_MAX !== 128 || RESULT_LIMIT_MAX !== 8192) {
    console.error('Frozen caps drifted!');
    process.exit(1);
  }
}

await main();
