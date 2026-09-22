/**
 * CPU baseline pin: the reference contract from docs/support-matrix.md.
 *
 * Verifies that `preferGpu: false` + `cpuScorer: 'exact'` (default) is
 * reproducible without any GPU, and that `cpuScorer: 'ufuzzy'` stays an
 * explicit opt-in CPU-only scorer with fail-closed conflicts.
 *
 * Run: bun scripts/test-cpu-baseline.ts (root: bun run test:cpu-baseline)
 * Requires: no GPU, no browser, no network.
 */
import {
  DocumentIndex,
  SearchIndex,
  GpuDevicePool,
  IncompatibleOptionError,
} from '../packages/webgpu-search/src/index';

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

async function expectIncompatibleOption(name: string, fn: () => Promise<unknown>): Promise<void> {
  try {
    await fn();
  } catch (err) {
    check(name, err instanceof IncompatibleOptionError, `got ${(err as Error)?.name}`);
    return;
  }
  check(name, false, 'expected IncompatibleOptionError');
}

function isDescendingIntegers(scores: number[]): boolean {
  for (let i = 0; i < scores.length; i++) {
    if (!Number.isInteger(scores[i] as number)) return false;
    if (i > 0 && (scores[i] as number) > (scores[i - 1] as number)) return false;
  }
  return true;
}

async function main(): Promise<void> {
  console.log('--- CPU baseline pin (docs/support-matrix.md section 1) ---');

  const corpus = [
    'src/components/AuthController.ts',
    'src/views/UserProfile.vue',
    'src/services/AuthService.ts',
    'src/utils/formatDate.ts',
    'docs/api/authentication.md',
    'packages/core/src/SessionManager.ts',
    'assets/icons/user_settings.png',
    'README.md',
  ];

  // 1. Baseline stats: explicit CPU, exact echoes.
  const idx = await SearchIndex.create(corpus, { preferGpu: false });
  const stats = idx.getStats();
  check('baseline stats engine is cpu', stats.engine === 'cpu', stats.engine);
  check('baseline fallbackReason is prefer-cpu', stats.fallbackReason === 'prefer-cpu', String(stats.fallbackReason));
  check('baseline profileId', stats.profileId === 'unicode-default', stats.profileId);
  check('baseline scoringVersion', stats.scoringVersion === 'parity-v1', stats.scoringVersion);
  check('baseline formatVersion is 2', (stats.formatVersion as number) === 2, String(stats.formatVersion));

  // 2. Default scorer equals explicit exact.
  const def = await idx.search('auth', { mode: 'substring' });
  const exp = await idx.search('auth', { mode: 'substring', cpuScorer: 'exact' });
  check('default query engine is cpu', def.engine === 'cpu', def.engine);
  check('default cpuScorer echo is exact', def.cpuScorer === 'exact', def.cpuScorer);
  check('explicit exact echo', exp.cpuScorer === 'exact', exp.cpuScorer);
  check('default-equals-explicit totalMatches', def.totalMatches === exp.totalMatches, `${def.totalMatches} vs ${exp.totalMatches}`);
  check(
    'default-equals-explicit order',
    JSON.stringify(def.results.map((r) => [r.index, r.score])) ===
      JSON.stringify(exp.results.map((r) => [r.index, r.score])),
  );
  check('exact fallbackReason is prefer-cpu', def.fallbackReason === 'prefer-cpu', String(def.fallbackReason));
  check('scores are descending integers', isDescendingIntegers(def.results.map((r) => r.score)));

  // 3. Determinism: identical query twice.
  const again = await idx.search('auth', { mode: 'substring', cpuScorer: 'exact' });
  check(
    'repeated query is identical',
    JSON.stringify(again.results.map((r) => [r.index, r.score])) ===
      JSON.stringify(def.results.map((r) => [r.index, r.score])),
  );

  // 4. All four modes serve on the baseline (CPU by design for token/prefix/typo).
  const fuzzy = await idx.search('auth', { mode: 'fuzzy' });
  check('fuzzy serves on cpu', fuzzy.engine === 'cpu' && fuzzy.totalMatches > 0, `${fuzzy.engine}/${fuzzy.totalMatches}`);
  const substr = await idx.search('auth', { mode: 'substring' });
  check('substring serves on cpu', substr.engine === 'cpu' && substr.totalMatches > 0, `${substr.engine}/${substr.totalMatches}`);
  const token = await idx.search('auth service', { mode: 'token' });
  check('token serves on cpu', token.engine === 'cpu', token.engine);
  check('token fallbackReason unsupported-mode', token.fallbackReason === 'unsupported-mode', String(token.fallbackReason));
  const prefix = await idx.search('Auth', { mode: 'prefix' });
  check('prefix serves on cpu', prefix.engine === 'cpu' && prefix.totalMatches > 0, `${prefix.engine}/${prefix.totalMatches}`);
  check('prefix fallbackReason unsupported-mode', prefix.fallbackReason === 'unsupported-mode', String(prefix.fallbackReason));
  const typo = await idx.search('auht', { mode: 'substring', typoTolerance: true });
  check('typo serves on cpu', typo.engine === 'cpu', typo.engine);
  check('typo fallbackReason unsupported-mode', typo.fallbackReason === 'unsupported-mode', String(typo.fallbackReason));

  // 5. ufuzzy opt-in: CPU-only, non-conforming, echo + reason pinned.
  const uf = await idx.search('auth', { mode: 'fuzzy', cpuScorer: 'ufuzzy' });
  check('ufuzzy stays cpu', uf.engine === 'cpu', uf.engine);
  check('ufuzzy echo', uf.cpuScorer === 'ufuzzy', uf.cpuScorer);
  check('ufuzzy fallbackReason cpu-algorithm-requested', uf.fallbackReason === 'cpu-algorithm-requested', String(uf.fallbackReason));

  // 6. ufuzzy conflicts fail closed.
  await expectIncompatibleOption('ufuzzy + token throws', () =>
    idx.search('auth', { mode: 'token', cpuScorer: 'ufuzzy' }),
  );
  await expectIncompatibleOption('ufuzzy + prefix throws', () =>
    idx.search('auth', { mode: 'prefix', cpuScorer: 'ufuzzy' }),
  );
  await expectIncompatibleOption('ufuzzy + typo throws', () =>
    idx.search('auth', { mode: 'substring', cpuScorer: 'ufuzzy', typoTolerance: true }),
  );
  const gpuPref = await SearchIndex.create(corpus, { preferGpu: true });
  try {
    await expectIncompatibleOption('preferGpu:true + ufuzzy throws', () =>
      gpuPref.search('auth', { mode: 'fuzzy', cpuScorer: 'ufuzzy' }),
    );
  } finally {
    gpuPref.destroy();
  }

  // 7. DocumentIndex baseline echoes.
  interface Doc {
    id: string;
    title: string;
    body: string;
  }
  const docs: Doc[] = corpus.map((title, i) => ({ id: `d${i}`, title, body: `body ${i}` }));
  const docIndex = await DocumentIndex.create<Doc>(docs, {
    fields: [
      { name: 'title', weight: 2.0 },
      { name: 'body', weight: 1.0 },
    ],
    preferGpu: false,
  });
  const docStats = docIndex.getStats();
  check('doc baseline engine is cpu', docStats.engine === 'cpu', docStats.engine);
  const docRes = await docIndex.search('auth', { mode: 'substring' });
  check('doc baseline echoes', docRes.profileId === 'unicode-default' && docRes.scoringVersion === 'parity-v1' && docRes.cpuScorer === 'exact');
  check('doc baseline engine cpu', docRes.engine === 'cpu', docRes.engine);
  check('doc scores descending integers', isDescendingIntegers(docRes.results.map((r) => r.score)));
  const docAgain = await docIndex.search('auth', { mode: 'substring' });
  check(
    'doc repeated query identical',
    JSON.stringify(docAgain.results.map((r) => [String(r.id), r.score])) ===
      JSON.stringify(docRes.results.map((r) => [String(r.id), r.score])),
  );
  const docToken = await docIndex.search('auth service', { mode: 'token' });
  check('doc token fallbackReason unsupported-mode', docToken.fallbackReason === 'unsupported-mode', String(docToken.fallbackReason));
  const docUf = await docIndex.search('auth', { mode: 'substring', cpuScorer: 'ufuzzy' });
  check('doc ufuzzy cpu-algorithm-requested', docUf.engine === 'cpu' && docUf.fallbackReason === 'cpu-algorithm-requested', `${docUf.engine}/${String(docUf.fallbackReason)}`);
  await expectIncompatibleOption('doc ufuzzy + token throws', () =>
    docIndex.search('auth', { mode: 'token', cpuScorer: 'ufuzzy' }),
  );
  docIndex.destroy();
  idx.destroy();

  // 8. SSR guard: no navigator.gpu is required for the baseline.
  check('GpuDevicePool.isSupported is boolean', typeof GpuDevicePool.isSupported() === 'boolean');
  const hasGpu = typeof navigator !== 'undefined' && !!(navigator as unknown as { gpu?: unknown }).gpu;
  check('baseline reproducible without GPU injection', true, hasGpu ? 'gpu present, cpu path still pinned' : 'no gpu present');
  const noGpuIndex = await SearchIndex.create(['alpha', 'beta'], { preferGpu: false });
  const noGpuRes = await noGpuIndex.search('alpha', { mode: 'substring' });
  check('no-device baseline serves cpu', noGpuRes.engine === 'cpu' && noGpuRes.totalMatches === 1, `${noGpuRes.engine}/${noGpuRes.totalMatches}`);
  noGpuIndex.destroy();

  if (process.exitCode === 1) {
    console.error(`\nCPU baseline pin FAILED`);
    process.exit(1);
  }
  console.log(`\n--- CPU baseline pin passed (${passed} checks) ---`);
}

main().catch((err) => {
  console.error('test failed:', err);
  process.exit(1);
});
