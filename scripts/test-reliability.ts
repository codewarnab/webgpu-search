/**
 * Phase 2 reliability proofs (docs/ISSUE-11-PLAN.md):
 * device-loss rebuild, resource cleanup, concurrent-query isolation,
 * deterministic fallback.
 *
 * Headless: uses vgpu/mock injected devices (non-executing) + the exact CPU
 * contract as the semantic oracle. The executing-hardware subset stays in
 * `scripts/test-regression.ts` (browser gate).
 *
 * Run: bun scripts/test-reliability.ts (root: bun run test:reliability)
 * Requires: no GPU, no browser, no network.
 */
import { createMockAdapter } from 'vgpu/mock';
import {
  DocumentIndex,
  SearchIndex,
  SearchWorkerClient,
  GpuDevicePool,
  scoreExactMatches,
  type DocumentSearchResponse,
} from '../packages/webgpu-search/src/index';

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
function isDescendingIntegers(scores: number[]): boolean {
  for (let i = 0; i < scores.length; i++) {
    if (!Number.isInteger(scores[i] as number)) return false;
    if (i > 0 && (scores[i] as number) > (scores[i - 1] as number)) return false;
  }
  return true;
}
function orderedPairs(res: { results: Array<{ index: number; score: number }> }): string {
  return JSON.stringify(res.results.map((r) => [r.index, r.score]));
}
function docOrderedPairs(res: DocumentSearchResponse<any>): string {
  return JSON.stringify(res.results.map((r) => [String(r.id), r.score, r.matchedField]));
}

async function main(): Promise<void> {
  console.log('--- Phase 2 reliability proofs (device-loss / cleanup / concurrency / fallback) ---');
  const mockAdapter = createMockAdapter({ features: ['timestamp-query'] as never });
  const mockDeviceWrapper = await mockAdapter.requestDevice();
  const mockDevice = mockDeviceWrapper.gpu as unknown as GPUDevice;

  const corpus = [
    'src/components/AuthController.ts',
    'src/views/UserProfile.vue',
    'src/services/AuthService.ts',
    'src/utils/formatDate.ts',
    'docs/api/authentication.md',
    'packages/core/src/SessionManager.ts',
  ];
  interface Doc { id: string; title: string; body: string }
  const docs: Doc[] = corpus.map((title, i) => ({ id: `d${i}`, title, body: `body ${i} auth service` }));

  // ============================================================ 1. Device-loss rebuild
  console.log('1. Device-loss rebuild (explicit path + identical semantics)');
  {
    // --- SearchIndex ---
    const idx = await SearchIndex.create(corpus, { device: mockDevice, preferGpu: true });
    ok('search-index starts on webgpu (mock)', idx.getStats().engine === 'webgpu', idx.getStats().engine);
    const listenersBefore = GpuDevicePool.getListenerCount();
    ok('search-index holds one device-loss listener', listenersBefore >= 1, String(listenersBefore));
    const preLoss = await idx.search('auth', { mode: 'substring', cpuScorer: 'exact' });
    const prePairs = orderedPairs(preLoss);

    GpuDevicePool.simulateDeviceLoss('phase2-test');
    ok('search-index falls back to cpu after loss', idx.getStats().engine === 'cpu', idx.getStats().engine);
    ok('search-index fallbackReason is device-lost', idx.getStats().fallbackReason === 'device-lost', String(idx.getStats().fallbackReason));

    const postLoss = await idx.search('auth', { mode: 'substring', cpuScorer: 'exact' });
    const baseline = await SearchIndex.create(corpus, { preferGpu: false });
    try {
      const ref = await baseline.search('auth', { mode: 'substring', cpuScorer: 'exact' });
      ok(
        'post-loss cpu matches baseline contract',
        postLoss.totalMatches === ref.totalMatches &&
          postLoss.candidateCount === ref.candidateCount &&
          postLoss.hasOverflow === ref.hasOverflow &&
          orderedPairs(postLoss) === orderedPairs(ref) &&
          postLoss.profileId === ref.profileId &&
          postLoss.scoringVersion === ref.scoringVersion,
        `loss=${orderedPairs(postLoss)} ref=${orderedPairs(ref)}`,
      );
      ok('post-loss scores are descending integers', isDescendingIntegers(postLoss.results.map((r) => r.score)));
    } finally {
      baseline.destroy();
    }

    const rebuilt = await idx.rebuildGpu({ device: mockDevice });
    ok('search-index rebuildGpu returns true', rebuilt === true);
    ok('search-index engine back on webgpu', idx.getStats().engine === 'webgpu', idx.getStats().engine);
    const postRebuild = await idx.search('auth', { mode: 'substring', cpuScorer: 'exact' });
    ok('rebuild preserves identical gpu-path semantics', orderedPairs(postRebuild) === prePairs, `${orderedPairs(postRebuild)} vs ${prePairs}`);
    // Second rebuild is a no-op true (already webgpu, no listener leak).
    const listenersMid = GpuDevicePool.getListenerCount();
    const rebuiltAgain = await idx.rebuildGpu({ device: mockDevice });
    ok('second rebuildGpu is no-op true', rebuiltAgain === true);
    ok('rebuild cycle leaks no listeners', GpuDevicePool.getListenerCount() === listenersMid, `${GpuDevicePool.getListenerCount()} vs ${listenersMid}`);
    // preferGpu:false rebuild is a no-op false.
    const cpuOnly = await SearchIndex.create(corpus, { preferGpu: false });
    try {
      ok('preferGpu:false rebuild is no-op false', (await cpuOnly.rebuildGpu()) === false);
    } finally {
      cpuOnly.destroy();
    }
    // Empty-corpus rebuild is a no-op false (nothing to upload).
    const emptyFlat = await SearchIndex.create([], { preferGpu: true });
    try {
      ok('empty-corpus rebuild is no-op false', (await emptyFlat.rebuildGpu({ device: mockDevice })) === false);
    } finally {
      emptyFlat.destroy();
    }
    // Destroyed index rebuild throws fail-closed.
    idx.destroy();
    let threw = false;
    try {
      await idx.rebuildGpu({ device: mockDevice });
    } catch {
      threw = true;
    }
    ok('destroyed search-index rebuild throws', threw);
  }
  {
    // --- DocumentIndex ---
    const didx = await DocumentIndex.create<Doc>(docs, {
      fields: [
        { name: 'title', weight: 2.0 },
        { name: 'body', weight: 1.0 },
      ],
      device: mockDevice,
      preferGpu: true,
    });
    ok('document-index starts on webgpu (mock)', didx.getStats().engine === 'webgpu', didx.getStats().engine);
    const pre = await didx.search('auth', { mode: 'substring' });
    const prePairs = docOrderedPairs(pre);

    GpuDevicePool.simulateDeviceLoss('phase2-test-doc');
    ok('document-index falls back to cpu after loss', didx.getStats().engine === 'cpu', didx.getStats().engine);
    ok('document-index fallbackReason is device-lost', didx.getStats().fallbackReason === 'device-lost', String(didx.getStats().fallbackReason));

    const postLoss = await didx.search('auth', { mode: 'substring' });
    const cpuRef = await DocumentIndex.create<Doc>(docs, {
      fields: [
        { name: 'title', weight: 2.0 },
        { name: 'body', weight: 1.0 },
      ],
      preferGpu: false,
    });
    try {
      const ref = await cpuRef.search('auth', { mode: 'substring' });
      ok(
        'document post-loss cpu matches baseline contract',
        postLoss.totalMatches === ref.totalMatches &&
          docOrderedPairs(postLoss) === docOrderedPairs(ref) &&
          postLoss.profileId === ref.profileId &&
          postLoss.scoringVersion === ref.scoringVersion,
        `loss=${docOrderedPairs(postLoss)} ref=${docOrderedPairs(ref)}`,
      );
    } finally {
      cpuRef.destroy();
    }

    const rebuilt = await didx.rebuildGpu({ device: mockDevice });
    ok('document rebuildGpu returns true', rebuilt === true);
    ok('document engine back on webgpu', didx.getStats().engine === 'webgpu', didx.getStats().engine);
    const postRebuild = await didx.search('auth', { mode: 'substring' });
    ok('document rebuild preserves identical semantics', docOrderedPairs(postRebuild) === prePairs, `${docOrderedPairs(postRebuild)} vs ${prePairs}`);
    // Mutations still work after rebuild (corpus retained, epoch advances).
    // Note: the mock device never executes WGSL (0-hit GPU path), so the
    // mutation check uses CPU-by-design `token` mode which reads the
    // retained rowTokens directly on every path.
    const epochBefore = didx.getStats().mutationEpoch;
    await didx.add({ id: 'd-new', title: 'Auth audit log', body: 'new record' });
    ok('mutation works after rebuild', didx.getStats().mutationEpoch === epochBefore + 1);
    const afterMut = await didx.search('audit', { mode: 'token' });
    ok('post-rebuild mutation is searchable', afterMut.totalMatches >= 1);
    didx.destroy();
    let threw = false;
    try {
      await didx.rebuildGpu({ device: mockDevice });
    } catch {
      threw = true;
    }
    ok('destroyed document-index rebuild throws', threw);
    // Empty-corpus document rebuild is a no-op false.
    const emptyDoc = await DocumentIndex.create<Doc>([], {
      fields: [
        { name: 'title', weight: 2.0 },
        { name: 'body', weight: 1.0 },
      ],
      preferGpu: true,
    });
    try {
      ok('empty-corpus document rebuild is no-op false', (await emptyDoc.rebuildGpu({ device: mockDevice })) === false);
    } finally {
      emptyDoc.destroy();
    }
  }

  // ============================================================ 2. Resource cleanup
  console.log('2. Resource cleanup (multi-index create/destroy, worker DESTROY)');
  {
    const baseListeners = GpuDevicePool.getListenerCount();
    const baseRef = GpuDevicePool.getRefCount();
    const N = 8;
    const flat: SearchIndex[] = [];
    const docIdxs: DocumentIndex<Doc>[] = [];
    for (let i = 0; i < N; i++) {
      flat.push(await SearchIndex.create(corpus, { device: mockDevice, preferGpu: true }));
      docIdxs.push(
        await DocumentIndex.create<Doc>(docs, {
          fields: [{ name: 'title', weight: 2.0 }, { name: 'body', weight: 1.0 }],
          device: mockDevice,
          preferGpu: true,
        }),
      );
    }
    ok('multi-index all on webgpu', flat.every((x) => x.getStats().engine === 'webgpu') && docIdxs.every((x) => x.getStats().engine === 'webgpu'));
    ok('listener count grows by 2N', GpuDevicePool.getListenerCount() === baseListeners + 2 * N, `${GpuDevicePool.getListenerCount()} vs ${baseListeners + 2 * N}`);
    // Injected devices are non-shared: refcount untouched.
    ok('injected devices keep shared refcount at baseline', GpuDevicePool.getRefCount() === baseRef, String(GpuDevicePool.getRefCount()));
    for (const x of flat) x.destroy();
    for (const x of docIdxs) x.destroy();
    ok('listener count returns to baseline after destroy', GpuDevicePool.getListenerCount() === baseListeners, `${GpuDevicePool.getListenerCount()} vs ${baseListeners}`);
    // Double-destroy safe + post-destroy stats read 0/empty.
    for (const x of flat) x.destroy();
    for (const x of docIdxs) x.destroy();
    ok('double-destroy safe', true);
    ok('post-destroy flat stats read 0', flat[0]!.getStats().size === 0 && flat[0]!.getStats().vramAllocatedBytes === 0);
    ok('post-destroy doc stats read 0', docIdxs[0]!.getStats().docCount === 0 && docIdxs[0]!.getStats().vramAllocatedBytes === 0);
    let threw = false;
    try {
      await flat[0]!.search('auth');
    } catch {
      threw = true;
    }
    ok('post-destroy search throws fail-closed', threw);
    // Symbol.dispose symmetry on both index types.
    const d1 = await SearchIndex.create(corpus, { preferGpu: false });
    (d1 as unknown as { [Symbol.dispose]: () => void })[Symbol.dispose]();
    let disposedThrew = false;
    try {
      await d1.search('auth');
    } catch {
      disposedThrew = true;
    }
    ok('SearchIndex Symbol.dispose destroys', disposedThrew);
    const d2 = await DocumentIndex.create<Doc>(docs, { fields: ['title'], preferGpu: false });
    d2[Symbol.dispose]();
    let disposed2Threw = false;
    try {
      await d2.search('auth');
    } catch {
      disposed2Threw = true;
    }
    ok('DocumentIndex Symbol.dispose destroys', disposed2Threw);
  }
  {
    // Worker DESTROY: pending isolation + teardown.
    const clients: SearchWorkerClient<Doc>[] = [];
    for (let i = 0; i < 4; i++) {
      const c = new SearchWorkerClient<Doc>({ stringIsolated: true });
      await c.init(docs, { fields: ['title', 'body'], preferGpu: false });
      clients.push(c);
    }
    const stats = await clients[0]!.getStats();
    ok('worker stats report live index', stats.docCount === docs.length, String(stats.docCount));
    for (const c of clients) await c.destroy();
    let threw = false;
    try {
      await clients[0]!.search('auth');
    } catch (e: any) {
      threw = String(e?.message ?? '').includes('destroyed');
    }
    ok('worker search after destroy throws', threw);
    // Double-destroy safe.
    for (const c of clients) await c.destroy();
    ok('worker double-destroy safe', true);
  }

  // ============================================================ 3. Concurrent-query isolation
  console.log('3. Concurrent-query isolation (direct + worker rapid typing)');
  {
    // Direct path: concurrent queries are independent — none auto-aborts.
    const idx = await SearchIndex.create(corpus, { preferGpu: false });
    try {
      const queries = ['auth', 'service', 'utils', 'docs', 'session', 'user'];
      const results = await Promise.all(queries.map((q) => idx.search(q, { mode: 'substring', limit: 10 })));
      ok(
        'direct concurrent queries stay isolated (echo + order)',
        results.every((r, i) => r.query === queries[i]),
        JSON.stringify(results.map((r) => r.query)),
      );
      // Each matches the exact-scorer oracle for its own query.
      const recordTokens = (idx as unknown as { recordTokens: Uint32Array[] }).recordTokens;
      const { normalizeText } = await import('../packages/webgpu-search/src/index');
      let oracleOk = true;
      for (let i = 0; i < queries.length; i++) {
        const qT = normalizeText(queries[i] as string, true).tokens;
        const direct = scoreExactMatches(recordTokens, qT, 'substring', 10, corpus);
        if (orderedPairs(results[i] as { results: Array<{ index: number; score: number }> }) !== orderedPairs(direct)) {
          oracleOk = false;
          break;
        }
      }
      ok('direct concurrent results match exact oracle each', oracleOk);
      // Pre-aborted signals reject with AbortError without poisoning siblings.
      const live = idx.search('auth', { mode: 'substring' });
      let aborted = false;
      try {
        await idx.search('service', { mode: 'substring', signal: AbortSignal.abort() });
      } catch (e: any) {
        aborted = e?.name === 'AbortError';
      }
      const liveRes = await live;
      ok('pre-aborted direct query rejects AbortError', aborted);
      ok('live sibling survives aborted sibling', liveRes.query === 'auth');
    } finally {
      idx.destroy();
    }

    // DocumentIndex direct path: same independence.
    const didx = await DocumentIndex.create<Doc>(docs, { fields: ['title', 'body'], preferGpu: false });
    try {
      const qs = ['auth', 'service', 'body'];
      const out = await Promise.all(qs.map((q) => didx.search(q, { mode: 'substring', limit: 10 })));
      ok(
        'document direct concurrent isolated',
        out.every((r, i) => r.query === qs[i]),
        JSON.stringify(out.map((r) => r.query)),
      );
    } finally {
      didx.destroy();
    }

    // Worker path: monotonic sequencing aborts superseded queries.
    const client = new SearchWorkerClient<Doc>({ stringIsolated: true });
    await client.init(docs, { fields: ['title', 'body'], preferGpu: false });
    try {
      const p1 = client.search('a');
      const p2 = client.search('au');
      const p3 = client.search('aut');
      const p4 = client.search('auth');
      let aborts = 0;
      for (const p of [p1, p2, p3]) {
        try {
          await p;
        } catch (e: any) {
          if (e?.name === 'AbortError') aborts++;
        }
      }
      const last = await p4;
      ok('worker rapid typing aborts superseded (3/3)', aborts === 3, `aborts=${aborts}`);
      ok('worker last query resolves with echo', last.query === 'auth' && last.totalMatches > 0);
      // Caller AbortSignal on worker path.
      let sigAborted = false;
      try {
        await client.search('auth', { signal: AbortSignal.abort() });
      } catch (e: any) {
        sigAborted = e?.name === 'AbortError';
      }
      ok('worker pre-aborted signal rejects AbortError', sigAborted);
    } finally {
      await client.destroy();
    }
  }

  // ============================================================ 4. Deterministic fallback
  console.log('4. Deterministic fallback (unavailable + lost == CPU contract)');
  {
    // GPU-unavailable (no device, headless) matches the exact oracle.
    const cpuIdx = await SearchIndex.create(corpus, { preferGpu: false });
    try {
      const res = await cpuIdx.search('auth', { mode: 'substring', cpuScorer: 'exact' });
      const recordTokens = (cpuIdx as unknown as { recordTokens: Uint32Array[] }).recordTokens;
      const { normalizeText } = await import('../packages/webgpu-search/src/index');
      const direct = scoreExactMatches(recordTokens, normalizeText('auth', true).tokens, 'substring', 50, corpus);
      ok(
        'gpu-unavailable matches exact oracle',
        res.engine === 'cpu' && orderedPairs(res) === orderedPairs(direct),
        `engine=${res.engine}`,
      );
      ok('gpu-unavailable fallbackReason prefer-cpu', res.fallbackReason === 'prefer-cpu', String(res.fallbackReason));
      const tokenRes = await cpuIdx.search('auth service', { mode: 'token' });
      ok('token cpu-by-design unsupported-mode', tokenRes.fallbackReason === 'unsupported-mode', String(tokenRes.fallbackReason));
    } finally {
      cpuIdx.destroy();
    }

    // GPU-lost matches the same oracle with device-lost reason.
    const gpuIdx = await SearchIndex.create(corpus, { device: mockDevice, preferGpu: true });
    try {
      GpuDevicePool.simulateDeviceLoss('phase2-fallback');
      const lost = await gpuIdx.search('auth', { mode: 'substring', cpuScorer: 'exact' });
      const refIdx = await SearchIndex.create(corpus, { preferGpu: false });
      try {
        const ref = await refIdx.search('auth', { mode: 'substring', cpuScorer: 'exact' });
        ok(
          'gpu-lost matches cpu contract identically',
          lost.totalMatches === ref.totalMatches &&
            lost.candidateCount === ref.candidateCount &&
            lost.hasOverflow === ref.hasOverflow &&
            orderedPairs(lost) === orderedPairs(ref),
        );
      } finally {
        refIdx.destroy();
      }
      ok('gpu-lost fallbackReason device-lost', lost.fallbackReason === 'device-lost', String(lost.fallbackReason));
      ok('gpu-lost echoes intact', lost.profileId === 'unicode-default' && lost.scoringVersion === 'parity-v1' && lost.cpuScorer === 'exact');
      // Unsupported-mode stays unsupported-mode even after loss (routing, not scorer).
      const lostToken = await gpuIdx.search('auth service', { mode: 'token' });
      ok('post-loss token still unsupported-mode', lostToken.fallbackReason === 'unsupported-mode', String(lostToken.fallbackReason));
      // Rebuild recovers to webgpu (explicit path, not silent).
      ok('fallback index rebuilds to webgpu', (await gpuIdx.rebuildGpu({ device: mockDevice })) === true);
      ok('rebuilt engine is webgpu', gpuIdx.getStats().engine === 'webgpu');
    } finally {
      gpuIdx.destroy();
    }

    // gpu-execution-error dispatch failure falls back to identical CPU semantics.
    const failIdx = await SearchIndex.create(corpus, { device: mockDevice, preferGpu: true });
    try {
      const handle = (failIdx as unknown as { gpuEngine: { search: (...a: any[]) => Promise<unknown> } }).gpuEngine;
      const orig = handle.search.bind(handle);
      (handle as { search: (...a: any[]) => Promise<unknown> }).search = async () => {
        throw new Error('injected dispatch failure');
      };
      try {
        const fb = await failIdx.search('auth', { mode: 'substring' });
        const refIdx = await SearchIndex.create(corpus, { preferGpu: false });
        try {
          const ref = await refIdx.search('auth', { mode: 'substring' });
          ok(
            'dispatch-failure fallback matches cpu contract',
            fb.engine === 'cpu' && orderedPairs(fb) === orderedPairs(ref),
          );
        } finally {
          refIdx.destroy();
        }
        ok('dispatch-failure reason gpu-execution-error', fb.fallbackReason === 'gpu-execution-error', String(fb.fallbackReason));
      } finally {
        (handle as { search: (...a: any[]) => Promise<unknown> }).search = orig;
      }
    } finally {
      failIdx.destroy();
    }

    // DocumentIndex GPU-unavailable vs lost parity.
    const dCpu = await DocumentIndex.create<Doc>(docs, { fields: ['title', 'body'], preferGpu: false });
    try {
      const r = await dCpu.search('auth', { mode: 'substring' });
      ok('document gpu-unavailable is cpu prefer-cpu', r.engine === 'cpu' && r.fallbackReason === 'prefer-cpu', `${r.engine}/${String(r.fallbackReason)}`);
    } finally {
      dCpu.destroy();
    }
  }

  console.log(`\n--- reliability proofs: ${passed} passed, ${failed} failed ---`);
  if (failed > 0) process.exit(1);
}

main().catch((err) => {
  console.error('test failed:', err);
  process.exit(1);
});
