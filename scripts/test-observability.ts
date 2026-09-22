import assert from 'node:assert';
import { createMockAdapter } from 'vgpu/mock';
import {
  DocumentIndex,
  SearchIndex,
  SearchWorkerClient,
  encodeSnapshot,
  restoreSnapshot,
  GpuDevicePool,
  QUERY_TOKENS_MAX,
  type DocumentId,
  type FallbackReason,
  type DocumentIndexStats,
} from '../packages/webgpu-search/src/index';

import { createVanillaSearchApp } from '../examples/vanilla/search-app';
import { useSearch } from '../examples/vue/useSearch';
import { createDocumentSearch } from '../examples/svelte/documentSearchStore';

interface TestDoc {
  id: string;
  title: string;
  category: string;
  content: string;
}

const SAMPLE_DOCS: TestDoc[] = [
  { id: '1', title: 'WebGPU Shading Pipelines', category: 'GPU', content: 'WGSL compute shaders parallel search' },
  { id: '2', title: 'Unicode Canonical Decomposition', category: 'Text', content: 'NFD and NFC normalization code points' },
  { id: '3', title: 'IndexedDB Snapshot Storage', category: 'Persistence', content: 'snapshot Little-Endian binary format with CRC32' },
  { id: '4', title: 'Worker Client Concurrency', category: 'Threading', content: 'Off-thread async search with abort controllers' },
];

async function runM7Tests() {
  console.log('--- Running  Observability & Framework Integration Tests ---');

  // =========================================================================
  // 1. DocumentIndexStats Telemetry Schema & Memory Breakdown
  // =========================================================================
  console.log('1. Testing DocumentIndexStats telemetry schema and memory breakdown...');
  {
    const index = await DocumentIndex.create(SAMPLE_DOCS, {
      fields: ['title', 'category', 'content'],
      preferGpu: false
    });

    const stats = index.getStats();
    assert.strictEqual(stats.docCount, 4);
    assert.strictEqual(stats.rowCount, 12);
    assert.strictEqual(stats.size, 4);
    assert.strictEqual(stats.tombstoneCount, 0);
    assert.strictEqual(stats.tombstoneRatio, 0);
    assert.strictEqual(stats.engine, 'cpu');
    assert.strictEqual(stats.fallbackReason, 'prefer-cpu');
    assert.strictEqual(stats.formatVersion, 4);
    assert.strictEqual(stats.scoringVersion, 'parity-v1');
    assert.strictEqual(stats.unicodeVersion, '16.0.0');
    assert.strictEqual(stats.profileId, 'unicode-default');
    assert.strictEqual(stats.mutationEpoch, 0);
    assert(typeof stats.buildTimeMs === 'number' && stats.buildTimeMs >= 0);
    assert.strictEqual(stats.restoreTimeMs, undefined);
    assert.strictEqual(stats.lastMutationTimeMs, undefined);

    // Memory breakdown validation
    assert.strictEqual(stats.memory.vramBytes, 0);
    assert(stats.memory.ramBytes > 0);
    assert.strictEqual(stats.memory.totalBytes, stats.memory.ramBytes);
    assert.strictEqual(stats.memory.tokenRamBytes, stats.tokenCount * 4);
    assert.strictEqual(stats.memory.offsetRamBytes, (stats.rowCount + 1) * 4);
    assert.strictEqual(stats.memory.ramBytes, stats.memory.tokenRamBytes + stats.memory.offsetRamBytes);

    index.destroy();
    console.log('   ✅ DocumentIndexStats schema and memory breakdown verified');
  }

  // =========================================================================
  // 2. Dynamic Mutation Telemetry & Epoch Counters
  // =========================================================================
  console.log('2. Testing mutation telemetry, lastMutationTimeMs, and epoch counters...');
  {
    const index = await DocumentIndex.create(SAMPLE_DOCS, {
      fields: ['title', 'content'],
      preferGpu: false
    });

    // A. Add mutation
    const addResult = await index.add({
      id: '5',
      title: 'Monaco Editor Integration',
      category: 'UI',
      content: 'QuickOpen fuzzy symbol search palette'
    });

    assert.strictEqual(addResult.mutationEpoch, 1);
    assert(typeof addResult.durationMs === 'number' && addResult.durationMs >= 0);
    let stats = index.getStats();
    assert.strictEqual(stats.mutationEpoch, 1);
    assert.strictEqual(stats.docCount, 5);
    assert.strictEqual(stats.rowCount, 10);
    assert.strictEqual(stats.lastMutationTimeMs, addResult.durationMs);

    // B. Update mutation
    const updateResult = await index.update({
      id: '5',
      title: 'Monaco QuickOpen Palette Revised',
      category: 'UI',
      content: 'Optimized sub-millisecond keyboard navigation'
    });

    assert.strictEqual(updateResult.mutationEpoch, 2);
    stats = index.getStats();
    assert.strictEqual(stats.mutationEpoch, 2);
    assert.strictEqual(stats.docCount, 5);
    assert.strictEqual(stats.tombstoneCount, 2); // 2 fields updated = 2 old rows tombstoned
    assert.strictEqual(stats.tombstoneRatio, 2 / 12);
    assert.strictEqual(stats.lastMutationTimeMs, updateResult.durationMs);

    // C. Remove mutation (triggers automatic compaction since 4 / 12 = 33.3% >= 25% threshold)
    const removeResult = await index.remove('5');
    assert.strictEqual(removeResult.mutationEpoch, 3);
    assert.strictEqual(removeResult.compacted, true);
    stats = index.getStats();
    assert.strictEqual(stats.mutationEpoch, 3);
    assert.strictEqual(stats.docCount, 4);
    assert.strictEqual(stats.tombstoneCount, 0); // compacted cleanly!
    assert.strictEqual(stats.tombstoneRatio, 0);
    assert.strictEqual(stats.rowCount, 8);
    assert.strictEqual(stats.lastMutationTimeMs, removeResult.durationMs);

    index.destroy();
    console.log('   ✅ Mutation telemetry and epoch advancement verified');
  }

  // =========================================================================
  // 3. Persistence Telemetry & restoreTimeMs
  // =========================================================================
  console.log('3. Testing persistence telemetry and restoreTimeMs tracking...');
  {
    const index = await DocumentIndex.create(SAMPLE_DOCS, {
      fields: ['title', 'category'],
      preferGpu: false
    });

    await index.add({ id: 'persisted-1', title: 'Persistence Test', category: 'Test', content: '' });
    const originalStats = index.getStats();
    assert.strictEqual(originalStats.mutationEpoch, 1);

    const snapshot = encodeSnapshot(index);
    index.destroy();

    const restoredIndex = await restoreSnapshot<TestDoc>(snapshot, { preferGpu: false });
    const restoredStats = restoredIndex.getStats();

    assert.strictEqual(restoredStats.docCount, 5);
    assert.strictEqual(restoredStats.mutationEpoch, 1);
    assert(typeof restoredStats.restoreTimeMs === 'number' && restoredStats.restoreTimeMs >= 0);
    assert.strictEqual(restoredStats.tombstoneCount, 0);
    assert.strictEqual(restoredStats.engine, 'cpu');
    assert.strictEqual(restoredStats.fallbackReason, 'prefer-cpu');

    restoredIndex.destroy();
    console.log('   ✅ Persistence restoreTimeMs and state preservation verified');
  }

  // =========================================================================
  // 4. Verification of All 9 FallbackReason Values
  // =========================================================================
  console.log('4. Testing comprehensive verification of all 9 FallbackReason values...');
  {
    // 4.1 'prefer-cpu'
    {
      const idx = await DocumentIndex.create(SAMPLE_DOCS, { fields: ['title'], preferGpu: false });
      assert.strictEqual(idx.getStats().fallbackReason, 'prefer-cpu');
      const res = await idx.search('pipeline');
      assert.strictEqual(res.fallbackReason, 'prefer-cpu');
      idx.destroy();
      console.log('   ✅ FallbackReason 1/9: prefer-cpu verified');
    }

    // 4.2 'below-threshold'
    {
      const idx = await DocumentIndex.create(SAMPLE_DOCS, {
        fields: ['title'],
        threshold: 100_000,
        preferGpu: undefined
      });
      assert.strictEqual(idx.getStats().fallbackReason, 'below-threshold');
      const res = await idx.search('pipeline');
      assert.strictEqual(res.fallbackReason, 'below-threshold');
      idx.destroy();
      console.log('   ✅ FallbackReason 2/9: below-threshold verified');
    }

    // 4.3 'query-too-long'
    {
      const idx = await DocumentIndex.create(SAMPLE_DOCS, { fields: ['title'], preferGpu: false });
      const longQuery = 'word '.repeat(QUERY_TOKENS_MAX + 5);
      const res = await idx.search(longQuery, { onQueryTooLong: 'cpu-fallback' });
      assert.strictEqual(res.fallbackReason, 'query-too-long');
      assert.strictEqual(res.engine, 'cpu');
      idx.destroy();
      console.log('   ✅ FallbackReason 3/9: query-too-long verified');
    }

    // 4.4 'cpu-algorithm-requested'
    {
      const idx = await DocumentIndex.create(SAMPLE_DOCS, { fields: ['title'], preferGpu: false });
      const res = await idx.search('pipeline', { cpuScorer: 'ufuzzy' });
      assert.strictEqual(res.fallbackReason, 'cpu-algorithm-requested');
      assert.strictEqual(res.engine, 'cpu');
      idx.destroy();
      console.log('   ✅ FallbackReason 4/9: cpu-algorithm-requested verified');
    }

    // 4.5 'webgpu-unsupported'
    {
      const originalNav = (globalThis as any).navigator;
      try {
        (globalThis as any).navigator = {};
        const idx = await DocumentIndex.create(SAMPLE_DOCS, {
          fields: ['title'],
          threshold: 1 // should attempt GPU but navigator.gpu is absent
        });
        assert.strictEqual(idx.getStats().fallbackReason, 'webgpu-unsupported');
        const res = await idx.search('pipeline');
        assert.strictEqual(res.fallbackReason, 'webgpu-unsupported');
        idx.destroy();
        console.log('   ✅ FallbackReason 5/9: webgpu-unsupported verified');
      } finally {
        (globalThis as any).navigator = originalNav;
      }
    }

    // 4.6 'device-request-failed'
    {
      const originalNav = (globalThis as any).navigator;
      try {
        (globalThis as any).navigator = {
          gpu: {
            requestAdapter: async () => {
              throw new Error('Adapter acquisition denied by security policy');
            }
          }
        };
        const idx = await DocumentIndex.create(SAMPLE_DOCS, {
          fields: ['title'],
          threshold: 1
        });
        assert.strictEqual(idx.getStats().fallbackReason, 'device-request-failed');
        const res = await idx.search('pipeline');
        assert.strictEqual(res.fallbackReason, 'device-request-failed');
        idx.destroy();
        console.log('   ✅ FallbackReason 6/9: device-request-failed verified');
      } finally {
        (globalThis as any).navigator = originalNav;
      }
    }

    // 4.7 'device-lost'
    {
      let deviceLostHandler: (() => void) | null = null;
      const originalOnDeviceLost = GpuDevicePool.onDeviceLost;
      GpuDevicePool.onDeviceLost = (fn: () => void) => {
        deviceLostHandler = fn;
        return () => { deviceLostHandler = null; };
      };

      try {
        const mockAdapter = createMockAdapter({ features: ['timestamp-query'] as any });
        const mockDeviceWrapper = await mockAdapter.requestDevice();
        const mockDevice = mockDeviceWrapper.gpu;

        const idx = await DocumentIndex.create(SAMPLE_DOCS, {
          fields: ['title'],
          device: mockDevice,
          threshold: 1
        });
        assert.strictEqual(idx.getStats().engine, 'webgpu');
        assert(deviceLostHandler !== null, 'deviceLostHandler should be registered');

        deviceLostHandler();
        const stats = idx.getStats();
        assert.strictEqual(stats.engine, 'cpu');
        assert.strictEqual(stats.fallbackReason, 'device-lost');

        const queryRes = await idx.search('pipeline');
        assert.strictEqual(queryRes.engine, 'cpu');
        assert.strictEqual(queryRes.fallbackReason, 'device-lost');

        idx.destroy();
        console.log('   ✅ FallbackReason 7/9: device-lost verified');
      } finally {
        GpuDevicePool.onDeviceLost = originalOnDeviceLost;
      }
    }

    // 4.8 'memory-budget-exceeded'
    {
      const tinyLimitsDevice: any = {
        limits: { maxBufferSize: 16, maxStorageBufferBindingSize: 16 } // Too small for 4 documents
      };
      const idx = await DocumentIndex.create(SAMPLE_DOCS, {
        fields: ['title', 'content'],
        device: tinyLimitsDevice,
        threshold: 1
      });
      assert.strictEqual(idx.getStats().fallbackReason, 'memory-budget-exceeded');
      assert.strictEqual(idx.getStats().engine, 'cpu');

      const queryRes = await idx.search('pipeline');
      assert.strictEqual(queryRes.engine, 'cpu');
      assert.strictEqual(queryRes.fallbackReason, 'memory-budget-exceeded');

      idx.destroy();
      console.log('   ✅ FallbackReason 8/9: memory-budget-exceeded verified');
    }

    // 4.9 'gpu-execution-error'
    {
      const mockAdapter = createMockAdapter({ features: ['timestamp-query'] as any });
      const mockDeviceWrapper = await mockAdapter.requestDevice();
      const mockDevice = mockDeviceWrapper.gpu;

      const idx = await DocumentIndex.create(SAMPLE_DOCS, {
        fields: ['title'],
        device: mockDevice,
        threshold: 1
      });
      assert.strictEqual(idx.getStats().engine, 'webgpu');

      // Inject query failure into mock device queue submit
      mockDevice.queue.submit = () => {
        throw new Error('GPU query execution failed inside hardware driver');
      };

      const res = await idx.search('pipeline');
      assert.strictEqual(res.engine, 'cpu');
      assert.strictEqual(res.fallbackReason, 'gpu-execution-error');
      idx.destroy();
      console.log('   ✅ FallbackReason 9/9: gpu-execution-error verified');
    }
  }

  // =========================================================================
  // 5. SearchWorkerClient Telemetry Forwarding
  // =========================================================================
  console.log('5. Testing SearchWorkerClient telemetry forwarding...');
  {
    const client = new SearchWorkerClient<TestDoc>({
      stringIsolated: true
    });

    await client.init(SAMPLE_DOCS, {
      fields: ['title', 'category', 'content'],
      preferGpu: false
    });

    const stats = await client.getStats();
    assert.strictEqual(stats.docCount, 4);
    assert.strictEqual(stats.rowCount, 12);
    assert.strictEqual(stats.engine, 'cpu');
    assert.strictEqual(stats.fallbackReason, 'prefer-cpu');
    assert.strictEqual(stats.mutationEpoch, 0);
    assert(stats.memory.ramBytes > 0);

    // Search through worker
    const res = await client.search('pipeline', { tag: 'mark' });
    assert(res.timings.totalMs >= 0);
    assert.strictEqual(res.fallbackReason, 'prefer-cpu');
    assert.strictEqual(res.results.length, 1);
    assert.strictEqual(res.results[0].id, '1');

    // Mutations through worker
    const addRes = await client.add({
      id: 'w-1',
      title: 'Worker Dynamic Doc',
      category: 'Worker',
      content: 'Added via SearchWorkerClient'
    });
    assert.strictEqual(addRes.mutationEpoch, 1);
    assert(addRes.durationMs >= 0);

    const updatedStats = await client.getStats();
    assert.strictEqual(updatedStats.mutationEpoch, 1);
    assert.strictEqual(updatedStats.docCount, 5);

    // Concurrency: simultaneous searches and getStats
    let priorAborted = false;
    const p1 = client.search('pipeline').catch((err) => {
      if (err?.name === 'AbortError') priorAborted = true;
    });
    const p2 = client.search('Worker');
    const p3 = client.getStats();

    const [, res2, stats3] = await Promise.all([p1, p2, p3]);
    assert.strictEqual(priorAborted, true, 'Superseded concurrent query should reject with AbortError');
    assert(res2 && res2.results.length >= 1);
    assert.strictEqual(stats3.docCount, 5);

    await client.destroy();
    console.log('   ✅ SearchWorkerClient telemetry and mutation forwarding verified');
  }

  // =========================================================================
  // 6. SearchIndex (Low-Level Hybrid Index) Telemetry Parity
  // =========================================================================
  console.log('6. Testing SearchIndex low-level telemetry parity...');
  {
    const items = ['alpha item', 'beta item', 'gamma item'];
    const sIndex = await SearchIndex.create(items, { preferGpu: false });

    const stats = sIndex.getStats();
    assert.strictEqual(stats.size, 3);
    assert.strictEqual(stats.engine, 'cpu');
    assert.strictEqual(stats.fallbackReason, 'prefer-cpu');
    assert(stats.memory !== undefined);
    assert.strictEqual(stats.memory.vramBytes, 0);
    assert(stats.memory.ramBytes > 0);
    assert.strictEqual(stats.memory.totalBytes, stats.memory.ramBytes);

    const searchRes = await sIndex.search('alpha');
    assert.strictEqual(searchRes.engine, 'cpu');
    assert.strictEqual(searchRes.fallbackReason, 'prefer-cpu');

    // Legacy ufuzzy fallback reason
    const ufuzzyRes = await sIndex.search('alpha', { cpuScorer: 'ufuzzy' });
    assert.strictEqual(ufuzzyRes.fallbackReason, 'cpu-algorithm-requested');

    sIndex.destroy();
    console.log('   ✅ SearchIndex telemetry parity verified');
  }

  // =========================================================================
  // 7. Framework Recipes: Vanilla Recipe Execution
  // =========================================================================
  console.log('7. Testing Vanilla search app recipe execution and lifecycle...');
  {
    // Mock minimal DOM container
    const createMockEl = () => ({
      value: '',
      textContent: '',
      style: {},
      innerHTML: '',
      addEventListener: () => {},
      removeEventListener: () => {},
      appendChild: () => {}
    });

    const mockDoc: any = {
      createElement: () => createMockEl()
    };

    const mockContainer: any = {
      innerHTML: '',
      ownerDocument: mockDoc,
      querySelector: () => createMockEl()
    };

    const idx = await DocumentIndex.create(SAMPLE_DOCS, { fields: ['title'], preferGpu: false });
    const app = createVanillaSearchApp(mockContainer as any, {
      index: idx,
      searchOptions: { mode: 'fuzzy', limit: 5 }
    });

    const searchRes = await app.search('shading');
    assert(searchRes !== null);
    assert.strictEqual(searchRes.results.length, 1);
    assert.strictEqual(searchRes.results[0].id, '1');

    // Add doc
    const addRes = await app.add({ id: 'v-1', title: 'Vanilla Added Item' });
    assert.strictEqual(addRes.mutationEpoch, 1);

    const stats = await app.refreshStats();
    assert(stats !== null);
    assert.strictEqual(stats.docCount, 5);

    app.destroy();
    idx.destroy();
    console.log('   ✅ Vanilla recipe execution and lifecycle verified');
  }

  // =========================================================================
  // 8. Framework Recipes: Vue Composable (useSearch)
  // =========================================================================
  console.log('8. Testing Vue useSearch composable recipe...');
  {
    const idx = await DocumentIndex.create(SAMPLE_DOCS, { fields: ['title', 'category'], preferGpu: false });
    const vueSearch = useSearch<TestDoc>({
      index: idx,
      debounceMs: 0
    });

    const searchRes = await vueSearch.search('decomposition');
    assert(searchRes !== null);
    assert.strictEqual(searchRes.results.length, 1);
    assert.strictEqual(searchRes.results[0].id, '2');
    assert.strictEqual(vueSearch.results.value.length, 1);
    assert.strictEqual(vueSearch.engine.value, 'cpu');
    assert.strictEqual(vueSearch.fallbackReason.value, 'prefer-cpu');

    // Add mutation
    const addRes = await vueSearch.add({ id: 'vue-1', title: 'Vue Composition Doc', category: 'Vue', content: '' });
    assert.strictEqual(addRes.mutationEpoch, 1);
    assert.strictEqual(vueSearch.mutationEpoch.value, 1);

    // Reactive typing watcher
    vueSearch.query.value = 'Pipelines';
    await new Promise((resolve) => setTimeout(resolve, 30));
    assert.strictEqual(vueSearch.results.value[0]?.id, '1');

    vueSearch.destroy();
    idx.destroy();
    console.log('   ✅ Vue useSearch composable verified');
  }

  // =========================================================================
  // 9. Framework Recipes: Svelte Store (createDocumentSearch)
  // =========================================================================
  console.log('9. Testing Svelte createDocumentSearch store recipe...');
  {
    const idx = await DocumentIndex.create(SAMPLE_DOCS, { fields: ['title', 'content'], preferGpu: false });
    const svelteStore = createDocumentSearch<TestDoc>({
      index: idx,
      debounceMs: 0
    });

    let currentState: any = null;
    const unsub = svelteStore.subscribe((state) => {
      currentState = state;
    });

    assert(currentState !== null);
    const searchRes = await svelteStore.search('storage');
    assert(searchRes !== null);
    assert.strictEqual(searchRes.results.length, 1);
    assert.strictEqual(searchRes.results[0].id, '3');
    assert.strictEqual(currentState.results.length, 1);
    assert.strictEqual(currentState.engine, 'cpu');

    // Add mutation
    const addRes = await svelteStore.add({ id: 'svelte-1', title: 'Svelte Store Doc', category: 'Svelte', content: '' });
    assert.strictEqual(addRes.mutationEpoch, 1);
    assert.strictEqual(currentState.mutationEpoch, 1);

    // Reactive typing watcher
    svelteStore.setQuery('Pipelines');
    await new Promise((resolve) => setTimeout(resolve, 30));
    assert.strictEqual(currentState.results[0]?.id, '1');

    unsub();
    svelteStore.destroy();
    idx.destroy();
    console.log('   ✅ Svelte createDocumentSearch store verified');
  }

  // =========================================================================
  // 10. Framework Recipes: React Hook Logic & Debounce/Abort State Machine
  // =========================================================================
  console.log('10. Testing React hook state machine & abort logic...');
  {
    const idx = await DocumentIndex.create(SAMPLE_DOCS, { fields: ['title'], preferGpu: false });

    // Emulate hook request sequencing & AbortController cancellation
    let latestSearchId = 0;
    let activeAbort: AbortController | null = null;
    let committedResults: any = null;

    async function hookSearch(query: string, delayMs: number) {
      const searchId = ++latestSearchId;
      if (activeAbort) activeAbort.abort();
      activeAbort = new AbortController();
      const signal = activeAbort.signal;

      try {
        if (delayMs > 0) {
          await new Promise((resolve) => setTimeout(resolve, delayMs));
        }
        if (signal.aborted) throw new Error('AbortError');
        const res = await idx.search(query, { signal });
        if (searchId === latestSearchId) {
          committedResults = res;
        }
        return res;
      } catch (err: any) {
        if (err.message === 'AbortError' || signal.aborted) return null;
        throw err;
      }
    }

    // Launch slow query 1, then quick query 2 immediately
    const p1 = hookSearch('shading', 50);
    const p2 = hookSearch('unicode', 10);
    await Promise.all([p1, p2]);

    assert(committedResults !== null);
    assert.strictEqual(committedResults.query, 'unicode');
    assert.strictEqual(committedResults.results[0].id, '2');

    idx.destroy();
    console.log('   ✅ React hook state machine & abort logic verified');
  }

  console.log('--- All Observability & Recipe Tests Passed! ✅ ---');
}

runM7Tests().catch((err) => {
  console.error('❌  Observability tests failed:', err);
  process.exit(1);
});
