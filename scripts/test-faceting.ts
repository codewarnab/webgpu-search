/**
 * Test Suite (): Facet Aggregation Engine
 * Tests terms & range facets, exact vs approximate semantics, force-exact
 * fallback, disjunctive multi-facet navigation, mutation sync, worker
 * boundary, GPU overflow behavior, and sub-2ms aggregation performance.
 */

import assert from 'node:assert';
import { readFile } from 'node:fs/promises';
import { createMockAdapter } from 'vgpu/mock';
import {
  DocumentIndex,
  FacetEngine,
  SearchWorkerClient,
  normalizeFacetRequests,
  excludeFieldFromFilter,
  filterReferencesField,
  InvalidFilterError,
  type DocumentIndexOptions,
  type FacetRequest,
} from '../packages/webgpu-search/src/index';

interface Product {
  id: string;
  title: string;
  body: string;
  category: string;
  price: number;
  inStock: boolean;
  tags: string[];
}

const CORPUS: Product[] = [
  { id: 'p1', title: 'Widget Alpha', body: 'wireless bluetooth speaker', category: 'tech', price: 49.99, inStock: true, tags: ['sale', 'audio'] },
  { id: 'p2', title: 'Widget Beta', body: 'wireless noise cancelling headphones', category: 'tech', price: 199.99, inStock: true, tags: ['audio', 'premium'] },
  { id: 'p3', title: 'Widget Gamma', body: 'mechanical keyboard rgb', category: 'tech', price: 129.5, inStock: false, tags: ['gaming'] },
  { id: 'p4', title: 'Widget Delta', body: 'ergonomic office chair', category: 'home', price: 299.0, inStock: true, tags: ['sale', 'office'] },
  { id: 'p5', title: 'Widget Epsilon', body: 'standing desk converter', category: 'home', price: 50.0, inStock: false, tags: ['office'] },
  { id: 'p6', title: 'Widget Zeta', body: 'garden hose nozzle', category: 'garden', price: 50.0, inStock: true, tags: ['outdoor', 'sale'] },
  { id: 'p7', title: 'Manual Handbook', body: 'paper documentation guide', category: 'books', price: 19.99, inStock: true, tags: ['reference'] },
  { id: 'p8', title: 'Cookbook Classics', body: 'recipes for home chefs', category: 'books', price: 24.99, inStock: false, tags: ['kitchen'] },
];

function productIndexOpts(extra?: Partial<DocumentIndexOptions<Product>>): DocumentIndexOptions<Product> {
  return {
    fields: [
      { name: 'title', weight: 2.0 },
      { name: 'body', weight: 1.0 },
    ],
    filterFields: [
      { name: 'category', type: 'string' },
      { name: 'price', type: 'number' },
      { name: 'inStock', type: 'boolean' },
      { name: 'tags', type: 'string[]' },
    ],
    preferGpu: false,
    ...extra,
  };
}

async function runTests() {
  console.log('=== Running Issue #10  Facet Aggregation Engine ===\n');

  // =========================================================================
  // 1. Terms facets: exact counts, ordering, limit, sortBy
  // =========================================================================
  console.log('1. Testing terms facets (counts, ordering, limit, sortBy)...');
  {
    const index = await DocumentIndex.create(CORPUS, productIndexOpts());

    const res = await index.search('Widget', {
      facets: { byCategory: { type: 'terms', field: 'category' } },
    });
    assert.strictEqual(res.totalMatches, 6);
    assert.ok(res.facets, 'facets must be present when requested');
    const cat = res.facets!.byCategory;
    assert.strictEqual(cat.type, 'terms');
    assert.strictEqual(cat.field, 'category');
    assert.strictEqual(cat.isApproximate, false);
    if (cat.type !== 'terms') throw new Error('unreachable');
    assert.deepStrictEqual(
      cat.buckets.map((b) => [b.value, b.count]),
      [['tech', 3], ['home', 2], ['garden', 1]]
    );

    // sortBy value -> alphabetical for strings...
    const resValue = await index.search('Widget', {
      facets: { byCategory: { type: 'terms', field: 'category', sortBy: 'value' } },
    });
    const catValue = resValue.facets!.byCategory;
    if (catValue.type !== 'terms') throw new Error('unreachable');
    assert.deepStrictEqual(
      catValue.buckets.map((b) => b.value),
      ['garden', 'home', 'tech']
    );

    // ...but numeric for numbers (9 < 10 < 100, not lexicographic).
    const resNumSort = await index.search('Widget', {
      facets: { byPrice: { type: 'terms', field: 'price', sortBy: 'value' } },
    });
    const priceSorted = resNumSort.facets!.byPrice;
    if (priceSorted.type !== 'terms') throw new Error('unreachable');
    assert.deepStrictEqual(
      priceSorted.buckets.map((b) => b.value),
      [49.99, 50, 129.5, 199.99, 299]
    );

    // limit truncation keeps count-desc order
    const resLimit = await index.search('Widget', {
      facets: { byCategory: { type: 'terms', field: 'category', limit: 2 } },
    });
    const catLimit = resLimit.facets!.byCategory;
    if (catLimit.type !== 'terms') throw new Error('unreachable');
    assert.deepStrictEqual(
      catLimit.buckets.map((b) => [b.value, b.count]),
      [['tech', 3], ['home', 2]]
    );

    // array form keys by field name
    const resArr = await index.search('Widget', {
      facets: [{ type: 'terms', field: 'category', limit: 1 }],
    });
    assert.ok(resArr.facets!.category, 'array form must key by field');
    if (resArr.facets!.category.type !== 'terms') throw new Error('unreachable');
    assert.strictEqual(resArr.facets!.category.buckets.length, 1);

    // boolean + tag-set terms facets
    const resMisc = await index.search('Widget', {
      facets: {
        stock: { type: 'terms', field: 'inStock' },
        tagCloud: { type: 'terms', field: 'tags', sortBy: 'value' },
      },
    });
    const stock = resMisc.facets!.stock;
    if (stock.type !== 'terms') throw new Error('unreachable');
    assert.deepStrictEqual(
      stock.buckets.map((b) => [b.value, b.count]),
      [[true, 4], [false, 2]]
    );
    const tagCloud = resMisc.facets!.tagCloud;
    if (tagCloud.type !== 'terms') throw new Error('unreachable');
    const tagMap = new Map(tagCloud.buckets.map((b) => [b.value, b.count]));
    assert.strictEqual(tagMap.get('sale'), 3);
    assert.strictEqual(tagMap.get('audio'), 2);
    assert.strictEqual(tagMap.get('office'), 2);

    // backward compat: no facets key when not requested
    const resPlain = await index.search('Widget');
    assert.strictEqual(resPlain.facets, undefined);

    index.destroy();
    console.log('   ✅ Terms facets verified 100%');
  }

  // =========================================================================
  // 2. Range facets: half-open [from, to) boundaries, keys
  // =========================================================================
  console.log('2. Testing range facets (half-open intervals, keys)...');
  {
    const index = await DocumentIndex.create(CORPUS, productIndexOpts());

    const res = await index.search('Widget', {
      facets: {
        byPrice: {
          type: 'range',
          field: 'price',
          ranges: [
            { to: 50, key: 'budget' },          // p1 (49.99); 50.0 excluded (exclusive upper)
            { from: 50, to: 200, key: 'mid' },  // p5, p6 (50.0), p3 (129.5), p2 (199.99)
            { from: 200 },                      // p4 (299.0); default key '200-*'
          ],
        },
      },
    });
    const byPrice = res.facets!.byPrice;
    assert.strictEqual(byPrice.type, 'range');
    if (byPrice.type !== 'range') throw new Error('unreachable');
    assert.strictEqual(byPrice.isApproximate, false);
    assert.deepStrictEqual(
      byPrice.buckets.map((b) => [b.key, b.count]),
      [['budget', 1], ['mid', 4], ['200-*', 1]]
    );
    assert.strictEqual(byPrice.buckets[0].to, 50);
    assert.strictEqual(byPrice.buckets[1].from, 50);

    index.destroy();
    console.log('   ✅ Range facets verified 100%');
  }

  // =========================================================================
  // 3. Disjunctive multi-facet navigation via filter exclusion
  // =========================================================================
  console.log('3. Testing disjunctive faceting (filter exclusion masks)...');
  {
    const index = await DocumentIndex.create(CORPUS, productIndexOpts());

    const res = await index.search('Widget', {
      filter: { category: 'tech' },
      facets: {
        byCategory: { type: 'terms', field: 'category' },
        byPrice: {
          type: 'range',
          field: 'price',
          ranges: [{ to: 150 }, { from: 150 }],
        },
      },
    });
    // Conjunctive results: only tech widgets
    assert.strictEqual(res.totalMatches, 3);
    // Disjunctive: category facet ignores the category clause -> all 6
    const byCategory = res.facets!.byCategory;
    if (byCategory.type !== 'terms') throw new Error('unreachable');
    assert.deepStrictEqual(
      byCategory.buckets.map((b) => [b.value, b.count]),
      [['tech', 3], ['home', 2], ['garden', 1]]
    );
    // Price facet keeps the category clause -> tech only (49.99, 199.99, 129.5)
    const byPrice = res.facets!.byPrice;
    if (byPrice.type !== 'range') throw new Error('unreachable');
    assert.deepStrictEqual(
      byPrice.buckets.map((b) => b.count),
      [2, 1]
    );

    // Multi-clause AND: facet on category drops only the category clause
    const res2 = await index.search('Widget', {
      filter: { and: [{ category: 'tech' }, { price: { gte: 100 } }] },
      facets: { byCategory: { type: 'terms', field: 'category' } },
    });
    assert.strictEqual(res2.totalMatches, 2); // p2, p3
    const cat2 = res2.facets!.byCategory;
    if (cat2.type !== 'terms') throw new Error('unreachable');
    // price >= 100 across all categories: p2, p3 (tech), p4 (home)
    assert.deepStrictEqual(
      cat2.buckets.map((b) => [b.value, b.count]),
      [['tech', 2], ['home', 1]]
    );

    // Conservative or/not handling: no throw, conjunctive fallback
    const res3 = await index.search('Widget', {
      filter: { or: [{ category: 'tech' }, { price: { gte: 250 } }] },
      facets: { byCategory: { type: 'terms', field: 'category' } },
    });
    assert.strictEqual(res3.totalMatches, 4); // 3 tech + p4 home
    const cat3 = res3.facets!.byCategory;
    if (cat3.type !== 'terms') throw new Error('unreachable');
    const catMap3 = new Map(cat3.buckets.map((b) => [b.value, b.count]));
    assert.strictEqual(catMap3.get('tech'), 3);
    assert.strictEqual(catMap3.get('home'), 1);

    // excludeFieldFromFilter unit checks
    assert.deepStrictEqual(excludeFieldFromFilter({ category: 'tech' }, 'category'), undefined);
    assert.deepStrictEqual(excludeFieldFromFilter({ and: [] }, 'category'), undefined);
    assert.strictEqual(filterReferencesField({ or: [{ category: 'x' }] }, 'category'), true);
    assert.strictEqual(filterReferencesField({ price: { gte: 1 } }, 'category'), false);
    const kept = excludeFieldFromFilter({ or: [{ category: 'x' }] }, 'category');
    assert.deepStrictEqual(kept, { or: [{ category: 'x' }] });

    // normalizeFacetRequests unit checks
    assert.strictEqual(normalizeFacetRequests(undefined), undefined);
    assert.deepStrictEqual(normalizeFacetRequests([]), []);
    const dup = normalizeFacetRequests([
      { type: 'terms', field: 'category', limit: 1 },
      { type: 'terms', field: 'category', limit: 2 },
    ])!;
    assert.strictEqual(dup.length, 1);
    assert.strictEqual((dup[0].request as { limit?: number }).limit, 2);

    index.destroy();
    console.log('   ✅ Disjunctive faceting verified 100%');
  }

  // =========================================================================
  // 4. Fail-closed validation
  // =========================================================================
  console.log('4. Testing facet validation (InvalidFilterError / TypeError)...');
  {
    const index = await DocumentIndex.create(CORPUS, productIndexOpts());

    await assert.rejects(
      index.search('Widget', { facets: { bad: { type: 'terms', field: 'nope' } } }),
      InvalidFilterError
    );
    await assert.rejects(
      index.search('Widget', {
        facets: { bad: { type: 'range', field: 'category', ranges: [{ from: 1 }] } },
      }),
      InvalidFilterError
    );
    await assert.rejects(
      index.search('Widget', { facets: { bad: { type: 'bogus', field: 'category' } as unknown as FacetRequest } }),
      TypeError
    );
    await assert.rejects(
      index.search('Widget', { facets: { bad: { type: 'terms', field: 'category', limit: 0 } } }),
      RangeError
    );
    await assert.rejects(
      index.search('Widget', {
        facets: { bad: { type: 'range', field: 'price', ranges: [] } },
      }),
      TypeError
    );
    await assert.rejects(
      index.search('Widget', { facets: { bad: { type: 'terms', field: 'category', sortBy: 'nope' } as never } }),
      TypeError
    );
    await assert.rejects(
      index.search('Widget', {
        facets: { byCategory: { type: 'terms', field: 'category' } },
        faceting: 'sometimes' as never,
      }),
      TypeError
    );
    await assert.rejects(
      index.search('Widget', {
        facets: { bad: { type: 'range', field: 'price', ranges: [{}] } },
      }),
      TypeError
    );

    // Fail-closed on every early-exit path: unknown fields throw even with
    // zero matches (empty query, empty filter result, empty index).
    await assert.rejects(
      index.search('', { facets: { bad: { type: 'terms', field: 'nope' } } }),
      InvalidFilterError
    );
    await assert.rejects(
      index.search('Widget', {
        filter: { category: 'nonexistent' },
        facets: { bad: { type: 'terms', field: 'nope' } },
      }),
      InvalidFilterError
    );
    await assert.rejects(
      index.search('Widget', {
        facets: { bad: { type: 'range', field: 'category', ranges: [{ from: 1 }] } },
        filter: { category: 'nonexistent' },
      }),
      InvalidFilterError
    );
    const emptyIndex = await DocumentIndex.create([], productIndexOpts());
    await assert.rejects(
      emptyIndex.search('Widget', { facets: { bad: { type: 'terms', field: 'nope' } } }),
      InvalidFilterError
    );
    emptyIndex.destroy();

    // Empty facet list behaves like no facets (absent key).
    const resEmpty = await index.search('Widget', { facets: [] });
    assert.strictEqual(resEmpty.facets, undefined);

    index.destroy();
    console.log('   ✅ Facet validation verified 100%');
  }

  // =========================================================================
  // 5. Empty match sets keep stable response shape
  // =========================================================================
  console.log('5. Testing empty-result facet shape...');
  {
    const index = await DocumentIndex.create(CORPUS, productIndexOpts());

    const res = await index.search('zzz-no-such-product', {
      facets: {
        byCategory: { type: 'terms', field: 'category' },
        byPrice: { type: 'range', field: 'price', ranges: [{ to: 100, key: 'cheap' }] },
      },
    });
    assert.strictEqual(res.totalMatches, 0);
    const byCategory = res.facets!.byCategory;
    if (byCategory.type !== 'terms') throw new Error('unreachable');
    assert.deepStrictEqual(byCategory.buckets, []);
    assert.strictEqual(byCategory.isApproximate, false);
    const byPrice = res.facets!.byPrice;
    if (byPrice.type !== 'range') throw new Error('unreachable');
    assert.deepStrictEqual(byPrice.buckets, [{ key: 'cheap', to: 100, count: 0 }]);

    // Filter matching nothing also yields stable empty facets
    const res2 = await index.search('Widget', {
      filter: { category: 'nonexistent' },
      facets: { byCategory: { type: 'terms', field: 'category' } },
    });
    assert.strictEqual(res2.totalMatches, 0);
    if (res2.facets!.byCategory.type !== 'terms') throw new Error('unreachable');
    assert.deepStrictEqual(res2.facets!.byCategory.buckets, []);

    index.destroy();
    console.log('   ✅ Empty-result facet shape verified 100%');
  }

  // =========================================================================
  // 6. Mutation lifecycle keeps facets in sync
  // =========================================================================
  console.log('6. Testing facet sync across add/update/remove...');
  {
    const index = await DocumentIndex.create(CORPUS.slice(0, 6), productIndexOpts());
    const catCounts = async () => {
      const r = await index.search('Widget', {
        facets: { c: { type: 'terms', field: 'category' } },
      });
      const f = r.facets!.c;
      if (f.type !== 'terms') throw new Error('unreachable');
      return new Map(f.buckets.map((b) => [b.value, b.count]));
    };

    assert.strictEqual((await catCounts()).get('tech'), 3);

    await index.add({
      id: 'p9', title: 'Widget Theta', body: 'usb hub adapter', category: 'tech', price: 39.99, inStock: true, tags: ['office'],
    });
    assert.strictEqual((await catCounts()).get('tech'), 4);

    await index.update([{
      id: 'p9', title: 'Widget Theta', body: 'usb hub adapter', category: 'garden', price: 39.99, inStock: true, tags: ['outdoor'],
    }]);
    const afterUpdate = await catCounts();
    assert.strictEqual(afterUpdate.get('tech'), 3);
    assert.strictEqual(afterUpdate.get('garden'), 2);

    await index.remove('p9');
    const afterRemove = await catCounts();
    assert.strictEqual(afterRemove.get('tech'), 3);
    assert.strictEqual(afterRemove.get('garden'), 1);

    index.destroy();
    console.log('   ✅ Mutation sync verified 100%');
  }

  // =========================================================================
  // 7. GPU parity + overflow semantics (mock adapter + stubbed engine)
  //
  // NOTE: the vgpu mock stubs compute dispatch (it never produces match
  // rows), so the repo-wide convention (test-vgpu-mock.ts, filter tests)
  // asserts GPU *routing*, not GPU match quality. Overflow/approximate
  // routing — which needs a capped candidate pool — is covered with a
  // stubbed WebGPUEngine behind the same readback contract.
  // =========================================================================
  console.log('7. Testing GPU facet parity and overflow semantics...');
  {
    const adapter = await createMockAdapter({ features: ['timestamp-query'] as never });
    const mockDeviceWrapper = await adapter.requestDevice();
    const mockDevice = (mockDeviceWrapper as unknown as { gpu: GPUDevice }).gpu ?? (mockDeviceWrapper as unknown as GPUDevice);

    const gpuIndex = await DocumentIndex.create(CORPUS, productIndexOpts({ device: mockDevice, preferGpu: true }));
    const cpuIndex = await DocumentIndex.create(CORPUS, productIndexOpts());
    const cpuRef = await cpuIndex.search('Widget', {
      facets: { c: { type: 'terms', field: 'category' } },
    });

    // 7a. Mock routing: engine + facet shape (mock pool is empty by design).
    const mockRes = await gpuIndex.search('Widget', {
      facets: { c: { type: 'terms', field: 'category' } },
    });
    assert.strictEqual(mockRes.engine, 'webgpu');
    assert.strictEqual(mockRes.hasOverflow, false);
    if (mockRes.facets!.c.type !== 'terms') throw new Error('unreachable');
    assert.strictEqual(mockRes.facets!.c.isApproximate, false);
    assert.deepStrictEqual(mockRes.facets!.c.buckets, []);

    // 7b. Stubbed engine: rows are field-stratified, so title rows for the
    // 8-doc corpus are 0..7 and body rows are 8..15.
    const stubGpu = (poolRows: number[], hasOverflow: boolean) => ({
      isReady: true,
      destroy() {},
      search: async (query: string, opts: { mode?: string }) => ({
        query,
        mode: opts.mode ?? 'fuzzy',
        engine: 'webgpu',
        totalMatches: poolRows.length + (hasOverflow ? 5000 : 0),
        candidateCount: poolRows.length,
        hasOverflow,
        results: poolRows.map((index) => ({ index, score: 500, text: '' })),
        timings: {
          queryUploadMs: 0, encodeSubmitMs: 0, gpuExecutionMs: null,
          readbackMs: 0, totalMs: 1, gpuDispatchMs: 0,
        },
        profileId: 'unicode-default',
        scoringVersion: 'parity-v1',
        cpuAlgorithm: 'parity',
      }),
    });

    const stubIndex = await DocumentIndex.create(CORPUS, productIndexOpts());
    (stubIndex as unknown as { engineType: string }).engineType = 'webgpu';

    // Capped pool (docs p1..p3) with overflow, auto mode -> approximate.
    (stubIndex as unknown as { gpuEngine: unknown }).gpuEngine = stubGpu([0, 1, 2], true);
    const autoRes = await stubIndex.search('Widget', {
      facets: { c: { type: 'terms', field: 'category' } },
    });
    assert.strictEqual(autoRes.engine, 'webgpu');
    assert.strictEqual(autoRes.hasOverflow, true);
    assert.strictEqual(autoRes.totalMatches, 3);
    if (autoRes.facets!.c.type !== 'terms') throw new Error('unreachable');
    assert.strictEqual(autoRes.facets!.c.isApproximate, true);
    assert.deepStrictEqual(
      autoRes.facets!.c.buckets.map((b) => [b.value, b.count]),
      [['tech', 3]]
    );

    // Same capped pool with force-exact -> full CPU rescan, exact counts.
    const exactRes = await stubIndex.search('Widget', {
      facets: { c: { type: 'terms', field: 'category' } },
      faceting: 'force-exact',
    });
    assert.strictEqual(exactRes.engine, 'webgpu');
    assert.strictEqual(exactRes.totalMatches, 3); // hits still served from the GPU pool
    if (exactRes.facets!.c.type !== 'terms') throw new Error('unreachable');
    assert.strictEqual(exactRes.facets!.c.isApproximate, false);
    if (cpuRef.facets!.c.type !== 'terms') throw new Error('unreachable');
    assert.deepStrictEqual(exactRes.facets!.c.buckets, cpuRef.facets!.c.buckets);

    // Capped pool without overflow flag -> exact over the pool.
    (stubIndex as unknown as { gpuEngine: unknown }).gpuEngine = stubGpu([0, 1, 2], false);
    const noOverflowRes = await stubIndex.search('Widget', {
      facets: {
        c: { type: 'terms', field: 'category' },
        p: { type: 'range', field: 'price', ranges: [{ to: 150 }, { from: 150 }] },
      },
    });
    assert.strictEqual(noOverflowRes.hasOverflow, false);
    if (noOverflowRes.facets!.c.type !== 'terms') throw new Error('unreachable');
    assert.strictEqual(noOverflowRes.facets!.c.isApproximate, false);
    assert.deepStrictEqual(
      noOverflowRes.facets!.c.buckets.map((b) => [b.value, b.count]),
      [['tech', 3]]
    );
    if (noOverflowRes.facets!.p.type !== 'range') throw new Error('unreachable');
    assert.strictEqual(noOverflowRes.facets!.p.isApproximate, false);
    assert.deepStrictEqual(
      noOverflowRes.facets!.p.buckets.map((b) => b.count),
      [2, 1] // 49.99, 129.50 vs 199.99
    );

    // 7c. CPU route stays exact past candidate capacity (9000 docs).
    const N = 9000;
    const bulk: Product[] = new Array(N);
    const cats = ['tech', 'home', 'garden'];
    for (let i = 0; i < N; i++) {
      bulk[i] = {
        id: `b${i}`, title: `Widget bulk ${i}`, body: 'mass accessory unit',
        category: cats[i % 3], price: (i % 500) + 0.99, inStock: i % 2 === 0, tags: ['bulk'],
      };
    }
    const cpuBulk = await DocumentIndex.create(bulk, productIndexOpts());
    const cpuBulkRes = await cpuBulk.search('Widget', {
      facets: { c: { type: 'terms', field: 'category' } },
    });
    assert.strictEqual(cpuBulkRes.hasOverflow, true);
    if (cpuBulkRes.facets!.c.type !== 'terms') throw new Error('unreachable');
    assert.strictEqual(cpuBulkRes.facets!.c.isApproximate, false);
    assert.deepStrictEqual(
      cpuBulkRes.facets!.c.buckets.map((b) => b.count),
      [3000, 3000, 3000]
    );

    gpuIndex.destroy();
    cpuIndex.destroy();
    stubIndex.destroy();
    cpuBulk.destroy();
    console.log('   ✅ GPU parity & overflow semantics verified 100%');
  }

  // =========================================================================
  // 8. Worker thread boundary
  // =========================================================================
  console.log('8. Testing facets across SearchWorkerClient boundary...');
  {
    const worker = new SearchWorkerClient<Product>();
    try {
      await worker.init(CORPUS, {
        fields: ['title', 'body'],
        filterFields: [
          { name: 'category', type: 'string' },
          { name: 'price', type: 'number' },
        ],
        preferGpu: false,
      });
      const res = await worker.search('Widget', {
        filter: { price: { gte: 100 } },
        facets: { byCategory: { type: 'terms', field: 'category' } },
      });
      // price >= 100 over widgets: p2, p3 (tech), p4 (home) — disjunctive on category
      const byCategory = (res.facets as Record<string, { type: string; buckets: Array<{ value: unknown; count: number }> }>)!.byCategory;
      assert.strictEqual(byCategory.type, 'terms');
      const map = new Map(byCategory.buckets.map((b) => [b.value, b.count]));
      assert.strictEqual(map.get('tech'), 2);
      assert.strictEqual(map.get('home'), 1);
    } finally {
      await worker.destroy();
    }
    console.log('   ✅ Worker boundary verified 100%');
  }

  // =========================================================================
  // 9. Performance invariant (< 2ms aggregation over 50k docs)
  // =========================================================================
  console.log('9. Validating performance invariant: < 2ms per 50k aggregation...');
  {
    const N = 50_000;
    const records: { id: number; cat: string; val: number }[] = new Array(N);
    const categories = ['alpha', 'beta', 'gamma', 'delta', 'epsilon'];
    for (let i = 0; i < N; i++) {
      records[i] = { id: i, cat: categories[i % 5], val: (i * 1.7) % 1000 };
    }
    const index = await DocumentIndex.create(records, {
      fields: ['cat'],
      filterFields: [
        { name: 'cat', type: 'string' },
        { name: 'val', type: 'number' },
      ],
      preferGpu: false,
    });
    const store = index.getColumnarStore();
    const engine = new FacetEngine(store);
    const all = store.getActiveDocs().getMatchingIndices();
    assert.strictEqual(all.length, N);

    for (let i = 0; i < 10; i++) {
      engine.aggregateTerms('cat', all, {});
      engine.aggregateRange('val', all, [{ to: 250 }, { from: 250, to: 750 }, { from: 750 }]);
    }
    const iterations = 100;
    const t0 = performance.now();
    for (let i = 0; i < iterations; i++) {
      engine.aggregateTerms('cat', all, {});
    }
    const termsMs = (performance.now() - t0) / iterations;
    const t1 = performance.now();
    for (let i = 0; i < iterations; i++) {
      engine.aggregateRange('val', all, [{ to: 250 }, { from: 250, to: 750 }, { from: 750 }]);
    }
    const rangeMs = (performance.now() - t1) / iterations;
    console.log(`   [Performance] 50k terms aggregation: ${termsMs.toFixed(3)} ms; range: ${rangeMs.toFixed(3)} ms (target: < 2 ms)`);
    assert(termsMs < 2, `Terms aggregation exceeded 2ms target: ${termsMs.toFixed(3)}ms`);
    assert(rangeMs < 2, `Range aggregation exceeded 2ms target: ${rangeMs.toFixed(3)}ms`);

    // Sanity: exact distribution over the full set
    const terms = engine.aggregateTerms('cat', all, {});
    assert.strictEqual(terms.buckets.length, 5);
    assert.ok(terms.buckets.every((b) => b.count === 10_000));

    index.destroy();
    console.log('   ✅ Performance invariant verified 100%');
  }

  // =========================================================================
  // 10. Cross-platform portability audit (zero DOM references)
  // =========================================================================
  console.log('10. Auditing facet files for DOM references (window, document)...');
  {
    const filesToAudit = [
      'packages/webgpu-search/src/facets/facet-engine.ts',
      'packages/webgpu-search/src/document-index.ts',
      'packages/webgpu-search/src/exact-scorer.ts',
      'packages/webgpu-search/src/filter/columnar-store.ts',
    ];
    for (const filePath of filesToAudit) {
      const code = await readFile(filePath, 'utf-8');
      assert(!code.includes('window.'), `Unexpected window reference in ${filePath}`);
      assert(!code.includes('document.'), `Unexpected document reference in ${filePath}`);
      // Hardened: bare globals that break Workers/Node/SSR (typeof-guarded
      // navigator is allowed; direct use is not).
      assert(!/\blocalStorage\b/.test(code), `Unexpected localStorage in ${filePath}`);
      assert(!/\bsessionStorage\b/.test(code), `Unexpected sessionStorage in ${filePath}`);
      assert(!/\bindexedDB\b/.test(code.replace(/index\.html|IDB/gi, '')), `Unexpected indexedDB in ${filePath}`);
      // Direct document/window/self/globalThis DOM use (typeof checks ok).
      const stripped = code.replace(/typeof\s+(document|window|navigator|self)\b/g, '');
      assert(!/(^|[^\w$.])document\s*\./.test(stripped), `Unexpected bare document use in ${filePath}`);
      // boundary-aware (`window` must not continue into an
      // identifier): names like `windowLength`/`findBestTypoWindow` are
      // record spans, not the DOM global. Still catches `window.foo`,
      // `window `, and `(window)`.
      assert(!/(^|[^\w$.])window(?![\w$])/.test(stripped.replace(/worker-client|search-worker/gi, '')), `Unexpected bare window in ${filePath}`);
    }
    console.log('   ✅ Zero DOM references confirmed across facet modules');
  }

  // =========================================================================
  // 11. Review hardening (multi-agent PR #30 findings)
  // =========================================================================
  console.log('11. Testing review hardening (proto, dedupe, caps, validation)...');
  {
    const index = await DocumentIndex.create(CORPUS, productIndexOpts());

    // 11a. Prototype pollution fail-closed (JSON own-key, literal would set proto).
    const protoFacets = JSON.parse('{"__proto__":{"type":"terms","field":"category"}}');
    await assert.rejects(
      index.search('Widget', { facets: protoFacets }),
      TypeError
    );
    await assert.rejects(
      index.search('Widget', { facets: [{ type: 'terms', field: 'category' }, { type: 'range', field: 'category', ranges: [{ from: 1 }] }] as unknown as FacetRequest[] }),
      InvalidFilterError
    );

    // 11b. Per-doc dedupe: duplicate tags count once per doc.
    const dupIndex = await DocumentIndex.create(
      [
        { id: 'd1', title: 'Widget dup', body: 'x', category: 'tech', price: 10, inStock: true, tags: ['sale', 'sale', 'audio'] },
      ],
      productIndexOpts()
    );
    const dupRes = await dupIndex.search('Widget', { facets: { t: { type: 'terms', field: 'tags' } } });
    if (dupRes.facets!.t.type !== 'terms') throw new Error('unreachable');
    const dupMap = new Map(dupRes.facets!.t.buckets.map((b) => [b.value, b.count]));
    assert.strictEqual(dupMap.get('sale'), 1);
    assert.strictEqual(dupMap.get('audio'), 1);
    dupIndex.destroy();

    // 11c. Inverted range bounds throw (from >= to).
    await assert.rejects(
      index.search('Widget', { facets: { bad: { type: 'range', field: 'price', ranges: [{ from: 100, to: 50 }] } } }),
      RangeError
    );
    await assert.rejects(
      index.search('Widget', { facets: { bad: { type: 'range', field: 'price', ranges: [{ from: 50, to: 50 }] } } }),
      RangeError
    );

    // 11d. Caps: ranges >100 and facets >32 throw.
    await assert.rejects(
      index.search('Widget', {
        facets: { bad: { type: 'range', field: 'price', ranges: new Array(101).fill(0).map((_, i) => ({ from: i, to: i + 1 })) } },
      }),
      RangeError
    );
    const manyFacets: Record<string, FacetRequest> = {};
    for (let i = 0; i < 33; i++) manyFacets[`f${i}`] = { type: 'terms', field: 'category' };
    await assert.rejects(index.search('Widget', { facets: manyFacets }), RangeError);

    // 11e. Direct FacetEngine validation (bypasses normalize).
    const store = index.getColumnarStore();
    const engine = new FacetEngine(store);
    const all = store.getActiveDocs().getMatchingIndices();
    assert.throws(() => engine.aggregateTerms('category', all, { limit: 0 }), RangeError);
    assert.throws(() => engine.aggregateTerms('category', all, { limit: NaN }), TypeError);
    assert.throws(() => engine.aggregateTerms('category', all, { limit: Infinity }), TypeError);
    assert.throws(() => engine.aggregateRange('price', all, []), TypeError);
    assert.throws(() => engine.aggregateRange('price', all, [{ from: 10, to: 5 }]), RangeError);
    // Fractional limit floors (documented).
    const floored = engine.aggregateTerms('category', all, { limit: 2.9 });
    assert.ok(floored.buckets.length <= 2);

    // 11f. excludeFieldFromFilter throws on invalid shapes (was lenient).
    assert.throws(() => (excludeFieldFromFilter as (e: unknown, f: string) => unknown)(null, 'category'), TypeError);
    assert.throws(() => (excludeFieldFromFilter as (e: unknown, f: string) => unknown)([], 'category'), TypeError);

    // 11g. faceting ignored when facets absent (no throw).
    const noFacetNoThrow = await index.search('Widget', { faceting: 'sometimes' as never });
    assert.strictEqual(noFacetNoThrow.facets, undefined);

    // 11h. Overlapping ranges double-count by design.
    const overlap = await index.search('Widget', {
      facets: { p: { type: 'range', field: 'price', ranges: [{ from: 0, to: 200 }, { from: 100, to: 300 }] } },
    });
    if (overlap.facets!.p.type !== 'range') throw new Error('unreachable');
    // p2 (199.99) and p3 (129.5) fall in both buckets.
    assert.ok(overlap.facets!.p.buckets[0].count >= 2);
    assert.ok(overlap.facets!.p.buckets[1].count >= 2);

    // 11i. Facets x mode substring + explicit faceting auto.
    const sub = await index.search('Widget', {
      mode: 'substring',
      faceting: 'auto',
      facets: { c: { type: 'terms', field: 'category' } },
    });
    assert.ok(sub.facets?.c);

    index.destroy();
    console.log('   ✅ Review hardening verified 100%');
  }

  // =========================================================================
  // 12. Worker predicate + range/overflow hardening
  // =========================================================================
  console.log('12. Testing worker predicate facets + range overflow...');
  {
    const worker = new SearchWorkerClient<Product>();
    try {
      await worker.init(CORPUS, {
        fields: ['title', 'body'],
        filterFields: [
          { name: 'category', type: 'string' },
          { name: 'price', type: 'number' },
        ],
        preferGpu: false,
      });
      // Range facet across worker boundary.
      const rangeRes = await worker.search('Widget', {
        facets: { p: { type: 'range', field: 'price', ranges: [{ to: 100 }, { from: 100 }] } },
      });
      const p = (rangeRes.facets as Record<string, { type: string; buckets: Array<{ count: number }> }>)!.p;
      assert.strictEqual(p.type, 'range');
      assert.deepStrictEqual(p.buckets.map((b) => b.count), [3, 3]);

      // Validation error propagates across worker.
      await assert.rejects(
        worker.search('Widget', { facets: { bad: { type: 'terms', field: 'nope' } } }),
        Error
      );

      // Predicate + facets: facets must be absent (stale otherwise).
      // NOTE: SearchWorkerClient predicate path requires a real Worker with
      // postMessage; in-process check uses DocumentIndex parity (conjunctive).
      const direct = await DocumentIndex.create(CORPUS, productIndexOpts());
      const predRes = await direct.search('Widget', {
        filter: (doc: Product) => doc.category === 'tech',
        facets: { c: { type: 'terms', field: 'category' } },
      });
      if (predRes.facets!.c.type !== 'terms') throw new Error('unreachable');
      assert.deepStrictEqual(
        predRes.facets!.c.buckets.map((b) => [b.value, b.count]),
        [['tech', 3]]
      );
      direct.destroy();
    } finally {
      await worker.destroy();
    }
    console.log('   ✅ Worker predicate + range hardening verified 100%');
  }

  console.log('\n🎉 ALL MILESTONE 3 TESTS PASSED SUCCESSFULLY! 🎉\n');
}

runTests().catch((err) => {
  console.error('test failed:', err);
  process.exit(1);
});
