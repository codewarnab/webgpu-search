/**
 * Test Suite: Columnar Attribute Index & Structured Pre-Filtering
 * Tests DocumentBitset, ColumnarStore, compileFilter, DocumentIndex pre-filtering,
 * mutations sync, SearchWorkerClient integration, and sub-50µs performance.
 */

import assert from 'node:assert';
import { readFile } from 'node:fs/promises';
import { createMockAdapter } from 'vgpu/mock';
import {
  DocumentBitset,
  ColumnarStore,
  compileFilter,
  DocumentIndex,
  SearchWorkerClient,
  InvalidFilterError,
  type FilterExpression,
  type DocumentIndexOptions
} from '../packages/webgpu-search/src/index';

async function runTests() {
  console.log('=== Running Issue #10  Columnar Attribute Index & Pre-Filtering ===\n');

  // =========================================================================
  // 1. DocumentBitset Unit Verification
  // =========================================================================
  console.log('1. Testing DocumentBitset bitwise operations & capacity...');
  {
    const bs = new DocumentBitset(64);
    assert.strictEqual(bs.capacity, 64);
    assert.strictEqual(bs.words.length, 2);
    assert.strictEqual(bs.popcount(), 0);
    assert.strictEqual(bs.has(0), false);
    assert.strictEqual(bs.has(31), false);
    assert.strictEqual(bs.has(63), false);

    bs.set(0);
    bs.set(31);
    bs.set(32);
    bs.set(63);
    assert.strictEqual(bs.has(0), true);
    assert.strictEqual(bs.has(31), true);
    assert.strictEqual(bs.has(32), true);
    assert.strictEqual(bs.has(63), true);
    assert.strictEqual(bs.popcount(), 4);
    assert.deepStrictEqual(bs.getMatchingIndices(), [0, 31, 32, 63]);

    // Dynamic geometric growth
    bs.set(100);
    assert.strictEqual(bs.has(100), true);
    assert(bs.capacity >= 101);
    assert.strictEqual(bs.popcount(), 5);

    // clear & toggle
    bs.clear(31);
    assert.strictEqual(bs.has(31), false);
    assert.strictEqual(bs.popcount(), 4);

    bs.toggle(31);
    assert.strictEqual(bs.has(31), true);
    assert.strictEqual(bs.popcount(), 5);

    bs.toggle(31);
    assert.strictEqual(bs.has(31), false);
    assert.strictEqual(bs.popcount(), 4);

    // Bitwise combinators
    const b1 = DocumentBitset.fromIndices([1, 2, 3, 4], 10);
    const b2 = DocumentBitset.fromIndices([3, 4, 5, 6], 10);

    const andRes = b1.and(b2);
    assert.deepStrictEqual(andRes.getMatchingIndices(), [3, 4]);

    const orRes = b1.or(b2);
    assert.deepStrictEqual(orRes.getMatchingIndices(), [1, 2, 3, 4, 5, 6]);

    const andNotRes = b1.andNot(b2);
    assert.deepStrictEqual(andNotRes.getMatchingIndices(), [1, 2]);

    const allBs = DocumentBitset.all(10);
    assert.strictEqual(allBs.popcount(), 10);
    const noneBs = DocumentBitset.none(10);
    assert.strictEqual(noneBs.popcount(), 0);

    console.log('   ✅ DocumentBitset unit specifications verified 100%');
  }

  // =========================================================================
  // 2. ColumnarStore Schema & Mutation Verification
  // =========================================================================
  console.log('2. Testing ColumnarStore typed columns & mutations...');
  {
    interface TestItem {
      id: string;
      category: string;
      price: number;
      inStock: boolean;
      tags: string[];
    }

    const items: TestItem[] = [
      { id: '1', category: 'electronics', price: 299.99, inStock: true, tags: ['sale', 'popular'] },
      { id: '2', category: 'books', price: 19.99, inStock: false, tags: ['bestseller'] },
      { id: '3', category: 'electronics', price: 99.99, inStock: true, tags: ['clearance'] }
    ];

    const store = new ColumnarStore<TestItem>([
      { name: 'category', type: 'string' },
      { name: 'price', type: 'number' },
      { name: 'inStock', type: 'boolean' },
      { name: 'tags', type: 'string[]' }
    ]);

    store.init(items);

    // Equality
    const elec = store.evaluateEquality('category', 'electronics');
    assert.deepStrictEqual(elec.getMatchingIndices(), [0, 2]);

    // Range
    const cheap = store.evaluateRange('price', { lt: 100 });
    assert.deepStrictEqual(cheap.getMatchingIndices(), [1, 2]);

    // Boolean
    const stock = store.evaluateEquality('inStock', true);
    assert.deepStrictEqual(stock.getMatchingIndices(), [0, 2]);

    // Tag set
    const sale = store.evaluateEquality('tags', 'sale');
    assert.deepStrictEqual(sale.getMatchingIndices(), [0]);

    // Mutations: add item 4
    store.add(3, {
      id: '4',
      category: 'books',
      price: 14.99,
      inStock: true,
      tags: ['sale']
    });

    assert.deepStrictEqual(
      store.evaluateEquality('category', 'books').getMatchingIndices(),
      [1, 3]
    );
    assert.deepStrictEqual(
      store.evaluateEquality('tags', 'sale').getMatchingIndices(),
      [0, 3]
    );

    // Mutation: update item 1 category to 'home'
    store.update(0, {
      id: '1',
      category: 'home',
      price: 299.99,
      inStock: true,
      tags: ['clearance']
    });

    assert.deepStrictEqual(
      store.evaluateEquality('category', 'electronics').getMatchingIndices(),
      [2]
    );
    assert.deepStrictEqual(
      store.evaluateEquality('category', 'home').getMatchingIndices(),
      [0]
    );
    assert.deepStrictEqual(
      store.evaluateEquality('tags', 'sale').getMatchingIndices(),
      [3]
    );

    // Mutation: remove item 4
    store.remove(3);
    assert.deepStrictEqual(
      store.evaluateEquality('category', 'books').getMatchingIndices(),
      [1]
    );

    console.log('   ✅ ColumnarStore typed columns & mutations verified 100%');
  }

  // =========================================================================
  // 3. compileFilter AST Evaluator Verification
  // =========================================================================
  console.log('3. Testing compileFilter AST combinators & validation...');
  {
    interface Record {
      dept: string;
      level: number;
      active: boolean;
      skills: string[];
    }

    const data: Record[] = [
      { dept: 'eng', level: 3, active: true, skills: ['ts', 'gpu'] },
      { dept: 'eng', level: 5, active: true, skills: ['wgsl', 'rust'] },
      { dept: 'design', level: 2, active: false, skills: ['figma'] },
      { dept: 'product', level: 4, active: true, skills: ['roadmap', 'agile'] }
    ];

    const store = new ColumnarStore<Record>([
      { name: 'dept', type: 'string' },
      { name: 'level', type: 'number' },
      { name: 'active', type: 'boolean' },
      { name: 'skills', type: 'string[]' }
    ]);
    store.init(data);

    // Implicit AND
    const r1 = compileFilter({ dept: 'eng', active: true }, store);
    assert.deepStrictEqual(r1.getMatchingIndices(), [0, 1]);

    // Explicit AND with range
    const r2 = compileFilter({
      and: [
        { dept: 'eng' },
        { level: { gte: 4 } }
      ]
    }, store);
    assert.deepStrictEqual(r2.getMatchingIndices(), [1]);

    // Explicit OR
    const r3 = compileFilter({
      or: [
        { dept: 'design' },
        { skills: { in: ['gpu', 'roadmap'] } }
      ]
    }, store);
    assert.deepStrictEqual(r3.getMatchingIndices(), [0, 2, 3]);

    // NOT combinator
    const r4 = compileFilter({
      not: { active: true }
    }, store);
    assert.deepStrictEqual(r4.getMatchingIndices(), [2]);

    // Vacuous truth and falsehood
    const rTruth = compileFilter({ and: [] }, store);
    assert.strictEqual(rTruth.popcount(), 4);
    const rFalsehood = compileFilter({ or: [] }, store);
    assert.strictEqual(rFalsehood.popcount(), 0);

    // InvalidFilterError throws on unindexed field
    assert.throws(
      () => compileFilter({ nonexistentField: 'test' }, store),
      InvalidFilterError
    );

    // InvalidFilterError throws on invalid operator
    assert.throws(
      () => compileFilter({ level: { unknownOp: 10 } as any }, store),
      InvalidFilterError
    );

    console.log('   ✅ compileFilter AST Evaluator verified 100%');
  }

  // =========================================================================
  // 4. DocumentIndex Integration & Parity Verification
  // =========================================================================
  console.log('4. Testing DocumentIndex structured pre-filtering integration...');
  {
    interface Article {
      id: string;
      title: string;
      body: string;
      category: string;
      views: number;
      published: boolean;
    }

    const corpus: Article[] = [
      { id: '1', title: 'WebGPU Compute Shaders', body: 'Parallel processing with WGSL', category: 'tech', views: 1500, published: true },
      { id: '2', title: 'TypeScript Best Practices', body: 'Typing large scale applications', category: 'tech', views: 800, published: true },
      { id: '3', title: 'Graphic Design Basics', body: 'Color theory and typography', category: 'design', views: 300, published: false },
      { id: '4', title: 'WebGPU Pipeline Tuning', body: 'Uniform buffers and workgroups', category: 'tech', views: 2200, published: true }
    ];

    const opts: DocumentIndexOptions<Article> = {
      fields: [
        { name: 'title', weight: 2.0 },
        { name: 'body', weight: 1.0 }
      ],
      filterFields: [
        { name: 'category', type: 'string' },
        { name: 'views', type: 'number' },
        { name: 'published', type: 'boolean' }
      ],
      preferGpu: false
    };

    const index = await DocumentIndex.create(corpus, opts);

    // Query 'WebGPU' with filter { views: { gte: 2000 } } -> matches doc '4' only
    const res1 = await index.search('WebGPU', {
      filter: { views: { gte: 2000 } }
    });
    assert.strictEqual(res1.totalMatches, 1);
    assert.strictEqual(res1.results[0].id, '4');

    // Query 'WebGPU' with filter { category: 'design' } -> 0 matches
    const resZero = await index.search('WebGPU', {
      filter: { category: 'design' }
    });
    assert.strictEqual(resZero.totalMatches, 0);

    // Mock WebGPU pipeline validation
    const adapter = await createMockAdapter({
      features: ['timestamp-query'] as any
    });
    const mockDeviceWrapper = await adapter.requestDevice();
    const mockDevice = (mockDeviceWrapper as any).gpu ?? mockDeviceWrapper;

    const gpuIndex = await DocumentIndex.create(corpus, {
      ...opts,
      device: mockDevice,
      preferGpu: true
    });

    const resGpu = await gpuIndex.search('WebGPU', {
      filter: { published: true }
    });
    assert.strictEqual(resGpu.engine, 'webgpu');

    gpuIndex.destroy();
    index.destroy();
    console.log('   ✅ DocumentIndex pre-filtering integration verified 100%');
  }

  // =========================================================================
  // 5. SearchWorkerClient Thread Boundary Verification
  // =========================================================================
  console.log('5. Testing SearchWorkerClient filter evaluation across thread boundary...');
  {
    interface WorkerDoc {
      id: string;
      title: string;
      category: string;
      score: number;
    }

    const workerDocs: WorkerDoc[] = [
      { id: '1', title: 'Apple Red', category: 'fruit', score: 10 },
      { id: '2', title: 'Banana Yellow', category: 'fruit', score: 20 },
      { id: '3', title: 'Carrot Orange', category: 'vegetable', score: 15 }
    ];

    const worker = new SearchWorkerClient<WorkerDoc>();
    try {
      await worker.init(workerDocs, {
        fields: ['title'],
        filterFields: [
          { name: 'category', type: 'string' },
          { name: 'score', type: 'number' }
        ],
        preferGpu: false
      });

      const res = await worker.search('Apple', {
        filter: { category: 'fruit', score: { lte: 15 } }
      });
      assert.strictEqual(res.totalMatches, 1);
      assert.strictEqual(res.results[0].id, '1');
    } finally {
      await worker.destroy();
    }

    console.log('   ✅ SearchWorkerClient thread boundary verified 100%');
  }

  // =========================================================================
  // 6. Performance Invariant (< 50µs for 10k rows)
  // =========================================================================
  console.log('6. Validating performance invariant: < 50µs per 10k rows...');
  {
    const N = 10_000;
    const records: { id: number; cat: string; val: number }[] = new Array(N);
    const categories = ['alpha', 'beta', 'gamma', 'delta'];

    for (let i = 0; i < N; i++) {
      records[i] = {
        id: i,
        cat: categories[i % 4],
        val: i * 1.5
      };
    }

    const store = new ColumnarStore<{ id: number; cat: string; val: number }>([
      { name: 'cat', type: 'string' },
      { name: 'val', type: 'number' }
    ]);
    store.init(records);

    const filter: FilterExpression = {
      and: [
        { cat: { in: ['alpha', 'gamma'] } },
        { val: { gte: 2000, lte: 8000 } }
      ]
    };

    // Warm up JIT (100 iterations)
    for (let i = 0; i < 100; i++) {
      compileFilter(filter, store);
    }

    // Benchmark 1000 iterations: averages out performance.now() (~1ms)
    // granularity noise that made 100-iteration runs flaky (35-68µs spread).
    const iterations = 1000;
    const t0 = performance.now();
    for (let i = 0; i < iterations; i++) {
      compileFilter(filter, store);
    }
    const t1 = performance.now();
    const avgMicros = ((t1 - t0) / iterations) * 1000;

    console.log(`   [Performance] 10k rows filter evaluation: ${avgMicros.toFixed(2)} µs (target: < 50 µs)`);
    assert(
      avgMicros < 50,
      `Filter evaluation exceeded 50µs target: ${avgMicros.toFixed(2)}µs`
    );
    console.log('   ✅ Performance invariant verified 100%');
  }

  // =========================================================================
  // 7. Cross-Platform & Portability Audit (Zero DOM references)
  // =========================================================================
  console.log('7. Auditing filter files for DOM references (window, document)...');
  {
    const filesToAudit = [
      'packages/webgpu-search/src/filter/bitset.ts',
      'packages/webgpu-search/src/filter/columnar-store.ts',
      'packages/webgpu-search/src/filter/filter-evaluator.ts'
    ];

    for (const filePath of filesToAudit) {
      const code = await readFile(filePath, 'utf-8');
      assert(!code.includes('window.'), `Unexpected window reference in ${filePath}`);
      assert(!code.includes('document.'), `Unexpected document reference in ${filePath}`);
    }
    console.log('   ✅ Zero DOM references confirmed across all filter modules');
  }

  console.log('\n🎉 ALL MILESTONE 2 TESTS PASSED SUCCESSFULLY! 🎉\n');
}

runTests().catch((err) => {
  console.error('test failed:', err);
  process.exit(1);
});
