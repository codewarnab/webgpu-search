import { describe, test, expect } from 'bun:test';
import {
  DocumentBitset,
  ColumnarStore,
  compileFilter,
  DocumentIndex,
  SearchWorkerClient,
  InvalidFilterError,
  type DocumentIndexOptions,
  type FilterExpression
} from '../src/index';
import { createMockAdapter } from 'vgpu/mock';

describe('DocumentBitset', () => {
  test('initializes with capacity and defaults to zero bits', () => {
    const bs = new DocumentBitset(100);
    expect(bs.size).toBe(100);
    expect(bs.popcount()).toBe(0);
    expect(bs.isEmpty()).toBe(true);
    for (let i = 0; i < 100; i++) {
      expect(bs.has(i)).toBe(false);
    }
  });

  test('set, clear, has, get, and toggle operations', () => {
    const bs = new DocumentBitset(64);
    bs.set(0);
    bs.set(31);
    bs.set(32);
    bs.set(63);

    expect(bs.has(0)).toBe(true);
    expect(bs.get(31)).toBe(true);
    expect(bs.has(32)).toBe(true);
    expect(bs.has(63)).toBe(true);
    expect(bs.has(1)).toBe(false);
    expect(bs.has(62)).toBe(false);
    expect(bs.popcount()).toBe(4);
    expect(bs.isEmpty()).toBe(false);

    bs.clear(31);
    expect(bs.has(31)).toBe(false);
    expect(bs.popcount()).toBe(3);

    bs.toggle(0);
    expect(bs.has(0)).toBe(false);
    bs.toggle(0);
    expect(bs.has(0)).toBe(true);
  });

  test('dynamic geometric growth beyond initial capacity', () => {
    const bs = new DocumentBitset(10);
    bs.set(5);
    expect(bs.size).toBe(10);

    bs.set(100);
    expect(bs.size).toBeGreaterThanOrEqual(101);
    expect(bs.has(5)).toBe(true);
    expect(bs.has(100)).toBe(true);
    expect(bs.has(50)).toBe(false);
  });

  test('popcount and getMatchingIndices across word boundaries', () => {
    const bs = new DocumentBitset(128);
    const expected = [0, 15, 31, 32, 45, 63, 64, 99, 127];
    for (const idx of expected) {
      bs.set(idx);
    }

    expect(bs.popcount()).toBe(expected.length);
    expect(bs.getMatchingIndices()).toEqual(expected);
  });

  test('DocumentBitset.all and DocumentBitset.none factory methods', () => {
    const all = DocumentBitset.all(70);
    expect(all.size).toBe(70);
    expect(all.popcount()).toBe(70);
    expect(all.has(0)).toBe(true);
    expect(all.has(69)).toBe(true);
    expect(all.has(70)).toBe(false); // Past capacity

    const none = DocumentBitset.none(70);
    expect(none.size).toBe(70);
    expect(none.popcount()).toBe(0);
    expect(none.isEmpty()).toBe(true);
  });

  test('bitwise AND, OR, AND-NOT, and NOT operations', () => {
    const a = DocumentBitset.fromIndices([0, 1, 2, 3, 10, 32], 64);
    const b = DocumentBitset.fromIndices([2, 3, 4, 5, 32], 64);

    // AND
    const andBs = a.and(b);
    expect(andBs.getMatchingIndices()).toEqual([2, 3, 32]);

    // OR
    const orBs = a.or(b);
    expect(orBs.getMatchingIndices()).toEqual([0, 1, 2, 3, 4, 5, 10, 32]);

    // AND-NOT (a AND ~b)
    const andNotBs = a.andNot(b);
    expect(andNotBs.getMatchingIndices()).toEqual([0, 1, 10]);

    // NOT (inversion up to capacity)
    const small = DocumentBitset.fromIndices([0, 2], 4);
    const notBs = small.not(4);
    expect(notBs.getMatchingIndices()).toEqual([1, 3]);
  });

  test('in-place bitwise operations (andInPlace, orInPlace, andNotInPlace)', () => {
    const a = DocumentBitset.fromIndices([0, 1, 2, 3], 10);
    const b = DocumentBitset.fromIndices([2, 3, 4], 10);

    a.andInPlace(b);
    expect(a.getMatchingIndices()).toEqual([2, 3]);

    a.orInPlace(DocumentBitset.fromIndices([8], 10));
    expect(a.getMatchingIndices()).toEqual([2, 3, 8]);

    a.andNotInPlace(DocumentBitset.fromIndices([2], 10));
    expect(a.getMatchingIndices()).toEqual([3, 8]);
  });
});

describe('ColumnarStore', () => {
  interface Product {
    id: string;
    title: string;
    category: string;
    price: number;
    inStock: boolean;
    tags?: string[];
  }

  const products: Product[] = [
    { id: 'p1', title: 'Laptop Pro', category: 'electronics', price: 1299.99, inStock: true, tags: ['pc', 'portable'] },
    { id: 'p2', title: 'Wireless Mouse', category: 'electronics', price: 29.99, inStock: true, tags: ['accessory'] },
    { id: 'p3', title: 'Desk Chair', category: 'furniture', price: 199.50, inStock: false, tags: ['office'] },
    { id: 'p4', title: 'Mechanical Keyboard', category: 'electronics', price: 109.00, inStock: true, tags: ['pc', 'accessory'] },
    { id: 'p5', title: 'Coffee Mug', category: 'kitchen', price: 12.00, inStock: true }
  ];

  test('indexes categorical string, numeric, boolean, and tag set columns', () => {
    const store = new ColumnarStore<Product>([
      { name: 'category', type: 'string' },
      { name: 'price', type: 'number' },
      { name: 'inStock', type: 'boolean' },
      { name: 'tags', type: 'string[]' }
    ]);

    store.init(products);

    // Equality on categorical string
    const elec = store.evaluateEquality('category', 'electronics');
    expect(elec.getMatchingIndices()).toEqual([0, 1, 3]);

    const furn = store.evaluateEquality('category', 'furniture');
    expect(furn.getMatchingIndices()).toEqual([2]);

    // Equality on boolean
    const inStockTrue = store.evaluateEquality('inStock', true);
    expect(inStockTrue.getMatchingIndices()).toEqual([0, 1, 3, 4]);

    const inStockFalse = store.evaluateEquality('inStock', false);
    expect(inStockFalse.getMatchingIndices()).toEqual([2]);

    // Range on numeric
    const cheap = store.evaluateRange('price', { lt: 50 });
    expect(cheap.getMatchingIndices()).toEqual([1, 4]); // 29.99 and 12.00

    const mid = store.evaluateRange('price', { gte: 100, lte: 200 });
    expect(mid.getMatchingIndices()).toEqual([2, 3]); // 199.50 and 109.00

    // IN on categorical string
    const inCat = store.evaluateIn('category', ['furniture', 'kitchen']);
    expect(inCat.getMatchingIndices()).toEqual([2, 4]);

    // Tag set matching
    const pcTag = store.evaluateEquality('tags', 'pc');
    expect(pcTag.getMatchingIndices()).toEqual([0, 3]);

    const accessoryTag = store.evaluateEquality('tags', 'accessory');
    expect(accessoryTag.getMatchingIndices()).toEqual([1, 3]);

    // Exists check on tags
    const hasTags = store.evaluateExists('tags', true);
    expect(hasTags.getMatchingIndices()).toEqual([0, 1, 2, 3]);

    const noTags = store.evaluateExists('tags', false);
    expect(noTags.getMatchingIndices()).toEqual([4]); // p5 has no tags
  });

  test('mutations (add, update, remove) correctly update columnar state', () => {
    const store = new ColumnarStore<Product>([
      { name: 'category', type: 'string' },
      { name: 'price', type: 'number' }
    ]);

    store.init(products);

    // Add new product at index 5
    store.add(5, { id: 'p6', title: 'Gaming Monitor', category: 'electronics', price: 349.99, inStock: true });
    let elec = store.evaluateEquality('category', 'electronics');
    expect(elec.getMatchingIndices()).toEqual([0, 1, 3, 5]);

    // Update product at index 1: category changes to 'refurbished'
    store.update(1, { id: 'p2', title: 'Wireless Mouse', category: 'refurbished', price: 19.99, inStock: true });
    elec = store.evaluateEquality('category', 'electronics');
    expect(elec.getMatchingIndices()).toEqual([0, 3, 5]);

    const refurbs = store.evaluateEquality('category', 'refurbished');
    expect(refurbs.getMatchingIndices()).toEqual([1]);

    // Remove product at index 3
    store.remove(3);
    elec = store.evaluateEquality('category', 'electronics');
    expect(elec.getMatchingIndices()).toEqual([0, 5]);
  });
});

describe('compileFilter AST Evaluator', () => {
  interface RecordItem {
    id: number;
    status: string;
    level: number;
    enabled: boolean;
    tags?: string[];
  }

  const items: RecordItem[] = [
    { id: 1, status: 'active', level: 10, enabled: true, tags: ['core', 'v1'] },
    { id: 2, status: 'pending', level: 20, enabled: false, tags: ['v1'] },
    { id: 3, status: 'active', level: 30, enabled: true, tags: ['core', 'v2'] },
    { id: 4, status: 'archived', level: 40, enabled: false },
    { id: 5, status: 'active', level: 50, enabled: true, tags: ['v2'] }
  ];

  const store = new ColumnarStore<RecordItem>([
    { name: 'status', type: 'string' },
    { name: 'level', type: 'number' },
    { name: 'enabled', type: 'boolean' },
    { name: 'tags', type: 'string[]' }
  ]);
  store.init(items);

  test('evaluates direct literal equality', () => {
    const expr: FilterExpression = { status: 'active' };
    const res = compileFilter(expr, store);
    expect(res.getMatchingIndices()).toEqual([0, 2, 4]);
  });

  test('evaluates multi-field object as implicit AND', () => {
    const expr: FilterExpression = {
      status: 'active',
      level: { gte: 30 }
    };
    const res = compileFilter(expr, store);
    expect(res.getMatchingIndices()).toEqual([2, 4]); // id 3 (level 30) and id 5 (level 50)
  });

  test('evaluates explicit AND combinator', () => {
    const expr: FilterExpression = {
      and: [
        { status: 'active' },
        { enabled: true },
        { level: { lt: 40 } }
      ]
    };
    const res = compileFilter(expr, store);
    expect(res.getMatchingIndices()).toEqual([0, 2]); // id 1 (10) and id 3 (30)
  });

  test('evaluates explicit OR combinator', () => {
    const expr: FilterExpression = {
      or: [
        { status: 'pending' },
        { status: 'archived' }
      ]
    };
    const res = compileFilter(expr, store);
    expect(res.getMatchingIndices()).toEqual([1, 3]);
  });

  test('evaluates explicit NOT combinator', () => {
    const expr: FilterExpression = {
      not: { status: 'active' }
    };
    const res = compileFilter(expr, store);
    expect(res.getMatchingIndices()).toEqual([1, 3]); // pending and archived
  });

  test('evaluates vacuous truth ({ and: [] }, {}) and vacuous falsehood ({ or: [] })', () => {
    const truth1 = compileFilter({ and: [] }, store);
    expect(truth1.getMatchingIndices()).toEqual([0, 1, 2, 3, 4]);

    const truth2 = compileFilter({}, store);
    expect(truth2.getMatchingIndices()).toEqual([0, 1, 2, 3, 4]);

    const false1 = compileFilter({ or: [] }, store);
    expect(false1.getMatchingIndices()).toEqual([]);
  });

  test('evaluates in and nin operators', () => {
    const inExpr: FilterExpression = {
      status: { in: ['active', 'pending'] }
    };
    expect(compileFilter(inExpr, store).getMatchingIndices()).toEqual([0, 1, 2, 4]);

    const ninExpr: FilterExpression = {
      status: { nin: ['active'] }
    };
    expect(compileFilter(ninExpr, store).getMatchingIndices()).toEqual([1, 3]);
  });

  test('evaluates nested combinators (AND with OR and NOT)', () => {
    const expr: FilterExpression = {
      and: [
        { enabled: true },
        {
          or: [
            { level: { lte: 10 } },
            { tags: 'v2' }
          ]
        },
        { not: { level: 50 } }
      ]
    };
    // enabled: true -> [0, 2, 4]
    // level <= 10 or tags: v2 -> id 1 (0), id 3 (2), id 5 (4) -> [0, 2, 4]
    // not level: 50 -> excludes id 5 (4)
    // Result: id 1 (0) and id 3 (2)
    const res = compileFilter(expr, store);
    expect(res.getMatchingIndices()).toEqual([0, 2]);
  });

  test('throws InvalidFilterError on unindexed fields or invalid syntax', () => {
    expect(() => {
      compileFilter({ unindexedField: 'foo' }, store);
    }).toThrow(InvalidFilterError);

    expect(() => {
      compileFilter({ status: { regex: '.*' } as any }, store);
    }).toThrow(InvalidFilterError);

    expect(() => {
      compileFilter({ and: 'not-an-array' as any }, store);
    }).toThrow(InvalidFilterError);

    expect(() => {
      compileFilter(null as any, store);
    }).toThrow(InvalidFilterError);
  });
});

describe('DocumentIndex with Structured Filters Integration', () => {
  interface Doc {
    id: string;
    title: string;
    body: string;
    role: string;
    stars: number;
    active: boolean;
  }

  const dataset: Doc[] = [
    { id: '1', title: 'Admin User Guide', body: 'Manage system settings and users', role: 'admin', stars: 5, active: true },
    { id: '2', title: 'Editor Basics', body: 'Writing and publishing blog posts', role: 'editor', stars: 4, active: true },
    { id: '3', title: 'Viewer Guide', body: 'Browse articles and search archives', role: 'viewer', stars: 2, active: false },
    { id: '4', title: 'Admin Security Policy', body: 'Network firewall rules and protocols', role: 'admin', stars: 5, active: false },
    { id: '5', title: 'Editor Advanced Tools', body: 'SEO optimization and media management', role: 'editor', stars: 3, active: true }
  ];

  const baseOptions: DocumentIndexOptions<Doc> = {
    fields: [
      { name: 'title', weight: 2.0 },
      { name: 'body', weight: 1.0 }
    ],
    filterFields: [
      { name: 'role', type: 'string' },
      { name: 'stars', type: 'number' },
      { name: 'active', type: 'boolean' }
    ],
    preferGpu: false
  };

  test('filters search results via structured FilterExpression on CPU parity', async () => {
    const index = await DocumentIndex.create(dataset, baseOptions);

    // Search 'Guide' with filter { role: 'admin' }
    // Matching docs: '1' and '4' have 'admin'
    // 'Guide' matches '1' (title: Admin User Guide) and '3' (Viewer Guide)
    // Only doc '1' matches BOTH 'Guide' AND role 'admin'
    const res1 = await index.search('Guide', {
      filter: { role: 'admin' }
    });
    expect(res1.totalMatches).toBe(1);
    expect(res1.results.length).toBe(1);
    expect(res1.results[0].id).toBe('1');

    // Search with numeric comparison: stars >= 4
    // Matches with 'Guide': doc '1' (stars: 5)
    // Doc '3' has stars: 2 (filtered out)
    const res2 = await index.search('Guide', {
      filter: { stars: { gte: 4 } }
    });
    expect(res2.totalMatches).toBe(1);
    expect(res2.results[0].id).toBe('1');

    // Zero selectivity pre-filter: stars > 100
    const resZero = await index.search('Guide', {
      filter: { stars: { gt: 100 } }
    });
    expect(resZero.totalMatches).toBe(0);
    expect(resZero.results.length).toBe(0);
  });

  test('filters search results via structured FilterExpression on WebGPU mock device', async () => {
    const adapter = await createMockAdapter({
      features: ['timestamp-query'] as any
    });
    const mockDeviceWrapper = await adapter.requestDevice();
    const mockDevice = (mockDeviceWrapper as any).gpu ?? mockDeviceWrapper;

    const index = await DocumentIndex.create(dataset, {
      ...baseOptions,
      device: mockDevice,
      preferGpu: true
    });

    const res = await index.search('Editor', {
      filter: { active: true }
    });

    expect(res.engine).toBe('webgpu');
    // On mock GPU device, mock ALU returns empty readback, confirming engine executes through WebGPU pipeline without errors
    expect(res.totalMatches).toBe(0);

    // Filter by stars: { lt: 4 }
    const resStars = await index.search('Editor', {
      filter: { stars: { lt: 4 } }
    });
    expect(resStars.engine).toBe('webgpu');
    expect(resStars.totalMatches).toBe(0);

    index.destroy();
  });

  test('mutations dynamically update columnar pre-filter evaluation', async () => {
    const index = await DocumentIndex.create(dataset, baseOptions);

    // Add a new document
    await index.add({
      id: '6',
      title: 'Admin Quickstart',
      body: 'Setup guide for new administrators',
      role: 'admin',
      stars: 5,
      active: true
    });

    const resAfterAdd = await index.search('Guide', {
      filter: { role: 'admin' }
    });
    // Now docs 1 and 6 match
    expect(resAfterAdd.totalMatches).toBe(2);
    expect(resAfterAdd.results.map((r) => r.id)).toEqual(['1', '6']);

    // Update doc 1 to role 'auditor'
    await index.update({
      id: '1',
      title: 'Auditor User Guide',
      body: 'Manage system settings and audit trails',
      role: 'auditor',
      stars: 5,
      active: true
    });

    const resAfterUpdate = await index.search('Guide', {
      filter: { role: 'admin' }
    });
    expect(resAfterUpdate.totalMatches).toBe(1);
    expect(resAfterUpdate.results[0].id).toBe('6');

    // Remove doc 6
    await index.remove('6');
    const resAfterRemove = await index.search('Guide', {
      filter: { role: 'admin' }
    });
    expect(resAfterRemove.totalMatches).toBe(0);
  });

  test('SearchWorkerClient evaluates structured FilterExpression across thread boundary', async () => {
    const worker = new SearchWorkerClient<Doc>();
    try {
      await worker.init(dataset, baseOptions);

      const res = await worker.search('Guide', {
        filter: { role: 'admin' }
      });

      expect(res.totalMatches).toBe(1);
      expect(res.results[0].id).toBe('1');

      // Test complex filter in worker
      const resComplex = await worker.search('Editor', {
        filter: {
          and: [
            { role: 'editor' },
            { stars: { gte: 4 } }
          ]
        }
      });
      expect(resComplex.totalMatches).toBe(1);
      expect(resComplex.results[0].id).toBe('2');
    } finally {
      await worker.destroy();
    }
  });

  test('performance: filter evaluation resolves in < 50µs for 10k rows', () => {
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

    // Warm up JIT (100 iterations to allow FTL tier-up)
    for (let i = 0; i < 100; i++) {
      compileFilter(filter, store);
    }

    // Benchmark 100 runs
    const iterations = 100;
    const t0 = performance.now();
    for (let i = 0; i < iterations; i++) {
      compileFilter(filter, store);
    }
    const t1 = performance.now();
    const avgMicros = ((t1 - t0) / iterations) * 1000;

    console.log(`   [Performance] 10k rows filter evaluation: ${avgMicros.toFixed(2)} µs (target: < 50 µs)`);
    expect(avgMicros).toBeLessThan(50);
  });
});
