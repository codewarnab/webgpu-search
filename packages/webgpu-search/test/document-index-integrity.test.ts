import { describe, test, expect } from 'bun:test';
import {
  ColumnarStore,
  compileFilter,
  compareIdsAsc,
  DocumentIndex,
  InvalidFilterError
} from '../src/index';

const cpu = { preferGpu: false } as const;

describe('applyBatch atomicity with throwing filter getters', () => {
  test('a throwing filter getter rejects the whole batch and leaves the index intact', async () => {
    const idx = await DocumentIndex.create([{ id: 1, title: 'apple', cat: 'a' }] as any[], {
      ...cpu,
      fields: ['title'],
      filterFields: [
        {
          name: 'cat',
          type: 'string',
          getter: (d: any) => {
            if (d.bad) throw new Error('boom');
            return d.cat;
          }
        }
      ]
    } as any);

    await expect(
      idx.add([{ id: 2, title: 'banana', cat: 'b' }, { id: 3, title: 'cherry', bad: true }] as any[])
    ).rejects.toThrow('boom');
    await expect(idx.update({ id: 1, title: 'apricot', bad: true } as any)).rejects.toThrow('boom');

    expect(idx.getStats().docCount).toBe(1);
    expect((await idx.search('banana')).results).toHaveLength(0);
    expect((await idx.search('apple', { filter: { cat: 'a' } })).results.map((r) => r.id)).toEqual([1]);

    // Subsequent mutations still work (no stale idToDocIndex / misaligned rows).
    await idx.add([{ id: 2, title: 'banana', cat: 'b' }, { id: 4, title: 'durian', cat: 'd' }] as any[]);
    await idx.update({ id: 4, title: 'eggplant', cat: 'e' } as any);
    expect((await idx.search('banana', { filter: { cat: 'b' } })).results.map((r) => r.id)).toEqual([2]);
    expect((await idx.search('eggplant', { filter: { cat: 'e' } })).results.map((r) => r.id)).toEqual([4]);
    expect((await idx.search('durian')).results).toHaveLength(0);
    expect(idx.getStats().docCount).toBe(3);
  });
});

describe('default getters ignore inherited Object.prototype members', () => {
  test('search field and filter field named "constructor"', async () => {
    const idx = await DocumentIndex.create(
      [{ id: 1, title: 'apple' }, { id: 2, title: 'pear', constructor: 'acme' }] as any[],
      { ...cpu, fields: ['title', 'constructor'], filterFields: [{ name: 'constructor', type: 'string' }] } as any
    );
    // Without own-property access, `Object` would be stringified as "function Object() { [native code] }".
    expect((await idx.search('native code')).results).toHaveLength(0);
    expect((await idx.search('acme')).results.map((r) => r.id)).toEqual([2]);
    expect(
      (await idx.search('apple', { filter: { constructor: { exists: false } } } as any)).results.map((r) => r.id)
    ).toEqual([1]);
  });

  test('ColumnarStore default getter reads own props and class accessors only', () => {
    class Item {
      constructor(private readonly c: string) {}
      get cat(): string {
        return this.c;
      }
    }
    const s = new ColumnarStore<any>([
      { name: 'toString', type: 'string' },
      { name: 'cat', type: 'string' }
    ]);
    s.init([{}, new Item('x')]);
    expect(compileFilter({ toString: { exists: true } } as any, s).getMatchingIndices()).toEqual([]);
    expect(compileFilter({ cat: 'x' }, s).getMatchingIndices()).toEqual([1]);
  });
});

describe('highlights with prototype-sensitive field names', () => {
  test('field named __proto__ keeps highlights as an own key', async () => {
    const idx = await DocumentIndex.create([{ id: 1, a: 'hello' }] as any[], {
      ...cpu,
      fields: [{ name: '__proto__', getter: (d: any) => d.a }]
    } as any);
    const r = (await idx.search('hello', { highlight: { tag: 'mark' } } as any)).results[0];
    expect(r).toBeDefined();
    expect(Object.getPrototypeOf(r.highlights)).toBe(Object.prototype);
    expect(Object.prototype.hasOwnProperty.call(r.highlights, '__proto__')).toBe(true);
    expect(Object.keys(r.highlights!)).toEqual(['__proto__']);
    expect(Array.isArray((r.highlights as any)['__proto__'])).toBe(true);
  });
});

describe('compareIdsAsc total order', () => {
  test('mixed number/string ids are transitive (numbers first)', () => {
    expect(compareIdsAsc(9, 10)).toBe(-1);
    expect(compareIdsAsc(10, '10a')).toBe(-1);
    expect(compareIdsAsc(9, '10a')).toBe(-1);
    expect(compareIdsAsc('10a', 9)).toBe(1);
    expect(compareIdsAsc(2, '2')).toBe(-1);
    expect(compareIdsAsc('10', '9')).toBe(-1);
    expect(compareIdsAsc('a', 'a')).toBe(0);
    expect(() => compareIdsAsc(NaN, 'a')).toThrow(TypeError);
    const ids: Array<string | number> = ['b', 10, '10a', 9, '9', 2, 'a'];
    for (const perm of [ids, [...ids].reverse(), [...ids].sort()]) {
      expect([...perm].sort(compareIdsAsc)).toEqual([2, 9, 10, '10a', '9', 'a', 'b']);
    }
  });

  test('search ordering of tied mixed ids is input-order independent', async () => {
    const run = async (docs: any[]) =>
      (await (await DocumentIndex.create(docs, { ...cpu, fields: ['t'] } as any)).search('x')).results.map(
        (r) => r.id
      );
    const a = await run([{ id: '10a', t: 'x' }, { id: 10, t: 'x' }, { id: 9, t: 'x' }]);
    const b = await run([{ id: 10, t: 'x' }, { id: 9, t: 'x' }, { id: '10a', t: 'x' }]);
    expect(a).toEqual([9, 10, '10a']);
    expect(b).toEqual(a);
  });
});

describe('ColumnarStore strict value coercion', () => {
  test('blank / non-finite numeric range bounds are rejected', () => {
    const s = new ColumnarStore<any>([{ name: 'n', type: 'number' }]);
    s.init([{ n: 0 }, { n: 5 }]);
    for (const bad of ['', '  ', 'abc', 'Infinity', NaN, Infinity]) {
      expect(() => compileFilter({ n: { gte: bad as any } }, s)).toThrow(InvalidFilterError);
    }
    expect(compileFilter({ n: { gte: '1' } }, s).getMatchingIndices()).toEqual([1]);
  });

  test('blank strings are missing for number fields at index and query time', () => {
    const s = new ColumnarStore<any>([{ name: 'n', type: 'number' }]);
    s.init([{ n: '' }, { n: ' 3 ' }, { n: 0 }]);
    expect(compileFilter({ n: { exists: true } }, s).getMatchingIndices()).toEqual([1, 2]);
    expect(compileFilter({ n: '' }, s).getMatchingIndices()).toEqual([]);
    expect(compileFilter({ n: 3 }, s).getMatchingIndices()).toEqual([1]);
  });

  test('boolean fields accept only booleans and "true"/"false"', () => {
    const s = new ColumnarStore<any>([{ name: 'b', type: 'boolean' }]);
    s.init([{ b: 'false' }, { b: false }, { b: true }, { b: 'true' }, { b: 1 }, { b: 'yes' }, {}]);
    expect(compileFilter({ b: false }, s).getMatchingIndices()).toEqual([0, 1]);
    expect(compileFilter({ b: true }, s).getMatchingIndices()).toEqual([2, 3]);
    expect(compileFilter({ b: 'false' }, s).getMatchingIndices()).toEqual([0, 1]);
    expect(compileFilter({ b: null }, s).getMatchingIndices()).toEqual([4, 5, 6]);
    expect(compileFilter({ b: { in: [true, null] } }, s).getMatchingIndices()).toEqual([2, 3, 4, 5, 6]);
    expect(() => compileFilter({ b: 'yes' }, s)).toThrow(InvalidFilterError);
    expect(() => compileFilter({ b: 0 } as any, s)).toThrow(InvalidFilterError);
    expect(() => compileFilter({ b: { in: [1] } } as any, s)).toThrow(InvalidFilterError);
  });
});

describe('ColumnarStore memory-proportional value postings', () => {
  const distinctBitsetBytes = (s: ColumnarStore<any>, name: string): number => {
    const col = s.getColumn(name) as any;
    const map: Map<unknown, any> = col.type === 'string' ? col.invertedIndex : col.tagInverted;
    let dense = 0;
    for (const pl of map.values()) if (pl.isDense) dense++;
    return dense;
  };

  test('high-cardinality values stay sparse; low-cardinality values go dense', () => {
    const N = 40_000;
    const s = new ColumnarStore<any>([
      { name: 'sku', type: 'string' },
      { name: 'cat', type: 'string' },
      { name: 'tags', type: 'string[]' }
    ]);
    s.init(Array.from({ length: N }, (_, i) => ({ sku: 's' + i, cat: 'c' + (i % 4), tags: ['t' + i, 'common'] })));
    expect(distinctBitsetBytes(s, 'sku')).toBe(0);
    expect(distinctBitsetBytes(s, 'cat')).toBe(4);
    expect(distinctBitsetBytes(s, 'tags')).toBe(1);

    expect(compileFilter({ sku: 's123' }, s).getMatchingIndices()).toEqual([123]);
    expect(compileFilter({ sku: ['s1', 's39999'] }, s).getMatchingIndices()).toEqual([1, 39999]);
    expect(compileFilter({ cat: 'c1', sku: { gte: 's39990', lt: 's4' } }, s).getMatchingIndices()).toEqual([
      39993, 39997
    ]);
    expect(compileFilter({ tags: 't7' }, s).getMatchingIndices()).toEqual([7]);
    expect(compileFilter({ tags: 'common' }, s).popcount()).toBe(N);
    expect(compileFilter({ tags: { nin: ['common'] } }, s).popcount()).toBe(0);
  });

  test('remove + compact drops dead codes and keeps filters/facets correct', () => {
    const s = new ColumnarStore<any>([
      { name: 'sku', type: 'string' },
      { name: 'tags', type: 'string[]' }
    ]);
    const N = 200;
    s.init(Array.from({ length: N }, (_, i) => ({ sku: 's' + i, tags: ['t' + (i % 3), 'u' + i] })));
    const map = new Map<number, number>();
    for (let i = 0; i < N; i++) {
      if (i % 2 === 0) s.remove(i);
      else map.set(i, map.size);
    }
    // Removed values are released from postings eagerly.
    expect((s.getColumn('sku') as any).invertedIndex.size).toBe(N / 2);
    expect((s.getColumn('tags') as any).tagInverted.has('u0')).toBe(false);

    s.compact(map, map.size);
    const col = s.getColumn('sku') as any;
    expect(col.stringTable).toHaveLength(N / 2);
    expect(col.stringToCode.has('s0')).toBe(false);
    expect(compileFilter({ sku: 's0' }, s).popcount()).toBe(0);
    expect(compileFilter({ sku: 's3' }, s).getMatchingIndices()).toEqual([1]);
    expect(compileFilter({ tags: 't0' }, s).popcount()).toBe(33); // odd i with i % 3 === 0
    const facet = s.termsDistribution('sku', [0, 1])!;
    expect(facet).toEqual([
      { value: 's1', count: 1 },
      { value: 's3', count: 1 }
    ]);
    // Re-adding a dropped value after compaction works.
    s.add(map.size, { sku: 's0', tags: ['u0'] });
    expect(compileFilter({ sku: 's0' }, s).getMatchingIndices()).toEqual([map.size]);
  });

  test('dense postings demote when they become sparse', () => {
    const s = new ColumnarStore<any>([{ name: 'cat', type: 'string' }]);
    const N = 4096;
    s.init(Array.from({ length: N }, (_, i) => ({ cat: i < 2048 ? 'hot' : 'x' + i })));
    const pl = () => (s.getColumn('cat') as any).invertedIndex.get(0);
    expect(pl().isDense).toBe(true);
    for (let i = 0; i < 2040; i++) s.remove(i);
    expect(pl().isDense).toBe(false);
    expect(compileFilter({ cat: 'hot' }, s).getMatchingIndices()).toEqual([2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047]);
  });
});

describe('ColumnarStore.prepare / addPrepared', () => {
  test('prepare runs getters without mutating; addPrepared applies', () => {
    let calls = 0;
    const s = new ColumnarStore<any>([
      { name: 'a', type: 'string', getter: (d: any) => { calls++; return d.a; } },
      { name: 'n', type: 'number' }
    ]);
    const row = s.prepare({ a: 'x', n: '7' });
    expect(calls).toBe(1);
    expect(s.getActiveDocs().popcount()).toBe(0);
    s.addPrepared(0, row);
    expect(calls).toBe(1);
    expect(compileFilter({ a: 'x', n: 7 }, s).getMatchingIndices()).toEqual([0]);
    expect(() => s.addPrepared(1, [])).toThrow(TypeError);
  });
});
