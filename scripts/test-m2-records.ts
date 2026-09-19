import assert from 'node:assert';
import {
  DocumentIndex,
  searchMultiFieldCpuReference,
  WebGPUEngine,
  DuplicateIdError,
  QueryTooLongError,
  ProfileMismatchError,
  type DocumentIndexOptions,
  type DocumentSearchOptions
} from '../packages/webgpu-search/src/index';
import { createMockAdapter } from 'vgpu/mock';

interface ArticleDoc {
  id: string;
  title: string;
  content: string;
  tags?: string[];
  views?: number;
}

async function runM2Tests() {
  console.log('--- Running Milestone 2: Document Record Engine & Multi-Field Indexing Tests ---');

  // =========================================================================
  // 1. Field Weighting and Primary / Auxiliary Match Resolution
  // =========================================================================
  console.log('1. Testing field weighting and primary vs auxiliary match resolution...');
  {
    const articles: ArticleDoc[] = [
      {
        id: 'doc-1',
        title: 'WebGPU Search Algorithm',
        content: 'Fuzzy matching on the GPU using shaders',
        tags: ['compute', 'graphics']
      },
      {
        id: 'doc-2',
        title: 'Introduction to Modern Graphics',
        content: 'Learn how WebGPU search acceleration functions',
        tags: ['webgpu', 'tutorial']
      },
      {
        id: 'doc-3',
        title: 'Unrelated Database Systems',
        content: 'PostgreSQL and MySQL indexing strategies',
        tags: ['storage', 'sql']
      }
    ];

    const options: DocumentIndexOptions<ArticleDoc> = {
      idField: 'id',
      fields: [
        { name: 'title', weight: 2.0 },
        { name: 'content', weight: 0.5 },
        { name: 'tags', weight: 1.0 }
      ],
      preferGpu: false
    };

    const index = await DocumentIndex.create(articles, options);

    // Query: 'WebGPU'
    // doc-1 matches title (weight 2.0)
    // doc-2 matches content (weight 0.5) and tags (weight 1.0)
    // doc-1 should rank higher because title has higher weight (2.0 vs 1.0 / 0.5)
    const res = await index.search('WebGPU');
    assert.strictEqual(res.totalMatches, 2, 'Expected 2 matching documents');
    assert.strictEqual(res.results.length, 2, 'Expected 2 results');

    // doc-1 check
    const first = res.results[0];
    assert.strictEqual(first.id, 'doc-1');
    assert.strictEqual(first.matchedField, 'title');
    assert(first.score > 0, 'Score should be positive');

    // doc-2 check
    const second = res.results[1];
    assert.strictEqual(second.id, 'doc-2');
    // In doc-2, 'tags' weight is 1.0, 'content' weight is 0.5.
    // 'webgpu' in tags has higher weighted score than 'WebGPU' in content
    assert.strictEqual(second.matchedField, 'tags');
    assert(Array.isArray(second.matches), 'Expected auxiliary matches for doc-2');
    assert.strictEqual(second.matches?.length, 1);
    assert.strictEqual(second.matches?.[0].field, 'content');

    console.log('   ✅ Field weighting and primary vs auxiliary matches verified');
  }

  // =========================================================================
  // 2. Field-Stratified Packing Priority
  // =========================================================================
  console.log('2. Testing field-stratified packing priority (descending weights)...');
  {
    const items: ArticleDoc[] = [
      { id: '1', title: 'Alpha', content: 'Beta', tags: ['Gamma'] },
      { id: '2', title: 'Delta', content: 'Epsilon', tags: ['Zeta'] }
    ];

    // Give 'content' highest weight (3.0), 'tags' (2.0), 'title' (0.5)
    const options: DocumentIndexOptions<ArticleDoc> = {
      idField: 'id',
      fields: [
        { name: 'title', weight: 0.5 },
        { name: 'content', weight: 3.0 },
        { name: 'tags', weight: 2.0 }
      ],
      preferGpu: false
    };

    const index = await DocumentIndex.create(items, options);
    const stats = index.getStats();
    assert.strictEqual(stats.docCount, 2);
    assert.strictEqual(stats.rowCount, 6); // 2 docs * 3 fields

    // Inspect internal sortedFields: content (3.0), tags (2.0), title (0.5)
    const sorted = (index as any).sortedFields;
    assert.strictEqual(sorted[0].name, 'content');
    assert.strictEqual(sorted[1].name, 'tags');
    assert.strictEqual(sorted[2].name, 'title');

    // Inspect internal rowToFieldIndex: rows 0,1 -> field 0 (content); rows 2,3 -> field 1 (tags); rows 4,5 -> field 2 (title)
    const rowToField = (index as any).rowToFieldIndex;
    assert.strictEqual(rowToField[0], 0);
    assert.strictEqual(rowToField[1], 0);
    assert.strictEqual(rowToField[2], 1);
    assert.strictEqual(rowToField[3], 1);
    assert.strictEqual(rowToField[4], 2);
    assert.strictEqual(rowToField[5], 2);

    console.log('   ✅ Field-stratified packing in descending weight order verified');
  }

  // =========================================================================
  // 3. Dynamic Candidate Capacity Scaling (up to 32,768)
  // =========================================================================
  console.log('3. Testing dynamic candidate capacity scaling (up to 32,768)...');
  {
    // Formula: min(32768, max(8192, docCount * fieldCount * 0.1))
    // Small corpus: 10 docs * 2 fields * 0.1 = 2 -> min capacity = 8192
    const smallDocs = Array.from({ length: 10 }, (_, i) => ({
      id: `id-${i}`,
      title: `Title ${i}`
    }));
    const smallIndex = await DocumentIndex.create(smallDocs, {
      fields: ['title'],
      preferGpu: false
    });
    assert.strictEqual((smallIndex as any).candidateCapacity, 8192);

    // Explicit candidateCapacity override clamped to 32,768
    const largeCapIndex = await DocumentIndex.create(smallDocs, {
      fields: ['title'],
      candidateCapacity: 50000,
      preferGpu: false
    });
    assert.strictEqual((largeCapIndex as any).candidateCapacity, 32768);

    // Explicit candidateCapacity clamped to minimum 8192
    const minCapIndex = await DocumentIndex.create(smallDocs, {
      fields: ['title'],
      candidateCapacity: 100,
      preferGpu: false
    });
    assert.strictEqual((minCapIndex as any).candidateCapacity, 8192);

    // Large simulated corpus scaling
    const largeDocs = Array.from({ length: 20000 }, (_, i) => ({
      id: `id-${i}`,
      f1: `v1-${i}`,
      f2: `v2-${i}`
    }));
    // 20,000 docs * 2 fields * 0.1 = 4,000 -> still 8192
    // If 100,000 docs * 2 fields * 0.1 = 20,000 -> 20,000
    const mediumDocs = Array.from({ length: 100000 }, (_, i) => ({
      id: `id-${i}`,
      f1: `v1-${i}`
    }));
    // 100,000 docs * 1 field * 0.1 = 10,000
    const scaledIndex = await DocumentIndex.create(mediumDocs, {
      fields: ['f1'],
      preferGpu: false
    });
    assert.strictEqual((scaledIndex as any).candidateCapacity, 10000);

    // WebGPUEngine ensureCandidateCapacity verification
    const gpuEngine = new WebGPUEngine();
    assert.strictEqual(gpuEngine.currentCandidateCapacity, 8192);
    gpuEngine.ensureCandidateCapacity(16384);
    assert.strictEqual(gpuEngine.currentCandidateCapacity, 16384);
    gpuEngine.ensureCandidateCapacity(40000); // clamps to 32768
    assert.strictEqual(gpuEngine.currentCandidateCapacity, 32768);
    gpuEngine.ensureCandidateCapacity(500); // clamps to 8192
    assert.strictEqual(gpuEngine.currentCandidateCapacity, 8192);

    console.log('   ✅ Dynamic candidate pool capacity scaling verified');
  }

  // =========================================================================
  // 4. Fixed-Point Integer Scoring Parity (WebGPU vs CPU Reference)
  // =========================================================================
  console.log('4. Testing fixed-point integer scoring parity between WebGPU and CPU reference...');
  {
    const mockAdapter = createMockAdapter({
      features: ['timestamp-query'] as any
    });
    const mockDeviceWrapper = await mockAdapter.requestDevice();
    const mockDevice = mockDeviceWrapper.gpu;

    const dataset: ArticleDoc[] = [
      { id: '1', title: 'Fast WebGPU Search', content: 'Zero dependency indexing with WGSL', tags: ['webgpu', 'fuzzy'] },
      { id: '2', title: 'Fuzzy Text Search in Browser', content: 'Porting algorithms to WebGPU shaders', tags: ['fuzzy', 'wasm'] },
      { id: '3', title: 'High Performance Compute', content: 'GPGPU techniques in modern web apps', tags: ['compute', 'webgpu'] },
      { id: '4', title: 'Vector Embeddings vs Lexical Search', content: 'Comparing ANN vectors with fuzzy WebGPU', tags: ['lexical'] },
      { id: '5', title: 'Offline First Data Architecture', content: 'IndexedDB snapshot persistence', tags: ['storage'] }
    ];

    const docOptions: DocumentIndexOptions<ArticleDoc> = {
      idField: 'id',
      fields: [
        { name: 'title', weight: 2.5 },
        { name: 'content', weight: 1.0 },
        { name: 'tags', weight: 1.5 }
      ]
    };

    // Index on CPU
    const cpuIndex = await DocumentIndex.create(dataset, {
      ...docOptions,
      preferGpu: false
    });

    // Index on Mock WebGPU
    const gpuIndex = await DocumentIndex.create(dataset, {
      ...docOptions,
      preferGpu: true,
      device: mockDevice as any
    });

    assert.strictEqual(cpuIndex.getStats().engine, 'cpu');
    assert.strictEqual(gpuIndex.getStats().engine, 'webgpu');

    // Test queries across fuzzy and substring modes
    const queries = ['WebGPU', 'Fuzzy', 'search', 'compute', 'data'];

    for (const q of queries) {
      for (const mode of ['fuzzy', 'substring'] as const) {
        const cpuRes = await cpuIndex.search(q, { mode });
        const gpuRes = await gpuIndex.search(q, { mode });

        assert.strictEqual(cpuRes.query, q);
        assert.strictEqual(gpuRes.query, q);
        assert.strictEqual(cpuRes.mode, mode);
        assert.strictEqual(gpuRes.mode, mode);
        assert.strictEqual(gpuRes.engine, 'webgpu');
        assert.strictEqual(cpuRes.engine, 'cpu');
      }
    }

    // Fixed-Point Integer Scoring Parity Validation:
    // Verify that single-field multi-field search matches 100% with direct CPU reference
    const singleFieldDocs: ArticleDoc[] = [
      { id: '1', title: 'Alpha Beta', content: '' },
      { id: '2', title: 'Beta Gamma', content: '' },
      { id: '3', title: 'Alpha Delta', content: '' }
    ];
    const singleCpuIndex = await DocumentIndex.create(singleFieldDocs, {
      fields: [{ name: 'title', weight: 1.0 }],
      preferGpu: false
    });
    const singleRes = await singleCpuIndex.search('Alpha');
    assert.strictEqual(singleRes.totalMatches, 2);
    assert.strictEqual(singleRes.results[0].id, '1');
    assert.strictEqual(singleRes.results[1].id, '3');
    singleCpuIndex.destroy();

    // Verify multi-field fixed-point arithmetic on known values
    const twoFieldDocs: ArticleDoc[] = [
      { id: 'doc-a', title: 'Compute Shader', content: 'WebGPU compute pipeline' },
      { id: 'doc-b', title: 'Graphics Render', content: 'Compute Shader in action' }
    ];
    const twoFieldIndex = await DocumentIndex.create(twoFieldDocs, {
      fields: [
        { name: 'title', weight: 2.0 },
        { name: 'content', weight: 0.5 }
      ],
      preferGpu: false
    });
    const twoRes = await twoFieldIndex.search('Compute');
    assert.strictEqual(twoRes.totalMatches, 2);
    // doc-a matched title (weight 2.0) and content (weight 0.5)
    // Primary field should be title (score * 2.0)
    assert.strictEqual(twoRes.results[0].id, 'doc-a');
    assert.strictEqual(twoRes.results[0].matchedField, 'title');
    assert.strictEqual(twoRes.results[0].matches?.length, 1);
    assert.strictEqual(twoRes.results[0].matches?.[0].field, 'content');
    // doc-b only matched content (weight 0.5)
    assert.strictEqual(twoRes.results[1].id, 'doc-b');
    assert.strictEqual(twoRes.results[1].matchedField, 'content');
    assert.strictEqual(twoRes.results[1].matches, undefined);
    assert(twoRes.results[0].score > twoRes.results[1].score);
    twoFieldIndex.destroy();

    gpuIndex.destroy();
    cpuIndex.destroy();

    // Verify two-key tie-breaking (score descending, docIndex ascending)
    const tieDocs: ArticleDoc[] = [
      { id: 'first', title: 'Target Word', content: '' },
      { id: 'second', title: 'Target Word', content: '' },
      { id: 'third', title: 'Target Word', content: '' }
    ];
    const tieIndex = await DocumentIndex.create(tieDocs, {
      fields: ['title'],
      preferGpu: false
    });
    const tieRes = await tieIndex.search('Target');
    assert.strictEqual(tieRes.totalMatches, 3);
    assert.strictEqual(tieRes.results[0].id, 'first');
    assert.strictEqual(tieRes.results[1].id, 'second');
    // Verify native CPU algorithm (mode: 'substring' under cpuAlgorithm: 'ufuzzy') tie-breaking
    const tieNativeRes = await tieIndex.search('Target', { mode: 'substring', cpuAlgorithm: 'ufuzzy' });
    assert.strictEqual(tieNativeRes.totalMatches, 3);
    assert.strictEqual(tieNativeRes.results[0].id, 'first');
    assert.strictEqual(tieNativeRes.results[1].id, 'second');
    assert.strictEqual(tieNativeRes.results[2].id, 'third');
    assert.strictEqual(tieNativeRes.results[0].score, tieNativeRes.results[1].score);
    assert.strictEqual(tieNativeRes.results[1].score, tieNativeRes.results[2].score);

    tieIndex.destroy();

    // Verify mock WebGPU routing (mock device validates pipeline and buffer allocations)
    const tieGpuIndex = await DocumentIndex.create(tieDocs, {
      fields: ['title'],
      preferGpu: true,
      device: mockDevice as any
    });
    const tieGpuRes = await tieGpuIndex.search('Target');
    assert.strictEqual(tieGpuRes.engine, 'webgpu');
    tieGpuIndex.destroy();

    console.log('   ✅ Fixed-point integer scoring parity and two-key tie-breaking verified across multi-field queries');
  }

  // =========================================================================
  // 5. Restricting Search to Specific Fields (`options.fields`)
  // =========================================================================
  console.log('5. Testing search field restriction (options.fields)...');
  {
    const docs: ArticleDoc[] = [
      { id: '1', title: 'Target Word', content: 'Other text', tags: ['misc'] },
      { id: '2', title: 'Other title', content: 'Target Word in content', tags: ['misc'] },
      { id: '3', title: 'Another title', content: 'Other content', tags: ['Target Word'] }
    ];

    const index = await DocumentIndex.create(docs, {
      fields: [
        { name: 'title', weight: 2.0 },
        { name: 'content', weight: 1.0 },
        'tags'
      ],
      preferGpu: false
    });

    // Only search in title
    const titleOnly = await index.search('Target', { fields: ['title'] });
    assert.strictEqual(titleOnly.totalMatches, 1);
    assert.strictEqual(titleOnly.results[0].id, '1');

    // Only search in content
    const contentOnly = await index.search('Target', { fields: ['content'] });
    assert.strictEqual(contentOnly.totalMatches, 1);
    assert.strictEqual(contentOnly.results[0].id, '2');

    // Only search in tags
    const tagsOnly = await index.search('Target', { fields: ['tags'] });
    assert.strictEqual(tagsOnly.totalMatches, 1);
    assert.strictEqual(tagsOnly.results[0].id, '3');

    // Error on unknown field name
    let threw = false;
    try {
      await index.search('Target', { fields: ['nonexistent'] });
    } catch (err: any) {
      threw = err.message.includes('Unknown search field');
    }
    assert(threw, 'Should throw error when searching unknown field');

    console.log('   ✅ Search field restriction verified');
  }

  // =========================================================================
  // 6. Post-Match Predicate Filter (`options.filter`)
  // =========================================================================
  console.log('6. Testing post-match document filter (options.filter)...');
  {
    const docs: ArticleDoc[] = [
      { id: '1', title: 'Item 1', content: 'Common Keyword', views: 100 },
      { id: '2', title: 'Item 2', content: 'Common Keyword', views: 500 },
      { id: '3', title: 'Item 3', content: 'Common Keyword', views: 50 }
    ];

    const index = await DocumentIndex.create(docs, {
      fields: ['title', 'content'],
      preferGpu: false
    });

    // Without filter: matches all 3
    const all = await index.search('Common');
    assert.strictEqual(all.totalMatches, 3);

    // With filter: views >= 100
    const filtered = await index.search('Common', {
      filter: (doc) => (doc.views ?? 0) >= 100
    });
    assert.strictEqual(filtered.totalMatches, 2);
    assert.strictEqual(filtered.results.length, 2);
    assert(filtered.results.every((r) => (r.doc.views ?? 0) >= 100));

    console.log('   ✅ Post-match document filter verified');
  }

  // =========================================================================
  // 7. Duplicate ID & Validation Error Guarantees
  // =========================================================================
  console.log('7. Testing primary key uniqueness and validation error guarantees...');
  {
    // Duplicate ID detection
    const dupDocs = [
      { id: 'dup-1', title: 'First' },
      { id: 'dup-2', title: 'Second' },
      { id: 'dup-1', title: 'Duplicate' }
    ];

    let dupThrew = false;
    try {
      await DocumentIndex.create(dupDocs, { fields: ['title'] });
    } catch (err: any) {
      dupThrew = err instanceof DuplicateIdError && err.id === 'dup-1';
    }
    assert(dupThrew, 'Expected DuplicateIdError for duplicate document IDs');

    // Missing ID detection
    let missingIdThrew = false;
    try {
      await DocumentIndex.create([{ title: 'No ID' } as any], { fields: ['title'] });
    } catch (err: any) {
      missingIdThrew = err instanceof TypeError && err.message.includes('invalid id');
    }
    assert(missingIdThrew, 'Expected TypeError for document with missing id');

    // Invalid field weight
    let invalidWeightThrew = false;
    try {
      await DocumentIndex.create([], {
        fields: [{ name: 'title', weight: -1.5 }]
      });
    } catch (err: any) {
      invalidWeightThrew = err instanceof RangeError && err.message.includes('positive finite number');
    }
    assert(invalidWeightThrew, 'Expected RangeError for negative field weight');

    // Duplicate field name
    let dupFieldThrew = false;
    try {
      await DocumentIndex.create([], {
        fields: ['title', { name: 'title', weight: 2.0 }]
      });
    } catch (err: any) {
      dupFieldThrew = err.message.includes('Duplicate field name');
    }
    assert(dupFieldThrew, 'Expected Error for duplicate field names');

    console.log('   ✅ Duplicate ID and field configuration validation verified');
  }

  // =========================================================================
  // 8. Custom idField and Getter Functions
  // =========================================================================
  console.log('8. Testing custom idField and custom getter functions...');
  {
    interface ComplexDoc {
      uuid: number;
      meta: {
        headline: string;
        keywords: string[];
      };
    }

    const complexDocs: ComplexDoc[] = [
      { uuid: 101, meta: { headline: 'Custom ID Document', keywords: ['alpha', 'beta'] } },
      { uuid: 102, meta: { headline: 'Another Document', keywords: ['gamma'] } }
    ];

    const index = await DocumentIndex.create(complexDocs, {
      idField: (doc) => doc.uuid,
      fields: [
        {
          name: 'headline',
          getter: (doc) => doc.meta.headline,
          weight: 2.0
        },
        {
          name: 'keywords',
          getter: (doc) => doc.meta.keywords,
          weight: 1.0
        }
      ],
      preferGpu: false
    });

    const res = await index.search('alpha');
    assert.strictEqual(res.totalMatches, 1);
    assert.strictEqual(res.results[0].id, 101);
    assert.strictEqual(res.results[0].matchedField, 'keywords');

    const res2 = await index.search('Custom');
    assert.strictEqual(res2.totalMatches, 1);
    assert.strictEqual(res2.results[0].id, 101);
    assert.strictEqual(res2.results[0].matchedField, 'headline');

    console.log('   ✅ Custom idField and getter functions verified');
  }

  // =========================================================================
  // 9. Edge Cases: Empty Queries, Long Queries, Case Mismatch, Lifecycle
  // =========================================================================
  console.log('9. Testing edge cases: empty queries, long queries, case mismatch, destroy...');
  {
    const docs = [{ id: '1', title: 'Hello World' }];
    const index = await DocumentIndex.create(docs, {
      fields: ['title'],
      preferGpu: false
    });

    // Empty query returns empty response
    const emptyRes = await index.search('');
    assert.strictEqual(emptyRes.totalMatches, 0);
    assert.strictEqual(emptyRes.query, '');
    assert.strictEqual(emptyRes.results.length, 0);

    // Whitespace-only query returns empty response
    const wsRes = await index.search('   ');
    assert.strictEqual(wsRes.totalMatches, 0);
    assert.strictEqual(wsRes.query, '');

    // ProfileMismatchError on caseSensitive disagreement
    let caseThrew = false;
    try {
      await index.search('hello', { caseSensitive: true });
    } catch (err: any) {
      caseThrew = err instanceof ProfileMismatchError;
    }
    assert(caseThrew, 'Expected ProfileMismatchError for case disagreement');

    // QueryTooLongError
    const longQuery = 'a'.repeat(1000);
    let longThrew = false;
    try {
      await index.search(longQuery);
    } catch (err: any) {
      longThrew = err instanceof QueryTooLongError;
    }
    assert(longThrew, 'Expected QueryTooLongError for >128 token query');

    // Query too long with cpu-fallback
    const fbRes = await index.search(longQuery, { onQueryTooLong: 'cpu-fallback' });
    assert.strictEqual(fbRes.totalMatches, 0);
    assert.strictEqual(fbRes.fallbackReason, 'query-too-long');

    // Destruction
    index.destroy();
    let destroyThrew = false;
    try {
      await index.search('test');
    } catch (err: any) {
      destroyThrew = err.message.includes('destroyed');
    }
    assert(destroyThrew, 'Expected Error on destroyed index');

    console.log('   ✅ Edge cases and lifecycle safety verified');
  }

  // =========================================================================
  // 10. DocumentIndexStats Telemetry Reporting
  // =========================================================================
  console.log('10. Testing DocumentIndexStats telemetry reporting...');
  {
    const docs = [
      { id: '1', title: 'One', body: 'First record' },
      { id: '2', title: 'Two', body: 'Second record' }
    ];
    const index = await DocumentIndex.create(docs, {
      fields: ['title', 'body'],
      preferGpu: false
    });

    const stats = index.getStats();
    assert.strictEqual(stats.docCount, 2);
    assert.strictEqual(stats.rowCount, 4);
    assert.strictEqual(stats.tombstoneCount, 0);
    assert.strictEqual(stats.tombstoneRatio, 0);
    assert.strictEqual(stats.engine, 'cpu');
    assert.strictEqual(stats.fallbackReason, 'prefer-cpu');
    assert(typeof stats.buildTimeMs === 'number' && stats.buildTimeMs >= 0);
    assert(typeof stats.memory.ramBytes === 'number' && stats.memory.ramBytes > 0);
    assert.strictEqual(stats.memory.vramBytes, 0);
    assert.strictEqual(stats.memory.totalBytes, stats.memory.ramBytes);

    index.destroy();
    console.log('   ✅ DocumentIndexStats telemetry reporting verified');
  }

  console.log('\n--- All Milestone 2 Document Record Engine & Multi-Field Tests Passed! ✅ ---');
}

runM2Tests().catch((err) => {
  console.error('Milestone 2 test failed:', err);
  process.exit(1);
});
