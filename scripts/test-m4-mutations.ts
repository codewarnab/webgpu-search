import assert from 'node:assert';
import {
  DocumentIndex,
  computeClampedHeadroomBytes,
  DuplicateIdError,
  DocumentNotFoundError,
  type DocumentIndexOptions,
  type MutationBatch
} from '../packages/webgpu-search/src/index';
import { createMockAdapter } from 'vgpu/mock';

interface ItemDoc {
  id: string;
  title: string;
  content: string;
  tags?: string[];
}

async function runM4Tests() {
  console.log('--- Running Milestone 4: Batched Dynamic Mutations & Memory Management Tests ---');

  // =========================================================================
  // 1. Dynamic Document Additions (add)
  // =========================================================================
  console.log('1. Testing dynamic document additions (add)...');
  {
    const initial: ItemDoc[] = [
      { id: '1', title: 'TypeScript Handbook', content: 'Static typing for JavaScript' },
      { id: '2', title: 'WebGPU Specification', content: 'Hardware acceleration for the web' }
    ];

    const index = await DocumentIndex.create(initial, {
      fields: [
        { name: 'title', weight: 2.0 },
        { name: 'content', weight: 1.0 }
      ],
      preferGpu: false
    });

    // Initial check
    const initialStats = index.getStats();
    assert.strictEqual(initialStats.docCount, 2);
    assert.strictEqual(initialStats.mutationEpoch, 0);

    // Single document add
    const addRes1 = await index.add({
      id: '3',
      title: 'Rust Compute Shaders',
      content: 'Writing WGSL and spirv shaders'
    });

    assert.strictEqual(addRes1.added, 1);
    assert.strictEqual(addRes1.updated, 0);
    assert.strictEqual(addRes1.removed, 0);
    assert.strictEqual(addRes1.mutationEpoch, 1);
    assert.strictEqual(addRes1.compacted, false);
    assert(addRes1.durationMs >= 0);

    // Read-your-writes check
    const searchRes1 = await index.search('Rust');
    assert.strictEqual(searchRes1.totalMatches, 1);
    assert.strictEqual(searchRes1.results[0].id, '3');
    assert.strictEqual(searchRes1.results[0].matchedField, 'title');

    // Array addition
    const addRes2 = await index.add([
      { id: '4', title: 'Go Concurrency', content: 'Goroutines and channels' },
      { id: '5', title: 'Python Asyncio', content: 'Event loop and coroutines' }
    ]);

    assert.strictEqual(addRes2.added, 2);
    assert.strictEqual(addRes2.mutationEpoch, 2);

    const searchRes2 = await index.search('Concurrency');
    assert.strictEqual(searchRes2.totalMatches, 1);
    assert.strictEqual(searchRes2.results[0].id, '4');

    const stats = index.getStats();
    assert.strictEqual(stats.docCount, 5);
    assert.strictEqual(stats.mutationEpoch, 2);
    assert.strictEqual(stats.lastMutationTimeMs, addRes2.durationMs);

    index.destroy();
    console.log('   ✅ Dynamic document additions (single and batch) verified');
  }

  // =========================================================================
  // 2. Dynamic Document Updates (update)
  // =========================================================================
  console.log('2. Testing dynamic document updates (update)...');
  {
    const initial: ItemDoc[] = [
      { id: 'doc-1', title: 'Original Title', content: 'Database storage indexing' },
      { id: 'doc-2', title: 'Secondary Item', content: 'Networking protocols' }
    ];

    const index = await DocumentIndex.create(initial, {
      fields: ['title', 'content'],
      preferGpu: false
    });

    // Update doc-1
    const updateRes = await index.update({
      id: 'doc-1',
      title: 'Updated Modern Title',
      content: 'Completely replaced content about WebGPU shaders'
    });

    assert.strictEqual(updateRes.added, 0);
    assert.strictEqual(updateRes.updated, 1);
    assert.strictEqual(updateRes.removed, 0);
    assert.strictEqual(updateRes.mutationEpoch, 1);

    // Old content must NO LONGER match
    const oldSearch = await index.search('Database');
    assert.strictEqual(oldSearch.totalMatches, 0, 'Old content must not match after update');

    // New content must match
    const newSearch = await index.search('Shaders');
    assert.strictEqual(newSearch.totalMatches, 1);
    assert.strictEqual(newSearch.results[0].id, 'doc-1');
    assert.strictEqual(newSearch.results[0].doc.title, 'Updated Modern Title');

    // Updating a non-existent document must throw DocumentNotFoundError
    let threwNotFound = false;
    try {
      await index.update({ id: 'non-existent', title: 'X', content: 'Y' });
    } catch (err) {
      assert(err instanceof DocumentNotFoundError);
      assert.strictEqual(err.id, 'non-existent');
      threwNotFound = true;
    }
    assert.strictEqual(threwNotFound, true, 'Updating non-existent document must throw DocumentNotFoundError');

    index.destroy();
    console.log('   ✅ Dynamic document updates and DocumentNotFoundError verified');
  }

  // =========================================================================
  // 3. Dynamic Document Deletions (remove)
  // =========================================================================
  console.log('3. Testing dynamic document deletions (remove)...');
  {
    const initial: ItemDoc[] = [
      { id: '1', title: 'Doc One', content: 'First entry' },
      { id: '2', title: 'Doc Two', content: 'Second entry' },
      { id: '3', title: 'Doc Three', content: 'Third entry' }
    ];

    const index = await DocumentIndex.create(initial, {
      fields: ['title', 'content'],
      preferGpu: false
    });

    // Remove single ID
    const removeRes1 = await index.remove('2');
    assert.strictEqual(removeRes1.removed, 1);
    assert.strictEqual(removeRes1.mutationEpoch, 1);

    // Doc 2 must no longer be found
    const searchRes = await index.search('Two');
    assert.strictEqual(searchRes.totalMatches, 0);

    // Other docs must still be searchable
    const searchOne = await index.search('One');
    assert.strictEqual(searchOne.totalMatches, 1);
    assert.strictEqual(searchOne.results[0].id, '1');

    // Removing non-existent ID is a safe no-op
    const removeRes2 = await index.remove('non-existent');
    assert.strictEqual(removeRes2.removed, 0);

    // Remove multiple IDs
    const removeRes3 = await index.remove(['1', '3']);
    assert.strictEqual(removeRes3.removed, 2);

    const stats = index.getStats();
    assert.strictEqual(stats.docCount, 0);

    // Search on empty index returns 0 hits cleanly
    const emptySearch = await index.search('One');
    assert.strictEqual(emptySearch.totalMatches, 0);
    assert.strictEqual(emptySearch.results.length, 0);

    index.destroy();
    console.log('   ✅ Dynamic document deletions (remove) verified');
  }

  // =========================================================================
  // 4. Batched Mutations (applyBatch) and Deterministic Ordering
  // =========================================================================
  console.log('4. Testing batched mutations (applyBatch) with deterministic sequence...');
  {
    const initial: ItemDoc[] = [
      { id: '1', title: 'Alpha', content: 'First' },
      { id: '2', title: 'Beta', content: 'Second' },
      { id: '3', title: 'Gamma', content: 'Third' }
    ];

    const index = await DocumentIndex.create(initial, {
      fields: ['title', 'content'],
      preferGpu: false
    });

    // Combined batch: remove '1', update '2', add '4'
    const batch: MutationBatch<ItemDoc> = {
      remove: ['1'],
      update: [{ id: '2', title: 'Beta Updated', content: 'Second Modified' }],
      add: [{ id: '4', title: 'Delta', content: 'Fourth' }]
    };

    const res = await index.applyBatch(batch);
    assert.strictEqual(res.removed, 1);
    assert.strictEqual(res.updated, 1);
    assert.strictEqual(res.added, 1);
    assert.strictEqual(res.mutationEpoch, 1);

    // Verify remove '1'
    assert.strictEqual((await index.search('Alpha')).totalMatches, 0);
    // Verify update '2'
    const s2 = await index.search('Modified');
    assert.strictEqual(s2.totalMatches, 1);
    assert.strictEqual(s2.results[0].id, '2');
    // Verify add '4'
    const s4 = await index.search('Delta');
    assert.strictEqual(s4.totalMatches, 1);
    assert.strictEqual(s4.results[0].id, '4');

    // Deterministic ordering: remove runs before add.
    // Re-creating the same ID within a single batch (remove '3' then add '3')
    const recreateBatch: MutationBatch<ItemDoc> = {
      remove: ['3'],
      add: [{ id: '3', title: 'Gamma Recreated', content: 'Third Reborn' }]
    };
    const recreateRes = await index.applyBatch(recreateBatch);
    assert.strictEqual(recreateRes.removed, 1);
    assert.strictEqual(recreateRes.added, 1);

    const s3 = await index.search('Reborn');
    assert.strictEqual(s3.totalMatches, 1);
    assert.strictEqual(s3.results[0].id, '3');

    // Removing and then updating the same ID within the same batch must fail,
    // because remove runs before update.
    let threwRemoveThenUpdate = false;
    try {
      await index.applyBatch({
        remove: ['4'],
        update: [{ id: '4', title: 'Delta Dead', content: 'Dead' }]
      });
    } catch (err) {
      assert(err instanceof DocumentNotFoundError);
      threwRemoveThenUpdate = true;
    }
    assert.strictEqual(threwRemoveThenUpdate, true, 'Remove then update same ID must throw DocumentNotFoundError');

    index.destroy();
    console.log('   ✅ Batched mutations execution sequence (remove -> update -> add) verified');
  }

  // =========================================================================
  // 5. Upsert Semantics
  // =========================================================================
  console.log('5. Testing upsert semantics...');
  {
    const initial: ItemDoc[] = [
      { id: 'user-1', title: 'Alice Smith', content: 'Engineer' }
    ];

    const index = await DocumentIndex.create(initial, {
      fields: ['title', 'content'],
      preferGpu: false
    });

    // Calling add with existing ID without upsert must throw DuplicateIdError
    let threwDup = false;
    try {
      await index.add({ id: 'user-1', title: 'Alice Brown', content: 'Designer' });
    } catch (err) {
      assert(err instanceof DuplicateIdError);
      assert.strictEqual(err.id, 'user-1');
      threwDup = true;
    }
    assert.strictEqual(threwDup, true, 'add() without upsert on existing ID must throw DuplicateIdError');

    // Calling add with upsert: true on existing ID updates the document
    const upsertRes1 = await index.add(
      { id: 'user-1', title: 'Alice Brown', content: 'Senior Designer' },
      { upsert: true }
    );
    assert.strictEqual(upsertRes1.added, 0);
    assert.strictEqual(upsertRes1.updated, 1);

    const s1 = await index.search('Designer');
    assert.strictEqual(s1.totalMatches, 1);
    assert.strictEqual(s1.results[0].doc.title, 'Alice Brown');

    // Calling add with upsert: true on NEW ID adds the document
    const upsertRes2 = await index.add(
      { id: 'user-2', title: 'Bob Jones', content: 'Product Manager' },
      { upsert: true }
    );
    assert.strictEqual(upsertRes2.added, 1);
    assert.strictEqual(upsertRes2.updated, 0);

    const s2 = await index.search('Bob');
    assert.strictEqual(s2.totalMatches, 1);

    index.destroy();
    console.log('   ✅ Upsert semantics verified');
  }

  // =========================================================================
  // 6. Two-Phase Validation & Atomic Guarantees
  // =========================================================================
  console.log('6. Testing two-phase validation & atomic failure guarantees...');
  {
    const initial: ItemDoc[] = [
      { id: 'valid-1', title: 'One', content: 'Alpha' },
      { id: 'valid-2', title: 'Two', content: 'Beta' }
    ];

    const index = await DocumentIndex.create(initial, {
      fields: ['title', 'content'],
      preferGpu: false
    });

    // Batch contains 1 valid add, but also duplicate IDs within add
    let threwBatch = false;
    try {
      await index.applyBatch({
        add: [
          { id: 'valid-3', title: 'Three', content: 'Gamma' },
          { id: 'dup-id', title: 'Four', content: 'Delta' },
          { id: 'dup-id', title: 'Five', content: 'Epsilon' }
        ]
      });
    } catch (err) {
      assert(err instanceof DuplicateIdError);
      threwBatch = true;
    }
    assert.strictEqual(threwBatch, true);

    // State MUST NOT have changed: valid-3 must not be present
    assert.strictEqual((await index.search('Gamma')).totalMatches, 0);
    const stats = index.getStats();
    assert.strictEqual(stats.docCount, 2, 'docCount must be unchanged on failed validation');
    assert.strictEqual(stats.mutationEpoch, 0, 'mutationEpoch must be unchanged on failed validation');

    // Batch with invalid document ID type
    let threwIdType = false;
    try {
      await index.applyBatch({
        add: [{ id: '' as any, title: 'Blank', content: 'Test' }]
      });
    } catch (err) {
      assert(err instanceof TypeError);
      threwIdType = true;
    }
    assert.strictEqual(threwIdType, true, 'Empty string ID must throw TypeError');

    index.destroy();
    console.log('   ✅ Two-phase atomic validation guarantees verified');
  }

  // =========================================================================
  // 7. Cumulative Offsets Preservation (No In-Place Offset Shifting)
  // =========================================================================
  console.log('7. Testing cumulative offsets preservation & tombstone filtering...');
  {
    // Create 6 documents
    const items: ItemDoc[] = [
      { id: 'd0', title: 'Apples', content: 'Red fruit' },
      { id: 'd1', title: 'Bananas', content: 'Yellow fruit' },
      { id: 'd2', title: 'Cherries', content: 'Small red fruit' },
      { id: 'd3', title: 'Dates', content: 'Sweet brown fruit' },
      { id: 'd4', title: 'Elderberries', content: 'Dark purple fruit' },
      { id: 'd5', title: 'Figs', content: 'Sweet green fruit' }
    ];

    const index = await DocumentIndex.create(items, {
      fields: ['title', 'content'],
      preferGpu: false
    });

    // Delete intermediate documents: d1 and d3
    await index.remove(['d1', 'd3']);

    // Check d0 (preceding) still matches
    const s0 = await index.search('Apples');
    assert.strictEqual(s0.totalMatches, 1);
    assert.strictEqual(s0.results[0].id, 'd0');

    // Check d2 (intermediate between deleted d1 and d3) matches accurately
    const s2 = await index.search('Cherries');
    assert.strictEqual(s2.totalMatches, 1);
    assert.strictEqual(s2.results[0].id, 'd2');

    // Check d4 and d5 (succeeding) match accurately without shifted offsets
    const s4 = await index.search('Elderberries');
    assert.strictEqual(s4.totalMatches, 1);
    assert.strictEqual(s4.results[0].id, 'd4');

    const s5 = await index.search('Figs');
    assert.strictEqual(s5.totalMatches, 1);
    assert.strictEqual(s5.results[0].id, 'd5');

    // Deleted items do not match
    assert.strictEqual((await index.search('Bananas')).totalMatches, 0);
    assert.strictEqual((await index.search('Dates')).totalMatches, 0);

    index.destroy();
    console.log('   ✅ Cumulative offsets preservation and tombstone filtering verified');
  }

  // =========================================================================
  // 8. CPU-Driven Compaction Pipeline (>= 25% Tombstones)
  // =========================================================================
  console.log('8. Testing CPU-driven compaction pipeline (>= 25% tombstones)...');
  {
    // 10 documents, 2 fields = 20 rows
    const items: ItemDoc[] = Array.from({ length: 10 }, (_, i) => ({
      id: `doc-${i}`,
      title: `Document Number ${i}`,
      content: `Content entry index ${i} for testing`
    }));

    const index = await DocumentIndex.create(items, {
      fields: [
        { name: 'title', weight: 2.0 },
        { name: 'content', weight: 1.0 }
      ],
      preferGpu: false
    });

    assert.strictEqual(index.getStats().tombstoneCount, 0);
    assert.strictEqual(index.getStats().tombstoneRatio, 0);

    // Delete 1 doc: 2 rows tombstoned out of 20 = 10% < 25% -> compacted: false
    const r1 = await index.remove('doc-0');
    assert.strictEqual(r1.compacted, false);
    assert.strictEqual(index.getStats().tombstoneCount, 2);

    // Delete 2 more docs (doc-1, doc-2):
    // 4 more rows tombstoned -> 6 total tombstones out of 20 = 30% >= 25% -> triggers compaction!
    const r2 = await index.remove(['doc-1', 'doc-2']);
    assert.strictEqual(r2.compacted, true, 'Deletions >= 25% must trigger CPU compaction');

    const compactedStats = index.getStats();
    assert.strictEqual(compactedStats.docCount, 7);
    assert.strictEqual(compactedStats.tombstoneCount, 0, 'Compaction must sweep tombstones to 0');
    assert.strictEqual(compactedStats.tombstoneRatio, 0);
    assert.strictEqual(compactedStats.rowCount, 14, 'Compacted rowCount should be 7 docs * 2 fields');

    // Remaining documents must continue to search cleanly
    for (let i = 3; i < 10; i++) {
      const res = await index.search(`Document Number ${i}`);
      assert.strictEqual(res.totalMatches, 1);
      assert.strictEqual(res.results[0].id, `doc-${i}`);
    }

    index.destroy();
    console.log('   ✅ CPU-driven compaction pipeline (>= 25% tombstones) verified');
  }

  // =========================================================================
  // 9. Clamped Dynamic Headroom Allocation
  // =========================================================================
  console.log('9. Testing clamped dynamic headroom allocation...');
  {
    // Unit test computeClampedHeadroomBytes directly
    const r1 = computeClampedHeadroomBytes(0);
    assert.strictEqual(r1, 16, 'Zero/negative bytes should clamp to minimum 16 B');

    const r2 = computeClampedHeadroomBytes(100, { growthFactor: 1.5 });
    assert.strictEqual(r2, 152, '100 * 1.5 = 150 -> 4-byte aligned = 152');

    // Mock device with a tight 1 MB storage binding limit
    const tightMockDevice = {
      limits: {
        maxBufferSize: 1048576, // 1 MB
        maxStorageBufferBindingSize: 1048576 // 1 MB
      }
    } as any;

    // Headroom that fits under 1 MB
    const r3 = computeClampedHeadroomBytes(500_000, { growthFactor: 1.5, device: tightMockDevice });
    assert.strictEqual(r3, 750_000, 'Fits within limit with growth factor');

    // Headroom overflow: 800 KB * 1.5 = 1.2 MB > 1 MB, but 800 KB <= 1 MB.
    // Must allocate exact 800 KB instead of failing!
    const r4 = computeClampedHeadroomBytes(800_000, { growthFactor: 1.5, device: tightMockDevice });
    assert.strictEqual(r4, 800_000, 'When headroom overflows limit but required fits, allocate exact required bytes');

    // Test DocumentIndex with initialCapacity and growthFactor options
    const index = await DocumentIndex.create<ItemDoc>(
      [{ id: '1', title: 'Alpha', content: 'A' }],
      {
        fields: ['title', 'content'],
        initialCapacity: 50,
        growthFactor: 2.0,
        preferGpu: false
      }
    );

    // Adding documents that fit within headroom
    const addRes = await index.add({ id: '2', title: 'Beta', content: 'B' });
    assert.strictEqual(addRes.compacted, false);
    assert.strictEqual(index.getStats().docCount, 2);

    index.destroy();
    console.log('   ✅ Clamped dynamic headroom computation verified');
  }

  // =========================================================================
  // 10. Read-Your-Writes & Mutex Concurrency
  // =========================================================================
  console.log('10. Testing read-your-writes and concurrency mutex protection...');
  {
    const index = await DocumentIndex.create<ItemDoc>(
      [
        { id: '1', title: 'Concurrency First', content: 'Thread safety' },
        { id: '2', title: 'Concurrency Second', content: 'Locks and mutex' }
      ],
      {
        fields: ['title', 'content'],
        preferGpu: false
      }
    );

    // Sequential operations observe immediate read-your-writes
    await index.add({ id: '3', title: 'Concurrency Third', content: 'Worker channels' });
    const s1 = await index.search('Third');
    assert.strictEqual(s1.totalMatches, 1);
    assert.strictEqual(s1.results[0].id, '3');

    await index.update({ id: '3', title: 'Concurrency Third Modified', content: 'Channels' });
    const s2 = await index.search('Modified');
    assert.strictEqual(s2.totalMatches, 1);

    await index.remove('3');
    const s3 = await index.search('Third');
    assert.strictEqual(s3.totalMatches, 0);

    index.destroy();
    console.log('   ✅ Read-your-writes and mutex consistency verified');
  }

  // =========================================================================
  // 11. WebGPU vs CPU Parity Across Mutations
  // =========================================================================
  console.log('11. Testing WebGPU vs CPU parity across all mutation operations...');
  {
    const adapter = createMockAdapter({
      features: ['timestamp-query'] as any
    });
    const mockDeviceWrapper = await adapter.requestDevice();
    const mockDevice = (mockDeviceWrapper as any).gpu ?? mockDeviceWrapper;

    const corpus: ItemDoc[] = [
      { id: 'p1', title: 'Architecture of Computers', content: 'Processors memory bus caches' },
      { id: 'p2', title: 'Operating Systems Internal', content: 'Virtual memory page tables scheduling' },
      { id: 'p3', title: 'Network Protocols Design', content: 'TCP IP routing packets flow' }
    ];

    const fields = [
      { name: 'title', weight: 2.0 },
      { name: 'content', weight: 1.0 }
    ];

    const gpuIndex = await DocumentIndex.create(corpus, {
      fields,
      device: mockDevice,
      preferGpu: true
    });

    const cpuIndex = await DocumentIndex.create(corpus, {
      fields,
      preferGpu: false
    });

    assert.strictEqual(gpuIndex.getStats().engine, 'webgpu');
    assert.strictEqual(cpuIndex.getStats().engine, 'cpu');
    assert.strictEqual(gpuIndex.getStats().docCount, cpuIndex.getStats().docCount);
    assert.strictEqual(gpuIndex.getStats().rowCount, cpuIndex.getStats().rowCount);

    // 1. Initial search on CPU & WebGPU pipeline
    const q1 = 'memory';
    const gpuR1 = await gpuIndex.search(q1);
    const cpuR1 = await cpuIndex.search(q1);
    assert.strictEqual(gpuR1.engine, 'webgpu');
    assert.strictEqual(cpuR1.engine, 'cpu');
    assert.strictEqual(cpuR1.totalMatches, 2);

    // 2. Add parity
    const newDoc: ItemDoc = {
      id: 'p4',
      title: 'Distributed Systems Memory',
      content: 'Shared memory and consistency models'
    };
    const gpuAddRes = await gpuIndex.add(newDoc);
    const cpuAddRes = await cpuIndex.add(newDoc);
    assert.strictEqual(gpuAddRes.added, 1);
    assert.strictEqual(cpuAddRes.added, 1);
    assert.strictEqual(gpuIndex.getStats().docCount, cpuIndex.getStats().docCount);
    assert.strictEqual(gpuIndex.getStats().rowCount, cpuIndex.getStats().rowCount);

    const cpuR2 = await cpuIndex.search(q1);
    assert.strictEqual(cpuR2.totalMatches, 3);
    const gpuR2 = await gpuIndex.search(q1);
    assert.strictEqual(gpuR2.engine, 'webgpu');

    // 3. Update parity
    const updatedDoc: ItemDoc = {
      id: 'p1',
      title: 'Quantum Architecture',
      content: 'Qubits superposition entanglement'
    };
    const gpuUpRes = await gpuIndex.update(updatedDoc);
    const cpuUpRes = await cpuIndex.update(updatedDoc);
    assert.strictEqual(gpuUpRes.updated, 1);
    assert.strictEqual(cpuUpRes.updated, 1);
    assert.strictEqual(gpuIndex.getStats().tombstoneCount, cpuIndex.getStats().tombstoneCount);

    const qQuantum = 'Quantum';
    const cpuR3 = await cpuIndex.search(qQuantum);
    assert.strictEqual(cpuR3.totalMatches, 1);
    assert.strictEqual(cpuR3.results[0].id, 'p1');
    const gpuR3 = await gpuIndex.search(qQuantum);
    assert.strictEqual(gpuR3.engine, 'webgpu');

    // 4. Remove parity
    const gpuRemRes = await gpuIndex.remove('p2');
    const cpuRemRes = await cpuIndex.remove('p2');
    assert.strictEqual(gpuRemRes.removed, 1);
    assert.strictEqual(cpuRemRes.removed, 1);
    assert.strictEqual(gpuIndex.getStats().docCount, cpuIndex.getStats().docCount);

    const qVirtual = 'Virtual';
    assert.strictEqual((await cpuIndex.search(qVirtual)).totalMatches, 0);

    // 5. Compaction parity
    const burst = Array.from({ length: 6 }, (_, i) => ({
      id: `burst-${i}`,
      title: `Burst Title ${i}`,
      content: `Burst Content ${i}`
    }));
    await gpuIndex.add(burst);
    await cpuIndex.add(burst);

    const removeBurst = ['burst-0', 'burst-1', 'burst-2', 'burst-3'];
    const gpuCompRes = await gpuIndex.remove(removeBurst);
    const cpuCompRes = await cpuIndex.remove(removeBurst);
    assert.strictEqual(gpuCompRes.compacted, true);
    assert.strictEqual(cpuCompRes.compacted, true);
    assert.strictEqual(gpuIndex.getStats().docCount, cpuIndex.getStats().docCount);
    assert.strictEqual(gpuIndex.getStats().tombstoneCount, 0);
    assert.strictEqual(cpuIndex.getStats().tombstoneCount, 0);

    const cpuFinal = await cpuIndex.search('Burst Title');
    assert.strictEqual(cpuFinal.totalMatches, 2);

    gpuIndex.destroy();
    cpuIndex.destroy();
    console.log('   ✅ WebGPU vs CPU parity across all mutations verified 100%');
  }

  // =========================================================================
  // 12. Highlighting Integrity on Mutated Documents
  // =========================================================================
  console.log('12. Testing highlighting integrity on mutated documents...');
  {
    const index = await DocumentIndex.create<ItemDoc>(
      [{ id: '1', title: 'Initial Document', content: 'Basic text' }],
      {
        fields: ['title', 'content'],
        preferGpu: false
      }
    );

    // Add document with German Eszett and indentation
    await index.add({
      id: '2',
      title: '    function süßeTräume() {',
      content: 'Indented German code with ß expansion 😀'
    });

    const res = await index.search('süsse', {
      tag: 'mark'
    });

    assert.strictEqual(res.totalMatches, 1);
    const hit = res.results[0];
    assert.strictEqual(hit.id, '2');

    // Verify title highlight range accounts for leading indentation
    const ranges = hit.highlights?.title;
    assert(ranges && ranges.length > 0, 'Highlights should be present');
    assert.strictEqual(ranges[0].start, 13); // '    function '.length = 13
    assert.strictEqual(ranges[0].end, 17); // 'süße' span is 4 chars -> 13 + 4 = 17

    // Verify tag injection
    const html = hit.highlightedText?.title;
    assert(html && html.includes('<mark>'), 'HTML output should contain <mark> tag');

    index.destroy();
    console.log('   ✅ Highlighting integrity on mutated documents verified');
  }

  console.log('\n--- All Milestone 4 Batched Dynamic Mutations Tests Passed! ✅ ---');
}

runM4Tests().catch((err) => {
  console.error('Milestone 4 test failed:', err);
  process.exit(1);
});
