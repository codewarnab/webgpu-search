import assert from 'node:assert';
import { execSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import {
  DocumentIndex,
  encodeSnapshot,
  restoreSnapshot,
  decodeSnapshotHeader,
  SNAPSHOT_MAGIC,
  SNAPSHOT_FORMAT_VERSION,
  SNAPSHOT_HEADER_BYTES,
  type DocumentId,
  type DocumentIndexStats,
  type HighlightRange
} from '../packages/webgpu-search/src/index';

import { CORE_FILES, generateMonacoRecords } from '../apps/monaco-palette/src/sample-data';
import { PaletteEngine } from '../apps/monaco-palette/src/palette-engine';
import type { MonacoFileRecord } from '../apps/monaco-palette/src/types';

import { generateStructuredLogs } from '../apps/log-viewer/src/log-generator';
import { LogEngine } from '../apps/log-viewer/src/log-engine';
import type { StructuredLogRecord } from '../apps/log-viewer/src/types';

import { CORE_DOCS, generateDocsRecords } from '../apps/docs-search/src/docs-data';
import { DocsEngine } from '../apps/docs-search/src/docs-engine';
import type { DocPageRecord } from '../apps/docs-search/src/types';

async function runProofAppTests() {
  console.log('--- Running  Proof Applications & Verification Tests ---');

  const rootDir = path.resolve(__dirname, '..');
  const monacoDir = path.join(rootDir, 'apps/monaco-palette');
  const logViewerDir = path.join(rootDir, 'apps/log-viewer');
  const docsDir = path.join(rootDir, 'apps/docs-search');

  // =========================================================================
  // 1. Proof Applications Monorepo Structure & File Contracts
  // =========================================================================
  console.log('1. Verifying Proof Applications monorepo structure and file contracts...');
  {
    // Ensure dist builds exist on clean clones
    if (!fs.existsSync(path.join(monacoDir, 'dist/index.html')) || !fs.existsSync(path.join(logViewerDir, 'dist/index.html')) || !fs.existsSync(path.join(docsDir, 'dist/index.html'))) {
      console.log('   ℹ️ Building apps on-demand for clean-clone distribution verification...');
      execSync('bun run build', { cwd: rootDir, stdio: 'inherit' });
    }
    // A. apps/monaco-palette files
    const monacoPkg = JSON.parse(fs.readFileSync(path.join(monacoDir, 'package.json'), 'utf8'));
    assert.strictEqual(monacoPkg.name, 'monaco-palette');
    assert.strictEqual(monacoPkg.dependencies['webgpu-search'], 'workspace:*');
    assert(fs.existsSync(path.join(monacoDir, 'tsconfig.json')));
    assert(fs.existsSync(path.join(monacoDir, 'vite.config.ts')));
    assert(fs.existsSync(path.join(monacoDir, 'index.html')));
    assert(fs.existsSync(path.join(monacoDir, 'src/main.ts')));
    assert(fs.existsSync(path.join(monacoDir, 'src/palette-engine.ts')));
    assert(fs.existsSync(path.join(monacoDir, 'src/sample-data.ts')));
    assert(fs.existsSync(path.join(monacoDir, 'src/types.ts')));
    assert(fs.existsSync(path.join(monacoDir, 'src/style.css')));
    assert(fs.existsSync(path.join(monacoDir, 'src/worker.ts')));
    assert(fs.existsSync(path.join(monacoDir, 'dist/index.html')), 'monaco-palette dist/index.html must exist');

    // B. apps/log-viewer files
    const logPkg = JSON.parse(fs.readFileSync(path.join(logViewerDir, 'package.json'), 'utf8'));
    assert.strictEqual(logPkg.name, 'log-viewer');
    assert.strictEqual(logPkg.dependencies['webgpu-search'], 'workspace:*');
    assert(fs.existsSync(path.join(logViewerDir, 'tsconfig.json')));
    assert(fs.existsSync(path.join(logViewerDir, 'vite.config.ts')));
    assert(fs.existsSync(path.join(logViewerDir, 'index.html')));
    assert(fs.existsSync(path.join(logViewerDir, 'src/main.ts')));
    assert(fs.existsSync(path.join(logViewerDir, 'src/log-engine.ts')));
    assert(fs.existsSync(path.join(logViewerDir, 'src/log-generator.ts')));
    assert(fs.existsSync(path.join(logViewerDir, 'src/virtual-grid.ts')));
    assert(fs.existsSync(path.join(logViewerDir, 'src/types.ts')));
    assert(fs.existsSync(path.join(logViewerDir, 'src/style.css')));
    assert(fs.existsSync(path.join(logViewerDir, 'src/worker.ts')));
    assert(fs.existsSync(path.join(logViewerDir, 'dist/index.html')), 'log-viewer dist/index.html must exist');

    // C. apps/docs-search files
    const docsPkg = JSON.parse(fs.readFileSync(path.join(docsDir, 'package.json'), 'utf8'));
    assert.strictEqual(docsPkg.name, 'docs-search');
    assert.strictEqual(docsPkg.dependencies['webgpu-search'], 'workspace:*');
    assert(fs.existsSync(path.join(docsDir, 'tsconfig.json')));
    assert(fs.existsSync(path.join(docsDir, 'vite.config.ts')));
    assert(fs.existsSync(path.join(docsDir, 'index.html')));
    assert(fs.existsSync(path.join(docsDir, 'src/main.ts')));
    assert(fs.existsSync(path.join(docsDir, 'src/docs-engine.ts')));
    assert(fs.existsSync(path.join(docsDir, 'src/docs-data.ts')));
    assert(fs.existsSync(path.join(docsDir, 'src/types.ts')));
    assert(fs.existsSync(path.join(docsDir, 'src/style.css')));
    assert(fs.existsSync(path.join(docsDir, 'src/worker.ts')));
    assert(fs.existsSync(path.join(docsDir, 'dist/index.html')), 'docs-search dist/index.html must exist');

    console.log('   ✅ Proof applications workspace contracts and build outputs verified');
  }

  // =========================================================================
  // 2. Monaco Palette Multi-Field Search & Highlight Verification
  // =========================================================================
  console.log('2. Testing Monaco Palette multi-field search and Unicode highlight projection...');
  {
    const records = generateMonacoRecords(600);
    assert.strictEqual(records.length, 600);

    const palette = new PaletteEngine({
      useWorker: false, // test main thread directly in headless bun
      preferGpu: false
    });

    await palette.init(records);

    // A. Query targeting filename (weight 3.0)
    const filenameRes = await palette.search('document-index', { mode: 'fuzzy', highlight: true });
    assert(filenameRes.totalMatches >= 1);
    assert.strictEqual(filenameRes.results[0]!.doc.filename, 'document-index.ts');
    assert.strictEqual(filenameRes.results[0]!.matchedField, 'filename');
    assert(filenameRes.results[0]!.highlightedText?.filename?.includes('<mark>'));

    // B. Query targeting symbols (weight 2.0)
    const symbolRes = await palette.search('setupPipelinesAndBuffers', { mode: 'fuzzy', highlight: true });
    assert(symbolRes.totalMatches >= 1);
    assert.strictEqual(symbolRes.results[0]!.doc.filename, 'webgpu-engine.ts');
    assert.strictEqual(symbolRes.results[0]!.matchedField, 'symbols');
    assert(symbolRes.results[0]!.highlightedText?.symbols?.includes('<mark>'));

    // C. Query targeting shader WGSL
    const shaderRes = await palette.search('computeFuzzyScore', { mode: 'fuzzy', highlight: true });
    assert(shaderRes.totalMatches >= 1);
    assert.strictEqual(shaderRes.results[0]!.doc.filename, 'fuzzy.wgsl');
    assert.strictEqual(shaderRes.results[0]!.doc.type, 'shader');

    // D. Dynamic mutation tests: Add
    const customRecord: MonacoFileRecord = {
      id: 'f-custom-quantum',
      filename: 'quantum_shading.wgsl',
      path: 'src/shaders/quantum/quantum_shading.wgsl',
      symbols: 'QuantumPipeline, executeEntangledCompute, measureQubits',
      type: 'shader',
      language: 'wgsl',
      description: 'Quantum state computation shader executing on virtual qubits',
      sizeBytes: 4096,
      lineCount: 110
    };
    await palette.addRecord(customRecord);

    const matchAdded = await palette.search('measureQubits', { mode: 'fuzzy' });
    assert.strictEqual(matchAdded.totalMatches, 1);
    assert.strictEqual(matchAdded.results[0]!.doc.id, 'f-custom-quantum');

    // E. Dynamic mutation tests: Update
    customRecord.symbols = 'QuantumPipeline, executeEntangledCompute, measureQubits, teleportState';
    await palette.updateRecord(customRecord);

    const matchUpdated = await palette.search('teleportState', { mode: 'fuzzy' });
    assert.strictEqual(matchUpdated.totalMatches, 1);
    assert.strictEqual(matchUpdated.results[0]!.doc.id, 'f-custom-quantum');

    // F. Dynamic mutation tests: Remove
    await palette.removeRecord('f-custom-quantum');
    const matchRemoved = await palette.search('teleportState', { mode: 'fuzzy' });
    assert.strictEqual(matchRemoved.totalMatches, 0);

    // G. Batch Add 200 records
    const batchRecords = generateMonacoRecords(200).map((r, i) => ({
      ...r,
      id: `batch-test-${i}`
    }));
    await palette.batchAdd(batchRecords);
    const stats = await palette.getStats();
    assert(stats !== null);
    assert.strictEqual(stats.docCount, 800);
    assert.strictEqual(stats.rowCount, 3208);
    assert.strictEqual(stats.tombstoneCount, 8);

    // H. Security: XSS sanitization in highlightedText
    const xssRecord: MonacoFileRecord = {
      id: 'f-custom-xss',
      filename: 'xss_payload_unique_<img src=x onerror=alert(1)>.ts',
      path: 'src/<script>alert("xss")</script>/xss.ts',
      symbols: 'UniqueXssSymbol, attack<T>',
      type: 'class',
      language: 'typescript',
      description: 'Hostile test record with markup injection',
      sizeBytes: 100,
      lineCount: 10
    };
    await palette.addRecord(xssRecord);
    const xssSearch = await palette.search('xss_payload_unique', { mode: 'fuzzy', highlight: true });
    assert.strictEqual(xssSearch.totalMatches, 1);
    assert.strictEqual(xssSearch.results[0]!.doc.id, 'f-custom-xss');
    const hlFilename = xssSearch.results[0]!.highlightedText?.filename;
    assert(hlFilename, 'Highlighted filename must exist');
    assert(!hlFilename.includes('<img'), 'Raw HTML tag <img must be escaped');
    assert(hlFilename.includes('&lt;img') || hlFilename.includes('&gt;'), 'HTML entities must be escaped');

    palette.destroy();
    console.log('   ✅ Monaco Palette multi-field search, highlights, and mutations verified');
  }

  // =========================================================================
  // 3. Log Viewer Procedural Generation & 100k High-Throughput Indexing
  // =========================================================================
  console.log('3. Testing Log Viewer procedural log generation and 100k record indexing...');
  {
    // A. Generation quality checks
    const logs10k = generateStructuredLogs(10000);
    assert.strictEqual(logs10k.length, 10000);
    assert.strictEqual(logs10k[0]!.id, 'log-0000001');
    assert.strictEqual(logs10k[9999]!.id, 'log-0010000');

    // Ensure distinct levels are present
    const levels = new Set(logs10k.map((l) => l.level));
    assert(levels.has('ERROR'), 'Should contain ERROR logs');
    assert(levels.has('WARN'), 'Should contain WARN logs');
    assert(levels.has('INFO'), 'Should contain INFO logs');
    assert(levels.has('DEBUG'), 'Should contain DEBUG logs');

    // B. High-throughput LogEngine test (10k dataset)
    const logEngine = new LogEngine({
      useWorker: false,
      preferGpu: false
    });
    await logEngine.init(logs10k);

    const stats10k = await logEngine.getStats();
    assert(stats10k !== null);
    assert.strictEqual(stats10k.docCount, 10000);
    assert.strictEqual(stats10k.rowCount, 40000); // 4 fields

    // Search common token across logs
    const errorSearch = await logEngine.search('timeout', { mode: 'fuzzy', limit: 50 });
    assert(errorSearch.totalMatches >= 1);
    assert(errorSearch.searchDurationMs >= 0);
    assert(errorSearch.results.length > 0);
    assert(errorSearch.results[0]!.doc.message.toLowerCase().includes('timeout'));

    // Search specific service
    const authSearch = await logEngine.search('auth-service', { mode: 'substring', limit: 20 });
    assert(authSearch.totalMatches >= 1);
    assert.strictEqual(authSearch.results[0]!.doc.service, 'auth-service');

    // Test filter parameter
    const filteredSearch = await logEngine.search('WebGPU', {
      mode: 'fuzzy',
      levelFilter: 'ERROR',
      limit: 20
    });
    for (const res of filteredSearch.results) {
      assert.strictEqual(res.doc.level, 'ERROR');
    }

    // C. Streaming batch mutations
    const newLogs = generateStructuredLogs(100, 10001);
    const appendRes = await logEngine.appendLogs(newLogs);
    assert(appendRes !== null);
    assert.strictEqual(appendRes.added, 100);

    const statsAfterAppend = await logEngine.getStats();
    assert.strictEqual(statsAfterAppend?.docCount, 10100);

    const removeRes = await logEngine.removeOldLogs(200);
    assert(removeRes !== null);
    assert.strictEqual(removeRes.removed, 200);

    const statsAfterRemove = await logEngine.getStats();
    assert.strictEqual(statsAfterRemove?.docCount, 9900);
    assert(statsAfterRemove?.tombstoneCount! >= 200);

    // D. Security: XSS sanitization in log search results
    const hostileLog: StructuredLogRecord = {
      id: 'log-xss-999',
      timestamp: new Date().toISOString(),
      level: 'ERROR',
      service: '<script>alert(1)</script>',
      message: 'UniqueFailureQuantum <img src=x onerror=alert(2)> module',
      traceId: 'tr-" onfocus="alert(3)',
      latencyMs: 120
    };
    await logEngine.appendLogs([hostileLog]);
    const hostileSearch = await logEngine.search('UniqueFailureQuantum', { mode: 'fuzzy', highlight: true });
    assert.strictEqual(hostileSearch.totalMatches, 1);
    const hlMessage = hostileSearch.results[0]!.highlightedText?.message;
    assert(hlMessage, 'Highlighted log message must exist');
    assert(!hlMessage.includes('<img'), 'Raw HTML tag <img must be escaped in highlighted log message');
    assert(hlMessage.includes('&lt;img') || hlMessage.includes('&gt;'), 'HTML entities must be escaped in highlighted log message');

    logEngine.destroy();
    console.log('   ✅ Log Viewer generation, search, filtering, and streaming mutations verified');
  }

  // =========================================================================
  // 4. Versioned Snapshot Persistence (snapshot Binary Format Roundtrip)
  // =========================================================================
  console.log('4. Testing snapshot Little-Endian binary format serialization & restore roundtrip...');
  {
    const testLogs = generateStructuredLogs(5000);
    const docIndex = await DocumentIndex.create(testLogs, {
      fields: ['message', 'service', 'level', 'traceId'],
      filterFields: [{ name: 'level' }, { name: 'service' }],
      preferGpu: false
    });

    // A. Pre-serialization search baseline
    const baselineQuery = 'deadlock';
    const baselineSearch = await docIndex.search(baselineQuery, { mode: 'fuzzy', limit: 25 });

    // B. Serialize to snapshot binary buffer
    const snapshotBuffer = encodeSnapshot(docIndex);
    assert(snapshotBuffer.byteLength > 56, 'Snapshot must be larger than 56-byte snapshot header');

    // C. Inspect 56-byte Little-Endian Header (8-byte aligned)
    const header = decodeSnapshotHeader(snapshotBuffer);
    assert.strictEqual(header.magic, SNAPSHOT_MAGIC); // 0x55324434 ('snapshot')
    assert.strictEqual(header.formatVersion, SNAPSHOT_FORMAT_VERSION); // 4
    assert.strictEqual(SNAPSHOT_HEADER_BYTES, 56);
    assert.strictEqual(SNAPSHOT_HEADER_BYTES % 8, 0, 'snapshot header must be 8-byte aligned');
    assert.strictEqual(header.profileId, 'unicode-default');
    assert.strictEqual(header.unicodeVersion, '16.0.0');
    assert.strictEqual(header.scoringVersion, 'parity-v1');
    assert.strictEqual(header.docCount, 5000);
    assert.strictEqual(header.rowCount, 20000); // 4 fields * 5000
    assert(header.tokenCount > 0);
    assert(header.schemaByteLength > 0);
    assert(header.docsByteLength > 0);
    assert(header.columnarByteLength! > 0, 'filter-configured snapshot must carry a columnar segment');
    assert(typeof header.checksum === 'number');

    // D. Restore from binary buffer
    const restoredIndex = await restoreSnapshot<StructuredLogRecord>(snapshotBuffer, {
      options: { preferGpu: false }
    });

    const restoredStats = restoredIndex.getStats();
    assert.strictEqual(restoredStats.docCount, 5000);
    assert.strictEqual(restoredStats.rowCount, 20000);
    assert.strictEqual(restoredStats.formatVersion, 4);
    assert(typeof restoredStats.restoreTimeMs === 'number' && restoredStats.restoreTimeMs >= 0);

    // E. Verify identical ranking and score parity
    const restoredSearch = await restoredIndex.search(baselineQuery, { mode: 'fuzzy', limit: 25 });
    assert.strictEqual(restoredSearch.totalMatches, baselineSearch.totalMatches);
    assert.strictEqual(restoredSearch.results.length, baselineSearch.results.length);

    for (let i = 0; i < baselineSearch.results.length; i++) {
      const baseItem = baselineSearch.results[i]!;
      const restItem = restoredSearch.results[i]!;
      assert.strictEqual(restItem.id, baseItem.id);
      assert.strictEqual(restItem.score, baseItem.score);
      assert.strictEqual(restItem.matchedField, baseItem.matchedField);
    }

    // E2. Structured filter + facet parity across the restore boundary.
    const baselineFiltered = await docIndex.search(baselineQuery, {
      mode: 'fuzzy',
      limit: 25,
      filter: { level: 'ERROR' },
      facets: { byLevel: { type: 'terms', field: 'level', limit: 10 } }
    });
    const restoredFiltered = await restoredIndex.search(baselineQuery, {
      mode: 'fuzzy',
      limit: 25,
      filter: { level: 'ERROR' },
      facets: { byLevel: { type: 'terms', field: 'level', limit: 10 } }
    });
    assert.strictEqual(restoredFiltered.totalMatches, baselineFiltered.totalMatches);
    assert.deepStrictEqual(restoredFiltered.facets, baselineFiltered.facets);

    // F. Decoupled Document Storage Persistence (docsByteLength = 0)
    const decoupledBuffer = encodeSnapshot(docIndex, { decoupled: true });
    const decoupledHeader = decodeSnapshotHeader(decoupledBuffer);
    assert.strictEqual(decoupledHeader.docsByteLength, 0, 'Decoupled snapshot must have docsByteLength = 0');
    assert(decoupledBuffer.byteLength < snapshotBuffer.byteLength, 'Decoupled snapshot must be smaller');

    // Restore decoupled with external documents
    const restoredDecoupled = await restoreSnapshot<StructuredLogRecord>(decoupledBuffer, {
      documents: testLogs,
      options: { preferGpu: false }
    });
    const decoupledSearch = await restoredDecoupled.search(baselineQuery, { mode: 'fuzzy', limit: 25 });
    assert.strictEqual(decoupledSearch.totalMatches, baselineSearch.totalMatches);
    assert.strictEqual(decoupledSearch.results[0]!.doc.message, baselineSearch.results[0]!.doc.message);

    docIndex.destroy();
    restoredIndex.destroy();
    restoredDecoupled.destroy();
    console.log('   ✅ snapshot Little-Endian binary format serialization & decoupled restore verified');
  }

  // =========================================================================
  // 5. Proof-App Feature Integration (prefix, filters, facets, autocomplete)
  // =========================================================================
  console.log('5. Testing  proof-app feature integration (prefix + type filter + autocomplete)...');
  {
    // A. Monaco palette: prefix symbol search with structured type filter.
    const palette = new PaletteEngine({ useWorker: false, preferGpu: false });
    await palette.init(generateMonacoRecords(600));

    const prefixRes = await palette.search('compute', { mode: 'prefix', limit: 20 });
    assert(prefixRes.totalMatches >= 1, 'prefix search must match symbol records');

    const shaderOnly = await palette.search('compute', {
      mode: 'prefix',
      limit: 20,
      typeFilter: 'shader'
    });
    assert(shaderOnly.totalMatches >= 1, 'shader-filtered prefix search must match');
    for (const r of shaderOnly.results) {
      assert.strictEqual(r.doc.type, 'shader');
    }
    assert(shaderOnly.totalMatches <= prefixRes.totalMatches);

    // Type facets are populated on every palette search.
    assert(shaderOnly.facets?.byType?.type === 'terms', 'palette search must return type facets');
    assert((shaderOnly.facets.byType as any).isApproximate === false);

    // Autocomplete suggestions resolve.
    const autocompleteRes = await palette.autocomplete('comp', { mode: 'prefix', limit: 5 });
    assert(autocompleteRes.suggestions.length >= 1, 'autocomplete must return completions');

    const inlineAutocomplete = await palette.search('comp', {
      mode: 'prefix',
      limit: 5,
      autocomplete: { mode: 'prefix', limit: 5 }
    });
    assert((inlineAutocomplete.suggestions?.length ?? 0) >= 1, 'inline autocomplete must return completions');

    // Palette snapshot snapshot roundtrip preserves prefix + type filtering.
    {
      const snap = await palette.serializeSnapshot();
      const snapHeader = decodeSnapshotHeader(snap);
      assert.strictEqual(snapHeader.magic, SNAPSHOT_MAGIC);
      assert.strictEqual(snapHeader.formatVersion, SNAPSHOT_FORMAT_VERSION);
      const fresh = new PaletteEngine({ useWorker: false, preferGpu: false });
      await fresh.init([]);
      await fresh.restoreSnapshot(snap);
      const after = await fresh.search('compute', { mode: 'prefix', limit: 20, typeFilter: 'shader' });
      assert.strictEqual(after.totalMatches, shaderOnly.totalMatches);
      fresh.destroy();
    }
    palette.destroy();

    // B. Log viewer: structured severity filter + timestamp range + facets.
    const logEngine = new LogEngine({ useWorker: false, preferGpu: false });
    const logs = generateStructuredLogs(5000);
    await logEngine.init(logs);

    const errorOnly = await logEngine.search('timeout', {
      mode: 'fuzzy',
      limit: 20,
      levelFilter: 'ERROR'
    });
    assert(errorOnly.totalMatches >= 1);
    for (const r of errorOnly.results) {
      assert.strictEqual(r.doc.level, 'ERROR');
    }
    assert(errorOnly.facets?.byLevel?.type === 'terms', 'log search must return level facets');
    assert(errorOnly.facets?.byLatency?.type === 'range', 'log search must return latency range facets');

    // Timestamp range narrows strictly on seeded data (range filter is applied,
    // not ignored): per-result timestamps must fall inside [mid, newest).
    const newest = logs[logs.length - 1]!.timestamp;
    const mid = logs[Math.floor(logs.length / 2)]!.timestamp;
    const rangeNarrow = await logEngine.search('timeout', {
      mode: 'fuzzy',
      limit: 50,
      timestampRange: { from: mid, to: newest }
    });
    const rangeWide = await logEngine.search('timeout', { mode: 'fuzzy', limit: 50 });
    assert(rangeNarrow.totalMatches < rangeWide.totalMatches, 'timestamp range must strictly narrow matches');
    for (const r of rangeNarrow.results) {
      assert(r.doc.timestamp >= mid && r.doc.timestamp < newest, 'range result timestamp in-window');
    }

    // snapshot snapshot roundtrip preserves structured filtering (restore + search parity).
    const snap = await logEngine.serializeSnapshot();
    const snapHeader = decodeSnapshotHeader(snap);
    assert.strictEqual(snapHeader.magic, SNAPSHOT_MAGIC);
    assert.strictEqual(snapHeader.formatVersion, 4);
    const restoredSnap = await restoreSnapshot<StructuredLogRecord>(snap, {
      options: { preferGpu: false }
    });
    const snapBaseline = await logEngine.search('timeout', {
      mode: 'fuzzy',
      limit: 20,
      levelFilter: 'ERROR'
    });
    const snapAfter = await restoredSnap.search('timeout', {
      mode: 'fuzzy',
      limit: 20,
      filter: { level: 'ERROR' }
    });
    assert.strictEqual(snapAfter.totalMatches, snapBaseline.totalMatches);
    restoredSnap.destroy();
    logEngine.destroy();
    console.log('   ✅  proof-app feature integration verified');
  }

  // =========================================================================
  // 6. Worker-Boundary Proof-App Integration (filter/facets/autocomplete survive)
  // =========================================================================
  console.log('6. Testing worker-boundary proof-app integration (mock worker)...');
  {
    const { SearchWorkerClient, startSearchWorker } = await import('../packages/webgpu-search/src/index');
    const createMockWorkerScope = () => {
      const clientListeners: Array<(e: any) => void> = [];
      const workerListeners: Array<(e: any) => void> = [];
      const clientWorker = {
        postMessage(data: any) {
          queueMicrotask(() => { for (const l of workerListeners) l({ data }); });
        },
        addEventListener(event: string, listener: any) {
          if (event === 'message') clientListeners.push(listener);
        },
        removeEventListener(event: string, listener: any) {
          if (event === 'message') {
            const i = clientListeners.indexOf(listener);
            if (i >= 0) clientListeners.splice(i, 1);
          }
        },
        terminate() { clientListeners.length = 0; workerListeners.length = 0; }
      };
      const workerScope = {
        postMessage(data: any) {
          queueMicrotask(() => { for (const l of clientListeners) l({ data }); });
        },
        addEventListener(event: string, listener: any) {
          if (event === 'message') workerListeners.push(listener);
        },
        removeEventListener(event: string, listener: any) {
          if (event === 'message') {
            const i = workerListeners.indexOf(listener);
            if (i >= 0) workerListeners.splice(i, 1);
          }
        }
      };
      startSearchWorker(workerScope);
      return { clientWorker };
    };

    // Palette worker path: prefix + type filter + facets + autocomplete.
    {
      const { clientWorker } = createMockWorkerScope();
      const client = new SearchWorkerClient<MonacoFileRecord>({ worker: clientWorker as any });
      const records = generateMonacoRecords(600);
      await client.init(records, {
        idField: 'id',
        fields: [
          { name: 'filename', weight: 3.0 },
          { name: 'symbols', weight: 2.0 },
          { name: 'path', weight: 1.0 },
          { name: 'description', weight: 0.5 }
        ],
        filterFields: [{ name: 'type' }, { name: 'language' }],
        preferGpu: false
      });
      const res = await client.search('compute', {
        mode: 'prefix',
        limit: 20,
        filter: { type: 'shader' },
        facets: { byType: { type: 'terms', field: 'type', limit: 10 } },
        autocomplete: { mode: 'prefix', limit: 5 }
      } as any);
      assert(res.totalMatches >= 1);
      assert((res.facets as any)?.byType?.type === 'terms');
      assert((res.suggestions?.length ?? 0) >= 1);
      await client.destroy();
    }

    // Log worker path: level + latency filter + facets.
    {
      const { clientWorker } = createMockWorkerScope();
      const client = new SearchWorkerClient<StructuredLogRecord>({ worker: clientWorker as any });
      const logs = generateStructuredLogs(2000);
      await client.init(logs as any, {
        idField: 'id',
        fields: [
          { name: 'message', weight: 2.0 },
          { name: 'service', weight: 1.5 },
          { name: 'level', weight: 1.0 },
          { name: 'traceId', weight: 1.2 }
        ],
        filterFields: [{ name: 'level' }, { name: 'service' }, { name: 'timestamp' }, { name: 'latencyMs', type: 'number' }],
        preferGpu: false
      });
      const res = await client.search('timeout', {
        mode: 'fuzzy',
        limit: 20,
        filter: { level: 'ERROR', latencyMs: { gte: 300 } },
        facets: {
          byLevel: { type: 'terms', field: 'level', limit: 10 },
          byLatency: { type: 'range', field: 'latencyMs', ranges: [{ to: 50 }, { from: 50, to: 300 }, { from: 300 }] }
        }
      } as any);
      assert(res.totalMatches >= 1);
      assert((res.facets as any)?.byLevel?.type === 'terms');
      await client.destroy();
    }

    // Docs worker path: section + version filter + facets + autocomplete.
    {
      const { clientWorker } = createMockWorkerScope();
      const client = new SearchWorkerClient<DocPageRecord>({ worker: clientWorker as any });
      const docs = generateDocsRecords(400);
      await client.init(docs as any, {
        idField: 'id',
        fields: [
          { name: 'title', weight: 3.0 },
          { name: 'tags', weight: 2.0 },
          { name: 'section', weight: 1.5 },
          { name: 'content', weight: 1.0 }
        ],
        filterFields: [{ name: 'section' }, { name: 'version' }],
        preferGpu: false
      });
      const res = await client.search('snapshot', {
        mode: 'fuzzy',
        limit: 20,
        filter: { section: 'Storage' },
        facets: { bySection: { type: 'terms', field: 'section', limit: 10 } },
        autocomplete: { mode: 'prefix', limit: 5 }
      } as any);
      assert(res.totalMatches >= 1);
      for (const item of res.results) {
        assert.strictEqual((item.doc as any).section, 'Storage');
      }
      assert((res.facets as any)?.bySection?.type === 'terms');
      assert((res.suggestions?.length ?? 0) >= 1);
      await client.destroy();
    }
    console.log('   ✅ Worker-boundary proof-app integration verified');
  }

  // =========================================================================
  // 7. Cross-Platform Safety Audit (Zero Unguarded DOM Globals)
  // =========================================================================
  console.log('7. Auditing core library cross-platform safety (zero unguarded DOM globals)...');
  {
    const srcDir = path.join(rootDir, 'packages/webgpu-search/src');
    const tsFiles: string[] = [];

    function walkDir(current: string) {
      const entries = fs.readdirSync(current, { withFileTypes: true });
      for (const entry of entries) {
        const full = path.join(current, entry.name);
        if (entry.isDirectory()) {
          walkDir(full);
        } else if (entry.isFile() && entry.name.endsWith('.ts')) {
          tsFiles.push(full);
        }
      }
    }
    walkDir(srcDir);

    for (const file of tsFiles) {
      const content = fs.readFileSync(file, 'utf8');
      // Strip comments + string literals before matching so prose like
      // "sliding window" does not trip the bare-global scan (same approach
      // as scripts/check-parity-lint.ts and the search-modes DOM test).
      const noBlock = content.replace(/\/\*[\s\S]*?\*\//g, (m) => '\n'.repeat((m.match(/\n/g) || []).length));
      const code = noBlock
        .split('\n')
        .map((line) => {
          const idx = line.indexOf('//');
          return idx >= 0 ? line.slice(0, idx) : line;
        })
        .join('\n')
        .replace(/'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|`(?:[^`\\]|\\.)*`/g, "''");

      // Check for unguarded window references (bare global, not windowLength
      // / TypoWindow identifiers which carry no boundary after `window`).
      if (/\bwindow\b/.test(code)) {
        assert(
          code.includes("typeof window") || content.includes("typeof window"),
          `Unguarded window reference in ${path.relative(rootDir, file)}`
        );
      }

      // Check for unguarded document references
      if (/\bdocument\./.test(code)) {
        assert(
          code.includes("typeof document"),
          `Unguarded document reference in ${path.relative(rootDir, file)}`
        );
      }
    }

    // Check core library package.json dependencies only retains @leeoniya/ufuzzy for legacy CPU fallback
    const corePkg = JSON.parse(fs.readFileSync(path.join(rootDir, 'packages/webgpu-search/package.json'), 'utf8'));
    assert.deepStrictEqual(
      Object.keys(corePkg.dependencies || {}),
      ['@leeoniya/ufuzzy'],
      'packages/webgpu-search must retain only @leeoniya/ufuzzy for legacy CPU fallback and no other dependencies'
    );

    console.log('   ✅ Core library 100% portable: zero unguarded DOM globals & zero runtime dependencies');
  }

  // =========================================================================
  // 8. Docs-Search Offline Documentation Engine (third proof app)
  // =========================================================================
  console.log('8. Testing docs-search offline documentation engine...');
  {
    assert(CORE_DOCS.length >= 10, 'docs corpus must ship curated core pages');
    const docs = generateDocsRecords(400);
    assert.strictEqual(docs.length, 400);
    assert.strictEqual(docs[0]!.id, 'd-001');

    const sections = new Set(docs.map((d) => d.section));
    for (const s of ['Guide', 'API', 'Storage', 'Reliability', 'Reference']) {
      assert(sections.has(s as any), `docs corpus must contain ${s} section`);
    }

    const engine = new DocsEngine({ useWorker: false, preferGpu: false });
    await engine.init(docs);

    // A. Title-weighted query resolves the snapshot guide.
    const titleRes = await engine.search('snapshot', { mode: 'fuzzy', highlight: true });
    assert(titleRes.totalMatches >= 1);
    assert(titleRes.results.some((r) => r.doc.id === 'd-004'), 'snapshot query must resolve the v4 guide');
    assert.strictEqual(titleRes.results[0]!.matchedField, 'title');
    assert(titleRes.results[0]!.highlightedText?.title?.includes('<mark>'));

    // B. Tags-weighted query resolves the device-loss recovery page.
    const tagRes = await engine.search('rebuildGpu', { mode: 'fuzzy', highlight: true });
    assert(tagRes.totalMatches >= 1);
    assert(tagRes.results.some((r) => r.doc.id === 'd-009'), 'rebuildGpu query must resolve the recovery page');

    // C. Section pre-filter narrows strictly + facets populate.
    const storageOnly = await engine.search('snapshot', {
      mode: 'fuzzy',
      limit: 20,
      sectionFilter: 'Storage'
    });
    assert(storageOnly.totalMatches >= 1);
    for (const r of storageOnly.results) {
      assert.strictEqual(r.doc.section, 'Storage');
    }
    assert(storageOnly.totalMatches <= titleRes.totalMatches);
    assert(storageOnly.facets?.bySection?.type === 'terms', 'docs search must return section facets');
    assert((storageOnly.facets.bySection as any).isApproximate === false);

    // D. Version filter narrows strictly.
    const v10Only = await engine.search('offline', { mode: 'fuzzy', limit: 50, versionFilter: '1.0' });
    const unfiltered = await engine.search('offline', { mode: 'fuzzy', limit: 50 });
    assert(v10Only.totalMatches >= 1);
    assert(v10Only.totalMatches <= unfiltered.totalMatches);
    for (const r of v10Only.results) {
      assert.strictEqual(r.doc.version, '1.0');
    }

    // E. Prefix search + autocomplete suggestions resolve.
    const prefixRes = await engine.search('snap', { mode: 'prefix', limit: 20 });
    assert(prefixRes.totalMatches >= 1, 'prefix search must match doc titles');
    const autoRes = await engine.autocomplete('snap', { mode: 'prefix', limit: 5 });
    assert(autoRes.suggestions.length >= 1, 'autocomplete must return completions');
    const inlineAuto = await engine.search('snap', {
      mode: 'prefix',
      limit: 5,
      autocomplete: { mode: 'prefix', limit: 5 }
    });
    assert((inlineAuto.suggestions?.length ?? 0) >= 1, 'inline autocomplete must return completions');

    // F. Incremental mutations: add → update → remove.
    // Identity is pinned with substring (contiguous) matching; fuzzy asserts recall.
    const customDoc: DocPageRecord = {
      id: 'd-custom-quantum',
      path: 'docs/guides/quantum-search.md',
      title: 'Quantum entangled offline search',
      section: 'Guide',
      content: 'Experimental entangled index measuring teleportState coherence across shards',
      tags: 'quantum, teleportState, offline-docs',
      version: '1.0',
      readingMinutes: 6
    };
    await engine.addRecord(customDoc);
    const matchAdded = await engine.search('teleportState', { mode: 'substring' });
    assert.strictEqual(matchAdded.totalMatches, 1);
    assert.strictEqual(matchAdded.results[0]!.doc.id, 'd-custom-quantum');
    const matchAddedFuzzy = await engine.search('teleportState', { mode: 'fuzzy' });
    assert(matchAddedFuzzy.results.some((r) => r.doc.id === 'd-custom-quantum'), 'fuzzy recall must include the new page');
    assert.strictEqual(matchAddedFuzzy.results[0]!.doc.id, 'd-custom-quantum', 'exact tag hit must rank first');

    customDoc.content = 'Experimental entangled index measuring teleportState fidelity across shards';
    await engine.updateRecord(customDoc);
    const matchUpdated = await engine.search('fidelity', { mode: 'substring' });
    assert(matchUpdated.results.some((r) => r.doc.id === 'd-custom-quantum'));

    await engine.removeRecord('d-custom-quantum');
    const matchRemoved = await engine.search('teleportState', { mode: 'substring' });
    assert.strictEqual(matchRemoved.totalMatches, 0);

    // G. Query cancel: pre-aborted signal rejects AbortError (never fallback).
    {
      const controller = new AbortController();
      controller.abort();
      let abortName: string | null = null;
      try {
        await engine.search('snapshot', { mode: 'fuzzy', signal: controller.signal });
      } catch (err: any) {
        abortName = err?.name ?? null;
      }
      assert.strictEqual(abortName, 'AbortError', 'pre-aborted docs search must reject AbortError');
    }

    // H. Recovery hook: CPU-by-design rebuildGpu returns false without throwing.
    assert.strictEqual(await engine.rebuildGpu(), false);

    // I. snapshot snapshot roundtrip preserves section filtering.
    {
      const snap = await engine.serializeSnapshot();
      const snapHeader = decodeSnapshotHeader(snap);
      assert.strictEqual(snapHeader.magic, SNAPSHOT_MAGIC);
      assert.strictEqual(snapHeader.formatVersion, SNAPSHOT_FORMAT_VERSION);
      const fresh = new DocsEngine({ useWorker: false, preferGpu: false });
      await fresh.init([]);
      await fresh.restoreSnapshot(snap);
      const after = await fresh.search('snapshot', { mode: 'fuzzy', limit: 20, sectionFilter: 'Storage' });
      assert.strictEqual(after.totalMatches, storageOnly.totalMatches);
      assert.strictEqual(after.results[0]!.id, storageOnly.results[0]!.id);
      assert.strictEqual(after.results[0]!.score, storageOnly.results[0]!.score);
      fresh.destroy();
    }

    // J. restoreSnapshot rejects non-ArrayBuffer + oversize buffers fail-closed.
    await assert.rejects(engine.restoreSnapshot('nope' as any), TypeError);
    {
      const { MAX_SNAPSHOT_BYTES } = await import('../packages/webgpu-search/src/index');
      let rejectName: string | null = null;
      try {
        await engine.restoreSnapshot(new ArrayBuffer((MAX_SNAPSHOT_BYTES as number) + 8));
      } catch (err: any) {
        rejectName = err?.name ?? null;
      }
      assert.strictEqual(rejectName, 'IncompatibleIndexError', 'oversize snapshot must reject fail-closed');
    }

    // K. Security: XSS sanitization in highlightedText.
    const xssDoc: DocPageRecord = {
      id: 'd-custom-xss',
      path: 'docs/<script>alert("xss")</script>/xss.md',
      title: 'xss_payload_unique_<img src=x onerror=alert(1)> guide',
      section: 'Guide',
      content: 'Hostile test page with markup injection',
      tags: 'xss, UniqueXssTag',
      version: '1.0',
      readingMinutes: 1
    };
    await engine.addRecord(xssDoc);
    const xssSearch = await engine.search('xss_payload_unique', { mode: 'fuzzy', highlight: true });
    assert.strictEqual(xssSearch.totalMatches, 1);
    const hlTitle = xssSearch.results[0]!.highlightedText?.title;
    assert(hlTitle, 'Highlighted title must exist');
    assert(!hlTitle.includes('<img'), 'Raw HTML tag <img must be escaped');
    assert(hlTitle.includes('&lt;img') || hlTitle.includes('&gt;'), 'HTML entities must be escaped');

    // L. Idempotent destroy.
    engine.destroy();
    engine.destroy();
    console.log('   ✅ Docs-search offline documentation engine verified');
  }

  // =========================================================================
  // 9. Proof-App Teardown Lint (worker-first + abort-safe + destroy-on-unmount)
  // =========================================================================
  console.log('9. Linting proof-app teardown, cancel, and error-path patterns...');
  {
    const apps = ['monaco-palette', 'log-viewer', 'docs-search'];
    for (const app of apps) {
      const mainSrc = fs.readFileSync(path.join(rootDir, `apps/${app}/src/main.ts`), 'utf8');
      assert(mainSrc.includes('AbortController'), `${app}/main.ts must cancel in-flight queries via AbortController`);
      assert(mainSrc.includes('AbortError'), `${app}/main.ts must swallow AbortError on superseded queries`);
      assert(mainSrc.includes('engine.destroy()'), `${app}/main.ts must tear down the engine`);
      assert(
        mainSrc.includes("addEventListener('pagehide'") || mainSrc.includes('addEventListener("pagehide"'),
        `${app}/main.ts must destroy resources on pagehide`
      );
      assert(mainSrc.includes('console.error'), `${app}/main.ts must surface error paths`);
    }

    const engines: Array<[string, string]> = [
      ['monaco-palette', 'palette-engine.ts'],
      ['log-viewer', 'log-engine.ts'],
      ['docs-search', 'docs-engine.ts']
    ];
    for (const [app, file] of engines) {
      const engineSrc = fs.readFileSync(path.join(rootDir, `apps/${app}/src/${file}`), 'utf8');
      assert(engineSrc.includes('SearchWorkerClient'), `${app}/${file} must support off-main-thread build via SearchWorkerClient`);
      assert(engineSrc.includes('new Worker('), `${app}/${file} must construct a dedicated worker`);
      assert(engineSrc.includes('signal'), `${app}/${file} must forward AbortSignal to search`);
      assert(engineSrc.includes('escapeHtml: true'), `${app}/${file} must request escaped original-text highlights`);
      assert(engineSrc.includes('.terminate()'), `${app}/${file} must terminate the worker on destroy`);
      assert(engineSrc.includes('destroy()'), `${app}/${file} must expose destroy()`);
      assert(engineSrc.includes('serialize') && engineSrc.includes('restore'), `${app}/${file} must support snapshot serialize/restore`);
    }

    for (const app of apps) {
      const workerSrc = fs.readFileSync(path.join(rootDir, `apps/${app}/src/worker.ts`), 'utf8');
      assert(workerSrc.includes('startSearchWorker'), `${app}/worker.ts must boot the dedicated worker entrypoint`);
    }
    console.log('   ✅ Proof-app teardown, cancel, and error-path patterns verified');
  }

  // =========================================================================
  // 10. Proof-App Public-API-Only Import Lint
  // =========================================================================
  console.log('10. Linting proof-app imports (public API entries only)...');
  {
    const allowedSpecifiers = new Set(['webgpu-search', 'webgpu-search/worker']);
    const appSrcDirs = ['apps/monaco-palette/src', 'apps/log-viewer/src', 'apps/docs-search/src'];

    function collectTsFiles(dir: string): string[] {
      const out: string[] = [];
      for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) out.push(...collectTsFiles(full));
        else if (entry.isFile() && entry.name.endsWith('.ts')) out.push(full);
      }
      return out;
    }

    for (const rel of appSrcDirs) {
      for (const file of collectTsFiles(path.join(rootDir, rel))) {
        const content = fs.readFileSync(file, 'utf8');
        // Only real module specifiers count: prose/data may mention
        // `packages/webgpu-search/src/...` paths (e.g. monaco file records).
        const specifiers = [
          ...content.matchAll(/(?:from\s+|import\s*\(\s*|require\s*\(\s*)['"]([^'"]+)['"]/g)
        ].map((m) => m[1]!);
        for (const spec of specifiers) {
          assert(
            !spec.includes('webgpu-search/src') && !spec.includes('packages/'),
            `Deep src import '${spec}' in ${path.relative(rootDir, file)}: proof apps must use public entries only`
          );
          if (spec === 'webgpu-search' || spec.startsWith('webgpu-search/')) {
            assert(
              allowedSpecifiers.has(spec),
              `Unsupported entry ${spec} in ${path.relative(rootDir, file)}: only 'webgpu-search' + 'webgpu-search/worker' are public`
            );
          }
        }
      }
    }
    console.log("   ✅ Proof apps consume only 'webgpu-search' + 'webgpu-search/worker'");
  }

  // =========================================================================
  // 11. Framework Recipe Audit (worker-first + abort-safe + destroy-on-unmount)
  // =========================================================================
  console.log('11. Auditing framework recipes for worker-first + abort-safe patterns...');
  {
    const recipes = [
      'examples/react/useDocumentSearch.ts',
      'examples/vue/useSearch.ts',
      'examples/svelte/documentSearchStore.ts',
      'examples/vanilla/search-app.ts'
    ];
    for (const rel of recipes) {
      const src = fs.readFileSync(path.join(rootDir, rel), 'utf8');
      assert(src.includes('SearchWorkerClient'), `${rel} must support worker-first init via SearchWorkerClient`);
      assert(src.includes('AbortController'), `${rel} must cancel superseded queries via AbortController`);
      assert(src.includes('AbortError'), `${rel} must tolerate AbortError on superseded queries`);
      assert(src.includes('destroy'), `${rel} must expose destroy() for unmount teardown`);
      assert(
        !src.includes('webgpu-search/src') && !src.includes('../packages/'),
        `${rel} must import from public entries only`
      );
    }

    const vueBox = fs.readFileSync(path.join(rootDir, 'examples/vue/SearchBox.vue'), 'utf8');
    assert(vueBox.includes('useSearch'), 'SearchBox.vue must consume the useSearch composable');
    const vueHook = fs.readFileSync(path.join(rootDir, 'examples/vue/useSearch.ts'), 'utf8');
    assert(vueHook.includes('onUnmounted') && vueHook.includes('destroy'), 'useSearch must destroy on unmount');
    const svelteBox = fs.readFileSync(path.join(rootDir, 'examples/svelte/SearchBox.svelte'), 'utf8');
    assert(svelteBox.includes('onDestroy') && svelteBox.includes('destroy'), 'SearchBox.svelte must destroy on destroy');
    const reactHook = fs.readFileSync(path.join(rootDir, 'examples/react/useDocumentSearch.ts'), 'utf8');
    assert(reactHook.includes('return () =>'), 'useDocumentSearch must return an effect cleanup that destroys');
    const vanillaApp = fs.readFileSync(path.join(rootDir, 'examples/vanilla/search-app.ts'), 'utf8');
    assert(vanillaApp.includes('removeEventListener'), 'createVanillaSearchApp must detach listeners on destroy');
    console.log('   ✅ Framework recipes are worker-first, abort-safe, and destroy-on-unmount');
  }

  // =========================================================================
  // 12. Proof-App Engine Cross-Platform Safety (DOM-free engines + data)
  // =========================================================================
  console.log('12. Auditing proof-app engines and data generators (DOM-free)...');
  {
    const domFreePatterns = [
      'apps/monaco-palette/src/palette-engine.ts',
      'apps/monaco-palette/src/sample-data.ts',
      'apps/monaco-palette/src/types.ts',
      'apps/monaco-palette/src/worker.ts',
      'apps/log-viewer/src/log-engine.ts',
      'apps/log-viewer/src/log-generator.ts',
      'apps/log-viewer/src/types.ts',
      'apps/log-viewer/src/worker.ts',
      'apps/docs-search/src/docs-engine.ts',
      'apps/docs-search/src/docs-data.ts',
      'apps/docs-search/src/types.ts',
      'apps/docs-search/src/worker.ts'
    ];
    for (const rel of domFreePatterns) {
      const content = fs.readFileSync(path.join(rootDir, rel), 'utf8');
      const noBlock = content.replace(/\/\*[\s\S]*?\*\//g, (m) => '\n'.repeat((m.match(/\n/g) || []).length));
      const code = noBlock
        .split('\n')
        .map((line) => {
          const idx = line.indexOf('//');
          return idx >= 0 ? line.slice(0, idx) : line;
        })
        .join('\n')
        .replace(/'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|`(?:[^`\\]|\\.)*`/g, "''");
      assert(!/\bwindow\b/.test(code), `Bare window global in ${rel}: engines/data must stay DOM-free`);
      assert(!/\bdocument\./.test(code), `Bare document reference in ${rel}: engines/data must stay DOM-free`);
    }
    console.log('   ✅ Proof-app engines and data generators are DOM-free');
  }

  console.log('\n--- All Proof Applications & Verification Tests Passed! ✅ ---');
}

runProofAppTests().catch((err) => {
  console.error('Fatal test failure in test-proof-apps:', err);
  process.exit(1);
});
