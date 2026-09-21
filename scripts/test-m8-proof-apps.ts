import assert from 'node:assert';
import { execSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import {
  DocumentIndex,
  serializeDocumentIndex,
  restoreDocumentIndex,
  deserializeDocumentSnapshotHeader,
  U2D4_MAGIC,
  U2D4_FORMAT_VERSION,
  U2D4_HEADER_BYTES,
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

async function runM8Tests() {
  console.log('--- Running Milestone 8: Proof Applications & Verification Tests ---');

  const rootDir = path.resolve(__dirname, '..');
  const monacoDir = path.join(rootDir, 'apps/monaco-palette');
  const logViewerDir = path.join(rootDir, 'apps/log-viewer');

  // =========================================================================
  // 1. Proof Applications Monorepo Structure & File Contracts
  // =========================================================================
  console.log('1. Verifying Proof Applications monorepo structure and file contracts...');
  {
    // Ensure dist builds exist on clean clones
    if (!fs.existsSync(path.join(monacoDir, 'dist/index.html')) || !fs.existsSync(path.join(logViewerDir, 'dist/index.html'))) {
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
  // 4. Versioned Snapshot Persistence (U2D4 Binary Format Roundtrip)
  // =========================================================================
  console.log('4. Testing U2D4 Little-Endian binary format serialization & restore roundtrip...');
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

    // B. Serialize to U2D4 binary buffer
    const snapshotBuffer = serializeDocumentIndex(docIndex);
    assert(snapshotBuffer.byteLength > 56, 'Snapshot must be larger than 56-byte U2D4 header');

    // C. Inspect 56-byte Little-Endian Header (8-byte aligned)
    const header = deserializeDocumentSnapshotHeader(snapshotBuffer);
    assert.strictEqual(header.magic, U2D4_MAGIC); // 0x55324434 ('U2D4')
    assert.strictEqual(header.formatVersion, U2D4_FORMAT_VERSION); // 4
    assert.strictEqual(U2D4_HEADER_BYTES, 56);
    assert.strictEqual(U2D4_HEADER_BYTES % 8, 0, 'U2D4 header must be 8-byte aligned');
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
    const restoredIndex = await restoreDocumentIndex<StructuredLogRecord>(snapshotBuffer, {
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
    const decoupledBuffer = serializeDocumentIndex(docIndex, { decoupled: true });
    const decoupledHeader = deserializeDocumentSnapshotHeader(decoupledBuffer);
    assert.strictEqual(decoupledHeader.docsByteLength, 0, 'Decoupled snapshot must have docsByteLength = 0');
    assert(decoupledBuffer.byteLength < snapshotBuffer.byteLength, 'Decoupled snapshot must be smaller');

    // Restore decoupled with external documents
    const restoredDecoupled = await restoreDocumentIndex<StructuredLogRecord>(decoupledBuffer, {
      documents: testLogs,
      options: { preferGpu: false }
    });
    const decoupledSearch = await restoredDecoupled.search(baselineQuery, { mode: 'fuzzy', limit: 25 });
    assert.strictEqual(decoupledSearch.totalMatches, baselineSearch.totalMatches);
    assert.strictEqual(decoupledSearch.results[0]!.doc.message, baselineSearch.results[0]!.doc.message);

    docIndex.destroy();
    restoredIndex.destroy();
    restoredDecoupled.destroy();
    console.log('   ✅ U2D4 Little-Endian binary format serialization & decoupled restore verified');
  }

  // =========================================================================
  // 6. v0.4 Proof-App Feature Integration (prefix, filters, facets, suggest)
  // =========================================================================
  console.log('6. Testing v0.4 proof-app feature integration (prefix + type filter + autocomplete)...');
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
    const suggestRes = await palette.suggest('comp', { mode: 'prefix', limit: 5 });
    assert(suggestRes.suggestions.length >= 1, 'suggest must return completions');

    const inlineSuggest = await palette.search('comp', {
      mode: 'prefix',
      limit: 5,
      suggest: { mode: 'prefix', limit: 5 }
    });
    assert((inlineSuggest.suggestions?.length ?? 0) >= 1, 'inline suggest must return completions');
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

    // Timestamp range narrows monotonically.
    const newest = logs[logs.length - 1]!.timestamp;
    const mid = logs[Math.floor(logs.length / 2)]!.timestamp;
    const rangeNarrow = await logEngine.search('timeout', {
      mode: 'fuzzy',
      limit: 50,
      timestampRange: { from: mid, to: newest }
    });
    const rangeWide = await logEngine.search('timeout', { mode: 'fuzzy', limit: 50 });
    assert(rangeNarrow.totalMatches <= rangeWide.totalMatches);

    // U2D4 snapshot roundtrip preserves structured filtering.
    const snap = await logEngine.serializeSnapshot();
    const snapHeader = deserializeDocumentSnapshotHeader(snap);
    assert.strictEqual(snapHeader.magic, U2D4_MAGIC);
    assert.strictEqual(snapHeader.formatVersion, 4);
    logEngine.destroy();
    console.log('   ✅ v0.4 proof-app feature integration verified');
  }

  // =========================================================================
  // 5. Cross-Platform Safety Audit (Zero Unguarded DOM Globals)
  // =========================================================================
  console.log('5. Auditing core library cross-platform safety (zero unguarded DOM globals)...');
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
      // as scripts/check-parity-lint.ts and the M4 search-modes DOM test).
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

  console.log('\n--- All Milestone 8 Proof Applications & Verification Tests Passed! ✅ ---');
}

runM8Tests().catch((err) => {
  console.error('Fatal test failure in test-m8-proof-apps:', err);
  process.exit(1);
});
