import { createMockAdapter } from 'vgpu/mock';
import {
    WebGPUEngine,
    CPUEngine,
    SearchIndex,
    packStringsToGPUBuffer,
    packUnicodeToGPUBuffer,
    serializeUnicodeDataset,
    deserializeUnicodeDataset,
    checkMemoryBudget,
    normalizeText,
    tokensEqual,
    IncompatibleIndexError,
    IncompatibleOptionError,
    QueryTooLongError,
    QUERY_TOKENS_MAX,
    RESULT_LIMIT_MAX,
} from '../packages/webgpu-search/src/index';
import type { SearchResponse } from '../packages/webgpu-search/src/index';

// ASCII-only source: non-ASCII strings built via String.fromCodePoint.
const FCP = String.fromCodePoint;
const SHARP_S = FCP(0xdf);
const GRIN = FCP(0x1f600);
const CJK = FCP(0x4eac);
const STRASSE = 'stra' + SHARP_S + 'e';

function generateTestStrings(count: number): string[] {
    const prefixes = ['src/components', 'src/views', 'src/utils', 'src/services'];
    const nouns = ['User', 'Account', 'Session', 'Auth', 'Order', 'Product'];
    const suffixes = ['Controller', 'Service', 'Handler', 'Manager', 'Provider'];
    const items: string[] = new Array(count);

    for (let i = 0; i < count; i++) {
        const p = prefixes[i % prefixes.length];
        const n = nouns[(i * 7 + 3) % nouns.length];
        const s = suffixes[(i * 17 + 5) % suffixes.length];
        items[i] = `${p}/${n}${s}_${i}.ts`;
    }
    return items;
}

async function runMockTests() {
    console.log('--- Running WebGPU Mock Tests (vgpu/mock) ---');

    const mockAdapter = createMockAdapter({
        features: ['timestamp-query'] as any
    });

    const mockDeviceWrapper = await mockAdapter.requestDevice();
    const mockDevice = mockDeviceWrapper.gpu;

    console.log('1. Testing WebGPUEngine initialization with vgpu mock device...');
    const engine = new WebGPUEngine();
    const initialized = await engine.init(mockDevice);

    if (!initialized || !engine.isReady) {
        throw new Error('Failed to initialize WebGPUEngine with mock device');
    }
    console.log('   ✅ WebGPUEngine initialized successfully with mock device');

    console.log('2. Verifying M3 buffers and pipelines (32 B uniform + 512 B query)...');
    const mockInstrumentation = (mockDevice as any).__vgpuMockInstrumentation;
    if (mockInstrumentation) {
        console.log(`   - Buffers created: ${mockInstrumentation.createBufferDescriptors.length}`);
        console.log(`   - Compute pipelines created: ${mockInstrumentation.createComputePipelineDescriptors.length}`);

        // M3 uniform header: 32 B (16 B-aligned), NOT the v0.1 272 B blob.
        const uniformBufferDesc = mockInstrumentation.createBufferDescriptors.find(
            (b: any) => b.label === 'Uniform Buffer'
        );
        if (!uniformBufferDesc || uniformBufferDesc.size !== 32) {
            throw new Error(`Uniform buffer size mismatch: expected 32, got ${uniformBufferDesc?.size}`);
        }
        console.log('   ✅ Uniform buffer (32 bytes) verified');

        // M3 persistent storage query buffer: 128 tokens * 4 = 512 B.
        const queryBufferDesc = mockInstrumentation.createBufferDescriptors.find(
            (b: any) => b.label === 'Query Buffer'
        );
        if (!queryBufferDesc || queryBufferDesc.size !== 512) {
            throw new Error(`Query buffer size mismatch: expected 512, got ${queryBufferDesc?.size}`);
        }
        console.log('   ✅ Query storage buffer (512 bytes) verified');

        // Output layout unchanged: 8 + 8192 * 8 = 65,544 B.
        const outputBufferDesc = mockInstrumentation.createBufferDescriptors.find(
            (b: any) => b.label === 'Output Buffer'
        );
        if (!outputBufferDesc || outputBufferDesc.size !== 65544) {
            throw new Error(`Output buffer size mismatch: expected 65544 (8192 candidates), got ${outputBufferDesc?.size}`);
        }
        console.log('   ✅ Candidate output buffer (8,192 slots = 65,544 bytes) verified');

        const stagingBufferDesc = mockInstrumentation.createBufferDescriptors.find(
            (b: any) => b.label === 'Staging Buffer'
        );
        if (!stagingBufferDesc || stagingBufferDesc.size !== 65544) {
            throw new Error(`Staging buffer size mismatch: expected 65544, got ${stagingBufferDesc?.size}`);
        }
        console.log('   ✅ Staging buffer (65,544 bytes) verified');
    }

    console.log('3. Testing packUnicodeToGPUBuffer (u32 scalars, no re-normalization)...');
    if (normalizeText(STRASSE, true).tokenCount !== 7) {
        throw new Error('Strasse fold sanity failed: expected 7 post-fold tokens');
    }
    const packedTokens = packUnicodeToGPUBuffer([STRASSE, 'ab', GRIN], { folded: true });
    if (packedTokens.rowCount !== 3 || packedTokens.tokenCount !== 7 + 2 + 1) {
        throw new Error(`packUnicode counts wrong: ${packedTokens.rowCount}/${packedTokens.tokenCount}`);
    }
    if (packedTokens.offsets.length !== 4 || packedTokens.offsets[0] !== 0 || packedTokens.offsets[3] !== packedTokens.tokenCount) {
        throw new Error('packUnicode offsets not monotonic/terminal');
    }
    if (packedTokens.recordsByteLength !== packedTokens.tokenCount * 4 || packedTokens.offsetsByteLength !== 16) {
        throw new Error('packUnicode byte lengths wrong');
    }
    // Pre-tokenized input takes the zero-renorm path (no strings involved).
    const preTok = [new Uint32Array([1, 2, 3]), new Uint32Array([4])];
    const packedPre = packUnicodeToGPUBuffer(preTok, { folded: true });
    if (packedPre.tokenCount !== 4 || packedPre.offsets[2] !== 4 || packedPre.tokens[3] !== 4) {
        throw new Error('packUnicode Uint32Array[] path wrong');
    }
    // Empty input returns rowCount 0 before input[0] discrimination.
    const packedEmpty = packUnicodeToGPUBuffer([], { folded: true });
    if (packedEmpty.rowCount !== 0 || packedEmpty.tokenCount !== 0) {
        throw new Error('packUnicode empty input wrong');
    }
    // slotBytes is throw-on-use.
    let slotThrew = false;
    try {
        packUnicodeToGPUBuffer(['a'], { slotBytes: 64 } as any);
    } catch (e: any) {
        slotThrew = e instanceof IncompatibleOptionError;
    }
    if (!slotThrew) throw new Error('packUnicode slotBytes must throw IncompatibleOptionError');
    console.log('   ✅ packUnicodeToGPUBuffer verified (folded tokens, offsets, empty, slotBytes)');

    console.log('4. Testing U2F2 serialization roundtrip + corrupt-header rejection...');
    const rt = packUnicodeToGPUBuffer(['hello', STRASSE, GRIN], { folded: true });
    const bytes = serializeUnicodeDataset(rt);
    const header = new Uint32Array(bytes, 0, 9);
    if (header[0] !== 0x55324632 || header[1] !== 2) {
        throw new Error(`U2F2 magic/version wrong: ${header[0].toString(16)}/${header[1]}`);
    }
    const back = deserializeUnicodeDataset(bytes);
    if (back.rowCount !== 3 || back.folded !== true || back.profileId !== 'unicode-default' ||
        back.unicodeVersion !== '16.0.0' || back.scoringVersion !== 'parity-v1' ||
        back.formatVersion !== 2 || back.nfcProbedVersion !== null) {
        throw new Error('deserialize field mismatch');
    }
    if (!tokensEqual(back.tokens, rt.tokens) || !tokensEqual(back.offsets, rt.offsets)) {
        throw new Error('serialize roundtrip payload mismatch');
    }
    const expectIncompatible = (label: string, fn: () => void) => {
        try {
            fn();
        } catch (e: any) {
            if (e instanceof IncompatibleIndexError) return;
            throw new Error(`${label}: wrong error type ${e?.name}`);
        }
        throw new Error(`${label}: expected IncompatibleIndexError`);
    };
    expectIncompatible('bad-magic', () => {
        const bad = new ArrayBuffer(36);
        new Uint32Array(bad)[0] = 0xdeadbeef;
        deserializeUnicodeDataset(bad);
    });
    expectIncompatible('truncated', () => deserializeUnicodeDataset(bytes.slice(0, bytes.byteLength - 4)));
    expectIncompatible('bad-checksum', () => {
        const tampered = bytes.slice(0);
        new Uint8Array(tampered)[40] ^= 0xff;
        deserializeUnicodeDataset(tampered);
    });
    expectIncompatible('neutered', () => deserializeUnicodeDataset(new ArrayBuffer(0)));
    expectIncompatible('legacy-v0.1', () => {
        const legacy = packStringsToGPUBuffer(['abc', 'def']);
        deserializeUnicodeDataset(legacy.recordsBufferData);
    });
    console.log('   ✅ U2F2 roundtrip + 5 corrupt/legacy rejections verified');

    console.log('5. Testing buffer packer and mock VRAM loading (packed/string/serialized)...');
    const strings = generateTestStrings(500);
    const packed = packUnicodeToGPUBuffer(strings, { folded: true });
    const { uploadTimeMs } = await engine.loadDataset(packed);
    console.log(`   ✅ Packed dataset loaded (500 items, ${packed.combinedByteLength} bytes, ${uploadTimeMs.toFixed(2)}ms)`);
    await engine.loadDataset(strings);
    await engine.loadDataset(serializeUnicodeDataset(packed));
    console.log('   ✅ string[] + serialized ArrayBuffer overloads verified');

    console.log('6. Testing search execution, empty query, and 0-row early return...');
    const emptyRes = await engine.search('', { mode: 'substring' });
    if (emptyRes.totalMatches !== 0 || emptyRes.results.length !== 0 || (emptyRes as any).query !== undefined && emptyRes.query !== '') {
        throw new Error('Empty query test failed');
    }
    await engine.loadDataset([]);
    const zeroRes = await engine.search('abc', { mode: 'substring' });
    if (zeroRes.totalMatches !== 0 || zeroRes.results.length !== 0) {
        throw new Error('0-row early return failed');
    }
    console.log('   ✅ Empty query + 0-row early return handled properly');

    engine.destroy();
    console.log('   ✅ Low-level engine resources cleaned up via engine.destroy()');

    console.log('7. Testing high-level SearchIndex API with injected mock device...');
    const searchIndex = await SearchIndex.create(strings, {
        device: mockDevice,
        preferGpu: true
    });

    const stats = searchIndex.getStats();
    if (stats.engine !== 'webgpu') {
        throw new Error(`Expected SearchIndex engine to be 'webgpu', got '${stats.engine}'`);
    }
    const expectVram = stats.tokenCount * 4 + (stats.size + 1) * 4;
    if (stats.vramAllocatedBytes !== expectVram) {
        throw new Error(`VRAM must be actual packed bytes (${expectVram}), got ${stats.vramAllocatedBytes}`);
    }
    console.log(`   ✅ SearchIndex created with engine: ${stats.engine}, VRAM: ${stats.vramAllocatedBytes} bytes (actual)`);

    const gpuSearchRes = await searchIndex.search('AuthController', { mode: 'fuzzy', limit: 20 });
    if (gpuSearchRes.engine !== 'webgpu') {
        throw new Error(`Expected search response engine to be 'webgpu', got '${gpuSearchRes.engine}'`);
    }
    console.log(`   ✅ SearchIndex search routed to WebGPU successfully (query: '${gpuSearchRes.query}')`);

    console.log('8. Testing M3 routing: non-ASCII + long queries reach WebGPU (M2 gate deleted)...');
    for (const q of [STRASSE, GRIN, CJK, 'a'.repeat(100)]) {
        const r = await searchIndex.search(q, { mode: 'substring', limit: 5 });
        if (r.engine !== 'webgpu') {
            throw new Error(`Query ${JSON.stringify(q.slice(0, 12))} must route to webgpu, got ${r.engine}`);
        }
    }
    const atCap = await searchIndex.search('a'.repeat(QUERY_TOKENS_MAX), { mode: 'substring', limit: 5 });
    if (atCap.engine !== 'webgpu') throw new Error('128-token query must route to webgpu');
    let tooLongThrew = false;
    try {
        await searchIndex.search('a'.repeat(QUERY_TOKENS_MAX + 2), { mode: 'substring' });
    } catch (e: any) {
        tooLongThrew = e instanceof QueryTooLongError;
    }
    if (!tooLongThrew) throw new Error('130-token query must throw QueryTooLongError');
    const cpuForced = await searchIndex.search('a'.repeat(QUERY_TOKENS_MAX + 2), { mode: 'substring', onQueryTooLong: 'cpu-fallback' });
    if (cpuForced.engine !== 'cpu') throw new Error('onQueryTooLong cpu-fallback must force CPU');
    searchIndex.destroy();
    console.log('   ✅ Non-ASCII/long routing + 128/130 token gates verified');

    console.log('9. Testing 5 bind-group entries + storage binding coverage...');
    const bgEngine = new WebGPUEngine();
    await bgEngine.init(mockDevice);
    await bgEngine.loadDataset(['alpha', 'beta']);
    await bgEngine.search('alpha', { mode: 'substring' });
    const bgDescs = (mockDevice as any).__vgpuMockInstrumentation.createBindGroupDescriptors;
    const lastBg = bgDescs[bgDescs.length - 1];
    const bindings = (lastBg.entries as any[]).map((e: any) => e.binding).sort((a: number, b: number) => a - b);
    if (bindings.length !== 5 || bindings.join(',') !== '0,1,2,3,4') {
        throw new Error(`Expected 5 bind-group entries 0..4, got ${bindings.join(',')}`);
    }
    const maxStorage = (mockDevice as any).limits.maxStorageBuffersPerShaderStage;
    if (!(maxStorage >= 4)) {
        throw new Error(`maxStorageBuffersPerShaderStage ${maxStorage} must cover 4 storage buffers`);
    }
    bgEngine.destroy();
    console.log('   ✅ 5 bindings (uniform/offsets/records/query/output) + binding limits verified');

    console.log('10. Testing per-buffer memory budget checks...');
    const tinyDevice = { limits: { maxBufferSize: 1024, maxStorageBufferBindingSize: 1024, maxComputeWorkgroupsPerDimension: 65535 } } as unknown as GPUDevice;
    const recOver = checkMemoryBudget(10_000, 64, tinyDevice);
    if (recOver.allowed || !recOver.reason?.includes('records')) throw new Error('records over-budget must fail closed');
    const offOver = checkMemoryBudget(10_000, 0, tinyDevice);
    if (offOver.allowed || !offOver.reason?.includes('offsets')) throw new Error('offsets over-budget must fail closed');
    const outOver = checkMemoryBudget(1, 1, tinyDevice, 10_000_000);
    if (outOver.allowed || !outOver.reason?.includes('output')) throw new Error('output over-budget must fail closed');
    const fits = checkMemoryBudget(10, 64);
    if (!fits.allowed) throw new Error('Small dataset must fit in fallback budget');
    console.log('   ✅ Per-buffer (records/offsets/output) fail-closed checks verified');

    console.log('11. Testing high-level SearchIndex CPU auto-routing fallback...');
    const smallStrings = ['apple', 'banana', 'orange', 'grape'];
    const cpuIndex = await SearchIndex.create(smallStrings, { threshold: 30000 });
    const cpuStats = cpuIndex.getStats();
    if (cpuStats.engine !== 'cpu') {
        throw new Error(`Expected small dataset to route to 'cpu', got '${cpuStats.engine}'`);
    }

    const cpuRes = await cpuIndex.search('an', { mode: 'fuzzy' });
    if (cpuRes.engine !== 'cpu') {
        throw new Error(`Expected search response engine to be 'cpu', got '${cpuRes.engine}'`);
    }
    if (cpuRes.results.length === 0) {
        throw new Error('Expected matches for query "an" in small dataset');
    }
    for (const r of cpuRes.results) {
        if (typeof r.score !== 'number' || typeof r.index !== 'number' || typeof r.text !== 'string') {
            throw new Error(`Invalid SearchResultItem contract: ${JSON.stringify(r)}`);
        }
    }
    console.log(`   ✅ CPU auto-routing verified (found ${cpuRes.results.length} matches with normalized scores)`);
    cpuIndex.destroy();

    console.log('12. Testing overflow boundaries (8191/8192/8193) on CPU parity...');
    for (const n of [8191, 8192, 8193]) {
        const items = Array.from({ length: n }, (_, i) => `ov-match-${i}`);
        const idx = await SearchIndex.create(items, { preferGpu: false });
        const res = await idx.search('ov-match-', { mode: 'substring' });
        const wantOverflow = n > RESULT_LIMIT_MAX;
        if (res.totalMatches !== n || res.hasOverflow !== wantOverflow || res.candidateCount !== Math.min(n, RESULT_LIMIT_MAX)) {
            throw new Error(`Overflow ${n}: total=${res.totalMatches} overflow=${res.hasOverflow} cand=${res.candidateCount}`);
        }
        idx.destroy();
    }
    console.log('   ✅ Overflow at-cap/over-cap multiset scope verified');

    console.log('13. Testing searchCold pipeline on WebGPUEngine...');
    const coldEngine = new WebGPUEngine();
    await coldEngine.init(mockDevice);
    const coldRes = await coldEngine.searchCold(strings, 'Auth', { mode: 'substring' });
    if (typeof coldRes.datasetUploadMs !== 'number' || typeof coldRes.coldTotalMs !== 'number') {
        throw new Error('searchCold failed to report datasetUploadMs or coldTotalMs');
    }
    console.log(`   ✅ searchCold verified (coldTotalMs: ${coldRes.coldTotalMs.toFixed(2)}ms, upload: ${coldRes.datasetUploadMs.toFixed(2)}ms)`);
    coldEngine.destroy();

    console.log('14. Testing empty dataset edge-case on SearchIndex...');
    const emptyIndex = await SearchIndex.create([], { preferGpu: true });
    const emptyStats = emptyIndex.getStats();
    if (emptyStats.engine !== 'cpu' || emptyStats.size !== 0) {
        throw new Error('Empty dataset must route to CPU with size 0');
    }
    const emptySearchResult = await emptyIndex.search('test');
    if (emptySearchResult.results.length !== 0 || emptySearchResult.totalMatches !== 0) {
        throw new Error('Expected 0 results for search on empty dataset');
    }
    emptyIndex.destroy();
    console.log('   ✅ Empty dataset handled cleanly without WebGPU allocation crash');

    console.log('15. Testing caseSensitivity parity on CPUEngine...');
    const cpuEngineTest = new CPUEngine();
    const testCases = ['AuthController.ts', 'authcontroller.ts', 'AUTHCONTROLLER.TS'];
    const caseSensitiveRes = cpuEngineTest.searchNative(testCases, 'Auth', 10, true);
    if (caseSensitiveRes.totalMatches !== 1) {
        throw new Error(`Expected exactly 1 case-sensitive match, got ${caseSensitiveRes.totalMatches}`);
    }
    const caseInsensitiveRes = cpuEngineTest.searchNative(testCases, 'Auth', 10, false);
    if (caseInsensitiveRes.totalMatches !== 3) {
        throw new Error(`Expected 3 case-insensitive matches, got ${caseInsensitiveRes.totalMatches}`);
    }
    console.log('   ✅ CPUEngine caseSensitive parameter verified');

    console.log('16. Testing SearchIndex limit contract on CPU...');
    const limitItems = Array.from({ length: 9_000 }, (_, i) => `match-${i}`);
    const limitIndex = await SearchIndex.create(limitItems, { preferGpu: false });
    const defaultLimit = await limitIndex.search('match-', { mode: 'substring' });
    const minimumLimit = await limitIndex.search('match-', { mode: 'substring', limit: 0 });
    const maximumLimit = await limitIndex.search('match-', { mode: 'substring', limit: 9_000 });
    const legacyAliasLimit = await limitIndex.search('match-', { mode: 'substring', maxResults: 3 });
    const nanLimit = await limitIndex.search('match-', { mode: 'substring', limit: NaN });
    if (defaultLimit.results.length !== 50) {
        throw new Error(`Expected default limit of 50, got ${defaultLimit.results.length}`);
    }
    if (minimumLimit.results.length !== 1) {
        throw new Error(`Expected limit=0 to clamp to 1, got ${minimumLimit.results.length}`);
    }
    if (maximumLimit.results.length !== 8_192) {
        throw new Error(`Expected limit=9000 to clamp to 8192, got ${maximumLimit.results.length}`);
    }
    if (legacyAliasLimit.results.length !== 3) {
        throw new Error(`Expected maxResults=3 alias to return 3 results, got ${legacyAliasLimit.results.length}`);
    }
    if (nanLimit.results.length !== 50) {
        throw new Error(`Expected limit=NaN to fall back to default 50, got ${nanLimit.results.length}`);
    }
    limitIndex.destroy();
    console.log('   ✅ SearchIndex limit defaults, bounds, and maxResults alias verified on CPU');

    console.log('17. Testing variable-length string support (>100 characters without truncation)...');
    const veryLongString = 'packages/core/extremely/long/nested/folder/path/to/some/deeply/embedded/internal/structure/that/exceeds/the/old/fiftynine/limit/DeepSpecialController.ts';
    if (veryLongString.length <= 100) {
        throw new Error('Test string must be > 100 chars');
    }
    const dynamicPacked = packUnicodeToGPUBuffer([veryLongString], { folded: true });
    if (dynamicPacked.tokenCount !== veryLongString.length) {
        throw new Error(`Expected packed tokenCount ${veryLongString.length}, got ${dynamicPacked.tokenCount}`);
    }
    if (dynamicPacked.recordsByteLength !== veryLongString.length * 4) {
        throw new Error('Records buffer must hold one u32 per post-fold token');
    }

    const longIndex = await SearchIndex.create([veryLongString], { preferGpu: false });
    const longSearch = await longIndex.search('DeepSpecialController');
    if (longSearch.results.length === 0 || longSearch.results[0].text !== veryLongString) {
        throw new Error('Failed to match search token located past character position 100');
    }
    longIndex.destroy();
    console.log(`   ✅ Variable-length strings verified (${veryLongString.length} chars preserved and matched past char 100)`);

    console.log('18. Testing direct string[] callers on WebGPUEngine.loadDataset and searchCold...');
    const directEngine = new WebGPUEngine();
    await directEngine.init(mockDevice);
    await directEngine.loadDataset(['apple', 'banana', 'orange']);
    const directColdRes = await directEngine.searchCold(['alpha', 'beta', 'gamma'], 'beta', { mode: 'substring' });
    if (typeof directColdRes.coldTotalMs !== 'number') {
        throw new Error('searchCold with string[] failed');
    }
    directEngine.destroy();
    console.log('   ✅ Direct string[] callers on WebGPUEngine verified');

    console.log('19. Testing concurrent searches and GPU fallback paths...');
    const concurrentIndex = await SearchIndex.create(strings, { device: mockDevice, preferGpu: true });
    const concurrentResults = await Promise.all([
        concurrentIndex.search('Auth', { mode: 'substring', limit: 20 }),
        concurrentIndex.search('Order', { mode: 'substring', limit: 20 }),
        concurrentIndex.search('Service', { mode: 'substring', limit: 20 })
    ]);
    const expectedQueries = ['Auth', 'Order', 'Service'];
    concurrentResults.forEach((result: SearchResponse, i: number) => {
        if (result.query !== expectedQueries[i] || result.engine !== 'webgpu') {
            throw new Error(`Concurrent search ${i} returned cross-talk: ${result.query}/${result.engine}`);
        }
    });
    console.log('   ✅ Concurrent searches stay isolated with no cross-talk');

    const gpuEngine = (concurrentIndex as any).gpuEngine;
    const originalSearch = gpuEngine.search.bind(gpuEngine);
    gpuEngine.search = async () => { throw new Error('injected GPU query failure'); };
    try {
        const fallbackResult = await concurrentIndex.search('Auth', { mode: 'substring', limit: 20 });
        if (fallbackResult.engine !== 'cpu' || fallbackResult.results.length === 0) {
            throw new Error('A GPU query failure must fall back to CPU results');
        }
    } finally {
        gpuEngine.search = originalSearch;
    }
    concurrentIndex.destroy();
    console.log('   ✅ GPU query errors fall back to CPU results');

    console.log('\n--- All vgpu/mock Tests Passed! (0ms GPU, 100% in-memory) ✅ ---');
}

runMockTests().catch(err => {
    console.error('Test failed:', err);
    process.exit(1);
});
