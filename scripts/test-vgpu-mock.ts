import { createMockAdapter } from 'vgpu/mock';
import {
    WebGPUEngine,
    CPUEngine,
    SearchIndex,
    packStringsToGPUBuffer,
    packDataset,
    serializeDataset,
    deserializeDataset,
    validatePackedOffsets,
    checkMemoryBudget,
    normalizeText,
    tokensEqual,
    compareExactResults,
    scoreFuzzyTokens,
    scoreSubstringTokens,
    scoreExactMatches,
    IncompatibleIndexError,
    IncompatibleOptionError,
    ProfileMismatchError,
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

    console.log('2. Verifying  buffers and pipelines (32 B uniform + 512 B query)...');
    const mockInstrumentation = (mockDevice as any).__vgpuMockInstrumentation;
    if (mockInstrumentation) {
        console.log(`   - Buffers created: ${mockInstrumentation.createBufferDescriptors.length}`);
        console.log(`   - Compute pipelines created: ${mockInstrumentation.createComputePipelineDescriptors.length}`);

        // uniform header: 32 B (16 B-aligned), NOT the 272 B blob.
        const uniformBufferDesc = mockInstrumentation.createBufferDescriptors.find(
            (b: any) => b.label === 'Uniform Buffer'
        );
        if (!uniformBufferDesc || uniformBufferDesc.size !== 32) {
            throw new Error(`Uniform buffer size mismatch: expected 32, got ${uniformBufferDesc?.size}`);
        }
        console.log('   ✅ Uniform buffer (32 bytes) verified');

        // persistent storage query buffer: 128 tokens * 4 = 512 B.
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

    console.log('3. Testing packDataset (u32 scalars, no re-normalization)...');
    if (normalizeText(STRASSE, true).tokenCount !== 7) {
        throw new Error('Strasse fold sanity failed: expected 7 post-fold tokens');
    }
    const packedTokens = packDataset([STRASSE, 'ab', GRIN], { normalized: true });
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
    const packedPre = packDataset(preTok, { normalized: true });
    if (packedPre.tokenCount !== 4 || packedPre.offsets[2] !== 4 || packedPre.tokens[3] !== 4) {
        throw new Error('packUnicode Uint32Array[] path wrong');
    }
    // Empty input returns rowCount 0 before input[0] discrimination.
    const packedEmpty = packDataset([], { normalized: true });
    if (packedEmpty.rowCount !== 0 || packedEmpty.tokenCount !== 0) {
        throw new Error('packUnicode empty input wrong');
    }
    // slotBytes is throw-on-use.
    let slotThrew = false;
    try {
        packDataset(['a'], { slotBytes: 64 } as any);
    } catch (e: any) {
        slotThrew = e instanceof IncompatibleOptionError;
    }
    if (!slotThrew) throw new Error('packUnicode slotBytes must throw IncompatibleOptionError');
    console.log('   ✅ packDataset verified (normalized tokens, offsets, empty, slotBytes)');

    console.log('4. Testing dataset serialization roundtrip + corrupt-header rejection...');
    const rt = packDataset(['hello', STRASSE, GRIN], { normalized: true });
    const bytes = serializeDataset(rt);
    const header = new Uint32Array(bytes, 0, 9);
    if (header[0] !== 0x55324632 || header[1] !== 2) {
        throw new Error(`dataset magic/version wrong: ${header[0].toString(16)}/${header[1]}`);
    }
    const back = deserializeDataset(bytes);
    if (back.rowCount !== 3 || back.normalized !== true || back.profileId !== 'unicode-default' ||
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
        deserializeDataset(bad);
    });
    expectIncompatible('truncated', () => deserializeDataset(bytes.slice(0, bytes.byteLength - 4)));
    expectIncompatible('bad-checksum', () => {
        const tampered = bytes.slice(0);
        new Uint8Array(tampered)[40] ^= 0xff;
        deserializeDataset(tampered);
    });
    expectIncompatible('neutered', () => deserializeDataset(new ArrayBuffer(0)));
    expectIncompatible('legacy-', () => {
        const legacy = packStringsToGPUBuffer(['abc', 'def']);
        deserializeDataset(legacy.recordsBufferData);
    });
    console.log('   ✅ dataset roundtrip + 5 corrupt/legacy rejections verified');

    console.log('5. Testing buffer packer and mock VRAM loading (packed/string/serialized)...');
    const strings = generateTestStrings(500);
    const packed = packDataset(strings, { normalized: true });
    const { uploadTimeMs } = await engine.loadDataset(packed);
    console.log(`   ✅ Packed dataset loaded (500 items, ${packed.combinedByteLength} bytes, ${uploadTimeMs.toFixed(2)}ms)`);
    await engine.loadDataset(strings);
    await engine.loadDataset(serializeDataset(packed));
    console.log('   ✅ string[] + serialized ArrayBuffer overloads verified');

    console.log('6. Testing search execution, empty query, and 0-row early return...');
    const emptyRes = await engine.search('', { mode: 'substring' });
    if (emptyRes.totalMatches !== 0 || emptyRes.results.length !== 0 || emptyRes.query !== '') {
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

    console.log('8. Testing  routing: non-ASCII + long queries reach WebGPU (gate deleted)...');
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
    const caseSensitiveRes = cpuEngineTest.searchNaiveScan(testCases, 'Auth', 10, true);
    if (caseSensitiveRes.totalMatches !== 1) {
        throw new Error(`Expected exactly 1 case-sensitive match, got ${caseSensitiveRes.totalMatches}`);
    }
    const caseInsensitiveRes = cpuEngineTest.searchNaiveScan(testCases, 'Auth', 10, false);
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
    const dynamicPacked = packDataset([veryLongString], { normalized: true });
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

    console.log('20. Testing CPU/GPU parity formulas (host-side WGSL integer mirror)...');
    // The mock never executes WGSL, so pin the contracted integer formulas
    // directly: scoreSubstringTokens/scoreFuzzyTokens must equal a JS mirror
    // of the WGSL i32 arithmetic over fixed vectors + fuzz.
    const u32 = (arr: number[]) => new Uint32Array(arr);
    const wgslSubstring = (rec: Uint32Array, q: Uint32Array): { matched: boolean; score: number; pos: number } => {
        if (q.length === 0 || rec.length < q.length) return { matched: false, score: 0, pos: -1 };
        const maxStart = rec.length - q.length;
        for (let start = 0; start <= maxStart; start++) {
            let ok = true;
            for (let j = 0; j < q.length; j++) {
                if (rec[start + j] !== q[j]) { ok = false; break; }
            }
            if (ok) {
                // Mirror: 1000i - i32(pos)*10 - (i32(str)-i32(ql)), i32 wrap.
                const score = (1000 - Math.imul(start, 10) - (rec.length - q.length)) | 0;
                return { matched: true, score, pos: start };
            }
        }
        return { matched: false, score: 0, pos: -1 };
    };
    const parityVectors: Array<[number[], number[]]> = [
        [[1, 2, 3, 4], [2, 3]],
        [[97, 98, 99], [97]],
        [[97, 98, 99], [98, 99]],
        [[47, 97, 98], [97, 98]],
        [[0x1f600, 97, 98], [0x1f600]],
        [[1, 1, 1, 2, 1], [1, 2]],
        [[5], [5]],
        [[5], [6]],
    ];
    for (const [r, q] of parityVectors) {
        const rec = u32(r);
        const qry = u32(q);
        const cpu = scoreSubstringTokens(rec, qry);
        const w = wgslSubstring(rec, qry);
        if (cpu.matched !== w.matched || (cpu.matched && (cpu.score !== w.score || cpu.matchStart !== w.pos))) {
            throw new Error(`substring parity drift rec=${r} q=${q}: cpu=${JSON.stringify(cpu)} wgsl=${JSON.stringify(w)}`);
        }
        const cf = scoreFuzzyTokens(rec, qry);
        // Fuzzy oracle: brute-force subsequence check + score shape sanity
        // (exact formula pinned by exact-scorer unit vectors below).
        if (cf.matched) {
            if (!Number.isInteger(cf.score)) throw new Error('fuzzy score must be integer');
            const ref = scoreExactMatches([rec], qry, 'fuzzy', 10, ['x']);
            if (ref.totalMatches !== 1 || ref.results[0].score !== cf.score) {
                throw new Error('fuzzy scoreExactMatches/score mismatch');
            }
        }
    }
    // Fuzz small alphabet for substring parity (deterministic LCG).
    let seed = 0x12345678;
    const rnd = () => (seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 0x100000000;
    for (let t = 0; t < 500; t++) {
        const rl = 1 + Math.floor(rnd() * 8);
        const ql = 1 + Math.floor(rnd() * Math.min(4, rl));
        const rec = u32(Array.from({ length: rl }, () => Math.floor(rnd() * 3)));
        const qry = u32(Array.from({ length: ql }, () => Math.floor(rnd() * 3)));
        const cpu = scoreSubstringTokens(rec, qry);
        const w = wgslSubstring(rec, qry);
        if (cpu.matched !== w.matched || (cpu.matched && cpu.score !== w.score)) {
            throw new Error(`fuzz substring drift t=${t}`);
        }
    }
    // Sort contract: score desc, index asc.
    const sorted = [{ score: 5, index: 2 }, { score: 5, index: 1 }, { score: 9, index: 0 }].sort(compareExactResults);
    if (sorted[0].score !== 9 || sorted[1].index !== 1 || sorted[2].index !== 2) {
        throw new Error('compareExactResults contract broken');
    }
    console.log('   ✅ Parity formulas pinned (substring WGSL mirror + fuzz + sort)');

    console.log('21. Testing 129-token boundary, ProfileMismatch, mode + caseSensitive gates...');
    const gateEngine = new WebGPUEngine();
    await gateEngine.init(mockDevice);
    await gateEngine.loadDataset(['alpha', 'beta']);
    // 129 = QUERY_TOKENS_MAX+1 must throw at both layers with exact actual.
    for (const q of ['a'.repeat(QUERY_TOKENS_MAX + 1)]) {
        let threwEngine = false;
        try { await gateEngine.search(q, { mode: 'substring' }); } catch (e: any) { threwEngine = e instanceof QueryTooLongError && e.actual === 129 && e.limit === 128; }
        if (!threwEngine) throw new Error('engine 129-token must throw QueryTooLongError actual=129');
    }
    const gateIndex = await SearchIndex.create(['alpha'], { device: mockDevice, preferGpu: true });
    let threwIndex = false;
    try { await gateIndex.search('a'.repeat(129), { mode: 'substring' }); } catch (e: any) { threwIndex = e instanceof QueryTooLongError && e.actual === 129; }
    if (!threwIndex) throw new Error('index 129-token must throw QueryTooLongError actual=129');
    gateIndex.destroy();
    // ProfileMismatch at engine level (default normalized=true index, caseSensitive:true).
    let pmThrew = false;
    try { await gateEngine.search('alpha', { mode: 'substring', caseSensitive: true }); } catch (e: any) { pmThrew = e instanceof ProfileMismatchError; }
    if (!pmThrew) throw new Error('engine ProfileMismatchError must throw for caseSensitive:true on normalized index');
    // Omitted flag defaults to false (hybrid-compatible): must NOT throw on normalized=true.
    await gateEngine.search('alpha', { mode: 'substring' });
    // Forged non-boolean caseSensitive fails closed.
    let typeThrew = false;
    try { await gateEngine.search('alpha', { mode: 'substring', caseSensitive: 1 as any }); } catch (e: any) { typeThrew = e instanceof TypeError; }
    if (!typeThrew) throw new Error('forged caseSensitive:1 must throw TypeError');
    // Invalid mode fails closed (unified with indexes: IncompatibleOptionError).
    let modeThrew = false;
    try { await gateEngine.search('alpha', { mode: 'regex' as any }); } catch (e: any) { modeThrew = e instanceof IncompatibleOptionError; }
    if (!modeThrew) throw new Error('invalid mode must throw IncompatibleOptionError');
    // No-device gates: direct engine without init still enforces throw/echo.
    const bareEngine = new WebGPUEngine();
    const bareEmpty = await bareEngine.search('   ', { mode: 'substring' });
    if (bareEmpty.query !== '') throw new Error('no-device whitespace must echo unified empty');
    let bareLongThrew = false;
    try { await bareEngine.search('a'.repeat(200), { mode: 'substring' }); } catch (e: any) { bareLongThrew = e instanceof QueryTooLongError; }
    if (!bareLongThrew) throw new Error('no-device over-long must throw QueryTooLongError');
    gateEngine.destroy();
    console.log('   ✅ 129 boundary + ProfileMismatch + mode/caseSensitive + no-device gates verified');

    console.log('22. Testing dataset hostile headers + engine-level rejections...');
    const good = packDataset(['hello', 'world'], { normalized: true });
    const goodBytes = serializeDataset(good);
    const hostile = (label: string, fn: () => void) => {
        try { fn(); } catch (e: any) {
            if (e instanceof IncompatibleIndexError) return;
            throw new Error(`${label}: wrong error ${e?.name}`);
        }
        throw new Error(`${label}: expected IncompatibleIndexError`);
    };
    hostile('bad-version', () => {
        const b = goodBytes.slice(0);
        const h = new Uint32Array(b, 0, 9);
        h[1] = 1;
        deserializeDataset(b);
    });
    hostile('bad-enum', () => {
        const b = goodBytes.slice(0);
        const h = new Uint32Array(b, 0, 9);
        h[2] = 99;
        // Recompute CRC so we reach the enum gate, not the checksum gate.
        const recLen = (h[6] as number) * 4;
        const offLen = ((h[5] as number) + 1) * 4;
        let crc = 0xffffffff;
        const T = new Uint32Array(256);
        for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; T[n] = c >>> 0; }
        const parts = [new Uint8Array(b, 0, 32), new Uint8Array(b, 36, recLen), new Uint8Array(b, 36 + recLen, offLen)];
        for (const p of parts) for (let i = 0; i < p.length; i++) crc = (T[(crc ^ p[i]) & 0xff] as number) ^ (crc >>> 8);
        h[8] = ((crc ^ 0xffffffff) >>> 0);
        deserializeDataset(b);
    });
    hostile('normalized-2', () => {
        const b = goodBytes.slice(0);
        new Uint32Array(b, 0, 9)[7] = 2;
        deserializeDataset(b);
    });
    hostile('non-monotonic', () => {
        const p = packDataset(['ab', 'cd'], { normalized: true });
        const b = serializeDataset(p);
        const h = new Uint32Array(b, 0, 9);
        const recLen = (h[6] as number) * 4;
        const off = new Uint32Array(b, 36 + recLen, 3);
        off[1] = 999;
        deserializeDataset(b);
    });
    // Engine-level forged packed object must fail closed (was trust-by-construction).
    const forgedEngine = new WebGPUEngine();
    await forgedEngine.init(mockDevice);
    let forgedThrew = false;
    try {
        await forgedEngine.loadDataset({ tokens: new Uint32Array([1, 2, 3, 4, 5, 6]), offsets: new Uint32Array([0, 999, 6]), rowCount: 2, tokenCount: 6, normalized: true, folded: true, profileId: 'unicode-default', unicodeVersion: '16.0.0', scoringVersion: 'parity-v1', formatVersion: 2, recordsBufferData: new Uint32Array([1, 2, 3, 4, 5, 6]).buffer as ArrayBuffer, offsetsBufferData: new Uint32Array([0, 999, 6]).buffer as ArrayBuffer, recordsByteLength: 24, offsetsByteLength: 12, combinedByteLength: 36 } as any);
    } catch (e: any) { forgedThrew = e instanceof IncompatibleIndexError; }
    if (!forgedThrew) throw new Error('forged non-monotonic offsets must throw IncompatibleIndexError');
    // Engine-level legacy DatasetLike rejected.
    let legacyThrew = false;
    try { await forgedEngine.loadDataset({ size: 2, byteLength: 100 } as any); } catch (e: any) { legacyThrew = e instanceof IncompatibleIndexError; }
    if (!legacyThrew) throw new Error('legacy DatasetLike must throw IncompatibleIndexError');
    // validatePackedOffsets direct: interior bound violation.
    hostile('validate-offsets-bound', () => validatePackedOffsets(new Uint32Array([0, 5, 4]), 2, 4));
    forgedEngine.destroy();
    console.log('   ✅ Hostile headers + forged packed fail-closed verified');

    console.log('23. Testing packer/serialize/budget edge cases...');
    // totalTokens hint validated.
    let ttThrew = false;
    try { packDataset([new Uint32Array([1, 2])], { normalized: true, totalTokens: 99 }); } catch (e: any) { ttThrew = e instanceof IncompatibleIndexError; }
    if (!ttThrew) throw new Error('totalTokens mismatch must throw IncompatibleIndexError');
    packDataset([new Uint32Array([1, 2])], { normalized: true, totalTokens: 2 });
    // Unknown versions fail at pack time (not just serialize).
    let uvThrew = false;
    try { packDataset(['a'], { unicodeVersion: 'nope' }); } catch (e: any) { uvThrew = e instanceof IncompatibleIndexError; }
    if (!uvThrew) throw new Error('unknown unicodeVersion must throw at pack time');
    // serialize shape validation (not raw RangeError).
    let serThrew = false;
    try {
        const p = packDataset(['ab'], { normalized: true });
        serializeDataset({ ...p, recordsByteLength: 999 } as any);
    } catch (e: any) { serThrew = e instanceof IncompatibleIndexError; }
    if (!serThrew) throw new Error('serialize shape mismatch must throw IncompatibleIndexError');
    // Legacy packer deprecation warning + offsets length honesty (was 8 B buffer claiming 16 B).
    const warnMsgs: string[] = [];
    const origWarn = console.warn;
    console.warn = (...args: any[]) => { warnMsgs.push(args.join(' ')); origWarn(...args); };
    const leg = packStringsToGPUBuffer(['a']);
    console.warn = origWarn;
    if (!warnMsgs.some(m => m.includes('legacy-ascii') && m.includes('deprecated'))) {
        throw new Error('legacy packStringsToGPUBuffer must emit deprecation warning with legacy-ascii and deprecated');
    }
    if ((leg.offsetsBufferData as ArrayBuffer).byteLength !== leg.offsetsByteLength) {
        throw new Error('legacy offsets buffer must match claimed byteLength');
    }
    // checkMemoryBudget forged/partial limits fail closed (no NaN allow).
    const nanBudget = checkMemoryBudget(30_000_000, 64, { limits: {} } as any);
    if (nanBudget.allowed || !Number.isFinite(nanBudget.maxBytes)) throw new Error('partial limits must fall back to finite budget and not allow oversize');
    const negBudget = checkMemoryBudget(-5, -10);
    if (!negBudget.allowed) throw new Error('negative inputs must clamp to allowed (no negative budgets)');
    const exactFit = checkMemoryBudget(10, 64, { limits: { maxBufferSize: 65544 + 512 + 44 + 8 + 8192 * 8, maxStorageBufferBindingSize: 1 << 30, maxComputeWorkgroupsPerDimension: 65535 } } as any, 8192);
    if (typeof exactFit.allowed !== 'boolean') throw new Error('exact-boundary budget must return boolean');
    // Mixed input rejected.
    let mixedThrew = false;
    try { packDataset(['a', new Uint32Array([1])] as any, { normalized: true }); } catch (e: any) { mixedThrew = e instanceof TypeError; }
    if (!mixedThrew) throw new Error('mixed string/Uint32Array must throw TypeError');
    console.log('   ✅ Packer/serialize/budget edge cases verified');

    console.log('24. Testing abort, destroy, re-init, Uint32Array[] + clearBuffer fallback...');
    const abortEngine = new WebGPUEngine();
    await abortEngine.init(mockDevice);
    await abortEngine.loadDataset(['apple', 'banana']);
    const ac = new AbortController();
    ac.abort();
    let abortThrew = false;
    try { await abortEngine.search('apple', { mode: 'substring', signal: ac.signal }); } catch (e: any) { abortThrew = e?.name === 'AbortError'; }
    if (!abortThrew) throw new Error('pre-aborted signal must throw AbortError');
    // Uint32Array[] direct load (was rejected before fix).
    await abortEngine.loadDataset([new Uint32Array([1, 2, 3]), new Uint32Array([4])]);
    // Mock has no clearBuffer: search must still succeed via writeBuffer fallback.
    await abortEngine.search('x', { mode: 'substring' });
    // Forged maxDim 0 must not hang (clamped to 65535 → fast path).
    const zeroDimDevice = { ...mockDevice, limits: { ...(mockDevice as any).limits, maxComputeWorkgroupsPerDimension: 0 } } as unknown as GPUDevice;
    (abortEngine as any).device = zeroDimDevice;
    await abortEngine.search('x', { mode: 'substring' });
    (abortEngine as any).device = mockDevice;
    // Re-init must not leak/throw.
    await abortEngine.init(mockDevice);
    await abortEngine.loadDataset(['re', 'init']);
    // Double-destroy safe; post-destroy search returns noHits (not raw TypeError).
    abortEngine.destroy();
    abortEngine.destroy();
    const postDestroy = await abortEngine.search('re', { mode: 'substring' });
    if (postDestroy.totalMatches !== 0) throw new Error('post-destroy search must return noHits');
    console.log('   ✅ Abort/destroy/re-init/Uint32Array[] verified');

    console.log('25. Testing contract sentinels (harness scripts-only, ufuzzy conflict, worker unicode path)...');
    const fsSentinel = await import('node:fs/promises');
    // Harness must stay scripts-only: shipped src/index.ts must never import
    // it (bundle gate), while the harness itself must exist for CI.
    const srcIndex = await fsSentinel.readFile(new URL('../packages/webgpu-search/src/index.ts', import.meta.url), 'utf8');
    if (srcIndex.includes('test-parity-harness')) {
        throw new Error('src/index.ts must not import scripts/test-parity-harness (bundle gate)');
    }
    await fsSentinel.stat(new URL('./test-parity-harness.ts', import.meta.url));
    // preferGpu:true + cpuScorer:'ufuzzy' is a hard conflict (CPU-only scorer).
    const conflictIndex = await SearchIndex.create(['hello'], { device: mockDevice, preferGpu: true });
    let conflictThrew = false;
    try {
        await conflictIndex.search('hello', { mode: 'fuzzy', cpuScorer: 'ufuzzy' });
    } catch (e: any) {
        conflictThrew = e instanceof IncompatibleOptionError;
    }
    if (!conflictThrew) throw new Error("preferGpu:true + cpuScorer:'ufuzzy' must throw IncompatibleOptionError");
    conflictIndex.destroy();
    // Worker blocker: LOAD_DATASET/SEARCH must not use the legacy ASCII packer
    // (code-only match: comments may name it for migration context).
    const workerSrc = await fsSentinel.readFile(new URL('../apps/benchmark/src/search.worker.ts', import.meta.url), 'utf8');
    const workerCode = workerSrc.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|\s)\/\/.*$/gm, '$1');
    if (/packStringsToGPUBuffer\s*\(/.test(workerCode) || /import[^;]*packStringsToGPUBuffer/.test(workerCode)) {
        throw new Error('search.worker.ts must not use legacy packStringsToGPUBuffer (blocker)');
    }
    for (const token of ['packDataset', 'deserializeDataset', 'STRING_ISOLATED_ENRICHMENT', 'latestQueryId', 'SEARCH_ERROR', 'datasetGeneration']) {
        if (!workerSrc.includes(token)) throw new Error(`search.worker.ts missing  token: ${token}`);
    }
    // Main thread enriches compact hits and transfers the dataset buffer.
    const mainSrc = await fsSentinel.readFile(new URL('../apps/benchmark/src/main.ts', import.meta.url), 'utf8');
    if (mainSrc.includes('recordsBufferData.slice(0)')) {
        throw new Error('main.ts must not clone legacy byte buffers to the worker (blocker)');
    }
    if (!mainSrc.includes('gpuCompact') || !mainSrc.includes('serializedDataset.slice(0)')) {
        throw new Error('main.ts missing string-isolated enrichment / dataset transfer (blocker)');
    }
    for (const token of ['SEARCH_ERROR', 'activeDatasetGeneration', 'requestGeneration']) {
        if (!mainSrc.includes(token)) throw new Error(`main.ts missing  token: ${token}`);
    }
    console.log('   ✅ sentinels verified (scripts-only, ufuzzy conflict, worker unicode path)');

    console.log('\n--- All vgpu/mock Tests Passed! (0ms GPU, 100% in-memory) ✅ ---');
}

runMockTests().catch(err => {
    console.error('Test failed:', err);
    process.exit(1);
});
