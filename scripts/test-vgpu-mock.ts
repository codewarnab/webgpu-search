import { createMockAdapter } from 'vgpu/mock';
import {
    WebGPUEngine,
    CPUEngine,
    SearchIndex,
    packStringsToGPUBuffer,
    checkMemoryBudget,
    sanitizeStringForSlot
} from '../packages/webgpu-search/src/index';

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

    console.log('2. Verifying allocated buffers and pipelines...');
    const mockInstrumentation = (mockDevice as any).__vgpuMockInstrumentation;
    if (mockInstrumentation) {
        console.log(`   - Buffers created: ${mockInstrumentation.createBufferDescriptors.length}`);
        console.log(`   - Compute pipelines created: ${mockInstrumentation.createComputePipelineDescriptors.length}`);

        // Verify uniform buffer (272 bytes)
        const uniformBufferDesc = mockInstrumentation.createBufferDescriptors.find(
            (b: any) => b.label === 'Uniform Buffer'
        );
        if (!uniformBufferDesc || uniformBufferDesc.size !== 272) {
            throw new Error(`Uniform buffer size mismatch: expected 272, got ${uniformBufferDesc?.size}`);
        }
        console.log('   ✅ Uniform buffer (272 bytes) verified');

        // Verify candidate output buffer (65,544 bytes = 8 + 8192 * 8)
        const outputBufferDesc = mockInstrumentation.createBufferDescriptors.find(
            (b: any) => b.label === 'Output Buffer'
        );
        if (!outputBufferDesc || outputBufferDesc.size !== 65544) {
            throw new Error(`Output buffer size mismatch: expected 65544 (8192 candidates), got ${outputBufferDesc?.size}`);
        }
        console.log('   ✅ Candidate output buffer (8,192 slots = 65,544 bytes) verified');

        // Verify staging buffer
        const stagingBufferDesc = mockInstrumentation.createBufferDescriptors.find(
            (b: any) => b.label === 'Staging Buffer'
        );
        if (!stagingBufferDesc || stagingBufferDesc.size !== 65544) {
            throw new Error(`Staging buffer size mismatch: expected 65544, got ${stagingBufferDesc?.size}`);
        }
        console.log('   ✅ Staging buffer (65,544 bytes) verified');
    }

    console.log('3. Testing buffer packer and mock VRAM loading...');
    const strings = generateTestStrings(500);
    const packed = packStringsToGPUBuffer(strings, 64);
    const { uploadTimeMs } = await engine.loadDataset({
        size: strings.length,
        strings,
        gpuBufferData: packed.bufferData,
        byteLength: packed.byteLength
    });
    console.log(`   ✅ Dataset loaded (500 items, ${packed.byteLength} bytes, ${uploadTimeMs.toFixed(2)}ms)`);

    console.log('4. Testing search execution and empty query handling...');
    const emptyRes = await engine.search('', { mode: 'substring' });
    if (emptyRes.totalMatches !== 0 || emptyRes.results.length !== 0) {
        throw new Error('Empty query test failed');
    }
    console.log('   ✅ Empty query edge-case handled properly');

    engine.destroy();
    console.log('   ✅ Low-level engine resources cleaned up via engine.destroy()');

    console.log('5. Testing high-level SearchIndex API with injected mock device...');
    const searchIndex = await SearchIndex.create(strings, {
        device: mockDevice,
        preferGpu: true
    });

    const stats = searchIndex.getStats();
    if (stats.engine !== 'webgpu') {
        throw new Error(`Expected SearchIndex engine to be 'webgpu', got '${stats.engine}'`);
    }
    console.log(`   ✅ SearchIndex created with engine: ${stats.engine}, VRAM: ${stats.vramAllocatedBytes} bytes`);

    const gpuSearchRes = await searchIndex.search('AuthController', { mode: 'fuzzy', limit: 20 });
    if (gpuSearchRes.engine !== 'webgpu') {
        throw new Error(`Expected search response engine to be 'webgpu', got '${gpuSearchRes.engine}'`);
    }
    console.log(`   ✅ SearchIndex search routed to WebGPU successfully (query: '${gpuSearchRes.query}')`);
    searchIndex.destroy();

    console.log('6. Testing high-level SearchIndex CPU auto-routing fallback...');
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
    // Verify unified score contract
    for (const r of cpuRes.results) {
        if (typeof r.score !== 'number' || typeof r.index !== 'number' || typeof r.text !== 'string') {
            throw new Error(`Invalid SearchResultItem contract: ${JSON.stringify(r)}`);
        }
    }
    console.log(`   ✅ CPU auto-routing verified (found ${cpuRes.results.length} matches with normalized scores)`);
    cpuIndex.destroy();

    console.log('7. Testing searchCold pipeline on WebGPUEngine...');
    const coldEngine = new WebGPUEngine();
    await coldEngine.init(mockDevice);
    const coldRes = await coldEngine.searchCold({
        size: strings.length,
        strings,
        gpuBufferData: packed.bufferData,
        byteLength: packed.byteLength
    }, 'Auth', { mode: 'substring' });
    if (typeof coldRes.datasetUploadMs !== 'number' || typeof coldRes.coldTotalMs !== 'number') {
        throw new Error('searchCold failed to report datasetUploadMs or coldTotalMs');
    }
    console.log(`   ✅ searchCold verified (coldTotalMs: ${coldRes.coldTotalMs.toFixed(2)}ms, upload: ${coldRes.datasetUploadMs.toFixed(2)}ms)`);
    coldEngine.destroy();

    console.log('8. Testing checkMemoryBudget and sanitizeStringForSlot...');
    const underBudget = checkMemoryBudget(10_000, 64, mockDevice);
    if (!underBudget.allowed) throw new Error('Expected 10,000 items to fit in budget');
    const overBudget = checkMemoryBudget(30_000_000, 64, mockDevice);
    if (overBudget.allowed) throw new Error('Expected 30M items to exceed 128MB budget');
    console.log('   ✅ checkMemoryBudget correctly enforces hardware allocation limits');

    const sanitized = sanitizeStringForSlot('Café crème naïve 🚀', 59);
    if (sanitized !== 'Cafe creme naive ??') {
        throw new Error(`Sanitization failed: got "${sanitized}"`);
    }
    console.log(`   ✅ sanitizeStringForSlot correctly strips diacritics and replaces non-ASCII: "${sanitized}"`);

    console.log('9. Testing empty dataset edge-case on SearchIndex...');
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

    console.log('10. Testing caseSensitivity parity on CPUEngine...');
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

    console.log('11. Testing SearchIndex limit contract on CPU...');
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

    console.log('12. Testing variable-length string support (>100 characters without truncation)...');
    const veryLongString = 'packages/core/extremely/long/nested/folder/path/to/some/deeply/embedded/internal/structure/that/exceeds/the/old/fiftynine/limit/DeepSpecialController.ts';
    if (veryLongString.length <= 100) {
        throw new Error('Test string must be > 100 chars');
    }
    const sanitizedLong = sanitizeStringForSlot(veryLongString);
    if (sanitizedLong.length !== veryLongString.length) {
        throw new Error(`Expected sanitizeStringForSlot without maxChars to preserve full length (${veryLongString.length}), got ${sanitizedLong.length}`);
    }

    const dynamicPacked = packStringsToGPUBuffer([veryLongString]);
    if (dynamicPacked.totalChars !== veryLongString.length) {
        throw new Error(`Expected packed totalChars to be ${veryLongString.length}, got ${dynamicPacked.totalChars}`);
    }
    if (dynamicPacked.recordsByteLength < veryLongString.length) {
        throw new Error('Records buffer byte length is smaller than string length');
    }

    const longIndex = await SearchIndex.create([veryLongString], { preferGpu: false });
    const longSearch = await longIndex.search('DeepSpecialController');
    if (longSearch.results.length === 0 || longSearch.results[0].text !== veryLongString) {
        throw new Error('Failed to match search token located past character position 100');
    }
    longIndex.destroy();
    console.log(`   ✅ Variable-length strings verified (${veryLongString.length} chars preserved and matched past char 100)`);

    console.log('13. Testing direct string[] callers on WebGPUEngine.loadDataset and searchCold...');
    const directEngine = new WebGPUEngine();
    await directEngine.init(mockDevice);
    await directEngine.loadDataset(['apple', 'banana', 'orange']);
    const directColdRes = await directEngine.searchCold(['alpha', 'beta', 'gamma'], 'beta', { mode: 'substring' });
    if (typeof directColdRes.coldTotalMs !== 'number') {
        throw new Error('searchCold with string[] failed');
    }
    directEngine.destroy();
    console.log('   ✅ Direct string[] callers on WebGPUEngine verified');

    console.log('\n--- All vgpu/mock Tests Passed! (0ms GPU, 100% in-memory) ✅ ---');
}

runMockTests().catch(err => {
    console.error('Test failed:', err);
    process.exit(1);
});

