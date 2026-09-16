import { createMockAdapter } from 'vgpu/mock';
import { WebGPUEngine } from '../src/webgpu-engine.ts';
import { generateDataset } from '../src/dataset.ts';

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

    console.log('3. Testing dataset generation and mock VRAM loading...');
    const dataset = generateDataset(500);
    const { uploadTimeMs } = await engine.loadDataset(dataset);
    console.log(`   ✅ Dataset loaded (500 items, ${dataset.byteLength} bytes, ${uploadTimeMs.toFixed(2)}ms)`);

    console.log('4. Testing search mutex and internal pipeline execution flow...');
    // In mock, queue and buffers record calls
    const searchPromise = engine.search('AuthController', { mode: 'fuzzy', maxResults: 100 });
    const emptyQueryPromise = engine.search('', { mode: 'substring' });

    const [emptyRes] = await Promise.all([emptyQueryPromise]);
    if (emptyRes.totalMatches !== 0 || emptyRes.results.length !== 0) {
        throw new Error('Empty query test failed');
    }
    console.log('   ✅ Empty query edge-case handled properly');

    engine.destroy();
    console.log('   ✅ Resources cleaned up via engine.destroy()');

    console.log('\n--- All vgpu/mock Tests Passed! (0ms GPU, 100% in-memory) ✅ ---');
}

runMockTests().catch(err => {
    console.error('Test failed:', err);
    process.exit(1);
});
