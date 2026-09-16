import puppeteer from 'puppeteer-core';
import path from 'path';

async function main() {
    console.log('--- Running WebGPU Top-K & Timing Regression Tests ---');
    const chromePath = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';

    const browser = await puppeteer.launch({
        executablePath: chromePath,
        headless: 'new',
        args: [
            '--enable-unsafe-webgpu',
            '--use-angle=d3d11',
            '--enable-features=Vulkan,DefaultANGLEVulkan,WebGPU',
            '--enable-gpu-rasterization',
            '--no-sandbox',
            '--disable-setuid-sandbox'
        ]
    });

    try {
        const page = await browser.newPage();

        page.on('console', msg => {
            const text = msg.text();
            if (text.includes('Error') || text.includes('WebGPU') || text.includes('Failed')) {
                console.log('[Browser Console]', text);
            }
        });

        let loaded = false;
        for (let i = 0; i < 10; i++) {
            try {
                await page.goto('http://localhost:5173/', { waitUntil: 'networkidle0', timeout: 5000 });
                loaded = true;
                break;
            } catch (err) {
                await new Promise(r => setTimeout(r, 1000));
            }
        }
        if (!loaded) {
            throw new Error('Failed to connect to http://localhost:5173/ after multiple attempts');
        }

        // Evaluate inside page context where WebGPU engine is initialized
        const testResults = await page.evaluate(async () => {
            const engine = (window as any).gpuEngine;
            if (!engine || !engine.isReady) {
                return { success: false, reason: 'WebGPU Engine not ready' };
            }

            const results: { name: string; passed: boolean; details?: any }[] = [];

            // 1. Test: Timing metrics separation
            const searchRes = await engine.search('Controller', { mode: 'substring', maxResults: 100 });
            const timings = searchRes.timings;
            const hasProperTimings = (
                typeof timings.encodeSubmitMs === 'number' &&
                typeof timings.readbackMs === 'number' &&
                typeof timings.totalMs === 'number' &&
                (timings.gpuExecutionMs === null || typeof timings.gpuExecutionMs === 'number')
            );
            results.push({
                name: 'Timing metrics separated (Submit, Exec, Readback, Total)',
                passed: hasProperTimings,
                details: timings
            });

            // 2. Test: Candidate collection across >1000 items with late high-scoring matches
            // Generate synthetic dataset of 2,500 items where items 0..1199 have low scores and 1200..1250 have high scores
            const count = 3000;
            const strings = new Array<string>(count);
            const byteLength = count * 64;
            const gpuBufferData = new ArrayBuffer(byteLength);
            const u32View = new Uint32Array(gpuBufferData);
            const u8View = new Uint8Array(gpuBufferData);

            for (let i = 0; i < count; i++) {
                // For i < 1200: path has match at end (low substring score): "long_prefix_path_controller"
                // For i >= 1200 && i < 1250: starts with exact query (highest substring score): "Controller_Special"
                const str = i >= 1200 && i < 1250
                    ? `Controller_Special_${i}.ts`
                    : `very_long_path_prefix_folder_name/Controller_${i}.ts`;
                strings[i] = str;

                const strLen = Math.min(str.length, 59);
                u32View[i * 16] = strLen;
                const baseByte = i * 64 + 4;
                for (let c = 0; c < strLen; c++) {
                    u8View[baseByte + c] = str.charCodeAt(c);
                }
            }

            await engine.loadDataset({
                size: count,
                strings,
                gpuBufferData,
                byteLength
            });

            const topResults = await engine.search('Controller', { mode: 'substring', maxResults: 50 });
            // The top results must contain the late high-scoring items (indices 1200-1249)
            const containsLateHighScorers = topResults.results.slice(0, 10).every((r: any) => r.index >= 1200 && r.index < 1250);
            results.push({
                name: 'Broad query (>1000 matches) preserves late high-scoring matches in top-K',
                passed: containsLateHighScorers,
                details: {
                    totalMatches: topResults.totalMatches,
                    top1Index: topResults.results[0]?.index,
                    top1Score: topResults.results[0]?.score,
                    candidateCount: topResults.candidateCount
                }
            });

            // 3. Test: Candidate overflow detection (>8192 items)
            const bigCount = 10000;
            const bigGpuBuffer = new ArrayBuffer(bigCount * 64);
            const bigU32 = new Uint32Array(bigGpuBuffer);
            const bigU8 = new Uint8Array(bigGpuBuffer);
            const bigStrings = new Array<string>(bigCount);

            for (let i = 0; i < bigCount; i++) {
                const s = `TestItem_${i}.ts`;
                bigStrings[i] = s;
                bigU32[i * 16] = s.length;
                const base = i * 64 + 4;
                for (let c = 0; c < s.length; c++) bigU8[base + c] = s.charCodeAt(c);
            }

            await engine.loadDataset({
                size: bigCount,
                strings: bigStrings,
                gpuBufferData: bigGpuBuffer,
                byteLength: bigCount * 64
            });

            const overflowRes = await engine.search('TestItem', { mode: 'substring', maxResults: 100 });
            results.push({
                name: 'Candidate pool overflow (>8192 matches) detected and reported',
                passed: overflowRes.hasOverflow === true && overflowRes.candidateCount === 8192,
                details: {
                    totalMatches: overflowRes.totalMatches,
                    hasOverflow: overflowRes.hasOverflow,
                    candidateCount: overflowRes.candidateCount
                }
            });

            return { success: true, results };
        });

        console.log('\n--- Test Suite Summary ---');
        console.dir(testResults, { depth: null });

        if (!testResults.success || testResults.results?.some((r: any) => !r.passed)) {
            console.error('Some tests failed!');
            process.exit(1);
        } else {
            console.log('\nAll regression tests passed successfully! ✅');
        }
    } finally {
        await browser.close();
    }
}

main().catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
});
