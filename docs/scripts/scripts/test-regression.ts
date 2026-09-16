import puppeteer from 'puppeteer-core';
import path from 'path';
import fs from 'fs';

function getChromeExecutablePath(): string {
    if (process.env.CHROME_BIN && fs.existsSync(process.env.CHROME_BIN)) {
        return process.env.CHROME_BIN;
    }
    if (process.env.PUPPETEER_EXECUTABLE_PATH && fs.existsSync(process.env.PUPPETEER_EXECUTABLE_PATH)) {
        return process.env.PUPPETEER_EXECUTABLE_PATH;
    }

    const platform = process.platform;
    if (platform === 'win32') {
        const progFiles = process.env.PROGRAMFILES || 'C:\\Program Files';
        const progFilesX86 = process.env['PROGRAMFILES(X86)'] || 'C:\\Program Files (x86)';
        const localAppData = process.env.LOCALAPPDATA || '';
        const candidates = [
            path.join(progFiles, 'Google\\Chrome\\Application\\chrome.exe'),
            path.join(progFilesX86, 'Google\\Chrome\\Application\\chrome.exe'),
            path.join(localAppData, 'Google\\Chrome\\Application\\chrome.exe'),
            path.join(progFiles, 'Microsoft\\Edge\\Application\\msedge.exe')
        ];
        for (const p of candidates) {
            if (fs.existsSync(p)) return p;
        }
    } else if (platform === 'darwin') {
        const candidates = [
            '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
            '/Applications/Chromium.app/Contents/MacOS/Chromium'
        ];
        for (const p of candidates) {
            if (fs.existsSync(p)) return p;
        }
    } else {
        const candidates = [
            '/usr/bin/google-chrome-stable',
            '/usr/bin/google-chrome',
            '/usr/bin/chromium-browser',
            '/usr/bin/chromium',
            '/snap/bin/chromium'
        ];
        for (const p of candidates) {
            if (fs.existsSync(p)) return p;
        }
    }

    throw new Error('Could not automatically find Chrome executable. Please set CHROME_BIN environment variable.');
}

async function main() {
    console.log('--- Running WebGPU Top-K & Timing Regression Tests ---');
    const chromePath = getChromeExecutablePath();
    console.log(`Using Chrome binary: ${chromePath}`);

    const args = [
        '--enable-unsafe-webgpu',
        '--enable-features=Vulkan,DefaultANGLEVulkan,WebGPU',
        '--enable-gpu-rasterization',
        '--no-sandbox',
        '--disable-setuid-sandbox'
    ];
    if (process.platform === 'win32') {
        args.push('--use-angle=d3d11');
    }

    const browser = await puppeteer.launch({
        executablePath: chromePath,
        headless: 'new',
        args
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
        const moduleUrl = `/@fs/${path.resolve('packages/webgpu-search/src/index.ts').replace(/\\/g, '/')}`;
        const testResults = await page.evaluate(async (moduleUrl) => {
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

            // 4. Test: CPU/GPU result parity for the supported ASCII contract.
            const { SearchIndex, WebGPUContextManager } = await import(moduleUrl);
            const parityItems = [
                'src/AuthController.ts',
                'src/auth-controller.test.ts',
                'src/AuthorizationService.ts',
                'docs/Authentication.md',
                'src/BillingController.ts'
            ];
            const gpuIndex = await SearchIndex.create(parityItems, { device: engine.device, preferGpu: true });
            const cpuIndex = await SearchIndex.create(parityItems, { preferGpu: false, threshold: Number.MAX_SAFE_INTEGER });
            for (const parityCase of [
                { query: 'Auth', mode: 'substring' as const, caseSensitive: false },
                { query: 'Auth', mode: 'substring' as const, caseSensitive: true },
                { query: 'athctl', mode: 'fuzzy' as const, caseSensitive: false }
            ]) {
                const [gpu, cpu] = await Promise.all([
                    gpuIndex.search(parityCase.query, { ...parityCase, limit: 5 }),
                    cpuIndex.search(parityCase.query, { ...parityCase, limit: 5 })
                ]);
                const gpuIndices = gpu.results.map((r: any) => r.index);
                const cpuIndices = cpu.results.map((r: any) => r.index);
                results.push({
                    name: `CPU/GPU parity: ${parityCase.mode}, caseSensitive=${parityCase.caseSensitive}`,
                    passed: JSON.stringify(gpuIndices) === JSON.stringify(cpuIndices),
                    details: { query: parityCase.query, gpuIndices, cpuIndices }
                });
            }

            // 5. Test: concurrent searches remain isolated and return query-specific results.
            const concurrent = await Promise.all([
                gpuIndex.search('Auth', { mode: 'substring', limit: 5 }),
                gpuIndex.search('Billing', { mode: 'substring', limit: 5 }),
                gpuIndex.search('Controller', { mode: 'substring', limit: 5 })
            ]);
            results.push({
                name: 'Concurrent searches are serialized without result cross-talk',
                passed: concurrent[0].query === 'Auth' &&
                    concurrent[1].query === 'Billing' &&
                    concurrent[2].query === 'Controller' &&
                    concurrent[1].results.every((r: any) => r.text.includes('Billing')),
                details: concurrent.map((r: any) => ({ query: r.query, indices: r.results.map((x: any) => x.index) }))
            });

            // 6. Test: a per-query GPU failure falls back to CPU with the same index.
            const fallbackIndex = await SearchIndex.create(parityItems, { device: engine.device, preferGpu: true });
            const originalSearch = (fallbackIndex as any).gpuEngine.search.bind((fallbackIndex as any).gpuEngine);
            (fallbackIndex as any).gpuEngine.search = async () => { throw new Error('injected GPU failure'); };
            const fallbackResult = await fallbackIndex.search('Billing', { mode: 'substring', limit: 5 });
            results.push({
                name: 'GPU query failure falls back to CPU',
                passed: fallbackResult.engine === 'cpu' && fallbackResult.results[0]?.text === 'src/BillingController.ts',
                details: { engine: fallbackResult.engine, results: fallbackResult.results }
            });
            (fallbackIndex as any).gpuEngine.search = originalSearch;

            // 7. Test: a device-loss notification switches existing indexes to CPU.
            const listeners: Set<(reason: string) => void> = (WebGPUContextManager as any).deviceLostListeners;
            for (const listener of [...listeners]) listener('injected device loss');
            const afterLoss = await gpuIndex.search('Billing', { mode: 'substring', limit: 5 });
            results.push({
                name: 'Device loss switches existing SearchIndex to CPU fallback',
                passed: gpuIndex.getStats().engine === 'cpu' && afterLoss.engine === 'cpu',
                details: { stats: gpuIndex.getStats(), responseEngine: afterLoss.engine }
            });

            fallbackIndex.destroy();
            gpuIndex.destroy();
            cpuIndex.destroy();

            return { success: true, results };
        }, moduleUrl);

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
