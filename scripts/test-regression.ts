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
            // Generate synthetic dataset of 3,000 variable-length items where
            // items 1200..1249 start with the query (highest substring score).
            // M3: variable-length u32 packing — plain strings in, the engine
            // normalizes to post-fold tokens (no 59-char truncation, no slots).
            const count = 3000;
            const strings = new Array<string>(count);

            for (let i = 0; i < count; i++) {
                // For i < 1200 or i >= 1250: match starts late (low substring score)
                // For 1200 <= i < 1250: starts with exact query (highest substring score)
                strings[i] = i >= 1200 && i < 1250
                    ? `Controller_Special_${i}.ts`
                    : `very_long_path_prefix_folder_name/Controller_${i}.ts`;
            }

            await engine.loadDataset({
                size: count,
                strings
            });

            const topResults = await engine.search('Controller', { mode: 'substring', maxResults: 50 });
            // The top results must contain the late high-scoring items (indices 1200-1249).
            // Set-membership (>=8/10) instead of exact order: identical-prefix
            // ties are stable today but brittle to scoring tweaks.
            const top10 = topResults.results.slice(0, 10);
            const inRange = top10.filter((r: any) => r.index >= 1200 && r.index < 1250).length;
            const containsLateHighScorers = inRange >= 8;
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
            const bigStrings = new Array<string>(bigCount);

            for (let i = 0; i < bigCount; i++) {
                bigStrings[i] = `TestItem_${i}.ts`;
            }

            await engine.loadDataset({
                size: bigCount,
                strings: bigStrings
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

            // 4. Test: unicode variable-length packing (astral/empty/whitespace rows).
            const FCP = String.fromCodePoint;
            const uniStrings = [
                'hello',
                'stra' + FCP(0xdf) + 'e',
                FCP(0x1f600),
                '',
                '   ',
                'packages/core/' + FCP(0x4eac) + '/DeepSpecialController.ts',
            ];
            await engine.loadDataset({ size: uniStrings.length, strings: uniStrings });
            const uniRes = await engine.search('DeepSpecial', { mode: 'substring', maxResults: 10 });
            const uniOk = uniRes.totalMatches >= 1 && uniRes.results.some((r: any) => r.text.includes('DeepSpecial'));
            results.push({
                name: 'Unicode rows (astral/empty/whitespace) pack + search without truncation',
                passed: uniOk,
                details: { totalMatches: uniRes.totalMatches, top: uniRes.results[0] }
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
