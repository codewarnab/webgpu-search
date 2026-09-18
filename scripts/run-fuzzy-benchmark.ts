import puppeteer from 'puppeteer-core';
import fs from 'fs';
import path from 'path';
import {
    getChromeExecutablePath,
    getChromeLaunchArgs,
    ensureBenchmarkServer,
    formatBenchmarkMarkdown
} from './browser-utils';

async function main() {
    console.log('--- WebGPU Fuzzy Search: Single-Algorithm Benchmark Runner (Fuzzy) ---');

    const args = process.argv.slice(2);
    const getArg = (name: string): string | undefined => {
        const prefix = `--${name}=`;
        const item = args.find(a => a.startsWith(prefix));
        return item ? item.slice(prefix.length) : undefined;
    };
    const isFast = args.includes('--fast');

    const sizesArg = getArg('sizes') || (isFast ? '10000,100000' : '10000,100000,500000,1000000,2000000');
    const corpusArg = getArg('corpus') || 'ascii';
    const queryArg = getArg('query');
    const warmupsArg = getArg('warmups') || (isFast ? '2' : '5');
    const samplesArg = getArg('samples') || (isFast ? '5' : '20');

    const chromePath = getChromeExecutablePath();
    console.log(`Discovered Chrome binary: ${chromePath}`);

    const server = await ensureBenchmarkServer(5173);

    const queryParams = new URLSearchParams({
        autorun: 'fuzzy',
        sizes: sizesArg,
        corpus: corpusArg,
        warmups: warmupsArg,
        samples: samplesArg
    });
    if (queryArg) {
        queryParams.set('query', queryArg);
    }

    const targetUrl = `${server.url}/?${queryParams.toString()}`;
    console.log(`Launching headless browser with WebGPU flags...`);

    const browser = await puppeteer.launch({
        executablePath: chromePath,
        headless: 'new',
        args: getChromeLaunchArgs(),
        protocolTimeout: 0
    });

    try {
        const page = await browser.newPage();
        await page.setViewport({ width: 1280, height: 950 });

        page.on('console', msg => {
            console.log(`[Browser Console ${msg.type()}]`, msg.text());
        });

        page.on('pageerror', err => {
            console.error('[Browser PageError]', err);
        });

        console.log(`Navigating to: ${targetUrl}`);
        await page.goto(targetUrl, { waitUntil: 'networkidle0', timeout: 30000 });

        console.log('Waiting for fuzzy benchmark completion...');
        const resultsHandle = await page.waitForFunction(() => {
            if ((window as any).__BENCHMARK_ERROR__) {
                throw new Error(`Benchmark execution failed in browser: ${(window as any).__BENCHMARK_ERROR__}`);
            }
            const res = (window as any).__BENCHMARK_RESULTS__;
            if (!res) return false;
            if (Array.isArray(res) && res.length > 0) return res;
            if (res.fuzzy && res.fuzzy.length > 0) return res;
            return false;
        }, { timeout: 360000, polling: 1000 });

        const rawData = await resultsHandle.jsonValue() as any;
        const rows: any[] = Array.isArray(rawData) ? rawData : (rawData.fuzzy || []);

        console.log('\n================ BENCHMARK RESULTS (FUZZY) ================\n');
        console.table(rows.map(r => ({
            Size: (r.datasetSize || 0).toLocaleString(),
            Corpus: r.corpusType || 'ascii',
            'VRAM (MB)': r.vramAllocation?.totalBytes ? (r.vramAllocation.totalBytes / (1024 * 1024)).toFixed(2) : 'N/A',
            'GPU Ret (med/p95)': r.gpuRetained ? `${r.gpuRetained.medianMs.toFixed(2)} / ${r.gpuRetained.p95Ms.toFixed(2)} ms` : 'N/A',
            'CPU Parity (med/p95)': r.cpuParity ? `${r.cpuParity.medianMs.toFixed(2)} / ${r.cpuParity.p95Ms.toFixed(2)} ms` : `${(r.cpuParityMs || 0).toFixed(2)} ms`,
            'uFuzzy (med)': r.ufuzzy ? `${r.ufuzzy.medianMs.toFixed(2)} ms` : `${(r.ufuzzyMs || 0).toFixed(2)} ms`,
            'JS Native (med)': r.jsNative ? `${r.jsNative.medianMs.toFixed(2)} ms` : `${(r.jsNativeMs || 0).toFixed(2)} ms`,
            'Speedup vs uFuzzy': `${r.retainedVsUfuzzySpeedup || 0}x`,
            'Speedup vs Parity': `${r.retainedVsParitySpeedup || 0}x`,
            Status: r.qualificationStatus || 'pending-hardware'
        })));

        // Screenshot
        const screenshotPath = path.resolve('fuzzy-benchmark-results.png');
        await page.screenshot({ path: screenshotPath, fullPage: true });
        console.log(`Saved screenshot to: ${screenshotPath}`);

        // Save JSON
        const jsonPath = path.resolve('fuzzy-benchmark-results.json');
        fs.writeFileSync(jsonPath, JSON.stringify(rawData, null, 2));
        console.log(`Saved JSON data to: ${jsonPath}`);

        console.log('\nFuzzy benchmark complete!');
    } finally {
        await browser.close().catch(() => {});
        await server.close().catch(() => {});
    }
    process.exit(0);
}

main().catch(err => {
    console.error('Fatal error running fuzzy benchmark:', err);
    process.exit(1);
});
