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
    console.log('--- WebGPU Fuzzy Search: Dual-Algorithm Benchmark Runner ---');

    // Parse optional CLI arguments
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

    // Ensure Vite server is active
    const server = await ensureBenchmarkServer(5173);

    const queryParams = new URLSearchParams({
        autorun: 'all',
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

        console.log('Benchmark running in browser context. Waiting for results...');
        const resultsHandle = await page.waitForFunction(() => {
            if ((window as any).__BENCHMARK_ERROR__) {
                throw new Error(`Benchmark execution failed in browser: ${(window as any).__BENCHMARK_ERROR__}`);
            }
            const res = (window as any).__BENCHMARK_RESULTS__;
            if (!res) return false;
            const hasSub = Array.isArray(res.substring) && res.substring.length > 0;
            const hasFuz = Array.isArray(res.fuzzy) && res.fuzzy.length > 0;
            return hasSub || hasFuz ? res : false;
        }, { timeout: 360000, polling: 1000 });

        const benchmarkData = await resultsHandle.jsonValue() as any;

        const formatRowForConsole = (r: any) => ({
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
        });

        if (benchmarkData.substring && benchmarkData.substring.length > 0) {
            console.log('\n================ EXACT SUBSTRING BENCHMARK RESULTS ================');
            console.table(benchmarkData.substring.map(formatRowForConsole));
        }

        if (benchmarkData.fuzzy && benchmarkData.fuzzy.length > 0) {
            console.log('\n================ FUZZY SUBSEQUENCE BENCHMARK RESULTS ================');
            console.table(benchmarkData.fuzzy.map(formatRowForConsole));
        }

        // Check qualification status
        const isQualified = benchmarkData.qualificationStatus === 'qualified';
        if (!isQualified) {
            console.warn('\n[HARDWARE QUALIFICATION NOTICE]');
            console.warn('Measurements ran on software adapter or mock WebGPU device.');
            console.warn('Cells are flagged as "pending-hardware" per unicode-contract.md §5.\n');
        } else {
            console.log('\n[HARDWARE QUALIFICATION]');
            console.log('Measurements qualified on physical GPU device.\n');
        }

        // Export Screenshots
        try {
            const screenshotPath = path.resolve('full-benchmark-ui.png');
            await page.screenshot({ path: screenshotPath, fullPage: true });
            console.log(`Saved screenshot: ${screenshotPath}`);

            await page.click('#tab-btn-substring');
            await new Promise(r => setTimeout(r, 100));
            const subChartEl = await page.$('#benchmark-chart-substring');
            if (subChartEl) {
                await subChartEl.screenshot({ path: path.resolve('chart_substring_benchmark.png') });
                console.log('Exported chart: chart_substring_benchmark.png');
            }

            await page.click('#tab-btn-fuzzy');
            await new Promise(r => setTimeout(r, 100));
            const fuzChartEl = await page.$('#benchmark-chart-fuzzy');
            if (fuzChartEl) {
                await fuzChartEl.screenshot({ path: path.resolve('chart_fuzzy_benchmark.png') });
                console.log('Exported chart: chart_fuzzy_benchmark.png');
            }
        } catch (e) {
            console.warn('Could not save some chart screenshots:', e);
        }

        // Save raw JSON and formatted Markdown summary
        const jsonPath = path.resolve('benchmark_results.json');
        fs.writeFileSync(jsonPath, JSON.stringify(benchmarkData, null, 2));
        console.log(`Saved benchmark JSON to: ${jsonPath}`);

        const mdContent = formatBenchmarkMarkdown(benchmarkData);
        const mdPath = path.resolve('benchmark_summary.md');
        fs.writeFileSync(mdPath, mdContent);
        console.log(`Saved benchmark Markdown summary to: ${mdPath}`);

        console.log('\nAll benchmarks complete successfully!');
    } finally {
        await browser.close().catch(() => {});
        await server.close().catch(() => {});
    }
    process.exit(0);
}

main().catch(err => {
    console.error('Fatal error running benchmark suite:', err);
    process.exit(1);
});
