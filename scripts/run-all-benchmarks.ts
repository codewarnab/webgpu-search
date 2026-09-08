import puppeteer from 'puppeteer-core';
import fs from 'fs';
import path from 'path';

async function main() {
    console.log('Launching Chrome with WebGPU enabled...');
    const chromePath = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe';

    const browser = await puppeteer.launch({
        executablePath: chromePath,
        headless: 'new',
        args: [
            '--enable-unsafe-webgpu',
            '--use-angle=d3d11',
            '--enable-features=Vulkan,DefaultANGLEVulkan,WebGPU',
            '--enable-gpu-rasterization',
            '--window-size=1280,950',
            '--no-sandbox',
            '--disable-setuid-sandbox'
        ]
    });

    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 950 });

    page.on('console', msg => {
        const text = msg.text();
        if (text.includes('WebGPU') || text.includes('Benchmark') || text.includes('Error')) {
            console.log('[Browser Console]', text);
        }
    });

    console.log('Navigating to http://127.0.0.1:5173/?autorun=all ...');
    await page.goto('http://127.0.0.1:5173/?autorun=all', { waitUntil: 'networkidle0' });

    console.log('Waiting for dual-algorithm benchmark to finish (Substring + Fuzzy)...');
    const resultsHandle = await page.waitForFunction(() => {
        const res = (window as any).__BENCHMARK_RESULTS__;
        return res && res.substring && res.fuzzy && res.substring.length > 0 && res.fuzzy.length > 0 ? res : false;
    }, { timeout: 240000 });

    const benchmarkData = await resultsHandle.jsonValue();

    console.log('\n================ SUBSTRING RESULTS ================\n');
    console.table(benchmarkData.substring.map((r: any) => ({
        Size: r.datasetSize.toLocaleString(),
        'GPU Retained (ms)': r.gpuRetained.totalMs,
        'uFuzzy (ms)': r.ufuzzyMs,
        'JS Native (ms)': r.jsNativeMs,
        'GPU Cold (ms)': r.gpuCold.totalMs,
        Speedup: `${r.retainedVsUfuzzySpeedup}x`,
        Winner: r.crossover.gpuRetainedBeatsUfuzzy ? 'GPU' : 'CPU'
    })));

    console.log('\n================ FUZZY RESULTS ================\n');
    console.table(benchmarkData.fuzzy.map((r: any) => ({
        Size: r.datasetSize.toLocaleString(),
        'GPU Retained (ms)': r.gpuRetained.totalMs,
        'uFuzzy (ms)': r.ufuzzyMs,
        'JS Native (ms)': r.jsNativeMs,
        'GPU Cold (ms)': r.gpuCold.totalMs,
        Speedup: `${r.retainedVsUfuzzySpeedup}x`,
        Winner: r.crossover.gpuRetainedBeatsUfuzzy ? 'GPU' : 'CPU'
    })));

    // Take screenshot
    const screenshotPath = path.resolve('full-benchmark-ui.png');
    await page.screenshot({ path: screenshotPath, fullPage: true });
    console.log(`\nScreenshot saved to: ${screenshotPath}`);

    // Trigger CSV export and verify download or write directly
    fs.writeFileSync('benchmark_results.json', JSON.stringify(benchmarkData, null, 2));
    console.log('Raw JSON saved to: benchmark_results.json');

    await browser.close();
    console.log('\nAll benchmarks complete!');
}

main().catch(err => {
    console.error('Error running benchmarks:', err);
    process.exit(1);
});
