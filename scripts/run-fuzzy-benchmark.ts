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
            '--window-size=1280,900',
            '--no-sandbox',
            '--disable-setuid-sandbox'
        ]
    });

    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 900 });

    page.on('console', msg => {
        const text = msg.text();
        if (text.includes('WebGPU') || text.includes('Benchmark') || text.includes('Error')) {
            console.log('[Browser Console]', text);
        }
    });

    console.log('Navigating to http://127.0.0.1:5173/?autorun=fuzzy ...');
    await page.goto('http://127.0.0.1:5173/?autorun=fuzzy', { waitUntil: 'networkidle0' });

    console.log('Waiting for WebGPU initialization and benchmark completion...');
    // Poll for window.__BENCHMARK_RESULTS__
    const results = await page.waitForFunction(() => {
        return (window as any).__BENCHMARK_RESULTS__;
    }, { timeout: 180000 });

    const benchmarkData = await results.jsonValue();
    console.log('\n================ BENCHMARK RESULTS (FUZZY) ================\n');
    console.table(benchmarkData.map((r: any) => ({
        Size: r.datasetSize.toLocaleString(),
        'GPU Retained (ms)': r.gpuRetained.totalMs,
        'uFuzzy (ms)': r.ufuzzyMs,
        'JS Native (ms)': r.jsNativeMs,
        'GPU Cold (ms)': r.gpuCold.totalMs,
        Speedup: `${r.retainedVsUfuzzySpeedup}x`,
        Winner: r.crossover.gpuRetainedBeatsUfuzzy ? 'GPU' : 'CPU'
    })));

    // Take a screenshot of the benchmark UI
    const screenshotPath = path.resolve('fuzzy-benchmark-results.png');
    await page.screenshot({ path: screenshotPath, fullPage: true });
    console.log(`\nScreenshot saved to: ${screenshotPath}`);

    // Save JSON data
    fs.writeFileSync('fuzzy-benchmark-results.json', JSON.stringify(benchmarkData, null, 2));

    await browser.close();
    console.log('\nAll done!');
}

main().catch(err => {
    console.error('Error running benchmark:', err);
    process.exit(1);
});
