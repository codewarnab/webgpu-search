import { WebGPUEngine, type SearchResult } from './webgpu-engine.ts';
import { CPUEngine, type CPUSearchResult } from './cpu-engine.ts';
import { generateDataset, type Dataset } from './dataset.ts';
import { BenchmarkRunner, type BenchmarkRowResult } from './benchmark.ts';
import { svgToPngBlob, generateBenchmarkCsv, downloadBlob, downloadFullReportZip } from './export.ts';

// State
let gpuEngine: WebGPUEngine;
let cpuEngine: CPUEngine;
let currentDataset: Dataset | null = null;
let benchmarkRunner: BenchmarkRunner;

let substringBenchmarkResults: BenchmarkRowResult[] = [];
let fuzzyBenchmarkResults: BenchmarkRowResult[] = [];

// DOM Elements
const webgpuBadge = document.getElementById('webgpu-status-badge')!;
const hwAdapter = document.getElementById('hw-adapter')!;
const hwVendor = document.getElementById('hw-vendor')!;
const hwBuffer = document.getElementById('hw-buffer')!;

const datasetSizeSelect = document.getElementById('dataset-size-select') as HTMLSelectElement;
const searchModeSelect = document.getElementById('search-mode-select') as HTMLSelectElement;
const btnReloadData = document.getElementById('btn-reload-data') as HTMLButtonElement;
const activeDatasetSize = document.getElementById('active-dataset-size')!;

const searchQueryInput = document.getElementById('search-query-input') as HTMLInputElement;
const meterGpu = document.getElementById('meter-gpu')!;
const meterGpuVal = document.getElementById('meter-gpu-val')!;
const meterGpuSub = document.getElementById('meter-gpu-sub')!;

const meterUfuzzy = document.getElementById('meter-ufuzzy')!;
const meterUfuzzyVal = document.getElementById('meter-ufuzzy-val')!;

const meterNative = document.getElementById('meter-native')!;
const meterNativeVal = document.getElementById('meter-native-val')!;

const resultsList = document.getElementById('results-list')!;
const resultsCountSummary = document.getElementById('results-count-summary')!;

const btnRunBenchmark = document.getElementById('btn-run-benchmark') as HTMLButtonElement;
const benchmarkStatus = document.getElementById('benchmark-status')!;
const benchmarkProgressBox = document.getElementById('benchmark-progress-box')!;
const benchmarkProgressBar = document.getElementById('benchmark-progress-bar')!;
const benchmarkProgressText = document.getElementById('benchmark-progress-text')!;

// Tabs and Panels
const tabBtnSubstring = document.getElementById('tab-btn-substring') as HTMLButtonElement;
const tabBtnFuzzy = document.getElementById('tab-btn-fuzzy') as HTMLButtonElement;
const panelSubstring = document.getElementById('panel-substring') as HTMLDivElement;
const panelFuzzy = document.getElementById('panel-fuzzy') as HTMLDivElement;

const tableBodySubstring = document.getElementById('table-body-substring')!;
const tableBodyFuzzy = document.getElementById('table-body-fuzzy')!;
const chartSubstring = document.getElementById('benchmark-chart-substring') as unknown as SVGSVGElement;
const chartFuzzy = document.getElementById('benchmark-chart-fuzzy') as unknown as SVGSVGElement;

// Download Buttons
const btnDownloadZip = document.getElementById('btn-download-zip') as HTMLButtonElement;
const btnDownloadCsv = document.getElementById('btn-download-csv') as HTMLButtonElement;
const btnDownloadPngSub = document.getElementById('btn-download-png-sub') as HTMLButtonElement;
const btnDownloadPngFuz = document.getElementById('btn-download-png-fuz') as HTMLButtonElement;

async function init() {
    gpuEngine = new WebGPUEngine();
    cpuEngine = new CPUEngine();
    benchmarkRunner = new BenchmarkRunner(gpuEngine, cpuEngine);
    (window as any).benchmarkRunner = benchmarkRunner;
    (window as any).gpuEngine = gpuEngine;
    (window as any).cpuEngine = cpuEngine;

    const isSupported = await gpuEngine.init();
    if (!isSupported) {
        webgpuBadge.textContent = 'WebGPU Unsupported';
        webgpuBadge.style.background = 'rgba(239, 68, 68, 0.2)';
        webgpuBadge.style.color = '#ef4444';
        hwAdapter.textContent = 'WebGPU not available in this browser';
        hwVendor.textContent = 'Use Chrome/Edge/Firefox with WebGPU enabled';
        return;
    }

    // Populate Hardware Info
    const info = gpuEngine.adapterInfo!;
    webgpuBadge.textContent = 'WebGPU Ready';
    webgpuBadge.style.background = 'rgba(16, 185, 129, 0.15)';
    webgpuBadge.style.color = '#10b981';
    hwAdapter.textContent = `${info.vendor} - ${info.device}`;
    hwVendor.textContent = `${info.architecture} (Timestamp Query: ${info.hasTimestampQuery ? 'Yes' : 'No'})`;
    hwBuffer.textContent = `${info.maxBufferSizeMB} MB`;

    // Load Initial Dataset
    await switchDataset(100_000);

    // Bind Event Listeners
    datasetSizeSelect.addEventListener('change', () => {
        const size = parseInt(datasetSizeSelect.value, 10);
        switchDataset(size);
    });

    btnReloadData.addEventListener('click', () => {
        const size = parseInt(datasetSizeSelect.value, 10);
        switchDataset(size);
    });

    searchModeSelect.addEventListener('change', () => {
        triggerSearch();
    });

    searchQueryInput.addEventListener('input', () => {
        triggerSearch();
    });

    // Preset Chips for Edge Cases
    document.querySelectorAll('.btn-chip').forEach(btn => {
        btn.addEventListener('click', () => {
            const query = (btn as HTMLElement).dataset.query || '';
            searchQueryInput.value = query;
            triggerSearch();
        });
    });

    // Tab Switching
    tabBtnSubstring.addEventListener('click', () => {
        tabBtnSubstring.classList.add('active');
        tabBtnFuzzy.classList.remove('active');
        panelSubstring.classList.add('active');
        panelFuzzy.classList.remove('active');
    });

    tabBtnFuzzy.addEventListener('click', () => {
        tabBtnFuzzy.classList.add('active');
        tabBtnSubstring.classList.remove('active');
        panelFuzzy.classList.add('active');
        panelSubstring.classList.remove('active');
    });

    // Benchmark Run
    btnRunBenchmark.addEventListener('click', () => {
        runFullBenchmark();
    });

    // Export Handlers
    btnDownloadZip.addEventListener('click', async () => {
        btnDownloadZip.disabled = true;
        btnDownloadZip.textContent = '⏳ Packaging ZIP...';
        try {
            await downloadFullReportZip(
                gpuEngine.adapterInfo,
                substringBenchmarkResults,
                fuzzyBenchmarkResults,
                chartSubstring,
                chartFuzzy
            );
        } catch (err) {
            console.error('ZIP generation failed:', err);
            alert('Failed to generate ZIP package: ' + err);
        } finally {
            btnDownloadZip.disabled = false;
            btnDownloadZip.textContent = '📦 Download Full Report (.ZIP: CSV + PNGs)';
        }
    });

    btnDownloadCsv.addEventListener('click', () => {
        const csv = generateBenchmarkCsv(gpuEngine.adapterInfo, substringBenchmarkResults, fuzzyBenchmarkResults);
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        downloadBlob(blob, `webgpu_search_benchmark_${new Date().toISOString().slice(0, 10)}.csv`);
    });

    btnDownloadPngSub.addEventListener('click', async () => {
        try {
            const blob = await svgToPngBlob(chartSubstring);
            downloadBlob(blob, `chart_substring_benchmark.png`);
        } catch (err) {
            alert('Could not export Substring chart image: ' + err);
        }
    });

    btnDownloadPngFuz.addEventListener('click', async () => {
        try {
            const blob = await svgToPngBlob(chartFuzzy);
            downloadBlob(blob, `chart_fuzzy_benchmark.png`);
        } catch (err) {
            alert('Could not export Fuzzy chart image: ' + err);
        }
    });

    // Initial search
    triggerSearch();

    // Check URL parameters for autorun (e.g. ?autorun=all or ?autorun=fuzzy)
    const params = new URLSearchParams(window.location.search);
    const autorun = params.get('autorun');
    if (autorun) {
        if (autorun === 'fuzzy' || autorun === 'substring') {
            searchModeSelect.value = autorun;
        }
        setTimeout(() => {
            runFullBenchmark();
        }, 300);
    }

    (window as any).__IS_INITIALIZED__ = true;
}

async function switchDataset(size: number) {
    btnReloadData.disabled = true;
    datasetSizeSelect.disabled = true;
    activeDatasetSize.textContent = `Generating ${size.toLocaleString()} items...`;

    // Allow UI to render loading state
    await new Promise(r => setTimeout(r, 20));

    const t0 = performance.now();
    currentDataset = generateDataset(size);
    const genTime = performance.now() - t0;

    activeDatasetSize.textContent = `Uploading ${size.toLocaleString()} items to GPU VRAM...`;
    await new Promise(r => setTimeout(r, 20));

    const { uploadTimeMs } = await gpuEngine.loadDataset(currentDataset);

    activeDatasetSize.textContent = `${size.toLocaleString()} items (Gen: ${genTime.toFixed(0)}ms, VRAM upload: ${uploadTimeMs.toFixed(1)}ms)`;
    btnReloadData.disabled = false;
    datasetSizeSelect.disabled = false;

    triggerSearch();
}

let searchDebounceTimer: any = null;

function triggerSearch() {
    clearTimeout(searchDebounceTimer);
    searchDebounceTimer = setTimeout(executeLiveSearch, 20);
}

async function executeLiveSearch() {
    if (!currentDataset) return;
    const query = searchQueryInput.value.trim();
    const mode = searchModeSelect.value as 'substring' | 'fuzzy';

    if (!query) {
        meterGpuVal.textContent = '-- ms';
        meterUfuzzyVal.textContent = '-- ms';
        meterNativeVal.textContent = '-- ms';
        meterGpuSub.textContent = 'Awaiting query';
        resultsList.innerHTML = '<div style="color: var(--text-muted); text-align: center; margin-top: 2rem;">Type in the box above to search.</div>';
        resultsCountSummary.textContent = 'Matched 0 items';
        resetMeterHighlights();
        return;
    }

    // 1. WebGPU Retained Search
    let gpuResult: SearchResult;
    try {
        gpuResult = await gpuEngine.search(query, { mode, maxResults: 1000 });
    } catch (err) {
        console.error('GPU search error:', err);
        return;
    }

    // 2. uFuzzy CPU Search
    const ufuzzyResult: CPUSearchResult = cpuEngine.searchUFuzzy(currentDataset.strings, query, 1000);

    // 3. Native JS CPU Search
    const nativeResult: CPUSearchResult = cpuEngine.searchNative(currentDataset.strings, query, 1000);

    // Update Meter Displays
    const gpuTotal = gpuResult.timings.totalMs;
    const ufuzzyTotal = ufuzzyResult.durationMs;
    const nativeTotal = nativeResult.durationMs;

    meterGpuVal.textContent = `${gpuTotal.toFixed(2)} ms`;
    meterGpuSub.textContent = `Dispatch: ${gpuResult.timings.gpuDispatchMs.toFixed(2)}ms | Readback: ${gpuResult.timings.readbackMs.toFixed(2)}ms`;

    meterUfuzzyVal.textContent = `${ufuzzyTotal.toFixed(2)} ms`;
    meterNativeVal.textContent = `${nativeTotal.toFixed(2)} ms`;

    // Highlight Winner
    resetMeterHighlights();
    const minTime = Math.min(gpuTotal, ufuzzyTotal, nativeTotal);
    if (minTime === gpuTotal) {
        meterGpu.classList.add('winner');
    } else if (minTime === ufuzzyTotal) {
        meterUfuzzy.classList.add('winner');
    } else {
        meterNative.classList.add('winner');
    }

    // Render Results List
    resultsCountSummary.textContent = `Found ${gpuResult.totalMatches.toLocaleString()} matches (WebGPU) | ${ufuzzyResult.totalMatches.toLocaleString()} (uFuzzy)`;
    renderResults(gpuResult.results, query);
}

function resetMeterHighlights() {
    meterGpu.classList.remove('winner');
    meterUfuzzy.classList.remove('winner');
    meterNative.classList.remove('winner');
}

function renderResults(results: Array<{ index: number; score: number; text?: string }>, query: string) {
    if (results.length === 0) {
        resultsList.innerHTML = '<div style="color: var(--text-muted); text-align: center; margin-top: 2rem;">No matching items found.</div>';
        return;
    }

    const displayCount = Math.min(results.length, 50);
    const fragment = document.createDocumentFragment();

    const lowerQuery = query.toLowerCase();

    for (let i = 0; i < displayCount; i++) {
        const item = results[i];
        const text = item.text || `Record #${item.index}`;

        const div = document.createElement('div');
        div.className = 'result-item';

        // Highlight matching substrings
        const lowerText = text.toLowerCase();
        const matchIdx = lowerText.indexOf(lowerQuery);

        let highlightedHtml = text;
        if (matchIdx >= 0) {
            const before = text.substring(0, matchIdx);
            const match = text.substring(matchIdx, matchIdx + query.length);
            const after = text.substring(matchIdx + query.length);
            highlightedHtml = `${escapeHtml(before)}<span class="result-match-highlight">${escapeHtml(match)}</span>${escapeHtml(after)}`;
        } else {
            highlightedHtml = escapeHtml(text);
        }

        div.innerHTML = `
            <div>${highlightedHtml}</div>
            <div class="result-score">#${item.index} | score: ${item.score}</div>
        `;
        fragment.appendChild(div);
    }

    resultsList.innerHTML = '';
    resultsList.appendChild(fragment);
}

function escapeHtml(str: string): string {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// Full Matrix Benchmark Execution (Runs BOTH Substring and Fuzzy algorithms)
async function runFullBenchmark() {
    btnRunBenchmark.disabled = true;
    btnDownloadZip.disabled = true;
    btnDownloadCsv.disabled = true;
    btnDownloadPngSub.disabled = true;
    btnDownloadPngFuz.disabled = true;

    benchmarkStatus.textContent = 'Benchmark running...';
    benchmarkProgressBox.style.display = 'block';

    tableBodySubstring.innerHTML = '';
    tableBodyFuzzy.innerHTML = '';
    substringBenchmarkResults = [];
    fuzzyBenchmarkResults = [];

    const sizes = [10_000, 100_000, 500_000, 1_000_000, 2_000_000];
    const query = searchQueryInput.value.trim() || 'AuthController';

    const totalSteps = sizes.length * 2; // Both algorithms

    try {
        // PHASE 1: Substring Algorithm
        tabBtnSubstring.click(); // Focus substring tab
        await benchmarkRunner.runBenchmark(
            sizes,
            query,
            'substring',
            3,
            (progress) => {
                const percent = Math.round((progress.currentStep / totalSteps) * 100);
                benchmarkProgressBar.style.width = `${percent}%`;
                benchmarkProgressText.textContent = `[1/2 Substring] ${progress.stepName}`;

                if (progress.currentRow) {
                    substringBenchmarkResults.push(progress.currentRow);
                    appendBenchmarkTableRow(tableBodySubstring, progress.currentRow);
                    drawBenchmarkChart(chartSubstring, substringBenchmarkResults);
                }
            }
        );

        // Allow UI to breathe
        await new Promise(r => setTimeout(r, 60));

        // PHASE 2: Fuzzy Algorithm
        tabBtnFuzzy.click(); // Focus fuzzy tab
        await benchmarkRunner.runBenchmark(
            sizes,
            query,
            'fuzzy',
            3,
            (progress) => {
                const currentTotal = sizes.length + progress.currentStep;
                const percent = Math.round((currentTotal / totalSteps) * 100);
                benchmarkProgressBar.style.width = `${percent}%`;
                benchmarkProgressText.textContent = `[2/2 Fuzzy] ${progress.stepName}`;

                if (progress.currentRow) {
                    fuzzyBenchmarkResults.push(progress.currentRow);
                    appendBenchmarkTableRow(tableBodyFuzzy, progress.currentRow);
                    drawBenchmarkChart(chartFuzzy, fuzzyBenchmarkResults);
                }
            }
        );

        benchmarkStatus.textContent = 'Benchmark Complete';
        benchmarkProgressText.textContent = `Completed full benchmark matrix for Substring and Fuzzy (10k to 2M rows)!`;

        // Store results globally
        (window as any).__BENCHMARK_RESULTS__ = {
            substring: substringBenchmarkResults,
            fuzzy: fuzzyBenchmarkResults
        };

        // Enable download buttons
        btnDownloadZip.disabled = false;
        btnDownloadCsv.disabled = false;
        btnDownloadPngSub.disabled = false;
        btnDownloadPngFuz.disabled = false;
    } catch (err) {
        console.error('Benchmark failed:', err);
        benchmarkStatus.textContent = 'Benchmark Failed';
        benchmarkProgressText.textContent = `Error: ${err}`;
    } finally {
        btnRunBenchmark.disabled = false;
    }
}

function appendBenchmarkTableRow(tbody: HTMLElement, row: BenchmarkRowResult) {
    const tr = document.createElement('tr');

    const crossoverBadge = row.crossover.gpuRetainedBeatsUfuzzy
        ? `<span class="tag-crossover-win">⚡ GPU Win (${row.retainedVsUfuzzySpeedup}x)</span>`
        : `<span class="tag-crossover-loss">CPU Wins (${(1 / row.retainedVsUfuzzySpeedup).toFixed(1)}x)</span>`;

    tr.innerHTML = `
        <td><strong>${row.datasetSize.toLocaleString()}</strong></td>
        <td>${row.gpuRetained.gpuDispatchMs} ms</td>
        <td>${row.gpuRetained.readbackMs} ms</td>
        <td style="color: var(--color-gpu); font-weight: 600;">${row.gpuRetained.totalMs} ms</td>
        <td style="color: var(--text-muted);">${row.gpuCold.totalMs} ms</td>
        <td style="color: var(--color-ufuzzy); font-weight: 600;">${row.ufuzzyMs} ms</td>
        <td style="color: var(--color-native);">${row.jsNativeMs} ms</td>
        <td><strong>${row.retainedVsUfuzzySpeedup}x</strong></td>
        <td>${crossoverBadge}</td>
    `;
    tbody.appendChild(tr);
}

function drawBenchmarkChart(svgElement: SVGSVGElement, results: BenchmarkRowResult[]) {
    if (!svgElement || results.length === 0) return;

    const width = 1000;
    const height = 260;
    const padding = { top: 30, right: 40, bottom: 40, left: 60 };

    const graphWidth = width - padding.left - padding.right;
    const graphHeight = height - padding.top - padding.bottom;

    // Find max value for Y scale
    let maxTime = 10;
    for (const r of results) {
        maxTime = Math.max(maxTime, r.gpuRetained.totalMs, r.ufuzzyMs, r.jsNativeMs);
    }
    maxTime = Math.ceil(maxTime * 1.15); // Add headroom

    // Clear SVG
    svgElement.innerHTML = '';

    // Draw horizontal grid lines
    const gridCount = 5;
    for (let i = 0; i <= gridCount; i++) {
        const yVal = (maxTime / gridCount) * i;
        const y = padding.top + graphHeight - (i / gridCount) * graphHeight;

        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', `${padding.left}`);
        line.setAttribute('y1', `${y}`);
        line.setAttribute('x2', `${width - padding.right}`);
        line.setAttribute('y2', `${y}`);
        line.setAttribute('stroke', '#24344d');
        line.setAttribute('stroke-dasharray', '3,3');
        svgElement.appendChild(line);

        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', `${padding.left - 10}`);
        text.setAttribute('y', `${y + 4}`);
        text.setAttribute('fill', '#64748b');
        text.setAttribute('font-size', '11');
        text.setAttribute('text-anchor', 'end');
        text.textContent = `${Math.round(yVal)}ms`;
        svgElement.appendChild(text);
    }

    // X coordinates mapped per item
    const getX = (idx: number) => {
        if (results.length === 1) return padding.left + graphWidth / 2;
        return padding.left + (idx / (results.length - 1)) * graphWidth;
    };

    const getY = (val: number) => {
        return padding.top + graphHeight - (val / maxTime) * graphHeight;
    };

    // Draw X labels
    for (let i = 0; i < results.length; i++) {
        const x = getX(i);
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', `${x}`);
        text.setAttribute('y', `${height - 12}`);
        text.setAttribute('fill', '#94a3b8');
        text.setAttribute('font-size', '11');
        text.setAttribute('text-anchor', 'middle');
        const sizeLabel = results[i].datasetSize >= 1_000_000
            ? `${results[i].datasetSize / 1_000_000}M`
            : `${results[i].datasetSize / 1_000}k`;
        text.textContent = sizeLabel;
        svgElement.appendChild(text);
    }

    // Helper to draw series polyline and points
    const drawSeries = (color: string, getter: (r: BenchmarkRowResult) => number) => {
        if (results.length > 1) {
            const points = results.map((r, idx) => `${getX(idx)},${getY(getter(r))}`).join(' ');
            const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
            polyline.setAttribute('fill', 'none');
            polyline.setAttribute('stroke', color);
            polyline.setAttribute('stroke-width', '2.5');
            polyline.setAttribute('points', points);
            svgElement.appendChild(polyline);
        }

        // Points
        results.forEach((r, idx) => {
            const cx = getX(idx);
            const cy = getY(getter(r));
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', `${cx}`);
            circle.setAttribute('cy', `${cy}`);
            circle.setAttribute('r', '5');
            circle.setAttribute('fill', color);
            circle.setAttribute('stroke', '#090d16');
            circle.setAttribute('stroke-width', '2');
            svgElement.appendChild(circle);
        });
    };

    // Draw lines: Native JS, uFuzzy, WebGPU Retained
    drawSeries('#a855f7', r => r.jsNativeMs);
    drawSeries('#f59e0b', r => r.ufuzzyMs);
    drawSeries('#38bdf8', r => r.gpuRetained.totalMs);
}

// Start application
window.addEventListener('DOMContentLoaded', () => {
    init().catch(console.error);
});
