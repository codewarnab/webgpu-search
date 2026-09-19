import {
    WebGPUEngine,
    CPUEngine,
    searchCpuReference,
    normalizeText,
    type SearchResult,
    type CPUSearchResult
} from 'webgpu-search';
import { generateDataset, type Dataset, type CorpusType } from './dataset';
import { BenchmarkRunner, type BenchmarkRowResult } from './benchmark';
import { svgToPngBlob, generateBenchmarkCsv, generateMarkdownSummary, downloadBlob } from './export';


// State
let gpuEngine: WebGPUEngine;
let cpuEngine: CPUEngine;
let currentDataset: Dataset | null = null;
let benchmarkRunner: BenchmarkRunner;

let searchWorker: Worker | null = null;
let workerReady = false;
let activeQuerySeq = 0;
// Dataset epoch: incremented on every switchDataset (synchronous with
// currentDataset swap). Sent with LOAD_DATASET/SEARCH; worker echoes it so
// stale-dataset hits (in-flight SEARCH racing a dataset switch) are dropped
// instead of enriching old indices against new strings.
let activeDatasetGeneration = 0;

let substringBenchmarkResults: BenchmarkRowResult[] = [];
let fuzzyBenchmarkResults: BenchmarkRowResult[] = [];

// DOM Elements
const webgpuBadge = document.getElementById('webgpu-status-badge')!;
const hwAdapter = document.getElementById('hw-adapter')!;
const hwVendor = document.getElementById('hw-vendor')!;
const hwBuffer = document.getElementById('hw-buffer')!;
const hwFps = document.getElementById('hw-fps');
const meterFpsVal = document.getElementById('meter-fps-val');
const meterFpsSub = document.getElementById('meter-fps-sub');
const benchmarkFpsVal = document.getElementById('benchmark-fps-val');
const hudFpsVal = document.getElementById('hud-fps-val');
const hudModeVal = document.getElementById('hud-mode-val');
const hudPulseDot = document.getElementById('hud-pulse-dot');

const datasetSizeSelect = document.getElementById('dataset-size-select') as HTMLSelectElement;
const corpusTypeSelect = document.getElementById('corpus-type-select') as HTMLSelectElement | null;
const searchModeSelect = document.getElementById('search-mode-select') as HTMLSelectElement;
const executionThreadSelect = document.getElementById('execution-thread-select') as HTMLSelectElement;
const btnReloadData = document.getElementById('btn-reload-data') as HTMLButtonElement;
const activeDatasetSize = document.getElementById('active-dataset-size')!;

const searchQueryInput = document.getElementById('search-query-input') as HTMLInputElement;
const meterGpu = document.getElementById('meter-gpu')!;
const meterGpuVal = document.getElementById('meter-gpu-val')!;
const meterGpuSub = document.getElementById('meter-gpu-sub')!;

const meterParity = document.getElementById('meter-parity') as HTMLDivElement | null;
const meterParityVal = document.getElementById('meter-parity-val') as HTMLDivElement | null;

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
const btnDownloadAll = document.getElementById('btn-download-all') as HTMLButtonElement;
const btnDownloadCsv = document.getElementById('btn-download-csv') as HTMLButtonElement;
const btnDownloadPngSub = document.getElementById('btn-download-png-sub') as HTMLButtonElement;
const btnDownloadPngFuz = document.getElementById('btn-download-png-fuz') as HTMLButtonElement;
const btnCopySummary = document.getElementById('btn-copy-summary') as HTMLButtonElement;

async function init() {
    gpuEngine = new WebGPUEngine();
    cpuEngine = new CPUEngine();
    benchmarkRunner = new BenchmarkRunner(gpuEngine, cpuEngine);
    (window as any).benchmarkRunner = benchmarkRunner;
    (window as any).gpuEngine = gpuEngine;
    (window as any).cpuEngine = cpuEngine;

    let isSupported = false;
    try {
        isSupported = await gpuEngine.init();
    } catch (err) {
        console.warn('WebGPU engine init threw an error, falling back to CPU:', err);
    }
    const warningBanner = document.getElementById('webgpu-warning-banner');

    if (!isSupported) {
        if (warningBanner) warningBanner.style.display = 'block';
        webgpuBadge.textContent = 'WebGPU Disabled (CPU Mode)';
        webgpuBadge.style.background = 'rgba(239, 68, 68, 0.2)';
        webgpuBadge.style.color = '#ef4444';
        hwAdapter.textContent = 'Disabled or Blocked by Browser';
        hwVendor.textContent = 'See troubleshooting steps above';
        hwBuffer.textContent = 'N/A';
        meterGpuVal.textContent = 'Disabled';
        meterGpuSub.textContent = 'WebGPU unavailable';
    } else {
        // Populate Hardware Info
        const info = gpuEngine.adapterInfo!;
        webgpuBadge.textContent = 'WebGPU Ready';
        webgpuBadge.style.background = 'rgba(16, 185, 129, 0.15)';
        webgpuBadge.style.color = '#10b981';
        hwAdapter.textContent = `${info.vendor} - ${info.device}`;
        hwVendor.textContent = `${info.architecture} (Timestamp Query: ${info.hasTimestampQuery ? 'Yes' : 'No'})`;
        hwBuffer.textContent = `${info.maxBufferSizeMB} MB`;
    }

    // Initialize Web Worker for background search offloading
    try {
        searchWorker = new Worker(new URL('./search.worker.ts', import.meta.url), { type: 'module' });
        searchWorker.postMessage({ type: 'INIT' });
        searchWorker.onmessage = (e: MessageEvent) => {
            const { type, payload } = e.data;
            if (type === 'INIT_DONE') {
                workerReady = true;
            } else if (type === 'DATASET_LOADED') {
                if (payload?.error) {
                    console.error('[main] Worker LOAD_DATASET failed:', payload.error);
                    // Surface worker load failures (previously console-only,
                    // badge kept claiming success).
                    activeDatasetSize.textContent = `Worker dataset error: ${String(payload.error).slice(0, 160)}`;
                }
            } else if (type === 'SEARCH_RESULTS') {
                if (payload.queryId !== activeQuerySeq) return;
                // Drop stale-dataset hits: request was sent for an older
                // dataset epoch, or the worker had not yet committed the
                // current epoch when it searched. Without this, old indices
                // enrich against new strings (wrong text / Record # placeholders).
                if (
                    typeof payload?.requestGeneration === 'number' &&
                    payload.requestGeneration !== activeDatasetGeneration
                ) {
                    return;
                }
                if (
                    typeof payload?.datasetGeneration === 'number' &&
                    typeof payload?.requestGeneration === 'number' &&
                    payload.datasetGeneration !== payload.requestGeneration
                ) {
                    return;
                }
                // M4 string-isolated enrichment: the worker returns compact
                // `{index,score}[]` + meta (no text clone); re-attach display
                // strings by index from our own copy (same contract as
                // SearchIndex enrichment over token-only engine results).
                let gpuResult = payload.gpuResult;
                if (!gpuResult && payload.gpuCompact && payload.gpuMeta && currentDataset) {
                    const hits = payload.gpuCompact as Array<{ index: number; score: number }>;
                    const strings = currentDataset.strings;
                    // Defensive: worker validates bounds, but filter OOB here
                    // too (never render a placeholder for a corrupt index
                    // without a warning).
                    const validHits = hits.filter((h) => Number.isInteger(h.index) && h.index >= 0 && h.index < strings.length);
                    if (validHits.length !== hits.length) {
                        console.warn(`[main] dropped ${hits.length - validHits.length} out-of-bounds compact hits (size ${strings.length}).`);
                    }
                    gpuResult = {
                        ...payload.gpuMeta,
                        results: validHits.map((h) => ({ index: h.index, score: h.score, text: strings[h.index] ?? '' }))
                    };
                    // One-time payload-size note for the before/after record:
                    // compact ~= hits*8 B vs full ~= compact + text UTF-16 units.
                    // Per-object structured-clone overhead means this is
                    // order-of-magnitude, not exact bytes.
                    if (!(window as any).__compactLogged) {
                        (window as any).__compactLogged = true;
                        let textChars = 0;
                        for (const h of validHits) textChars += (strings[h.index] ?? '').length;
                        console.info(
                            `[main] string-isolated SEARCH return: ${validHits.length} hits, ` +
                            `compact ~${validHits.length * 8} B vs full ~${validHits.length * 8 + textChars} B ` +
                            `(~${(textChars / Math.max(1, validHits.length * 8)).toFixed(1)}x smaller clone; approximate)`
                        );
                    }
                }
                let parityResult = payload.parityResult ?? null;
                if (!parityResult && currentDataset && payload.query) {
                    const norm = normalizeText(payload.query, true);
                    parityResult = searchCpuReference(
                        currentDataset.recordTokens,
                        norm.tokens,
                        (payload.gpuMeta?.mode || 'substring') as 'substring' | 'fuzzy',
                        1000,
                        currentDataset.strings
                    );
                }
                handleSearchResults(gpuResult, parityResult, payload.ufuzzyResult, payload.nativeResult, payload.query);
            } else if (type === 'SEARCH_ERROR') {
                // Worker-side search failures (invalid mode, QueryTooLong,
                // GPU loss) previously only console.error'd in the worker and
                // left stale results on screen. Surface instead of going stale.
                if (payload?.queryId !== activeQuerySeq) return;
                if (
                    typeof payload?.requestGeneration === 'number' &&
                    payload.requestGeneration !== activeDatasetGeneration
                ) {
                    return;
                }
                console.error('[main] Worker SEARCH failed:', payload?.error);
                resultsCountSummary.textContent = `Search error: ${String(payload?.error ?? 'unknown').slice(0, 200)}`;
                meterGpuVal.textContent = 'Error';
                meterGpuSub.textContent = String(payload?.error ?? 'worker search failed').slice(0, 160);
            }
        };
    } catch (workerErr) {
        console.warn('Dedicated Web Worker setup failed, falling back to main thread:', workerErr);
    }

    // UI Frame Rate (FPS) Telemetry
    let frameCount = 0;
    let lastFpsTime = performance.now();
    function trackFps() {
        frameCount++;
        const now = performance.now();
        const elapsed = now - lastFpsTime;
        if (elapsed >= 500) {
            const fps = Math.round((frameCount * 1000) / elapsed);
            const isWorker = (executionThreadSelect?.value || 'worker') === 'worker';
            const color = fps >= 100 ? '#10b981' : (fps >= 50 ? '#f59e0b' : '#ef4444');
            const subText = fps >= 100 ? '0ms Frame Drop' : (fps >= 50 ? 'Micro-stutters' : 'Severe UI Jank');
            const modeText = isWorker ? '⚡ Worker Offloaded' : '🖥️ Main Thread';

            if (hwFps) {
                hwFps.textContent = `${fps} FPS`;
                hwFps.style.color = color;
            }
            if (meterFpsVal) {
                meterFpsVal.textContent = `${fps} FPS`;
                meterFpsVal.style.color = color;
            }
            if (meterFpsSub) {
                meterFpsSub.textContent = subText;
            }
            if (benchmarkFpsVal) {
                benchmarkFpsVal.textContent = `${fps} FPS`;
                benchmarkFpsVal.style.color = color;
            }
            if (hudFpsVal) {
                hudFpsVal.textContent = `${fps} FPS`;
                hudFpsVal.style.color = color;
            }
            if (hudModeVal) {
                hudModeVal.textContent = modeText;
            }
            if (hudPulseDot) {
                hudPulseDot.style.background = color;
                hudPulseDot.style.boxShadow = `0 0 10px ${color}`;
            }

            frameCount = 0;
            lastFpsTime = now;
        }
        requestAnimationFrame(trackFps);
    }
    requestAnimationFrame(trackFps);

    // Load Initial Dataset
    await switchDataset(100_000);

    // Bind Event Listeners
    datasetSizeSelect.addEventListener('change', () => {
        const size = parseInt(datasetSizeSelect.value, 10);
        const corpus = (corpusTypeSelect?.value || 'ascii') as CorpusType;
        switchDataset(size, corpus);
    });

    if (corpusTypeSelect) {
        corpusTypeSelect.addEventListener('change', () => {
            const size = parseInt(datasetSizeSelect.value, 10);
            const corpus = (corpusTypeSelect.value || 'ascii') as CorpusType;
            switchDataset(size, corpus);
        });
    }

    if (executionThreadSelect) {
        executionThreadSelect.addEventListener('change', () => {
            triggerSearch();
        });
    }

    btnReloadData.addEventListener('click', () => {
        const size = parseInt(datasetSizeSelect.value, 10);
        const corpus = (corpusTypeSelect?.value || 'ascii') as CorpusType;
        switchDataset(size, corpus);
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

    // Helper to generate a clean, hardware-branded CSV filename
    const getBenchmarkCsvFilename = () => {
        const info = gpuEngine?.adapterInfo;
        const rawName = info ? `${info.vendor}_${info.device}` : (gpuEngine?.isReady ? 'gpu' : 'cpu_fallback');
        const slug = rawName
            .replace(/[^a-zA-Z0-9_-]/g, '_')
            .replace(/_+/g, '_')
            .replace(/^_|_$/g, '')
            .substring(0, 32);
        return `benchmark_${slug}_${new Date().toISOString().slice(0, 10)}.csv`;
    };

    // Export Handlers (Direct downloads: NO ZIP)
    btnDownloadAll.addEventListener('click', async () => {
        btnDownloadAll.disabled = true;
        btnDownloadAll.textContent = '⏳ Downloading 3 Files...';
        try {
            // 1. Substring Chart PNG
            const subBlob = await svgToPngBlob(chartSubstring);
            downloadBlob(subBlob, 'chart_substring_benchmark.png');

            // Brief pause so browser doesn't drop multiple downloads
            await new Promise(r => setTimeout(r, 200));

            // 2. Fuzzy Chart PNG
            const fuzBlob = await svgToPngBlob(chartFuzzy);
            downloadBlob(fuzBlob, 'chart_fuzzy_benchmark.png');

            await new Promise(r => setTimeout(r, 200));

            // 3. CSV Spreadsheet
            const csv = generateBenchmarkCsv(gpuEngine.adapterInfo, substringBenchmarkResults, fuzzyBenchmarkResults);
            const csvBlob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
            downloadBlob(csvBlob, getBenchmarkCsvFilename());
        } catch (err) {
            console.error('Download error:', err);
            alert('Failed to download benchmark assets: ' + err);
        } finally {
            btnDownloadAll.disabled = false;
            btnDownloadAll.textContent = '📥 Download All (2 Images + 1 CSV File)';
        }
    });

    btnDownloadCsv.addEventListener('click', () => {
        const csv = generateBenchmarkCsv(gpuEngine.adapterInfo, substringBenchmarkResults, fuzzyBenchmarkResults);
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        downloadBlob(blob, getBenchmarkCsvFilename());
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

    if (btnCopySummary) {
        btnCopySummary.addEventListener('click', async () => {
            try {
                const markdown = generateMarkdownSummary(gpuEngine.adapterInfo, substringBenchmarkResults, fuzzyBenchmarkResults);
                await navigator.clipboard.writeText(markdown);
                const originalText = btnCopySummary.textContent;
                btnCopySummary.textContent = '✅ Copied to Clipboard!';
                setTimeout(() => {
                    btnCopySummary.textContent = originalText;
                }, 2500);
            } catch (err) {
                console.error('Failed to copy summary:', err);
                alert('Could not copy summary to clipboard: ' + err);
            }
        });
    }

    // Initial search
    triggerSearch();

    // Check URL parameters for autorun (e.g. ?autorun=all or ?autorun=fuzzy)
    const params = new URLSearchParams(window.location.search);
    const autorun = params.get('autorun');
    if (params.get('corpus') && corpusTypeSelect) {
        corpusTypeSelect.value = params.get('corpus')!;
    }
    if (params.get('query') && searchQueryInput) {
        searchQueryInput.value = params.get('query')!;
    }
    console.log('[main] Application initialized. Autorun mode:', autorun);
    if (autorun) {
        if (autorun === 'fuzzy' || autorun === 'substring') {
            searchModeSelect.value = autorun;
        }
        setTimeout(() => {
            runFullBenchmark();
        }, 300);
    }

    (window as any).gpuEngine = gpuEngine;
    (window as any).cpuEngine = cpuEngine;
    (window as any).__IS_INITIALIZED__ = true;
}

async function switchDataset(size: number, corpusType: CorpusType = 'ascii') {
    btnReloadData.disabled = true;
    datasetSizeSelect.disabled = true;
    if (corpusTypeSelect) corpusTypeSelect.disabled = true;
    activeDatasetSize.textContent = `Generating ${size.toLocaleString()} ${corpusType.toUpperCase()} items...`;

    // Allow UI to render loading state
    await new Promise(r => setTimeout(r, 20));

    const t0 = performance.now();
    activeDatasetGeneration += 1;
    const thisGeneration = activeDatasetGeneration;
    currentDataset = generateDataset(size, { corpusType });
    const genTime = performance.now() - t0;

    activeDatasetSize.textContent = gpuEngine.isReady
        ? `Uploading ${size.toLocaleString()} items to GPU VRAM...`
        : `Preparing ${size.toLocaleString()} items in RAM...`;
    await new Promise(r => setTimeout(r, 20));

    const { uploadTimeMs } = await gpuEngine.loadDataset(currentDataset.strings);

    // Sync dataset with background Web Worker (M4 unicode path). The U2F2
    // serialized buffer is transferable: the `.slice(0)` copy is moved with
    // a transfer list (copy-then-move, not zero-copy, since the original is
    // retained for reuse; neutered on arrival is a worker-side error, never
    // silent). `strings` still structured-clones in full -- that clone cost
    // is real and reported by the worker (`stringsChars`); it is irreducible
    // while the worker runs CPU comparison. Legacy v0.1 byte buffers are
    // gone (the M3 engine ignored them whenever `strings` was present --
    // pure clone waste).
    if (searchWorker && currentDataset) {
        const serializedCopy = currentDataset.serializedU2F2.slice(0);
        searchWorker.postMessage({
            type: 'LOAD_DATASET',
            payload: {
                strings: currentDataset.strings,
                serialized: serializedCopy,
                datasetGeneration: thisGeneration,
                sentTimestamp: performance.now()
            }
        }, [serializedCopy]);
    }

    activeDatasetSize.textContent = gpuEngine.isReady
        ? `${size.toLocaleString()} items (${corpusType.toUpperCase()}) (Gen: ${genTime.toFixed(0)}ms, VRAM upload: ${uploadTimeMs.toFixed(1)}ms)`
        : `${size.toLocaleString()} items (${corpusType.toUpperCase()}) (Gen: ${genTime.toFixed(0)}ms, CPU Ready)`;
    btnReloadData.disabled = false;
    datasetSizeSelect.disabled = false;
    if (corpusTypeSelect) corpusTypeSelect.disabled = false;

    triggerSearch();
}

let searchDebounceTimer: any = null;

function triggerSearch() {
    clearTimeout(searchDebounceTimer);
    searchDebounceTimer = setTimeout(executeLiveSearch, 20);
}

function handleSearchResults(
    gpuResult: SearchResult | null,
    parityResult: { durationMs: number; totalMatches: number; results: any[] } | null,
    ufuzzyResult: CPUSearchResult | null,
    nativeResult: CPUSearchResult | null,
    query: string
) {
    const parityTotal = parityResult?.durationMs ?? 0;
    const ufuzzyTotal = ufuzzyResult?.durationMs ?? 0;
    const nativeTotal = nativeResult?.durationMs ?? 0;

    if (meterParityVal) {
        meterParityVal.textContent = parityResult ? `${parityTotal.toFixed(2)} ms` : '-- ms';
    }
    meterUfuzzyVal.textContent = ufuzzyResult ? `${ufuzzyTotal.toFixed(2)} ms` : '-- ms';
    meterNativeVal.textContent = nativeResult ? `${nativeTotal.toFixed(2)} ms` : '-- ms';

    resetMeterHighlights();

    const times: { el: HTMLElement | null; time: number }[] = [];
    if (parityResult && meterParity) times.push({ el: meterParity, time: parityTotal });
    if (ufuzzyResult) times.push({ el: meterUfuzzy, time: ufuzzyTotal });
    if (nativeResult) times.push({ el: meterNative, time: nativeTotal });

    if (gpuResult) {
        const gpuTotal = gpuResult.timings.totalMs;
        meterGpuVal.textContent = `${gpuTotal.toFixed(2)} ms`;
        const execLabel = gpuResult.timings.gpuExecutionMs !== null
            ? `Exec: ${gpuResult.timings.gpuExecutionMs.toFixed(2)}ms | `
            : '';
        meterGpuSub.textContent = `${execLabel}Submit: ${gpuResult.timings.encodeSubmitMs.toFixed(2)}ms | Readback: ${gpuResult.timings.readbackMs.toFixed(2)}ms`;

        times.push({ el: meterGpu, time: gpuTotal });
        times.sort((a, b) => a.time - b.time);
        if (times[0] && times[0].el) {
            times[0].el.classList.add('winner');
        }

        const overflowBadge = gpuResult.hasOverflow
            ? ` (⚠️ pool overflow: top ${gpuResult.results.length} of ${gpuResult.totalMatches.toLocaleString()})`
            : '';
        resultsCountSummary.textContent = `Found ${gpuResult.totalMatches.toLocaleString()} matches (WebGPU)${overflowBadge} | ${parityResult ? parityResult.totalMatches.toLocaleString() : 0} (CPU Parity) | ${ufuzzyResult ? ufuzzyResult.totalMatches.toLocaleString() : 0} (uFuzzy)`;
        renderResults(gpuResult.results, query);
    } else {
        meterGpuVal.textContent = 'Disabled';
        meterGpuSub.textContent = 'WebGPU unavailable';

        times.sort((a, b) => a.time - b.time);
        if (times[0] && times[0].el) {
            times[0].el.classList.add('winner');
        }

        resultsCountSummary.textContent = `Found ${parityResult ? parityResult.totalMatches.toLocaleString() : 0} matches (CPU Parity) | ${ufuzzyResult ? ufuzzyResult.totalMatches.toLocaleString() : 0} (uFuzzy) | ${nativeResult ? nativeResult.totalMatches.toLocaleString() : 0} (JS Native)`;
        if (parityResult) {
            renderResults(parityResult.results, query);
        } else if (ufuzzyResult) {
            renderResults(ufuzzyResult.results, query);
        }
    }
}

function resetMeterHighlights() {
    meterGpu.classList.remove('winner');
    if (meterParity) meterParity.classList.remove('winner');
    meterUfuzzy.classList.remove('winner');
    meterNative.classList.remove('winner');
}

async function executeLiveSearch() {
    if (!currentDataset) return;
    const query = searchQueryInput.value.trim();
    const mode = searchModeSelect.value as 'substring' | 'fuzzy';
    const threadMode = executionThreadSelect?.value || 'worker';

    if (!query) {
        meterGpuVal.textContent = '-- ms';
        if (meterParityVal) meterParityVal.textContent = '-- ms';
        meterUfuzzyVal.textContent = '-- ms';
        meterNativeVal.textContent = '-- ms';
        meterGpuSub.textContent = 'Awaiting query';
        resultsList.innerHTML = '<div style="color: var(--text-muted); text-align: center; margin-top: 2rem;">Type in the box above to search.</div>';
        resultsCountSummary.textContent = 'Matched 0 items';
        resetMeterHighlights();
        return;
    }

    const queryId = ++activeQuerySeq;
    const requestGeneration = activeDatasetGeneration;

    // 1. Offload to Web Worker if worker mode selected and worker is ready
    if (threadMode === 'worker' && searchWorker && workerReady) {
        searchWorker.postMessage({
            type: 'SEARCH',
            payload: {
                queryId,
                query,
                mode,
                limit: 1000,
                runCpuComparison: true,
                datasetGeneration: requestGeneration
            }
        });
        return;
    }

    // 2. Direct Main Thread Execution (UI thread)
    let gpuResult: SearchResult | null = null;
    if (gpuEngine.isReady) {
        try {
            gpuResult = await gpuEngine.search(query, { mode, maxResults: 1000 });
        } catch (err) {
            console.error('GPU search error:', err);
        }
    }

    let parityResult: { durationMs: number; totalMatches: number; results: any[] } | null = null;
    if (currentDataset && query) {
        const norm = normalizeText(query, true);
        parityResult = searchCpuReference(
            currentDataset.recordTokens,
            norm.tokens,
            mode,
            1000,
            currentDataset.strings
        );
    }

    const ufuzzyResult: CPUSearchResult = cpuEngine.searchUFuzzy(currentDataset.strings, query, 1000);
    const nativeResult: CPUSearchResult = cpuEngine.searchNative(currentDataset.strings, query, 1000);

    if (queryId === activeQuerySeq) {
        handleSearchResults(gpuResult, parityResult, ufuzzyResult, nativeResult, query);
    }
}

function renderResults(results: Array<{ index: number; score?: number; text?: string }>, query: string) {
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
            highlightedHtml = `${escapeHtml(before)}<mark>${escapeHtml(match)}</mark>${escapeHtml(after)}`;
        } else {
            highlightedHtml = escapeHtml(text);
        }

        const scoreInfo = item.score !== undefined ? ` | score: ${item.score}` : '';
        div.innerHTML = `
            <div>${highlightedHtml}</div>
            <div class="result-score">#${item.index}${scoreInfo}</div>
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
    btnDownloadAll.disabled = true;
    btnDownloadCsv.disabled = true;
    btnDownloadPngSub.disabled = true;
    btnDownloadPngFuz.disabled = true;
    if (btnCopySummary) btnCopySummary.disabled = true;

    benchmarkStatus.textContent = 'Benchmark running...';
    benchmarkProgressBox.style.display = 'block';

    tableBodySubstring.innerHTML = '';
    tableBodyFuzzy.innerHTML = '';
    substringBenchmarkResults = [];
    fuzzyBenchmarkResults = [];

    const params = new URLSearchParams(window.location.search);
    const autorun = params.get('autorun');
    const sizesParam = params.get('sizes');
    const corpusParam = params.get('corpus') as CorpusType | null;
    const queryParam = params.get('query');
    const warmupsParam = params.get('warmups');
    const samplesParam = params.get('samples');

    const sizes = sizesParam
        ? sizesParam.split(',').map(s => parseInt(s.trim(), 10)).filter(n => !isNaN(n) && n > 0)
        : [10_000, 100_000, 500_000, 1_000_000, 2_000_000];
    const query = queryParam || searchQueryInput.value.trim() || 'AuthController';
    const corpus: CorpusType = (corpusParam && ['ascii', 'cjk', 'emoji'].includes(corpusParam))
        ? corpusParam
        : ((corpusTypeSelect?.value as CorpusType) || 'ascii');
    const pWarmups = warmupsParam ? parseInt(warmupsParam, 10) : NaN;
    const warmups = Number.isFinite(pWarmups) && pWarmups >= 1 ? Math.min(pWarmups, 100) : 5;
    const pSamples = samplesParam ? parseInt(samplesParam, 10) : NaN;
    const samples = Number.isFinite(pSamples) && pSamples >= 1 ? Math.min(pSamples, 100) : 20;

    const runSubstring = autorun !== 'fuzzy';
    const runFuzzy = autorun !== 'substring';
    const totalSteps = (runSubstring ? sizes.length : 0) + (runFuzzy ? sizes.length : 0);

    console.log('[main] Starting runFullBenchmark:', { autorun, sizes, corpus, warmups, samples, runSubstring, runFuzzy });

    try {
        if (runSubstring) {
            // PHASE 1: Substring Algorithm
            tabBtnSubstring.click(); // Focus substring tab
            const subResults = await benchmarkRunner.runBenchmark(
                sizes,
                query,
                'substring',
                { sizes, query, mode: 'substring', corpusType: corpus, warmups, samples },
                (progress) => {
                    const percent = Math.round((progress.currentStep / totalSteps) * 100);
                    benchmarkProgressBar.style.width = `${percent}%`;
                    benchmarkProgressText.textContent = `[1/${runFuzzy ? '2' : '1'} Substring] ${progress.stepName}`;

                    if (progress.currentRow) {
                        substringBenchmarkResults.push(progress.currentRow);
                        appendBenchmarkTableRow(tableBodySubstring, progress.currentRow);
                        drawBenchmarkChart(
                            chartSubstring,
                            substringBenchmarkResults,
                            `WebGPU vs CPU: Exact Substring Search (${corpus.toUpperCase()})`,
                            getChartMeta(query, substringBenchmarkResults)
                        );
                    }
                }
            );
            if (substringBenchmarkResults.length === 0 && subResults.length > 0) {
                substringBenchmarkResults = subResults;
            }
            console.log(`[main] Substring benchmark finished with ${substringBenchmarkResults.length} rows.`);

            // Allow UI to breathe
            if (runFuzzy) {
                await new Promise(r => setTimeout(r, 60));
            }
        }

        if (runFuzzy) {
            // PHASE 2: Fuzzy Algorithm
            tabBtnFuzzy.click(); // Focus fuzzy tab
            const stepOffset = runSubstring ? sizes.length : 0;
            const fuzResults = await benchmarkRunner.runBenchmark(
                sizes,
                query,
                'fuzzy',
                { sizes, query, mode: 'fuzzy', corpusType: corpus, warmups, samples },
                (progress) => {
                    const currentTotal = stepOffset + progress.currentStep;
                    const percent = Math.round((currentTotal / totalSteps) * 100);
                    benchmarkProgressBar.style.width = `${percent}%`;
                    benchmarkProgressText.textContent = `[${runSubstring ? '2/2' : '1/1'} Fuzzy] ${progress.stepName}`;

                    if (progress.currentRow) {
                        fuzzyBenchmarkResults.push(progress.currentRow);
                        appendBenchmarkTableRow(tableBodyFuzzy, progress.currentRow);
                        drawBenchmarkChart(
                            chartFuzzy,
                            fuzzyBenchmarkResults,
                            `WebGPU vs CPU: Fuzzy Subsequence Search (${corpus.toUpperCase()})`,
                            getChartMeta(query, fuzzyBenchmarkResults)
                        );
                    }
                }
            );
            if (fuzzyBenchmarkResults.length === 0 && fuzResults.length > 0) {
                fuzzyBenchmarkResults = fuzResults;
            }
            console.log(`[main] Fuzzy benchmark finished with ${fuzzyBenchmarkResults.length} rows.`);
        }

        benchmarkStatus.textContent = 'Benchmark Complete';
        benchmarkProgressText.textContent = `Completed benchmark matrix (${sizes.map(s => s.toLocaleString()).join(', ')} rows)!`;

        // Store results globally
        (window as any).__BENCHMARK_RESULTS__ = {
            substring: substringBenchmarkResults,
            fuzzy: fuzzyBenchmarkResults,
            corpus,
            query,
            qualificationStatus: (substringBenchmarkResults[0]?.qualificationStatus || fuzzyBenchmarkResults[0]?.qualificationStatus || 'pending-hardware'),
            hardwareQualified: (substringBenchmarkResults[0]?.hardwareQualified ?? fuzzyBenchmarkResults[0]?.hardwareQualified ?? false),
            adapterInfo: (window as any).gpuEngine?.adapterInfo || null,
            timestamp: new Date().toISOString()
        };

        // Enable download buttons
        btnDownloadAll.disabled = false;
        btnDownloadCsv.disabled = false;
        btnDownloadPngSub.disabled = false;
        btnDownloadPngFuz.disabled = false;
        if (btnCopySummary) btnCopySummary.disabled = false;

        // Re-synchronize VRAM with active playground dataset
        const restoreSize = parseInt(datasetSizeSelect.value, 10);
        const restoreCorpus = (corpusTypeSelect?.value || 'ascii') as CorpusType;
        await switchDataset(restoreSize, restoreCorpus);
    } catch (err: any) {
        console.error('Benchmark failed:', err);
        (window as any).__BENCHMARK_ERROR__ = String(err?.stack || err?.message || err);
        benchmarkStatus.textContent = 'Benchmark Failed';
        benchmarkProgressText.textContent = `Error: ${err}`;
    } finally {
        btnRunBenchmark.disabled = false;
    }
}

function appendBenchmarkTableRow(tbody: HTMLElement, row: BenchmarkRowResult) {
    const tr = document.createElement('tr');

    const statusBadge = row.qualificationStatus === 'qualified'
        ? (row.crossover.gpuRetainedBeatsUfuzzy
            ? `<span class="tag-crossover-win">⚡ GPU Win (${row.retainedVsUfuzzySpeedup}x)</span>`
            : `<span class="tag-crossover-loss">CPU Wins (${(1 / (row.retainedVsUfuzzySpeedup || 1)).toFixed(1)}x)</span>`)
        : `<span class="tag-crossover-loss" title="Running under software rasterizer or mock device">Pending-HW (${row.retainedVsUfuzzySpeedup}x)</span>`;

    const gpuRetained = row.gpuRetained.medianMs > 0 ? `${row.gpuRetained.medianMs} / ${row.gpuRetained.p95Ms} ms` : 'N/A';
    const gpuColdTotal = row.gpuCold.totalMs > 0 ? `${row.gpuCold.totalMs} ms` : 'N/A';
    const parityMs = row.cpuParity ? `${row.cpuParity.medianMs} / ${row.cpuParity.p95Ms} ms` : (row.cpuParityMs ? `${row.cpuParityMs} ms` : '-');
    const ufuzzyMs = row.ufuzzy ? `${row.ufuzzy.medianMs} / ${row.ufuzzy.p95Ms} ms` : `${row.ufuzzyMs} ms`;
    const jsNativeMs = row.jsNative ? `${row.jsNative.medianMs} ms` : `${row.jsNativeMs} ms`;
    const speedupParityText = row.retainedVsParitySpeedup > 0 ? `<strong>${row.retainedVsParitySpeedup}x</strong>` : '-';
    const speedupUfuzzyText = row.retainedVsUfuzzySpeedup > 0 ? `<strong>${row.retainedVsUfuzzySpeedup}x</strong>` : '-';

    const uiFpsHtml = row.uiTelemetry
        ? `<div style="font-size: 0.8rem; line-height: 1.25;">
             <span style="color: #10b981; font-weight: 600;">⚡ ${row.uiTelemetry.workerFps} FPS</span>
             <span style="color: var(--text-muted); font-size: 0.72rem; display: block;">(${row.uiTelemetry.mainThreadFps} FPS, ${row.uiTelemetry.jankSpikes} jank)</span>
           </div>`
        : `<span style="color: var(--text-muted); font-size: 0.8rem;">~60 FPS</span>`;

    tr.innerHTML = `
        <td><strong>${row.datasetSize.toLocaleString()}</strong></td>
        <td><span style="font-size: 0.75rem; text-transform: uppercase; color: var(--text-muted); font-weight: 600;">${row.corpusType || 'ASCII'}</span></td>
        <td style="color: var(--color-gpu); font-weight: 600;">${gpuRetained}</td>
        <td style="color: var(--text-muted);">${gpuColdTotal}</td>
        <td style="color: #f43f5e; font-weight: 600;">${parityMs}</td>
        <td style="color: var(--color-ufuzzy); font-weight: 600;">${ufuzzyMs}</td>
        <td style="color: var(--color-native);">${jsNativeMs}</td>
        <td>${speedupParityText}</td>
        <td>${speedupUfuzzyText}</td>
        <td>${uiFpsHtml}</td>
        <td>${statusBadge}</td>
    `;
    tbody.appendChild(tr);
}

function getChartMeta(query: string, results: BenchmarkRowResult[]): { subtitle: string; speedupBadge: string } {
    const hw = gpuEngine?.adapterInfo;
    const gpuName = hw ? `${hw.vendor} - ${hw.device} (${hw.architecture})` : 'WebGPU Disabled (CPU Fallback)';
    const subtitle = `Hardware: ${gpuName} | Query: "${query}" | Dataset: 10k to 2M rows`;

    let maxSpeedup = 0;
    let maxSpeedupSize = 0;
    for (const r of results) {
        if (r.retainedVsUfuzzySpeedup > maxSpeedup) {
            maxSpeedup = r.retainedVsUfuzzySpeedup;
            maxSpeedupSize = r.datasetSize;
        }
    }

    const speedupBadge = maxSpeedup > 1
        ? `⚡ Max GPU Speedup: ${maxSpeedup}x (${maxSpeedupSize.toLocaleString()} rows)`
        : (!gpuEngine?.isReady ? 'WebGPU Disabled' : '');

    return { subtitle, speedupBadge };
}

function drawBenchmarkChart(
    svgElement: SVGSVGElement,
    results: BenchmarkRowResult[],
    title: string,
    meta: { subtitle: string; speedupBadge: string }
) {
    if (!svgElement || results.length === 0) return;

    const width = 1000;
    const height = 340;
    const padding = { top: 85, right: 70, bottom: 45, left: 65 };

    const graphWidth = width - padding.left - padding.right;
    const graphHeight = height - padding.top - padding.bottom;

    // Find max value for Y scale
    let maxTime = 10;
    for (const r of results) {
        const gpuTime = r.gpuRetained.medianMs || r.gpuRetained.totalMs || 0;
        const parityTime = r.cpuParity?.medianMs || r.cpuParityMs || 0;
        const ufuzzyTime = r.ufuzzy?.medianMs || r.ufuzzyMs || 0;
        const nativeTime = r.jsNative?.medianMs || r.jsNativeMs || 0;
        maxTime = Math.max(maxTime, gpuTime, parityTime, ufuzzyTime, nativeTime);
    }
    maxTime = Math.ceil(maxTime * 1.15); // Add headroom

    // Clear SVG
    svgElement.innerHTML = '';

    // 1. Background Card & Border
    const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    bg.setAttribute('width', `${width}`);
    bg.setAttribute('height', `${height}`);
    bg.setAttribute('fill', '#080c14');
    bg.setAttribute('rx', '8');
    svgElement.appendChild(bg);

    const border = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    border.setAttribute('width', `${width}`);
    border.setAttribute('height', `${height}`);
    border.setAttribute('fill', 'none');
    border.setAttribute('stroke', 'rgba(255, 255, 255, 0.08)');
    border.setAttribute('stroke-width', '1');
    border.setAttribute('rx', '8');
    svgElement.appendChild(border);

    // 2. Title & Subtitle
    const titleEl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    titleEl.setAttribute('x', '30');
    titleEl.setAttribute('y', '32');
    titleEl.setAttribute('fill', '#f8fafc');
    titleEl.setAttribute('font-size', '16');
    titleEl.setAttribute('font-weight', '700');
    titleEl.setAttribute('font-family', '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif');
    titleEl.textContent = title;
    svgElement.appendChild(titleEl);

    const subtitleEl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    subtitleEl.setAttribute('x', '30');
    subtitleEl.setAttribute('y', '54');
    subtitleEl.setAttribute('fill', '#94a3b8');
    subtitleEl.setAttribute('font-size', '12');
    subtitleEl.setAttribute('font-family', '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif');
    subtitleEl.textContent = meta.subtitle;
    svgElement.appendChild(subtitleEl);

    // 3. Embedded Legend (top right)
    const legendItems = [
        { color: '#38bdf8', label: 'WebGPU' },
        { color: '#f43f5e', label: 'CPU Parity' },
        { color: '#f59e0b', label: 'uFuzzy' },
        { color: '#a855f7', label: 'Native' }
    ];

    let legendX = 460;
    legendItems.forEach(item => {
        const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        dot.setAttribute('cx', `${legendX}`);
        dot.setAttribute('cy', '27');
        dot.setAttribute('r', '5');
        dot.setAttribute('fill', item.color);
        svgElement.appendChild(dot);

        const lbl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        lbl.setAttribute('x', `${legendX + 9}`);
        lbl.setAttribute('y', '31');
        lbl.setAttribute('fill', '#f8fafc');
        lbl.setAttribute('font-size', '11');
        lbl.setAttribute('font-weight', '600');
        lbl.setAttribute('font-family', '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif');
        lbl.textContent = item.label;
        svgElement.appendChild(lbl);

        legendX += 120;
    });

    // 4. Speedup Highlight Badge (top right below legend)
    if (meta.speedupBadge) {
        const badgeEl = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        badgeEl.setAttribute('x', `${width - 30}`);
        badgeEl.setAttribute('y', '54');
        badgeEl.setAttribute('fill', '#10b981');
        badgeEl.setAttribute('font-size', '12');
        badgeEl.setAttribute('font-weight', '700');
        badgeEl.setAttribute('text-anchor', 'end');
        badgeEl.setAttribute('font-family', '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif');
        badgeEl.textContent = meta.speedupBadge;
        svgElement.appendChild(badgeEl);
    }

    // 5. Y-Axis Title
    const yAxisLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    yAxisLabel.setAttribute('x', `${padding.left}`);
    yAxisLabel.setAttribute('y', '74');
    yAxisLabel.setAttribute('fill', '#64748b');
    yAxisLabel.setAttribute('font-size', '10');
    yAxisLabel.setAttribute('font-weight', '600');
    yAxisLabel.setAttribute('font-family', '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif');
    yAxisLabel.textContent = '▲ Latency (ms) [lower is better]';
    svgElement.appendChild(yAxisLabel);

    // 6. X-Axis Title
    const xAxisLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    xAxisLabel.setAttribute('x', `${padding.left + graphWidth / 2}`);
    xAxisLabel.setAttribute('y', `${height - 10}`);
    xAxisLabel.setAttribute('fill', '#64748b');
    xAxisLabel.setAttribute('font-size', '11');
    xAxisLabel.setAttribute('font-weight', '600');
    xAxisLabel.setAttribute('text-anchor', 'middle');
    xAxisLabel.setAttribute('font-family', '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif');
    xAxisLabel.textContent = 'Dataset Size (Number of Items)';
    svgElement.appendChild(xAxisLabel);

    // 7. Horizontal Grid lines & Y values
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
        text.setAttribute('x', `${padding.left - 8}`);
        text.setAttribute('y', `${y + 4}`);
        text.setAttribute('fill', '#64748b');
        text.setAttribute('font-size', '11');
        text.setAttribute('text-anchor', 'end');
        text.setAttribute('font-family', 'monospace');
        text.textContent = `${Math.round(yVal)}ms`;
        svgElement.appendChild(text);
    }

    const getX = (idx: number) => {
        if (results.length === 1) return padding.left + graphWidth / 2;
        return padding.left + (idx / (results.length - 1)) * graphWidth;
    };

    const getY = (val: number) => {
        return padding.top + graphHeight - (val / maxTime) * graphHeight;
    };

    // 8. X-Ticks and Labels
    for (let i = 0; i < results.length; i++) {
        const x = getX(i);
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', `${x}`);
        text.setAttribute('y', `${padding.top + graphHeight + 18}`);
        text.setAttribute('fill', '#94a3b8');
        text.setAttribute('font-size', '11');
        text.setAttribute('font-weight', '600');
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('font-family', '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif');
        const sizeLabel = results[i].datasetSize >= 1_000_000
            ? `${results[i].datasetSize / 1_000_000}M`
            : `${results[i].datasetSize / 1_000}k`;
        text.textContent = sizeLabel;
        svgElement.appendChild(text);
    }

    // 9. Draw series lines, points, and value labels
    const drawSeries = (color: string, getter: (r: BenchmarkRowResult) => number, labelOffsetY: number) => {
        if (results.length > 1) {
            const points = results.map((r, idx) => `${getX(idx)},${getY(getter(r))}`).join(' ');
            const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
            polyline.setAttribute('fill', 'none');
            polyline.setAttribute('stroke', color);
            polyline.setAttribute('stroke-width', '2.5');
            polyline.setAttribute('points', points);
            svgElement.appendChild(polyline);
        }

        results.forEach((r, idx) => {
            const cx = getX(idx);
            const val = getter(r);
            const cy = getY(val);

            // Point dot
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', `${cx}`);
            circle.setAttribute('cy', `${cy}`);
            circle.setAttribute('r', '4.5');
            circle.setAttribute('fill', color);
            circle.setAttribute('stroke', '#0c1220');
            circle.setAttribute('stroke-width', '2');
            svgElement.appendChild(circle);

            // Value text callout on points
            if (idx >= results.length - 2 || val > 10 || results.length <= 3) {
                const valText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                valText.setAttribute('x', `${cx}`);
                valText.setAttribute('y', `${cy + labelOffsetY}`);
                valText.setAttribute('fill', color);
                valText.setAttribute('font-size', '10');
                valText.setAttribute('font-weight', '700');
                valText.setAttribute('text-anchor', 'middle');
                valText.setAttribute('font-family', 'monospace');
                valText.textContent = `${val.toFixed(1)}ms`;
                svgElement.appendChild(valText);
            }
        });
    };

    drawSeries('#a855f7', r => r.jsNative?.medianMs ?? r.jsNativeMs, -8);
    drawSeries('#f59e0b', r => r.ufuzzy?.medianMs ?? r.ufuzzyMs, -8);
    drawSeries('#f43f5e', r => r.cpuParity?.medianMs ?? r.cpuParityMs ?? 0, -8);
    if (results.some(r => (r.gpuRetained.medianMs ?? r.gpuRetained.totalMs) > 0)) {
        drawSeries('#38bdf8', r => r.gpuRetained.medianMs || r.gpuRetained.totalMs, 14);
    }
}

// Start application
if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', () => {
        init().catch(console.error);
    });
} else {
    init().catch(console.error);
}
