import { CPUEngine, WebGPUEngine } from 'webgpu-search';
import { BenchmarkRunner, type BenchmarkRowResult } from './benchmark';
import type { CorpusType } from './dataset';
import {
  downloadBlob,
  generateBenchmarkCsv,
  generateMarkdownSummary,
  svgToPngBlob
} from './export';

const runButton = document.querySelector<HTMLButtonElement>('#benchmark-run');
const status = document.querySelector<HTMLElement>('#benchmark-status');
const progress = document.querySelector<HTMLElement>('#benchmark-progress');
const corpusSelect = document.querySelector<HTMLSelectElement>('#benchmark-corpus');
const modeSelect = document.querySelector<HTMLSelectElement>('#benchmark-mode');
const queryInput = document.querySelector<HTMLInputElement>('#benchmark-query');
const summary = document.querySelector<HTMLElement>('#benchmark-summary');
const hardwareSummary = document.querySelector<HTMLElement>('#benchmark-hardware');
const chart = document.querySelector<SVGSVGElement>('#benchmark-chart');
const tableBody = document.querySelector<HTMLTableSectionElement>('#benchmark-table-body');
const details = document.querySelector<HTMLElement>('#benchmark-details');
const exportCsv = document.querySelector<HTMLButtonElement>('#export-csv');
const exportMarkdown = document.querySelector<HTMLButtonElement>('#export-markdown');
const exportChart = document.querySelector<HTMLButtonElement>('#export-chart');

let currentRows: BenchmarkRowResult[] = [];
let currentMode: 'substring' | 'fuzzy' = 'substring';
let currentCorpus: CorpusType = 'ascii';
let adapterInfo: WebGPUEngine['adapterInfo'] = null;

function formatMs(value: number): string {
  return value > 0 ? `${value.toFixed(2)} ms` : '—';
}

function addCell(row: HTMLTableRowElement, value: string): void {
  const cell = document.createElement('td');
  cell.textContent = value;
  row.append(cell);
}

function setStatus(message: string): void {
  if (status) status.textContent = message;
}

function renderTable(rows: BenchmarkRowResult[]): void {
  if (!tableBody) return;
  tableBody.replaceChildren();
  for (const result of rows) {
    const row = document.createElement('tr');
    addCell(row, result.datasetSize.toLocaleString());
    addCell(row, result.corpusType.toUpperCase());
    addCell(row, result.gpuRetained.medianMs > 0
      ? `${formatMs(result.gpuRetained.medianMs)} / ${formatMs(result.gpuRetained.p95Ms)}`
      : 'GPU did not run');
    addCell(row, formatMs(result.gpuCold.uploadMs));
    addCell(row, `${formatMs(result.cpuParity.medianMs)} / ${formatMs(result.cpuParity.p95Ms)}`);
    addCell(row, `${formatMs(result.ufuzzy.medianMs)} / ${formatMs(result.ufuzzy.p95Ms)}`);
    addCell(row, `${formatMs(result.jsNative.medianMs)} / ${formatMs(result.jsNative.p95Ms)}`);
    addCell(row, `${(result.vramAllocation.totalBytes / (1024 * 1024)).toFixed(1)} MB`);
    addCell(row, `${result.matchCount.gpu.toLocaleString()} / ${result.matchCount.cpuParity.toLocaleString()}`);
    addCell(row, `${result.uiTelemetry.jankSpikes} spikes · ${result.uiTelemetry.mainThreadJankMs.toFixed(1)} ms max`);
    addCell(row, result.gpuRetained.medianMs <= 0
      ? 'GPU not run'
      : result.qualificationStatus === 'qualified' ? 'Qualified' : 'Pending hardware');
    tableBody.append(row);
  }
}

function renderDetails(rows: BenchmarkRowResult[]): void {
  if (!details) return;
  details.replaceChildren();
  for (const result of rows) {
    const item = document.createElement('p');
    item.textContent = `${result.datasetSize.toLocaleString()} rows · ${result.corpusType.toUpperCase()} · normalize ${formatMs(result.packing.normalizeMs)} · pack ${formatMs(result.packing.packMs)} · upload ${formatMs(result.packing.uploadMs)} · estimated VRAM ${(result.vramAllocation.totalBytes / (1024 * 1024)).toFixed(2)} MB · overflow ${result.hasOverflow ? 'yes' : 'no'}`;
    details.append(item);
  }
}

function renderChart(rows: BenchmarkRowResult[]): void {
  if (!chart) return;
  chart.replaceChildren();
  const width = 900;
  const rowHeight = 88;
  const height = Math.max(190, rows.length * rowHeight + 75);
  chart.setAttribute('viewBox', `0 0 ${width} ${height}`);
  chart.setAttribute('height', `${height}`);
  chart.setAttribute('aria-label', `Median ${currentMode} latency for selected dataset sizes`);
  const max = Math.max(1, ...rows.flatMap(result => [
    result.gpuRetained.medianMs,
    result.cpuParity.medianMs,
    result.ufuzzy.medianMs,
    result.jsNative.medianMs
  ]));
  const scale = (width - 245) / max;
  const svgEl = (name: string, attrs: Record<string, string>, text?: string): SVGElement => {
    const element = document.createElementNS('http://www.w3.org/2000/svg', name);
    for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
    if (text !== undefined) element.textContent = text;
    return element;
  };
  chart.append(svgEl('rect', { x: '0', y: '0', width: `${width}`, height: `${height}`, fill: '#ffffff' }));
  chart.append(svgEl('text', { x: '18', y: '25', fill: '#29292d', 'font-size': '12', 'font-family': 'sans-serif', 'font-weight': '600' }, `Median search latency · ${currentMode}`));
  rows.forEach((result, index) => {
    const top = 48 + index * rowHeight;
    chart.append(svgEl('text', { x: '18', y: `${top + 13}`, fill: '#55565b', 'font-size': '10', 'font-family': 'monospace' }, `${(result.datasetSize / 1000).toLocaleString()}k rows`));
    const values = [
      ['WebGPU retained', result.gpuRetained.medianMs],
      ['CPU exact', result.cpuParity.medianMs],
      ['uFuzzy', result.ufuzzy.medianMs],
      ['JS native', result.jsNative.medianMs]
    ] as const;
    values.forEach(([label, value], series) => {
      const y = top + 25 + series * 14;
      chart.append(svgEl('text', { x: '18', y: `${y + 8}`, fill: '#68696e', 'font-size': '9', 'font-family': 'sans-serif' }, label));
      const barWidth = value > 0 ? Math.max(2, value * scale) : 0;
      chart.append(svgEl('rect', { x: '125', y: `${y}`, width: `${barWidth}`, height: '9', rx: '2', fill: series === 0 ? '#252529' : '#a5a5aa' }));
      chart.append(svgEl('text', { x: `${Math.min(width - 8, 134 + barWidth)}`, y: `${y + 8}`, fill: '#55565b', 'font-size': '9', 'font-family': 'monospace' }, value > 0 ? `${value.toFixed(2)} ms` : 'not run'));
    });
  });
}

function selectedSizes(): number[] {
  return [...document.querySelectorAll<HTMLInputElement>('input[name="size"]:checked')]
    .map(input => Number(input.value))
    .filter(Number.isFinite)
    .sort((a, b) => a - b);
}

async function runBenchmark(): Promise<void> {
  if (!runButton || !corpusSelect || !modeSelect || !queryInput || !summary) return;
  const sizes = selectedSizes();
  const query = queryInput.value.trim();
  if (sizes.length === 0) {
    setStatus('Choose at least one dataset size.');
    return;
  }
  if (!query) {
    setStatus('Enter a search query.');
    queryInput.focus();
    return;
  }

  runButton.disabled = true;
  summary.hidden = true;
  if (progress) progress.style.width = '0%';
  currentRows = [];
  currentMode = modeSelect.value as 'substring' | 'fuzzy';
  currentCorpus = corpusSelect.value as CorpusType;

  const gpuEngine = new WebGPUEngine();
  const cpuEngine = new CPUEngine();
  try {
    setStatus('Checking WebGPU support…');
    let gpuReady = false;
    try {
      gpuReady = await gpuEngine.init();
    } catch {
      gpuReady = false;
    }
    adapterInfo = gpuEngine.adapterInfo;
    setStatus(gpuReady
      ? `WebGPU ready · running ${sizes.length} size${sizes.length === 1 ? '' : 's'} with 5 warmups + 20 samples…`
      : `WebGPU unavailable · CPU baselines will run for ${sizes.length} size${sizes.length === 1 ? '' : 's'}…`);

    const runner = new BenchmarkRunner(gpuEngine, cpuEngine);
    currentRows = await runner.runBenchmark(sizes, query, currentMode, {
      sizes,
      query,
      mode: currentMode,
      corpusType: currentCorpus,
      warmups: 5,
      samples: 20,
      onProgress: state => {
        setStatus(state.stepName);
        if (progress) progress.style.width = `${Math.round((state.currentStep / state.totalSteps) * 100)}%`;
      }
    });

    if (hardwareSummary) {
      const adapter = adapterInfo ? `${adapterInfo.vendor} · ${adapterInfo.device}` : 'No WebGPU adapter';
      const qualified = currentRows.some(result => result.hardwareQualified);
      hardwareSummary.textContent = gpuReady
        ? `${adapter} · ${qualified ? 'physical hardware identified' : 'pending hardware qualification (software/mock adapters are not dedicated-GPU results)'}`
        : 'WebGPU did not initialize. GPU timings are not reported; only CPU baselines are available.';
    }
    renderTable(currentRows);
    renderDetails(currentRows);
    renderChart(currentRows);
    summary.hidden = false;
    setStatus(`Complete · ${currentRows.length} result${currentRows.length === 1 ? '' : 's'} · ${currentCorpus.toUpperCase()} · ${currentMode}.`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus(`Benchmark stopped: ${message}`);
  } finally {
    gpuEngine.destroy();
    runButton.disabled = false;
  }
}

runButton?.addEventListener('click', () => { void runBenchmark(); });

corpusSelect?.addEventListener('change', () => {
  if (!queryInput) return;
  const suggested: Record<CorpusType, string> = { ascii: 'AuthController', cjk: 'ユーザー', emoji: '🧑‍💻' };
  queryInput.value = suggested[corpusSelect.value as CorpusType] ?? suggested.ascii;
});

exportCsv?.addEventListener('click', () => {
  if (currentRows.length === 0) return;
  const csv = generateBenchmarkCsv(
    adapterInfo,
    currentMode === 'substring' ? currentRows : [],
    currentMode === 'fuzzy' ? currentRows : []
  );
  downloadBlob(new Blob([csv], { type: 'text/csv;charset=utf-8' }), `webgpu-search-${currentMode}-${currentCorpus}.csv`);
});

exportMarkdown?.addEventListener('click', () => {
  if (currentRows.length === 0) return;
  const markdown = generateMarkdownSummary(
    adapterInfo,
    currentMode === 'substring' ? currentRows : [],
    currentMode === 'fuzzy' ? currentRows : []
  );
  downloadBlob(new Blob([markdown], { type: 'text/markdown;charset=utf-8' }), `webgpu-search-${currentMode}-${currentCorpus}.md`);
});

exportChart?.addEventListener('click', async () => {
  if (!chart || currentRows.length === 0) return;
  try {
    const blob = await svgToPngBlob(chart, 900, Math.max(190, currentRows.length * 88 + 75));
    downloadBlob(blob, `webgpu-search-${currentMode}-chart.png`);
  } catch (error) {
    setStatus(`Chart export failed: ${error instanceof Error ? error.message : String(error)}`);
  }
});

// Browser regression gate hook (scripts/test-regression.ts waits for
// `window.__IS_INITIALIZED__` and drives `window.gpuEngine` directly).
// The benchmark itself creates per-run engines lazily on click; this
// separate eagerly-initialized engine exists only so the CI browser gate
// can exercise real WGSL dispatch without driving the full UI.
const gateGpuEngine = new WebGPUEngine();
const gateCpuEngine = new CPUEngine();
async function initRegressionGate(): Promise<void> {
  try {
    await gateGpuEngine.init();
  } catch {
    // Engine stays CPU-fallback; the gate reports readiness either way.
  }
  (window as unknown as Record<string, unknown>).gpuEngine = gateGpuEngine;
  (window as unknown as Record<string, unknown>).cpuEngine = gateCpuEngine;
  (window as unknown as Record<string, unknown>).__IS_INITIALIZED__ = true;
}
void initRegressionGate();
