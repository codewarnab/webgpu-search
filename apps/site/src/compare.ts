import {
  normalizeText,
  packDataset,
  scoreExactMatches,
  WebGPUEngine,
  type SearchMode,
  type SearchResultItem
} from 'webgpu-search';
import './style.css';

const sizeInput = document.querySelector<HTMLSelectElement>('#compare-size');
const modeInput = document.querySelector<HTMLSelectElement>('#compare-mode');
const queryInput = document.querySelector<HTMLInputElement>('#compare-query');
const runButton = document.querySelector<HTMLButtonElement>('#compare-run');
const status = document.querySelector<HTMLElement>('#compare-status');
const resultsPanel = document.querySelector<HTMLElement>('#compare-results');
const fields = {
  device: document.querySelector<HTMLElement>('#compare-device'),
  gpuMedian: document.querySelector<HTMLElement>('#gpu-median'),
  gpuDetail: document.querySelector<HTMLElement>('#gpu-detail'),
  cpuMedian: document.querySelector<HTMLElement>('#cpu-median'),
  cpuDetail: document.querySelector<HTMLElement>('#cpu-detail'),
  agreement: document.querySelector<HTMLElement>('#agreement'),
  agreementDetail: document.querySelector<HTMLElement>('#agreement-detail'),
  note: document.querySelector<HTMLElement>('#compare-note'),
  hits: document.querySelector<HTMLOListElement>('#compare-hits')
};

const prefixes = ['src/auth', 'src/search', 'src/components', 'packages/engine', 'apps/docs', 'examples/palette'];
const nouns = ['Auth', 'User', 'Profile', 'Order', 'Invoice', 'Search', 'Query', 'Worker', 'Document', 'Session', 'Token', 'Index'];
const suffixes = ['Controller', 'Service', 'Handler', 'Manager', 'Provider', 'Repository', 'View', 'Worker', 'Index'];
const MEASUREMENTS = 9;
const WARMUPS = 2;

function generateCorpus(size: number): string[] {
  return Array.from({ length: size }, (_, i) =>
    `${prefixes[i % prefixes.length]}/${nouns[(i * 7 + 3) % nouns.length]}${nouns[(i * 13 + 5) % nouns.length]}${suffixes[(i * 17 + 1) % suffixes.length]}_${i}.ts`
  );
}

function median(samples: number[]): number {
  const sorted = [...samples].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] ?? 0;
}

function p95(samples: number[]): number {
  const sorted = [...samples].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * 0.95) - 1)] ?? 0;
}

function formatMs(value: number): string {
  return `${value < 10 ? value.toFixed(2) : value.toFixed(1)} ms`;
}

function setStatus(message: string): void {
  if (status) status.textContent = message;
}

function matchesAgree(gpu: SearchResultItem[], cpu: SearchResultItem[], gpuTotal: number, cpuTotal: number): boolean {
  return gpuTotal === cpuTotal && gpu.length === cpu.length &&
    gpu.every((item, i) => item.index === cpu[i]?.index && item.score === cpu[i]?.score);
}

function renderHits(items: SearchResultItem[]): void {
  if (!fields.hits) return;
  fields.hits.replaceChildren();
  for (const item of items.slice(0, 8)) {
    const row = document.createElement('li');
    row.textContent = `${item.text} · score ${item.score}`;
    fields.hits.append(row);
  }
  if (items.length === 0) {
    const row = document.createElement('li');
    row.textContent = 'No matches for this query.';
    fields.hits.append(row);
  }
}

async function runComparison(): Promise<void> {
  if (!sizeInput || !modeInput || !queryInput || !runButton || !resultsPanel) return;
  const size = Number(sizeInput.value);
  const mode = modeInput.value as SearchMode;
  const query = queryInput.value.trim();
  if (!query) {
    setStatus('Enter a query before starting the comparison.');
    queryInput.focus();
    return;
  }

  runButton.disabled = true;
  resultsPanel.hidden = true;
  let engine: WebGPUEngine | null = new WebGPUEngine();
  try {
    setStatus('Preparing the local corpus…');
    let gpuReady = false;
    let gpuFailure = '';
    const strings = generateCorpus(size);
    const tokens = strings.map(value => normalizeText(value, true).tokens);
    const queryTokens = normalizeText(query, true).tokens;
    const packed = packDataset(tokens, { normalized: true });

    try {
      setStatus('Checking WebGPU support…');
      gpuReady = await engine.init();
      if (gpuReady) {
        setStatus('Uploading the corpus to WebGPU…');
        await engine.loadDataset(packed);
      }
    } catch (error) {
      gpuReady = false;
      gpuFailure = error instanceof Error ? error.message : String(error);
    }
    const adapter = engine.adapterInfo;
    setStatus('Warming up, then collecting 9 measurements per engine…');

    const gpuSamples: number[] = [];
    let gpuLast: SearchResultItem[] = [];
    let gpuTotal = 0;
    let gpuOverflow = false;
    if (gpuReady) {
      try {
        for (let i = 0; i < WARMUPS; i++) {
          await engine.search(query, { mode, maxResults: 100 });
        }
        for (let i = 0; i < MEASUREMENTS; i++) {
          const start = performance.now();
          const response = await engine.search(query, { mode, maxResults: 100 });
          gpuSamples.push(performance.now() - start);
          gpuLast = response.results.map(item => ({ ...item, text: strings[item.index] ?? item.text }));
          gpuTotal = response.totalMatches;
          gpuOverflow = response.hasOverflow;
        }
      } catch (error) {
        gpuReady = false;
        gpuFailure = error instanceof Error ? error.message : String(error);
        gpuSamples.length = 0;
        gpuLast = [];
        gpuTotal = 0;
      }
    }

    setStatus('Measuring the exact CPU route…');
    const cpuRun = () => scoreExactMatches(tokens, queryTokens, mode, 100, strings);
    for (let i = 0; i < WARMUPS; i++) cpuRun();
    const cpuSamples: number[] = [];
    let cpuLast: SearchResultItem[] = [];
    let cpuTotal = 0;
    for (let i = 0; i < MEASUREMENTS; i++) {
      const start = performance.now();
      const response = cpuRun();
      cpuSamples.push(performance.now() - start);
      cpuLast = response.results;
      cpuTotal = response.totalMatches;
    }

    if (fields.gpuMedian) fields.gpuMedian.textContent = gpuReady ? formatMs(median(gpuSamples)) : 'Not available';
    if (fields.gpuDetail) fields.gpuDetail.textContent = gpuReady
      ? `p95 ${formatMs(p95(gpuSamples))} · ${gpuTotal.toLocaleString()} matches`
      : gpuFailure ? 'WebGPU search failed; no GPU timing recorded' : 'WebGPU unavailable; no GPU timing recorded';
    if (fields.cpuMedian) fields.cpuMedian.textContent = formatMs(median(cpuSamples));
    if (fields.cpuDetail) fields.cpuDetail.textContent = `p95 ${formatMs(p95(cpuSamples))} · ${cpuTotal.toLocaleString()} matches`;

    const agree = gpuReady && matchesAgree(gpuLast, cpuLast, gpuTotal, cpuTotal);
    if (fields.agreement) fields.agreement.textContent = gpuReady ? (gpuOverflow ? 'Overflow' : agree ? 'Match' : 'Different') : 'CPU only';
    if (fields.agreementDetail) fields.agreementDetail.textContent = gpuReady
      ? `${gpuTotal.toLocaleString()} WebGPU matches · ${cpuTotal.toLocaleString()} CPU matches${gpuOverflow ? ' · candidate cap exceeded' : ''}`
      : gpuFailure ? 'WebGPU search failed; CPU-only results shown' : 'WebGPU was not available in this browser';
    if (fields.device) {
      const deviceName = adapter ? [adapter.vendor, adapter.device].filter(Boolean).join(' · ') : 'No WebGPU adapter';
      fields.device.textContent = gpuReady
        ? `WebGPU ran on ${deviceName}. Measurements are local to this browser and include the search call, not index setup.${gpuOverflow ? ' Its candidate capacity was exceeded, so the returned top results may be incomplete.' : ''}`
        : gpuFailure
          ? `WebGPU search failed (${gpuFailure}); only exact CPU results are shown. No GPU timing or speedup is reported.`
          : 'WebGPU did not initialize. Only exact CPU results are shown; there is no GPU timing or speedup.';
    }
    if (fields.note) fields.note.textContent = `Corpus: ${size.toLocaleString()} generated records · Mode: ${mode} · Query: “${query}”. Dataset normalization, packing, and any GPU upload are excluded from warm-search measurements.`;
    renderHits(gpuReady ? gpuLast : cpuLast);
    resultsPanel.hidden = false;
    setStatus(`Completed ${MEASUREMENTS} timed searches per available engine.`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus(`Comparison failed: ${message}`);
  } finally {
    engine?.destroy();
    engine = null;
    runButton.disabled = false;
  }
}

if (runButton) runButton.addEventListener('click', () => { void runComparison(); });
