import { SearchIndex } from 'webgpu-search';
import '@fontsource-variable/inter';
import './style.css';
import { formatMs, runQuickCompare } from './quick-benchmark';

const copyButton = document.querySelector<HTMLButtonElement>('#copy-install');

const qbSize = document.querySelector<HTMLSelectElement>('#qb-size');
const qbQuery = document.querySelector<HTMLInputElement>('#qb-query');
const qbRun = document.querySelector<HTMLButtonElement>('#qb-run');
const qbStatus = document.querySelector<HTMLElement>('#qb-status');
const qbPanel = document.querySelector<HTMLElement>('#qb-results');
const qbFields = {
  gpu: document.querySelector<HTMLElement>('#qb-gpu'),
  gpuDetail: document.querySelector<HTMLElement>('#qb-gpu-d'),
  cpu: document.querySelector<HTMLElement>('#qb-cpu'),
  cpuDetail: document.querySelector<HTMLElement>('#qb-cpu-d'),
  agree: document.querySelector<HTMLElement>('#qb-agree'),
  agreeDetail: document.querySelector<HTMLElement>('#qb-agree-d'),
  device: document.querySelector<HTMLElement>('#qb-device'),
  note: document.querySelector<HTMLElement>('#qb-note'),
  verdict: document.querySelector<HTMLElement>('#qb-verdict'),
  chart: document.querySelector<SVGSVGElement>('#qb-chart')
};

const pgData = document.querySelector<HTMLTextAreaElement>('#pg-data');
const pgQuery = document.querySelector<HTMLInputElement>('#pg-query');
const pgRun = document.querySelector<HTMLButtonElement>('#pg-run');
const pgStatus = document.querySelector<HTMLElement>('#pg-status');
const pgResults = document.querySelector<HTMLOListElement>('#pg-results');

function renderQbChart(
  chart: SVGSVGElement,
  entries: Array<{ label: string; medianMs: number; matches: number }>
): void {
  chart.replaceChildren();
  const width = 900;
  const rowHeight = 26;
  const height = 58 + entries.length * rowHeight;
  chart.setAttribute('viewBox', `0 0 ${width} ${height}`);
  const max = Math.max(0.01, ...entries.map(entry => entry.medianMs));
  const fastest = Math.min(...entries.map(entry => entry.medianMs));
  const scale = (width - 330) / max;
  const svgEl = (name: string, attrs: Record<string, string>, text?: string): SVGElement => {
    const element = document.createElementNS('http://www.w3.org/2000/svg', name);
    for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
    if (text !== undefined) element.textContent = text;
    return element;
  };
  chart.append(svgEl('text', { x: '18', y: '26', fill: '#29292d', 'font-size': '13', 'font-family': 'sans-serif', 'font-weight': '600' }, 'Median search latency'));
  entries.forEach((entry, index) => {
    const y = 40 + index * rowHeight;
    const isFastest = entry.medianMs <= fastest;
    chart.append(svgEl('text', { x: '18', y: `${y + 12}`, fill: isFastest ? '#29292d' : '#68696e', 'font-size': '11', 'font-family': 'sans-serif', 'font-weight': isFastest ? '600' : '400' }, entry.label));
    const barWidth = Math.max(2, entry.medianMs * scale);
    chart.append(svgEl('rect', { x: '210', y: `${y + 3}`, width: `${barWidth}`, height: '11', rx: '2', fill: isFastest ? '#252529' : '#a5a5aa' }));
    chart.append(svgEl('text', { x: `${Math.min(width - 8, 220 + barWidth)}`, y: `${y + 12}`, fill: '#55565b', 'font-size': '11', 'font-family': 'monospace' },
      `${formatMs(entry.medianMs)} · ${entry.matches.toLocaleString()} matches`));
  });
}

async function runHomepageBenchmark(): Promise<void> {
  if (!qbSize || !qbQuery || !qbRun || !qbStatus || !qbPanel) return;
  const query = qbQuery.value.trim();
  if (!query) {
    qbStatus.textContent = 'Enter a query before running the benchmark.';
    qbQuery.focus();
    return;
  }
  qbRun.disabled = true;
  qbPanel.hidden = true;
  try {
    const result = await runQuickCompare(Number(qbSize.value), 'fuzzy', query, message => {
      qbStatus.textContent = message;
    });
    if (qbFields.gpu) qbFields.gpu.textContent = result.gpuRan ? formatMs(result.gpuMedianMs) : 'n/a';
    if (qbFields.gpuDetail) qbFields.gpuDetail.textContent = result.gpuRan
      ? `p95 ${formatMs(result.gpuP95Ms)} · ${result.gpuMatches.toLocaleString()} matches`
      : 'WebGPU unavailable — no GPU timing recorded';
    if (qbFields.cpu) qbFields.cpu.textContent = formatMs(result.cpuMedianMs);
    if (qbFields.cpuDetail) qbFields.cpuDetail.textContent =
      `p95 ${formatMs(result.cpuP95Ms)} · ${result.cpuMatches.toLocaleString()} matches`;
    if (qbFields.agree) qbFields.agree.textContent = result.gpuRan ? (result.agree ? 'Match' : 'Different') : 'CPU only';
    if (qbFields.agreeDetail) qbFields.agreeDetail.textContent = result.gpuRan
      ? `${result.gpuMatches.toLocaleString()} WebGPU · ${result.cpuMatches.toLocaleString()} CPU`
      : 'WebGPU was not available in this browser';
    if (qbFields.device) qbFields.device.textContent = result.deviceLabel;
    if (qbFields.chart) {
      const entries = [
        ...(result.gpuRan ? [{ label: 'WebGPU', medianMs: result.gpuMedianMs, matches: result.gpuMatches }] : []),
        { label: 'CPU · exact', medianMs: result.cpuMedianMs, matches: result.cpuMatches },
        ...result.externals.filter(external => external.ran)
          .map(external => ({ label: external.name, medianMs: external.medianMs, matches: external.matches }))
      ];
      renderQbChart(qbFields.chart, entries);
    }
    if (qbFields.verdict) {
      if (!result.gpuRan) {
        qbFields.verdict.textContent = 'WebGPU unavailable here — CPU-only result shown.';
      } else if (result.gpuMedianMs > 0 && result.cpuMedianMs > 0) {
        const ratio = result.cpuMedianMs / result.gpuMedianMs;
        if (ratio >= 1.1) {
          qbFields.verdict.textContent =
            `WebGPU ${ratio.toFixed(1)}× faster than CPU here.`;
        } else if (ratio <= 1 / 1.1) {
          qbFields.verdict.textContent =
            `CPU ${(1 / ratio).toFixed(1)}× faster than WebGPU here.`;
        } else {
          qbFields.verdict.textContent = 'Too close to call — WebGPU and CPU tied within noise.';
        }
      } else {
        qbFields.verdict.textContent = 'Timed run complete — see medians below.';
      }
    }
    if (qbFields.note) qbFields.note.textContent =
      `CPU route: scoreExactMatches (webgpu-search exact scorer); uFuzzy via CPUEngine, Fuse.js defaults · Corpus: ${result.corpusSize.toLocaleString()} generated records · Query: “${result.query}”. Setup is excluded from warm-search timing.`;
    qbPanel.hidden = false;
    qbStatus.textContent = 'Completed 7 timed searches per available engine.';
  } catch (error) {
    qbStatus.textContent = `Benchmark failed: ${error instanceof Error ? error.message : String(error)}`;
  } finally {
    qbRun.disabled = false;
  }
}

async function runPlayground(): Promise<void> {
  if (!pgData || !pgQuery || !pgRun || !pgStatus || !pgResults) return;
  const lines = pgData.value.split('\n').map(line => line.trim()).filter(Boolean).slice(0, 200);
  const query = pgQuery.value.trim();
  if (lines.length === 0) {
    pgStatus.textContent = 'Add at least one line of text.';
    return;
  }
  if (!query) {
    pgStatus.textContent = 'Enter a query.';
    pgQuery.focus();
    return;
  }
  pgRun.disabled = true;
  pgStatus.textContent = 'Searching your text…';
  try {
    const index = await SearchIndex.create(lines, { preferGpu: false });
    try {
      const response = await index.search(query, { mode: 'fuzzy', limit: 8 });
      pgResults.replaceChildren();
      for (const item of response.results) {
        const row = document.createElement('li');
        row.textContent = `${item.text} · score ${item.score}`;
        pgResults.append(row);
      }
      if (response.results.length === 0) {
        const row = document.createElement('li');
        row.textContent = 'No matches in your text.';
        pgResults.append(row);
      }
      pgStatus.textContent = `${response.totalMatches} match${response.totalMatches === 1 ? '' : 'es'} · CPU, local only.`;
    } finally {
      index.destroy();
    }
  } catch (error) {
    pgStatus.textContent = `Playground failed: ${error instanceof Error ? error.message : String(error)}`;
  } finally {
    pgRun.disabled = false;
  }
}

qbRun?.addEventListener('click', () => { void runHomepageBenchmark(); });
pgRun?.addEventListener('click', () => { void runPlayground(); });

copyButton?.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText('npm install webgpu-search');
    copyButton.textContent = 'Copied';
    window.setTimeout(() => { copyButton.textContent = 'Copy'; }, 1500);
  } catch {
    copyButton.textContent = 'Select command';
  }
});
