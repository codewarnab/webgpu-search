import { SearchIndex } from 'webgpu-search';
import './style.css';
import { formatMs, runQuickCompare } from './quick-benchmark';

const demoQuery = document.querySelector<HTMLInputElement>('#demo-query');
const demoStatus = document.querySelector<HTMLElement>('#demo-status');
const demoResults = document.querySelector<HTMLUListElement>('#demo-results');
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
  note: document.querySelector<HTMLElement>('#qb-note')
};

const pgData = document.querySelector<HTMLTextAreaElement>('#pg-data');
const pgQuery = document.querySelector<HTMLInputElement>('#pg-query');
const pgRun = document.querySelector<HTMLButtonElement>('#pg-run');
const pgStatus = document.querySelector<HTMLElement>('#pg-status');
const pgResults = document.querySelector<HTMLOListElement>('#pg-results');

const sampleFiles = [
  'src/auth/AuthController.ts', 'src/auth/AuthSessionService.ts', 'src/auth/authorizeRequest.ts',
  'src/auth/authentication.test.ts', 'src/workers/SearchWorkerClient.ts', 'src/search/SearchIndex.ts',
  'src/search/DocumentIndex.ts', 'src/search/AutocompleteProvider.ts', 'src/profiles/UserProfileView.tsx',
  'src/profiles/ProfileSearchService.ts', 'src/billing/InvoiceSearchIndex.ts', 'src/billing/PaymentController.ts',
  'packages/engine/src/webgpu-engine.ts', 'packages/engine/src/cpu-engine.ts',
  'packages/engine/src/context-manager.ts', 'docs/search/worker-guide.md',
  'docs/search/unicode-normalization.md', 'examples/command-palette/search.ts'
];

let searchIndex: SearchIndex | null = null;
let querySequence = 0;

function showEmptyState(message: string): void {
  if (!demoResults) return;
  const row = document.createElement('li');
  row.className = 'demo-empty';
  row.textContent = message;
  demoResults.replaceChildren(row);
}

function renderRows(items: Array<{ text: string; score: number }>): void {
  if (!demoResults) return;
  if (items.length === 0) {
    showEmptyState('No sample results. Try “auth”, “worker”, or “profile”.');
    return;
  }
  demoResults.replaceChildren(...items.slice(0, 4).map(item => {
    const row = document.createElement('li');
    row.className = 'demo-result';
    const content = document.createElement('div');
    content.className = 'demo-result-text';
    const name = document.createElement('div');
    name.className = 'demo-result-name';
    name.textContent = item.text.split('/').at(-1) ?? item.text;
    const path = document.createElement('div');
    path.className = 'demo-result-path';
    path.textContent = item.text;
    const score = document.createElement('span');
    score.className = 'demo-result-score';
    score.textContent = `score ${item.score}`;
    content.append(name, path);
    row.append(content, score);
    return row;
  }));
}

async function runDemoSearch(): Promise<void> {
  if (!demoQuery || !demoStatus || !searchIndex) return;
  const query = demoQuery.value.trim();
  const sequence = ++querySequence;
  if (!query) {
    demoStatus.textContent = 'Type to search the local sample.';
    showEmptyState('Search runs locally in your browser.');
    return;
  }
  demoStatus.textContent = 'Searching local sample…';
  try {
    const response = await searchIndex.search(query, { mode: 'fuzzy', limit: 4 });
    if (sequence !== querySequence) return;
    demoStatus.textContent = `${response.totalMatches.toLocaleString()} sample matches · ${response.timings.totalMs.toFixed(2)} ms`;
    renderRows(response.results.map(({ text, score }) => ({ text, score })));
  } catch {
    if (sequence !== querySequence) return;
    demoStatus.textContent = 'Search unavailable';
    showEmptyState('The local demo could not complete this query.');
  }
}

async function initializeDemo(): Promise<void> {
  if (!demoQuery || !demoStatus || !demoResults) return;
  try {
    searchIndex = await SearchIndex.create(sampleFiles, { preferGpu: false });
    demoQuery.addEventListener('input', () => { void runDemoSearch(); });
    await runDemoSearch();
  } catch {
    demoStatus.textContent = 'Demo could not start';
    showEmptyState('Try the full benchmark to check your browser setup.');
  }
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
    if (qbFields.gpu) qbFields.gpu.textContent = result.gpuRan ? formatMs(result.gpuMedianMs) : 'Not available';
    if (qbFields.gpuDetail) qbFields.gpuDetail.textContent = result.gpuRan
      ? `p95 ${formatMs(result.gpuP95Ms)} · ${result.gpuMatches.toLocaleString()} matches`
      : 'WebGPU unavailable; no GPU timing recorded';
    if (qbFields.cpu) qbFields.cpu.textContent = formatMs(result.cpuMedianMs);
    if (qbFields.cpuDetail) qbFields.cpuDetail.textContent =
      `p95 ${formatMs(result.cpuP95Ms)} · ${result.cpuMatches.toLocaleString()} matches`;
    if (qbFields.agree) qbFields.agree.textContent = result.gpuRan ? (result.agree ? 'Match' : 'Different') : 'CPU only';
    if (qbFields.agreeDetail) qbFields.agreeDetail.textContent = result.gpuRan
      ? `${result.gpuMatches.toLocaleString()} WebGPU · ${result.cpuMatches.toLocaleString()} CPU`
      : 'WebGPU was not available in this browser';
    if (qbFields.device) qbFields.device.textContent = result.deviceLabel;
    if (qbFields.note) qbFields.note.textContent =
      `Corpus: ${result.corpusSize.toLocaleString()} generated records · Query: “${result.query}”. Setup is excluded from warm-search timing.`;
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

void initializeDemo();
