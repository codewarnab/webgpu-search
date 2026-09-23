import { SearchIndex } from 'webgpu-search';
import './style.css';

const demoQuery = document.querySelector<HTMLInputElement>('#demo-query');
const demoStatus = document.querySelector<HTMLElement>('#demo-status');
const demoResults = document.querySelector<HTMLUListElement>('#demo-results');
const copyButton = document.querySelector<HTMLButtonElement>('#copy-install');

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
    showEmptyState('Try the full comparison page to check your browser setup.');
  }
}

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
