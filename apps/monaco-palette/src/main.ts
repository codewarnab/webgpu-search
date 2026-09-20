import { PaletteEngine } from './palette-engine';
import { generateMonacoRecords, CORE_FILES } from './sample-data';
import type { MonacoFileRecord, MonacoPaletteSearchResult } from './types';

// DOM Elements
const searchInput = document.getElementById('search-input') as HTMLInputElement;
const resultsList = document.getElementById('results-list') as HTMLUListElement;
const modeSelect = document.getElementById('mode-select') as HTMLSelectElement;
const engineSelect = document.getElementById('engine-select') as HTMLSelectElement;
const workerSelect = document.getElementById('worker-select') as HTMLSelectElement;
const highlightToggle = document.getElementById('highlight-toggle') as HTMLInputElement;
const engineBadge = document.getElementById('engine-badge') as HTMLSpanElement;

// Preview Elements
const previewTitle = document.getElementById('preview-title') as HTMLDivElement;
const previewMeta = document.getElementById('preview-meta') as HTMLDivElement;
const previewDesc = document.getElementById('preview-desc') as HTMLDivElement;
const previewSymbols = document.getElementById('preview-symbols') as HTMLDivElement;
const previewCode = document.getElementById('preview-code') as HTMLPreElement;

// Telemetry Elements
const statRecords = document.getElementById('stat-records') as HTMLSpanElement;
const statRows = document.getElementById('stat-rows') as HTMLSpanElement;
const statMatches = document.getElementById('stat-matches') as HTMLSpanElement;
const statLatency = document.getElementById('stat-latency') as HTMLSpanElement;
const statVram = document.getElementById('stat-vram') as HTMLSpanElement;
const statRam = document.getElementById('stat-ram') as HTMLSpanElement;
const statEpoch = document.getElementById('stat-epoch') as HTMLSpanElement;

// Modal Elements
const addModal = document.getElementById('add-modal') as HTMLDivElement;
const btnAddModal = document.getElementById('btn-add-modal') as HTMLButtonElement;
const btnBatchAdd = document.getElementById('btn-batch-add') as HTMLButtonElement;
const btnReset = document.getElementById('btn-reset') as HTMLButtonElement;
const btnModalCancel = document.getElementById('btn-modal-cancel') as HTMLButtonElement;
const btnModalSave = document.getElementById('btn-modal-save') as HTMLButtonElement;
const formFilename = document.getElementById('form-filename') as HTMLInputElement;
const formPath = document.getElementById('form-path') as HTMLInputElement;
const formSymbols = document.getElementById('form-symbols') as HTMLInputElement;
const formDesc = document.getElementById('form-desc') as HTMLInputElement;

// Application State
let records = generateMonacoRecords(600);
let activeResults: MonacoPaletteSearchResult[] = [];
let selectedIndex = 0;
let currentAbortController: AbortController | null = null;

const engine = new PaletteEngine({
  useWorker: true,
  preferGpu: true
});

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

async function updateTelemetry(): Promise<void> {
  const stats = await engine.getStats();
  if (!stats) return;

  statRecords.textContent = stats.docCount.toLocaleString();
  statRows.textContent = stats.rowCount.toLocaleString();
  statVram.textContent = formatBytes(stats.memory.vramBytes);
  statRam.textContent = formatBytes(stats.memory.ramBytes);
  statEpoch.textContent = String(stats.mutationEpoch);

  if (stats.engine === 'webgpu') {
    engineBadge.textContent = 'WebGPU Active';
    engineBadge.className = 'badge badge-gpu';
  } else {
    engineBadge.textContent = `CPU Fallback (${stats.fallbackReason || 'prefer-cpu'})`;
    engineBadge.className = 'badge badge-cpu';
  }
}

function updatePreview(record: MonacoFileRecord | null): void {
  if (!record) {
    previewTitle.textContent = 'Select a file to inspect';
    previewMeta.textContent = 'Path and symbols overview';
    previewDesc.textContent = '-';
    previewSymbols.textContent = '-';
    previewCode.textContent = '// Select a file from the list';
    return;
  }

  previewTitle.textContent = record.filename;
  previewMeta.textContent = `${record.path} · ${record.language.toUpperCase()} · ${record.lineCount} lines (${formatBytes(record.sizeBytes)})`;
  previewDesc.textContent = record.description;
  previewSymbols.textContent = record.symbols;

  previewCode.textContent = `// ${record.filename}
// ${record.description}

export interface ${record.filename.replace(/[^a-zA-Z0-9]/g, '_')}_Config {
  readonly id: string;
  readonly enabled: boolean;
}

export class ${record.symbols.split(',')[0]?.trim() || 'Component'} {
  constructor(private readonly config: ${record.filename.replace(/[^a-zA-Z0-9]/g, '_')}_Config) {}

  public async execute(): Promise<void> {
    console.log("Executing in ${record.path}");
  }
}`;
}

function renderResults(): void {
  resultsList.innerHTML = '';

  if (activeResults.length === 0) {
    const li = document.createElement('li');
    li.className = 'result-item';
    li.style.color = 'var(--text-muted)';
    li.style.textAlign = 'center';
    li.style.padding = '32px 16px';
    li.textContent = searchInput.value.trim() ? 'No matching files or symbols found.' : 'Type to search...';
    resultsList.appendChild(li);
    updatePreview(null);
    return;
  }

  activeResults.forEach((res, idx) => {
    const li = document.createElement('li');
    li.className = `result-item ${idx === selectedIndex ? 'selected' : ''}`;

    const kind = res.doc.type;
    const kindLabel = kind.charAt(0).toUpperCase();

    const highlightedFilename = res.highlightedText?.filename || res.doc.filename;
    const highlightedPath = res.highlightedText?.path || res.doc.path;
    const highlightedSymbols = res.highlightedText?.symbols || res.doc.symbols;

    li.innerHTML = `
      <div class="item-header">
        <div class="item-left">
          <span class="kind-icon kind-${kind}">${kindLabel}</span>
          <span class="item-filename">${highlightedFilename}</span>
        </div>
        <div class="item-right">
          <span class="field-badge">match: ${res.matchedField}</span>
          <span class="score-badge">${res.score}</span>
        </div>
      </div>
      <div class="item-path">${highlightedPath}</div>
      <div class="item-symbols">${highlightedSymbols}</div>
    `;

    li.addEventListener('click', () => {
      selectedIndex = idx;
      renderSelection();
    });

    resultsList.appendChild(li);
  });

  renderSelection();
}

function renderSelection(): void {
  const items = resultsList.querySelectorAll('.result-item');
  items.forEach((item, idx) => {
    if (idx === selectedIndex) {
      item.classList.add('selected');
      item.scrollIntoView({ block: 'nearest' });
    } else {
      item.classList.remove('selected');
    }
  });

  const selected = activeResults[selectedIndex]?.doc || null;
  updatePreview(selected);
}

async function performSearch(): Promise<void> {
  const query = searchInput.value.trim();

  if (currentAbortController) {
    currentAbortController.abort();
  }
  currentAbortController = new AbortController();

  const mode = modeSelect.value as 'fuzzy' | 'substring';
  const highlight = highlightToggle.checked;

  try {
    const searchRes = await engine.search(query, {
      mode,
      highlight,
      limit: 50,
      signal: currentAbortController.signal
    });

    activeResults = searchRes.results;
    selectedIndex = 0;
    statMatches.textContent = searchRes.totalMatches.toLocaleString();
    statLatency.textContent = `${searchRes.searchDurationMs.toFixed(2)} ms`;

    renderResults();
  } catch (err: any) {
    if (err?.name === 'AbortError') return;
    console.error('[Search Error]', err);
  }
}

// Event Listeners
searchInput.addEventListener('input', () => {
  performSearch();
});

searchInput.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowDown') {
    e.preventDefault();
    if (activeResults.length > 0) {
      selectedIndex = (selectedIndex + 1) % activeResults.length;
      renderSelection();
    }
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    if (activeResults.length > 0) {
      selectedIndex = (selectedIndex - 1 + activeResults.length) % activeResults.length;
      renderSelection();
    }
  } else if (e.key === 'Escape') {
    e.preventDefault();
    searchInput.value = '';
    performSearch();
  }
});

modeSelect.addEventListener('change', () => performSearch());
highlightToggle.addEventListener('change', () => performSearch());

engineSelect.addEventListener('change', async () => {
  const preferGpu = engineSelect.value === 'webgpu';
  await engine.setPreferGpu(preferGpu);
  await updateTelemetry();
  await performSearch();
});

workerSelect.addEventListener('change', async () => {
  const useWorker = workerSelect.value === 'worker';
  await engine.setUseWorker(useWorker);
  await updateTelemetry();
  await performSearch();
});

// Modal Actions
btnAddModal.addEventListener('click', () => {
  addModal.style.display = 'flex';
  formFilename.focus();
});

btnModalCancel.addEventListener('click', () => {
  addModal.style.display = 'none';
});

btnModalSave.addEventListener('click', async () => {
  const filename = formFilename.value.trim() || 'new_symbol.ts';
  const path = formPath.value.trim() || `src/custom/${filename}`;
  const symbols = formSymbols.value.trim() || 'CustomSymbol, execute';
  const desc = formDesc.value.trim() || 'User created symbol record';

  const newDoc: MonacoFileRecord = {
    id: `custom-${Date.now()}`,
    filename,
    path,
    symbols,
    type: 'class',
    language: 'typescript',
    description: desc,
    sizeBytes: 2048,
    lineCount: 85
  };

  await engine.addRecord(newDoc);
  addModal.style.display = 'none';
  formFilename.value = '';
  formPath.value = '';
  formSymbols.value = '';
  formDesc.value = '';

  await updateTelemetry();
  await performSearch();
});

btnBatchAdd.addEventListener('click', async () => {
  const newRecords = generateMonacoRecords(200).map((r, i) => ({
    ...r,
    id: `batch-${Date.now()}-${i}`
  }));

  await engine.batchAdd(newRecords);
  await updateTelemetry();
  await performSearch();
});

btnReset.addEventListener('click', async () => {
  records = [...CORE_FILES];
  await engine.init(records);
  await updateTelemetry();
  await performSearch();
});

// Initialize
async function bootstrap(): Promise<void> {
  await engine.init(records);
  await updateTelemetry();
  await performSearch();
}

bootstrap().catch((err) => {
  console.error('[Bootstrap Error]', err);
});
