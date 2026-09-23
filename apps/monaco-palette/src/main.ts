import { CPUEngine } from 'webgpu-search';
import Fuse from 'fuse.js';
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
const suggestToggle = document.getElementById('suggest-toggle') as HTMLInputElement;
const typeSelect = document.getElementById('type-select') as HTMLSelectElement;
const langSelect = document.getElementById('lang-select') as HTMLSelectElement;
const suggestBar = document.getElementById('suggest-bar') as HTMLDivElement;
const suggestList = document.getElementById('suggest-list') as HTMLDivElement;
const facetBar = document.getElementById('facet-bar') as HTMLDivElement;
const facetList = document.getElementById('facet-list') as HTMLDivElement;
const engineBadge = document.getElementById('engine-badge') as HTMLSpanElement;
const engineHint = document.getElementById('engine-hint') as HTMLParagraphElement | null;

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

type EngineChoice = 'webgpu' | 'cpu' | 'ufuzzy' | 'fuse' | 'native';

const engine = new PaletteEngine({
  useWorker: true,
  preferGpu: true
});
const cpuEngine = new CPUEngine();

function currentEngine(): EngineChoice {
  const value = engineSelect.value;
  if (value === 'cpu' || value === 'ufuzzy' || value === 'fuse' || value === 'native') return value;
  return 'webgpu';
}

function isCompetitor(choice: EngineChoice): boolean {
  return choice === 'ufuzzy' || choice === 'fuse' || choice === 'native';
}

function competitorLabel(choice: EngineChoice): string {
  if (choice === 'ufuzzy') return 'uFuzzy — popular library';
  if (choice === 'fuse') return 'Fuse.js — popular library';
  if (choice === 'native') return 'Simple scan — slowest';
  return '';
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  }[c] || c));
}

/**
 * Defense-in-depth sanitizer for `highlightedText` inserted via `innerHTML`.
 * Re-escapes everything except `<mark>` tags.
 */
function sanitizeHighlighted(html: string): string {
  return escapeHtml(html)
    .replace(/&lt;mark&gt;/g, '<mark>')
    .replace(/&lt;\/mark&gt;/g, '</mark>');
}

/** Plain-English field names for "Found in …". */
function friendlyField(raw: string): string {
  const key = raw.trim().toLowerCase();
  if (key === 'filename') return 'File name';
  if (key === 'symbols') return 'Exported names';
  if (key === 'path') return 'File location';
  if (key === 'description') return 'Summary';
  if (key === 'type') return 'File type';
  if (key === 'language') return 'Language';
  if (key === 'best match') return 'Best match';
  return raw || 'File';
}

/** Guarded select assignment: unknown facet values reset to ALL. */
function setSelectGuarded(sel: HTMLSelectElement, value: string): void {
  const exists = Array.from(sel.options).some((o) => o.value === value);
  sel.value = exists ? value : 'ALL';
}

function setBadgeForCompetitor(choice: EngineChoice): void {
  engineBadge.textContent = competitorLabel(choice);
  engineBadge.className = 'badge badge-cpu';
}

async function updateTelemetry(): Promise<void> {
  const choice = currentEngine();
  if (isCompetitor(choice)) {
    const stats = await engine.getStats().catch(() => null);
    if (stats) {
      statRecords.textContent = stats.docCount.toLocaleString();
      statRows.textContent = stats.rowCount.toLocaleString();
      statVram.textContent = formatBytes(stats.memory.vramBytes);
      statRam.textContent = formatBytes(stats.memory.ramBytes);
      statEpoch.textContent = String(stats.mutationEpoch);
    }
    setBadgeForCompetitor(choice);
    return;
  }

  const stats = await engine.getStats();
  if (!stats) return;

  statRecords.textContent = stats.docCount.toLocaleString();
  statRows.textContent = stats.rowCount.toLocaleString();
  statVram.textContent = formatBytes(stats.memory.vramBytes);
  statRam.textContent = formatBytes(stats.memory.ramBytes);
  statEpoch.textContent = String(stats.mutationEpoch);

  if (stats.engine === 'webgpu') {
    engineBadge.textContent = 'Fast mode on';
    engineBadge.className = 'badge badge-gpu';
  } else {
    engineBadge.textContent = 'Standard mode — same results';
    engineBadge.className = 'badge badge-cpu';
  }
}

function updateEngineHint(): void {
  if (!engineHint) return;
  const choice = currentEngine();
  if (choice === 'ufuzzy' || choice === 'fuse') {
    engineHint.textContent = "Popular libraries use their own matching — “How to match” is ignored and scores can’t be compared with ours.";
  } else if (choice === 'native') {
    engineHint.textContent = "Simple scan looks for exact words only — it is the slowest option and ignores “How to match”.";
  } else {
    engineHint.textContent = "Our two modes return identical results. Popular libraries rank differently, so scores can’t be compared directly.";
  }
  workerSelect.disabled = isCompetitor(choice);
  workerSelect.title = isCompetitor(choice)
    ? 'Background mode only applies to our search, not popular libraries.'
    : '';
}

function updatePreview(record: MonacoFileRecord | null): void {
  if (!record) {
    previewTitle.textContent = 'Click a file to read more';
    previewMeta.textContent = 'File, language and size appear here';
    previewDesc.textContent = '-';
    previewSymbols.textContent = '-';
    previewCode.textContent = '// Code preview appears here';
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
    li.style.color = 'var(--muted)';
    li.style.textAlign = 'center';
    li.style.padding = '32px 16px';
    li.textContent = searchInput.value.trim() ? 'No files match. Try fewer or different words.' : 'Type above to search the files.';
    resultsList.appendChild(li);
    updatePreview(null);
    return;
  }

  activeResults.forEach((res, idx) => {
    const li = document.createElement('li');
    li.className = `result-item ${idx === selectedIndex ? 'selected' : ''}`;

    const kind = res.doc.type;
    const kindLabel = escapeHtml(kind.charAt(0).toUpperCase());

    const highlightedFilename = res.highlightedText?.filename ? sanitizeHighlighted(res.highlightedText.filename) : escapeHtml(res.doc.filename);
    const highlightedPath = res.highlightedText?.path ? sanitizeHighlighted(res.highlightedText.path) : escapeHtml(res.doc.path);
    const highlightedSymbols = res.highlightedText?.symbols ? sanitizeHighlighted(res.highlightedText.symbols) : escapeHtml(res.doc.symbols);

    li.innerHTML = `
      <div class="item-header">
        <div class="item-left">
          <span class="kind-icon kind-${escapeHtml(kind)}">${kindLabel}</span>
          <span class="item-filename">${highlightedFilename}</span>
        </div>
        <div class="item-right">
          <span class="field-badge">Found in ${escapeHtml(friendlyField(res.matchedField))}</span>
          <span class="score-badge">Relevance ${res.score}</span>
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

function renderSuggestions(suggestions: Array<{ text: string; score: number }>): void {
  suggestList.innerHTML = '';
  if (suggestions.length === 0) {
    suggestBar.style.display = 'none';
    return;
  }
  suggestBar.style.display = 'flex';
  for (const s of suggestions) {
    const chip = document.createElement('button');
    chip.className = 'suggest-chip';
    chip.type = 'button';
    chip.textContent = s.text;
    chip.title = 'Click to search this suggestion';
    chip.addEventListener('click', () => {
      searchInput.value = s.text;
      performSearch();
    });
    suggestList.appendChild(chip);
  }
}

function renderFacets(facets: Record<string, any> | undefined): void {
  facetList.innerHTML = '';
  const byType = facets?.byType;
  if (!byType || byType.type !== 'terms' || !Array.isArray(byType.buckets) || byType.buckets.length === 0) {
    facetBar.style.display = 'none';
    return;
  }
  facetBar.style.display = 'flex';
  for (const b of byType.buckets) {
    const chip = document.createElement('button');
    chip.className = 'facet-chip';
    chip.type = 'button';
    chip.textContent = `${String(b.value)} · ${Number(b.count).toLocaleString()} files`;
    chip.title = `Show only ${String(b.value)} files`;
    chip.addEventListener('click', () => {
      setSelectGuarded(typeSelect, String(b.value));
      performSearch();
    });
    facetList.appendChild(chip);
  }
}

function getFilteredRecords(typeFilter: string, languageFilter: string): MonacoFileRecord[] {
  return engine.getRecords().filter((r) =>
    (typeFilter === 'ALL' || r.type === typeFilter) &&
    (languageFilter === 'ALL' || r.language === languageFilter)
  );
}

function facetsForRecords(docs: MonacoFileRecord[]): Record<string, any> | undefined {
  const counts = new Map<string, number>();
  for (const d of docs) counts.set(d.type, (counts.get(d.type) ?? 0) + 1);
  const buckets = [...counts.entries()]
    .map(([value, count]) => ({ value, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 10);
  if (buckets.length === 0) return undefined;
  return { byType: { type: 'terms', buckets } };
}

/** Popular-library search over the same files. Scores use each library's own ranking. */
async function searchCompetitor(
  query: string,
  choice: EngineChoice,
  typeFilter: string,
  languageFilter: string,
  limit: number
): Promise<{ results: MonacoPaletteSearchResult[]; totalMatches: number; durationMs: number; facets: Record<string, any> | undefined }> {
  const filtered = getFilteredRecords(typeFilter, languageFilter);
  const clean = query.trim();
  if (!clean) {
    const docs = filtered.slice(0, limit);
    return {
      results: docs.map((doc, i) => ({ id: doc.id, score: Math.max(1, 50 - i), matchedField: 'Best match', doc })),
      totalMatches: filtered.length,
      durationMs: 0,
      facets: facetsForRecords(filtered)
    };
  }

  if (choice === 'ufuzzy') {
    const strings = filtered.map((r) => `${r.filename} ${r.symbols} ${r.path} ${r.description}`);
    const t0 = performance.now();
    const out = cpuEngine.searchWithUFuzzy(strings, clean, limit);
    const durationMs = performance.now() - t0;
    const results: MonacoPaletteSearchResult[] = out.results
      .map((item) => filtered[item.index])
      .filter((doc): doc is MonacoFileRecord => Boolean(doc))
      .map((doc) => ({
        id: doc.id,
        score: 0,
        matchedField: 'Best match',
        doc
      }));
    // Reuse uFuzzy's own order; keep its rank scores for display honesty.
    out.results.forEach((item, i) => {
      if (results[i]) results[i].score = item.score;
    });
    const matchedDocs = out.results
      .map((item) => filtered[item.index])
      .filter((doc): doc is MonacoFileRecord => Boolean(doc));
    return { results, totalMatches: out.totalMatches, durationMs, facets: facetsForRecords(matchedDocs) };
  }

  if (choice === 'fuse') {
    const fuse = new Fuse(filtered, {
      keys: [
        { name: 'filename', weight: 2 },
        { name: 'symbols', weight: 1.5 },
        { name: 'path', weight: 1 }
      ],
      threshold: 0.4,
      ignoreLocation: true,
      includeScore: true
    });
    const t0 = performance.now();
    const found = fuse.search(clean);
    const durationMs = performance.now() - t0;
    const results: MonacoPaletteSearchResult[] = found.slice(0, limit).map((hit, i) => {
      const relevance = hit.score === undefined
        ? Math.max(1, 90 - i)
        : Math.max(1, Math.round((1 - Math.min(1, hit.score)) * 100));
      return {
        id: hit.item.id,
        score: relevance,
        matchedField: 'Best match',
        doc: hit.item
      };
    });
    return {
      results,
      totalMatches: found.length,
      durationMs,
      facets: facetsForRecords(found.map((hit) => hit.item))
    };
  }

  // native: simple exact-word scan for reference
  const t0 = performance.now();
  const q = clean.toLowerCase();
  const matched = filtered.filter((r) =>
    `${r.filename} ${r.symbols} ${r.path} ${r.description}`.toLowerCase().includes(q)
  );
  const durationMs = performance.now() - t0;
  const results: MonacoPaletteSearchResult[] = matched.slice(0, limit).map((doc, i) => {
    const hayFilename = doc.filename.toLowerCase();
    const haySymbols = doc.symbols.toLowerCase();
    const hayPath = doc.path.toLowerCase();
    const field = hayFilename.includes(q) ? 'File name' : haySymbols.includes(q) ? 'Exported names' : hayPath.includes(q) ? 'File location' : 'Summary';
    return { id: doc.id, score: Math.max(1, 80 - i), matchedField: field, doc };
  });
  return { results, totalMatches: matched.length, durationMs, facets: facetsForRecords(matched) };
}

async function performSearch(): Promise<void> {
  const query = searchInput.value.trim();

  if (currentAbortController) {
    currentAbortController.abort();
  }
  currentAbortController = new AbortController();

  const mode = modeSelect.value as 'fuzzy' | 'substring' | 'prefix' | 'token';
  const highlight = highlightToggle.checked;
  const withSuggest = suggestToggle.checked;
  const typeFilter = typeSelect.value;
  const languageFilter = langSelect.value;
  const choice = currentEngine();
  updateEngineHint();

  if (isCompetitor(choice)) {
    setBadgeForCompetitor(choice);
    try {
      const res = await searchCompetitor(query, choice, typeFilter, languageFilter, 50);
      activeResults = res.results;
      selectedIndex = 0;
      statMatches.textContent = res.totalMatches.toLocaleString();
      statLatency.textContent = `${res.durationMs.toFixed(2)} ms`;
      renderSuggestions([]);
      renderFacets(res.facets);
      renderResults();
    } catch (err: any) {
      if (err?.name === 'AbortError') return;
      console.error('[Search Error]', err);
    }
    return;
  }

  try {
    const searchRes = await engine.search(query, {
      mode,
      highlight,
      limit: 50,
      typeFilter,
      languageFilter,
      signal: currentAbortController.signal,
      ...(withSuggest && query ? { autocomplete: { mode: 'prefix', limit: 5 } } : {})
    });

    activeResults = searchRes.results;
    selectedIndex = 0;
    statMatches.textContent = searchRes.totalMatches.toLocaleString();
    statLatency.textContent = `${searchRes.searchDurationMs.toFixed(2)} ms`;

    renderSuggestions(searchRes.suggestions ?? []);
    renderFacets(searchRes.facets);
    renderResults();
    await updateTelemetry();
  } catch (err: any) {
    if (err?.name === 'AbortError') return;
    console.error('[Search Error]', err);
  }
}

// Event Listeners
let searchDebounceTimer: ReturnType<typeof setTimeout> | null = null;
searchInput.addEventListener('input', () => {
  if (searchDebounceTimer) clearTimeout(searchDebounceTimer);
  searchDebounceTimer = setTimeout(() => {
    performSearch();
  }, 60);
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
  } else if (e.key === 'Enter') {
    e.preventDefault();
    const selected = activeResults[selectedIndex]?.doc;
    if (selected) {
      updatePreview(selected);
      previewCode.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  } else if (e.key === 'Escape') {
    e.preventDefault();
    searchInput.value = '';
    performSearch();
  }
});

modeSelect.addEventListener('change', () => performSearch());
highlightToggle.addEventListener('change', () => performSearch());
suggestToggle.addEventListener('change', () => performSearch());
typeSelect.addEventListener('change', () => performSearch());
langSelect.addEventListener('change', () => performSearch());

engineSelect.addEventListener('change', async () => {
  const choice = currentEngine();
  updateEngineHint();
  if (choice === 'webgpu' || choice === 'cpu') {
    await engine.setPreferGpu(choice === 'webgpu');
    await updateTelemetry();
  } else {
    setBadgeForCompetitor(choice);
    await updateTelemetry();
  }
  await performSearch();
});

workerSelect.addEventListener('change', async () => {
  const useWorker = workerSelect.value === 'worker';
  await engine.setUseWorker(useWorker);
  await updateTelemetry();
  await performSearch();
});

// Modal Actions
function closeModal(): void {
  addModal.style.display = 'none';
}
btnAddModal.addEventListener('click', () => {
  addModal.style.display = 'flex';
  formFilename.focus();
});

btnModalCancel.addEventListener('click', closeModal);
addModal.addEventListener('click', (e) => {
  if (e.target === addModal) closeModal();
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && addModal.style.display === 'flex') closeModal();
});

btnModalSave.addEventListener('click', async () => {
  btnModalSave.disabled = true;
  try {
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
    closeModal();
    formFilename.value = '';
    formPath.value = '';
    formSymbols.value = '';
    formDesc.value = '';

    await updateTelemetry();
    await performSearch();
  } finally {
    btnModalSave.disabled = false;
  }
});

btnBatchAdd.addEventListener('click', async () => {
  btnBatchAdd.disabled = true;
  try {
    const newRecords = generateMonacoRecords(200).map((r, i) => ({
      ...r,
      id: `batch-${Date.now()}-${i}`
    }));

    await engine.batchAdd(newRecords);
    await updateTelemetry();
    await performSearch();
  } finally {
    btnBatchAdd.disabled = false;
  }
});

btnReset.addEventListener('click', async () => {
  records = [...CORE_FILES];
  await engine.init(records);
  await updateTelemetry();
  await performSearch();
});

// Initialize
async function bootstrap(): Promise<void> {
  updateEngineHint();
  await engine.init(records);
  await updateTelemetry();
  await performSearch();
}

// Teardown: release GPU buffers + terminate the worker on navigation.
window.addEventListener('pagehide', () => {
  if (searchDebounceTimer) clearTimeout(searchDebounceTimer);
  if (currentAbortController) currentAbortController.abort();
  engine.destroy();
});

bootstrap().catch((err) => {
  console.error('[Bootstrap Error]', err);
});
