import { DocsEngine } from './docs-engine';
import { generateDocsRecords, CORE_DOCS } from './docs-data';
import type { DocPageRecord, DocsSearchResult } from './types';

// DOM Elements
const searchInput = document.getElementById('search-input') as HTMLInputElement;
const resultsList = document.getElementById('results-list') as HTMLUListElement;
const modeSelect = document.getElementById('mode-select') as HTMLSelectElement;
const engineSelect = document.getElementById('engine-select') as HTMLSelectElement;
const workerSelect = document.getElementById('worker-select') as HTMLSelectElement;
const highlightToggle = document.getElementById('highlight-toggle') as HTMLInputElement;
const suggestToggle = document.getElementById('suggest-toggle') as HTMLInputElement;
const sectionSelect = document.getElementById('section-select') as HTMLSelectElement;
const versionSelect = document.getElementById('version-select') as HTMLSelectElement;
const suggestBar = document.getElementById('suggest-bar') as HTMLDivElement;
const suggestList = document.getElementById('suggest-list') as HTMLDivElement;
const facetBar = document.getElementById('facet-bar') as HTMLDivElement;
const facetList = document.getElementById('facet-list') as HTMLDivElement;
const engineBadge = document.getElementById('engine-badge') as HTMLSpanElement;
const idbStatus = document.getElementById('idb-status') as HTMLSpanElement;

// Preview Elements
const previewTitle = document.getElementById('preview-title') as HTMLDivElement;
const previewMeta = document.getElementById('preview-meta') as HTMLDivElement;
const previewContent = document.getElementById('preview-content') as HTMLDivElement;
const previewTags = document.getElementById('preview-tags') as HTMLDivElement;
const previewPath = document.getElementById('preview-path') as HTMLPreElement;

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
const formTitle = document.getElementById('form-title') as HTMLInputElement;
const formSection = document.getElementById('form-section') as HTMLInputElement;
const formTags = document.getElementById('form-tags') as HTMLInputElement;
const formContent = document.getElementById('form-content') as HTMLInputElement;

// Persistence / recovery buttons
const btnSaveIdb = document.getElementById('btn-save-idb') as HTMLButtonElement;
const btnRestoreIdb = document.getElementById('btn-restore-idb') as HTMLButtonElement;
const btnClearIdb = document.getElementById('btn-clear-idb') as HTMLButtonElement;
const btnRebuildGpu = document.getElementById('btn-rebuild-gpu') as HTMLButtonElement;

// Application State
let records = generateDocsRecords(400);
let activeResults: DocsSearchResult[] = [];
let selectedIndex = 0;
let currentAbortController: AbortController | null = null;

const engine = new DocsEngine({
  useWorker: true,
  preferGpu: true
});

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

/** Guarded select assignment: unknown facet values reset to ALL. */
function setSelectGuarded(sel: HTMLSelectElement, value: string): void {
  const exists = Array.from(sel.options).some((o) => o.value === value);
  sel.value = exists ? value : 'ALL';
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

function updatePreview(record: DocPageRecord | null): void {
  if (!record) {
    previewTitle.textContent = 'Select a page to inspect';
    previewMeta.textContent = 'Section and version overview';
    previewContent.textContent = '-';
    previewTags.textContent = '-';
    previewPath.textContent = '// Path will appear here';
    return;
  }

  previewTitle.textContent = record.title;
  previewMeta.textContent = `${record.section} · v${record.version} · ${record.readingMinutes} min read`;
  previewContent.textContent = record.content;
  previewTags.textContent = record.tags;
  previewPath.textContent = record.path;
}

function renderResults(): void {
  resultsList.innerHTML = '';

  if (activeResults.length === 0) {
    const li = document.createElement('li');
    li.className = 'result-item';
    li.style.color = 'var(--text-muted)';
    li.style.textAlign = 'center';
    li.style.padding = '32px 16px';
    li.textContent = searchInput.value.trim() ? 'No matching pages found.' : 'Type to search...';
    resultsList.appendChild(li);
    updatePreview(null);
    return;
  }

  activeResults.forEach((res, idx) => {
    const li = document.createElement('li');
    li.className = `result-item ${idx === selectedIndex ? 'selected' : ''}`;

    const section = res.doc.section;
    const sectionLabel = escapeHtml(section.charAt(0).toUpperCase());

    const highlightedTitle = res.highlightedText?.title ? sanitizeHighlighted(res.highlightedText.title) : escapeHtml(res.doc.title);
    const highlightedPath = res.highlightedText?.path ? sanitizeHighlighted(res.highlightedText.path) : escapeHtml(res.doc.path);
    const highlightedContent = res.highlightedText?.content ? sanitizeHighlighted(res.highlightedText.content) : escapeHtml(res.doc.content);

    li.innerHTML = `
      <div class="item-header">
        <div class="item-left">
          <span class="kind-icon kind-${escapeHtml(section)}">${sectionLabel}</span>
          <span class="item-filename">${highlightedTitle}</span>
        </div>
        <div class="item-right">
          <span class="field-badge">match: ${escapeHtml(res.matchedField)}</span>
          <span class="score-badge">${res.score}</span>
        </div>
      </div>
      <div class="item-path">${highlightedPath}</div>
      <div class="item-symbols">${highlightedContent}</div>
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
  suggestBar.style.display = 'block';
  for (const s of suggestions) {
    const chip = document.createElement('button');
    chip.className = 'suggest-chip';
    chip.textContent = `${s.text} (${s.score})`;
    chip.addEventListener('click', () => {
      searchInput.value = s.text;
      performSearch();
    });
    suggestList.appendChild(chip);
  }
}

function renderFacets(facets: Record<string, any> | undefined): void {
  facetList.innerHTML = '';
  const bySection = facets?.bySection;
  if (!bySection || bySection.type !== 'terms' || !Array.isArray(bySection.buckets) || bySection.buckets.length === 0) {
    facetBar.style.display = 'none';
    return;
  }
  facetBar.style.display = 'block';
  for (const b of bySection.buckets) {
    const chip = document.createElement('button');
    chip.className = 'facet-chip';
    chip.textContent = `${String(b.value)} · ${b.count}`;
    chip.addEventListener('click', () => {
      setSelectGuarded(sectionSelect, String(b.value));
      performSearch();
    });
    facetList.appendChild(chip);
  }
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
  const sectionFilter = sectionSelect.value;
  const versionFilter = versionSelect.value;

  try {
    const searchRes = await engine.search(query, {
      mode,
      highlight,
      limit: 50,
      sectionFilter,
      versionFilter,
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
      previewPath.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
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
sectionSelect.addEventListener('change', () => performSearch());
versionSelect.addEventListener('change', () => performSearch());

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
  formTitle.focus();
});

btnModalCancel.addEventListener('click', () => {
  addModal.style.display = 'none';
});

btnModalSave.addEventListener('click', async () => {
  btnModalSave.disabled = true;
  try {
    const title = formTitle.value.trim() || 'Untitled offline page';
    const section = formSection.value.trim() || 'Guide';
    const tags = formTags.value.trim() || 'offline-docs';
    const content = formContent.value.trim() || 'User created offline documentation page';

    const newDoc: DocPageRecord = {
      id: `custom-${Date.now()}`,
      path: `docs/custom/${title.toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 48)}.md`,
      title,
      section: section as DocPageRecord['section'],
      content,
      tags,
      version: '1.0',
      readingMinutes: 4
    };

    await engine.addRecord(newDoc);
    addModal.style.display = 'none';
    formTitle.value = '';
    formSection.value = '';
    formTags.value = '';
    formContent.value = '';

    await updateTelemetry();
    await performSearch();
  } finally {
    btnModalSave.disabled = false;
  }
});

btnBatchAdd.addEventListener('click', async () => {
  const newRecords = generateDocsRecords(200).map((r, i) => ({
    ...r,
    id: `batch-${Date.now()}-${i}`
  }));

  await engine.batchAdd(newRecords);
  await updateTelemetry();
  await performSearch();
});

btnReset.addEventListener('click', async () => {
  records = [...CORE_DOCS];
  await engine.init(records);
  await updateTelemetry();
  await performSearch();
});

// Offline bundle persistence + GPU recovery
btnSaveIdb.addEventListener('click', async () => {
  btnSaveIdb.disabled = true;
  try {
    const res = await engine.saveSnapshotToIDB();
    idbStatus.textContent = `Saved ${(res.byteLength / 1024).toFixed(1)} KB in ${res.durationMs.toFixed(1)}ms`;
  } catch (err) {
    console.error('[Save IDB Error]', err);
    idbStatus.textContent = 'Save failed — see console';
  } finally {
    btnSaveIdb.disabled = false;
  }
});

btnRestoreIdb.addEventListener('click', async () => {
  btnRestoreIdb.disabled = true;
  try {
    const res = await engine.restoreSnapshotFromIDB();
    idbStatus.textContent = `Restored ${res.recordCount.toLocaleString()} pages in ${res.durationMs.toFixed(1)}ms`;
    await updateTelemetry();
    await performSearch();
  } catch (err) {
    console.error('[Restore IDB Error]', err);
    idbStatus.textContent = 'Restore failed — see console';
  } finally {
    btnRestoreIdb.disabled = false;
  }
});

btnClearIdb.addEventListener('click', async () => {
  await engine.clearIDB();
  idbStatus.textContent = 'Snapshot cleared';
});

btnRebuildGpu.addEventListener('click', async () => {
  btnRebuildGpu.disabled = true;
  try {
    const rebuilt = await engine.rebuildGpu();
    idbStatus.textContent = rebuilt ? 'GPU pipeline rebuilt' : 'Rebuild skipped (worker path restores via snapshot)';
    await updateTelemetry();
    await performSearch();
  } catch (err) {
    console.error('[Rebuild GPU Error]', err);
    idbStatus.textContent = 'Rebuild failed — see console';
  } finally {
    btnRebuildGpu.disabled = false;
  }
});

// Teardown: release GPU buffers + terminate the worker on navigation.
window.addEventListener('pagehide', () => {
  if (searchDebounceTimer) clearTimeout(searchDebounceTimer);
  if (currentAbortController) currentAbortController.abort();
  engine.destroy();
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
