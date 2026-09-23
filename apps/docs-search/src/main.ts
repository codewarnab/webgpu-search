import { CPUEngine } from 'webgpu-search';
import Fuse from 'fuse.js';
import { DocsEngine } from './docs-engine';
import { generateDocsRecords, CORE_DOCS, docBody } from './docs-data';
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
const engineBadge = document.getElementById('engine-badge') as HTMLSpanElement | null;
const engineHint = document.getElementById('engine-hint') as HTMLParagraphElement | null;
const idbStatus = document.getElementById('idb-status') as HTMLSpanElement;

// Preview Elements
const previewTitle = document.getElementById('preview-title') as HTMLHeadingElement;
const previewMeta = document.getElementById('preview-meta') as HTMLDivElement;
const previewMetaTop = document.getElementById('preview-meta-top') as HTMLSpanElement | null;
const previewContent = document.getElementById('preview-content') as HTMLDivElement;
const previewTags = document.getElementById('preview-tags') as HTMLDivElement;
const previewPath = document.getElementById('preview-path') as HTMLPreElement;
const statMatchesLine = document.getElementById('stat-matches-line') as HTMLSpanElement | null;
const statLatencyLine = document.getElementById('stat-latency-line') as HTMLSpanElement | null;

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

type EngineChoice = 'webgpu' | 'cpu' | 'ufuzzy' | 'fuse' | 'native';

const engine = new DocsEngine({
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
  if (key === 'title') return 'Title';
  if (key === 'content') return 'Summary';
  if (key === 'tags') return 'Keywords';
  if (key === 'section') return 'Topic';
  if (key === 'path') return 'File location';
  if (key === 'fuzzy match') return 'Best match';
  return raw || 'Page';
}

/** Guarded select assignment: unknown facet values reset to ALL. */
function setSelectGuarded(sel: HTMLSelectElement, value: string): void {
  const exists = Array.from(sel.options).some((o) => o.value === value);
  sel.value = exists ? value : 'ALL';
}

function setBadgeForCompetitor(choice: EngineChoice): void {
  if (!engineBadge) return;
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
    if (engineBadge) {
      engineBadge.textContent = 'Fast mode on';
      engineBadge.className = 'badge badge-gpu';
    }
  } else {
    if (engineBadge) {
      engineBadge.textContent = 'Standard mode — same results';
      engineBadge.className = 'badge badge-cpu';
    }
  }
}

function updateEngineHint(): void {
  if (!engineHint) return;
  const choice = currentEngine();
  if (choice === 'ufuzzy' || choice === 'fuse') {
    engineHint.textContent = 'Popular libraries use their own matching — “How to match” is ignored and scores can’t be compared with ours.';
  } else if (choice === 'native') {
    engineHint.textContent = 'Simple scan looks for exact words only — it is the slowest option and ignores “How to match”.';
  } else {
    engineHint.textContent = 'Our two modes return identical results. Popular libraries rank differently, so scores can’t be compared directly.';
  }
  workerSelect.disabled = isCompetitor(choice);
  workerSelect.title = isCompetitor(choice)
    ? 'Background mode only applies to our search, not popular libraries.'
    : '';
}

function updatePreview(record: DocPageRecord | null): void {
  if (!record) {
    previewTitle.textContent = 'Welcome to Acme Docs';
    if (previewMetaTop) previewMetaTop.textContent = 'Offline docs';
    previewMeta.textContent = 'Pick a page on the left to start reading';
    previewContent.innerHTML = `<p>This is an embedded docs site. Search above to filter pages, or browse the list on the left.</p><h2>Getting started</h2><p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.</p>`;
    previewTags.textContent = '-';
    previewPath.textContent = '// File location appears here';
    return;
  }

  previewTitle.textContent = record.title;
  if (previewMetaTop) previewMetaTop.textContent = `${record.section} · v${record.version}`;
  previewMeta.textContent = `${record.section} · v${record.version} · ${record.readingMinutes} min read`;
  const body = docBody(record);
  const paras = body.split('\n\n').filter(Boolean);
  previewContent.innerHTML = '';
  paras.forEach((p, i) => {
    if (i === 0) {
      const lead = document.createElement('p');
      lead.textContent = p;
      previewContent.appendChild(lead);
      const h = document.createElement('h2');
      h.textContent = 'Overview';
      previewContent.appendChild(h);
    } else {
      const el = document.createElement('p');
      el.textContent = p;
      previewContent.appendChild(el);
    }
  });
  const listHead = document.createElement('h3');
  listHead.textContent = 'On this page';
  previewContent.appendChild(listHead);
  const ul = document.createElement('ul');
  for (const kw of record.tags.split(',').map((s) => s.trim()).filter(Boolean).slice(0, 4)) {
    const li = document.createElement('li');
    li.textContent = kw;
    ul.appendChild(li);
  }
  previewContent.appendChild(ul);
  previewTags.textContent = record.tags;
  previewPath.textContent = record.path;
}

function setSearchMeta(line: string, latency: string): void {
  if (statMatchesLine) statMatchesLine.textContent = line;
  if (statLatencyLine) statLatencyLine.textContent = latency;
  statMatches.textContent = line.replace(/[^0-9,]/g, '') || '0';
  statLatency.textContent = latency || '–';
}

function renderResults(): void {
  resultsList.innerHTML = '';
  const browsing = !searchInput.value.trim();

  if (activeResults.length === 0) {
    const li = document.createElement('li');
    li.className = 'empty-note';
    li.textContent = browsing
      ? 'No pages in this section yet.'
      : 'No pages match. Try fewer or different words.';
    resultsList.appendChild(li);
    if (browsing) updatePreview(null);
    return;
  }

  let lastSection = '';
  activeResults.forEach((res, idx) => {
    if (browsing && res.doc.section !== lastSection) {
      lastSection = res.doc.section;
      const header = document.createElement('li');
      header.className = 'nav-group-label';
      header.textContent = lastSection;
      resultsList.appendChild(header);
    }
    const li = document.createElement('li');
    li.className = `result-item ${idx === selectedIndex ? 'selected' : ''}`;

    const section = res.doc.section;
    const sectionLabel = escapeHtml(section.charAt(0).toUpperCase());

    const highlightedTitle = res.highlightedText?.title ? sanitizeHighlighted(res.highlightedText.title) : escapeHtml(res.doc.title);
    const highlightedPath = res.highlightedText?.path ? sanitizeHighlighted(res.highlightedText.path) : escapeHtml(res.doc.path);
    const highlightedContent = res.highlightedText?.content ? sanitizeHighlighted(res.highlightedText.content) : escapeHtml(res.doc.content);

    const metaBadges = browsing ? '' : `
        <div class="item-right">
          <span class="field-badge">Found in ${escapeHtml(friendlyField(res.matchedField))}</span>
          <span class="score-badge">Relevance ${res.score}</span>
        </div>`;

    li.innerHTML = `
      <div class="item-header">
        <div class="item-left">
          <span class="kind-icon kind-${escapeHtml(section)}">${sectionLabel}</span>
          <span class="item-filename">${highlightedTitle}</span>
        </div>${metaBadges}
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
  const bySection = facets?.bySection;
  if (!bySection || bySection.type !== 'terms' || !Array.isArray(bySection.buckets) || bySection.buckets.length === 0) {
    facetBar.style.display = 'none';
    return;
  }
  facetBar.style.display = 'flex';
  for (const b of bySection.buckets) {
    const chip = document.createElement('button');
    chip.className = 'facet-chip';
    chip.type = 'button';
    chip.textContent = `${String(b.value)} · ${Number(b.count).toLocaleString()} pages`;
    chip.title = `Show only ${String(b.value)} pages`;
    chip.addEventListener('click', () => {
      setSelectGuarded(sectionSelect, String(b.value));
      performSearch();
    });
    facetList.appendChild(chip);
  }
}

function getFilteredRecords(sectionFilter: string, versionFilter: string): DocPageRecord[] {
  return engine.getRecords().filter((r) =>
    (sectionFilter === 'ALL' || r.section === sectionFilter) &&
    (versionFilter === 'ALL' || r.version === versionFilter)
  );
}

function facetsForDocs(docs: DocPageRecord[]): Record<string, any> | undefined {
  const counts = new Map<string, number>();
  for (const d of docs) counts.set(d.section, (counts.get(d.section) ?? 0) + 1);
  const buckets = [...counts.entries()]
    .map(([value, count]) => ({ value, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 10);
  if (buckets.length === 0) return undefined;
  return { bySection: { type: 'terms', buckets } };
}

/** Popular-library search over the same pages. Scores use each library's own ranking. */
async function searchCompetitor(
  query: string,
  choice: EngineChoice,
  sectionFilter: string,
  versionFilter: string,
  limit: number
): Promise<{ results: DocsSearchResult[]; totalMatches: number; durationMs: number; facets: Record<string, any> | undefined }> {
  const filtered = getFilteredRecords(sectionFilter, versionFilter);
  const clean = query.trim();
  if (!clean) {
    const docs = filtered.slice(0, limit);
    return {
      results: docs.map((doc, i) => ({ id: doc.id, score: Math.max(1, 50 - i), matchedField: 'Page', doc })),
      totalMatches: filtered.length,
      durationMs: 0,
      facets: facetsForDocs(filtered)
    };
  }

  if (choice === 'ufuzzy') {
    const strings = filtered.map((r) => `${r.title} ${r.tags} ${r.content}`);
    const t0 = performance.now();
    const out = cpuEngine.searchWithUFuzzy(strings, clean, limit);
    const durationMs = performance.now() - t0;
    const results: DocsSearchResult[] = out.results
      .map((item) => filtered[item.index])
      .filter((doc): doc is DocPageRecord => Boolean(doc))
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
      .filter((doc): doc is DocPageRecord => Boolean(doc));
    return { results, totalMatches: out.totalMatches, durationMs, facets: facetsForDocs(matchedDocs) };
  }

  if (choice === 'fuse') {
    const fuse = new Fuse(filtered, {
      keys: [
        { name: 'title', weight: 2 },
        { name: 'tags', weight: 1.5 },
        { name: 'content', weight: 1 }
      ],
      threshold: 0.4,
      ignoreLocation: true,
      includeScore: true
    });
    const t0 = performance.now();
    const found = fuse.search(clean);
    const durationMs = performance.now() - t0;
    const results: DocsSearchResult[] = found.slice(0, limit).map((hit, i) => {
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
      facets: facetsForDocs(found.map((hit) => hit.item))
    };
  }

  // native: simple exact-word scan for reference
  const t0 = performance.now();
  const q = clean.toLowerCase();
  const matched = filtered.filter((r) =>
    `${r.title} ${r.content} ${r.tags} ${r.path}`.toLowerCase().includes(q)
  );
  const durationMs = performance.now() - t0;
  const results: DocsSearchResult[] = matched.slice(0, limit).map((doc, i) => {
    const hayTitle = doc.title.toLowerCase();
    const hayTags = doc.tags.toLowerCase();
    const hayContent = doc.content.toLowerCase();
    const field = hayTitle.includes(q) ? 'Title' : hayTags.includes(q) ? 'Keywords' : hayContent.includes(q) ? 'Summary' : 'Page';
    return { id: doc.id, score: Math.max(1, 80 - i), matchedField: field, doc };
  });
  return { results, totalMatches: matched.length, durationMs, facets: facetsForDocs(matched) };
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
  const choice = currentEngine();
  updateEngineHint();

  if (isCompetitor(choice)) {
    setBadgeForCompetitor(choice);
    try {
      const res = await searchCompetitor(query, choice, sectionFilter, versionFilter, 50);
      activeResults = res.results;
      selectedIndex = 0;
      const label = query
        ? `${res.totalMatches.toLocaleString()} results`
        : `${res.totalMatches.toLocaleString()} pages`;
      setSearchMeta(label, query ? `${res.durationMs.toFixed(2)} ms` : '');
      renderSuggestions([]);
      renderFacets(res.facets);
      renderResults();
    } catch (err: any) {
      if (err?.name === 'AbortError') return;
      console.error('[Search Error]', err);
    }
    return;
  }

  // Browse mode: empty query shows the docs navigation instead of no hits.
  if (!query) {
    const docs = getFilteredRecords(sectionFilter, versionFilter).slice(0, 100);
    activeResults = docs.map((doc) => ({ id: doc.id, score: 50, matchedField: 'Page', doc }));
    selectedIndex = 0;
    setSearchMeta(`${getFilteredRecords(sectionFilter, versionFilter).length.toLocaleString()} pages`, '');
    renderSuggestions([]);
    renderFacets(facetsForDocs(getFilteredRecords(sectionFilter, versionFilter)));
    renderResults();
    await updateTelemetry();
    return;
  }

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
    const label = `${searchRes.totalMatches.toLocaleString()} results`;
    const engineName = searchRes.engine === 'webgpu' ? 'Fast mode' : 'Standard mode';
    setSearchMeta(`${label} · ${engineName}`, `${searchRes.searchDurationMs.toFixed(2)} ms`);

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

// Press "/" or Cmd/Ctrl+K anywhere to focus search, like popular docs sites.
document.addEventListener('keydown', (e) => {
  const target = e.target as HTMLElement | null;
  const typing = target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT');
  if ((e.key === '/' && !typing) || ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k')) {
    e.preventDefault();
    searchInput.focus();
  }
});

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
  formTitle.focus();
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
    const title = formTitle.value.trim() || 'Untitled page';
    const rawSection = formSection.value.trim() || 'Guide';
    const allowed = ['Guide', 'API', 'Storage', 'Reliability', 'Reference'];
    const section = allowed.includes(rawSection) ? rawSection : 'Guide';
    const tags = formTags.value.trim() || 'docs';
    const content = formContent.value.trim() || 'A new doc page added from the demo.';

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
    closeModal();
    formTitle.value = '';
    formSection.value = '';
    formTags.value = '';
    formContent.value = '';

    idbStatus.textContent = `Added “${title.slice(0, 40)}” to this session`;
    await updateTelemetry();
    await performSearch();
  } finally {
    btnModalSave.disabled = false;
  }
});

btnBatchAdd.addEventListener('click', async () => {
  btnBatchAdd.disabled = true;
  try {
    const newRecords = generateDocsRecords(200).map((r, i) => ({
      ...r,
      id: `batch-${Date.now()}-${i}`
    }));

    await engine.batchAdd(newRecords);
    idbStatus.textContent = `Added 200 sample pages`;
    await updateTelemetry();
    await performSearch();
  } finally {
    btnBatchAdd.disabled = false;
  }
});

btnReset.addEventListener('click', async () => {
  records = [...CORE_DOCS];
  await engine.init(records);
  idbStatus.textContent = 'Back to starter docs';
  await updateTelemetry();
  await performSearch();
});

// Offline bundle persistence + GPU recovery
btnSaveIdb.addEventListener('click', async () => {
  btnSaveIdb.disabled = true;
  try {
    const res = await engine.saveSnapshotToIDB();
    idbStatus.textContent = `Saved ${(res.byteLength / 1024).toFixed(1)} KB · ready offline`;
  } catch (err) {
    console.error('[Save IDB Error]', err);
    idbStatus.textContent = 'Couldn’t save — please try again';
  } finally {
    btnSaveIdb.disabled = false;
  }
});

btnRestoreIdb.addEventListener('click', async () => {
  btnRestoreIdb.disabled = true;
  try {
    const res = await engine.restoreSnapshotFromIDB();
    idbStatus.textContent = `Opened ${res.recordCount.toLocaleString()} pages · ready`;
    await updateTelemetry();
    await performSearch();
  } catch (err) {
    console.error('[Restore IDB Error]', err);
    idbStatus.textContent = 'No saved copy found yet';
  } finally {
    btnRestoreIdb.disabled = false;
  }
});

btnClearIdb.addEventListener('click', async () => {
  await engine.clearIDB();
  idbStatus.textContent = 'Saved copy deleted';
});

btnRebuildGpu.addEventListener('click', async () => {
  btnRebuildGpu.disabled = true;
  try {
    const rebuilt = await engine.rebuildGpu();
    idbStatus.textContent = rebuilt ? 'Fast search restarted' : 'Background mode restores from a saved copy instead';
    await updateTelemetry();
    await performSearch();
  } catch (err) {
    console.error('[Rebuild GPU Error]', err);
    idbStatus.textContent = 'Couldn’t restart fast search';
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
  updateEngineHint();
  await engine.init(records);
  await updateTelemetry();
  await performSearch();
}

bootstrap().catch((err) => {
  console.error('[Bootstrap Error]', err);
});
