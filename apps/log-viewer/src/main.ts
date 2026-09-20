import { LogEngine } from './log-engine';
import { generateStructuredLogs } from './log-generator';
import { VirtualGrid, type VirtualGridItem } from './virtual-grid';
import type { StructuredLogRecord } from './types';

// DOM Elements
const searchInput = document.getElementById('search-input') as HTMLInputElement;
const levelSelect = document.getElementById('level-select') as HTMLSelectElement;
const modeSelect = document.getElementById('mode-select') as HTMLSelectElement;
const engineSelect = document.getElementById('engine-select') as HTMLSelectElement;
const workerSelect = document.getElementById('worker-select') as HTMLSelectElement;
const engineBadge = document.getElementById('engine-badge') as HTMLSpanElement;

// HUD Elements
const hudTotalLogs = document.getElementById('hud-total-logs') as HTMLSpanElement;
const hudGpuRows = document.getElementById('hud-gpu-rows') as HTMLSpanElement;
const hudMatches = document.getElementById('hud-matches') as HTMLSpanElement;
const hudLatency = document.getElementById('hud-latency') as HTMLSpanElement;
const hudVram = document.getElementById('hud-vram') as HTMLSpanElement;
const hudRam = document.getElementById('hud-ram') as HTMLSpanElement;
const hudEpoch = document.getElementById('hud-epoch') as HTMLSpanElement;
const hudIdb = document.getElementById('hud-idb') as HTMLSpanElement;

// Action Buttons
const btnLoad10k = document.getElementById('btn-load-10k') as HTMLButtonElement;
const btnLoad50k = document.getElementById('btn-load-50k') as HTMLButtonElement;
const btnLoad100k = document.getElementById('btn-load-100k') as HTMLButtonElement;
const btnToggleStream = document.getElementById('btn-toggle-stream') as HTMLButtonElement;
const btnSaveIdb = document.getElementById('btn-save-idb') as HTMLButtonElement;
const btnRestoreIdb = document.getElementById('btn-restore-idb') as HTMLButtonElement;
const btnClearIdb = document.getElementById('btn-clear-idb') as HTMLButtonElement;

// Drawer Elements
const detailDrawer = document.getElementById('detail-drawer') as HTMLDivElement;
const btnCloseDrawer = document.getElementById('btn-close-drawer') as HTMLButtonElement;
const detailId = document.getElementById('detail-id') as HTMLSpanElement;
const detailTime = document.getElementById('detail-time') as HTMLSpanElement;
const detailLevel = document.getElementById('detail-level') as HTMLSpanElement;
const detailService = document.getElementById('detail-service') as HTMLSpanElement;
const detailMessage = document.getElementById('detail-message') as HTMLSpanElement;
const detailTrace = document.getElementById('detail-trace') as HTMLSpanElement;
const detailLatency = document.getElementById('detail-latency') as HTMLSpanElement;

const gridViewport = document.getElementById('grid-viewport') as HTMLDivElement;

// Application State
let currentLogCount = 10000;
let isStreaming = false;
let streamTimer: number | null = null;
let currentAbortController: AbortController | null = null;
let nextLogId = currentLogCount + 1;

const engine = new LogEngine({
  useWorker: true,
  preferGpu: true
});

const grid = new VirtualGrid({
  container: gridViewport,
  rowHeight: 36,
  onRowClick: (item: VirtualGridItem) => {
    openDetailDrawer(item.record);
  }
});

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

async function updateHUD(): Promise<void> {
  const stats = await engine.getStats();
  if (!stats) return;

  hudTotalLogs.textContent = stats.docCount.toLocaleString();
  hudGpuRows.textContent = stats.rowCount.toLocaleString();
  hudVram.textContent = formatBytes(stats.memory.vramBytes);
  hudRam.textContent = formatBytes(stats.memory.ramBytes);
  hudEpoch.textContent = String(stats.mutationEpoch);

  if (stats.engine === 'webgpu') {
    engineBadge.textContent = 'WebGPU Ready';
    engineBadge.className = 'badge-tag badge-gpu';
  } else {
    engineBadge.textContent = `CPU (${stats.fallbackReason || 'prefer-cpu'})`;
    engineBadge.className = 'badge-tag badge-cpu';
  }
}

function openDetailDrawer(record: StructuredLogRecord): void {
  detailId.textContent = record.id;
  detailTime.textContent = record.timestamp;
  detailLevel.textContent = record.level;
  detailService.textContent = record.service;
  detailMessage.textContent = record.message;
  detailTrace.textContent = record.traceId;
  detailLatency.textContent = `${record.latencyMs} ms`;
  detailDrawer.classList.add('open');
}

btnCloseDrawer.addEventListener('click', () => {
  detailDrawer.classList.remove('open');
});

async function performSearch(): Promise<void> {
  const query = searchInput.value.trim();
  const mode = modeSelect.value as 'fuzzy' | 'substring';
  const levelFilter = levelSelect.value;

  if (currentAbortController) {
    currentAbortController.abort();
  }
  const controller = new AbortController();
  currentAbortController = controller;

  if (!query) {
    let records = engine.getRecords();
    if (levelFilter !== 'ALL') {
      records = records.filter((r) => r.level === levelFilter);
    }
    const items: VirtualGridItem[] = records.map((r) => ({ record: r }));
    grid.setItems(items, { resetScroll: false });
    hudMatches.textContent = records.length.toLocaleString();
    hudLatency.textContent = '0.00 ms';
    return;
  }

  try {
    const searchRes = await engine.search(query, {
      mode,
      highlight: true,
      limit: 1000,
      levelFilter,
      signal: controller.signal
    });

    if (controller !== currentAbortController) return;

    grid.setSearchResults(searchRes.results);
    hudMatches.textContent = searchRes.totalMatches.toLocaleString();
    hudLatency.textContent = `${searchRes.searchDurationMs.toFixed(2)} ms`;
  } catch (err: any) {
    if (err?.name === 'AbortError') return;
    console.error('[Search Error]', err);
  }
}

async function loadDataset(count: number): Promise<void> {
  if (isStreaming) toggleStream();
  currentLogCount = count;
  nextLogId = count + 1;
  const initialLogs = generateStructuredLogs(count);
  await engine.init(initialLogs);
  await updateHUD();
  await performSearch();
}

// Ingestion Stream
function toggleStream(): void {
  isStreaming = !isStreaming;
  if (isStreaming) {
    btnToggleStream.textContent = '⏸ Pause Live Stream';
    btnToggleStream.classList.add('btn-streaming');

    let isIngesting = false;
    async function streamTick() {
      if (!isStreaming) return;
      if (isIngesting) {
        streamTimer = window.setTimeout(streamTick, 500);
        return;
      }
      isIngesting = true;
      try {
        const batchSize = 100;
        const newLogs = generateStructuredLogs(batchSize, nextLogId);
        nextLogId += batchSize;

        await engine.appendLogs(newLogs);
        await updateHUD();

        // If user isn't searching, refresh grid with current level filter
        if (!searchInput.value.trim()) {
          let records = engine.getRecords();
          const levelFilter = levelSelect.value;
          if (levelFilter !== 'ALL') {
            records = records.filter((r) => r.level === levelFilter);
          }
          const items: VirtualGridItem[] = records.slice(-1000).map((r) => ({ record: r }));
          grid.setItems(items, { resetScroll: false });
          hudMatches.textContent = records.length.toLocaleString();
        } else {
          // If searching, update matches
          await performSearch();
        }
      } finally {
        isIngesting = false;
        if (isStreaming) {
          streamTimer = window.setTimeout(streamTick, 500);
        }
      }
    }

    streamTimer = window.setTimeout(streamTick, 500);
  } else {
    btnToggleStream.textContent = '▶ Start Live Stream';
    btnToggleStream.classList.remove('btn-streaming');
    if (streamTimer !== null) {
      clearTimeout(streamTimer);
      streamTimer = null;
    }
  }
}

// Preset Handlers
btnLoad10k.addEventListener('click', () => loadDataset(10000));
btnLoad50k.addEventListener('click', () => loadDataset(50000));
btnLoad100k.addEventListener('click', () => loadDataset(100000));
btnToggleStream.addEventListener('click', toggleStream);

// Persistence Handlers
btnSaveIdb.addEventListener('click', async () => {
  btnSaveIdb.disabled = true;
  try {
    const res = await engine.saveSnapshotToIDB();
    hudIdb.textContent = `${formatBytes(res.byteLength)} (${res.durationMs.toFixed(1)}ms)`;
  } catch (err) {
    console.error('[Save IDB Error]', err);
    hudIdb.textContent = 'Save Failed';
  } finally {
    btnSaveIdb.disabled = false;
  }
});

btnRestoreIdb.addEventListener('click', async () => {
  if (isStreaming) toggleStream();
  btnRestoreIdb.disabled = true;
  try {
    const res = await engine.restoreSnapshotFromIDB();
    hudIdb.textContent = `Restored ${res.recordCount.toLocaleString()} docs in ${res.durationMs.toFixed(1)}ms`;
    await updateHUD();
    await performSearch();
  } catch (err) {
    console.error('[Restore IDB Error]', err);
    hudIdb.textContent = 'Restore Failed';
  } finally {
    btnRestoreIdb.disabled = false;
  }
});

btnClearIdb.addEventListener('click', async () => {
  await engine.clearIDB();
  hudIdb.textContent = 'Cleared';
});

// Controls Handlers with Debounce
let searchDebounceTimer: ReturnType<typeof setTimeout> | null = null;
searchInput.addEventListener('input', () => {
  if (searchDebounceTimer) clearTimeout(searchDebounceTimer);
  searchDebounceTimer = setTimeout(() => {
    performSearch();
  }, 60);
});

levelSelect.addEventListener('change', () => performSearch());
modeSelect.addEventListener('change', () => performSearch());

engineSelect.addEventListener('change', async () => {
  const preferGpu = engineSelect.value === 'webgpu';
  await engine.setPreferGpu(preferGpu);
  await updateHUD();
  await performSearch();
});

workerSelect.addEventListener('change', async () => {
  const useWorker = workerSelect.value === 'worker';
  await engine.setUseWorker(useWorker);
  await updateHUD();
  await performSearch();
});

// Initialize
async function bootstrap(): Promise<void> {
  await loadDataset(currentLogCount);
}

bootstrap().catch((err) => {
  console.error('[Log Viewer Bootstrap Error]', err);
});
