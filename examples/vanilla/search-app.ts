import {
  DocumentIndex,
  SearchWorkerClient,
  type DocumentId,
  type DocumentIndexOptions,
  type DocumentSearchOptions,
  type DocumentSearchResultItem,
  type DocumentSearchResponse,
  type DocumentIndexStats,
  type MutationResult,
  type AddOptions,
  type SearchMode,
} from 'webgpu-search';

export interface VanillaSearchAppOptions<TDoc = any> {
  /** Initial documents */
  initialDocs?: TDoc[];
  /** Index options */
  indexOptions?: DocumentIndexOptions<TDoc>;
  /** Optional pre-instantiated DocumentIndex */
  index?: DocumentIndex<TDoc>;
  /** Optional pre-instantiated SearchWorkerClient */
  workerClient?: SearchWorkerClient<TDoc>;
  /** Optional worker instance or factory */
  worker?: Worker | (() => Worker);
  /** Debounce delay in ms (default: 150) */
  debounceMs?: number;
  /** Search options */
  searchOptions?: DocumentSearchOptions<TDoc>;
}

export interface VanillaSearchAppHandle<TDoc = any> {
  search: (query: string) => Promise<DocumentSearchResponse<TDoc> | null>;
  add: (docs: TDoc | TDoc[], options?: AddOptions) => Promise<MutationResult>;
  update: (docs: TDoc | TDoc[]) => Promise<MutationResult>;
  remove: (ids: DocumentId | DocumentId[]) => Promise<MutationResult>;
  refreshStats: () => Promise<DocumentIndexStats | null>;
  destroy: () => void;
}

/**
 * Initializes an interactive vanilla TypeScript search application into a container element.
 * Provides live debounced input handling, highlight rendering, telemetry HUD updates, and mutations.
 */
export function createVanillaSearchApp<TDoc extends Record<string, any>>(
  container: HTMLElement,
  options: VanillaSearchAppOptions<TDoc> = {}
): VanillaSearchAppHandle<TDoc> {
  const doc = container.ownerDocument || (typeof document !== 'undefined' ? document : null);
  if (!doc) {
    throw new Error('[webgpu-search] createVanillaSearchApp requires a DOM environment.');
  }

  const {
    initialDocs = [],
    indexOptions,
    index: externalIndex,
    workerClient: externalWorkerClient,
    worker,
    debounceMs = 150,
    searchOptions = { mode: 'fuzzy', highlight: true, tag: 'mark', limit: 20 },
  } = options;

  let currentMode: SearchMode = searchOptions.mode || 'fuzzy';
  let indexInstance: DocumentIndex<TDoc> | null = null;
  let workerClientInstance: SearchWorkerClient<TDoc> | null = null;
  let isOwned = false;

  let searchId = 0;
  let activeAbortController: AbortController | null = null;
  let debounceTimer: any = null;
  let currentStats: DocumentIndexStats | null = null;

  // Build UI Shell
  container.innerHTML = `
    <div class="vanilla-search-root" style="max-width: 800px; margin: 0 auto; font-family: system-ui, -apple-system, sans-serif;">
      <div style="display: flex; gap: 8px; margin-bottom: 12px;">
        <input type="text" class="vsearch-input" placeholder="Type to search..." style="flex: 1; padding: 10px 14px; font-size: 16px; border-radius: 6px; border: 1px solid #ccc;" />
        <button class="vsearch-clear-btn" style="padding: 0 14px; border: 1px solid #ccc; border-radius: 6px; background: #fff; cursor: pointer;">Clear</button>
        <button class="vsearch-mode-btn" style="padding: 0 14px; border: 1px solid #0066cc; border-radius: 6px; background: #0066cc; color: #fff; cursor: pointer;">Mode: ${currentMode}</button>
      </div>

      <div style="display: flex; gap: 8px; margin-bottom: 12px;">
        <button class="vsearch-add-btn" style="padding: 6px 12px; border: 1px solid #28a745; border-radius: 6px; background: #28a745; color: #fff; cursor: pointer;">+ Add Document</button>
      </div>

      <div class="vsearch-telemetry" style="background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 12px; margin-bottom: 16px; font-size: 13px;">
        <div class="vsearch-hud-status" style="margin-bottom: 6px; display: flex; gap: 10px; align-items: center;">
          <strong>Engine:</strong> <span class="vsearch-engine-badge" style="padding: 2px 8px; border-radius: 4px; background: #6c757d; color: #fff; font-weight: 600;">INIT</span>
          <span class="vsearch-fallback-pill" style="display: none; background: #ffeeba; color: #856404; padding: 2px 6px; border-radius: 4px;"></span>
          <span class="vsearch-status-label" style="color: #666;">Booting...</span>
        </div>
        <div class="vsearch-hud-metrics" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 8px;">
          <div><strong>Docs:</strong> <span class="vsearch-m-docs">0</span></div>
          <div><strong>VRAM:</strong> <span class="vsearch-m-vram">0 KB</span></div>
          <div><strong>RAM:</strong> <span class="vsearch-m-ram">0 KB</span></div>
          <div><strong>Latency:</strong> <span class="vsearch-m-latency">0 ms</span></div>
          <div><strong>Epoch:</strong> <span class="vsearch-m-epoch">0</span></div>
        </div>
      </div>

      <div class="vsearch-count" style="margin-bottom: 8px; font-size: 13px; color: #666;">Ready</div>
      <ul class="vsearch-results" style="list-style: none; padding: 0; margin: 0;"></ul>
    </div>
  `;

  const inputEl = container.querySelector<HTMLInputElement>('.vsearch-input')!;
  const clearBtn = container.querySelector<HTMLButtonElement>('.vsearch-clear-btn')!;
  const modeBtn = container.querySelector<HTMLButtonElement>('.vsearch-mode-btn')!;
  const addBtn = container.querySelector<HTMLButtonElement>('.vsearch-add-btn')!;
  const engineBadge = container.querySelector<HTMLElement>('.vsearch-engine-badge')!;
  const fallbackPill = container.querySelector<HTMLElement>('.vsearch-fallback-pill')!;
  const statusLabel = container.querySelector<HTMLElement>('.vsearch-status-label')!;
  const mDocs = container.querySelector<HTMLElement>('.vsearch-m-docs')!;
  const mVram = container.querySelector<HTMLElement>('.vsearch-m-vram')!;
  const mRam = container.querySelector<HTMLElement>('.vsearch-m-ram')!;
  const mLatency = container.querySelector<HTMLElement>('.vsearch-m-latency')!;
  const mEpoch = container.querySelector<HTMLElement>('.vsearch-m-epoch')!;
  const countEl = container.querySelector<HTMLElement>('.vsearch-count')!;
  const resultsEl = container.querySelector<HTMLUListElement>('.vsearch-results')!;

  function updateHUD(stats: DocumentIndexStats | null, latencyMs?: number, fallback?: string): void {
    if (!stats) return;
    engineBadge.textContent = stats.engine.toUpperCase();
    engineBadge.style.background = stats.engine === 'webgpu' ? '#28a745' : '#e0a800';

    const effFallback = fallback || stats.fallbackReason;
    if (effFallback) {
      fallbackPill.style.display = 'inline';
      fallbackPill.textContent = `Fallback: ${effFallback}`;
    } else {
      fallbackPill.style.display = 'none';
    }

    mDocs.textContent = String(stats.docCount);
    mVram.textContent = `${(stats.memory.vramBytes / 1024).toFixed(1)} KB`;
    mRam.textContent = `${(stats.memory.ramBytes / 1024).toFixed(1)} KB`;
    mEpoch.textContent = String(stats.mutationEpoch);
    if (latencyMs !== undefined) {
      mLatency.textContent = `${latencyMs.toFixed(2)} ms`;
    }
  }

  function renderResults(results: DocumentSearchResultItem<TDoc>[], totalMatches: number): void {
    countEl.textContent = `Showing ${results.length} of ${totalMatches} matches`;
    resultsEl.innerHTML = '';

    for (const item of results) {
      const li = doc.createElement('li');
      li.style.cssText = 'padding: 12px; border-bottom: 1px solid #eee; display: flex; flex-direction: column; gap: 4px;';

      const titleRow = doc.createElement('div');
      titleRow.style.cssText = 'display: flex; justify-content: space-between; align-items: center;';

      const titleSpan = doc.createElement('span');
      titleSpan.style.cssText = 'font-weight: 600; font-size: 16px;';
      if (item.highlightedText?.title) {
        titleSpan.innerHTML = item.highlightedText.title;
      } else {
        titleSpan.textContent = String(item.doc?.title || item.id);
      }

      const scoreTag = doc.createElement('span');
      scoreTag.style.cssText = 'font-size: 12px; color: #888; background: #eee; padding: 2px 6px; border-radius: 4px;';
      scoreTag.textContent = `Score: ${item.score} (${item.matchedField})`;

      titleRow.appendChild(titleSpan);
      titleRow.appendChild(scoreTag);
      li.appendChild(titleRow);

      if (item.doc?.description) {
        const descDiv = doc.createElement('div');
        descDiv.style.cssText = 'font-size: 14px; color: #444;';
        if (item.highlightedText?.description) {
          descDiv.innerHTML = item.highlightedText.description;
        } else {
          descDiv.textContent = String(item.doc.description);
        }
        li.appendChild(descDiv);
      }

      resultsEl.appendChild(li);
    }
  }

  async function refreshStats(): Promise<DocumentIndexStats | null> {
    try {
      let s: DocumentIndexStats | null = null;
      if (workerClientInstance) {
        s = await workerClientInstance.getStats();
      } else if (indexInstance) {
        s = indexInstance.getStats();
      }
      if (s) {
        currentStats = s;
        updateHUD(s);
      }
      return s;
    } catch {
      return null;
    }
  }

  async function executeSearch(query: string): Promise<DocumentSearchResponse<TDoc> | null> {
    if (!indexInstance && !workerClientInstance) return null;

    const currentSearchId = ++searchId;

    if (activeAbortController) {
      activeAbortController.abort();
    }
    const abortController = new AbortController();
    activeAbortController = abortController;

    statusLabel.textContent = 'Searching...';

    try {
      let res: DocumentSearchResponse<TDoc>;
      if (workerClientInstance) {
        res = await workerClientInstance.search(query, {
          ...searchOptions,
          mode: currentMode,
          signal: abortController.signal,
        });
      } else if (indexInstance) {
        res = await indexInstance.search(query, {
          ...searchOptions,
          mode: currentMode,
          signal: abortController.signal,
        });
      } else {
        return null;
      }

      if (currentSearchId === searchId) {
        statusLabel.textContent = 'Ready';
        renderResults(res.results, res.totalMatches);
        updateHUD(currentStats, res.timings.totalMs, res.fallbackReason);
      }
      return res;
    } catch (err: any) {
      if (err.name === 'AbortError' || currentSearchId !== searchId) {
        return null;
      }
      statusLabel.textContent = `Error: ${err.message}`;
      return null;
    }
  }

  // Event handlers
  const handleInput = () => {
    if (debounceTimer) clearTimeout(debounceTimer);
    const q = inputEl.value;
    if (debounceMs <= 0) {
      executeSearch(q);
    } else {
      debounceTimer = setTimeout(() => {
        executeSearch(q);
      }, debounceMs);
    }
  };

  const handleClear = () => {
    inputEl.value = '';
    executeSearch('');
  };

  const handleModeToggle = () => {
    currentMode = currentMode === 'fuzzy' ? 'substring' : 'fuzzy';
    modeBtn.textContent = `Mode: ${currentMode}`;
    executeSearch(inputEl.value);
  };

  const handleAdd = async () => {
    const epoch = currentStats ? currentStats.mutationEpoch : 0;
    const newDoc = {
      id: `doc-${Date.now()}`,
      title: `Dynamic Document #${epoch + 1}`,
      description: 'Added via createVanillaSearchApp runtime mutation.',
    } as any;

    if (workerClientInstance) {
      await workerClientInstance.add(newDoc);
    } else if (indexInstance) {
      await indexInstance.add(newDoc);
    }
    await refreshStats();
    if (inputEl.value.trim().length > 0) {
      await executeSearch(inputEl.value);
    }
  };

  inputEl.addEventListener('input', handleInput);
  clearBtn.addEventListener('click', handleClear);
  modeBtn.addEventListener('click', handleModeToggle);
  addBtn.addEventListener('click', handleAdd);

  // Initialize
  (async () => {
    statusLabel.textContent = 'Initializing engine...';
    try {
      if (externalWorkerClient) {
        workerClientInstance = externalWorkerClient;
        isOwned = false;
      } else if (externalIndex) {
        indexInstance = externalIndex;
        isOwned = false;
      } else if (worker) {
        const client = new SearchWorkerClient<TDoc>({ worker });
        await client.init(indexOptions);
        if (initialDocs.length > 0) {
          await client.add(initialDocs);
        }
        workerClientInstance = client;
        isOwned = true;
      } else {
        const defaultFields = indexOptions?.fields ?? ['title', 'description'];
        const idx = await DocumentIndex.create(initialDocs, {
          ...indexOptions,
          fields: defaultFields,
        });
        indexInstance = idx;
        isOwned = true;
      }

      statusLabel.textContent = 'Ready';
      await refreshStats();
      if (inputEl.value.trim().length > 0) {
        executeSearch(inputEl.value);
      }
    } catch (err: any) {
      statusLabel.textContent = `Init Error: ${err.message}`;
    }
  })();

  function destroy(): void {
    inputEl.removeEventListener('input', handleInput);
    clearBtn.removeEventListener('click', handleClear);
    modeBtn.removeEventListener('click', handleModeToggle);
    addBtn.removeEventListener('click', handleAdd);

    if (activeAbortController) {
      activeAbortController.abort();
      activeAbortController = null;
    }
    if (debounceTimer) {
      clearTimeout(debounceTimer);
      debounceTimer = null;
    }
    if (isOwned) {
      if (workerClientInstance) {
        workerClientInstance.destroy();
        workerClientInstance = null;
      }
      if (indexInstance) {
        indexInstance.destroy();
        indexInstance = null;
      }
    }
    container.innerHTML = '';
  }

  return {
    search: executeSearch,
    add: async (docs, addOpts) => {
      let r: MutationResult;
      if (workerClientInstance) {
        r = await workerClientInstance.add(docs, addOpts);
      } else if (indexInstance) {
        r = await indexInstance.add(docs, addOpts);
      } else {
        throw new Error('Index not initialized');
      }
      await refreshStats();
      return r;
    },
    update: async (docs) => {
      let r: MutationResult;
      if (workerClientInstance) {
        r = await workerClientInstance.update(docs);
      } else if (indexInstance) {
        r = await indexInstance.update(docs);
      } else {
        throw new Error('Index not initialized');
      }
      await refreshStats();
      return r;
    },
    remove: async (ids) => {
      let r: MutationResult;
      if (workerClientInstance) {
        r = await workerClientInstance.remove(ids);
      } else if (indexInstance) {
        r = await indexInstance.remove(ids);
      } else {
        throw new Error('Index not initialized');
      }
      await refreshStats();
      return r;
    },
    refreshStats,
    destroy,
  };
}
