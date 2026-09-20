<script lang="ts">
  import { onDestroy } from 'svelte';
  import { createDocumentSearch } from './documentSearchStore';

  interface LogDoc {
    id: string;
    level: string;
    message: string;
    service: string;
  }

  const SAMPLE_LOGS: LogDoc[] = [
    { id: '1', level: 'ERROR', message: 'Failed to bind WebGPU storage buffer layout', service: 'pipeline' },
    { id: '2', level: 'WARN', message: 'Query token length exceeds max tokens; falling back to CPU', service: 'tokenizer' },
    { id: '3', level: 'INFO', message: 'U2D3 binary snapshot restored from IndexedDB cache', service: 'storage' },
    { id: '4', level: 'INFO', message: 'Worker thread dispatched compute pass in 0.42ms', service: 'worker' },
  ];

  const search = createDocumentSearch<LogDoc>({
    initialDocs: SAMPLE_LOGS,
    indexOptions: {
      fields: [
        { name: 'message', weight: 2.0 },
        { name: 'service', weight: 1.5 },
        { name: 'level', weight: 1.0 },
      ],
    },
    searchOptions: {
      mode: 'fuzzy',
      highlight: true,
      tag: 'mark',
      limit: 10,
    },
  });

  const handleAddLog = async () => {
    await search.add({
      id: `log-${Date.now()}`,
      level: 'DEBUG',
      message: `Dynamic runtime log entry #${$search.mutationEpoch + 1}`,
      service: 'telemetry',
    });
  };

  onDestroy(() => {
    search.destroy();
  });
</script>

<div class="search-box">
  <h2>WebGPU Search (Svelte Recipe)</h2>

  <!-- Search Input -->
  <div class="input-bar">
    <input
      type="text"
      value={$search.query}
      on:input={(e) => search.setQuery((e.target as HTMLInputElement).value)}
      placeholder="Type to search logs..."
      class="search-input"
    />
    {#if $search.query}
      <button class="btn" on:click={() => search.setQuery('')}>Clear</button>
    {/if}
  </div>

  <!-- Actions -->
  <div class="action-bar">
    <button class="btn btn-primary" disabled={$search.isIndexing} on:click={handleAddLog}>
      + Add Log Entry
    </button>
  </div>

  <!-- Telemetry HUD -->
  <div class="telemetry-card">
    <div class="status-row">
      <strong>Engine:</strong>
      <span class="badge" class:badge-gpu={$search.engine === 'webgpu'} class:badge-cpu={$search.engine === 'cpu'}>
        {$search.engine ? $search.engine.toUpperCase() : 'INITIALIZING'}
      </span>
      {#if $search.fallbackReason}
        <span class="badge-fallback">Fallback: {$search.fallbackReason}</span>
      {/if}
      <span>Status: {$search.isSearching ? 'Searching...' : $search.isIndexing ? 'Indexing...' : $search.isReady ? 'Ready' : 'Booting'}</span>
    </div>

    <div class="metrics-grid">
      <div><strong>Docs:</strong> {$search.stats ? $search.stats.docCount : 0}</div>
      <div><strong>VRAM:</strong> {$search.stats ? ($search.stats.memory.vramBytes / 1024).toFixed(1) + ' KB' : '0 KB'}</div>
      <div><strong>RAM:</strong> {$search.stats ? ($search.stats.memory.ramBytes / 1024).toFixed(1) + ' KB' : '0 KB'}</div>
      <div><strong>Latency:</strong> {$search.timings ? $search.timings.totalMs.toFixed(2) + ' ms' : '0 ms'}</div>
      <div><strong>Epoch:</strong> {$search.mutationEpoch}</div>
    </div>
  </div>

  <!-- Error State -->
  {#if $search.error}
    <div class="error-banner">Error: {$search.error.message}</div>
  {/if}

  <!-- Results Count -->
  <div class="results-header">
    Showing {$search.results.length} of {$search.totalMatches} matches
  </div>

  <!-- Results List -->
  <ul class="results-list">
    {#each $search.results as item (item.id)}
      <li class="result-item">
        <div class="result-title-row">
          <span class="result-title">
            {#if item.highlightedText?.message}
              {@html item.highlightedText.message}
            {:else}
              {item.doc.message}
            {/if}
          </span>
          <span class="score-tag">Score: {item.score} ({item.matchedField})</span>
        </div>

        <div class="category-tag">[{item.doc.level}] {item.doc.service}</div>
      </li>
    {/each}
  </ul>
</div>

<style>
  .search-box {
    max-width: 800px;
    margin: 0 auto;
    font-family: system-ui, -apple-system, sans-serif;
    padding: 20px;
  }
  .input-bar {
    display: flex;
    gap: 10px;
    margin-bottom: 15px;
  }
  .search-input {
    flex: 1;
    padding: 10px 14px;
    font-size: 16px;
    border-radius: 6px;
    border: 1px solid #ccc;
  }
  .btn {
    padding: 6px 14px;
    border-radius: 6px;
    border: 1px solid #ccc;
    cursor: pointer;
  }
  .btn-primary {
    background: #0066cc;
    color: #fff;
    border: none;
  }
  .action-bar {
    display: flex;
    gap: 10px;
    margin-bottom: 15px;
  }
  .telemetry-card {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 20px;
    font-size: 13px;
  }
  .status-row {
    display: flex;
    gap: 12px;
    align-items: center;
    margin-bottom: 8px;
  }
  .badge {
    padding: 2px 8px;
    border-radius: 4px;
    color: #fff;
    font-weight: 600;
  }
  .badge-gpu {
    background: #28a745;
  }
  .badge-cpu {
    background: #e0a800;
  }
  .badge-fallback {
    background: #ffeeba;
    color: #856404;
    padding: 2px 6px;
    border-radius: 4px;
  }
  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 8px;
  }
  .error-banner {
    color: #d9534f;
    background: #fdf7f7;
    padding: 10px;
    border-radius: 6px;
    margin-bottom: 15px;
  }
  .results-header {
    margin-bottom: 10px;
    color: #666;
    font-size: 14px;
  }
  .results-list {
    list-style: none;
    padding: 0;
    margin: 0;
  }
  .result-item {
    padding: 12px;
    border-bottom: 1px solid #eee;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .result-title-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .result-title {
    font-weight: 600;
    font-size: 16px;
  }
  .score-tag {
    font-size: 12px;
    color: #888;
    background: #eee;
    padding: 2px 6px;
    border-radius: 4px;
  }
  .category-tag {
    font-size: 12px;
    color: #0066cc;
  }
</style>
