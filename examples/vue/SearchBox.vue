<template>
  <div class="search-container">
    <h2>WebGPU Fuzzy Search (Vue 3 Recipe)</h2>

    <!-- Search Input Bar -->
    <div class="input-bar">
      <input
        v-model="query"
        type="text"
        placeholder="Type to search (e.g. 'webgpu', 'fuzzy', 'worker')..."
        class="search-input"
      />
      <button v-if="query" class="btn" @click="query = ''">Clear</button>
    </div>

    <!-- Actions -->
    <div class="action-bar">
      <button class="btn btn-primary" :disabled="isIndexing" @click="handleAddDoc">
        + Add Document
      </button>
    </div>

    <!-- Observability Telemetry HUD -->
    <div class="telemetry-card">
      <div class="status-row">
        <span><strong>Engine:</strong></span>
        <span class="badge" :class="engine === 'webgpu' ? 'badge-gpu' : 'badge-cpu'">
          {{ engine ? engine.toUpperCase() : 'INITIALIZING' }}
        </span>
        <span v-if="fallbackReason" class="badge-fallback">
          Fallback: {{ fallbackReason }}
        </span>
        <span class="status-text">
          Status: {{ isSearching ? 'Searching...' : isIndexing ? 'Indexing...' : isReady ? 'Ready' : 'Booting' }}
        </span>
      </div>

      <div class="metrics-grid">
        <div><strong>Docs:</strong> {{ stats ? stats.docCount : 0 }}</div>
        <div><strong>VRAM:</strong> {{ stats ? (stats.memory.vramBytes / 1024).toFixed(1) + ' KB' : '0 KB' }}</div>
        <div><strong>RAM:</strong> {{ stats ? (stats.memory.ramBytes / 1024).toFixed(1) + ' KB' : '0 KB' }}</div>
        <div><strong>Latency:</strong> {{ timings ? timings.totalMs.toFixed(2) + ' ms' : '0 ms' }}</div>
        <div><strong>Epoch:</strong> {{ mutationEpoch }}</div>
      </div>
    </div>

    <!-- Error State -->
    <div v-if="error" class="error-banner">
      Error: {{ error.message }}
    </div>

    <!-- Results Count -->
    <div class="results-header">
      Showing {{ results.length }} of {{ totalMatches }} matches
    </div>

    <!-- Results List -->
    <ul class="results-list">
      <li v-for="item in results" :key="item.id" class="result-item">
        <div class="result-title-row">
          <span
            v-if="item.highlightedText?.title"
            class="result-title"
            v-html="item.highlightedText.title"
          />
          <span v-else class="result-title">{{ item.doc.title }}</span>
          <span class="score-tag">Score: {{ item.score }} ({{ item.matchedField }})</span>
        </div>

        <div class="category-tag">{{ item.doc.category }}</div>

        <div
          v-if="item.highlightedText?.description"
          class="result-desc"
          v-html="item.highlightedText.description"
        />
        <div v-else class="result-desc">{{ item.doc.description }}</div>
      </li>
    </ul>
  </div>
</template>

<script setup lang="ts">
import { useSearch } from './useSearch';

interface ArticleDoc {
  id: string;
  title: string;
  category: string;
  description: string;
}

const SAMPLE_ARTICLES: ArticleDoc[] = [
  { id: '1', title: 'WebGPU Compute Pipelines', category: 'Graphics', description: 'WGSL compute shaders for high-throughput string matching.' },
  { id: '2', title: 'Unicode Canonical Decomposition', category: 'Text', description: 'Handling combining diacritics, astral surrogate pairs, and case folds.' },
  { id: '3', title: 'IndexedDB Snapshot Persistence', category: 'Storage', description: 'U2D3 Little-Endian binary format with CRC32 integrity verification.' },
  { id: '4', title: 'Vue 3 Reactivity and Composables', category: 'Framework', description: 'Deep dive into shallowRef, custom stores, and component lifecycles.' },
];

const {
  query,
  results,
  totalMatches,
  isSearching,
  isIndexing,
  isReady,
  error,
  stats,
  engine,
  fallbackReason,
  timings,
  mutationEpoch,
  add,
} = useSearch<ArticleDoc>({
  initialDocs: SAMPLE_ARTICLES,
  indexOptions: {
    fields: [
      { name: 'title', weight: 2.0 },
      { name: 'category', weight: 1.2 },
      { name: 'description', weight: 1.0 },
    ],
  },
  searchOptions: {
    mode: 'fuzzy',
    highlight: true,
    tag: 'mark',
    limit: 10,
  },
});

const handleAddDoc = async () => {
  await add({
    id: `doc-${Date.now()}`,
    title: `Dynamic Entry #${mutationEpoch.value + 1}`,
    category: 'Runtime',
    description: 'Added reactively via composable mutation method.',
  });
};
</script>

<style scoped>
.search-container {
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
.result-desc {
  font-size: 14px;
  color: #444;
}
</style>
