# Vue 3 Integration Recipe (`useSearch`)

Idiomatic Vue 3 Composition API recipe for `webgpu-search`. Provides reactive `ref` and `shallowRef` bindings, query debouncing, live dynamic mutations, off-thread Web Worker delegation, and real-time observability telemetry.

---

## Features

- **Vue 3 Composition API**: Fully typed `ref`, `shallowRef`, and `watch` primitives that integrate seamlessly into `<script setup lang="ts">`.
- **Automatic Debounced Watcher**: Watches `query` changes and dispatches searches with configurable debounce delays, auto-aborting superseded queries.
- **Dedicated Web Worker Support**: Move heavy token processing and GPU dispatch to a Web Worker via `SearchWorkerClient` without blocking the Vue reactivity loop.
- **Reactive Telemetry**: Live engine status (`webgpu` / `cpu`), fallback reasons, memory metrics (`vramBytes`, `ramBytes`), and query latency timings.
- **Highlighting**: Seamless HTML mark generation with `v-html`.

---

## Installation

```bash
bun add webgpu-search
# or
npm install webgpu-search
```

---

## Usage in `<script setup lang="ts">`

```vue
<script setup lang="ts">
import { useSearch } from './useSearch';

interface BookDoc {
  id: string;
  title: string;
  author: string;
}

const books: BookDoc[] = [
  { id: '1', title: 'The Rust Programming Language', author: 'Steve Klabnik' },
  { id: '2', title: 'Designing Data-Intensive Applications', author: 'Martin Kleppmann' },
  { id: '3', title: 'WebGPU Handbook', author: 'GPU Community' },
];

const { query, results, totalMatches, isSearching, engine, timings } = useSearch<BookDoc>({
  initialDocs: books,
  indexOptions: {
    fields: [
      { name: 'title', weight: 2.0 },
      { name: 'author', weight: 1.0 },
    ],
  },
  searchOptions: {
    mode: 'fuzzy',
    highlight: true,
    tag: 'mark',
    escapeHtml: true,
  },
});
</script>

<template>
  <div class="search-box">
    <input v-model="query" placeholder="Search books..." />
    <div v-if="isSearching">Scoring on {{ engine }}...</div>
    <ul>
      <li v-for="item in results" :key="item.id">
        <span v-if="item.highlightedText?.title" v-html="item.highlightedText.title" />
        <span v-else>{{ item.doc.title }}</span>
        <small>({{ item.doc.author }})</small>
      </li>
    </ul>
  </div>
</template>
```

---

## Nuxt 3 & Web Worker Setup

When using Nuxt 3 with SSR, instantiate the search client on the client-side inside `onMounted` or guard with `import.meta.client`:

```ts
const { query, results } = useSearch<BookDoc>({
  worker: () => new Worker(new URL('./search.worker.ts', import.meta.url), { type: 'module' }),
  initialDocs: myBooks,
  indexOptions: { fields: ['title', 'author'] },
});
```
