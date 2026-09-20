# Svelte Integration Recipe (`createDocumentSearch`)

Svelte reactive store recipe for `webgpu-search`. Conforms to Svelte's `Readable` store contract (`subscribe`), enabling reactive `$search` state access across Svelte 3, 4, and 5 with zero boilerplate.

---

## Features

- **Native Svelte Store Contract**: Works seamlessly with Svelte's auto-subscription `$store` syntax.
- **Off-Thread Web Worker Ready**: Offload candidate ranking to a dedicated Web Worker without locking the browser UI thread.
- **Built-In Query Debouncing**: Auto-cancels in-flight searches when the user types rapidly.
- **Dynamic Batched Mutations**: Add, update, or remove records on the fly.
- **Real-Time Observability**: Telemetry for engine state, CPU fallback reason, memory allocations, and query latencies.

---

## Installation

```bash
bun add webgpu-search
# or
npm install webgpu-search
```

---

## Basic Usage

```svelte
<script lang="ts">
  import { createDocumentSearch } from './documentSearchStore';

  interface PackageDoc {
    id: string;
    name: string;
    desc: string;
  }

  const search = createDocumentSearch<PackageDoc>({
    initialDocs: [
      { id: '1', name: 'svelte', desc: 'Cybernetically enhanced web apps' },
      { id: '2', name: 'webgpu-search', desc: 'Ultra-fast hybrid fuzzy search' },
    ],
    indexOptions: {
      fields: ['name', 'desc'],
    },
    searchOptions: {
      mode: 'fuzzy',
      highlight: true,
      tag: 'mark',
    },
  });
</script>

<input
  value={$search.query}
  on:input={(e) => search.setQuery(e.currentTarget.value)}
  placeholder="Search packages..."
/>

<div>Engine: {$search.engine} | Matches: {$search.totalMatches}</div>

<ul>
  {#each $search.results as item (item.id)}
    <li>
      {@html item.highlightedText?.name || item.doc.name}
      <small>{item.doc.desc}</small>
    </li>
  {/each}
</ul>
```
