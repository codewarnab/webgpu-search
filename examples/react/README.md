# React Integration Recipe (`useDocumentSearch`)

Production-ready React integration recipe for `webgpu-search`. Provides an ergonomic, non-blocking hook with automatic query debouncing, race condition elimination, dynamic mutations, Unicode-exact highlight rendering, and observability telemetry.

---

## Features

- **WebGPU Acceleration with Graceful CPU Fallback**: Automatically harnesses client WebGPU compute pipelines and falls back cleanly to the multi-field reference CPU scorer on unsupported hardware or small corpora.
- **Dedicated Web Worker Support**: Run tokenization, dispatch, and ranking completely off the main thread via `SearchWorkerClient` with zero UI freezing.
- **Race Condition Immunity**: Monotonic query counters and `AbortController` cancellation ensure slow earlier queries never overwrite fresh results.
- **Live Dynamic Mutations**: Incremental batched `add`, `update`, and `remove` methods update the search state and mutation epoch without reloading or reallocating whole datasets.
- **Unicode-Exact Highlighting**: Highlight ranges and `<mark>` tags map accurately to original UTF-16 strings, preserving indentation and grapheme clusters.
- **Real-Time Observability**: Exposes engine type (`webgpu` vs `cpu`), fallback reasons, memory metrics (`vramBytes`, `ramBytes`, `totalBytes`), and query latency timings.

---

## Installation

```bash
bun add webgpu-search
# or
npm install webgpu-search
```

---

## Basic In-Memory Usage

```tsx
import React from 'react';
import { useDocumentSearch } from './useDocumentSearch';

interface FileItem {
  id: string;
  name: string;
  path: string;
}

const files: FileItem[] = [
  { id: '1', name: 'App.tsx', path: 'src/App.tsx' },
  { id: '2', name: 'SearchWorker.ts', path: 'src/workers/SearchWorker.ts' },
  { id: '3', name: 'fuzzy.wgsl', path: 'src/shaders/fuzzy.wgsl' },
];

export function FilePalette() {
  const { query, setQuery, results, timings, engine } = useDocumentSearch<FileItem>({
    initialDocs: files,
    indexOptions: {
      fields: [
        { name: 'name', weight: 2.0 },
        { name: 'path', weight: 1.0 },
      ],
    },
    searchOptions: {
      mode: 'fuzzy',
      highlight: true,
      tag: 'mark',
      escapeHtml: true,
      limit: 20,
    },
  });

  return (
    <div>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="QuickOpen file..."
      />

      <div>
        Engine: {engine} | Latency: {timings?.totalMs.toFixed(2)} ms
      </div>

      <ul>
        {results.map((hit) => (
          <li key={String(hit.id)}>
            {hit.highlightedText?.name ? (
              <div dangerouslySetInnerHTML={{ __html: hit.highlightedText.name }} />
            ) : (
              <div>{hit.doc.name}</div>
            )}
            <small>{hit.doc.path}</small>
          </li>
        ))}
      </ul>
    </div>
  );
}
```

---

## Off-Thread Web Worker Usage (Recommended for Large Corpora)

To search 50,000+ records off the main thread, pass a worker factory:

```tsx
const { query, setQuery, results } = useDocumentSearch<FileItem>({
  worker: () => new Worker(new URL('./search.worker.ts', import.meta.url), { type: 'module' }),
  initialDocs: largeCorpus,
  indexOptions: {
    fields: ['title', 'content'],
  },
});
```

Where `search.worker.ts` contains:

```ts
import { startSearchWorker } from 'webgpu-search/worker';
startSearchWorker();
```

---

## Dynamic Mutations & Observability

```tsx
const { add, remove, stats, mutationEpoch } = useDocumentSearch<Doc>({ ... });

// Add documents at runtime
await add({ id: 'doc-99', title: 'New Item' });

// Inspect memory and hardware allocations
console.log('VRAM bytes:', stats?.memory.vramBytes);
console.log('RAM bytes:', stats?.memory.ramBytes);
console.log('Fallback Reason:', stats?.fallbackReason);
console.log('Mutation Epoch:', mutationEpoch);
```
