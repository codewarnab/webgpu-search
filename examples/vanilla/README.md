# Vanilla JavaScript / TypeScript Integration Recipe

Zero-framework integration recipe for embedding `webgpu-search` directly into vanilla HTML/TS applications, Web Components, Electron, or browser extensions.

---

## Features

- **Zero Framework Runtime Dependencies**: Direct DOM bindings without React, Vue, or Svelte.
- **Embedded Telemetry HUD**: Shows live engine state, CPU fallback reason, VRAM/RAM allocation, latency, and mutation epoch.
- **Dynamic In-Place Mutations**: Reactive additions and deletions without rebuilding the entire index.
- **Lifecycle Teardown**: Clean `destroy()` method that removes event listeners and releases GPU buffers.

---

## Installation

```bash
bun add webgpu-search
# or
npm install webgpu-search
```

---

## Usage

```ts
import { createVanillaSearchApp } from './search-app';

const container = document.getElementById('search-app');

const app = createVanillaSearchApp(container, {
  initialDocs: [
    { id: '1', title: 'Getting Started', description: 'Quickstart guide for webgpu-search' },
    { id: '2', title: 'Architecture', description: 'Deep dive into GPU buffer layouts' },
  ],
  indexOptions: {
    fields: ['title', 'description'],
  },
  searchOptions: {
    mode: 'fuzzy',
    highlight: true,
    tag: 'mark',
  },
});

// Tear down when container is removed
// app.destroy();
```
