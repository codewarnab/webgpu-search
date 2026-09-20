# Framework Integration Recipes

This directory contains official framework recipes and patterns for integrating `webgpu-search` into modern web applications:

| Recipe | Directory | Primary APIs | Use Case |
|---|---|---|---|
| **React** | [`examples/react/`](./react) | `useDocumentSearch`, `SearchComponent` | React 18 / 19, Next.js, Remix, Vite React |
| **Vue 3** | [`examples/vue/`](./vue) | `useSearch`, `SearchBox.vue` | Vue 3, Nuxt 3, Vite Vue |
| **Svelte** | [`examples/svelte/`](./svelte) | `createDocumentSearch`, `SearchBox.svelte` | Svelte 4 / 5, SvelteKit, Vite Svelte |
| **Vanilla** | [`examples/vanilla/`](./vanilla) | `createVanillaSearchApp`, `index.html` | Vanilla HTML/TS, Web Components, Electron |

---

## Architectural Principles

1. **Zero Runtime Coupling**: `packages/webgpu-search` published to npm has **0 runtime framework dependencies**. These recipes demonstrate how to wrap the core engine in idiomatic reactive primitives.
2. **Worker-First Architecture**: Every recipe supports running off-thread in a Web Worker via `SearchWorkerClient` to guarantee zero UI thread stutter.
3. **Observability & Telemetry**: Every recipe surfaces real-time metrics: active engine (`webgpu` vs `cpu`), fallback reason, memory allocations (`vramBytes`, `ramBytes`), and query latency timings.
4. **Clean Lifecycle & Abort Safety**: Queries in-flight when a user types rapidly are automatically cancelled using `AbortController` and monotonic request IDs. Resources are destroyed on component unmount.
