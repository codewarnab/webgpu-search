/**
 * Dedicated Web Worker entrypoint for webgpu-search (webgpu-search/worker).
 *
 * SSR & Main-Thread Safe: guards self, importScripts, and postMessage to avoid
 * executing on the main thread, Node.js, or SSR environments.
 */

export function startSearchWorker(): void {
  // Scaffolding for v0.3 worker message listener (M5 implementation)
}

export const isDedicatedWorker =
  typeof self !== 'undefined' &&
  typeof (self as any).importScripts === 'function' &&
  typeof (self as any).postMessage === 'function';

if (isDedicatedWorker) {
  startSearchWorker();
}
