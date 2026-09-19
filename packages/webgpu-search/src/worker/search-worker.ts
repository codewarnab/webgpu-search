import { DocumentIndex } from '../document-index';
import {
  serializeError,
  type WorkerAbortPayload,
  type WorkerInitPayload,
  type WorkerMutatePayload,
  type WorkerRequest,
  type WorkerResponse,
  type WorkerRestorePayload,
  type WorkerSearchPayload
} from './protocol';

/**
 * Dedicated Web Worker entrypoint for webgpu-search (webgpu-search/worker).
 *
 * SSR & Main-Thread Safe: guards self, postMessage, DedicatedWorkerGlobalScope,
 * and importScripts to avoid executing on the main thread, Node.js, or SSR environments.
 */

const g = globalThis as any;

export const isDedicatedWorker =
  typeof self !== 'undefined' &&
  typeof (self as any).postMessage === 'function' &&
  (
    typeof (self as any).importScripts === 'function' ||
    (typeof g.DedicatedWorkerGlobalScope !== 'undefined' && self instanceof g.DedicatedWorkerGlobalScope) ||
    (typeof g.Bun !== 'undefined' && g.Bun.isMainThread === false)
  );

export function startSearchWorker(customScope?: any): void {
  const scope = customScope !== undefined ? customScope : (isDedicatedWorker && typeof self !== 'undefined' ? self : undefined);
  if (!scope || typeof scope.postMessage !== 'function') {
    return;
  }

  let index: DocumentIndex<any> | null = null;
  let activeAbortController: AbortController | null = null;
  let latestQueryId: number = 0;

  const onMessage = async (event: MessageEvent) => {
    const req = event?.data as WorkerRequest;
    if (!req || typeof req.id !== 'number' || typeof req.type !== 'string') {
      return;
    }

    switch (req.type) {
      case 'INIT': {
        try {
          if (index) {
            index.destroy();
            index = null;
          }
          const payload = req.payload as WorkerInitPayload<any>;
          const records = Array.isArray(payload?.records) ? payload.records : [];
          index = await DocumentIndex.create(records, payload?.options ?? { fields: [] });
          scope.postMessage({ id: req.id, success: true } satisfies WorkerResponse);
        } catch (err) {
          scope.postMessage({
            id: req.id,
            success: false,
            error: serializeError(err)
          } satisfies WorkerResponse);
        }
        break;
      }

      case 'SEARCH': {
        const payload = req.payload as WorkerSearchPayload<any>;
        const queryId = payload?.queryId ?? 0;

        if (!index) {
          scope.postMessage({
            id: req.id,
            success: false,
            error: serializeError(
              new Error('[webgpu-search] Worker search index is not initialized. Call init() first.')
            )
          } satisfies WorkerResponse);
          break;
        }

        if (queryId < latestQueryId) {
          // Superseded by newer query
          break;
        }
        latestQueryId = queryId;

        if (activeAbortController) {
          activeAbortController.abort();
        }
        activeAbortController = new AbortController();
        const signal = activeAbortController.signal;

        try {
          const resp = await index.search(payload.query, {
            ...payload.options,
            signal
          });

          if (signal.aborted || queryId < latestQueryId) {
            break;
          }

          // String-isolated enrichment: strip `doc` across thread boundary to eliminate structured-clone overhead
          if (payload.stringIsolated !== false && resp.results) {
            for (let i = 0; i < resp.results.length; i++) {
              (resp.results[i] as any).doc = undefined;
            }
          }

          scope.postMessage({
            id: req.id,
            success: true,
            result: resp
          } satisfies WorkerResponse);
        } catch (err: any) {
          if (signal.aborted || queryId < latestQueryId || err?.name === 'AbortError') {
            break;
          }
          scope.postMessage({
            id: req.id,
            success: false,
            error: serializeError(err)
          } satisfies WorkerResponse);
        }
        break;
      }

      case 'ABORT': {
        const payload = req.payload as WorkerAbortPayload;
        if (payload && payload.queryId >= latestQueryId) {
          if (activeAbortController) {
            activeAbortController.abort();
            activeAbortController = null;
          }
        }
        break;
      }

      case 'MUTATE': {
        if (!index) {
          scope.postMessage({
            id: req.id,
            success: false,
            error: serializeError(
              new Error('[webgpu-search] Worker search index is not initialized.')
            )
          } satisfies WorkerResponse);
          break;
        }
        try {
          const payload = req.payload as WorkerMutatePayload<any>;
          const res = await index.applyBatch(payload.batch, payload.options);
          scope.postMessage({
            id: req.id,
            success: true,
            result: res
          } satisfies WorkerResponse);
        } catch (err) {
          scope.postMessage({
            id: req.id,
            success: false,
            error: serializeError(err)
          } satisfies WorkerResponse);
        }
        break;
      }

      case 'SERIALIZE': {
        if (!index) {
          scope.postMessage({
            id: req.id,
            success: false,
            error: serializeError(
              new Error('[webgpu-search] Worker search index is not initialized.')
            )
          } satisfies WorkerResponse);
          break;
        }
        try {
          const buf = index.serialize();
          scope.postMessage(
            { id: req.id, success: true, result: buf } satisfies WorkerResponse,
            [buf]
          );
        } catch (err) {
          scope.postMessage({
            id: req.id,
            success: false,
            error: serializeError(err)
          } satisfies WorkerResponse);
        }
        break;
      }

      case 'RESTORE': {
        if (!index) {
          scope.postMessage({
            id: req.id,
            success: false,
            error: serializeError(
              new Error('[webgpu-search] Worker search index is not initialized.')
            )
          } satisfies WorkerResponse);
          break;
        }
        try {
          const payload = req.payload as WorkerRestorePayload;
          index.restore(payload.buffer, payload.options);
          scope.postMessage({ id: req.id, success: true } satisfies WorkerResponse);
        } catch (err) {
          scope.postMessage({
            id: req.id,
            success: false,
            error: serializeError(err)
          } satisfies WorkerResponse);
        }
        break;
      }

      case 'STATS': {
        if (!index) {
          scope.postMessage({
            id: req.id,
            success: false,
            error: serializeError(
              new Error('[webgpu-search] Worker search index is not initialized.')
            )
          } satisfies WorkerResponse);
          break;
        }
        try {
          const stats = index.getStats();
          scope.postMessage({
            id: req.id,
            success: true,
            result: stats
          } satisfies WorkerResponse);
        } catch (err) {
          scope.postMessage({
            id: req.id,
            success: false,
            error: serializeError(err)
          } satisfies WorkerResponse);
        }
        break;
      }

      case 'DESTROY': {
        if (index) {
          index.destroy();
          index = null;
        }
        if (activeAbortController) {
          activeAbortController.abort();
          activeAbortController = null;
        }
        scope.postMessage({ id: req.id, success: true } satisfies WorkerResponse);
        break;
      }

      default:
        break;
    }
  };

  if (typeof scope.addEventListener === 'function') {
    scope.addEventListener('message', onMessage);
  } else {
    scope.onmessage = onMessage;
  }
}

if (isDedicatedWorker) {
  startSearchWorker();
}
