import { DocumentIndex } from '../document-index';
import { restoreDocumentIndex } from '../snapshot-codec';
import type { WorkerMessageType } from '../types';
import {
  serializeError,
  type WorkerAbortPayload,
  type WorkerInitPayload,
  type WorkerMutatePayload,
  type WorkerRequest,
  type WorkerResponse,
  type WorkerRestorePayload,
  type WorkerSearchPayload,
  type WorkerSerializePayload
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
  !(typeof g.SharedWorkerGlobalScope !== 'undefined' && self instanceof g.SharedWorkerGlobalScope) &&
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
  if ((scope as any).__webgpu_search_worker_started) {
    return;
  }
  (scope as any).__webgpu_search_worker_started = true;

  let index: DocumentIndex<any> | null = null;
  let activeAbortController: AbortController | null = null;
  let latestQueryId: number = 0;

  type WorkerHandler = (req: WorkerRequest) => Promise<void> | void;

  /**
   * Table-driven worker dispatch: maps each WorkerMessageType to its handler.
   * Replaces the previous 8-case switch; responses are identical.
   */
  const handlers = new Map<WorkerMessageType, WorkerHandler>([
    [
      'INIT',
      async (req) => {
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
      }
    ],
    [
      'SEARCH',
      async (req) => {
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
          return;
        }

        if (queryId < latestQueryId) {
          // Superseded by newer query
          return;
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
            return;
          }

          // String-isolated enrichment: strip `doc` across thread boundary to eliminate structured-clone overhead
          if (payload.stringIsolated !== false && resp.results) {
            resp.results = resp.results.map((item) => ({ ...item, doc: undefined }));
          }

          scope.postMessage({
            id: req.id,
            success: true,
            result: resp
          } satisfies WorkerResponse);
        } catch (err: any) {
          if (signal.aborted || queryId < latestQueryId) {
            return;
          }
          scope.postMessage({
            id: req.id,
            success: false,
            error: serializeError(err)
          } satisfies WorkerResponse);
        }
      }
    ],
    [
      'ABORT',
      (req) => {
        const payload = req.payload as WorkerAbortPayload;
        if (payload && payload.queryId >= latestQueryId) {
          if (activeAbortController) {
            activeAbortController.abort();
            activeAbortController = null;
          }
        }
      }
    ],
    [
      'MUTATE',
      async (req) => {
        if (!index) {
          scope.postMessage({
            id: req.id,
            success: false,
            error: serializeError(
              new Error('[webgpu-search] Worker search index is not initialized.')
            )
          } satisfies WorkerResponse);
          return;
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
      }
    ],
    [
      'SERIALIZE',
      (req) => {
        if (!index) {
          scope.postMessage({
            id: req.id,
            success: false,
            error: serializeError(
              new Error('[webgpu-search] Worker search index is not initialized.')
            )
          } satisfies WorkerResponse);
          return;
        }
        try {
          const payload = req.payload as WorkerSerializePayload | undefined;
          const buf = index.serialize(payload?.options);
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
      }
    ],
    [
      'RESTORE',
      async (req) => {
        try {
          const payload = req.payload as WorkerRestorePayload;
          if (index) {
            index.destroy();
            index = null;
          }
          index = await restoreDocumentIndex(payload.buffer, payload.options);
          scope.postMessage({ id: req.id, success: true } satisfies WorkerResponse);
        } catch (err) {
          scope.postMessage({
            id: req.id,
            success: false,
            error: serializeError(err)
          } satisfies WorkerResponse);
        }
      }
    ],
    [
      'STATS',
      (req) => {
        if (!index) {
          scope.postMessage({
            id: req.id,
            success: false,
            error: serializeError(
              new Error('[webgpu-search] Worker search index is not initialized.')
            )
          } satisfies WorkerResponse);
          return;
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
      }
    ],
    [
      'DESTROY',
      (req) => {
        if (index) {
          index.destroy();
          index = null;
        }
        if (activeAbortController) {
          activeAbortController.abort();
          activeAbortController = null;
        }
        scope.postMessage({ id: req.id, success: true } satisfies WorkerResponse);
      }
    ]
  ]);

  const onMessage = async (event: MessageEvent) => {
    const req = event?.data as WorkerRequest;
    if (!req || typeof req.id !== 'number' || typeof req.type !== 'string') {
      return;
    }

    const handler = handlers.get(req.type as WorkerMessageType);
    if (!handler) {
      scope.postMessage({
        id: req.id,
        success: false,
        error: serializeError(
          new Error(`[webgpu-search] Unknown worker request type: ${(req as any).type}`)
        )
      } satisfies WorkerResponse);
      return;
    }
    await handler(req);
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
