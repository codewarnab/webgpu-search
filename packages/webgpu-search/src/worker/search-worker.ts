import { DocumentIndex } from '../document-index';
import { decodeSnapshotHeader, restoreDocumentIndex } from '../snapshot-codec';
import type { WorkerMessageType } from '../types';
import {
  serializeError,
  type WorkerAbortPayload,
  type WorkerInitPayload,
  type WorkerMutatePayload,
  type WorkerRequest,
  type WorkerResponse,
  type WorkerRestorePayload,
  type WorkerRestoreResult,
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
  let activeQueryId: number = 0;
  // Highest SEARCH queryId seen (updated on arrival, before queueing).
  let latestQueryId: number = 0;
  // Every queryId <= abortedUpTo was cancelled by the client.
  let abortedUpTo: number = -1;

  /**
   * Message serialization (reader/writer queue). State-mutating requests
   * (INIT / RESTORE / MUTATE / DESTROY) run exclusively, after every earlier
   * request has settled. Read requests (SEARCH / SERIALIZE / STATS) wait only
   * for earlier exclusive requests, so concurrent searches keep their
   * supersede/abort semantics. ABORT is handled immediately on arrival.
   */
  let allTail: Promise<unknown> = Promise.resolve();
  let exclusiveTail: Promise<unknown> = Promise.resolve();
  const EXCLUSIVE_TYPES: ReadonlySet<string> = new Set(['INIT', 'RESTORE', 'MUTATE', 'DESTROY']);

  /**
   * Session epoch, bumped when DESTROY arrives. Requests are tagged with the
   * epoch they arrived in; replies (and not-yet-started work) from an older
   * epoch are dropped so a reused worker never answers a new client's
   * request ids with a previous session's results.
   */
  let sessionEpoch = 0;
  const requestEpoch = new WeakMap<object, number>();
  const reply = (req: WorkerRequest, msg: WorkerResponse, transfer?: Transferable[]): void => {
    if (requestEpoch.get(req) !== sessionEpoch) {
      return;
    }
    if (transfer) {
      scope.postMessage(msg, transfer);
    } else {
      scope.postMessage(msg);
    }
  };

  const swapIndex = (next: DocumentIndex<any> | null): void => {
    const prev = index;
    index = next;
    if (prev && prev !== next) {
      prev.destroy();
    }
  };

  const abortActiveSearch = (): void => {
    if (activeAbortController) {
      activeAbortController.abort();
      activeAbortController = null;
    }
  };

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
          const payload = req.payload as WorkerInitPayload<any>;
          const records = Array.isArray(payload?.records) ? payload.records : [];
          // Build-then-swap: a failed INIT leaves the previous index live.
          const next = await DocumentIndex.create(records, payload?.options ?? { fields: [] });
          swapIndex(next);
          reply(req, { id: req.id, success: true } satisfies WorkerResponse);
        } catch (err) {
          reply(req, {
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
          reply(req, {
            id: req.id,
            success: false,
            error: serializeError(
              new Error('[webgpu-search] Worker search index is not initialized. Call init() first.')
            )
          } satisfies WorkerResponse);
          return;
        }

        if (queryId < latestQueryId || queryId <= abortedUpTo) {
          // Superseded by a newer query or cancelled while queued.
          return;
        }

        abortActiveSearch();
        const controller = new AbortController();
        activeAbortController = controller;
        activeQueryId = queryId;
        const signal = controller.signal;

        try {
          const resp = await index.search(payload.query, {
            ...payload.options,
            signal
          });

          if (activeAbortController === controller) {
            activeAbortController = null;
          }
          if (signal.aborted || queryId < latestQueryId) {
            return;
          }

          // String-isolated enrichment: strip `doc` across thread boundary to eliminate structured-clone overhead
          if (payload.stringIsolated !== false && resp.results) {
            resp.results = resp.results.map((item) => ({ ...item, doc: undefined }));
          }

          reply(req, {
            id: req.id,
            success: true,
            result: resp
          } satisfies WorkerResponse);
        } catch (err: any) {
          if (activeAbortController === controller) {
            activeAbortController = null;
          }
          if (signal.aborted || queryId < latestQueryId) {
            return;
          }
          reply(req, {
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
        const queryId = payload?.queryId;
        if (typeof queryId !== 'number' || !Number.isFinite(queryId)) {
          return;
        }
        if (queryId > abortedUpTo) {
          abortedUpTo = queryId;
        }
        if (activeAbortController && activeQueryId <= queryId) {
          abortActiveSearch();
        }
      }
    ],
    [
      'MUTATE',
      async (req) => {
        if (!index) {
          reply(req, {
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
          reply(req, {
            id: req.id,
            success: true,
            result: res
          } satisfies WorkerResponse);
        } catch (err) {
          reply(req, {
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
          reply(req, {
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
          reply(req, { id: req.id, success: true, result: buf } satisfies WorkerResponse, [buf]);
        } catch (err) {
          reply(req, {
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
          const hasCallerDocs = Array.isArray(payload?.options?.documents);
          // Validate-then-swap: a corrupt snapshot leaves the previous index live.
          const header = decodeSnapshotHeader(payload?.buffer);
          const next = await restoreDocumentIndex(payload?.buffer, payload?.options);
          let result: WorkerRestoreResult;
          try {
            result = {
              fields: next.getSortedFields().map((f) => ({ name: f.name, weight: f.weight })),
              filterFields: next.getFilterFieldDefinitions().map((ff) => ({
                name: ff.name as string,
                ...(ff.type !== undefined ? { type: ff.type } : {})
              })),
              ...(typeof next.options.idField === 'string' ? { idField: next.options.idField } : {}),
              ...(header.docsByteLength > 0 && !hasCallerDocs ? { records: next.getRecords() } : {})
            };
          } catch (err) {
            next.destroy();
            throw err;
          }
          swapIndex(next);
          reply(req, { id: req.id, success: true, result } satisfies WorkerResponse);
        } catch (err) {
          reply(req, {
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
          reply(req, {
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
          reply(req, {
            id: req.id,
            success: true,
            result: stats
          } satisfies WorkerResponse);
        } catch (err) {
          reply(req, {
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
        swapIndex(null);
        abortActiveSearch();
        reply(req, { id: req.id, success: true } satisfies WorkerResponse);
      }
    ]
  ]);

  const runGuarded = (handler: WorkerHandler, req: WorkerRequest) => async (): Promise<void> => {
    if (requestEpoch.get(req) !== sessionEpoch) {
      return; // Superseded by a later DESTROY before it started.
    }
    try {
      await handler(req);
    } catch (err) {
      reply(req, {
        id: req.id,
        success: false,
        error: serializeError(err)
      } satisfies WorkerResponse);
    }
  };

  const onMessage = (event: MessageEvent): Promise<void> => {
    const req = event?.data as WorkerRequest;
    if (!req || typeof req.id !== 'number' || typeof req.type !== 'string') {
      return Promise.resolve();
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
      return Promise.resolve();
    }

    // Never let a rejected run (e.g. reply() throwing DataCloneError) poison
    // the queue tails and silently skip every later request.
    const guarded = runGuarded(handler, req);
    const run = (): Promise<void> => guarded().catch(() => undefined);

    if (req.type === 'SEARCH') {
      // Supersede on arrival so a newer query cancels the in-flight one even
      // while it waits behind a queued state mutation.
      const queryId = (req.payload as WorkerSearchPayload<any> | undefined)?.queryId ?? 0;
      if (queryId > latestQueryId) {
        latestQueryId = queryId;
        if (activeAbortController && activeQueryId < queryId) {
          abortActiveSearch();
        }
      }
    } else if (req.type === 'DESTROY') {
      // Cancel in-flight work early; the index teardown itself is queued
      // behind any pending INIT/RESTORE so it cannot be undone by them.
      // Query sequencing restarts so a reused (non-owned) worker accepts a
      // fresh client's queryIds.
      abortActiveSearch();
      latestQueryId = 0;
      abortedUpTo = -1;
      sessionEpoch++;
    }

    requestEpoch.set(req, sessionEpoch);

    if (req.type === 'ABORT') {
      return run();
    }

    if (EXCLUSIVE_TYPES.has(req.type)) {
      const p = allTail.then(run);
      allTail = p;
      exclusiveTail = p;
      return p;
    }
    const p = exclusiveTail.then(run);
    allTail = Promise.all([allTail, p]);
    return p;
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
