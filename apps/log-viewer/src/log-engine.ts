import {
  DocumentIndex,
  SearchWorkerClient,
  saveIndexToIDB,
  loadIndexFromIDB,
  serializeDocumentIndex,
  restoreDocumentIndex,
  deleteIndexFromIDB,
  type DocumentIndexStats,
  type DocumentSearchResponse,
  type MutationResult
} from 'webgpu-search';
import type { LogViewerSearchResult, StructuredLogRecord } from './types';

export interface LogEngineOptions {
  useWorker?: boolean;
  preferGpu?: boolean;
}

export const IDB_DATABASE_NAME = 'webgpu-log-viewer-snapshots';
export const IDB_SNAPSHOT_KEY = 'latest-100k-snapshot';

export class LogEngine {
  private mainIndex: DocumentIndex<StructuredLogRecord> | null = null;
  private workerClient: SearchWorkerClient<StructuredLogRecord> | null = null;
  private workerInstance: Worker | null = null;
  private records: StructuredLogRecord[] = [];
  private useWorker: boolean = false;
  private preferGpu: boolean = true;
  private isDestroyed: boolean = false;
  private rebuildGeneration: number = 0;

  private readonly fieldDefs = [
    { name: 'message', weight: 2.0 },
    { name: 'service', weight: 1.5 },
    { name: 'level', weight: 1.0 },
    { name: 'traceId', weight: 1.2 }
  ];

  constructor(options?: LogEngineOptions) {
    this.useWorker = options?.useWorker ?? false;
    this.preferGpu = options?.preferGpu ?? true;
  }

  async init(records: StructuredLogRecord[]): Promise<void> {
    this.records = [...records];
    await this.rebuildIndex();
  }

  private async rebuildIndex(): Promise<void> {
    if (this.isDestroyed) return;
    const currentGen = ++this.rebuildGeneration;

    if (this.mainIndex) {
      this.mainIndex.destroy();
      this.mainIndex = null;
    }
    if (this.workerClient) {
      await this.workerClient.destroy().catch(() => {});
      this.workerClient = null;
    }
    if (this.workerInstance) {
      this.workerInstance.terminate();
      this.workerInstance = null;
    }

    if (this.useWorker && typeof Worker !== 'undefined') {
      try {
        const worker = new Worker(
          new URL('./worker.ts', import.meta.url),
          { type: 'module' }
        );
        const client = new SearchWorkerClient<StructuredLogRecord>({
          worker
        });

        await client.init(this.records, {
          idField: 'id',
          fields: this.fieldDefs,
          preferGpu: this.preferGpu,
          candidateCapacity: 32768
        });

        if (this.isDestroyed || this.rebuildGeneration !== currentGen) {
          await client.destroy().catch(() => {});
          worker.terminate();
          return;
        }

        this.workerInstance = worker;
        this.workerClient = client;
        return;
      } catch (err) {
        console.warn('[LogEngine] Worker init failed, using main thread:', err);
        this.useWorker = false;
        if (this.workerInstance) {
          this.workerInstance.terminate();
          this.workerInstance = null;
        }
        this.workerClient = null;
      }
    }

    const mainIdx = await DocumentIndex.create(this.records, {
      idField: 'id',
      fields: this.fieldDefs,
      preferGpu: this.preferGpu,
      candidateCapacity: 32768
    });

    if (this.isDestroyed || this.rebuildGeneration !== currentGen) {
      mainIdx.destroy();
      return;
    }

    this.mainIndex = mainIdx;
  }

  async setUseWorker(useWorker: boolean): Promise<void> {
    if (this.useWorker === useWorker) return;
    this.useWorker = useWorker;
    await this.rebuildIndex();
  }

  async setPreferGpu(preferGpu: boolean): Promise<void> {
    if (this.preferGpu === preferGpu) return;
    this.preferGpu = preferGpu;
    await this.rebuildIndex();
  }

  async search(
    query: string,
    options?: {
      mode?: 'fuzzy' | 'substring';
      highlight?: boolean;
      limit?: number;
      levelFilter?: string;
      signal?: AbortSignal;
    }
  ): Promise<{
    results: LogViewerSearchResult[];
    totalMatches: number;
    searchDurationMs: number;
    engine: 'webgpu' | 'cpu';
    hasOverflow: boolean;
  }> {
    const start = performance.now();
    const mode = options?.mode ?? 'fuzzy';
    const highlight = options?.highlight ?? true;
    const limit = options?.limit ?? 100;
    const levelFilter = options?.levelFilter;

    const filter = levelFilter && levelFilter !== 'ALL'
      ? (doc: StructuredLogRecord) => doc.level === levelFilter
      : undefined;

    let response: DocumentSearchResponse<StructuredLogRecord>;

    if (this.useWorker && this.workerClient) {
      response = await this.workerClient.search(query, {
        mode,
        highlight,
        tag: 'mark',
        escapeHtml: true,
        limit,
        filter,
        signal: options?.signal,
        candidateCapacity: 32768
      } as any);
    } else if (this.mainIndex) {
      response = await this.mainIndex.search(query, {
        mode,
        highlight,
        tag: 'mark',
        escapeHtml: true,
        limit,
        filter,
        signal: options?.signal
      });
    } else {
      return {
        results: [],
        totalMatches: 0,
        searchDurationMs: 0,
        engine: 'cpu',
        hasOverflow: false
      };
    }

    const duration = performance.now() - start;

    const formatted: LogViewerSearchResult[] = response.results.map((item) => ({
      id: String(item.id),
      score: item.score,
      matchedField: item.matchedField,
      doc: item.doc,
      highlightedText: item.highlightedText,
      highlights: item.highlights
    }));

    return {
      results: formatted,
      totalMatches: response.totalMatches,
      searchDurationMs: duration,
      engine: response.engine,
      hasOverflow: response.hasOverflow
    };
  }

  async appendLogs(newLogs: StructuredLogRecord[]): Promise<MutationResult | null> {
    this.records.push(...newLogs);
    if (this.useWorker && this.workerClient) {
      return await this.workerClient.add(newLogs);
    } else if (this.mainIndex) {
      return await this.mainIndex.add(newLogs);
    }
    return null;
  }

  async removeOldLogs(count: number): Promise<MutationResult | null> {
    if (count <= 0 || this.records.length === 0) return null;
    const toRemove = this.records.slice(0, count);
    this.records = this.records.slice(count);
    const ids = toRemove.map((r) => r.id);

    if (this.useWorker && this.workerClient) {
      return await this.workerClient.remove(ids);
    } else if (this.mainIndex) {
      return await this.mainIndex.remove(ids);
    }
    return null;
  }

  async saveSnapshotToIDB(): Promise<{ byteLength: number; durationMs: number }> {
    const start = performance.now();
    let byteLength = 0;

    if (this.useWorker && this.workerClient) {
      const buf = await this.workerClient.serialize();
      byteLength = buf.byteLength;
      await saveIndexToIDB(buf, {
        dbName: IDB_DATABASE_NAME,
        key: IDB_SNAPSHOT_KEY
      });
    } else if (this.mainIndex) {
      const buf = serializeDocumentIndex(this.mainIndex);
      byteLength = buf.byteLength;
      await saveIndexToIDB(buf, {
        dbName: IDB_DATABASE_NAME,
        key: IDB_SNAPSHOT_KEY
      });
    }

    return {
      byteLength,
      durationMs: performance.now() - start
    };
  }

  async restoreSnapshotFromIDB(): Promise<{ recordCount: number; durationMs: number }> {
    const start = performance.now();

    const loaded = await loadIndexFromIDB<StructuredLogRecord>({
      dbName: IDB_DATABASE_NAME,
      key: IDB_SNAPSHOT_KEY
    });

    if (!loaded) {
      throw new Error('No snapshot found in IndexedDB');
    }

    await this.restoreSnapshot(loaded.snapshot);

    return {
      recordCount: this.records.length,
      durationMs: performance.now() - start
    };
  }

  async clearIDB(): Promise<void> {
    await deleteIndexFromIDB({ dbName: IDB_DATABASE_NAME, key: IDB_SNAPSHOT_KEY }).catch(() => {});
  }

  async serializeSnapshot(): Promise<ArrayBuffer> {
    if (this.useWorker && this.workerClient) {
      return await this.workerClient.serialize();
    } else if (this.mainIndex) {
      return serializeDocumentIndex(this.mainIndex);
    }
    throw new Error('Index not initialized');
  }

  async restoreSnapshot(buffer: ArrayBuffer): Promise<void> {
    if (this.useWorker && this.workerClient) {
      await this.workerClient.restore(buffer);
      this.records = this.workerClient.getRecords();
    } else {
      const restored = await restoreDocumentIndex<StructuredLogRecord>(buffer, {
        options: {
          preferGpu: this.preferGpu,
          candidateCapacity: 32768
        }
      });
      if (this.mainIndex) this.mainIndex.destroy();
      this.mainIndex = restored;
      this.records = restored.getRecords();
    }
  }

  async getStats(): Promise<DocumentIndexStats | null> {
    if (this.useWorker && this.workerClient) {
      return await this.workerClient.getStats();
    } else if (this.mainIndex) {
      return this.mainIndex.getStats();
    }
    return null;
  }

  getRecords(): StructuredLogRecord[] {
    return this.records;
  }

  destroy(): void {
    this.isDestroyed = true;
    if (this.mainIndex) {
      this.mainIndex.destroy();
      this.mainIndex = null;
    }
    if (this.workerClient) {
      this.workerClient.destroy().catch(() => {});
      this.workerClient = null;
    }
    if (this.workerInstance) {
      this.workerInstance.terminate();
      this.workerInstance = null;
    }
  }
}
