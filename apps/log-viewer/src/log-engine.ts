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
  type FacetRequest,
  type FacetResult,
  type FilterExpression,
  type MutationResult
} from 'webgpu-search';
import type { LogViewerSearchResult, StructuredLogRecord } from './types';

export interface LogEngineOptions {
  useWorker?: boolean;
  preferGpu?: boolean;
}

export const IDB_DATABASE_NAME = 'webgpu-log-viewer-snapshots';
export const IDB_SNAPSHOT_KEY = 'latest-100k-snapshot';

export interface LogSearchOptions {
  mode?: 'fuzzy' | 'substring' | 'prefix' | 'token';
  highlight?: boolean;
  limit?: number;
  levelFilter?: string;
  serviceFilter?: string;
  /** Half-open latency window [minMs, maxMs). */
  latencyRange?: { minMs?: number; maxMs?: number };
  /** Half-open ISO timestamp window [from, to). Lexicographic = chronological for ISO-8601. */
  timestampRange?: { from?: string; to?: string };
  /** Facet requests; defaults to level terms + latency ranges. `false` disables. */
  facets?: Record<string, FacetRequest> | false;
  signal?: AbortSignal;
}

export interface LogSearchResult {
  results: LogViewerSearchResult[];
  totalMatches: number;
  searchDurationMs: number;
  engine: 'webgpu' | 'cpu';
  hasOverflow: boolean;
  facets?: Record<string, FacetResult>;
}

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

  /**
   * columnar filter attributes. `timestamp` stays a string so ISO-8601
   * lexicographic range compares equal chronological order on both the main
   * thread and the string-isolated worker (no custom getter crosses the
   * worker boundary).
   */
  private readonly filterFieldDefs = [
    { name: 'level', type: 'string' as const },
    { name: 'service', type: 'string' as const },
    { name: 'timestamp', type: 'string' as const },
    { name: 'latencyMs', type: 'number' as const }
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
          filterFields: this.filterFieldDefs,
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
      filterFields: this.filterFieldDefs,
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

  /**
   * compiles level/service/latency/timestamp constraints into a
   * single structured `FilterExpression` evaluated via columnar bitsets
   * (pre-match, O(N/32) bitwise) instead of per-record predicates.
   */
  buildFilter(options?: LogSearchOptions): FilterExpression | undefined {
    const clauses: FilterExpression[] = [];
    const levelFilter = options?.levelFilter;
    if (levelFilter !== undefined && levelFilter !== 'ALL') {
      clauses.push({ level: levelFilter });
    }
    const serviceFilter = options?.serviceFilter;
    if (serviceFilter !== undefined && serviceFilter !== 'ALL') {
      clauses.push({ service: serviceFilter });
    }
    const latencyRange = options?.latencyRange;
    if (latencyRange !== undefined && (latencyRange.minMs !== undefined || latencyRange.maxMs !== undefined)) {
      const comp: Record<string, number> = {};
      if (latencyRange.minMs !== undefined && Number.isFinite(latencyRange.minMs)) {
        comp.gte = latencyRange.minMs;
      }
      if (latencyRange.maxMs !== undefined && Number.isFinite(latencyRange.maxMs)) {
        comp.lt = latencyRange.maxMs;
      }
      if (Object.keys(comp).length > 0) {
        clauses.push({ latencyMs: comp as { gte?: number; lt?: number } });
      }
    }
    const timestampRange = options?.timestampRange;
    if (timestampRange !== undefined && (timestampRange.from !== undefined || timestampRange.to !== undefined)) {
      const comp: Record<string, string> = {};
      if (typeof timestampRange.from === 'string' && timestampRange.from.trim() !== '' && !Number.isNaN(new Date(timestampRange.from).getTime())) {
        comp.gte = timestampRange.from;
      }
      if (typeof timestampRange.to === 'string' && timestampRange.to.trim() !== '' && !Number.isNaN(new Date(timestampRange.to).getTime())) {
        comp.lt = timestampRange.to;
      }
      if (Object.keys(comp).length > 0) {
        clauses.push({ timestamp: comp as { gte?: string; lt?: string } });
      }
    }
    if (clauses.length === 0) return undefined;
    if (clauses.length === 1) return clauses[0];
    return { and: clauses };
  }

  defaultFacets(): Record<string, FacetRequest> {
    return {
      byLevel: { type: 'terms', field: 'level', limit: 10 },
      byService: { type: 'terms', field: 'service', limit: 10 },
      byLatency: {
        type: 'range',
        field: 'latencyMs',
        ranges: [
          { to: 50, key: 'fast-<50ms' },
          { from: 50, to: 300, key: 'normal-50-300ms' },
          { from: 300, to: 800, key: 'slow-300-800ms' },
          { from: 800, key: 'critical-800ms+' }
        ]
      }
    };
  }

  async search(
    query: string,
    options?: LogSearchOptions
  ): Promise<LogSearchResult> {
    const start = performance.now();
    const mode = options?.mode ?? 'fuzzy';
    const highlight = options?.highlight ?? true;
    const limit = options?.limit ?? 100;

    const filter = this.buildFilter(options);
    const facets = options?.facets === false ? undefined : (options?.facets ?? this.defaultFacets());

    let response: DocumentSearchResponse<StructuredLogRecord>;

    const searchOpts = {
      mode,
      highlight,
      tag: 'mark',
      escapeHtml: true,
      limit,
      ...(filter ? { filter } : {}),
      ...(facets ? { facets } : {}),
      signal: options?.signal
    };

    if (this.useWorker && this.workerClient) {
      response = await this.workerClient.search(query, searchOpts);
    } else if (this.mainIndex) {
      response = await this.mainIndex.search(query, searchOpts);
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
      hasOverflow: response.hasOverflow,
      ...(response.facets ? { facets: response.facets } : {})
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
    if (!buffer || typeof (buffer as ArrayBuffer).byteLength !== 'number') {
      throw new TypeError('[LogEngine] restoreSnapshot expects an ArrayBuffer.');
    }
    const { MAX_SNAPSHOT_BYTES } = await import('webgpu-search');
    if ((buffer as ArrayBuffer).byteLength > (MAX_SNAPSHOT_BYTES as number)) {
      const { IncompatibleIndexError } = await import('webgpu-search');
      throw new IncompatibleIndexError(`snapshot-bytes<=${MAX_SNAPSHOT_BYTES}`, (buffer as ArrayBuffer).byteLength);
    }
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
