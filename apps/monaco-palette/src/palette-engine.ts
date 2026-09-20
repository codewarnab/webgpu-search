import {
  DocumentIndex,
  SearchWorkerClient,
  type DocumentIndexStats,
  type DocumentSearchResponse
} from 'webgpu-search';
import type { MonacoFileRecord, MonacoPaletteSearchResult } from './types';

export interface PaletteEngineOptions {
  useWorker?: boolean;
  preferGpu?: boolean;
  candidateCapacity?: number;
}

export class PaletteEngine {
  private mainIndex: DocumentIndex<MonacoFileRecord> | null = null;
  private workerClient: SearchWorkerClient<MonacoFileRecord> | null = null;
  private workerInstance: Worker | null = null;
  private records: MonacoFileRecord[] = [];
  private useWorker: boolean = false;
  private preferGpu: boolean = true;
  private candidateCapacity: number = 8192;
  private isDestroyed: boolean = false;
  private rebuildGeneration: number = 0;

  private readonly fieldDefs = [
    { name: 'filename', weight: 3.0 },
    { name: 'symbols', weight: 2.0 },
    { name: 'path', weight: 1.0 },
    { name: 'description', weight: 0.5 }
  ];

  constructor(options?: PaletteEngineOptions) {
    this.useWorker = options?.useWorker ?? false;
    this.preferGpu = options?.preferGpu ?? true;
    this.candidateCapacity = options?.candidateCapacity ?? 8192;
  }

  async init(records: MonacoFileRecord[]): Promise<void> {
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
        const client = new SearchWorkerClient<MonacoFileRecord>({
          worker
        });

        await client.init(this.records, {
          idField: 'id',
          fields: this.fieldDefs,
          preferGpu: this.preferGpu,
          candidateCapacity: this.candidateCapacity
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
        console.warn('[PaletteEngine] Worker initialization failed, falling back to main thread:', err);
        this.useWorker = false;
        if (this.workerInstance) {
          this.workerInstance.terminate();
          this.workerInstance = null;
        }
        this.workerClient = null;
      }
    }

    // Main thread DocumentIndex
    const mainIdx = await DocumentIndex.create(this.records, {
      idField: 'id',
      fields: this.fieldDefs,
      preferGpu: this.preferGpu,
      candidateCapacity: this.candidateCapacity
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
      signal?: AbortSignal;
    }
  ): Promise<{
    results: MonacoPaletteSearchResult[];
    totalMatches: number;
    searchDurationMs: number;
    engine: 'webgpu' | 'cpu';
    hasOverflow: boolean;
  }> {
    const start = performance.now();
    const mode = options?.mode ?? 'fuzzy';
    const highlight = options?.highlight ?? true;
    const limit = options?.limit ?? 50;

    let response: DocumentSearchResponse<MonacoFileRecord>;

    if (this.useWorker && this.workerClient) {
      response = await this.workerClient.search(query, {
        mode,
        highlight,
        tag: 'mark',
        escapeHtml: true,
        limit,
        signal: options?.signal
      });
    } else if (this.mainIndex) {
      response = await this.mainIndex.search(query, {
        mode,
        highlight,
        tag: 'mark',
        escapeHtml: true,
        limit,
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

    const formattedResults: MonacoPaletteSearchResult[] = response.results.map((item) => ({
      id: String(item.id),
      score: item.score,
      matchedField: item.matchedField,
      doc: item.doc,
      highlightedText: item.highlightedText,
      highlights: item.highlights
    }));

    return {
      results: formattedResults,
      totalMatches: response.totalMatches,
      searchDurationMs: duration,
      engine: response.engine,
      hasOverflow: response.hasOverflow
    };
  }

  async addRecord(record: MonacoFileRecord): Promise<void> {
    this.records.push(record);
    if (this.useWorker && this.workerClient) {
      await this.workerClient.add(record);
    } else if (this.mainIndex) {
      await this.mainIndex.add(record);
    }
  }

  async updateRecord(record: MonacoFileRecord): Promise<void> {
    const idx = this.records.findIndex((r) => r.id === record.id);
    if (idx >= 0) {
      this.records[idx] = record;
    }
    if (this.useWorker && this.workerClient) {
      await this.workerClient.update(record);
    } else if (this.mainIndex) {
      await this.mainIndex.update(record);
    }
  }

  async removeRecord(id: string): Promise<void> {
    const idx = this.records.findIndex((r) => r.id === id);
    if (idx >= 0) {
      this.records.splice(idx, 1);
    }
    if (this.useWorker && this.workerClient) {
      await this.workerClient.remove(id);
    } else if (this.mainIndex) {
      await this.mainIndex.remove(id);
    }
  }

  async batchAdd(records: MonacoFileRecord[]): Promise<void> {
    this.records.push(...records);
    if (this.useWorker && this.workerClient) {
      await this.workerClient.add(records);
    } else if (this.mainIndex) {
      await this.mainIndex.add(records);
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

  getRecords(): MonacoFileRecord[] {
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
