import {
  DocumentIndex,
  SearchWorkerClient,
  saveIndexToIDB,
  loadIndexFromIDB,
  serializeDocumentIndex,
  restoreDocumentIndex,
  deleteIndexFromIDB,
  type AutocompleteOptions,
  type DocumentIndexStats,
  type DocumentSearchResponse,
  type FacetRequest,
  type FacetResult,
  type FilterExpression,
  type SuggestionItem
} from 'webgpu-search';
import type { DocPageRecord, DocsSearchResult } from './types';

export interface DocsEngineOptions {
  useWorker?: boolean;
  preferGpu?: boolean;
  candidateCapacity?: number;
}

export interface DocsSearchOptions {
  mode?: 'fuzzy' | 'substring' | 'prefix' | 'token';
  highlight?: boolean;
  limit?: number;
  signal?: AbortSignal;
  /** Restrict to a doc section (e.g. 'Guide' | 'API'). 'ALL' disables. */
  sectionFilter?: string;
  /** Restrict to a doc version (e.g. '1.0'). 'ALL' disables. */
  versionFilter?: string;
  /** Request autocomplete suggestions alongside results. */
  autocomplete?: boolean | AutocompleteOptions;
  /** Facet requests; defaults to section terms. `false` disables. */
  facets?: Record<string, FacetRequest> | false;
}

export interface DocsSearchOutcome {
  results: DocsSearchResult[];
  totalMatches: number;
  searchDurationMs: number;
  engine: 'webgpu' | 'cpu';
  fallbackReason?: string;
  hasOverflow: boolean;
  suggestions?: SuggestionItem<DocPageRecord>[];
  facets?: Record<string, FacetResult>;
}

export const IDB_DATABASE_NAME = 'webgpu-docs-search-snapshots';
export const IDB_SNAPSHOT_KEY = 'latest-docs-snapshot';

export class DocsEngine {
  private mainIndex: DocumentIndex<DocPageRecord> | null = null;
  private workerClient: SearchWorkerClient<DocPageRecord> | null = null;
  private workerInstance: Worker | null = null;
  private records: DocPageRecord[] = [];
  private useWorker: boolean = false;
  private preferGpu: boolean = true;
  private candidateCapacity: number = 8192;
  private isDestroyed: boolean = false;
  private rebuildGeneration: number = 0;

  private readonly fieldDefs = [
    { name: 'title', weight: 3.0 },
    { name: 'tags', weight: 2.0 },
    { name: 'section', weight: 1.5 },
    { name: 'content', weight: 1.0 }
  ];

  /** columnar filter attributes enabling structured section/version filters + section facets. */
  private readonly filterFieldDefs = [
    { name: 'section', type: 'string' as const },
    { name: 'version', type: 'string' as const }
  ];

  constructor(options?: DocsEngineOptions) {
    this.useWorker = options?.useWorker ?? false;
    this.preferGpu = options?.preferGpu ?? true;
    this.candidateCapacity = options?.candidateCapacity ?? 8192;
  }

  async init(records: DocPageRecord[]): Promise<void> {
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
        const client = new SearchWorkerClient<DocPageRecord>({ worker });

        await client.init(this.records, {
          idField: 'id',
          fields: this.fieldDefs,
          filterFields: this.filterFieldDefs,
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
        console.warn('[DocsEngine] Worker initialization failed, falling back to main thread:', err);
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

  /** structured section/version pre-filtering via columnar bitsets. */
  buildFilter(sectionFilter?: string, versionFilter?: string): FilterExpression | undefined {
    const clauses: FilterExpression[] = [];
    if (sectionFilter !== undefined && sectionFilter !== 'ALL') {
      clauses.push({ section: sectionFilter });
    }
    if (versionFilter !== undefined && versionFilter !== 'ALL') {
      clauses.push({ version: versionFilter });
    }
    if (clauses.length === 0) return undefined;
    if (clauses.length === 1) return clauses[0];
    return { and: clauses };
  }

  defaultFacets(): Record<string, FacetRequest> {
    return {
      bySection: { type: 'terms', field: 'section', limit: 10 },
      byVersion: { type: 'terms', field: 'version', limit: 10 }
    };
  }

  async search(query: string, options?: DocsSearchOptions): Promise<DocsSearchOutcome> {
    const start = performance.now();
    const mode = options?.mode ?? 'fuzzy';
    const highlight = options?.highlight ?? true;
    const limit = options?.limit ?? 50;
    const filter = this.buildFilter(options?.sectionFilter, options?.versionFilter);
    const autocomplete = options?.autocomplete;
    const facets = options?.facets === false ? undefined : (options?.facets ?? this.defaultFacets());

    let response: DocumentSearchResponse<DocPageRecord>;

    const searchOpts = {
      mode,
      highlight,
      tag: 'mark',
      escapeHtml: true,
      limit,
      signal: options?.signal,
      ...(filter ? { filter } : {}),
      ...(facets ? { facets } : {}),
      ...(autocomplete !== undefined ? { autocomplete } : {})
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

    const formattedResults: DocsSearchResult[] = response.results.map((item) => ({
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
      ...(response.fallbackReason ? { fallbackReason: response.fallbackReason } : {}),
      hasOverflow: response.hasOverflow,
      ...(response.suggestions ? { suggestions: response.suggestions } : {}),
      ...(response.facets ? { facets: response.facets } : {})
    };
  }

  /** Canonical autocomplete primitive for docs search boxes. */
  async autocomplete(
    query: string,
    options?: AutocompleteOptions
  ): Promise<{ suggestions: SuggestionItem<DocPageRecord>[]; queryDurationMs: number }> {
    if (this.useWorker && this.workerClient) {
      const t0 = performance.now();
      const res = await this.workerClient.search(query, {
        limit: 1,
        highlight: false,
        autocomplete: options ?? { mode: 'prefix', limit: 5 }
      } as any);
      return { suggestions: res.suggestions ?? [], queryDurationMs: performance.now() - t0 };
    } else if (this.mainIndex) {
      return this.mainIndex.autocomplete(query, options);
    }
    return { suggestions: [], queryDurationMs: 0 };
  }

  async addRecord(record: DocPageRecord): Promise<void> {
    this.records.push(record);
    if (this.useWorker && this.workerClient) {
      await this.workerClient.add(record);
    } else if (this.mainIndex) {
      await this.mainIndex.add(record);
    }
  }

  async updateRecord(record: DocPageRecord): Promise<void> {
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

  async batchAdd(records: DocPageRecord[]): Promise<void> {
    this.records.push(...records);
    if (this.useWorker && this.workerClient) {
      await this.workerClient.add(records);
    } else if (this.mainIndex) {
      await this.mainIndex.add(records);
    }
  }

  /**
   * Recovery hook: re-acquire the GPU device after a device-lost fallback.
   * Main-thread indexes delegate to `DocumentIndex.rebuildGpu()`; worker
   * indexes recover via `restore()` / re-`init()` instead, so this returns
   * `false` on the worker path by design (never throws for routing).
   */
  async rebuildGpu(): Promise<boolean> {
    if (this.useWorker && this.workerClient) {
      return false;
    } else if (this.mainIndex) {
      return this.mainIndex.rebuildGpu();
    }
    return false;
  }

  async serializeSnapshot(): Promise<ArrayBuffer> {
    if (this.useWorker && this.workerClient) {
      return this.workerClient.serialize();
    } else if (this.mainIndex) {
      return serializeDocumentIndex(this.mainIndex);
    }
    throw new Error('[DocsEngine] Index not initialized');
  }

  async restoreSnapshot(buffer: ArrayBuffer): Promise<void> {
    if (!buffer || typeof (buffer as ArrayBuffer).byteLength !== 'number') {
      throw new TypeError('[DocsEngine] restoreSnapshot expects an ArrayBuffer.');
    }
    const { MAX_SNAPSHOT_BYTES, IncompatibleIndexError } = await import('webgpu-search');
    if ((buffer as ArrayBuffer).byteLength > (MAX_SNAPSHOT_BYTES as number)) {
      throw new IncompatibleIndexError(`snapshot-bytes<=${MAX_SNAPSHOT_BYTES}`, (buffer as ArrayBuffer).byteLength);
    }
    if (this.useWorker && this.workerClient) {
      await this.workerClient.restore(buffer);
      this.records = this.workerClient.getRecords();
    } else if (this.mainIndex) {
      const restored = await restoreDocumentIndex<DocPageRecord>(buffer, {
        options: { preferGpu: this.preferGpu, candidateCapacity: this.candidateCapacity }
      });
      this.mainIndex.destroy();
      this.mainIndex = restored;
      this.records = restored.getRecords();
    } else {
      throw new Error('[DocsEngine] Index not initialized');
    }
  }

  async saveSnapshotToIDB(): Promise<{ byteLength: number; durationMs: number }> {
    const start = performance.now();
    const buf = await this.serializeSnapshot();
    await saveIndexToIDB(buf, {
      dbName: IDB_DATABASE_NAME,
      key: IDB_SNAPSHOT_KEY
    });
    return { byteLength: buf.byteLength, durationMs: performance.now() - start };
  }

  async restoreSnapshotFromIDB(): Promise<{ recordCount: number; durationMs: number }> {
    const start = performance.now();
    const loaded = await loadIndexFromIDB<DocPageRecord>({
      dbName: IDB_DATABASE_NAME,
      key: IDB_SNAPSHOT_KEY
    });
    if (!loaded) {
      throw new Error('No snapshot found in IndexedDB');
    }
    await this.restoreSnapshot(loaded.snapshot);
    return { recordCount: this.records.length, durationMs: performance.now() - start };
  }

  async clearIDB(): Promise<void> {
    await deleteIndexFromIDB({ dbName: IDB_DATABASE_NAME, key: IDB_SNAPSHOT_KEY }).catch(() => {});
  }

  async getStats(): Promise<DocumentIndexStats | null> {
    if (this.useWorker && this.workerClient) {
      return await this.workerClient.getStats();
    } else if (this.mainIndex) {
      return this.mainIndex.getStats();
    }
    return null;
  }

  getRecords(): DocPageRecord[] {
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
