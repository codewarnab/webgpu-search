import {
  DocumentIndex,
  SearchWorkerClient,
  type AutocompleteOptions,
  type DocumentIndexStats,
  type DocumentSearchResponse,
  type FilterExpression,
  type SuggestOptions,
  type SuggestionItem
} from 'webgpu-search';
import type { MonacoFileRecord, MonacoPaletteSearchResult } from './types';

export interface PaletteEngineOptions {
  useWorker?: boolean;
  preferGpu?: boolean;
  candidateCapacity?: number;
}

export interface PaletteSearchOptions {
  mode?: 'fuzzy' | 'substring' | 'prefix' | 'token';
  highlight?: boolean;
  limit?: number;
  signal?: AbortSignal;
  /** Restrict to a symbol kind (e.g. 'class' | 'shader'). 'ALL' disables. */
  typeFilter?: string;
  /** Restrict to a language (e.g. 'typescript' | 'wgsl'). 'ALL' disables. */
  languageFilter?: string;
  /** Request autocomplete suggestions alongside results. */
  autocomplete?: boolean | AutocompleteOptions;
  /** @deprecated Use autocomplete. */
  suggest?: boolean | SuggestOptions;
}

export interface PaletteSearchResult {
  results: MonacoPaletteSearchResult[];
  totalMatches: number;
  searchDurationMs: number;
  engine: 'webgpu' | 'cpu';
  hasOverflow: boolean;
  suggestions?: SuggestionItem<MonacoFileRecord>[];
  facets?: DocumentSearchResponse<MonacoFileRecord>['facets'];
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

  /** columnar filter attributes enabling structured type/language filters + type facets. */
  private readonly filterFieldDefs = [
    { name: 'type', type: 'string' as const },
    { name: 'language', type: 'string' as const }
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

  /**
   * structured type/language pre-filtering via columnar bitsets,
   * native 'prefix' symbol search, and autocomplete suggestions.
   */
  buildFilter(typeFilter?: string, languageFilter?: string): FilterExpression | undefined {
    const clauses: FilterExpression[] = [];
    if (typeFilter !== undefined && typeFilter !== 'ALL') {
      clauses.push({ type: typeFilter });
    }
    if (languageFilter !== undefined && languageFilter !== 'ALL') {
      clauses.push({ language: languageFilter });
    }
    if (clauses.length === 0) return undefined;
    if (clauses.length === 1) return clauses[0];
    return { and: clauses };
  }

  async search(
    query: string,
    options?: PaletteSearchOptions
  ): Promise<PaletteSearchResult> {
    const start = performance.now();
    const mode = options?.mode ?? 'fuzzy';
    const highlight = options?.highlight ?? true;
    const limit = options?.limit ?? 50;
    const filter = this.buildFilter(options?.typeFilter, options?.languageFilter);
    const autocomplete = options?.autocomplete ?? options?.suggest;

    let response: DocumentSearchResponse<MonacoFileRecord>;

    const searchOpts = {
      mode,
      highlight,
      tag: 'mark',
      escapeHtml: true,
      limit,
      signal: options?.signal,
      ...(filter ? { filter } : {}),
      // Type-facet distribution powers the kind breakdown in the palette UI.
      facets: { byType: { type: 'terms' as const, field: 'type', limit: 10 } },
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
      hasOverflow: response.hasOverflow,
      ...(response.suggestions ? { suggestions: response.suggestions } : {}),
      ...(response.facets ? { facets: response.facets } : {})
    };
  }

  /** First-party autocomplete primitive for symbol navigation. */
  async suggest(
    query: string,
    options?: AutocompleteOptions
  ): Promise<{ suggestions: SuggestionItem<MonacoFileRecord>[]; queryDurationMs: number }> {
    if (this.useWorker && this.workerClient) {
      const t0 = performance.now();
      // Worker client has no dedicated suggest RPC; fan out via a
      // suggest-only worker search (suggestions stay index-wide by design).
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

  /** Canonical autocomplete primitive for symbol navigation. */
  async autocomplete(
    query: string,
    options?: AutocompleteOptions
  ): Promise<{ suggestions: SuggestionItem<MonacoFileRecord>[]; queryDurationMs: number }> {
    return this.suggest(query, options);
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

  async serializeSnapshot(): Promise<ArrayBuffer> {
    if (this.useWorker && this.workerClient) {
      return this.workerClient.serialize();
    } else if (this.mainIndex) {
      return this.mainIndex.serialize();
    }
    throw new Error('[PaletteEngine] Index not initialized');
  }

  async restoreSnapshot(buffer: ArrayBuffer): Promise<void> {
    if (!buffer || typeof (buffer as ArrayBuffer).byteLength !== 'number') {
      throw new TypeError('[PaletteEngine] restoreSnapshot expects an ArrayBuffer.');
    }
    const { MAX_SNAPSHOT_BYTES, IncompatibleIndexError } = await import('webgpu-search');
    if ((buffer as ArrayBuffer).byteLength > (MAX_SNAPSHOT_BYTES as number)) {
      throw new IncompatibleIndexError(`snapshot-bytes<=${MAX_SNAPSHOT_BYTES}`, (buffer as ArrayBuffer).byteLength);
    }
    if (this.useWorker && this.workerClient) {
      await this.workerClient.restore(buffer);
      this.records = this.workerClient.getRecords();
    } else if (this.mainIndex) {
      const { restoreDocumentIndex } = await import('webgpu-search');
      const restored = await restoreDocumentIndex<MonacoFileRecord>(buffer, {
        options: { preferGpu: this.preferGpu, candidateCapacity: this.candidateCapacity }
      });
      this.mainIndex.destroy();
      this.mainIndex = restored;
      this.records = restored.getRecords();
    } else {
      throw new Error('[PaletteEngine] Index not initialized');
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
