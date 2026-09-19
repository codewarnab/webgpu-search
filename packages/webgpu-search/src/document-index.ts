import type {
  AddOptions,
  DocumentId,
  DocumentIndexOptions,
  DocumentIndexStats,
  DocumentSearchOptions,
  DocumentSearchResponse,
  MutationBatch,
  MutationResult
} from './types';

/**
 * High-level multi-field document search index with dynamic mutations and highlighting.
 * Full implementation lands across M2 (record engine), M3 (highlighting), and M4 (mutations).
 */
export class DocumentIndex<TDoc = Record<string, unknown>> {
  readonly options: DocumentIndexOptions<TDoc>;

  constructor(options: DocumentIndexOptions<TDoc>) {
    this.options = options;
  }

  static async create<TDoc = Record<string, unknown>>(
    _records: TDoc[],
    options: DocumentIndexOptions<TDoc>
  ): Promise<DocumentIndex<TDoc>> {
    return new DocumentIndex<TDoc>(options);
  }

  async search(
    _query: string,
    _options?: DocumentSearchOptions<TDoc>
  ): Promise<DocumentSearchResponse<TDoc>> {
    throw new Error('DocumentIndex.search is scheduled for M2 implementation.');
  }

  async add(_docs: TDoc | TDoc[], _options?: AddOptions): Promise<MutationResult> {
    throw new Error('DocumentIndex.add is scheduled for M4 implementation.');
  }

  async update(_docs: TDoc | TDoc[]): Promise<MutationResult> {
    throw new Error('DocumentIndex.update is scheduled for M4 implementation.');
  }

  async remove(_ids: DocumentId | DocumentId[]): Promise<MutationResult> {
    throw new Error('DocumentIndex.remove is scheduled for M4 implementation.');
  }

  async applyBatch(_batch: MutationBatch<TDoc>, _options?: AddOptions): Promise<MutationResult> {
    throw new Error('DocumentIndex.applyBatch is scheduled for M4 implementation.');
  }

  serialize(): ArrayBuffer {
    throw new Error('DocumentIndex.serialize is scheduled for M6 implementation.');
  }

  restore(_buffer: ArrayBuffer, _options?: { transfer?: boolean }): void {
    throw new Error('DocumentIndex.restore is scheduled for M6 implementation.');
  }

  getStats(): DocumentIndexStats {
    throw new Error('DocumentIndex.getStats is scheduled for M7 implementation.');
  }

  destroy(): void {
    // Teardown hook
  }

  [Symbol.dispose](): void {
    this.destroy();
  }
}
