import type {
  AddOptions,
  DocumentId,
  DocumentIndexOptions,
  DocumentIndexStats,
  DocumentSearchOptions,
  DocumentSearchResponse,
  MutationBatch,
  MutationResult,
  WorkerClientOptions
} from '../types';

/**
 * First-party asynchronous client managing an off-thread search worker.
 * Full implementation lands in M5 (monotonic query ID, AbortError, error rehydration).
 */
export class SearchWorkerClient<TDoc = Record<string, unknown>> {
  readonly options?: WorkerClientOptions;

  constructor(options?: WorkerClientOptions) {
    this.options = options;
  }

  async init(_options?: DocumentIndexOptions<TDoc>): Promise<void> {
    throw new Error('SearchWorkerClient.init is scheduled for M5 implementation.');
  }

  async search(
    _query: string,
    _options?: DocumentSearchOptions<TDoc>
  ): Promise<DocumentSearchResponse<TDoc>> {
    throw new Error('SearchWorkerClient.search is scheduled for M5 implementation.');
  }

  async add(_docs: TDoc | TDoc[], _options?: AddOptions): Promise<MutationResult> {
    throw new Error('SearchWorkerClient.add is scheduled for M5 implementation.');
  }

  async update(_docs: TDoc | TDoc[]): Promise<MutationResult> {
    throw new Error('SearchWorkerClient.update is scheduled for M5 implementation.');
  }

  async remove(_ids: DocumentId | DocumentId[]): Promise<MutationResult> {
    throw new Error('SearchWorkerClient.remove is scheduled for M5 implementation.');
  }

  async applyBatch(_batch: MutationBatch<TDoc>, _options?: AddOptions): Promise<MutationResult> {
    throw new Error('SearchWorkerClient.applyBatch is scheduled for M5 implementation.');
  }

  async serialize(): Promise<ArrayBuffer> {
    throw new Error('SearchWorkerClient.serialize is scheduled for M5 implementation.');
  }

  async restore(_buffer: ArrayBuffer, _options?: { transfer?: boolean }): Promise<void> {
    throw new Error('SearchWorkerClient.restore is scheduled for M5 implementation.');
  }

  async getStats(): Promise<DocumentIndexStats> {
    throw new Error('SearchWorkerClient.getStats is scheduled for M5 implementation.');
  }

  async destroy(): Promise<void> {
    // Teardown hook
  }

  async [Symbol.asyncDispose](): Promise<void> {
    await this.destroy();
  }
}
