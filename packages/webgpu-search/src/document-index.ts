/**
 * High-level multi-field document search index with dynamic mutations and highlighting.
 *
 * This file is the public `DocumentIndex` facade. It owns:
 * - construction-time fail-closed validation (fields, removed `extensions`,
 *   power preference, text profile) and state construction,
 * - lifecycle gating (destroyed checks, mutation mutex scheduling, the
 *   search-generation snapshot),
 * - the public getter/serializer surface,
 * - snapshot restore orchestration (`restore`, `fromSnapshot`,
 *   `fromSnapshotData`).
 *
 * All implementation lives in the `document/` sibling modules operating on
 * the internal `DocumentIndexState` bundle:
 * - `document/build.ts`       : ingestion, batch mutations, GPU bootstrap,
 *                               compaction, destroy.
 * - `document/query.ts`       : search validation/execution + autocomplete.
 * - `document/filter-facet.ts`: filter compilation + facet aggregation.
 * - `document/snapshot.ts`    : snapshot merge/apply + restore guards.
 */

import { assertValidPowerPreference } from './gpu-device-pool';
import {
  IncompatibleOptionError,
  ProfileMismatchError,
  SCORING_VERSION,
  SNAPSHOT_FORMAT_VERSION,
  UNICODE_VERSION,
  type TextProfileId,
} from './text-profile';
import { nowMs, throwIfAborted } from './guard';
import {
  applyBatchOps,
  createDocumentIndexState,
  compactSync,
  destroyIndex,
  enqueue,
  initializeDocumentIndex,
  rebuildGpuEngine,
  compactCpu,
} from './document/build';
import { runAutocomplete, runSearch } from './document/query';
import { applySnapshotData, mergeSnapshotRestoreOptions } from './document/snapshot';
import {
  decodeSnapshot,
  encodeSnapshot,
  restoreSnapshot,
  type RestoredDocumentSnapshot,
} from './snapshot-codec';
import type {
  AddOptions,
  DocumentId,
  DocumentIndexOptions,
  DocumentIndexStats,
  DocumentSearchOptions,
  DocumentSearchResponse,
  FilterFieldDefinition,
  MutationBatch,
  MutationResult,
  RestoreDocumentIndexOptions,
  SearchHooks,
  SerializeDocumentIndexOptions,
  AutocompleteOptions,
  SuggestResponse,
} from './types';
import type { ColumnarStore } from './filtering/columnar-store';
import type { DocumentIndexState, InternalField } from './document/state';

export type { InternalField } from './document/state';
export { assertSnapshotFilterGettersSatisfied } from './document/snapshot';

/**
 * High-level multi-field document search index with dynamic mutations and highlighting.
 * Facade over the `document/` implementation modules.
 */
export class DocumentIndex<TDoc = Record<string, unknown>> {
  readonly options: DocumentIndexOptions<TDoc>;
  private readonly state: DocumentIndexState<TDoc>;

  constructor(options: DocumentIndexOptions<TDoc>) {
    if (!options || !Array.isArray(options.fields) || options.fields.length === 0) {
      throw new TypeError('[webgpu-search] DocumentIndex expects options.fields to be a non-empty array.');
    }
    if ((options as unknown as Record<string, unknown>).extensions !== undefined) {
      throw new IncompatibleOptionError(
        'hooks',
        "[webgpu-search] extensions was removed; use hooks."
      );
    }
    // Fail-closed: unknown powerPreference throws even on CPU-only paths.
    assertValidPowerPreference(options.powerPreference);
    if (options.textProfile !== undefined && options.textProfile !== 'unicode-default') {
      throw new ProfileMismatchError('unicode-default', options.textProfile, 'textProfile');
    }
    this.options = options;
    this.state = createDocumentIndexState(options);
  }

  static async create<TDoc = Record<string, unknown>>(
    records: TDoc[],
    options: DocumentIndexOptions<TDoc>
  ): Promise<DocumentIndex<TDoc>> {
    if (!Array.isArray(records)) {
      throw new TypeError('[webgpu-search] DocumentIndex.create expects records: TDoc[].');
    }
    if (options.slotBytes !== undefined) {
      throw new IncompatibleOptionError(
        'slotBytes',
        '[webgpu-search] slotBytes is not supported (fixed slots removed).'
      );
    }
    if (options.textProfile !== undefined && options.textProfile !== 'unicode-default') {
      throw new ProfileMismatchError('unicode-default', options.textProfile, 'textProfile');
    }

    const index = new DocumentIndex<TDoc>(options);
    await index.initialize(records);
    return index;
  }

  private async initialize(records: TDoc[]): Promise<void> {
    await initializeDocumentIndex(this.state, records);
  }

  async search(
    query: string,
    options: DocumentSearchOptions<TDoc> = {}
  ): Promise<DocumentSearchResponse<TDoc>> {
    if (this.state.isDestroyed) {
      throw new Error('[webgpu-search] DocumentIndex has been destroyed.');
    }
    throwIfAborted(options.signal);
    await this.state.searchMutex;
    if (this.state.isDestroyed) {
      throw new Error('[webgpu-search] DocumentIndex has been destroyed.');
    }
    throwIfAborted(options.signal);
    const gen = this.state.generation;

    return runSearch(this.state, query, options, gen);
  }

  async autocomplete(
    query: string,
    options: AutocompleteOptions = {}
  ): Promise<SuggestResponse<TDoc>> {
    if (this.state.isDestroyed) {
      throw new Error('[webgpu-search] DocumentIndex has been destroyed.');
    }
    return runAutocomplete(this.state, query, options);
  }

  async add(docs: TDoc | TDoc[], options?: AddOptions): Promise<MutationResult> {
    const list = Array.isArray(docs) ? docs : [docs];
    return this.applyBatch({ add: list }, options);
  }

  async update(docs: TDoc | TDoc[]): Promise<MutationResult> {
    const list = Array.isArray(docs) ? docs : [docs];
    return this.applyBatch({ update: list });
  }

  async remove(ids: DocumentId | DocumentId[]): Promise<MutationResult> {
    const list = Array.isArray(ids) ? ids : [ids];
    return this.applyBatch({ remove: list });
  }

  async applyBatch(batch: MutationBatch<TDoc>, options?: AddOptions): Promise<MutationResult> {
    if (this.state.isDestroyed) {
      throw new Error('[webgpu-search] DocumentIndex has been destroyed.');
    }
    if (!batch || typeof batch !== 'object') {
      throw new TypeError('[webgpu-search] applyBatch expects batch object.');
    }
    return enqueue(this.state, () => applyBatchOps(this.state, batch, options));
  }

  compactCpu(): void {
    compactCpu(this.state);
  }

  compactSync(): void {
    compactSync(this.state);
  }

  getRecords(): TDoc[] {
    return this.state.records.slice();
  }

  getDocIds(): DocumentId[] {
    return this.state.docIds.slice();
  }

  getSortedFields(): InternalField<TDoc>[] {
    return this.state.sortedFields;
  }

  getRowTokens(): Uint32Array[] {
    return this.state.rowTokens;
  }

  getTotalTokens(): number {
    return this.state.totalTokens;
  }

  getTombstoneCount(): number {
    return this.state.tombstones.size;
  }

  isNormalized(): boolean {
    return this.state.normalized;
  }

  getProfileId(): TextProfileId {
    return this.state.profileId;
  }

  getUnicodeVersion(): string {
    return UNICODE_VERSION;
  }

  getScoringVersion(): string {
    return SCORING_VERSION;
  }

  getColumnarStore(): ColumnarStore<TDoc> {
    return this.state.columnarStore;
  }

  getFilterFieldDefinitions(): FilterFieldDefinition<TDoc>[] {
    return this.state.filterFieldDefinitions.slice();
  }

  /**
   * index-level search hooks (shallow copy; function references
   * are shared with the index).
   */
  getHooks(): SearchHooks<TDoc> | undefined {
    return this.state.indexHooks === undefined ? undefined : { ...this.state.indexHooks };
  }

  serialize(options?: SerializeDocumentIndexOptions): ArrayBuffer {
    if (this.state.isDestroyed) {
      throw new Error('[webgpu-search] DocumentIndex has been destroyed.');
    }
    return encodeSnapshot(this, options);
  }

  async restore(buffer: ArrayBuffer, options?: RestoreDocumentIndexOptions<TDoc>): Promise<void> {
    if (this.state.isDestroyed) {
      throw new Error('[webgpu-search] DocumentIndex has been destroyed.');
    }
    const t0 = nowMs();
    await enqueue(this.state, async () => {
      const snapshot = decodeSnapshot<TDoc>(buffer, options);
      await applySnapshotData(this.state, snapshot, options);
      this.state.restoreTimeMs = nowMs() - t0;
    });
  }

  static async fromSnapshot<TDoc = Record<string, unknown>>(
    buffer: ArrayBuffer,
    options?: RestoreDocumentIndexOptions<TDoc>
  ): Promise<DocumentIndex<TDoc>> {
    return restoreSnapshot<TDoc>(buffer, options);
  }

  static async fromSnapshotData<TDoc = Record<string, unknown>>(
    snapshot: RestoredDocumentSnapshot<TDoc>,
    options?: RestoreDocumentIndexOptions<TDoc>
  ): Promise<DocumentIndex<TDoc>> {
    const t0 = nowMs();
    const mergedOptions = mergeSnapshotRestoreOptions<TDoc>(snapshot, options);
    const index = new DocumentIndex<TDoc>(mergedOptions);
    await applySnapshotData(index.state, snapshot, options);
    index.state.restoreTimeMs = nowMs() - t0;
    return index;
  }

  /**
   * Explicit device-loss rebuild path (Phase 2 reliability proof).
   *
   * After a `fallbackReason: 'device-lost'` (or `gpu-execution-error`)
   * transition, attempts to re-acquire a GPU device and re-upload the
   * retained in-memory corpus. On success the engine returns to `'webgpu'`
   * with identical match semantics; on failure the index stays on CPU with
   * an explicit `fallbackReason` and returns `false` (never throws for
   * GPU-unavailable).
   *
   * - `preferGpu: false` indexes are CPU-by-design: rebuild is a no-op
   *   returning `false`.
   * - Pass a fresh injected `device` after loss (e.g. a new mock device
   *   in headless tests); otherwise the creation-time `options.device` is
   *   reused, else the shared pool is tried. A supplied device is persisted
   *   into `options.device` for subsequent rebuilds.
   * - Throws only when the index itself is destroyed.
   */
  async rebuildGpu(options?: { device?: GPUDevice; powerPreference?: GPUPowerPreference }): Promise<boolean> {
    return rebuildGpuEngine(this.state, options);
  }

  getStats(): DocumentIndexStats {
    const s = this.state;
    const adapter = s.gpuEngine?.adapterInfo;
    const docCount = s.idToDocIndex.size;
    const rowCount = s.rowTokens.length;
    const tombstoneCount = s.tombstones.size;
    const tombstoneRatio = rowCount > 0 ? tombstoneCount / rowCount : 0;
    const tokenRamBytes = s.totalTokens * 4;
    const offsetRamBytes = (rowCount + 1) * 4;
    const ramBytes = tokenRamBytes + offsetRamBytes;

    return {
      size: docCount,
      engine: s.engineState.engine,
      vramAllocatedBytes: s.vramAllocatedBytes,
      adapterVendor: adapter?.vendor,
      adapterRenderer: adapter?.renderer,
      profileId: s.profileId,
      unicodeVersion: UNICODE_VERSION,
      scoringVersion: SCORING_VERSION,
      tokenCount: s.totalTokens,
      normalized: s.normalized,
      formatVersion: SNAPSHOT_FORMAT_VERSION,
      docCount,
      rowCount,
      tombstoneCount,
      tombstoneRatio,
      buildTimeMs: s.buildTimeMs,
      restoreTimeMs: s.restoreTimeMs,
      lastMutationTimeMs: s.lastMutationTimeMs,
      mutationEpoch: s.mutationEpoch,
      memory: {
        vramBytes: s.vramAllocatedBytes,
        ramBytes,
        totalBytes: s.vramAllocatedBytes + ramBytes,
        tokenRamBytes,
        offsetRamBytes
      },
      fallbackReason: s.engineState.fallbackReason
    };
  }

  destroy(): void {
    if (this.state.isDestroyed) {
      return;
    }
    destroyIndex(this.state);
  }

  [Symbol.dispose](): void {
    this.destroy();
  }
}
