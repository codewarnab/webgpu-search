/**
 * Internal shared-state contract for the decomposed document-index modules.
 *
 * The public `DocumentIndex` facade in `document-index.ts` owns lifecycle
 * concerns (construction-time fail-closed validation, mutation mutex
 * scheduling, destroy) and delegates the implementation to the sibling
 * modules in `document/`:
 * - `build.ts`      : ingestion/index building, batch mutations, GPU engine
 *                     bootstrap, compaction (the imperative mutation shell).
 * - `query.ts`      : `search`, `autocomplete`, inline suggestions, and
 *                     highlight enrichment.
 * - `filter-facet.ts`: search filter compilation + facet aggregation.
 * - `snapshot.ts`   : snapshot application/restoration.
 *
 * `DocumentIndexState` is an internal implementation detail only: it is
 * never exported from the package entrypoint and never exposed as public
 * API. Modules mutate its fields directly, exactly as the pre-refactor
 * private class members did — including the `null`-sentinel slots for
 * tombstoned documents. Keeping the mutable fields in one explicit
 * interface (instead of prototype tricks) keeps the cross-module contract
 * type-checked and cohesive.
 */

import type { EngineState } from '../engine-state';
import type { WebGPUEngine } from '../webgpu-engine';
import type { ColumnarStore } from '../filtering/columnar-store';
import type { FieldScoreDefinition } from '../exact-scorer';
import type { TextProfileId } from '../text-profile';
import type {
  DocumentId,
  DocumentIndexOptions,
  FilterFieldDefinition,
  SearchHooks,
} from '../types';

/**
 * Normalized search-field definition (weight-stratified). Reserved for
 * internal modules; the public `getSortedFields()` accessor returns live
 * instances of this shape.
 */
export interface InternalField<TDoc> extends FieldScoreDefinition {
  name: string;
  weight: number;
  getter: (doc: TDoc) => string | string[] | undefined | null;
  originalIndex: number;
}

/**
 * All mutable document-index internals, grouped into one explicit,
 * type-checked bundle owned by the facade and passed to the `document/`
 * modules by reference.
 */
export interface DocumentIndexState<TDoc> {
  /** User options (kept by reference; `restore` may write hooks/device back into it). */
  options: DocumentIndexOptions<TDoc>;

  /**
   * Frozen normalization polarity: token streams are case-folded when
   * `normalized` is `true` (i.e. `options.caseSensitive` is falsy).
   */
  normalized: boolean;
  profileId: TextProfileId;
  preferGpu: boolean;
  initialCapacity: number;
  growthFactor: number;

  // Active documents state
  records: TDoc[];
  docIds: DocumentId[];
  idToDocIndex: Map<DocumentId, number>;
  getId: (doc: TDoc) => DocumentId;

  // Fields and weights (stratified: sorted by weight desc)
  sortedFields: InternalField<TDoc>[];
  fieldNameToIndex: Map<string, number>;

  // Tokens and row mapping
  rowTokens: Uint32Array[];
  rowToDocIndex: number[];
  rowToFieldIndex: number[];
  docToRowIndices: number[][]; // [docIndex][fieldIndex] -> row index
  rawFieldStrings: string[][]; // [docIndex][fieldIndex]
  totalTokens: number;
  candidateCapacity: number;
  buildTimeMs: number;
  restoreTimeMs?: number;
  lastMutationTimeMs?: number;
  mutationEpoch: number;
  tombstones: Set<number>; // tombstone candidate filtering

  // Concurrency + lifecycle (owned by the facade, read by modules)
  searchMutex: Promise<unknown>;
  generation: number;
  isDestroyed: boolean;

  // Filter columnar + hooks
  columnarStore: ColumnarStore<TDoc>;
  filterFieldDefinitions: FilterFieldDefinition<TDoc>[];
  indexHooks: SearchHooks<TDoc> | undefined;

  // Engine lifecycle
  engineState: EngineState;
  gpuEngine: WebGPUEngine | null;
  vramAllocatedBytes: number;
  unsubscribeDeviceLost?: () => void;
}
