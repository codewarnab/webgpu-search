/**
 * Document-index ingestion, mutation, engine-bootstrap, and compaction
 * implementation (extracted verbatim from the pre-refactor monolithic
 * `document-index.ts`). The public `DocumentIndex` facade delegates here.
 *
 * Scope of this module:
 * - Option normalization for search fields, filter fields, and id accessors
 *   (construction path, fail-closed error order preserved).
 * - Initial corpus ingestion (`initializeDocumentIndex`).
 * - Two-phase atomic batch mutations (`applyBatchOps`).
 * - CPU compaction + GPU (re)initialization and device-loss subscription.
 * - Index destruction.
 *
 * All functions operate on the shared `DocumentIndexState` bundle and must
 * only be invoked while the index-level invariants hold (e.g. mutations are
 * serialized through the `searchMutex` by the facade).
 */

import { readOwnField, ColumnarStore, type PreparedColumnarRow } from '../filtering/columnar-store';
import { packDataset, checkMemoryBudget } from '../dataset-packing';
import { normalizeText } from '../text-normalization';
import { transitionEngineState, initialEngineState } from '../engine-state';
import { WebGPUEngine } from '../webgpu-engine';
import { GpuDevicePool } from '../gpu-device-pool';
import { normalizeSearchHooks } from '../hooks';
import { nowMs } from '../guard';
import { DuplicateIdError, DocumentNotFoundError } from '../text-profile';
import type {
  AddOptions,
  DocumentId,
  DocumentIndexOptions,
  FilterFieldDefinition,
  FilterFieldType,
  MutationBatch,
  MutationResult,
} from '../types';
import type { DocumentIndexState, InternalField } from './state';

/** Own-property write that is safe for arbitrary user field names: a plain
 * `obj[key] = v` with key `"__proto__"` would replace the prototype instead
 * of creating an entry. (Shared with query-side highlight/result writers.) */
export function setOwnKey<V>(obj: Record<string, V>, key: string, value: V): void {
  Object.defineProperty(obj, key, { value, enumerable: true, configurable: true, writable: true });
}

/**
 * Derive the document-id accessor from `options.idField` (string property
 * name or function). Called during construction; throws `TypeError` on a
 * malformed idField exactly as the pre-refactor constructor did.
 */
export function deriveIdAccessor<TDoc>(
  options: DocumentIndexOptions<TDoc>
): (doc: TDoc) => DocumentId {
  if (typeof options.idField === 'function') {
    return options.idField;
  } else if (typeof options.idField === 'string') {
    const prop = options.idField;
    return (doc: any) => doc[prop];
  } else if (options.idField === undefined) {
    return (doc: any) => doc.id;
  } else {
    throw new TypeError('[webgpu-search] idField must be a string property name or function.');
  }
}

/**
 * Normalize + deduplicate search-field definitions and stratify them
 * (sorted descending by weight, ties broken by original declaration order).
 * Fail-closed: empty names, invalid weights, duplicate names throw.
 */
export function normalizeIndexFields<TDoc>(
  options: DocumentIndexOptions<TDoc>
): InternalField<TDoc>[] {
  const seenNames = new Set<string>();
  const normalized: InternalField<TDoc>[] = options.fields.map((f, originalIndex) => {
    let name: string;
    let weight = 1.0;
    let getter: (doc: TDoc) => string | string[] | undefined | null;

    if (typeof f === 'string') {
      if (f.trim().length === 0) {
        throw new TypeError('[webgpu-search] Field name string must not be empty.');
      }
      name = f;
      getter = (doc: any) => readOwnField(doc, name);
    } else if (f && typeof f === 'object') {
      if (typeof f.name !== 'string' || f.name.length === 0) {
        throw new TypeError('[webgpu-search] Field definition requires a non-empty string name.');
      }
      name = f.name;
      if (f.weight !== undefined) {
        if (typeof f.weight !== 'number' || !Number.isFinite(f.weight) || f.weight <= 0) {
          throw new RangeError(
            `[webgpu-search] Field "${name}" weight must be a positive finite number, got ${f.weight}.`
          );
        }
        weight = f.weight;
      }
      if (f.getter !== undefined) {
        if (typeof f.getter !== 'function') {
          throw new TypeError(`[webgpu-search] Field "${name}" getter must be a function.`);
        }
        getter = f.getter;
      } else {
        getter = (doc: any) => readOwnField(doc, name);
      }
    } else {
      throw new TypeError('[webgpu-search] Field must be a string or FieldDefinition object.');
    }

    if (seenNames.has(name)) {
      throw new Error(`[webgpu-search] Duplicate field name: "${name}".`);
    }
    seenNames.add(name);

    return { name, weight, getter, originalIndex };
  });

  // Field-stratified packing: sort fields descending by weight
  return [...normalized].sort((a, b) => {
    if (b.weight !== a.weight) return b.weight > a.weight ? 1 : -1;
    if (a.originalIndex !== b.originalIndex) return a.originalIndex > b.originalIndex ? 1 : -1;
    return 0;
  });
}

/**
 * Normalize + deduplicate filter-field definitions (name, type, getter).
 * Fail-closed: empty names, invalid getters, duplicate names throw.
 */
export function normalizeFilterFields<TDoc>(
  options: DocumentIndexOptions<TDoc>
): FilterFieldDefinition<TDoc>[] {
  const normalizedFilterFields: FilterFieldDefinition<TDoc>[] = [];
  if (options.filterFields !== undefined) {
    if (!Array.isArray(options.filterFields)) {
      throw new TypeError('[webgpu-search] DocumentIndex expects options.filterFields to be an array.');
    }
    const seenFilterNames = new Set<string>();
    for (let i = 0; i < options.filterFields.length; i++) {
      const ff = options.filterFields[i];
      let name: string;
      let type: FilterFieldType | undefined = undefined;
      let getter: ((doc: TDoc) => any) | undefined = undefined;
      let hasGetter = false;

      if (typeof ff === 'string') {
        if (ff.trim().length === 0) {
          throw new TypeError('[webgpu-search] Filter field name string must not be empty.');
        }
        name = ff;
        getter = (doc: any) => readOwnField(doc, name);
        hasGetter = false;
      } else if (ff && typeof ff === 'object') {
        if (typeof ff.name !== 'string' || ff.name.trim().length === 0) {
          throw new TypeError('[webgpu-search] Filter field definition requires a non-empty string name.');
        }
        name = ff.name;
        type = ff.type;
        if (ff.getter !== undefined) {
          if (typeof ff.getter !== 'function') {
            throw new TypeError(`[webgpu-search] Filter field "${name}" getter must be a function.`);
          }
          getter = ff.getter;
          hasGetter = true;
        } else {
          getter = (doc: any) => readOwnField(doc, name);
          hasGetter = false;
        }
      } else {
        throw new TypeError('[webgpu-search] Filter field must be a string or FilterFieldDefinition object.');
      }

      if (seenFilterNames.has(name)) {
        throw new Error(`[webgpu-search] Duplicate filter field name: "${name}".`);
      }
      seenFilterNames.add(name);
      normalizedFilterFields.push(hasGetter ? { name, type, getter, hasGetter: true } : { name, type, getter });
    }
  }
  return normalizedFilterFields;
}

/**
 * Construct the shared mutable-state bundle from validated options.
 * Mirrors the pre-refactor constructor field-by-field: hooks are validated
 * first, then capacity defaults, then the id accessor, then search fields,
 * then filter fields — preserving the original error precedence.
 */
export function createDocumentIndexState<TDoc>(
  options: DocumentIndexOptions<TDoc>
): DocumentIndexState<TDoc> {
  // fail-closed extension hook validation at construction.
  const indexHooks = normalizeSearchHooks(options.hooks);

  const initialCapacity = typeof options.initialCapacity === 'number' && Number.isFinite(options.initialCapacity) && options.initialCapacity > 0
    ? Math.floor(options.initialCapacity)
    : 0;
  const growthFactor = typeof options.growthFactor === 'number' && Number.isFinite(options.growthFactor) && options.growthFactor >= 1.0
    ? options.growthFactor
    : 1.5;

  const normalized = !(options.caseSensitive ?? false);
  const preferGpu = options.preferGpu ?? false;
  const getId = deriveIdAccessor<TDoc>(options);
  const sortedFields = normalizeIndexFields<TDoc>(options);
  const filterFieldDefinitions = normalizeFilterFields<TDoc>(options);

  const fieldNameToIndex: Map<string, number> = new Map();
  for (let i = 0; i < sortedFields.length; i++) {
    fieldNameToIndex.set(sortedFields[i]!.name, i);
  }

  const columnarStore = new ColumnarStore<TDoc>(filterFieldDefinitions, {
    initialCapacity,
    growthFactor
  });

  return {
    options,
    normalized,
    profileId: 'unicode-default',
    preferGpu,
    initialCapacity,
    growthFactor,

    records: [],
    docIds: [],
    idToDocIndex: new Map<DocumentId, number>(),
    getId,

    sortedFields,
    fieldNameToIndex,

    rowTokens: [],
    rowToDocIndex: [],
    rowToFieldIndex: [],
    docToRowIndices: [],
    rawFieldStrings: [],
    totalTokens: 0,
    candidateCapacity: 8192,
    buildTimeMs: 0,
    lastMutationTimeMs: undefined,
    mutationEpoch: 0,
    tombstones: new Set<number>(),

    searchMutex: Promise.resolve(),
    generation: 0,
    isDestroyed: false,

    columnarStore,
    filterFieldDefinitions,
    indexHooks,

    engineState: initialEngineState(),
    gpuEngine: null,
    vramAllocatedBytes: 0,
    unsubscribeDeviceLost: undefined
  };
}

/**
 * Initial corpus ingestion (`DocumentIndex.create`). Builds the doc/id maps,
 * the field-stratified row packing, the columnar store, and attempts the
 * GPU engine when the corpus clears the configured routing thresholds.
 */
export async function initializeDocumentIndex<TDoc>(s: DocumentIndexState<TDoc>, records: TDoc[]): Promise<void> {
  const t0 = nowMs();
  const docCount = records.length;
  const fieldCount = s.sortedFields.length;
  const totalRows = docCount * fieldCount;

  s.records = records.slice();
  s.docIds = new Array(docCount);
  s.idToDocIndex = new Map<DocumentId, number>();

  for (let i = 0; i < docCount; i++) {
    const doc = s.records[i];
    if (doc === null || typeof doc !== 'object') {
      throw new TypeError(`[webgpu-search] Document at index ${i} must be an object.`);
    }
    const id = s.getId(doc);
    if (
      id === null ||
      id === undefined ||
      (typeof id !== 'string' && typeof id !== 'number') ||
      (typeof id === 'number' && !Number.isFinite(id))
    ) {
      throw new TypeError(
        `[webgpu-search] Document at index ${i} has invalid id: ${String(id)}. DocumentId must be a non-empty string or finite number.`
      );
    }
    if (typeof id === 'string' && id.length === 0) {
      throw new TypeError(
        `[webgpu-search] Document at index ${i} has an empty string id. DocumentId must not be empty.`
      );
    }
    if (s.idToDocIndex.has(id)) {
      throw new DuplicateIdError(id);
    }
    s.docIds[i] = id;
    s.idToDocIndex.set(id, i);
  }

  // Dynamic Candidate Pool Scaling (Section 3.1):
  // candidateCapacity = min(32768, max(8192, docCount * fieldCount * 0.1))
  const rawCap = s.options.candidateCapacity;
  const validUserCap = typeof rawCap === 'number' && Number.isFinite(rawCap) && rawCap > 0
    ? Math.floor(rawCap)
    : Math.floor(docCount * fieldCount * 0.1);
  s.candidateCapacity = Math.min(32768, Math.max(8192, validUserCap));

  // Field-Stratified Row Packing:
  // Rows 0..N-1: Primary fields (highest weight).
  // Rows N..2N-1: Secondary fields.
  // Rows 2N..3N-1: Lower-weight fields.
  s.rowTokens = new Array(totalRows);
  s.rowToDocIndex = new Array(totalRows);
  s.rowToFieldIndex = new Array(totalRows);
  s.docToRowIndices = Array.from({ length: docCount }, () => new Array(fieldCount));
  s.rawFieldStrings = Array.from({ length: docCount }, () => new Array(fieldCount));

  let corpusTokens = 0;
  let rowIdx = 0;

  for (let f = 0; f < fieldCount; f++) {
    const field = s.sortedFields[f];
    for (let d = 0; d < docCount; d++) {
      const doc = s.records[d];
      const rawVal = field.getter(doc);
      let rawStr = '';
      if (rawVal !== null && rawVal !== undefined) {
        if (Array.isArray(rawVal)) {
          rawStr = rawVal.filter((x) => x !== null && x !== undefined).join(' ');
        } else {
          rawStr = typeof rawVal === 'string' ? rawVal : String(rawVal);
        }
      }
      s.rawFieldStrings[d][f] = rawStr;
      const norm = normalizeText(rawStr, s.normalized);
      s.rowTokens[rowIdx] = norm.tokens;
      s.rowToDocIndex[rowIdx] = d;
      s.rowToFieldIndex[rowIdx] = f;
      s.docToRowIndices[d][f] = rowIdx;
      corpusTokens += norm.tokenCount;
      rowIdx++;
    }
  }

  s.totalTokens = corpusTokens;
  s.columnarStore.init(s.records);

  if (docCount === 0) {
    s.engineState = transitionEngineState(
      s.engineState,
      'cpu',
      s.options.preferGpu === false ? 'prefer-cpu' : 'below-threshold',
    );
    s.buildTimeMs = nowMs() - t0;
    return;
  }

  const threshold = s.options.threshold ?? 30_000;
  const shouldAttemptGpu = s.options.preferGpu === false
    ? false
    : s.preferGpu || totalRows >= threshold;

  if (s.options.preferGpu === false) {
    s.engineState = transitionEngineState(s.engineState, 'cpu', 'prefer-cpu');
  } else if (!s.preferGpu && totalRows < threshold) {
    s.engineState = transitionEngineState(s.engineState, 'cpu', 'below-threshold');
  }

  if (shouldAttemptGpu) {
    const measuredAvgBytes = totalRows > 0 ? (corpusTokens * 4) / totalRows : 0;
    const budget = checkMemoryBudget(totalRows, measuredAvgBytes, s.options.device, s.candidateCapacity);
    if (!budget.allowed) {
      console.warn(`[webgpu-search] ${budget.reason} Falling back to CPU.`);
      s.engineState = transitionEngineState(s.engineState, 'cpu', 'memory-budget-exceeded');
    } else {
      let gpu: WebGPUEngine | null = null;
      try {
        gpu = new WebGPUEngine();
        const initialized = await gpu.init(
          s.options.device !== undefined || s.options.powerPreference !== undefined
            ? { device: s.options.device, powerPreference: s.options.powerPreference }
            : undefined
        );

        if (initialized && gpu.isReady) {
          gpu.ensureCandidateCapacity(s.candidateCapacity);
          const packed = packDataset(s.rowTokens, {
            normalized: s.normalized,
            totalTokens: corpusTokens
          });
          const effectiveCap = s.initialCapacity > 0 ? s.initialCapacity : Math.max(16, docCount);
          const rowCap = effectiveCap * fieldCount;
          const avgTokens = totalRows > 0 ? corpusTokens / totalRows : 16;
          const tokenCap = Math.max(Math.floor(rowCap * avgTokens), 64);
          await gpu.loadDataset(packed, {
            rowCapacity: rowCap,
            tokenCapacity: tokenCap,
            growthFactor: s.initialCapacity > 0 ? 1.0 : s.growthFactor
          });

          s.gpuEngine = gpu;
          s.engineState = transitionEngineState(s.engineState, 'webgpu');
          s.vramAllocatedBytes = gpu.vramAllocatedBytes;
          gpu = null; // Ownership transferred

          subscribeDeviceLost(s);
        } else {
          s.engineState = transitionEngineState(
            s.engineState,
            'cpu',
            typeof navigator === 'undefined' || !('gpu' in navigator) || !navigator.gpu
              ? 'webgpu-unsupported'
              : 'device-request-failed',
          );
        }
      } catch (gpuErr) {
        console.warn('[webgpu-search] WebGPU initialization failed, falling back to CPU:', gpuErr);
        s.engineState = transitionEngineState(
          s.engineState,
          'cpu',
          typeof navigator === 'undefined' || !('gpu' in navigator) || !navigator.gpu
            ? 'webgpu-unsupported'
            : 'device-request-failed',
        );
      } finally {
        if (gpu) {
          gpu.destroy();
        }
      }
    }
  } else {
    s.engineState = transitionEngineState(
      s.engineState,
      'cpu',
      s.engineState.fallbackReason,
    );
  }

  s.buildTimeMs = nowMs() - t0;
}

/**
 * Fail-closed document-id validation shared by batch mutation paths.
 */
export function validateDocumentId(id: unknown, context: string): void {
  if (
    id === null ||
    id === undefined ||
    (typeof id !== 'string' && typeof id !== 'number') ||
    (typeof id === 'number' && !Number.isFinite(id))
  ) {
    throw new TypeError(
      `[webgpu-search] Invalid document ID in ${context}: ${String(id)}. DocumentId must be a non-empty string or finite number.`
    );
  }
  if (typeof id === 'string' && id.length === 0) {
    throw new TypeError(
      `[webgpu-search] Document ID in ${context} must not be an empty string.`
    );
  }
}

/** Field-token preparation result for a single document (pre-columnar). */
export interface PreparedDocFields<TDoc> {
  id: DocumentId;
  doc: TDoc;
  rawStrings: string[];
  tokens: Uint32Array[];
  tokenCount: number;
}

/** A prepared document with its columnar pre-filter row attached. */
export interface PreparedDocMutation<TDoc> extends PreparedDocFields<TDoc> {
  columnar: PreparedColumnarRow;
}

/**
 * Extract raw strings + normalized tokens for every field of one document
 * (mirrors the ingestion row-packing rule for arrays/scalars).
 */
export function prepareDocFields<TDoc>(
  s: DocumentIndexState<TDoc>,
  doc: TDoc,
  id: DocumentId
): PreparedDocFields<TDoc> {
  const fieldCount = s.sortedFields.length;
  const rawStrings = new Array<string>(fieldCount);
  const tokens = new Array<Uint32Array>(fieldCount);
  let tokenCount = 0;

  for (let f = 0; f < fieldCount; f++) {
    const field = s.sortedFields[f];
    const rawVal = field.getter(doc);
    let rawStr = '';
    if (rawVal !== null && rawVal !== undefined) {
      if (Array.isArray(rawVal)) {
        rawStr = rawVal.filter((x) => x !== null && x !== undefined).join(' ');
      } else {
        rawStr = typeof rawVal === 'string' ? rawVal : String(rawVal);
      }
    }
    rawStrings[f] = rawStr;
    const norm = normalizeText(rawStr, s.normalized);
    tokens[f] = norm.tokens;
    tokenCount += norm.tokenCount;
  }

  return { id, doc, rawStrings, tokens, tokenCount };
}

/**
 * Serialize a mutation through the index-wide search mutex. The returned
 * promise resolves with the mutation result; the stored mutex tail never
 * rejects (failures surface only through the returned promise).
 */
export function enqueue<TDoc, T>(s: DocumentIndexState<TDoc>, fn: () => Promise<T>): Promise<T> {
  const p = s.searchMutex.then(fn, fn);
  s.searchMutex = p.catch(() => {});
  return p;
}

/**
 * Two-phase atomic batch mutation (remove → update → add) executed with the
 * mutation mutex already held. Phase 1 validates the whole batch and
 * prepares all field/columnar work before touching state, so a throwing
 * getter (or any validation failure) leaves the index untouched.
 */
export async function applyBatchOps<TDoc>(
  s: DocumentIndexState<TDoc>,
  batch: MutationBatch<TDoc>,
  options?: AddOptions
): Promise<MutationResult> {
  if (s.isDestroyed) {
    throw new Error('[webgpu-search] DocumentIndex has been destroyed.');
  }
  const tStart = nowMs();
  const upsert = options?.upsert ?? false;

  // PHASE 1: ATOMIC TWO-PHASE VALIDATION (State remains untouched on error)
  const removeIds: DocumentId[] = [];
  const willRemoveSet = new Set<DocumentId>();
  if (batch.remove !== undefined) {
    if (!Array.isArray(batch.remove)) {
      throw new TypeError('[webgpu-search] batch.remove must be an array of document IDs.');
    }
    for (let i = 0; i < batch.remove.length; i++) {
      const id = batch.remove[i];
      validateDocumentId(id, `remove at index ${i}`);
      removeIds.push(id);
      if (s.idToDocIndex.has(id)) {
        willRemoveSet.add(id);
      }
    }
  }

  const preparedUpdates: PreparedDocMutation<TDoc>[] = [];
  const updatedIdsInBatch = new Set<DocumentId>();
  if (batch.update !== undefined) {
    if (!Array.isArray(batch.update)) {
      throw new TypeError('[webgpu-search] batch.update must be an array of documents.');
    }
    for (let i = 0; i < batch.update.length; i++) {
      const doc = batch.update[i];
      if (doc === null || typeof doc !== 'object') {
        throw new TypeError(`[webgpu-search] batch.update document at index ${i} must be an object.`);
      }
      const id = s.getId(doc);
      validateDocumentId(id, `update at index ${i}`);
      if (!s.idToDocIndex.has(id) || willRemoveSet.has(id)) {
        throw new DocumentNotFoundError(id);
      }
      if (updatedIdsInBatch.has(id)) {
        throw new DuplicateIdError(id);
      }
      updatedIdsInBatch.add(id);
      // Filter getters run here (before any mutation) so a throwing
      // getter rejects the whole batch with the index untouched.
      const prepared = prepareDocFields(s, doc, id);
      preparedUpdates.push({ ...prepared, columnar: s.columnarStore.prepare(doc) });
    }
  }

  const preparedAdds: PreparedDocMutation<TDoc>[] = [];
  const addedIdsInBatch = new Set<DocumentId>();

  if (batch.add !== undefined) {
    if (!Array.isArray(batch.add)) {
      throw new TypeError('[webgpu-search] batch.add must be an array of documents.');
    }
    for (let i = 0; i < batch.add.length; i++) {
      const doc = batch.add[i];
      if (doc === null || typeof doc !== 'object') {
        throw new TypeError(`[webgpu-search] batch.add document at index ${i} must be an object.`);
      }
      const id = s.getId(doc);
      validateDocumentId(id, `add at index ${i}`);

      const existsInIndex = s.idToDocIndex.has(id) && !willRemoveSet.has(id);
      const existsInBatch = addedIdsInBatch.has(id);

      if (existsInIndex || existsInBatch) {
        if (!upsert) {
          throw new DuplicateIdError(id);
        }
      }
      addedIdsInBatch.add(id);

      const prepared = prepareDocFields(s, doc, id);
      preparedAdds.push({ ...prepared, columnar: s.columnarStore.prepare(doc) });
    }
  }

  // PHASE 2: DETERMINISTIC EXECUTION (remove -> update -> add)
  s.generation++;
  let removedCount = 0;
  let updatedCount = 0;
  let addedCount = 0;

  // A. Remove
  for (let i = 0; i < removeIds.length; i++) {
    const id = removeIds[i];
    const d = s.idToDocIndex.get(id);
    if (d !== undefined) {
      const rows = s.docToRowIndices[d];
      if (rows) {
        for (let f = 0; f < rows.length; f++) {
          s.tombstones.add(rows[f]);
        }
      }
      s.idToDocIndex.delete(id);
      s.records[d] = null as any;
      s.docIds[d] = null as any;
      s.rawFieldStrings[d] = null as any;
      s.docToRowIndices[d] = null as any;
      s.columnarStore.remove(d);
      removedCount++;
    }
  }

  // B. Update
  const newRowTokensToAppend: Uint32Array[] = [];
  const newRowDocIndicesToAppend: number[] = [];
  const newRowFieldIndicesToAppend: number[] = [];
  let newTokensToAppendCount = 0;

  for (let i = 0; i < preparedUpdates.length; i++) {
    const upd = preparedUpdates[i];
    const oldD = s.idToDocIndex.get(upd.id)!;
    const oldRows = s.docToRowIndices[oldD];
    if (oldRows) {
      for (let f = 0; f < oldRows.length; f++) {
        s.tombstones.add(oldRows[f]);
      }
    }
    s.records[oldD] = null as any;
    s.docIds[oldD] = null as any;
    s.rawFieldStrings[oldD] = null as any;
    s.docToRowIndices[oldD] = null as any;
    s.columnarStore.remove(oldD);

    const newD = s.records.length;
    s.records.push(upd.doc);
    s.docIds.push(upd.id);
    s.idToDocIndex.set(upd.id, newD);
    s.rawFieldStrings.push(upd.rawStrings);
    s.columnarStore.addPrepared(newD, upd.columnar);

    const docRows: number[] = new Array(s.sortedFields.length);
    for (let f = 0; f < s.sortedFields.length; f++) {
      const rowIdx = s.rowTokens.length + newRowTokensToAppend.length;
      newRowTokensToAppend.push(upd.tokens[f]);
      newRowDocIndicesToAppend.push(newD);
      newRowFieldIndicesToAppend.push(f);
      docRows[f] = rowIdx;
      newTokensToAppendCount += upd.tokens[f].length;
    }
    s.docToRowIndices.push(docRows);
    updatedCount++;
  }

  // C. Add
  for (let i = 0; i < preparedAdds.length; i++) {
    const ad = preparedAdds[i];
    if (s.idToDocIndex.has(ad.id)) {
      const oldD = s.idToDocIndex.get(ad.id)!;
      const oldRows = s.docToRowIndices[oldD];
      if (oldRows) {
        for (let f = 0; f < oldRows.length; f++) {
          s.tombstones.add(oldRows[f]);
        }
      }
      s.records[oldD] = null as any;
      s.docIds[oldD] = null as any;
      s.rawFieldStrings[oldD] = null as any;
      s.docToRowIndices[oldD] = null as any;
      s.columnarStore.remove(oldD);
      updatedCount++;
    } else {
      addedCount++;
    }

    const newD = s.records.length;
    s.records.push(ad.doc);
    s.docIds.push(ad.id);
    s.idToDocIndex.set(ad.id, newD);
    s.rawFieldStrings.push(ad.rawStrings);
    s.columnarStore.addPrepared(newD, ad.columnar);

    const docRows: number[] = new Array(s.sortedFields.length);
    for (let f = 0; f < s.sortedFields.length; f++) {
      const rowIdx = s.rowTokens.length + newRowTokensToAppend.length;
      newRowTokensToAppend.push(ad.tokens[f]);
      newRowDocIndicesToAppend.push(newD);
      newRowFieldIndicesToAppend.push(f);
      docRows[f] = rowIdx;
      newTokensToAppendCount += ad.tokens[f].length;
    }
    s.docToRowIndices.push(docRows);
  }

  // Append newly prepared rows to rowTokens, rowToDocIndex, rowToFieldIndex
  if (newRowTokensToAppend.length > 0) {
    for (let r = 0; r < newRowTokensToAppend.length; r++) {
      s.rowTokens.push(newRowTokensToAppend[r]);
      s.rowToDocIndex.push(newRowDocIndicesToAppend[r]);
      s.rowToFieldIndex.push(newRowFieldIndicesToAppend[r]);
    }
    s.totalTokens += newTokensToAppendCount;
  }

  // D. Check Compaction / GPU Headroom
  const totalRows = s.rowTokens.length;
  const tombstoneCount = s.tombstones.size;
  const tombstoneRatio = totalRows > 0 ? tombstoneCount / totalRows : 0;

  let compacted = false;
  const isGpu = s.engineState.engine === 'webgpu' && s.gpuEngine !== null && s.gpuEngine.isReady;
  const gpuHeadroomExhausted = isGpu && !s.gpuEngine!.canFitHeadroom(totalRows, s.totalTokens);

  const activeDocCount = s.idToDocIndex.size;
  const newCandidateCap = Math.min(32768, Math.max(8192, Math.floor(activeDocCount * s.sortedFields.length * 0.1)));
  if (newCandidateCap > s.candidateCapacity) {
    s.candidateCapacity = newCandidateCap;
    s.gpuEngine?.ensureCandidateCapacity(newCandidateCap);
  }

  try {
    if (tombstoneRatio >= 0.25 || gpuHeadroomExhausted) {
      await compactIndex(s);
      compacted = true;
    } else if (isGpu && newRowTokensToAppend.length > 0) {
      await syncAppendedRowsToGpu(s, newRowTokensToAppend, totalRows, newTokensToAppendCount);
    } else if (!isGpu && (s.preferGpu || totalRows >= (s.options.threshold ?? 30_000)) && s.options.preferGpu !== false && activeDocCount > 0) {
      await tryInitializeGpuEngine(s);
    }
  } catch (err) {
    console.warn('[webgpu-search] GPU sync failed during mutation; falling back to CPU:', err);
    if (s.gpuEngine) {
      try { s.gpuEngine.destroy(); } catch {}
      s.gpuEngine = null;
    }
    s.engineState = transitionEngineState(s.engineState, 'cpu', 'gpu-execution-error');
    s.vramAllocatedBytes = 0;
  }

  if (s.engineState.engine === 'cpu' && !s.engineState.fallbackReason) {
    s.engineState = transitionEngineState(
      s.engineState,
      'cpu',
      s.options.preferGpu === false ? 'prefer-cpu' : 'below-threshold',
    );
  }

  s.mutationEpoch++;
  const durationMs = nowMs() - tStart;
  s.lastMutationTimeMs = durationMs;

  return {
    added: addedCount,
    updated: updatedCount,
    removed: removedCount,
    mutationEpoch: s.mutationEpoch,
    compacted,
    durationMs
  };
}

/**
 * Incremental GPU sync of appended mutation rows (offsets continue at the
 * pre-append corpus token count). VRAM accounting is refreshed afterwards.
 */
export async function syncAppendedRowsToGpu<TDoc>(
  s: DocumentIndexState<TDoc>,
  newRows: Uint32Array[],
  totalRows: number,
  newTokensCount: number
): Promise<void> {
  if (!s.gpuEngine || !s.gpuEngine.isReady) return;
  const numNewRows = newRows.length;
  const oldTotalTokens = s.totalTokens - newTokensCount;

  const newOffsets = new Uint32Array(numNewRows);
  let runningOffset = oldTotalTokens;
  for (let i = 0; i < numNewRows; i++) {
    runningOffset += newRows[i].length;
    newOffsets[i] = runningOffset;
  }

  const newTokens = new Uint32Array(newTokensCount);
  let pos = 0;
  for (let i = 0; i < numNewRows; i++) {
    newTokens.set(newRows[i], pos);
    pos += newRows[i].length;
  }

  await s.gpuEngine.appendRows(newTokens, newOffsets, totalRows);
  s.vramAllocatedBytes = s.gpuEngine.vramAllocatedBytes;
}

/**
 * CPU-side compaction: drop tombstoned slots, re-pack documents/ids/rows
 * into the field-stratified layout, and remap columnar store indices.
 */
export function compactCpu<TDoc>(s: DocumentIndexState<TDoc>): void {
  const activeDocCount = s.idToDocIndex.size;
  const fieldCount = s.sortedFields.length;
  const totalRows = activeDocCount * fieldCount;

  const activeOldDocIndices: number[] = [];
  for (let d = 0; d < s.records.length; d++) {
    const id = s.docIds[d];
    if (id !== null && id !== undefined && s.idToDocIndex.get(id) === d) {
      activeOldDocIndices.push(d);
    }
  }

  const newRecords: TDoc[] = new Array(activeDocCount);
  const newDocIds: DocumentId[] = new Array(activeDocCount);
  const newIdToDocIndex = new Map<DocumentId, number>();
  const newRawFieldStrings: string[][] = Array.from({ length: activeDocCount }, () => new Array(fieldCount));

  for (let newD = 0; newD < activeDocCount; newD++) {
    const oldD = activeOldDocIndices[newD];
    const doc = s.records[oldD];
    const id = s.docIds[oldD];
    newRecords[newD] = doc as TDoc;
    newDocIds[newD] = id as DocumentId;
    newIdToDocIndex.set(id as DocumentId, newD);
    for (let f = 0; f < fieldCount; f++) {
      newRawFieldStrings[newD][f] = s.rawFieldStrings[oldD][f];
    }
  }

  const newRowTokens: Uint32Array[] = new Array(totalRows);
  const newRowToDocIndex: number[] = new Array(totalRows);
  const newRowToFieldIndex: number[] = new Array(totalRows);
  const newDocToRowIndices: number[][] = Array.from({ length: activeDocCount }, () => new Array(fieldCount));

  let corpusTokens = 0;
  let rowIdx = 0;

  for (let f = 0; f < fieldCount; f++) {
    for (let newD = 0; newD < activeDocCount; newD++) {
      const oldD = activeOldDocIndices[newD];
      const oldRowIdx = s.docToRowIndices[oldD][f];
      const tokens = s.rowTokens[oldRowIdx];
      newRowTokens[rowIdx] = tokens;
      newRowToDocIndex[rowIdx] = newD;
      newRowToFieldIndex[rowIdx] = f;
      newDocToRowIndices[newD][f] = rowIdx;
      corpusTokens += tokens.length;
      rowIdx++;
    }
  }

  s.records = newRecords;
  s.docIds = newDocIds;
  s.idToDocIndex = newIdToDocIndex;
  s.rawFieldStrings = newRawFieldStrings;
  s.rowTokens = newRowTokens;
  s.rowToDocIndex = newRowToDocIndex;
  s.rowToFieldIndex = newRowToFieldIndex;
  s.docToRowIndices = newDocToRowIndices;
  s.totalTokens = corpusTokens;
  const oldToNewDocIndexMap = new Map<number, number>();
  for (let newD = 0; newD < activeDocCount; newD++) {
    oldToNewDocIndexMap.set(activeOldDocIndices[newD], newD);
  }
  s.columnarStore.compact(oldToNewDocIndexMap, activeDocCount);
  s.tombstones.clear();
}

/** True when the row layout is exactly field-stratified (compaction no-op). */
function isStratified<TDoc>(s: DocumentIndexState<TDoc>): boolean {
  const docCount = s.records.length;
  if (docCount === 0) return true;
  for (let r = 0; r < s.rowTokens.length; r++) {
    if (
      s.rowToFieldIndex[r] !== Math.floor(r / docCount) ||
      s.rowToDocIndex[r] !== (r % docCount)
    ) {
      return false;
    }
  }
  return true;
}

/**
 * Public synchronous compaction: compacts CPU state when tombstones exist or
 * the row layout is no longer stratified, discarding the GPU engine (which
 * cannot represent the compacted layout without a full re-upload).
 */
export function compactSync<TDoc>(s: DocumentIndexState<TDoc>): void {
  if (s.tombstones.size > 0 || !isStratified(s)) {
    compactCpu(s);
    if (s.gpuEngine) {
      s.gpuEngine.destroy();
      s.gpuEngine = null;
      s.engineState = transitionEngineState(
        s.engineState,
        'cpu',
        s.options.preferGpu === false ? 'prefer-cpu' : 'below-threshold',
      );
    }
  }
}

/**
 * Full compaction with GPU reload: compact CPU state first, then re-upload
 * the compacted corpus into the GPU engine (or reset it for an empty index).
 */
export async function compactIndex<TDoc>(s: DocumentIndexState<TDoc>): Promise<void> {
  compactCpu(s);

  if (s.engineState.engine === 'webgpu' && s.gpuEngine && s.gpuEngine.isReady) {
    const activeDocCount = s.idToDocIndex.size;
    const totalRows = s.rowTokens.length;
    if (activeDocCount === 0) {
      const packed = packDataset([], { normalized: s.normalized });
      await s.gpuEngine.loadDataset(packed);
      s.vramAllocatedBytes = s.gpuEngine.vramAllocatedBytes;
    } else {
      const packed = packDataset(s.rowTokens, {
        normalized: s.normalized,
        totalTokens: s.totalTokens
      });
      const rowCap = Math.max(totalRows, 16);
      const tokenCap = Math.max(s.totalTokens, 64);
      await s.gpuEngine.loadDataset(packed, {
        rowCapacity: rowCap,
        tokenCapacity: tokenCap,
        growthFactor: s.growthFactor
      });
      s.vramAllocatedBytes = s.gpuEngine.vramAllocatedBytes;
    }
  }
}

/**
 * Single device-loss subscription (re-subscribes replace the old one so
 * rebuild cycles never leak listeners). On loss the GPU engine is torn
 * down, VRAM accounting is zeroed, and the state moves to CPU with
 * `fallbackReason: 'device-lost'` — CPU corpora (`rowTokens`,
 * `records`) are retained so `rebuildGpu()` can re-upload identically.
 */
export function subscribeDeviceLost<TDoc>(s: DocumentIndexState<TDoc>): void {
  if (s.unsubscribeDeviceLost) {
    s.unsubscribeDeviceLost();
    s.unsubscribeDeviceLost = undefined;
  }
  s.unsubscribeDeviceLost = GpuDevicePool.onDeviceLost(() => {
    console.warn('[webgpu-search] GPU device lost, falling back to CPU.');
    if (s.gpuEngine) {
      s.gpuEngine.destroy();
      s.gpuEngine = null;
    }
    s.vramAllocatedBytes = 0;
    s.engineState = transitionEngineState(s.engineState, 'cpu', 'device-lost');
  });
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
export async function rebuildGpuEngine<TDoc>(
  s: DocumentIndexState<TDoc>,
  options?: { device?: GPUDevice; powerPreference?: GPUPowerPreference }
): Promise<boolean> {
  if (s.isDestroyed) {
    throw new Error('[webgpu-search] DocumentIndex has been destroyed.');
  }
  if (s.gpuEngine !== null && s.gpuEngine.isReady && s.engineState.engine === 'webgpu') {
    return true;
  }
  if (s.options.preferGpu === false) {
    return false;
  }
  if (s.idToDocIndex.size === 0) {
    return false;
  }
  if (options?.device !== undefined) {
    (s.options as { device?: GPUDevice }).device = options.device;
  }
  if (options?.powerPreference !== undefined) {
    (s.options as { powerPreference?: GPUPowerPreference }).powerPreference = options.powerPreference;
  }
  if (s.gpuEngine) {
    try { s.gpuEngine.destroy(); } catch {}
    s.gpuEngine = null;
  }
  s.vramAllocatedBytes = 0;
  await tryInitializeGpuEngine(s);
  return s.engineState.engine === 'webgpu';
}

/**
 * Attempt (re)initialization of the GPU engine from retained CPU state,
 * honoring preferGpu/threshold/memory-budget routing. Never throws for
 * GPU-unavailable environments; engine failures degrade to CPU fallback.
 */
export async function tryInitializeGpuEngine<TDoc>(s: DocumentIndexState<TDoc>): Promise<void> {
  const totalRows = s.rowTokens.length;
  const threshold = s.options.threshold ?? 30_000;
  const shouldAttempt = s.options.preferGpu === false
    ? false
    : s.preferGpu || totalRows >= threshold;

  if (!shouldAttempt) {
    s.engineState = transitionEngineState(
      s.engineState,
      'cpu',
      s.options.preferGpu === false ? 'prefer-cpu' : 'below-threshold',
    );
    return;
  }

  const measuredAvgBytes = totalRows > 0 ? (s.totalTokens * 4) / totalRows : 0;
  const budget = checkMemoryBudget(totalRows, measuredAvgBytes, s.options.device, s.candidateCapacity);
  if (!budget.allowed) {
    s.engineState = transitionEngineState(s.engineState, 'cpu', 'memory-budget-exceeded');
    return;
  }

  try {
    const gpu = new WebGPUEngine();
    const initialized = await gpu.init(
      s.options.device !== undefined || s.options.powerPreference !== undefined
        ? { device: s.options.device, powerPreference: s.options.powerPreference }
        : undefined
    );
    if (initialized && gpu.isReady) {
      gpu.ensureCandidateCapacity(s.candidateCapacity);
      const packed = packDataset(s.rowTokens, {
        normalized: s.normalized,
        totalTokens: s.totalTokens
      });
      const effectiveCap = s.initialCapacity > 0 ? s.initialCapacity : Math.max(16, s.records.length);
      const rowCap = effectiveCap * s.sortedFields.length;
      const avgTokens = totalRows > 0 ? s.totalTokens / totalRows : 16;
      const tokenCap = Math.max(Math.floor(rowCap * avgTokens), 64);
      await gpu.loadDataset(packed, {
        rowCapacity: rowCap,
        tokenCapacity: tokenCap,
        growthFactor: s.growthFactor
      });
      s.gpuEngine = gpu;
      s.engineState = transitionEngineState(s.engineState, 'webgpu');
      s.vramAllocatedBytes = gpu.vramAllocatedBytes;
      subscribeDeviceLost(s);
    } else {
      s.engineState = transitionEngineState(
        s.engineState,
        'cpu',
        typeof navigator === 'undefined' || !('gpu' in navigator) || !navigator.gpu
          ? 'webgpu-unsupported'
          : 'device-request-failed',
      );
    }
  } catch {
    s.engineState = transitionEngineState(
      s.engineState,
      'cpu',
      typeof navigator === 'undefined' || !('gpu' in navigator) || !navigator.gpu
        ? 'webgpu-unsupported'
        : 'device-request-failed',
    );
  }
}

/**
 * Destroy the index: release the GPU engine + device-loss listener, drop the
 * corpus, and reset the engine state so post-destroy calls fail closed.
 */
export function destroyIndex<TDoc>(s: DocumentIndexState<TDoc>): void {
  s.isDestroyed = true;
  s.generation++;
  if (s.unsubscribeDeviceLost) {
    s.unsubscribeDeviceLost();
    s.unsubscribeDeviceLost = undefined;
  }
  if (s.gpuEngine) {
    s.gpuEngine.destroy();
    s.gpuEngine = null;
  }
  s.records = [];
  s.docIds = [];
  s.idToDocIndex.clear();
  s.rowTokens = [];
  s.rowToDocIndex = [];
  s.rowToFieldIndex = [];
  s.docToRowIndices = [];
  s.rawFieldStrings = [];
  s.totalTokens = 0;
  s.tombstones.clear();
  s.columnarStore = new ColumnarStore<TDoc>(s.filterFieldDefinitions, {
    initialCapacity: s.initialCapacity,
    growthFactor: s.growthFactor
  });
  s.engineState = transitionEngineState(
    s.engineState,
    'cpu',
    s.engineState.fallbackReason,
  );
  s.vramAllocatedBytes = 0;
  const p = s.searchMutex.then(() => {}, () => {});
  s.searchMutex = p.catch(() => {});
}
