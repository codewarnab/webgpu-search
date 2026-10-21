/**
 * Snapshot restore/apply implementation for `DocumentIndex` (extracted
 * verbatim from the pre-refactor monolithic `document-index.ts`).
 *
 * Scope of this module:
 * - `assertSnapshotFilterGettersSatisfied`: fail-closed custom filter-getter
 *   restore guard (shared by static `fromSnapshotData` and instance restores;
 *   re-exported from the public `document-index.ts` facade).
 * - `mergeSnapshotRestoreOptions`: pure merge of a reset snapshot + user
 *   restore options into construction options (includes the profile/getter/
 *   hook fail-closed guards for the static restore path).
 * - `configureFromSchema` / `applySnapshotData`: rebuild live index state
 *   from a decoded snapshot, including the CPU re-platform and the
 *   GPU bootstrap attempt.
 *
 * This module must never import the `DocumentIndex` facade at runtime
 * (the facade imports from here); static restore constructs the instance
 * in `document-index.ts` before calling `applySnapshotData`.
 */

import { readOwnField, ColumnarStore } from '../filtering/columnar-store';
import { packDataset, checkMemoryBudget } from '../dataset-packing';
import { transitionEngineState } from '../engine-state';
import { WebGPUEngine } from '../webgpu-engine';
import { assertValidPowerPreference } from '../gpu-device-pool';
import { normalizeSearchHooks, assertHooksSatisfied } from '../hooks';
import { IncompatibleIndexError, IncompatibleOptionError, ProfileMismatchError } from '../text-profile';
import type { RestoredDocumentSnapshot } from '../snapshot-codec';
import type {
  DocumentId,
  DocumentIndexOptions,
  DocumentIndexSchema,
  FilterFieldDefinition,
  FilterFieldType,
  RestoreDocumentIndexOptions,
  SearchHooks,
} from '../types';
import { subscribeDeviceLost } from './build';
import type { DocumentIndexState, InternalField } from './state';

/**
 * fail-closed custom filter-getter restore guard (mirrors the
 * hookIds idiom). Snapshots recording `hasGetter: true` for a filter field
 * require the caller to supply a matching getter override via
 * `options.options.filterFields`; otherwise columnar rebuild via default
 * `doc[name]` would silently change filter semantics.
 */
export function assertSnapshotFilterGettersSatisfied(
  snapshotFilterFields: Array<{ name: string; hasGetter?: boolean }> | undefined,
  userFilterFields: Array<string | { name: string; getter?: unknown }> | undefined
): void {
  if (!snapshotFilterFields || snapshotFilterFields.length === 0) return;
  const needed = snapshotFilterFields.filter((ff) => ff && ff.hasGetter === true);
  if (needed.length === 0) return;
  const userGetterNames = new Set<string>();
  if (Array.isArray(userFilterFields)) {
    for (const uf of userFilterFields) {
      if (uf && typeof uf === 'object' && typeof (uf as { name?: unknown }).name === 'string') {
        const u = uf as { name: string; getter?: unknown };
        if (typeof u.getter === 'function') userGetterNames.add(u.name);
      }
    }
  }
  for (const ff of needed) {
    if (!userGetterNames.has(ff.name)) {
      throw new IncompatibleIndexError(
        `filter-getter:${ff.name}`,
        'missing custom getter override for snapshot filter field (supply options.options.filterFields with getter)'
      );
    }
  }
}

/**
 * Merge a decoded snapshot with user restore options into construction
 * options (verbatim from the pre-refactor `fromSnapshotData` data-merge
 * block; instance construction stays in the facade). Includes the fail-closed
 * profile guards, the filter-getter guard, and the hook restore guard.
 */
export function mergeSnapshotRestoreOptions<TDoc = Record<string, unknown>>(
  snapshot: RestoredDocumentSnapshot<TDoc>,
  options?: RestoreDocumentIndexOptions<TDoc>
): DocumentIndexOptions<TDoc> {
  // Fail-closed profile guards (never read under the wrong profile).
  // `textProfile` only admits 'unicode-default'; any other value throws
  // `ProfileMismatchError`. An explicit `caseSensitive` override must agree
  // with the snapshot polarity (`!header.normalized`); otherwise the
  // retained token stream would be queried under the wrong normalization.
  const rawTextProfile = (options?.options as { textProfile?: unknown } | undefined)?.textProfile;
  if (rawTextProfile !== undefined && rawTextProfile !== 'unicode-default') {
    throw new ProfileMismatchError('unicode-default', rawTextProfile, 'textProfile');
  }
  const overrideCase = options?.options?.caseSensitive;
  if (overrideCase !== undefined && overrideCase === snapshot.header.normalized) {
    throw new ProfileMismatchError(!snapshot.header.normalized, overrideCase, 'caseSensitive');
  }
  // fail-closed custom filter-getter guard. Snapshot `hasGetter`
  // entries require a matching getter override; otherwise restore would
  // silently rebuild columnar via default `doc[name]` (presence cleared).
  assertSnapshotFilterGettersSatisfied(
    snapshot.schema.filterFields as Array<{ name: string; hasGetter?: boolean }> | undefined,
    options?.options?.filterFields as Array<string | { name: string; getter?: unknown }> | undefined
  );
  const userFields = options?.options?.fields;
  const userFieldMap = new Map<string, any>();
  if (Array.isArray(userFields)) {
    for (const uf of userFields) {
      if (typeof uf === 'string') {
        userFieldMap.set(uf, uf);
      } else if (uf && typeof uf === 'object' && typeof uf.name === 'string') {
        userFieldMap.set(uf.name, uf);
      }
    }
  }

  const resolvedFields = snapshot.schema.fields.map((sf) => {
    const uf = userFieldMap.get(sf.name);
    const getter = typeof uf === 'object' && uf !== null && typeof uf.getter === 'function'
      ? uf.getter
      : undefined;
    return {
      name: sf.name,
      weight: sf.weight,
      getter
    };
  });

  const mergedOptions: DocumentIndexOptions<TDoc> = {
    ...(options?.options as any),
    fields: resolvedFields,
    idField: options?.options?.idField ?? snapshot.schema.idField ?? 'id',
    caseSensitive: options?.options?.caseSensitive ?? snapshot.schema.caseSensitive ?? !(snapshot.header.normalized),
    device: (options?.device ?? options?.options?.device) as GPUDevice | undefined,
    preferGpu: options?.options?.preferGpu ?? snapshot.schema.preferGpu,
    threshold: options?.options?.threshold ?? snapshot.schema.threshold,
    candidateCapacity: options?.options?.candidateCapacity ?? snapshot.schema.candidateCapacity,
    initialCapacity: options?.options?.initialCapacity ?? snapshot.schema.initialCapacity,
    growthFactor: options?.options?.growthFactor ?? snapshot.schema.growthFactor,
    filterFields: options?.options?.filterFields ?? (snapshot.schema.filterFields as any),
    hooks: options?.options?.hooks
  };

  // fail-closed hook restore guard. Snapshots recording hookIds
  // require matching handlers via `options.options.hooks`; closures
  // are never serialized, only declarative IDs.
  assertHooksSatisfied(
    snapshot.schema.hookIds,
    normalizeSearchHooks(mergedOptions.hooks)
  );

  return mergedOptions;
}

/**
 * Rebuild the field/id accessors from a snapshot schema (verbatim from the
 * pre-refactor `configureFromSchema`). User-supplied field getters take
 * precedence over the default own-property read.
 */
export function configureFromSchema<TDoc>(
  s: DocumentIndexState<TDoc>,
  schema: DocumentIndexSchema,
  options?: RestoreDocumentIndexOptions<TDoc>
): void {
  const userFields = options?.options?.fields;
  const userFieldMap = new Map<string, any>();
  if (Array.isArray(userFields)) {
    for (const uf of userFields) {
      if (typeof uf === 'string') {
        userFieldMap.set(uf, uf);
      } else if (uf && typeof uf === 'object' && typeof uf.name === 'string') {
        userFieldMap.set(uf.name, uf);
      }
    }
  }

  const normalized: InternalField<TDoc>[] = schema.fields.map(
    (f: { name: string; weight: number }, originalIndex: number) => {
    const name = f.name;
    if (typeof name !== 'string' || name.length === 0) {
      throw new TypeError('[webgpu-search] Restored schema field requires a non-empty string name.');
    }
    const weight = f.weight ?? 1.0;
    if (typeof weight !== 'number' || !Number.isFinite(weight) || weight <= 0) {
      throw new RangeError(
        `[webgpu-search] Restored field "${name}" weight must be a positive finite number, got ${String(weight)}.`
      );
    }
    const uf = userFieldMap.get(name);
    let getter: (doc: TDoc) => string | string[] | undefined | null;
    if (uf && typeof uf === 'object' && typeof uf.getter === 'function') {
      getter = uf.getter;
    } else {
      getter = (doc: any) => readOwnField(doc, name);
    }
    return { name, weight, getter, originalIndex };
  });

  s.sortedFields = [...normalized].sort((a, b) => {
    if (b.weight !== a.weight) return b.weight > a.weight ? 1 : -1;
    if (a.originalIndex !== b.originalIndex) return a.originalIndex > b.originalIndex ? 1 : -1;
    return 0;
  });

  s.fieldNameToIndex.clear();
  for (let i = 0; i < s.sortedFields.length; i++) {
    s.fieldNameToIndex.set(s.sortedFields[i].name, i);
  }

  if (typeof options?.options?.idField === 'function') {
    s.getId = options.options.idField;
  } else if (typeof options?.options?.idField === 'string') {
    const prop = options.options.idField;
    s.getId = (doc: any) => doc[prop];
  } else if (typeof schema.idField === 'string') {
    const prop = schema.idField;
    s.getId = (doc: any) => doc[prop];
  } else if (options?.options?.idField === undefined && schema.idField === undefined) {
    s.getId = (doc: any) => doc.id;
  }
}

/**
 * Apply decoded snapshot data to live index state (verbatim from the
 * pre-refactor `applySnapshotData`). Assumes the mutation mutex is held and
 * the constructor-time profile guards have already run (instance restores in
 * the facade, static restores in `fromSnapshotData`). Ends by bumping
 * `generation` so in-flight searches observe the replace.
 */
export async function applySnapshotData<TDoc>(
  s: DocumentIndexState<TDoc>,
  snapshot: RestoredDocumentSnapshot<TDoc>,
  options?: RestoreDocumentIndexOptions<TDoc>
): Promise<void> {
  if ((options?.options as unknown as Record<string, unknown> | undefined)?.extensions !== undefined) {
    throw new IncompatibleOptionError(
      'hooks',
      "[webgpu-search] extensions was removed; use hooks."
    );
  }
  // Fail-closed profile guards (never read under the wrong profile).
  // The live index normalization is fixed at construction; a snapshot
  // built under the opposite polarity must be rejected, not reinterpreted.
  // An explicit `caseSensitive`/`textProfile` override must agree with both
  // the live index and the snapshot.
  const rawTextProfile = (options?.options as { textProfile?: unknown } | undefined)?.textProfile;
  if (rawTextProfile !== undefined && rawTextProfile !== 'unicode-default') {
    throw new ProfileMismatchError('unicode-default', rawTextProfile, 'textProfile');
  }
  const snapshotCaseSensitive = !snapshot.header.normalized;
  const liveCaseSensitive = !s.normalized;
  if (snapshotCaseSensitive !== liveCaseSensitive) {
    throw new ProfileMismatchError(liveCaseSensitive, snapshotCaseSensitive, 'caseSensitive');
  }
  const overrideCaseApply = options?.options?.caseSensitive;
  if (overrideCaseApply !== undefined && overrideCaseApply !== snapshotCaseSensitive) {
    throw new ProfileMismatchError(snapshotCaseSensitive, overrideCaseApply, 'caseSensitive');
  }
  // fail-closed hook restore guard for instance `restore()`.
  // Restore-supplied handlers (if any) replace the live index hooks;
  // otherwise the live hooks must satisfy the snapshot.
  const restoreHooks = normalizeSearchHooks(options?.options?.hooks);
  if (restoreHooks !== undefined) {
    s.indexHooks = restoreHooks;
    (s.options as { hooks?: SearchHooks<TDoc> }).hooks = restoreHooks;
  }
  assertHooksSatisfied(snapshot.schema.hookIds, s.indexHooks);

  // fail-closed custom filter-getter guard for instance restore.
  // Live filter defs (plus any restore override) must supply getters for
  // snapshot `hasGetter` fields; otherwise columnar rebuild would silently
  // change semantics. When the override supplies the getters, adopt it so
  // `columnarStore.init` uses the correct accessors.
  {
    const overrideFF = options?.options?.filterFields as
      | Array<string | { name: string; type?: FilterFieldType; getter?: (doc: TDoc) => any }>
      | undefined;
    const liveCustom = new Set<string>();
    for (const ff of s.filterFieldDefinitions) {
      if ((ff as { hasGetter?: boolean }).hasGetter === true) liveCustom.add(ff.name as string);
    }
    const overrideGetterNames = new Set<string>();
    if (Array.isArray(overrideFF)) {
      for (const uf of overrideFF) {
        if (uf && typeof uf === 'object' && typeof (uf as { getter?: unknown }).getter === 'function') {
          overrideGetterNames.add((uf as { name: string }).name);
        }
      }
    }
    const needed = ((snapshot.schema.filterFields as Array<{ name: string; hasGetter?: boolean }> | undefined) ?? []).filter(
      (ff) => ff && ff.hasGetter === true
    );
    for (const ff of needed) {
      if (!liveCustom.has(ff.name) && !overrideGetterNames.has(ff.name)) {
        throw new IncompatibleIndexError(
          `filter-getter:${ff.name}`,
          'missing custom getter override for snapshot filter field (supply options.options.filterFields with getter)'
        );
      }
    }
    if (Array.isArray(overrideFF) && overrideFF.length > 0) {
      const normalized: FilterFieldDefinition<TDoc>[] = [];
      const seen = new Set<string>();
      for (const uf of overrideFF) {
        if (typeof uf === 'string') {
          if (seen.has(uf)) continue;
          seen.add(uf);
          normalized.push({ name: uf as never, getter: ((doc: any) => (doc as any)[uf]) as never });
        } else if (uf && typeof uf === 'object' && typeof (uf as { name?: unknown }).name === 'string') {
          const o = uf as { name: string; type?: FilterFieldType; getter?: (doc: TDoc) => any };
          if (seen.has(o.name)) continue;
          seen.add(o.name);
          const hasCustom = typeof o.getter === 'function';
          normalized.push(
            hasCustom
              ? { name: o.name as never, type: o.type, getter: o.getter as never, hasGetter: true }
              : { name: o.name as never, type: o.type, getter: ((doc: any) => (doc as any)[o.name]) as never }
          );
        }
      }
      if (normalized.length > 0) {
        s.filterFieldDefinitions = normalized;
        s.columnarStore = new ColumnarStore<TDoc>(normalized, {
          initialCapacity: s.initialCapacity,
          growthFactor: s.growthFactor
        });
      }
    }
  }

  configureFromSchema(s, snapshot.schema, options);

  if (s.unsubscribeDeviceLost) {
    s.unsubscribeDeviceLost();
    s.unsubscribeDeviceLost = undefined;
  }
  if (s.gpuEngine) {
    s.gpuEngine.destroy();
    s.gpuEngine = null;
  }
  s.tombstones.clear();
  s.mutationEpoch =
    typeof snapshot.schema.mutationEpoch === 'number' &&
    Number.isFinite(snapshot.schema.mutationEpoch) &&
    snapshot.schema.mutationEpoch >= 0
      ? Math.floor(snapshot.schema.mutationEpoch)
      : 0;

  const docCount = snapshot.header.docCount;
  const fieldCount = s.sortedFields.length;
  const totalRows = snapshot.header.rowCount;

  s.records = snapshot.records.slice();
  s.docIds = snapshot.docIds.slice();
  s.idToDocIndex.clear();

  if (s.records.length > 0 && s.docIds.length === 0) {
    s.docIds = new Array(docCount);
    for (let i = 0; i < docCount; i++) {
      const id = s.getId(s.records[i]);
      s.docIds[i] = id;
      s.idToDocIndex.set(id, i);
    }
  } else {
    for (let i = 0; i < s.docIds.length; i++) {
      s.idToDocIndex.set(s.docIds[i], i);
    }
  }

  // If both records and docIds are present, ensure exact ID correspondence alignment
  if (s.records.length === docCount && s.docIds.length === docCount) {
    const recMap = new Map<DocumentId, TDoc>();
    for (const rec of s.records) {
      recMap.set(s.getId(rec), rec);
    }
    if (recMap.size === docCount) {
      const aligned: TDoc[] = new Array(docCount);
      let allFound = true;
      for (let i = 0; i < docCount; i++) {
        const rec = recMap.get(s.docIds[i]);
        if (rec === undefined) {
          allFound = false;
          break;
        }
        aligned[i] = rec;
      }
      if (allFound) {
        s.records = aligned;
      }
    }
  }

  if (s.records.length === 0 && docCount > 0) {
    const idProp = typeof s.options.idField === 'string' ? s.options.idField : 'id';
    s.records = new Array(docCount);
    for (let i = 0; i < docCount; i++) {
      s.records[i] = { [idProp]: s.docIds[i] } as any;
    }
  }

  s.columnarStore.init(s.records);

  const rawCap = s.options.candidateCapacity;
  const validUserCap = typeof rawCap === 'number' && Number.isFinite(rawCap) && rawCap > 0
    ? Math.floor(rawCap)
    : Math.floor(docCount * fieldCount * 0.1);
  s.candidateCapacity = Math.min(32768, Math.max(8192, validUserCap));

  s.rowTokens = new Array(totalRows);
  s.rowToDocIndex = new Array(totalRows);
  s.rowToFieldIndex = new Array(totalRows);
  s.docToRowIndices = Array.from({ length: docCount }, () => new Array(fieldCount));
  s.rawFieldStrings = Array.from({ length: docCount }, () => new Array(fieldCount));

  for (let r = 0; r < totalRows; r++) {
    const start = snapshot.offsets[r];
    const end = snapshot.offsets[r + 1];
    s.rowTokens[r] = snapshot.tokens.subarray(start, end);

    const f = fieldCount > 0 && docCount > 0 ? Math.floor(r / docCount) : 0;
    const d = docCount > 0 ? r % docCount : 0;
    s.rowToDocIndex[r] = d;
    s.rowToFieldIndex[r] = f;
    if (s.docToRowIndices[d]) {
      s.docToRowIndices[d][f] = r;
    }
  }

  s.totalTokens = snapshot.header.tokenCount;

  if (s.records.length === docCount && docCount > 0) {
    for (let d = 0; d < docCount; d++) {
      const doc = s.records[d];
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
        s.rawFieldStrings[d][f] = rawStr;
      }
    }
  }

  s.engineState = transitionEngineState(
    s.engineState,
    'cpu',
    s.engineState.fallbackReason,
  );
  s.vramAllocatedBytes = 0;

  if (docCount > 0) {
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
      const measuredAvgBytes = totalRows > 0 ? (s.totalTokens * 4) / totalRows : 0;
      const deviceToUse = options?.device ?? s.options.device;
      const powerPreference = options?.options?.powerPreference ?? s.options.powerPreference;
      assertValidPowerPreference(powerPreference);
      const budget = checkMemoryBudget(totalRows, measuredAvgBytes, deviceToUse, s.candidateCapacity);
      if (!budget.allowed) {
        s.engineState = transitionEngineState(s.engineState, 'cpu', 'memory-budget-exceeded');
      } else {
        try {
          const gpu = new WebGPUEngine();
          const initialized = await gpu.init(
            deviceToUse !== undefined || powerPreference !== undefined
              ? { device: deviceToUse ?? undefined, powerPreference }
              : undefined
          );
          if (initialized && gpu.isReady) {
            gpu.ensureCandidateCapacity(s.candidateCapacity);
            const packed = packDataset(s.rowTokens, {
              normalized: s.normalized,
              totalTokens: s.totalTokens
            });
            const effectiveCap = s.initialCapacity > 0 ? s.initialCapacity : Math.max(16, docCount);
            const rowCap = effectiveCap * fieldCount;
            const avgTokens = totalRows > 0 ? s.totalTokens / totalRows : 16;
            const tokenCap = Math.max(Math.floor(rowCap * avgTokens), 64);
            await gpu.loadDataset(packed, {
              rowCapacity: rowCap,
              tokenCapacity: tokenCap,
              growthFactor: s.initialCapacity > 0 ? 1.0 : s.growthFactor
            });

            s.gpuEngine = gpu;
            s.engineState = transitionEngineState(s.engineState, 'webgpu');
            s.vramAllocatedBytes = gpu.vramAllocatedBytes;

            subscribeDeviceLost(s);
          } else {
            s.engineState = transitionEngineState(s.engineState, 'cpu', 'device-request-failed');
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
    }
  }

  s.generation++;
}
