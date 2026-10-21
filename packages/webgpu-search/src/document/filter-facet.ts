/**
 * Search filter compilation + facet aggregation (extracted verbatim from
 * the pre-refactor monolithic `document-index.ts`). The public
 * `DocumentIndex` facade delegates here via `document/query.ts`.
 *
 * Scope of this module:
 * - `compileFilterPhase`: compiles `FilterExpression` to a bitset, measures
 *   `filteringMs`, enforces abort/budget gates, and composes extension
 *   predicates conjunctively.
 * - `buildFacetResults`: aggregates facet buckets over an explicit
 *   query-matched doc set with disjunctive filter exclusion.
 *
 * All functions operate on the shared `DocumentIndexState` bundle and are
 * read-only with respect to index internals.
 */

import type { DocumentBitset } from '../filtering/doc-bitset';
import { compileFilter } from '../filtering/compile-filter';
import { excludeFieldFromFilter, type FacetEngine, type NormalizedFacet } from '../faceting/facet-engine';
import { nowMs, throwIfAborted } from '../guard';
import { throwIfBudgetAborted, assertTimeBudget, type NormalizedCostBudget } from '../diagnostics';
import type {
  DocumentSearchOptions,
  FacetResult,
  FilterExpression,
  SearchHooks,
} from '../types';
import type { DocumentIndexState } from './state';

/**
 * Normalized cost budget shape (validated via `normalizeCostBudgetOptions`,
 * which may return `undefined` when no budget is configured). Shared by the
 * search validation, filter compilation, and engine routing phases.
 */
export type CostBudget = NormalizedCostBudget;
/**
 * Compile structured/predicate filters (extracted from `search`).
 * Compiles `FilterExpression` to a bitset, measures `filteringMs`,
 * enforces abort/budget gates, and composes extension predicates.
 * Empty bitsets return early with `filteringMs: 0` so callers can take
 * the `noHits` path with identical telemetry. No behavior change.
 */
export function compileFilterPhase<TDoc>(
  s: DocumentIndexState<TDoc>,
  filter: DocumentSearchOptions<TDoc>['filter'],
  effectiveHooks: SearchHooks<TDoc> | undefined,
  wantsDiagnostics: boolean,
  signal: AbortSignal | undefined,
  budget: CostBudget | undefined,
  queryStartMs: number
) {
  let filterBitset: DocumentBitset | undefined = undefined;
  let filterPredicate: ((doc: TDoc) => boolean) | undefined = undefined;
  let structuredFilter: FilterExpression | undefined = undefined;
  let filteringMs = 0;

  const tFilter0 = wantsDiagnostics ? nowMs() : 0;
  if (filter !== undefined) {
    if (typeof filter === 'function') {
      filterPredicate = filter;
    } else if (typeof filter === 'object' && filter !== null) {
      structuredFilter = filter as FilterExpression;
      filterBitset = compileFilter(structuredFilter, s.columnarStore);
      if (filterBitset.isEmpty()) {
        return { filterBitset, filterPredicate, structuredFilter, filteringMs };
      }
    } else {
      throw new TypeError('[webgpu-search] options.filter must be a function or FilterExpression.');
    }
  }
  if (wantsDiagnostics) filteringMs = nowMs() - tFilter0;
  throwIfAborted(signal);
  throwIfBudgetAborted(budget);
  assertTimeBudget(queryStartMs, budget);

  // extension filter predicates compose conjunctively (AND) with
  // `options.filter` functions. Index-level vs per-query hooks already
  // merged via `resolveEffectiveHooks` (per-query wins per key).
  const extensionPredicate = effectiveHooks?.filterPredicate;
  if (extensionPredicate !== undefined) {
    const basePredicate = filterPredicate;
    if (basePredicate === undefined) {
      filterPredicate = extensionPredicate;
    } else {
      const optFn = basePredicate;
      const extFn = extensionPredicate;
      filterPredicate = (doc: TDoc) => {
        const d = doc as TDoc;
        return optFn(d) && extFn(d);
      };
    }
  }

  return { filterBitset, filterPredicate, structuredFilter, filteringMs };
}

/**
 * Aggregates facet buckets over an explicit query-matched doc set with
 * disjunctive filter exclusion. Each facet ignores structured filter
 * clauses on its own field (`excludeFieldFromFilter`) while keeping all
 * other clauses; function predicates stay conjunctive. `or`/`not` branches
 * referencing the facet field are kept verbatim (conservative, aliased).
 * Cost is O(F*n) with F bitset allocs plus an optional extra unfiltered
 * parity scan when a structured filter is present.
 */
export function buildFacetResults<TDoc>(
  s: DocumentIndexState<TDoc>,
  facetEngine: FacetEngine<TDoc>,
  facetSpec: NormalizedFacet[],
  queryMatched: Iterable<number>,
  structuredFilter: FilterExpression | undefined,
  filterPredicate: ((doc: TDoc) => boolean) | undefined,
  isApproximate: boolean
): Record<string, FacetResult> {
  const base: number[] = [];
  for (const dIdx of queryMatched) {
    const doc = s.records[dIdx];
    if (doc === null || doc === undefined) continue;
    if (filterPredicate && !filterPredicate(doc)) continue;
    base.push(dIdx);
  }
  const out: Record<string, FacetResult> = {};
  for (const facet of facetSpec) {
    let candidates: number[] = base;
    if (structuredFilter !== undefined) {
      const excluded = excludeFieldFromFilter(structuredFilter, facet.request.field);
      if (excluded !== undefined) {
        const mask = compileFilter(excluded, s.columnarStore);
        candidates = base.filter((d) => mask.has(d));
      }
    }
    const result = facetEngine.aggregateOne(facet, candidates, isApproximate);
    if (facet.name === '__proto__') {
      Object.defineProperty(out, facet.name, { value: result, enumerable: true, configurable: true, writable: true });
    } else {
      out[facet.name] = result;
    }
  }
  return out;
}
