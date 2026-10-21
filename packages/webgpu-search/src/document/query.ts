/**
 * Document-index query implementation (extracted verbatim from the
 * pre-refactor monolithic `document-index.ts). The public `DocumentIndex`
 * facade delegates `search` / `autocomplete` here.
 *
 * Scope of this module:
 * - `validateSearchOptions`: fail-closed option validation (query shape,
 *   mode/scorer/profile checks, token/prefix/typo normalization, hooks,
 *   budget/diagnostics, query-length gates, facet, ranking/autocomplete,
 *   field scoping, tokenizer expansion).
 * - `runSearch`: GPU/CPU routed execution with facets, inline
 *   autocomplete, diagnostics, and abort/budget gates.
 * - `runAutocomplete` + `computeSuggestions`: standalone and inline
 *   suggestion enumeration.
 * - `enrichResults` / `enrichHighlights` / `resortResultsAfterScoring`:
 *   Top-K highlight + post-match hook pipeline.
 * - `routeEngine`: candidate-budget + broad-query + field-restriction
 *   routing decisions.
 * - `aggregateDocMatches`: per-document best-field aggregation delegator.
 *
 * All functions operate on the shared `DocumentIndexState` bundle;
 * mutations are serialized through the `searchMutex` by the facade.
 */

import { CPUEngine } from '../cpu-engine';
import {
  scoreExactMatchesMultiField,
  scoreFuzzyTokens,
  scoreSubstringTypoTokens,
  type CpuModeOptions
} from '../exact-scorer';
import {
  normalizeTypoTolerance,
  type NormalizedTypoOptions
} from '../search/typo-tolerance';
import {
  normalizeTokenMatchOptions,
  type NormalizedTokenMatchOptions
} from '../search/token-search';
import {
  normalizePrefixOptions,
  assertPrefixLengthForQuery,
  scorePrefixTokens,
  type NormalizedPrefixOptions
} from '../search/prefix-search';
import {
  compareRanked,
  isExactTokenMatch,
  normalizeTieBreakers,
  type RankableCandidate
} from '../ranking';
import {
  aggregateDocMatches as aggregateDocMatchesPure,
  accumulateDocMatch,
} from '../document-aggregate';
import {
  normalizeAutocompleteOptions,
  type NormalizedAutocompleteOptions
} from '../autocomplete';
import {
  resolveEffectiveHooks,
  getTokenTermsForQuery,
  applyScoringHook,
  applyPostProcess,
} from '../hooks';
import {
  clampLimit,
  throwIfAborted,
  abortError,
  nowMs,
  assertValidMode
} from '../guard';
import {
  throwIfBudgetAborted,
  assertTimeBudget,
  assertCandidateBudget,
  computeFilterSelectivity,
  isBroadQueryHeuristic,
  isBroadSelectivity,
  broadQueryRouteWarning,
  broadSelectivityWarning,
  candidateOverflowWarning,
  normalizeCostBudgetOptions
} from '../diagnostics';
import {
  QUERY_TOKENS_MAX,
  SCORING_VERSION,
  normalizeCpuScorer,
  IncompatibleOptionError,
  ProfileMismatchError,
  QueryTooLongError,
} from '../text-profile';
import { transitionEngineState } from '../engine-state';
import type { WebGPUEngine } from '../webgpu-engine';
import { normalizeText } from '../text-normalization';
import type { DocumentBitset } from '../filtering/doc-bitset';
import {
  FacetEngine,
  normalizeFacetRequests
} from '../faceting/facet-engine';
import { alignHighlights, renderHighlightedText } from '../highlight';
import { compileFilterPhase, buildFacetResults, type CostBudget } from './filter-facet';
import { setOwnKey } from './build';
import type { DocumentIndexState } from './state';
import type {
  AutocompleteOptions,
  DocumentSearchOptions,
  DocumentSearchResponse,
  DocumentSearchResultItem,
  EngineType,
  FacetResult,
  HighlightRange,
  QueryDiagnostics,
  QueryDiagnosticsTimings,
  SearchHooks,
  SearchMode,
  SearchTimings,
  SuggestionItem,
  SuggestResponse,
  TieBreakerCriterion,
} from '../types';

/**
 * Validate search options fail-closed (extracted from `search`).
 * Covers query shape, mode/scorer/profile checks, token/prefix/typo
 * normalization, hooks, budget/diagnostics, query-length gates, facet,
 * ranking/autocomplete, field scoping, and tokenizer expansion.
 * No behavior change: code moved verbatim from `search`.
 */
export function validateSearchOptions<TDoc>(
  s: DocumentIndexState<TDoc>,
  query: string,
  options: DocumentSearchOptions<TDoc>
) {
  if (typeof query !== 'string') {
    throw new TypeError(`[webgpu-search] search expects query: string, got ${typeof query}.`);
  }
  if ((options as unknown as Record<string, unknown>).cpuAlgorithm !== undefined) {
    throw new IncompatibleOptionError(
      'cpuScorer',
      "[webgpu-search] cpuAlgorithm was removed; use cpuScorer: 'exact' | 'ufuzzy'."
    );
  }
  if ((options as unknown as Record<string, unknown>).extensions !== undefined) {
    throw new IncompatibleOptionError(
      'hooks',
      "[webgpu-search] extensions was removed; use hooks."
    );
  }
  if ((options as unknown as Record<string, unknown>).suggest !== undefined) {
    throw new IncompatibleOptionError(
      'autocomplete',
      "[webgpu-search] suggest was removed; use autocomplete."
    );
  }

  const {
    mode = 'fuzzy',
    caseSensitive = false,
    signal,
    cpuScorer: cpuScorerOpt,
    onQueryTooLong = 'throw',
    fields: searchFields,
    facets,
    faceting
  } = options;

  const cpuScorer = normalizeCpuScorer(cpuScorerOpt) ?? 'exact';
  if (caseSensitive === s.normalized) {
    throw new ProfileMismatchError(!s.normalized, caseSensitive);
  }
  if (s.preferGpu && cpuScorer === 'ufuzzy') {
    throw new IncompatibleOptionError(
      'cpuScorer',
      "cpuScorer:'ufuzzy' is CPU-only; use preferGpu:false or cpuScorer:'exact'."
    );
  }
  assertValidMode(mode);
  // fail-closed option validation up front so malformed
  // token/prefix/typo shapes throw identically on GPU and CPU paths.
  // 'fuzzy' validates typo shape but ignores it (subsequence matching is
  // inherently typo-tolerant); 'substring'/'token'/'prefix' honor it.
  const tokenOpts: NormalizedTokenMatchOptions = normalizeTokenMatchOptions(options.tokenMatch);
  const prefixOpts: NormalizedPrefixOptions = normalizePrefixOptions(options.prefixMatch);
  const typo: NormalizedTypoOptions = normalizeTypoTolerance(options.typoTolerance);
  if (mode === 'prefix' && options.prefixMatch?.exactCase !== undefined && prefixOpts.exactCase !== caseSensitive) {
    throw new ProfileMismatchError(caseSensitive, prefixOpts.exactCase, 'prefixMatch.exactCase');
  }
  // Legacy ufuzzy/native scorers only implement fuzzy/substring-exact.
  if (cpuScorer === 'ufuzzy' && (mode === 'token' || mode === 'prefix' || typo.enabled)) {
    throw new IncompatibleOptionError(
      'cpuScorer',
      `cpuScorer:'ufuzzy' supports only exact 'fuzzy'/'substring' modes without typo tolerance (got mode '${mode}'${typo.enabled ? ' with typoTolerance' : ''}). Use cpuScorer:'exact'.`
    );
  }
  // fail-closed extension hook validation up front so malformed
  // hook shapes throw identically on GPU and CPU paths (before early exits).
  const effectiveHooks = resolveEffectiveHooks(s.indexHooks, options.hooks);
  // Threaded into every exact CPU call below (single normalization).
  const cpuModeOptions: CpuModeOptions = {
    tokenMatch: { operator: tokenOpts.operator, minMatchCount: tokenOpts.minMatchCount },
    prefixMatch: prefixOpts.prefixLength !== undefined
      ? { prefixLength: prefixOpts.prefixLength, exactCase: prefixOpts.exactCase }
      : { exactCase: prefixOpts.exactCase },
    typoTolerance: {
      enabled: typo.enabled,
      maxDistance: typo.maxDistance,
      minWordLengthForOneTypo: typo.minWordLengthForOneTypo,
      minWordLengthForTwoTypos: typo.minWordLengthForTwoTypos,
      prefixExactLength: typo.prefixExactLength
    }
  };
  // fail-closed cost-budget + diagnostics validation up front so
  // malformed budgets throw identically on GPU and CPU paths and on empty
  // corpora/queries (before early exits). Enforcement is independent of the
  // diagnostics flag: budgets constrain even when telemetry is off.
  const budget = normalizeCostBudgetOptions(options.budget);
  if (options.diagnostics !== undefined && typeof options.diagnostics !== 'boolean') {
    throw new TypeError('[webgpu-search] options.diagnostics must be a boolean.');
  }
  const wantsDiagnostics = options.diagnostics === true;
  const needsClock = wantsDiagnostics || budget?.maxExecutionTimeMs !== undefined;
  const queryStartMs = needsClock ? nowMs() : 0;
  const diagWarnings: string[] = [];

  const rawTrimmed = query.trim();
  let forceCpu = false;
  if (rawTrimmed.length > QUERY_TOKENS_MAX * 4) {
    if (rawTrimmed.length > 1_000_000) {
      let cpCount = 0;
      for (const _ch of rawTrimmed) cpCount++;
      const est = cpCount * 3;
      if (est > QUERY_TOKENS_MAX) {
        if (onQueryTooLong !== 'cpu-fallback') {
          throw new QueryTooLongError(QUERY_TOKENS_MAX, est, s.profileId);
        }
        forceCpu = true;
      }
    }
  }

  const normalizedQuery = normalizeText(query, s.normalized);
  const queryTokenCount = normalizedQuery.tokenCount;
  if (queryTokenCount > QUERY_TOKENS_MAX) {
    if (onQueryTooLong === 'cpu-fallback') {
      forceCpu = true;
    } else {
      throw new QueryTooLongError(QUERY_TOKENS_MAX, queryTokenCount, s.profileId);
    }
  }

  const clampedLimit = clampLimit(options.limit ?? options.maxResults ?? 50);
  throwIfAborted(signal);
  throwIfBudgetAborted(budget);

  // Hoisted prefixLength check: fail-closed even on empty queries/corpora.
  // Skipped for empty queries to match the scorer's early noMatch.
  if (mode === 'prefix' && normalizedQuery.tokens.length > 0) {
    assertPrefixLengthForQuery(prefixOpts, normalizedQuery.tokens.length);
  }

  // custom tokenizer terms for 'token' mode (CPU-only).
  // Other modes ignore the tokenizer. Empty queries skip hook invocation
  // (no-match by design); non-empty token queries expand via the hook so
  // scoring and highlights share one term set (score-highlight symmetry).
  let customTokenTerms: Uint32Array[] | undefined = undefined;
  if (effectiveHooks?.tokenizer !== undefined && mode === 'token' && !normalizedQuery.isEmpty) {
    customTokenTerms = getTokenTermsForQuery(
      query,
      normalizedQuery.tokens,
      s.normalized,
      effectiveHooks.tokenizer
    );
    cpuModeOptions.tokenTermsOverride = customTokenTerms;
  }

  // (): validate facet contracts fail-fast, before early exits.
  // An empty facet list is equivalent to not requesting facets (absent key).
  // `faceting` is ignored unless facets are requested (no throw on absent).
  const facetSpec = normalizeFacetRequests(facets);
  const wantsFacets = facetSpec !== undefined && facetSpec.length > 0;
  let facetingMode: 'auto' | 'force-exact' = 'auto';
  if (wantsFacets && faceting !== undefined) {
    if (faceting !== 'auto' && faceting !== 'force-exact') {
      throw new TypeError(`[webgpu-search] Invalid faceting option: "${String(faceting)}". Must be 'auto' or 'force-exact'.`);
    }
    facetingMode = faceting;
  }
  const facetEngine = wantsFacets ? new FacetEngine<TDoc>(s.columnarStore) : null;
  if (wantsFacets && facetEngine && facetSpec) {
    // Fail-closed field validation up front: unknown facet fields and
    // range-on-non-number must throw even when an empty match set would
    // otherwise take an early noHits exit below.
    facetEngine.validateRequests(facetSpec);
  }

  // Inline autocomplete config: deterministic ranking hierarchy.
  // All scope validation (ranking, autocomplete, fields) is fail-closed before
  // early exits so malformed shapes throw identically on empty and
  // non-empty corpora/queries.
  if (options.ranking !== undefined && (typeof options.ranking !== 'object' || options.ranking === null || Array.isArray(options.ranking))) {
    throw new TypeError('[webgpu-search] options.ranking must be an object.');
  }
  const tieBreakers: TieBreakerCriterion[] = normalizeTieBreakers(
    options.ranking?.tieBreakers
  );
  let suggestSpec: NormalizedAutocompleteOptions | undefined = undefined;
  const suggestRaw = options.autocomplete;
  if (suggestRaw !== undefined && suggestRaw !== false) {
    suggestSpec = normalizeAutocompleteOptions(suggestRaw as AutocompleteOptions | boolean);
    // Inline autocomplete inherits the search ranking hierarchy unless the
    // autocomplete object carries its own explicit tieBreakers.
    const rawSuggest = suggestRaw as AutocompleteOptions;
    if (
      typeof rawSuggest === 'object' &&
      rawSuggest !== null &&
      rawSuggest.tieBreakers === undefined &&
      options.ranking?.tieBreakers !== undefined
    ) {
      suggestSpec.tieBreakers = [...tieBreakers];
    }
    if (suggestSpec.field !== undefined && !s.fieldNameToIndex.has(suggestSpec.field)) {
      throw new Error(`[webgpu-search] Unknown autocomplete field: "${suggestSpec.field}".`);
    }
  }
  // Hoisted search-field validation (fail-closed before empty-query,
  // empty-corpus, and empty-filter early exits below).
  let allowedFieldIndices: Set<number> | undefined = undefined;
  if (searchFields !== undefined) {
    if (!Array.isArray(searchFields)) {
      throw new TypeError('[webgpu-search] options.fields must be an array of string field names.');
    }
    if (searchFields.length !== 0) {
      allowedFieldIndices = new Set<number>();
      for (const fName of searchFields) {
        if (typeof fName !== 'string') {
          throw new TypeError('[webgpu-search] options.fields elements must be strings.');
        }
        const idx = s.fieldNameToIndex.get(fName);
        if (idx === undefined) {
          throw new Error(`[webgpu-search] Unknown search field: "${fName}".`);
        }
        allowedFieldIndices.add(idx);
      }
    }
  }
  const hasEmptyFieldList = searchFields !== undefined && searchFields.length === 0;

  return {
    mode,
    caseSensitive,
    signal,
    cpuScorer,
    tokenOpts,
    prefixOpts,
    typo,
    effectiveHooks,
    cpuModeOptions,
    budget,
    wantsDiagnostics,
    queryStartMs,
    diagWarnings,
    normalizedQuery,
    clampedLimit,
    forceCpu,
    facetSpec,
    wantsFacets,
    facetingMode,
    facetEngine,
    tieBreakers,
    suggestSpec,
    allowedFieldIndices,
    hasEmptyFieldList,
    customTokenTerms
  };
}

/**
 * Route to WebGPU vs CPU (extracted from `search`).
 * Applies candidate-budget enforcement, broad-query CPU routing, and
 * field-restriction routing. Mutates `diagWarnings` for the broad-query
 * warning exactly as before. No behavior change.
 */
export function routeEngine<TDoc>(
  s: DocumentIndexState<TDoc>,
  params: {
    mode: SearchMode;
    typoEnabled: boolean;
    cpuScorer: string;
    forceCpu: boolean;
    allowedFieldIndices: Set<number> | undefined;
    normalizedQueryTokenCount: number;
    filterBitset: DocumentBitset | undefined;
    budget: CostBudget | undefined;
    queryStartMs: number;
    wantsDiagnostics: boolean;
    diagWarnings: string[];
  }
) {
  const {
    mode,
    typoEnabled,
    cpuScorer,
    forceCpu,
    allowedFieldIndices,
    normalizedQueryTokenCount,
    filterBitset,
    budget,
    queryStartMs,
    wantsDiagnostics,
    diagWarnings
  } = params;
  // token/prefix modes and typo-tolerant queries are CPU-only
  // (exact-only WGSL kernels) and skip GPU dispatch with fallbackReason
  // 'unsupported-mode' (parity boundary).
  const isGpuSupportedMode: boolean =
    (mode === 'fuzzy' || mode === 'substring') && !typoEnabled;

  // broad-query pre-dispatch guard + candidate ceiling.
  // `filteredCandidateCount` is the exact post-filter population to score;
  // exceeding `maxCandidates` throws before any scoring work. Short queries
  // over massive corpora route to the CPU streaming scan to avoid GPU
  // buffer saturation and driver timeouts (TDR). With function-predicate
  // filters the pre-predicate population is checked (conservative — the
  // true post-predicate count is unknowable pre-scan).
  const activeDocCount = s.idToDocIndex.size;
  const filteredCandidateCount = filterBitset !== undefined
    ? filterBitset.popcount()
    : activeDocCount;
  assertCandidateBudget(filteredCandidateCount, budget);
  let broadQueryCpuRoute = false;
  if (isBroadQueryHeuristic(normalizedQueryTokenCount, activeDocCount)) {
    broadQueryCpuRoute = true;
    // Suppress the GPU-avoidance warning when already CPU-by-design
    // (token/prefix/typo modes, explicit ufuzzy, query-too-long fallback):
    // routing outcome is correct but the stated cause would misattribute.
    const cpuByDesign = !isGpuSupportedMode || cpuScorer === 'ufuzzy' || forceCpu;
    if (wantsDiagnostics && !cpuByDesign) {
      diagWarnings.push(broadQueryRouteWarning(activeDocCount, normalizedQueryTokenCount));
    }
  }
  assertTimeBudget(queryStartMs, budget);

  // When field-restricted, route to CPU to prevent unselected high-priority
  // fields from saturating GPU candidate buffer before low-priority allowed fields.
  const isFieldRestricted = allowedFieldIndices !== undefined && allowedFieldIndices.size < s.sortedFields.length;

  // 1. WebGPU execution path (exact fuzzy/substring only; token/prefix
  // and typo queries skip dispatch via isGpuSupportedMode above).
  const gpuHandle = s.gpuEngine;
  const useGpu = !forceCpu &&
    !isFieldRestricted &&
    !broadQueryCpuRoute &&
    isGpuSupportedMode &&
    cpuScorer !== 'ufuzzy' &&
    s.engineState.engine === 'webgpu' &&
    gpuHandle !== null &&
    gpuHandle.isReady;

  return {
    isGpuSupportedMode,
    broadQueryCpuRoute,
    isFieldRestricted,
    filteredCandidateCount,
    useGpu,
    gpuHandle
  };
}

/**
 * Aggregate per-row matches into per-document best-field hits (thin
 * delegator over the pure `document-aggregate` helper so GPU/CPU paths
 * share one ranking implementation; unit-tested without a device).
 * No behavior change.
 */
export function aggregateDocMatches<TDoc>(
  s: DocumentIndexState<TDoc>,
  docMatches: Map<number, { bestScore: number; bestFieldIdx: number; fieldScores: Map<number, number> }>,
  normalizedQueryTokens: Uint32Array,
  tieBreakers: TieBreakerCriterion[]
) {
  return aggregateDocMatchesPure<TDoc>(docMatches, {
    sortedFields: s.sortedFields,
    docToRowIndices: s.docToRowIndices,
    rowTokens: s.rowTokens,
    docIds: s.docIds,
    records: s.records,
    normalizedQueryTokens,
    tieBreakers
  });
}

/**
 * deterministic re-sort after `scoringHook` boosts.
 * Reconstructs rank keys from index state (field weight, exactness via
 * `isExactTokenMatch` on the winning row, matched length, id, docIndex)
 * with updated scores, then sorts via `compareRanked` + `tieBreakers`.
 * Preserves the total-order contract (deterministic across engines).
 */
export function resortResultsAfterScoring<TDoc>(
  s: DocumentIndexState<TDoc>,
  results: DocumentSearchResultItem<TDoc>[],
  queryTokens: Uint32Array,
  tieBreakers: TieBreakerCriterion[]
): void {
  if (results.length <= 1) return;
  const keys: RankableCandidate[] = new Array(results.length);
  for (let i = 0; i < results.length; i++) {
    const item = results[i] as DocumentSearchResultItem<TDoc>;
    const dIdx = s.idToDocIndex.get(item.id);
    const fIdx = s.fieldNameToIndex.get(item.matchedField);
    if (dIdx !== undefined && fIdx !== undefined) {
      const rowIdx = s.docToRowIndices[dIdx]?.[fIdx];
      const rowTok = rowIdx !== undefined ? s.rowTokens[rowIdx] : undefined;
      const weight = s.sortedFields[fIdx]?.weight ?? 1.0;
      keys[i] = {
        score: item.score,
        fieldWeight: weight,
        isExactMatch: rowTok !== undefined ? isExactTokenMatch(rowTok, queryTokens) : false,
        matchedLength: rowTok !== undefined ? rowTok.length : 0,
        id: item.id,
        docIndex: dIdx
      };
    } else {
      // Defensive fallback (unreachable in normal flow: resort runs before
      // postProcess, so IDs/fields resolve). Positional docIndex preserves
      // the total-order guarantee via unique positions.
      keys[i] = {
        score: item.score,
        fieldWeight: 1.0,
        isExactMatch: false,
        matchedLength: 0,
        id: item.id,
        docIndex: i
      };
    }
  }
  const order = results.map((_, i) => i);
  order.sort((ia, ib) => compareRanked(keys[ia] as RankableCandidate, keys[ib] as RankableCandidate, tieBreakers));
  const sorted = order.map((i) => results[i] as DocumentSearchResultItem<TDoc>);
  for (let i = 0; i < results.length; i++) {
    results[i] = sorted[i] as DocumentSearchResultItem<TDoc>;
  }
}

export function enrichHighlights<TDoc>(
  s: DocumentIndexState<TDoc>,
  results: DocumentSearchResultItem<TDoc>[],
  query: string,
  mode: SearchMode,
  options: DocumentSearchOptions<TDoc>,
  customTokenTerms?: Uint32Array[],
  precomputedTokens?: Uint32Array
): void {
  const shouldHighlight = options.highlightOptions?.highlight ?? options.highlight ?? true;
  if (!shouldHighlight) return;

  const tag = options.highlightOptions?.tag ?? options.tag;
  const highlightFieldsRule = options.highlightOptions?.fields ?? options.highlightFields ?? 'all-matched';
  const escapeHtml = options.highlightOptions?.escapeHtml ?? options.escapeHtml ?? false;

  if (
    typeof highlightFieldsRule === 'string' &&
    highlightFieldsRule !== 'all-matched' &&
    highlightFieldsRule !== 'matched-field' &&
    highlightFieldsRule !== 'all-fields'
  ) {
    throw new TypeError(
      `[webgpu-search] Invalid highlightFields option: "${highlightFieldsRule}". Must be 'all-matched', 'matched-field', 'all-fields', or an array of field names.`
    );
  }

  const queryTokens = precomputedTokens ?? normalizeText(query, s.normalized).tokens;
  // thread token/prefix/typo options so highlight ranges track
  // the serving scorer (score-highlight symmetry across all modes).
  // custom tokenizer terms replace the default split when supplied.
  const alignOpts = {
    mode,
    normalized: s.normalized,
    queryTokens,
    tokenMatch: options.tokenMatch,
    prefixMatch: options.prefixMatch,
    typoTolerance: options.typoTolerance,
    ...(customTokenTerms !== undefined ? { tokenTermsOverride: customTokenTerms } : {})
  };

  for (let i = 0; i < results.length; i++) {
    const item = results[i];
    const dIdx = s.idToDocIndex.get(item.id);
    if (dIdx === undefined) continue;

    const highlights: Record<string, HighlightRange[]> = {};
    const highlightedText: Record<string, string> = {};

    if (highlightFieldsRule === 'all-fields') {
      for (let f = 0; f < s.sortedFields.length; f++) {
        const fieldDef = s.sortedFields[f];
        const rawStr = s.rawFieldStrings[dIdx][f] ?? '';
        const ranges = alignHighlights(rawStr, query, alignOpts);
        if (ranges.length > 0) {
          setOwnKey(highlights, fieldDef.name, ranges);
          if (tag) {
            setOwnKey(highlightedText, fieldDef.name, renderHighlightedText(rawStr, ranges, tag, escapeHtml));
          }
        }
      }
    } else if (Array.isArray(highlightFieldsRule)) {
      for (let k = 0; k < highlightFieldsRule.length; k++) {
        const fName = highlightFieldsRule[k];
        const fIdx = s.fieldNameToIndex.get(fName);
        if (fIdx !== undefined) {
          const rawStr = s.rawFieldStrings[dIdx][fIdx] ?? '';
          const ranges = alignHighlights(rawStr, query, alignOpts);
          if (ranges.length > 0) {
            setOwnKey(highlights, fName, ranges);
            if (tag) {
              setOwnKey(highlightedText, fName, renderHighlightedText(rawStr, ranges, tag, escapeHtml));
            }
          }
        }
      }
    } else {
      // 'matched-field' or 'all-matched' (default)
      const primaryFIdx = s.fieldNameToIndex.get(item.matchedField);
      if (primaryFIdx !== undefined) {
        const primaryRaw = s.rawFieldStrings[dIdx][primaryFIdx] ?? '';
        const ranges = alignHighlights(primaryRaw, query, alignOpts);
        setOwnKey(highlights, item.matchedField, ranges);
        if (tag) {
          setOwnKey(highlightedText, item.matchedField, renderHighlightedText(primaryRaw, ranges, tag, escapeHtml));
        }
      }

      if (highlightFieldsRule === 'all-matched' && item.matches) {
        for (let m = 0; m < item.matches.length; m++) {
          const aux = item.matches[m];
          const auxFIdx = s.fieldNameToIndex.get(aux.field);
          if (auxFIdx !== undefined) {
            const auxRaw = s.rawFieldStrings[dIdx][auxFIdx] ?? '';
            const ranges = alignHighlights(auxRaw, query, alignOpts);
            aux.highlights = ranges;
            setOwnKey(highlights, aux.field, ranges);
            if (tag) {
              setOwnKey(highlightedText, aux.field, renderHighlightedText(auxRaw, ranges, tag, escapeHtml));
            }
          }
        }
      }
    }

    // Synchronize auxiliary match highlights across all modes if highlights exist
    if (item.matches) {
      for (let m = 0; m < item.matches.length; m++) {
        const aux = item.matches[m];
        if (Object.prototype.hasOwnProperty.call(highlights, aux.field)) {
          aux.highlights = highlights[aux.field];
        }
      }
    }

    item.highlights = highlights;
    if (tag) {
      item.highlightedText = highlightedText;
    }
  }
}

/**
 * Enrich Top-K results with highlights + post-match hooks (extracted).
 * Runs highlight enrichment, scoring-hook boost + deterministic re-sort,
 * and postProcess transform. Returns updated results with phase timings.
 * No behavior change.
 */
export function enrichResults<TDoc>(
  s: DocumentIndexState<TDoc>,
  results: DocumentSearchResultItem<TDoc>[],
  query: string,
  mode: SearchMode,
  options: DocumentSearchOptions<TDoc>,
  customTokenTerms: Uint32Array[] | undefined,
  normalizedQueryTokens: Uint32Array,
  effectiveHooks: SearchHooks<TDoc> | undefined,
  tieBreakers: TieBreakerCriterion[],
  wantsDiagnostics: boolean
) {
  const tHl0 = wantsDiagnostics ? nowMs() : 0;
  enrichHighlights(s, results, query, mode, options, customTokenTerms, normalizedQueryTokens);
  const highlightMs = wantsDiagnostics ? nowMs() - tHl0 : 0;

  // post-match extension pipeline (Top-K only). Scoring boosts
  // apply identically on GPU and CPU paths (parity preserved), followed
  // by a deterministic re-sort; postProcess is the final transform.
  // hook time joins the scoring bucket on all paths for comparability.
  const tHook0 = wantsDiagnostics ? nowMs() : 0;
  if (effectiveHooks?.scoringHook !== undefined) {
    applyScoringHook(results, effectiveHooks.scoringHook, query);
    resortResultsAfterScoring(s, results, normalizedQueryTokens, tieBreakers);
  }
  let out = results;
  if (effectiveHooks?.postProcess !== undefined) {
    out = applyPostProcess(results, effectiveHooks.postProcess);
  }
  const hookMs = wantsDiagnostics ? nowMs() - tHook0 : 0;
  return { results: out, highlightMs, hookMs };
}

/**
 * Synchronous suggestion enumeration shared by `autocomplete()` and inline
 * `search({ autocomplete })`. Callers pass pre-normalized query tokens and a
 * validated spec; `searchFields` scopes the scan when the autocomplete spec
 * carries no explicit field.
 */
export function computeSuggestions<TDoc>(
  s: DocumentIndexState<TDoc>,
  query: string,
  queryTokens: Uint32Array,
  spec: NormalizedAutocompleteOptions,
  searchFields?: ReadonlySet<number>
): SuggestionItem<TDoc>[] {
  let fieldIndices: number[];
  if (spec.field !== undefined) {
    fieldIndices = [s.fieldNameToIndex.get(spec.field) as number];
  } else if (searchFields !== undefined) {
    fieldIndices = [...searchFields];
  } else {
    fieldIndices = s.sortedFields.map((_, i) => i);
  }

  const typoForSuggest = normalizeTypoTolerance(
    spec.fuzzyDistance > 0
      ? { enabled: true, maxDistance: spec.fuzzyDistance as 1 | 2 }
      : undefined
  );
  const prefixOpts = normalizePrefixOptions(undefined);

  interface SuggestHit {
    keys: RankableCandidate;
    text: string;
    dIdx: number;
    fIdx: number;
    rawStr: string;
  }
  const candidates: SuggestHit[] = [];

  for (let d = 0; d < s.records.length; d++) {
    const doc = s.records[d];
    if (doc === null || doc === undefined) continue;
    const id = s.docIds[d];
    if (id === null || id === undefined) continue;
    const rows = s.docToRowIndices[d];
    if (!rows) continue;
    for (let k = 0; k < fieldIndices.length; k++) {
      const fIdx = fieldIndices[k] as number;
      const rowIdx = rows[fIdx];
      if (rowIdx === undefined) continue;
      if (s.tombstones.has(rowIdx)) continue;
      const rowTok = s.rowTokens[rowIdx];
      if (!rowTok || rowTok.length === 0) continue;
      const rawStr = s.rawFieldStrings[d]?.[fIdx] ?? '';
      if (rawStr.length === 0) continue;

      let matched = false;
      let rawScore = 0;
      if (spec.mode === 'prefix') {
        const r = scorePrefixTokens(rowTok, queryTokens, prefixOpts, typoForSuggest);
        matched = r.matched;
        rawScore = r.score;
      } else if (spec.fuzzyDistance > 0) {
        const r = scoreSubstringTypoTokens(rowTok, queryTokens, typoForSuggest);
        matched = r.matched;
        rawScore = r.score;
      } else {
        const r = scoreFuzzyTokens(rowTok, queryTokens);
        matched = r.matched;
        rawScore = r.score;
      }
      if (!matched) continue;
      const weight = s.sortedFields[fIdx].weight;
      candidates.push({
        keys: {
          score: Math.round(rawScore * weight),
          fieldWeight: weight,
          isExactMatch: isExactTokenMatch(rowTok, queryTokens),
          matchedLength: rowTok.length,
          id,
          docIndex: d
        },
        text: rawStr,
        dIdx: d,
        fIdx,
        rawStr
      });
    }
  }

  candidates.sort((a, b) => compareRanked(a.keys, b.keys, spec.tieBreakers));
  const top = candidates.length > spec.limit ? candidates.slice(0, spec.limit) : candidates;
  const out: SuggestionItem<TDoc>[] = [];
  for (let i = 0; i < top.length; i++) {
    const c = top[i] as SuggestHit;
    const highlightMode = spec.mode === 'prefix' ? 'prefix' : (spec.fuzzyDistance > 0 ? 'substring' : 'fuzzy');
    let matchedRanges: HighlightRange[] = [];
    try {
      matchedRanges = alignHighlights(c.rawStr, query, {
        mode: highlightMode,
        normalized: s.normalized,
        queryTokens,
        typoTolerance: spec.fuzzyDistance > 0
          ? { enabled: true, maxDistance: spec.fuzzyDistance as 1 | 2 }
          : undefined
      });
    } catch (err) {
      // Only tolerate expected highlight-alignment failures for a single
      // candidate (scorer/highlight length gates, polarity mismatch);
      // programmer errors and I/O faults must propagate.
      if (
        err instanceof IncompatibleOptionError ||
        err instanceof ProfileMismatchError ||
        err instanceof RangeError
      ) {
        matchedRanges = [];
      } else {
        throw err;
      }
    }
    out.push({
      text: c.text,
      score: c.keys.score,
      type: spec.mode === 'prefix' ? 'completion' : 'did-you-mean',
      matchedRanges,
      docId: c.keys.id,
      doc: s.records[c.dIdx] as TDoc
    });
  }
  return out;
}

/**
 * First-party autocomplete / did-you-mean primitive (extracted verbatim
 * from `DocumentIndex.autocomplete`).
 */
export async function runAutocomplete<TDoc>(
  s: DocumentIndexState<TDoc>,
  query: string,
  options: AutocompleteOptions = {}
): Promise<SuggestResponse<TDoc>> {
  if (typeof query !== 'string') {
    throw new TypeError(`[webgpu-search] autocomplete expects query: string, got ${typeof query}.`);
  }
  const spec = normalizeAutocompleteOptions(options);
  if (spec.field !== undefined && !s.fieldNameToIndex.has(spec.field)) {
    throw new Error(`[webgpu-search] Unknown autocomplete field: "${spec.field}".`);
  }
  const t0 = nowMs();
  const normalized = normalizeText(query, s.normalized);
  if (normalized.tokenCount > QUERY_TOKENS_MAX) {
    throw new QueryTooLongError(QUERY_TOKENS_MAX, normalized.tokenCount, s.profileId);
  }
  if (normalized.isEmpty || s.idToDocIndex.size === 0) {
    return { suggestions: [], queryDurationMs: nowMs() - t0 };
  }
  const allowed = spec.field !== undefined
    ? new Set<number>([s.fieldNameToIndex.get(spec.field) as number])
    : undefined;
  const suggestions = computeSuggestions(s, query, normalized.tokens, spec, allowed);
  return { suggestions, queryDurationMs: nowMs() - t0 };
}

/**
 * Full search execution (extracted verbatim from
 * `DocumentIndex.search`, minus the facade-level destroyed/mutex/generation
 * prologue which stays in `document-index.ts`).
 *
 * @param gen generation snapshot taken after the mutex wait; mid-flight
 *            mutations bump `generation` and abort in-flight searches.
 */
export async function runSearch<TDoc>(
  s: DocumentIndexState<TDoc>,
  query: string,
  options: DocumentSearchOptions<TDoc>,
  gen: number
): Promise<DocumentSearchResponse<TDoc>> {
  const validated = validateSearchOptions(s, query, options);
  const {
    mode,
    caseSensitive,
    signal,
    cpuScorer,
    typo,
    effectiveHooks,
    cpuModeOptions,
    budget,
    wantsDiagnostics,
    queryStartMs,
    diagWarnings,
    normalizedQuery,
    clampedLimit,
    forceCpu,
    facetSpec,
    wantsFacets,
    facetingMode,
    facetEngine,
    tieBreakers,
    suggestSpec,
    allowedFieldIndices,
    hasEmptyFieldList,
    customTokenTerms
  } = validated;

  const filterState = compileFilterPhase(
    s,
    options.filter,
    effectiveHooks,
    wantsDiagnostics,
    signal,
    budget,
    queryStartMs
  );
  const { filterBitset, filterPredicate, structuredFilter, filteringMs } = filterState;
  /**
   * assemble response diagnostics (undefined unless requested).
   * `filteringMs` is read at call time so early exits after compilation
   * report measured filter cost; `totalMs` is wall-clock from query entry
   * (end-to-end including inline autocomplete; see `autocompleteMs` for the slice).
   * Scoring includes post-match scoring-hook time on all paths.
   */
  const buildDiagnostics = (
    routedEngine: EngineType,
    scoringMs: number,
    highlightMs: number,
    facetingMs: number | undefined,
    hasOverflow: boolean,
    autocompleteMs?: number
  ): QueryDiagnostics | undefined => {
    if (!wantsDiagnostics) return undefined;
    const scanned = s.idToDocIndex.size;
    const selectivity = filterBitset !== undefined
      ? computeFilterSelectivity(filterBitset.popcount(), scanned)
      : 1.0;
    const totalMs = nowMs() - queryStartMs;
    const timings: QueryDiagnosticsTimings = {
      filteringMs,
      scoringMs,
      highlightMs,
      ...(facetingMs !== undefined ? { facetingMs } : {}),
      ...(autocompleteMs !== undefined ? { autocompleteMs } : {}),
      totalMs
    };
    const diag: QueryDiagnostics = {
      scannedCandidates: scanned,
      filterSelectivity: selectivity,
      routedEngine,
      hasOverflow,
      timings
    };
    if (diagWarnings.length > 0) diag.warnings = [...diagWarnings];
    return diag;
  };
  /**
   * post-hoc broad-query + overflow warnings (non-fatal, gated on
   * `diagnostics:true` — routing decisions stay ungated for safety).
   * `rawTotalMatches` carries the pre-filter pool count on the GPU path so
   * the overflow sentence stays accurate; `facetsExact` suppresses the
   * stale "approximate" claim after a force-exact rescan.
   */
  const pushPostHocWarnings = (
    totalMatches: number,
    hasOverflow: boolean,
    opts?: { rawTotalMatches?: number; facetsRequested?: boolean; facetsExact?: boolean }
  ): void => {
    if (!wantsDiagnostics) return;
    const scanned = s.idToDocIndex.size;
    const selectivity = scanned > 0 ? totalMatches / scanned : 0;
    if (isBroadSelectivity(selectivity, scanned)) {
      diagWarnings.push(broadSelectivityWarning(selectivity, scanned));
    }
    if (hasOverflow) {
      diagWarnings.push(
        candidateOverflowWarning(totalMatches, s.candidateCapacity, {
          facetsRequested: opts?.facetsRequested ?? false,
          facetsExact: opts?.facetsExact ?? false,
          rawTotalMatches: opts?.rawTotalMatches,
        })
      );
    }
  };

  const noHits = (q: string): DocumentSearchResponse<TDoc> => {
    // even trivial exits honor caller aborts and time budgets.
    throwIfAborted(signal);
    throwIfBudgetAborted(budget);
    assertTimeBudget(queryStartMs, budget);
    const diag = buildDiagnostics('cpu', 0, 0, wantsFacets ? 0 : undefined, false);
    // Public totalMs is scorer wall-clock (0 on no-hit: nothing scored);
    // diagnostics.totalMs carries the wall-clock including validation.
    return {
      query: q,
      mode,
      engine: 'cpu',
      totalMatches: 0,
      candidateCount: 0,
      hasOverflow: false,
      results: [],
      timings: {
        queryUploadMs: 0,
        encodeSubmitMs: 0,
        gpuExecutionMs: null,
        readbackMs: 0,
        totalMs: 0,
        gpuDispatchMs: 0
      },
      profileId: s.profileId,
      scoringVersion: SCORING_VERSION,
      cpuScorer,
      fallbackReason: forceCpu ? 'query-too-long' : s.engineState.fallbackReason,
      ...(wantsFacets && facetEngine && facetSpec
        ? { facets: facetEngine.emptyResults(facetSpec) }
        : {}),
      ...(diag ? { diagnostics: diag } : {})
    };
  };

  if (filterBitset !== undefined && filterBitset.isEmpty()) {
    return noHits(query);
  }

  if (normalizedQuery.isEmpty) {
    return noHits('');
  }
  if (s.idToDocIndex.size === 0) {
    return noHits(query);
  }
  if (hasEmptyFieldList) {
    return noHits(query);
  }

  const route = routeEngine(s, {
    mode,
    typoEnabled: typo.enabled,
    cpuScorer,
    forceCpu,
    allowedFieldIndices,
    normalizedQueryTokenCount: normalizedQuery.tokens.length,
    filterBitset,
    budget,
    queryStartMs,
    wantsDiagnostics,
    diagWarnings
  });
  const { broadQueryCpuRoute, isGpuSupportedMode, useGpu, gpuHandle } = route;

  if (useGpu && gpuHandle !== null) {
    // narrow GPU try to dispatch only so extension hook
    // errors (filterPredicate / scoringHook / postProcess) propagate
    // directly without destroying the engine or double-invoking via CPU
    // fallback. Only dispatch failures fall through to CPU.
    let gpuResult: Awaited<ReturnType<WebGPUEngine['search']>> | undefined = undefined;
    try {
      gpuResult = await gpuHandle.search(query, {
        ...options,
        mode,
        limit: s.candidateCapacity,
        caseSensitive,
        signal
      });

      if (s.isDestroyed || s.generation !== gen) {
        throw abortError();
      }
      throwIfAborted(signal);
      // GPU dispatch is async wall-clock work — enforce the caller
      // deadline and budget abort before touching the readback.
      throwIfBudgetAborted(budget);
      assertTimeBudget(queryStartMs, budget);
    } catch (err: any) {
      if (err.name === 'AbortError') {
        throw err;
      }
      // Mirror search-index: mode-routed rejections (token/prefix/typo via
      // direct or raced calls) are not engine failures — keep a healthy GPU
      // up and fall through to exact CPU.
      const modeRouted: boolean =
        err instanceof IncompatibleOptionError && (err.option === 'mode' || err.option === 'typoTolerance');
      if (!modeRouted) {
        console.warn('[webgpu-search] GPU document search failed, falling back to CPU:', err);
        s.engineState = transitionEngineState(s.engineState, 'cpu', 'gpu-execution-error');
        if (s.gpuEngine) {
          try { s.gpuEngine.destroy(); } catch {}
          s.gpuEngine = null;
        }
      }
      gpuResult = undefined;
    }

    if (gpuResult !== undefined) {
      const gpuResultOk = gpuResult;

      // WebGPU candidate readback & fixed-point score enrichment
      const docMatches = new Map<number, {
        bestScore: number;
        bestFieldIdx: number;
        fieldScores: Map<number, number>;
      }>();
      // unfiltered query match set for facet aggregation (filter applied below).
      const queryMatchedAll = wantsFacets ? new Set<number>() : null;

      // hook errors propagate (no fallback / no double-invoke).
      // Only GPU dispatch failures fall back; readback + hook pipeline runs
      // outside the dispatch try above.
      for (let i = 0; i < gpuResultOk.results.length; i++) {
        const item = gpuResultOk.results[i];
        const r = item.index;
        if (s.tombstones.has(r)) continue;
        const fIdx = s.rowToFieldIndex[r];
        if (allowedFieldIndices && !allowedFieldIndices.has(fIdx)) continue;
        const dIdx = s.rowToDocIndex[r];
        if (queryMatchedAll !== null && s.records[dIdx]) {
          queryMatchedAll.add(dIdx);
        }
        if (filterBitset && !filterBitset.has(dIdx)) continue;
        const doc = s.records[dIdx];
        if (!doc) continue;
        if (filterPredicate && !filterPredicate(doc)) continue;

        const rawScore = item.score;
        const fDef = s.sortedFields[fIdx];
        const weightedScore = Math.round(rawScore * fDef.weight);

        accumulateDocMatch(docMatches, dIdx, fIdx, weightedScore);
      }

      const hits = aggregateDocMatches(s, docMatches, normalizedQuery.tokens, tieBreakers);

      const totalMatches = hits.length;
      const candidateCount = Math.min(totalMatches, s.candidateCapacity);
      let results = hits.slice(0, clampedLimit).map((h) => h.item);
      const gpuEnriched = enrichResults(s, results, query, mode, options, customTokenTerms, normalizedQuery.tokens, effectiveHooks, tieBreakers, wantsDiagnostics);
      results = gpuEnriched.results;
      const gpuHighlightMs = gpuEnriched.highlightMs;
      const gpuHookMs = gpuEnriched.hookMs;

      // facet aggregation. Exact when the GPU pool covered all matches;
      // approximate over the top pool on overflow unless force-exact rescan.
      let gpuFacets: Record<string, FacetResult> | undefined = undefined;
      let gpuFacetMs: number | undefined = undefined;
      let gpuFacetExact = true;
      if (wantsFacets && facetEngine && facetSpec && queryMatchedAll !== null) {
        const facetT0 = nowMs();
        let facetCandidates: Iterable<number> = queryMatchedAll;
        let facetIsApproximate = gpuResultOk.hasOverflow;
        if (gpuResultOk.hasOverflow && facetingMode === 'force-exact') {
          const exact = scoreExactMatchesMultiField(
            s.records.length,
            s.sortedFields,
            s.rowTokens,
            s.rowToDocIndex,
            s.rowToFieldIndex,
            normalizedQuery.tokens,
            mode,
            clampedLimit,
            s.candidateCapacity,
            allowedFieldIndices,
            s.tombstones,
            undefined,
            true,
            cpuModeOptions
          );
          facetCandidates = exact.allMatchedDocIndices ?? exact.results.map((r) => r.docIndex);
          facetIsApproximate = false;
        }
        gpuFacets = buildFacetResults(
          s, facetEngine, facetSpec, facetCandidates, structuredFilter, filterPredicate, facetIsApproximate
        );
        const facetMs = nowMs() - facetT0;
        gpuFacetMs = facetMs;
        gpuFacetExact = !facetIsApproximate;
      }

      // deadline + post-hoc warnings before returning GPU results.
      // Inline autocomplete runs before the diagnostics snapshot so
      // diagnostics.totalMs covers end-to-end latency (autocomplete bucket).
      const tGpuSuggest0 = wantsDiagnostics && suggestSpec ? nowMs() : 0;
      const gpuSuggestions = suggestSpec
        ? computeSuggestions(s, query, normalizedQuery.tokens, suggestSpec, allowedFieldIndices)
        : undefined;
      const gpuSuggestMs = wantsDiagnostics && suggestSpec ? nowMs() - tGpuSuggest0 : undefined;
      throwIfAborted(signal);
      throwIfBudgetAborted(budget);
      assertTimeBudget(queryStartMs, budget);
      pushPostHocWarnings(totalMatches, gpuResultOk.hasOverflow, {
        rawTotalMatches: gpuResultOk.totalMatches,
        facetsRequested: wantsFacets,
        facetsExact: gpuFacetExact,
      });
      // Scoring = engine scan + post-match hooks (facet time excluded —
      // no double-counting, no mutation of the engine timings object).
      const gpuEngineTotal = gpuResultOk.timings.totalMs;
      const gpuDiag = buildDiagnostics(
        'webgpu',
        gpuEngineTotal + gpuHookMs,
        gpuHighlightMs,
        gpuFacetMs,
        gpuResultOk.hasOverflow,
        gpuSuggestMs
      );

      return {
        query: gpuResultOk.query,
        mode: gpuResultOk.mode,
        engine: 'webgpu',
        totalMatches,
        candidateCount,
        hasOverflow: gpuResultOk.hasOverflow,
        results,
        timings: {
          ...gpuResultOk.timings,
          totalMs: gpuEngineTotal + (gpuFacetMs ?? 0) + gpuHighlightMs + gpuHookMs,
        },
        profileId: s.profileId,
        scoringVersion: SCORING_VERSION,
        cpuScorer,
        ...(gpuFacets ? { facets: gpuFacets } : {}),
        ...(gpuDiag ? { diagnostics: gpuDiag } : {}),
        ...(gpuSuggestions ? { suggestions: gpuSuggestions } : {})
      };
    }
  }

  // 2. CPU execution path
  if (s.isDestroyed || s.generation !== gen) {
    throw abortError();
  }
  throwIfAborted(signal);
  throwIfBudgetAborted(budget);
  assertTimeBudget(queryStartMs, budget);
  const t0 = nowMs();

  if (cpuScorer === 'ufuzzy') {
    const cpuEngine = new CPUEngine();
    const docMatches = new Map<number, {
      bestScore: number;
      bestFieldIdx: number;
      fieldScores: Map<number, number>;
    }>();
    // unfiltered query match set for exact facet aggregation.
    const legacyMatchedAll = wantsFacets ? new Set<number>() : null;

    for (let fIdx = 0; fIdx < s.sortedFields.length; fIdx++) {
      if (allowedFieldIndices && !allowedFieldIndices.has(fIdx)) continue;
      const fDef = s.sortedFields[fIdx];
      const fieldStrings = s.records.map((doc, dIdx) => (doc && s.rawFieldStrings[dIdx] ? s.rawFieldStrings[dIdx][fIdx] : ''));
      const legacyResult = mode === 'fuzzy'
        ? cpuEngine.searchWithUFuzzy(fieldStrings, query, s.records.length, caseSensitive)
        : cpuEngine.searchNaiveScan(fieldStrings, query, s.records.length, caseSensitive);

      for (const hit of legacyResult.results) {
        const dIdx = hit.index;
        // Explicit tombstone guard for symmetry with GPU/exact paths:
        // removed docs have null records (columnar presence also cleared).
        if (!s.records[dIdx] || s.docIds[dIdx] === null || s.docIds[dIdx] === undefined) continue;
        if (legacyMatchedAll !== null) {
          legacyMatchedAll.add(dIdx);
        }
        if (filterBitset && !filterBitset.has(dIdx)) continue;
        const doc = s.records[dIdx];
        if (!doc) continue;
        if (filterPredicate && !filterPredicate(doc)) continue;
        const weightedScore = Math.round(hit.score * fDef.weight);
        accumulateDocMatch(docMatches, dIdx, fIdx, weightedScore);
      }
    }

    const hits = aggregateDocMatches(s, docMatches, normalizedQuery.tokens, tieBreakers);

    const totalMatches = hits.length;
    const candidateCount = Math.min(totalMatches, s.candidateCapacity);
    let results = hits.slice(0, clampedLimit).map((h) => h.item);
    const legacyEnriched = enrichResults(s, results, query, mode, options, customTokenTerms, normalizedQuery.tokens, effectiveHooks, tieBreakers, wantsDiagnostics);
    results = legacyEnriched.results;
    const legacyHighlightMs = legacyEnriched.highlightMs;
    const preFacetMs = nowMs() - t0;
    // scoring bucket covers scan + ranking + post-match hooks;
    // highlight enrichment is metered separately above.
    const legacyScoringMs = Math.max(0, preFacetMs - legacyHighlightMs);

    // CPU evaluates the full match set, so legacy facets are exact
    // w.r.t. the serving (ufuzzy/native) match set; see types for caveat.
    let legacyFacets: Record<string, FacetResult> | undefined = undefined;
    let legacyFacetMs = 0;
    if (wantsFacets && facetEngine && facetSpec && legacyMatchedAll !== null) {
      const fT0 = nowMs();
      legacyFacets = buildFacetResults(
        s, facetEngine, facetSpec, legacyMatchedAll, structuredFilter, filterPredicate, false
      );
      legacyFacetMs = nowMs() - fT0;
    }

    // inline autocomplete precedes the diagnostics snapshot so
    // diagnostics.totalMs covers end-to-end latency (autocompleteMs bucket).
    const tLegacySuggest0 = wantsDiagnostics && suggestSpec ? nowMs() : 0;
    const legacySuggestions = suggestSpec
      ? computeSuggestions(s, query, normalizedQuery.tokens, suggestSpec, allowedFieldIndices)
      : undefined;
    const legacySuggestMs = wantsDiagnostics && suggestSpec ? nowMs() - tLegacySuggest0 : undefined;
    throwIfAborted(signal);
    throwIfBudgetAborted(budget);
    assertTimeBudget(queryStartMs, budget);
    pushPostHocWarnings(totalMatches, totalMatches > s.candidateCapacity, {
      facetsRequested: wantsFacets,
      facetsExact: true,
    });
    const legacyDiag = buildDiagnostics(
      'cpu',
      legacyScoringMs,
      legacyHighlightMs,
      wantsFacets ? legacyFacetMs : undefined,
      totalMatches > s.candidateCapacity,
      legacySuggestMs
    );

    return {
      query,
      mode,
      engine: 'cpu',
      totalMatches,
      candidateCount,
      hasOverflow: totalMatches > s.candidateCapacity,
      results,
      timings: {
        queryUploadMs: 0,
        encodeSubmitMs: 0,
        gpuExecutionMs: null,
        readbackMs: 0,
        totalMs: preFacetMs + legacyFacetMs,
        gpuDispatchMs: 0
      },
      profileId: s.profileId,
      scoringVersion: SCORING_VERSION,
      cpuScorer,
      fallbackReason: 'cpu-algorithm-requested',
      ...(legacyFacets ? { facets: legacyFacets } : {}),
      ...(legacyDiag ? { diagnostics: legacyDiag } : {}),
      ...(legacySuggestions ? { suggestions: legacySuggestions } : {})
    };
  }

  // Default exact CPU algorithm (collect full match set only when faceting).
  // passes the deterministic hierarchy + doc IDs so pre-truncation order
  // already reflects (score, weight, exact, length, id).
  const parityResult = scoreExactMatchesMultiField(
    s.records.length,
    s.sortedFields,
    s.rowTokens,
    s.rowToDocIndex,
    s.rowToFieldIndex,
    normalizedQuery.tokens,
    mode,
    clampedLimit,
    s.candidateCapacity,
    allowedFieldIndices,
    s.tombstones,
    (filterBitset || filterPredicate) ? (dIdx) => {
      if (filterBitset && !filterBitset.has(dIdx)) return false;
      if (filterPredicate) {
        const doc = s.records[dIdx];
        return doc !== null && doc !== undefined && filterPredicate(doc);
      }
      return true;
    } : undefined,
    wantsFacets,
    cpuModeOptions,
    { tieBreakers, docIds: s.docIds as unknown as ReadonlyArray<string | number | null | undefined> }
  );

  if (s.isDestroyed || s.generation !== gen) {
    throw abortError();
  }
  throwIfAborted(signal);
  throwIfBudgetAborted(budget);
  assertTimeBudget(queryStartMs, budget);

  const idProp = typeof s.options.idField === 'string' ? s.options.idField : 'id';
  let enrichedResults: DocumentSearchResultItem<TDoc>[] = [];
  for (let i = 0; i < parityResult.results.length; i++) {
    const hit = parityResult.results[i];
    const id = s.docIds[hit.docIndex];
    if (id === null || id === undefined) continue;
    const doc = s.records[hit.docIndex] ?? ({ [idProp]: id } as any);
    enrichedResults.push({
      id,
      doc,
      score: hit.score,
      matchedField: hit.matchedField,
      matches: hit.matches
    });
  }
  const parityEnriched = enrichResults(s, enrichedResults, query, mode, options, customTokenTerms, normalizedQuery.tokens, effectiveHooks, tieBreakers, wantsDiagnostics);
  enrichedResults = parityEnriched.results;
  const parityHighlightMs = parityEnriched.highlightMs;
  const parityHookMs = parityEnriched.hookMs;
  const parityScoringMs = parityResult.durationMs + parityHookMs;

  // exact facet aggregation. Disjunctive facets need the unfiltered
  // query match set, so a structured filter triggers one extra unfiltered
  // parity scan (results above stay filtered and correctly ranked; O(F*n)
  // per-facet re-evaluation + F bitset allocs by design for disjunctive
  // semantics). Wall-clock facet work is included in totalMs; future
  // attribute facet work to diagnostics.timings.facetingMs.
  let parityFacets: Record<string, FacetResult> | undefined = undefined;
  const facetT0 = nowMs();
  let facetScanMs = 0;
  if (wantsFacets && facetEngine && facetSpec) {
    let facetCandidates: Iterable<number> =
      parityResult.allMatchedDocIndices ?? parityResult.results.map((r) => r.docIndex);
    if (structuredFilter !== undefined) {
      const unfiltered = scoreExactMatchesMultiField(
        s.records.length,
        s.sortedFields,
        s.rowTokens,
        s.rowToDocIndex,
        s.rowToFieldIndex,
        normalizedQuery.tokens,
        mode,
        clampedLimit,
        s.candidateCapacity,
        allowedFieldIndices,
        s.tombstones,
        undefined,
        true,
        cpuModeOptions
      );
      facetCandidates = unfiltered.allMatchedDocIndices ?? unfiltered.results.map((r) => r.docIndex);
      facetScanMs = unfiltered.durationMs;
    }
    parityFacets = buildFacetResults(
      s, facetEngine, facetSpec, facetCandidates, structuredFilter, filterPredicate, false
    );
  }
  const facetMs = wantsFacets ? Math.max(0, nowMs() - facetT0 - facetScanMs) : 0;
  // the disjunctive unfiltered rescan serves facets, so it joins the
  // faceting bucket (scoring stays scan + post-match hooks).
  const parityFacetingMs = wantsFacets ? facetScanMs + facetMs : undefined;

  // Public totalMs is scorer wall-clock: scan + hooks + highlight + facets
  // (excludes inline autocomplete; diagnostics.totalMs is end-to-end).
  const timings: SearchTimings = {
    queryUploadMs: 0,
    encodeSubmitMs: 0,
    gpuExecutionMs: null,
    readbackMs: 0,
    totalMs: parityScoringMs + parityHighlightMs + facetScanMs + facetMs,
    gpuDispatchMs: 0
  };

  let effectiveFallbackReason = s.engineState.fallbackReason;
  if (forceCpu) {
    effectiveFallbackReason = 'query-too-long';
  } else if (!isGpuSupportedMode) {
    // token/prefix/typo queries are CPU-by-design (exact-only
    // WGSL kernels) — recorded per the exact boundary.
    effectiveFallbackReason = 'unsupported-mode';
  } else if (useGpu && gpuHandle !== null) {
    effectiveFallbackReason = 'gpu-execution-error';
  } else if (broadQueryCpuRoute) {
    // broad-query CPU routing is a routing decision (like
    // field-restriction below), not a scorer request — leave the reason
    // as-is and surface the decision in diagnostics.warnings instead.
  }
  // Note: field-restricted routing (isFieldRestricted) intentionally leaves
  // fallbackReason as-is — it is a routing decision, not a scorer request,
  // so reusing 'cpu-algorithm-requested' would mislead telemetry.

  // inline autocomplete precedes the diagnostics snapshot so
  // diagnostics.totalMs covers end-to-end latency (autocompleteMs bucket).
  const tParitySuggest0 = wantsDiagnostics && suggestSpec ? nowMs() : 0;
  const paritySuggestions = suggestSpec
    ? computeSuggestions(s, query, normalizedQuery.tokens, suggestSpec, allowedFieldIndices)
    : undefined;
  const paritySuggestMs = wantsDiagnostics && suggestSpec ? nowMs() - tParitySuggest0 : undefined;
  throwIfAborted(signal);
  throwIfBudgetAborted(budget);
  assertTimeBudget(queryStartMs, budget);
  pushPostHocWarnings(parityResult.totalMatches, parityResult.hasOverflow, {
    facetsRequested: wantsFacets,
    facetsExact: true,
  });
  const parityDiag = buildDiagnostics(
    'cpu',
    parityScoringMs,
    parityHighlightMs,
    parityFacetingMs,
    parityResult.hasOverflow,
    paritySuggestMs
  );

  return {
    query,
    mode,
    engine: 'cpu',
    totalMatches: parityResult.totalMatches,
    candidateCount: parityResult.candidateCount,
    hasOverflow: parityResult.hasOverflow,
    results: enrichedResults,
    timings,
    profileId: s.profileId,
    scoringVersion: SCORING_VERSION,
    cpuScorer,
    fallbackReason: effectiveFallbackReason,
    ...(parityFacets ? { facets: parityFacets } : {}),
    ...(parityDiag ? { diagnostics: parityDiag } : {}),
    ...(paritySuggestions ? { suggestions: paritySuggestions } : {})
  };
}
