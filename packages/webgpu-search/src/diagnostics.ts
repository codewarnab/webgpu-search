/**
 * v0.4 query diagnostics, cost budgets & broad-query safeguards (Issue #10 M7).
 *
 * Host applications (IDE palettes, log viewers, data grids) need execution
 * deadlines, memory ceilings, candidate overflow telemetry, and protection
 * against broad queries saturating the GPU dispatch path. This module owns
 * the portable, zero-dependency policy layer:
 * - `normalizeCostBudgetOptions`: fail-closed validation of caller budgets.
 * - `assertTimeBudget` / `assertCandidateBudget`: fail-closed enforcement
 *   throwing `CostBudgetExceededError` (`'time'` / `'candidates'`).
 * - Broad-query heuristics: short queries over massive corpora route to the
 *   CPU streaming scan pre-dispatch; post-hoc selectivity > 80% over massive
 *   corpora emits a non-fatal warning.
 * - `computeFilterSelectivity`: filter narrowing ratio in `[0, 1]`.
 *
 * Timing granularity: budget enforcement is best-effort at phase boundaries
 * (post-filter, post-score, post-highlight, post-facet). Long synchronous
 * scans run to completion and then throw — work is discarded, never returned
 * as stale partials. There is no intra-scan preemption. Callers needing hard
 * deadlines must also use `abortSignal` for cooperative cancellation.
 *
 * Token semantics: `queryTokenCount` throughout M7 is post-fold Unicode code
 * points (including spaces), not whitespace words — e.g. `"auth"` is 4
 * tokens. The <=2-token pre-dispatch gate therefore fires only on 1–2
 * character queries, which are near-universally broad under fuzzy/substring.
 *
 * Telemetry overhead is sub-microsecond when disabled: clocks and warning
 * strings are gated on `diagnostics:true` (or an active time budget for the
 * entry clock). Phase timing itself lives in `document-index.ts` and
 * `hybrid-index.ts` so this module never touches engine internals.
 *
 * Portable: no DOM refs (`window`, `document`, `navigator`). Wall-clock reads
 * go through `nowMs()` (`runtime-guards.ts`), which avoids bare `performance`
 * globals for Web Worker / Node.js / SSR safety.
 */

import { CostBudgetExceededError } from './errors';
import { nowMs, throwIfAborted } from './runtime-guards';
import type { CostBudgetOptions } from './types';

/**
 * Post-hoc selectivity above which a query counts as broad: matching more
 * than 80% of a massive dataset risks GPU buffer saturation (TDR) and
 * candidate overflow, so the engine prefers the CPU streaming scan and
 * records a warning.
 */
export const BROAD_QUERY_SELECTIVITY_THRESHOLD = 0.8 as const;

/**
 * Minimum active document count for broad-query safeguards to engage.
 * Below this the GPU dispatch path handles even full-corpus matches without
 * saturation risk, so no routing or warnings apply.
 */
export const BROAD_QUERY_MIN_DOCS = 5000 as const;

/**
 * Pre-dispatch heuristic: queries with at most this many post-fold tokens
 * over a massive corpus (>= `BROAD_QUERY_MIN_DOCS` docs) are assumed broad
 * (e.g. single-character palette/log queries) and route to CPU before any
 * GPU dispatch. Selectivity is unknowable pre-scoring, so token count is the
 * only cheap, deterministic proxy available at routing time.
 */
export const BROAD_QUERY_SHORT_QUERY_TOKENS = 2 as const;

/** Cost budget with defaults resolved (all fields optional; undefined = no budget). */
export interface NormalizedCostBudget {
  maxExecutionTimeMs?: number;
  maxCandidates?: number;
  abortSignal?: AbortSignal;
}

/**
 * Validate and normalize caller cost budgets (fail-closed).
 * - `undefined` resolves to `undefined` (no budget enforcement).
 * - Must be a non-null object; `null`/arrays/primitives throw `TypeError`.
 * - `maxExecutionTimeMs` must be a finite number > 0 when provided
 *   (fractional milliseconds allowed); `0`, negatives, `NaN`, and `Infinity`
 *   throw `RangeError`. Note: a deadline alone never enables anything — it
 *   only constrains; tiny deadlines fail the query fail-closed. Deadlines are
 *   enforced at phase boundaries only (best-effort, post-scan discard — see
 *   `assertTimeBudget`); they do not preempt in-flight scans.
 * - `maxCandidates` must be an integer >= 1 when provided; fractions,
 *   `0`/negatives, and non-finite values throw `RangeError`. Structured
 *   filters are enforced pre-scan on the exact post-filter population;
 *   function-predicate filters report selectivity `1.0` and enforce the
 *   ceiling on the pre-predicate population (conservative fail-closed —
 *   the true post-predicate count is unknowable without scanning).
 * - `abortSignal` passes through untouched when provided (worker transport
 *   strips it pre-clone; see `worker-client.ts`); non-object values throw
 *   `TypeError`. Forged `{ aborted: true }` objects are honored via
 *   `throwIfAborted` at enforcement points.
 * - Unknown `max*` keys throw `TypeError` fail-closed (typo'd limits such as
 *   `{ maxCandidate: 5 }` must not silently disable the ceiling). Other
 *   unknown keys are ignored forward-compatibly.
 */
export function normalizeCostBudgetOptions(
  raw: CostBudgetOptions | undefined
): NormalizedCostBudget | undefined {
  if (raw === undefined) return undefined;
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
    throw new TypeError('[webgpu-search] budget must be an object.');
  }
  const record = raw as Record<string, unknown>;
  const KNOWN_BUDGET_KEYS = new Set(['maxExecutionTimeMs', 'maxCandidates', 'abortSignal']);
  for (const key of Object.keys(record)) {
    if (!KNOWN_BUDGET_KEYS.has(key) && /^max/i.test(key)) {
      throw new TypeError(
        `[webgpu-search] Unknown budget key "${key}". Did you mean "maxExecutionTimeMs" or "maxCandidates"?`
      );
    }
  }
  const out: NormalizedCostBudget = {};

  if (record.maxExecutionTimeMs !== undefined) {
    const v = record.maxExecutionTimeMs;
    if (typeof v !== 'number') {
      throw new TypeError(
        `[webgpu-search] budget.maxExecutionTimeMs must be a number, got ${typeof v}.`
      );
    }
    if (!Number.isFinite(v) || v <= 0) {
      throw new RangeError(
        `[webgpu-search] budget.maxExecutionTimeMs must be a finite number > 0, got ${String(v)}.`
      );
    }
    out.maxExecutionTimeMs = v;
  }

  if (record.maxCandidates !== undefined) {
    const v = record.maxCandidates;
    if (typeof v !== 'number') {
      throw new TypeError(
        `[webgpu-search] budget.maxCandidates must be a number, got ${typeof v}.`
      );
    }
    if (!Number.isInteger(v) || (v as number) < 1) {
      throw new RangeError(
        `[webgpu-search] budget.maxCandidates must be an integer >= 1, got ${String(v)}.`
      );
    }
    out.maxCandidates = v;
  }

  if (record.abortSignal !== undefined) {
    const v = record.abortSignal;
    if (typeof v !== 'object' || v === null) {
      throw new TypeError('[webgpu-search] budget.abortSignal must be an AbortSignal.');
    }
    out.abortSignal = v as AbortSignal;
  }

  if (
    out.maxExecutionTimeMs === undefined &&
    out.maxCandidates === undefined &&
    out.abortSignal === undefined
  ) {
    return undefined;
  }
  return out;
}

/** Throw `AbortError` when the budget abort signal is engaged (safe for forged objects). */
export function throwIfBudgetAborted(budget: NormalizedCostBudget | undefined): void {
  if (budget?.abortSignal == null) return;
  throwIfAborted(budget.abortSignal);
}

/**
 * Fail-closed deadline enforcement (best-effort, phase-boundary granularity).
 * Throws `CostBudgetExceededError` (`budgetType: 'time'`) when wall-clock
 * elapsed since `startMs` exceeds the configured `maxExecutionTimeMs`.
 * No-op without a time budget. Call at every phase boundary (post-filter,
 * post-score, post-highlight, post-facet) so over-budget queries abort
 * instead of returning stale partial work.
 *
 * This does NOT preempt in-flight synchronous scans: a long CPU scan runs to
 * completion and then throws (fail-closed discard, never partial results).
 */
export function assertTimeBudget(
  startMs: number,
  budget: NormalizedCostBudget | undefined
): void {
  const limit = budget?.maxExecutionTimeMs;
  if (limit === undefined) return;
  const elapsed = nowMs() - startMs;
  if (elapsed > limit) {
    throw new CostBudgetExceededError('time', limit, elapsed);
  }
}

/**
 * Fail-closed candidate-ceiling enforcement. Throws
 * `CostBudgetExceededError` (`budgetType: 'candidates'`) when the
 * post-filter candidate count to score exceeds `maxCandidates`. Call once
 * pre-scoring (nothing is scored on throw) and never on no-hit early exits
 * (empty query / corpus / empty filter score zero candidates by design).
 * With function-predicate filters the pre-predicate population is checked
 * (conservative; the true post-predicate count is unknowable pre-scan).
 */
export function assertCandidateBudget(
  candidatesToScore: number,
  budget: NormalizedCostBudget | undefined
): void {
  const limit = budget?.maxCandidates;
  if (limit === undefined) return;
  if (candidatesToScore > limit) {
    throw new CostBudgetExceededError('candidates', limit, candidatesToScore);
  }
}

/**
 * Filter narrowing ratio in `[0, 1]`: `matched / total`.
 * - No filter (`matched === total` by convention, including empty corpora)
 *   yields `1.0` (no narrowing). Function-predicate filters also report
 *   `1.0` with `filteringMs ≈ 0`; their evaluation cost lands in `scoringMs`.
 * - Empty corpus with a structured filter yields `0` (nothing can match).
 * - Result is clamped to `[0, 1]` defensively (popcount can never exceed the
 *   active count, but forged inputs must not leak `NaN`/`Infinity`).
 */
export function computeFilterSelectivity(matched: number, total: number): number {
  if (total <= 0) return matched <= 0 ? 1.0 : 0;
  if (matched <= 0) return 0;
  if (matched >= total) return 1.0;
  const ratio = matched / total;
  if (!Number.isFinite(ratio)) return 1.0;
  if (ratio < 0) return 0;
  if (ratio > 1) return 1.0;
  return ratio;
}

/**
 * Pre-dispatch broad-query heuristic: true when the corpus is massive
 * (`docCount >= BROAD_QUERY_MIN_DOCS`) and the query is short
 * (`queryTokenCount <= BROAD_QUERY_SHORT_QUERY_TOKENS`). `queryTokenCount`
 * is post-fold Unicode code points (including spaces), not whitespace words.
 * Short queries (single characters, symbol prefixes) match a large fraction
 * of any sizable corpus, so routing them to the CPU streaming scan
 * pre-dispatch avoids GPU buffer saturation and driver timeouts (TDR).
 * Deterministic and O(1). Callers suppress the route warning when the query
 * is already CPU-by-design (token/prefix/typo modes, explicit ufuzzy) since
 * the stated GPU-avoidance cause would misattribute.
 */
export function isBroadQueryHeuristic(queryTokenCount: number, docCount: number): boolean {
  if (!Number.isFinite(queryTokenCount) || !Number.isFinite(docCount)) return false;
  if (docCount < BROAD_QUERY_MIN_DOCS) return false;
  if (queryTokenCount <= 0) return false;
  return queryTokenCount <= BROAD_QUERY_SHORT_QUERY_TOKENS;
}

/**
 * Post-hoc broad-query check: true when the observed match selectivity
 * (`totalMatches / docCount`) exceeds `BROAD_QUERY_SELECTIVITY_THRESHOLD` on
 * a massive corpus. Callers record a non-fatal warning (never throw):
 * the result set is complete and correctly ranked, but hosts should narrow
 * the query or add filters on repeat.
 */
export function isBroadSelectivity(selectivity: number, docCount: number): boolean {
  if (!Number.isFinite(selectivity) || !Number.isFinite(docCount)) return false;
  if (docCount < BROAD_QUERY_MIN_DOCS) return false;
  return selectivity > BROAD_QUERY_SELECTIVITY_THRESHOLD;
}

/** Non-fatal warning recorded when the pre-dispatch heuristic routes to CPU. */
export function broadQueryRouteWarning(docCount: number, queryTokenCount: number): string {
  return (
    `[webgpu-search] broad-query: ${String(queryTokenCount)}-token query over ` +
    `${String(docCount)} documents routed to CPU streaming scan to avoid GPU ` +
    `saturation (pre-dispatch heuristic).`
  );
}

/** Non-fatal warning recorded when post-hoc selectivity exceeds 80% on a massive corpus. */
export function broadSelectivityWarning(selectivity: number, docCount: number): string {
  const pct = Number.isFinite(selectivity) ? (selectivity * 100).toFixed(1) : 'unknown';
  return (
    `[webgpu-search] broad-query: query matched ${pct}% of ${String(docCount)} ` +
    `documents; consider narrowing the query or adding filters.`
  );
}

export interface CandidateOverflowWarningOptions {
  /** Whether facets were requested (default true for backward compat). */
  facetsRequested?: boolean;
  /** True when facets were recomputed exactly (force-exact rescan). */
  facetsExact?: boolean;
  /** Raw pre-filter match count when it differs from the post-filter count. */
  rawTotalMatches?: number;
}

/**
 * Non-fatal warning recorded when matches overflow the candidate pool.
 * Tells callers the remediation knob (`candidateCapacity`) instead of only
 * stating the overflow. The facets clause is included only when facets were
 * requested and remain approximate; `force-exact` rescans set `facetsExact`
 * to suppress the stale "approximate" claim. When `rawTotalMatches` differs
 * (GPU raw pool vs post-filter hits), both counts are reported so the
 * sentence stays factually accurate.
 */
export function candidateOverflowWarning(
  totalMatches: number,
  candidateCapacity: number,
  opts?: CandidateOverflowWarningOptions
): string {
  const facetsRequested = opts?.facetsRequested ?? true;
  const facetsExact = opts?.facetsExact ?? false;
  const raw = opts?.rawTotalMatches;
  const countClause =
    raw !== undefined && raw !== totalMatches
      ? `${String(raw)} raw matches (${String(totalMatches)} post-filter) exceed `
      : `${String(totalMatches)} matches exceed `;
  let msg =
    `[webgpu-search] candidate overflow: ${countClause}` +
    `candidate pool capacity ${String(candidateCapacity)}; raise candidateCapacity ` +
    `or narrow the query / add filters.`;
  if (facetsRequested && !facetsExact) {
    msg += ' facets (if requested) are approximate.';
  }
  return msg;
}
