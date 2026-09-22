/**
 * shared runtime guards.
 *
 * Single source of truth for limit clamping, abort errors, wall-clock
 * reads, and search-mode validation so `search-index.ts`,
 * `document-index.ts`, `exact-scorer.ts`, and `webgpu-engine.ts`
 * cannot drift. Portable: no DOM refs (Worker/Node/SSR safe).
 */

import { RESULT_LIMIT_MAX } from './text-profile';
import { IncompatibleOptionError } from './errors';
import type { SearchMode } from './types';

export const DEFAULT_LIMIT = 50;

/**
 * Shared set of valid search modes. Single source of truth for
 * `assertValidMode()` so `search-index.ts`, `document-index.ts`,
 * `webgpu-engine.ts`, and `exact-scorer.ts` cannot drift.
 */
export const VALID_SEARCH_MODES: ReadonlySet<SearchMode> = new Set<SearchMode>([
  'fuzzy',
  'substring',
  'token',
  'prefix',
]);

/**
 * Throw `IncompatibleOptionError('mode')` unless `mode` is a valid search mode.
 * Centralizes the `mode !== ...` chain previously duplicated across
 * search-index, document-index, webgpu-engine, and exact-scorer.
 */
export function assertValidMode(mode: unknown): asserts mode is SearchMode {
  if (!VALID_SEARCH_MODES.has(mode as SearchMode)) {
    throw new IncompatibleOptionError(
      'mode',
      `Unknown search mode '${String(mode)}'. Expected 'fuzzy', 'substring', 'token', or 'prefix'.`
    );
  }
}

/**
 * Clamp a caller-supplied limit to 1..RESULT_LIMIT_MAX.
 * Coerced via Number(): non-finite / NaN / undefined / objects without a
 * numeric value → DEFAULT_LIMIT (50); numeric strings coerce ('3' → 3);
 * null → 0 → 1; fractions floored.
 */
export function clampLimit(raw: unknown): number {
  const n: number = typeof raw === 'number' ? raw : Number(raw);
  if (!Number.isFinite(n)) return DEFAULT_LIMIT;
  const floored: number = Math.floor(n);
  if (floored < 1) return 1;
  if (floored > RESULT_LIMIT_MAX) return RESULT_LIMIT_MAX;
  return floored;
}

/** Monotonic-ish wall clock without a bare `performance` global. */
export function nowMs(): number {
  const g = globalThis as unknown as {
    performance?: { now?: unknown };
  };
  if (typeof g.performance?.now === 'function') {
    return (g.performance.now as () => number)();
  }
  return Date.now();
}

/** AbortError without requiring a DOM `DOMException` global. */
export function abortError(): Error {
  const G = globalThis as unknown as {
    DOMException?: new (message: string, name: string) => Error;
  };
  if (typeof G.DOMException === 'function') {
    return new G.DOMException('Search aborted', 'AbortError');
  }
  const err = new Error('Search aborted');
  err.name = 'AbortError';
  return err;
}

/** Throw `abortError()` if signal is aborted (safe for forged objects). */
export function throwIfAborted(signal: unknown): void {
  if (signal == null) return;
  let aborted = false;
  try {
    aborted =
      (signal as { aborted?: unknown }).aborted === true ||
      (typeof AbortSignal !== 'undefined' &&
        signal instanceof AbortSignal &&
        signal.aborted);
  } catch {
    return;
  }
  if (aborted) throw abortError();
}

/** True when every scalar is ASCII (<=0x7F). Kept for compat; the GPU gate uses printable-ASCII below. */
export function isAsciiTokens(tokens: Uint32Array): boolean {
  for (let i = 0; i < tokens.length; i++) {
    if ((tokens[i] as number) > 0x7f) return false;
  }
  return true;
}

/**
 * True when every scalar is printable ASCII (0x20..0x7E). Legacy helper kept
 * for compat; all valid queries route to WebGPU regardless of script
 * (the legacy ASCII-only GPU gate is deleted). Not used in the exact path.
 */
export function isPrintableAsciiTokens(tokens: Uint32Array): boolean {
  for (let i = 0; i < tokens.length; i++) {
    const c: number = tokens[i] as number;
    if (c < 0x20 || c > 0x7e) return false;
  }
  return true;
}
