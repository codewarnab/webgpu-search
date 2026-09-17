/**
 * v0.2 Unicode text profile: version constants, profile types, error classes.
 * Frozen values mirror docs/unicode-contract.md (M1). Portable: no DOM refs.
 *
 * M1 scope note: this module freezes the *contract shape* (constants, types,
 * error classes). Full enforcement lands incrementally: query-length and
 * option-conflict checks are enforced in `hybrid-index.ts` on a pre-fold
 * code-point approximation; exact post-fold enforcement + parity scorer land
 * in M2 (`unicode-preprocess.ts` / `cpu-reference.ts`).
 */

export const UNICODE_VERSION = '16.0.0' as const;
export const SCORING_VERSION = 'parity-v1' as const;
export const FORMAT_VERSION = 2 as const;
export const SERIALIZED_MAGIC = 0x55324632 as const; // 'U2F2'
export const QUERY_TOKENS_MAX = 128 as const;
export const RESULT_LIMIT_MAX = 8192 as const;

export type TextProfileId = 'unicode-default';

/**
 * Which CPU scorer the caller wants.
 *
 * - `'parity'`: contract default; the v0.2 parity scorer (`cpu-reference.ts`).
 *   M1 echoes the request but the CPU path still serves the legacy
 *   uFuzzy/native scorer until M2 lands — see `SearchOptions.cpuAlgorithm`.
 * - `'ufuzzy'`: explicit opt-in to the legacy uFuzzy CPU scorer (CPU-only,
 *   never routes to WebGPU).
 */
export type CpuAlgorithm = 'parity' | 'ufuzzy';

export type OnQueryTooLong = 'throw' | 'cpu-fallback';

/**
 * M1 interim code-point counter (pre-fold approximation).
 *
 * Counts Unicode code points via string iteration, so astral characters count
 * as 1 and each lone surrogate counts as 1 (matching the per-surrogate U+FFFD
 * policy). M2 replaces query/record sizing with exact post-fold token counts
 * from `unicode-preprocess.ts`. Portable: no DOM refs.
 */
export function countUnicodeCodePoints(s: string): number {
  if (!s) return 0;
  return [...s].length;
}

export class QueryTooLongError extends RangeError {
  limit: number;
  actual: number;
  profileId: string;
  constructor(limit: number, actual: number, profileId: string) {
    super(`Query of ${actual} tokens exceeds limit of ${limit} (${profileId})`);
    this.name = 'QueryTooLongError';
    this.limit = limit;
    this.actual = actual;
    this.profileId = profileId;
  }
}

/**
 * Thrown when a serialized v0.1 buffer (no magic) is loaded as v0.2.
 * M1 shape-only stub — first thrown by `loadDataset` validation in M3.
 */
export class IncompatibleIndexError extends Error {
  expected: unknown;
  actual: unknown;
  constructor(expected: unknown, actual: unknown) {
    super(
      `Incompatible index (expected ${String(expected)}, got ${String(actual)}). Rebuild required.`
    );
    this.name = 'IncompatibleIndexError';
    this.expected = expected;
    this.actual = actual;
  }
}

export class ProfileMismatchError extends Error {
  expected: unknown;
  actual: unknown;
  constructor(expected: unknown, actual: unknown) {
    super(
      `Profile mismatch (expected caseSensitive=${String(expected)}, got ${String(actual)}). ` +
        `caseSensitive/profile is fixed at index-construction time: recreate the index with ` +
        `IndexOptions.caseSensitive=${String(expected)} or retry the query with ` +
        `caseSensitive=${String(expected)}.`
    );
    this.name = 'ProfileMismatchError';
    this.expected = expected;
    this.actual = actual;
  }
}

export class IncompatibleOptionError extends Error {
  option: string;
  reason: string;
  constructor(option: string, reason: string) {
    super(`Incompatible option ${option}: ${reason}`);
    this.name = 'IncompatibleOptionError';
    this.option = option;
    this.reason = reason;
  }
}
