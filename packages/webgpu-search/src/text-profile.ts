/**
 * v0.2 Unicode text profile: version constants, profile types, error classes.
 * Frozen values mirror docs/unicode-contract.md (M2: CPU post-fold enforced;
 * GPU stays legacy until the M3 engine swap). Portable: no DOM refs.
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
 * - `'ufuzzy'`: explicit opt-in to the legacy uFuzzy CPU scorer (CPU-only,
 *   explicitly non-conforming scores, never routes to WebGPU, excluded from
 *   the differential matrix).
 */
export type CpuAlgorithm = 'parity' | 'ufuzzy';

export type OnQueryTooLong = 'throw' | 'cpu-fallback';

/**
 * @deprecated M1 interim pre-fold code-point counter. M2 sizes queries and
 * records with exact post-fold token counts from `normalizeText()`.
 * Kept for compat only; do not use for budgets/limits (e.g. `Strasse` is 6
 * pre-fold vs 7 post-fold). Removal in v0.3.
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
  property: string;
  constructor(expected: unknown, actual: unknown, property: string = 'caseSensitive') {
    super(
      `Profile mismatch (expected ${property}=${String(expected)}, got ${String(actual)}). ` +
        `${property} is fixed at index-construction time: recreate the index with ` +
        `IndexOptions.${property}=${String(expected)} or retry the query with ` +
        `${property}=${String(expected)}.`
    );
    this.name = 'ProfileMismatchError';
    this.expected = expected;
    this.actual = actual;
    this.property = property;
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
