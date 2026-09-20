/**
 * v0.2 Unicode text profile: version constants, profile types, error classes.
 * Frozen values mirror docs/unicode-contract.md (M3: u32 representation swap
 * landed on both paths; M4 owns the differential harness). Portable: no DOM refs.
 */

export const UNICODE_VERSION = '16.0.0' as const;
export const SCORING_VERSION = 'parity-v1' as const;
export const FORMAT_VERSION = 2 as const;
export const SERIALIZED_MAGIC = 0x55324632 as const; // 'U2F2'
export const QUERY_TOKENS_MAX = 128 as const;
export const RESULT_LIMIT_MAX = 8192 as const;

// v0.3 Document persistence constants (U2D3)
export const DOC_FORMAT_VERSION = 3 as const;
export const SERIALIZED_DOC_MAGIC = 0x55324433 as const; // 'U2D3'
export const SERIALIZED_DOC_HEADER_BYTES = 48 as const;

// v0.4 Document persistence constants (U2D4)
export const U2D4_MAGIC = 0x55324434 as const; // 'U2D4'
export const U2D4_FORMAT_VERSION = 4 as const;
export const FORMAT_VERSION_4 = 4 as const;
export const DOC_FORMAT_VERSION_4 = 4 as const;
export const U2D4_HEADER_BYTES = 56 as const;


export type TextProfileId = 'unicode-default';

/** Enum maps for the v0.2 binary header (U2F2). Unknown enum → IncompatibleIndexError. */
export const PROFILE_TO_ENUM: Record<TextProfileId, number> = { 'unicode-default': 1 };
export const ENUM_TO_PROFILE: Record<number, TextProfileId> = { 1: 'unicode-default' };
export const UNICODE_VERSION_TO_ENUM: Record<string, number> = { '16.0.0': 1 };
export const ENUM_TO_UNICODE_VERSION: Record<number, string> = { 1: '16.0.0' };
export const SCORING_TO_ENUM: Record<string, number> = { 'parity-v1': 1 };
export const ENUM_TO_SCORING: Record<number, string> = { 1: 'parity-v1' };

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

// Re-export all error classes from ./errors for backwards-compatible imports
export {
  QueryTooLongError,
  IncompatibleIndexError,
  ProfileMismatchError,
  IncompatibleOptionError,
  DuplicateIdError,
  DocumentNotFoundError,
  WebGPUSearchError,
  IncompatibleHookError,
  CostBudgetExceededError,
  InvalidFilterError
} from './errors';

