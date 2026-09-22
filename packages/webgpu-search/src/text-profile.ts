/**
 * Unicode text profile: version constants, profile types, error classes.
 * Frozen wire values mirror docs/text-normalization.md. Portable: no DOM refs.
 */

import { IncompatibleOptionError } from './errors';

export const UNICODE_VERSION = '16.0.0' as const;
export const SCORING_VERSION = 'parity-v1' as const;
// Canonical dataset (packed token stream) version + magic.
// Wire bytes: magic 'dataset' (0x55324632), little-endian header.
export const DATASET_FORMAT_VERSION = 2 as const;
export const DATASET_MAGIC = 0x55324632 as const;
export const QUERY_TOKENS_MAX = 128 as const;
export const RESULT_LIMIT_MAX = 8192 as const;

// Legacy document snapshot (read-only migration path). Version 3,
// magic 'legacy snapshot' (0x55324433), 48-byte header.
export const LEGACY_SNAPSHOT_VERSION = 3 as const;
export const LEGACY_SNAPSHOT_MAGIC = 0x55324433 as const;
export const LEGACY_SNAPSHOT_HEADER_BYTES = 48 as const;

// Canonical document snapshot. Version 4, magic 'snapshot' (0x55324434),
// 56-byte header. Byte values are frozen; only exported names changed.
export const SNAPSHOT_FORMAT_VERSION = 4 as const;
export const SNAPSHOT_MAGIC = 0x55324434 as const;
export const SNAPSHOT_HEADER_BYTES = 56 as const;


export type TextProfileId = 'unicode-default';

/** Enum maps for the dataset binary header. Unknown enum → IncompatibleIndexError. */
export const PROFILE_TO_ENUM: Record<TextProfileId, number> = { 'unicode-default': 1 };
export const ENUM_TO_PROFILE: Record<number, TextProfileId> = { 1: 'unicode-default' };
export const UNICODE_VERSION_TO_ENUM: Record<string, number> = { '16.0.0': 1 };
export const ENUM_TO_UNICODE_VERSION: Record<number, string> = { 1: '16.0.0' };
export const SCORING_TO_ENUM: Record<string, number> = { 'parity-v1': 1 };
export const ENUM_TO_SCORING: Record<number, string> = { 1: 'parity-v1' };

/**
 * Which CPU scorer the caller wants.
 *
 * - `'exact'`: contract default; the exact scorer (`exact-scorer.ts`).
 * - `'ufuzzy'`: explicit opt-in to the uFuzzy CPU scorer (CPU-only,
 * explicitly non-conforming scores, never routes to WebGPU, excluded from
 * the differential matrix).
 */
export type CpuScorer = 'exact' | 'ufuzzy';

/**
 * Normalize a caller-supplied scorer fail-closed.
 * Accepts only 'exact' | 'ufuzzy'; unknown values throw `IncompatibleOptionError`.
 */
export function normalizeCpuScorer(raw: CpuScorer | undefined): CpuScorer | undefined {
  if (raw === undefined) return undefined;
  if (raw !== 'exact' && raw !== 'ufuzzy') {
    throw new IncompatibleOptionError(
      'cpuScorer',
      `Unknown cpuScorer '${String(raw)}'. Expected 'exact' or 'ufuzzy'.`
    );
  }
  return raw;
}

export type OnQueryTooLong = 'throw' | 'cpu-fallback';

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
