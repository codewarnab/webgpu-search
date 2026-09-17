/**
 * v0.2 Unicode text profile: version constants, profile types, error classes.
 * Frozen values mirror docs/unicode-contract.md (M1). Portable: no DOM refs.
 */

export const UNICODE_VERSION = '16.0.0' as const;
export const SCORING_VERSION = 'parity-v1' as const;
export const FORMAT_VERSION = 2 as const;
export const SERIALIZED_MAGIC = 0x55324632 as const; // 'U2F2'
export const QUERY_TOKENS_MAX = 128 as const;
export const RESULT_LIMIT_MAX = 8192 as const;

export type TextProfileId = 'unicode-default';

export type CpuAlgorithm = 'parity' | 'ufuzzy';

export type OnQueryTooLong = 'throw' | 'cpu-fallback';

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
      `Profile mismatch (expected ${String(expected)}, got ${String(actual)}). ` +
        `caseSensitive/profile is fixed at index-construction time.`
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
