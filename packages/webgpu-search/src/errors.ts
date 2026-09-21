/**
 * Error Hierarchy for webgpu-search.
 * Portable across browser main thread, Web Workers, Node.js, and SSR (zero DOM references).
 */

/**
 * Base error class for all webgpu-search specific errors.
 */
export class WebGPUSearchError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'WebGPUSearchError';
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * Thrown when snapshot restore encounters unregistered, mismatched, or corrupted extension hooks.
 */
export class IncompatibleHookError extends WebGPUSearchError {
  hookId: string;
  reason: string;

  constructor(hookId: string, reason: string, message?: string) {
    super(message ?? `Incompatible hook "${hookId}": ${reason}`);
    this.name = 'IncompatibleHookError';
    this.hookId = hookId;
    this.reason = reason;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * Thrown when search execution exceeds configured time or candidate limits.
 */
export class CostBudgetExceededError extends WebGPUSearchError {
  budgetType: 'time' | 'candidates';
  limit: number;
  actual: number;

  constructor(
    budgetType: 'time' | 'candidates',
    limit: number,
    actual: number,
    message?: string
  ) {
    super(
      message ??
        `Cost budget exceeded: ${budgetType} limit is ${limit}, reached ${actual}`
    );
    this.name = 'CostBudgetExceededError';
    this.budgetType = budgetType;
    this.limit = limit;
    this.actual = actual;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * Thrown when filter expressions contain invalid syntax, unsupported operators, or reference unindexed fields.
 */
export class InvalidFilterError extends WebGPUSearchError {
  field?: string;
  reason: string;

  constructor(reason: string, field?: string, message?: string) {
    super(
      message ??
        (field
          ? `Invalid filter on field "${field}": ${reason}`
          : `Invalid filter: ${reason}`)
    );
    this.name = 'InvalidFilterError';
    this.field = field;
    this.reason = reason;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

// Core engine errors extending WebGPUSearchError for unified error catching
export class QueryTooLongError extends RangeError {
  limit: number;
  actual: number;
  profileId: string;
  constructor(limit: number, actual: number, profileId: string) {
    super(`Query ${actual} tokens exceeds limit ${limit} (${profileId})`);
    this.name = 'QueryTooLongError';
    this.limit = limit;
    this.actual = actual;
    this.profileId = profileId;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

export class IncompatibleIndexError extends WebGPUSearchError {
  expected: unknown;
  actual: unknown;
  constructor(expected: unknown, actual: unknown) {
    super(
      `Incompatible index (expected ${String(expected)}, got ${String(actual)}). Rebuild.`
    );
    this.name = 'IncompatibleIndexError';
    this.expected = expected;
    this.actual = actual;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

export class ProfileMismatchError extends WebGPUSearchError {
  expected: unknown;
  actual: unknown;
  property: string;
  constructor(expected: unknown, actual: unknown, property: string = 'caseSensitive') {
    super(
      `Profile mismatch (${property}: expected ${String(expected)}, got ${String(actual)}). ` +
        `Rebuild the index or retry with ${property}=${String(expected)}.`
    );
    this.name = 'ProfileMismatchError';
    this.expected = expected;
    this.actual = actual;
    this.property = property;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

export class IncompatibleOptionError extends WebGPUSearchError {
  option: string;
  reason: string;
  constructor(option: string, reason: string) {
    super(`Incompatible option ${option}: ${reason}`);
    this.name = 'IncompatibleOptionError';
    this.option = option;
    this.reason = reason;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

export class DuplicateIdError extends WebGPUSearchError {
  id: string | number;
  constructor(id: string | number, message?: string) {
    super(message ?? `Duplicate document ID: ${String(id)}`);
    this.name = 'DuplicateIdError';
    this.id = id;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

export class DocumentNotFoundError extends WebGPUSearchError {
  id: string | number;
  constructor(id: string | number, message?: string) {
    super(message ?? `Document not found: ${String(id)}`);
    this.name = 'DocumentNotFoundError';
    this.id = id;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

