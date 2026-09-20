/**
 * v0.4 Error Hierarchy for webgpu-search.
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

// Re-export existing core errors for convenient unified error imports
export {
  QueryTooLongError,
  IncompatibleIndexError,
  ProfileMismatchError,
  IncompatibleOptionError,
  DuplicateIdError,
  DocumentNotFoundError
} from './text-profile';
