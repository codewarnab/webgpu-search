import type {
  AddOptions,
  DocumentIndexOptions,
  DocumentSearchOptions,
  MutationBatch,
  RestoreDocumentIndexOptions,
  SerializeDocumentIndexOptions,
  WorkerClientOptions,
  WorkerMessageType,
  WorkerRequest,
  WorkerResponse
} from '../types';
import {
  QueryTooLongError,
  IncompatibleIndexError,
  ProfileMismatchError,
  IncompatibleOptionError,
  DuplicateIdError,
  DocumentNotFoundError
} from '../text-profile';
import { abortError } from '../runtime-guards';

export type {
  WorkerClientOptions,
  WorkerMessageType,
  WorkerRequest,
  WorkerResponse
};

export interface SerializedWorkerError {
  name: string;
  message: string;
  stack?: string;
  details?: Record<string, unknown>;
}

export interface WorkerInitPayload<TDoc = Record<string, unknown>> {
  options: DocumentIndexOptions<TDoc>;
  records?: TDoc[];
}

export interface WorkerSearchPayload<TDoc = Record<string, unknown>> {
  queryId: number;
  query: string;
  options?: DocumentSearchOptions<TDoc>;
  stringIsolated?: boolean;
}

export interface WorkerAbortPayload {
  queryId: number;
}

export interface WorkerMutatePayload<TDoc = Record<string, unknown>> {
  batch: MutationBatch<TDoc>;
  options?: AddOptions;
}

export interface WorkerSerializePayload {
  options?: SerializeDocumentIndexOptions;
}

export interface WorkerRestorePayload {
  buffer: ArrayBuffer;
  options?: RestoreDocumentIndexOptions;
}

/**
 * Serialize an Error object (including custom library errors) into a structured-clone-safe format.
 */
export function serializeError(err: unknown): SerializedWorkerError {
  if (err instanceof Error || (err && typeof err === 'object' && 'name' in err && 'message' in err)) {
    const e = err as any;
    const name: string = typeof e.name === 'string' ? e.name : 'Error';
    const message: string = typeof e.message === 'string' ? e.message : String(err);
    const stack: string | undefined = typeof e.stack === 'string' ? e.stack : undefined;
    const details: Record<string, unknown> = {};

    if ('limit' in e) details.limit = e.limit;
    if ('actual' in e) details.actual = e.actual;
    if ('profileId' in e) details.profileId = e.profileId;
    if ('expected' in e) details.expected = e.expected;
    if ('property' in e) details.property = e.property;
    if ('option' in e) details.option = e.option;
    if ('reason' in e) details.reason = e.reason;
    if ('id' in e) details.id = e.id;
    if (e.details && typeof e.details === 'object') {
      Object.assign(details, e.details);
    }

    return {
      name,
      message,
      stack,
      details: Object.keys(details).length > 0 ? details : undefined
    };
  }

  return {
    name: 'Error',
    message: String(err)
  };
}

/**
 * Rehydrate a serialized error into its exact error class instance across the thread boundary.
 */
export function deserializeError(serialized: SerializedWorkerError): Error {
  if (!serialized || typeof serialized !== 'object') {
    return new Error('Unknown worker error');
  }

  const { name, message, stack, details = {} } = serialized;
  let error: Error;

  switch (name) {
    case 'QueryTooLongError':
      error = new QueryTooLongError(
        (details.limit as number) ?? 128,
        (details.actual as number) ?? 0,
        (details.profileId as string) ?? 'unicode-default'
      );
      break;
    case 'IncompatibleIndexError':
      error = new IncompatibleIndexError(details.expected, details.actual);
      break;
    case 'ProfileMismatchError':
      error = new ProfileMismatchError(
        details.expected,
        details.actual,
        (details.property as string) ?? 'caseSensitive'
      );
      break;
    case 'IncompatibleOptionError':
      error = new IncompatibleOptionError(
        (details.option as string) ?? 'option',
        (details.reason as string) ?? message
      );
      break;
    case 'DuplicateIdError':
      error = new DuplicateIdError(details.id as string | number, message);
      break;
    case 'DocumentNotFoundError':
      error = new DocumentNotFoundError(details.id as string | number, message);
      break;
    case 'AbortError':
      error = abortError();
      break;
    case 'TypeError':
      error = new TypeError(message);
      break;
    case 'RangeError':
      error = new RangeError(message);
      break;
    default:
      error = new Error(message);
      error.name = name;
      break;
  }

  if (stack) {
    error.stack = stack;
  }

  // Preserve any extra details if not already assigned
  for (const [key, value] of Object.entries(details)) {
    if (!(key in error)) {
      (error as any)[key] = value;
    }
  }

  return error;
}
