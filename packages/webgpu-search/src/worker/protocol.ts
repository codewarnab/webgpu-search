import type {
  AddOptions,
  DocumentIndexOptions,
  DocumentSearchOptions,
  MutationBatch,
  WorkerClientOptions,
  WorkerMessageType,
  WorkerRequest,
  WorkerResponse
} from '../types';

export type {
  WorkerClientOptions,
  WorkerMessageType,
  WorkerRequest,
  WorkerResponse
};

export interface WorkerInitPayload<TDoc = Record<string, unknown>> {
  options: DocumentIndexOptions<TDoc>;
}

export interface WorkerSearchPayload<TDoc = Record<string, unknown>> {
  query: string;
  options?: DocumentSearchOptions<TDoc>;
}

export interface WorkerMutatePayload<TDoc = Record<string, unknown>> {
  batch: MutationBatch<TDoc>;
  options?: AddOptions;
}

export interface WorkerRestorePayload {
  buffer: ArrayBuffer;
  transfer?: boolean;
}
