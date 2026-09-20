export type LogLevel = 'INFO' | 'WARN' | 'ERROR' | 'DEBUG';

export interface StructuredLogRecord {
  id: string;
  timestamp: string;
  level: LogLevel;
  service: string;
  message: string;
  traceId: string;
  latencyMs: number;
}

export interface LogViewerSearchResult {
  id: string;
  score: number;
  matchedField: string;
  doc: StructuredLogRecord;
  highlightedText?: Record<string, string>;
  highlights?: Record<string, Array<{ start: number; end: number }>>;
}

export interface LogViewerTelemetry {
  totalLogs: number;
  totalRows: number;
  totalMatches: number;
  searchDurationMs: number;
  engine: 'webgpu' | 'cpu';
  fallbackReason?: string;
  vramBytes: number;
  ramBytes: number;
  mutationEpoch: number;
  tombstoneCount: number;
  tombstoneRatio: number;
  idbSnapshotBytes?: number;
  idbRestoreDurationMs?: number;
}
