import type { QuickBenchmarkResult } from './quick-benchmark';

export interface HistoryRecord {
  at: string;
  corpusSize: number;
  mode: string;
  query: string;
  gpuRan: boolean;
  gpuMedianMs: number;
  cpuMedianMs: number;
  gpuTier: string;
  adapterShort: string;
}

const KEY = 'bench:history:v1';
const MAX_ENTRIES = 20;

export function loadHistory(): HistoryRecord[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as HistoryRecord[]).slice(0, MAX_ENTRIES) : [];
  } catch {
    return [];
  }
}

export function saveRun(result: QuickBenchmarkResult): HistoryRecord[] {
  const record: HistoryRecord = {
    at: new Date().toISOString(),
    corpusSize: result.corpusSize,
    mode: result.mode,
    query: result.query.slice(0, 60),
    gpuRan: result.gpuRan,
    gpuMedianMs: Math.round(result.gpuMedianMs * 100) / 100,
    cpuMedianMs: Math.round(result.cpuMedianMs * 100) / 100,
    gpuTier: result.gpuTier.tier,
    adapterShort: result.adapter
      ? `${result.adapter.vendor} ${result.adapter.device}`.trim().slice(0, 80)
      : 'no adapter'
  };
  const next = [record, ...loadHistory()].slice(0, MAX_ENTRIES);
  try {
    localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    // quota/private mode: history just doesn't persist
  }
  return next;
}

export function clearHistory(): void {
  try {
    localStorage.removeItem(KEY);
  } catch {
    // ignore
  }
}
