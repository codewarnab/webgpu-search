import type { LogLevel, StructuredLogRecord } from './types';

const SERVICES = [
  'auth-service',
  'api-gateway',
  'payment-gateway',
  'gpu-indexer',
  'db-cluster',
  'cache-proxy',
  'pipeline-worker',
  'search-engine'
];

const ERROR_TEMPLATES = [
  'Context timeout after 5000ms waiting for WebGPU fence completion',
  'Failed to acquire device: adapter request timed out or device lost',
  'Database deadlock detected during row lock acquisition on table orders',
  'JWT token signature verification failed: expired token timestamp',
  'Heap out of memory: VRAM buffer allocation exceeded 134217728 bytes',
  'Connection pool exhausted: maximum active connections reached (500/500)',
  'Worker thread pool crashed unexpectedly with uncaught DOMException',
  'CRC32 checksum mismatch on restored snapshot binary header: expected 0x5a2f1b, got 0x3e8a91'
];

const WARN_TEMPLATES = [
  'Candidate pool saturation warning: candidateCount reached 8192 capacity',
  'High latency spike observed on upstream RPC payment_verify (842ms)',
  'Slow query detected: table scan on customer_records took 312ms',
  'Tombstone ratio reached 26%: triggering CPU compaction repacking',
  'Rate limit threshold approaching 90% for client IP 192.168.1.45',
  'Deprecated API called: packStringsToGPUBuffer legacy ascii mode',
  'Adapter lacks timestamp-query capability: falling back to wall-clock timing'
];

const INFO_TEMPLATES = [
  'NFC normalization pass completed successfully in 0.38ms',
  'IndexedDB snapshot committed cleanly with CRC32 checksum 0x7a8b9c1d',
  'Worker thread pool initialized with 8 dedicated execution contexts',
  'Batch mutation applied: added 200 records, mutation epoch advanced to 14',
  'Healthcheck probe status OK for pod webgpu-search-worker-node-8',
  'VRAM buffer allocated cleanly: 65544 bytes staging buffer ready',
  'Subsequence greedy match aligned with 100% score-highlight parity',
  'Field-stratified token packing completed across 4 target fields'
];

const DEBUG_TEMPLATES = [
  'Binding layout group 0: offset 0, size 65544, access read_write',
  'Cache hit for key session:98471b in Redis cluster shard 4',
  'Dispatched workgroups vec3(625, 1, 1) across 80000 total rows',
  // Deliberately historical fixture: logs in the wild contain legacy magic + manager names.
  'Parsed 32-bit Little-Endian magic constant 0x55324433 (U2D3)',
  'Acquired shared device lock in WebGPUContextManager',
  'Tombstones filter evaluated: 0 dropped candidates in candidate readback'
];

function generateHex(len: number, seed: number): string {
  let hex = '';
  for (let i = 0; i < len; i++) {
    const val = (seed * 9301 + 49297 + i * 233280) % 16;
    hex += Math.floor(Math.abs(val)).toString(16);
  }
  return hex;
}

export function generateStructuredLogs(count: number, startId: number = 1, baseTimeMs?: number): StructuredLogRecord[] {
  const records = new Array<StructuredLogRecord>(count);
  // Frozen-benchmark anchor: callers (bench-snapshot-matrix) pass
  // BENCHMARK_LOG_BASE_TIME_MS so snapshot bytes are reproducible run-to-run.
  // Default preserves the legacy wall-clock behavior.
  const now = baseTimeMs ?? Date.now();

  for (let i = 0; i < count; i++) {
    const idNum = startId + i;
    const id = `log-${String(idNum).padStart(7, '0')}`;
    const service = SERVICES[i % SERVICES.length]!;

    let level: LogLevel = 'INFO';
    let message = '';
    let latency = 5 + (i * 7) % 250;

    const mod = i % 100;
    if (mod < 5) {
      level = 'ERROR';
      message = ERROR_TEMPLATES[i % ERROR_TEMPLATES.length]!;
      latency += 800;
    } else if (mod < 20) {
      level = 'WARN';
      message = WARN_TEMPLATES[i % WARN_TEMPLATES.length]!;
      latency += 250;
    } else if (mod < 75) {
      level = 'INFO';
      message = INFO_TEMPLATES[i % INFO_TEMPLATES.length]!;
    } else {
      level = 'DEBUG';
      message = DEBUG_TEMPLATES[i % DEBUG_TEMPLATES.length]!;
      latency = 1 + (i % 8);
    }

    const timestampMs = now - (count - i) * 80;
    const timestamp = new Date(timestampMs).toISOString();
    const traceId = `tr-${generateHex(8, i + 100)}`;

    records[i] = {
      id,
      timestamp,
      level,
      service,
      message,
      traceId,
      latencyMs: latency
    };
  }

  return records;
}
