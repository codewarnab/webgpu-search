/**
 * Dataset generation and U2F2 unicode packing for search benchmarks.
 *
 * M4 (Issue #7): the legacy v0.1 ASCII-mangling byte packer
 * (`packStringsToGPUBuffer`, NFKD + `?` replacement) is deleted from this
 * flow. Datasets are packed with the code-point-safe unicode pipeline
 * (`packUnicodeToGPUBuffer` → `serializeUnicodeDataset`); the serialized
 * U2F2 buffer is what crosses the worker boundary (transferable, zero-copy
 * with a transfer list — unlike `strings`, which always structured-clones).
 */
import {
  packUnicodeToGPUBuffer,
  serializeUnicodeDataset,
} from 'webgpu-search';

export interface Dataset {
  size: number;
  strings: string[];
  /** U2F2 serialized unicode dataset (header + u32 records + u32 offsets). */
  serializedU2F2: ArrayBuffer;
  serializedByteLength: number;
  /** Post-fold code-point total (exact, from the unicode packer). */
  tokenCount: number;
  /** records + offsets bytes actually allocated (no 64 B fiction). */
  packedBytes: number;
  folded: boolean;
}

const PREFIXES = [
  'src/components', 'src/views', 'src/utils', 'src/hooks', 'src/store',
  'src/services', 'src/api', 'src/lib', 'crates/core/src', 'crates/engine/src',
  'packages/client', 'packages/server', 'internal/router', 'internal/auth',
  'cmd/server', 'pkg/middleware', 'tests/e2e', 'docs/tutorials', 'scripts/ci',
  'node_modules/@tanstack', 'node_modules/@trpc', 'node_modules/vite', 'vendor/bundle'
];

const NOUNS = [
  'User', 'Account', 'Session', 'Profile', 'Auth', 'Token', 'Order', 'Invoice',
  'Payment', 'Billing', 'Product', 'Catalog', 'Item', 'Cart', 'Checkout', 'Delivery',
  'Search', 'Filter', 'Index', 'Query', 'Table', 'Column', 'Database', 'Cache',
  'Router', 'Socket', 'Worker', 'Queue', 'Job', 'Scheduler', 'Metric', 'Telemetry',
  'Log', 'Trace', 'Span', 'Config', 'Setting', 'Theme', 'Color', 'Canvas', 'Shader',
  'Texture', 'Mesh', 'Pipeline', 'Buffer', 'Array', 'Vector', 'Matrix', 'Engine'
];

const SUFFIXES = [
  'Controller', 'Service', 'Handler', 'Manager', 'Provider', 'Context', 'Hook',
  'Component', 'View', 'Modal', 'Drawer', 'Dropdown', 'Tooltip', 'Button', 'Input',
  'Validator', 'Serializer', 'Parser', 'Transformer', 'Adapter', 'Repository',
  'Client', 'Gateway', 'Middleware', 'Reducer', 'Dispatcher', 'Observer', 'Factory'
];

const EXTENSIONS = ['.ts', '.tsx', '.rs', '.go', '.py', '.js', '.jsx', '.json', '.wgsl', '.css'];

/**
 * Generate N synthetic code paths/symbols plus the U2F2 transfer buffer.
 * Pack mode is folded (`caseSensitive:false`), matching the benchmark
 * engine default — per-query mismatch would throw ProfileMismatchError.
 */
export function generateDataset(count: number, onProgress?: (percent: number) => void): Dataset {
  const strings = new Array<string>(count);
  const prefixLen = PREFIXES.length;
  const nounLen = NOUNS.length;
  const suffixLen = SUFFIXES.length;
  const extLen = EXTENSIONS.length;

  const reportInterval = Math.max(10000, Math.floor(count / 10));

  for (let i = 0; i < count; i++) {
    // Deterministic pseudo-random generation
    const p = PREFIXES[i % prefixLen];
    const n1 = NOUNS[(i * 7 + 3) % nounLen];
    const n2 = NOUNS[(i * 13 + 11) % nounLen];
    const s = SUFFIXES[(i * 17 + 5) % suffixLen];
    const ext = EXTENSIONS[(i * 23 + 7) % extLen];

    // Format: prefix/NounNounSuffix_1234.ext
    const path = `${p}/${n1}${n2}${s}_${i}${ext}`;
    strings[i] = path;

    if (onProgress && (i + 1) % reportInterval === 0) {
      onProgress(Math.round(((i + 1) / count) * 100));
    }
  }

  const folded = true;
  const packed = packUnicodeToGPUBuffer(strings, { folded });
  const serializedU2F2 = serializeUnicodeDataset(packed);

  return {
    size: count,
    strings,
    serializedU2F2,
    serializedByteLength: serializedU2F2.byteLength,
    tokenCount: packed.tokenCount,
    packedBytes: packed.combinedByteLength,
    folded
  };
}
