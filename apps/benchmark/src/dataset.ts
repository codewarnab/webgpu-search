/**
 * Dataset generation and GPU buffer packing for search benchmarks.
 */

export interface Dataset {
    size: number;
    strings: string[];
    // Packed WebGPU buffer: 64 bytes (16 u32 words) per row
    gpuBufferData: ArrayBuffer;
    byteLength: number;
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
 * Generate N synthetic code paths/symbols and pack into JS strings & GPU byte layout
 */
export function generateDataset(count: number, onProgress?: (percent: number) => void): Dataset {
    const strings = new Array<string>(count);
    const byteLength = count * 64; // 16 u32s = 64 bytes per row
    const gpuBufferData = new ArrayBuffer(byteLength);
    const u32View = new Uint32Array(gpuBufferData);
    const u8View = new Uint8Array(gpuBufferData);

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

        // Pack into GPU buffer
        // Row layout (64 bytes):
        // Offset 0 (u32 0): string length (up to 59 chars)
        // Offset 4..63 (u32 1..15): ASCII characters
        const strLen = Math.min(path.length, 59);
        u32View[i * 16] = strLen;

        const baseByte = i * 64 + 4;
        for (let c = 0; c < strLen; c++) {
            u8View[baseByte + c] = path.charCodeAt(c);
        }

        if (onProgress && (i + 1) % reportInterval === 0) {
            onProgress(Math.round(((i + 1) / count) * 100));
        }
    }

    return {
        size: count,
        strings,
        gpuBufferData,
        byteLength
    };
}
