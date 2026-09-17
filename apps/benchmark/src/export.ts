import type { BenchmarkRowResult } from './benchmark';
import type { AdapterInfo } from 'webgpu-search';

/**
 * Converts an inline SVGSVGElement into a high-resolution PNG Blob
 */
export async function svgToPngBlob(svgElement: SVGSVGElement, width = 1000, height = 340): Promise<Blob> {
    const xml = new XMLSerializer().serializeToString(svgElement);
    const svgBlob = new Blob([xml], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(svgBlob);

    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
            const canvas = document.createElement('canvas');
            const scale = 2; // 2x for sharp rendering
            canvas.width = width * scale;
            canvas.height = height * scale;
            const ctx = canvas.getContext('2d');
            if (!ctx) {
                URL.revokeObjectURL(url);
                reject(new Error('Canvas 2D context not available'));
                return;
            }

            ctx.scale(scale, scale);
            // Draw dark background matching UI theme
            ctx.fillStyle = '#111827';
            ctx.fillRect(0, 0, width, height);
            ctx.drawImage(img, 0, 0, width, height);

            URL.revokeObjectURL(url);
            canvas.toBlob(blob => {
                if (blob) resolve(blob);
                else reject(new Error('Canvas toBlob failed'));
            }, 'image/png');
        };
        img.onerror = (e) => {
            URL.revokeObjectURL(url);
            reject(e);
        };
        img.src = url;
    });
}

/**
 * Generates an Excel-compatible CSV string from benchmark results and system specs
 */
export function generateBenchmarkCsv(
    adapterInfo: AdapterInfo | null,
    substringResults: BenchmarkRowResult[],
    fuzzyResults: BenchmarkRowResult[]
): string {
    const lines: string[] = [];
    const isGpuActive = !!adapterInfo;
    const gpuDevice = adapterInfo?.device || (isGpuActive ? 'WebGPU Device' : 'WebGPU Disabled (CPU Fallback Mode)');
    const gpuVendor = adapterInfo?.vendor || (isGpuActive ? 'Unknown Vendor' : 'N/A');
    const gpuArch = adapterInfo?.architecture || (isGpuActive ? 'Default' : 'N/A');
    const gpuRenderer = adapterInfo?.renderer || (isGpuActive ? gpuDevice : 'N/A');
    const gpuBufferMB = adapterInfo?.maxBufferSizeMB || 0;
    const gpuStorageMB = adapterInfo?.maxStorageBindingSizeMB || 0;
    const maxWorkgroups = adapterInfo?.maxComputeWorkgroupsPerDimension || 0;
    const maxInvocations = adapterInfo?.maxComputeInvocationsPerWorkgroup || 0;
    const hasTimestamp = adapterInfo?.hasTimestampQuery ? 'Yes' : 'No';

    // 1. Comprehensive System & GPU Metadata Header Block
    lines.push('# ========================================================');
    lines.push('# WebGPU vs CPU (uFuzzy) Search Benchmark Export');
    lines.push(`# Export Timestamp: ${new Date().toISOString()}`);
    lines.push(`# WebGPU Status: ${isGpuActive ? 'Active & Hardware Accelerated' : 'Disabled (Running in CPU Mode)'}`);
    lines.push(`# GPU Device: "${gpuDevice.replace(/"/g, '""')}"`);
    lines.push(`# GPU Vendor: "${gpuVendor.replace(/"/g, '""')}"`);
    lines.push(`# GPU Architecture: "${gpuArch.replace(/"/g, '""')}"`);
    lines.push(`# GPU Hardware Renderer: "${gpuRenderer.replace(/"/g, '""')}"`);
    lines.push(`# Max VRAM Buffer Size: ${gpuBufferMB} MB`);
    lines.push(`# Max Storage Binding Size: ${gpuStorageMB} MB`);
    lines.push(`# Max Compute Workgroups Per Dim: ${maxWorkgroups}`);
    lines.push(`# Max Compute Invocations Per Workgroup: ${maxInvocations}`);
    lines.push(`# Hardware Timestamp Queries: ${hasTimestamp}`);
    lines.push(`# Browser / OS User Agent: "${navigator.userAgent.replace(/"/g, '""')}"`);
    lines.push('# ========================================================');
    lines.push('');

    // 2. Data Table with GPU Details in Every Single Row
    lines.push([
        'Algorithm',
        'Dataset Size',
        'Query',
        'GPU Device',
        'GPU Vendor',
        'GPU Architecture',
        'Max VRAM Buffer (MB)',
        'GPU Retained Total (ms)',
        'GPU Encode/Submit (ms)',
        'GPU Execution ms (Timestamp Query)',
        'GPU Readback (ms)',
        'GPU Cold Total (ms)',
        'GPU Cold Upload (ms)',
        'uFuzzy CPU (ms)',
        'JS Native CPU (ms)',
        'Speedup vs uFuzzy',
        'Speedup vs Native',
        'Winner',
        'Worker UI FPS',
        'Main Thread UI FPS',
        'Main Thread Freeze (ms)',
        'Estimated Dropped Frames',
        'GPU Total Matches',
        'Candidate Overflow',
        'uFuzzy Total Matches'
    ].join(','));

    const addRows = (rows: BenchmarkRowResult[]) => {
        for (const r of rows) {
            const hasGpuTiming = r.gpuRetained.totalMs > 0;
            lines.push([
                r.mode.toUpperCase(),
                r.datasetSize,
                `"${r.query.replace(/"/g, '""')}"`,
                `"${gpuDevice.replace(/"/g, '""')}"`,
                `"${gpuVendor.replace(/"/g, '""')}"`,
                `"${gpuArch.replace(/"/g, '""')}"`,
                gpuBufferMB,
                hasGpuTiming ? r.gpuRetained.totalMs : 'N/A',
                hasGpuTiming ? r.gpuRetained.encodeSubmitMs : 'N/A',
                hasGpuTiming && r.gpuRetained.gpuExecutionMs !== null ? r.gpuRetained.gpuExecutionMs : 'N/A',
                hasGpuTiming ? r.gpuRetained.readbackMs : 'N/A',
                hasGpuTiming ? r.gpuCold.totalMs : 'N/A',
                hasGpuTiming ? r.gpuCold.uploadMs : 'N/A',
                r.ufuzzyMs,
                r.jsNativeMs,
                r.retainedVsUfuzzySpeedup > 0 ? `${r.retainedVsUfuzzySpeedup}x` : 'N/A',
                r.retainedVsNativeSpeedup > 0 ? `${r.retainedVsNativeSpeedup}x` : 'N/A',
                hasGpuTiming ? (r.crossover.gpuRetainedBeatsUfuzzy ? 'GPU' : 'uFuzzy CPU') : 'uFuzzy CPU (GPU Off)',
                r.uiTelemetry?.workerFps ?? 120,
                r.uiTelemetry?.mainThreadFps ?? 60,
                r.uiTelemetry?.mainThreadJankMs ?? 0,
                r.uiTelemetry?.droppedFrames ?? 0,
                r.matchCount.gpu,
                r.hasOverflow ? 'YES (>8192 matches)' : 'NO',
                r.matchCount.ufuzzy
            ].join(','));
        }
    };

    addRows(substringResults);
    addRows(fuzzyResults);

    return lines.join('\n');
}

/**
 * Formats a clean, shareable Markdown summary report of the benchmark results
 */
export function generateMarkdownSummary(
    adapterInfo: AdapterInfo | null,
    substringResults: BenchmarkRowResult[],
    fuzzyResults: BenchmarkRowResult[]
): string {
    const gpuName = adapterInfo ? `${adapterInfo.vendor} - ${adapterInfo.device} (${adapterInfo.architecture})` : 'WebGPU Disabled (CPU Mode)';
    const dateStr = new Date().toISOString().slice(0, 10);

    const formatTable = (rows: BenchmarkRowResult[], title: string) => {
        const lines: string[] = [];
        lines.push(`### ${title}`);
        lines.push('| Dataset Size | WebGPU Retained | uFuzzy (CPU) | Native JS | Speedup vs uFuzzy | UI Frame Rate (Worker vs Main) |');
        lines.push('| :--- | :--- | :--- | :--- | :--- | :--- |');
        for (const r of rows) {
            const gpuTime = r.gpuRetained.totalMs > 0 ? `${r.gpuRetained.totalMs} ms` : 'N/A';
            const speedup = r.retainedVsUfuzzySpeedup > 0 ? `**${r.retainedVsUfuzzySpeedup}x**` : '-';
            const fpsStr = r.uiTelemetry ? `Measured ${r.uiTelemetry.workerFps} FPS vs estimated ${r.uiTelemetry.mainThreadFps} FPS` : 'Not recorded';
            lines.push(`| **${r.datasetSize.toLocaleString()}** | ${gpuTime} | ${r.ufuzzyMs} ms | ${r.jsNativeMs} ms | ${speedup} | ${fpsStr} |`);
        }
        return lines.join('\n');
    };

    return [
        `# WebGPU vs CPU Search Benchmark Report`,
        `- **Hardware**: ${gpuName}`,
        `- **Date**: ${dateStr}`,
        `- **UI responsiveness**: Worker FPS is measured by the benchmark page; main-thread FPS is an estimate derived from blocking search time. Neither is a performance guarantee.`,
        `- **Timing scope**: Retained values are end-to-end search timings. Hardware timestamp-query availability and candidate overflow are included in the CSV export.`,
        '',
        formatTable(substringResults, '1. Exact Substring Search Matrix'),
        '',
        formatTable(fuzzyResults, '2. Fuzzy Subsequence Search Matrix'),
        '',
        `*Generated by [WebGPU Fuzzy Search](https://webgpu-fuzzy-search.vercel.app)*`
    ].join('\n');
}

/**
 * Triggers a browser file download for a Blob
 */
export function downloadBlob(blob: Blob, filename: string) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

