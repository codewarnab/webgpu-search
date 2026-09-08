import JSZip from 'jszip';
import type { BenchmarkRowResult } from './benchmark.ts';
import type { AdapterInfo } from './webgpu-engine.ts';

/**
 * Converts an inline SVGSVGElement into a high-resolution PNG Blob
 */
export async function svgToPngBlob(svgElement: SVGSVGElement, width = 1000, height = 260): Promise<Blob> {
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
    lines.push('# WebGPU vs uFuzzy Search Benchmark Export');
    lines.push(`# Generated At: ${new Date().toISOString()}`);
    lines.push(`# GPU Vendor: ${adapterInfo?.vendor || 'Unknown'}`);
    lines.push(`# GPU Device: ${adapterInfo?.device || 'Unknown'}`);
    lines.push(`# GPU Architecture: ${adapterInfo?.architecture || 'Unknown'}`);
    lines.push(`# Max VRAM Buffer (MB): ${adapterInfo?.maxBufferSizeMB || 0}`);
    lines.push(`# User Agent: "${navigator.userAgent.replace(/"/g, '""')}"`);
    lines.push('');
    lines.push([
        'Algorithm',
        'Dataset Size',
        'Query',
        'GPU Retained (ms)',
        'GPU Dispatch (ms)',
        'GPU Readback (ms)',
        'GPU Cold Total (ms)',
        'GPU Cold Upload (ms)',
        'uFuzzy CPU (ms)',
        'JS Native CPU (ms)',
        'Speedup vs uFuzzy',
        'Speedup vs Native',
        'Retained Winner',
        'GPU Total Matches',
        'uFuzzy Total Matches'
    ].join(','));

    const addRows = (rows: BenchmarkRowResult[]) => {
        for (const r of rows) {
            lines.push([
                r.mode.toUpperCase(),
                r.datasetSize,
                `"${r.query.replace(/"/g, '""')}"`,
                r.gpuRetained.totalMs,
                r.gpuRetained.gpuDispatchMs,
                r.gpuRetained.readbackMs,
                r.gpuCold.totalMs,
                r.gpuCold.uploadMs,
                r.ufuzzyMs,
                r.jsNativeMs,
                r.retainedVsUfuzzySpeedup,
                r.retainedVsNativeSpeedup,
                r.crossover.gpuRetainedBeatsUfuzzy ? 'GPU' : 'CPU',
                r.matchCount.gpu,
                r.matchCount.ufuzzy
            ].join(','));
        }
    };

    addRows(substringResults);
    addRows(fuzzyResults);

    return lines.join('\n');
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

/**
 * Packages all benchmark assets (CSV + PNGs + Specs JSON) into a single ZIP file
 */
export async function downloadFullReportZip(
    adapterInfo: AdapterInfo | null,
    substringResults: BenchmarkRowResult[],
    fuzzyResults: BenchmarkRowResult[],
    substringSvg: SVGSVGElement | null,
    fuzzySvg: SVGSVGElement | null
) {
    const zip = new JSZip();

    // 1. Add CSV
    const csvContent = generateBenchmarkCsv(adapterInfo, substringResults, fuzzyResults);
    zip.file('benchmark_results.csv', csvContent);

    // 2. Add System Specs JSON
    const specs = {
        timestamp: new Date().toISOString(),
        hardware: adapterInfo,
        browser: navigator.userAgent,
        summary: {
            substringRuns: substringResults.length,
            fuzzyRuns: fuzzyResults.length,
            testedSizes: substringResults.map(r => r.datasetSize)
        }
    };
    zip.file('system_specs.json', JSON.stringify(specs, null, 2));

    // 3. Add PNG charts
    if (substringSvg) {
        try {
            const blob = await svgToPngBlob(substringSvg);
            zip.file('chart_substring_benchmark.png', blob);
        } catch (err) {
            console.warn('Could not render substring PNG:', err);
        }
    }

    if (fuzzySvg) {
        try {
            const blob = await svgToPngBlob(fuzzySvg);
            zip.file('chart_fuzzy_benchmark.png', blob);
        } catch (err) {
            console.warn('Could not render fuzzy PNG:', err);
        }
    }

    // Generate ZIP and trigger download
    const zipBlob = await zip.generateAsync({ type: 'blob' });
    const dateStr = new Date().toISOString().slice(0, 10);
    downloadBlob(zipBlob, `webgpu_search_benchmark_${dateStr}.zip`);
}
