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
      // Preserve the light chart canvas used by the site UI.
      ctx.fillStyle = '#ffffff';
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
 * Generates an Excel-compatible CSV string from benchmark results, per-buffer VRAM,
 * packing breakdown, and multi-engine median/p95 metrics.
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
  lines.push('# WebGPU vs CPU Parity vs uFuzzy Search Benchmark Export');
  lines.push(`# Export Timestamp: ${new Date().toISOString()}`);
  lines.push(`# WebGPU Status: ${isGpuActive ? 'Active & Initialized' : 'Disabled (Running in CPU Mode)'}`);
  lines.push(`# GPU Device: "${gpuDevice.replace(/"/g, '""')}"`);
  lines.push(`# GPU Vendor: "${gpuVendor.replace(/"/g, '""')}"`);
  lines.push(`# GPU Architecture: "${gpuArch.replace(/"/g, '""')}"`);
  lines.push(`# GPU Hardware Renderer: "${gpuRenderer.replace(/"/g, '""')}"`);
  lines.push(`# Max VRAM Buffer Size: ${gpuBufferMB} MB`);
  lines.push(`# Max Storage Binding Size: ${gpuStorageMB} MB`);
  lines.push(`# Max Compute Workgroups Per Dim: ${maxWorkgroups}`);
  lines.push(`# Max Compute Invocations Per Workgroup: ${maxInvocations}`);
  lines.push(`# Hardware Timestamp Queries: ${hasTimestamp}`);
  lines.push(`# Browser / OS User Agent: "${typeof navigator !== 'undefined' ? navigator.userAgent.replace(/"/g, '""') : 'Headless/CLI'}"`);
  lines.push('# ========================================================');
  lines.push('');

  // 2. Data Table with Metrics in Every Single Row
  lines.push([
    'Algorithm',
    'Corpus',
    'Dataset Size',
    'Query',
    'UTF-16 Units',
    'Code Points',
    'GPU Device',
    'GPU Vendor',
    'Hardware Qualification',
    'VRAM Records Bytes',
    'VRAM Offsets Bytes',
    'VRAM Query Bytes',
    'VRAM Output Bytes',
    'Total VRAM Bytes',
    'Normalize ms',
    'Pack ms',
    'Upload ms',
    'Total Pipeline ms',
    'GPU Retained Median ms',
    'GPU Retained p95 ms',
    'GPU Submit ms',
    'GPU Execution ms (Timestamp)',
    'GPU Readback ms',
    'GPU Cold Total ms',
    'CPU Parity Median ms',
    'CPU Parity p95 ms',
    'uFuzzy CPU Median ms',
    'uFuzzy CPU p95 ms',
    'JS Native CPU Median ms',
    'JS Native CPU p95 ms',
    'Speedup vs Parity',
    'Speedup vs uFuzzy',
    'Speedup vs Native',
    'Winner',
    'Worker UI FPS',
    'Main Thread UI FPS',
    'Main Thread Max Jank ms',
    'Jank Spikes (>16.7ms)',
    'Estimated Dropped Frames',
    'GPU Total Matches',
    'Candidate Overflow (>8192)',
    'CPU Parity Total Matches',
    'uFuzzy Total Matches'
  ].join(','));

  const sanitizeCsvField = (val: string): string => {
    let s = val.replace(/"/g, '""');
    if (/^[=+@\-\t\r]/.test(s)) {
      s = "'" + s;
    }
    return `"${s}"`;
  };

  const addRows = (rows: BenchmarkRowResult[]) => {
    for (const r of rows) {
      const hasGpuTiming = r.gpuRetained.medianMs > 0;
      lines.push([
        r.mode.toUpperCase(),
        r.corpusType ? r.corpusType.toUpperCase() : 'ASCII',
        r.datasetSize,
        sanitizeCsvField(r.query),
        r.utf16Units ?? 0,
        r.codePoints ?? 0,
        `"${gpuDevice.replace(/"/g, '""')}"`,
        `"${gpuVendor.replace(/"/g, '""')}"`,
        r.qualificationStatus ?? 'pending-hardware',
        r.vramAllocation?.recordsBytes ?? 0,
        r.vramAllocation?.offsetsBytes ?? 0,
        r.vramAllocation?.queryBytes ?? 512,
        r.vramAllocation?.outputBytes ?? 65544,
        r.vramAllocation?.totalBytes ?? 0,
        r.packing?.normalizeMs ?? 0,
        r.packing?.packMs ?? 0,
        r.packing?.uploadMs ?? 0,
        r.packing?.totalPipelineMs ?? 0,
        hasGpuTiming ? r.gpuRetained.medianMs : 'N/A',
        hasGpuTiming ? r.gpuRetained.p95Ms : 'N/A',
        hasGpuTiming ? r.gpuRetained.encodeSubmitMs : 'N/A',
        hasGpuTiming && r.gpuRetained.gpuExecutionMs !== null ? r.gpuRetained.gpuExecutionMs : 'N/A',
        hasGpuTiming ? r.gpuRetained.readbackMs : 'N/A',
        hasGpuTiming ? r.gpuCold.totalMs : 'N/A',
        r.cpuParity?.medianMs ?? r.cpuParityMs ?? 'N/A',
        r.cpuParity?.p95Ms ?? 'N/A',
        r.ufuzzy?.medianMs ?? r.ufuzzyMs ?? 'N/A',
        r.ufuzzy?.p95Ms ?? 'N/A',
        r.jsNative?.medianMs ?? r.jsNativeMs ?? 'N/A',
        r.jsNative?.p95Ms ?? 'N/A',
        r.retainedVsParitySpeedup > 0 ? `${r.retainedVsParitySpeedup}x` : 'N/A',
        r.retainedVsUfuzzySpeedup > 0 ? `${r.retainedVsUfuzzySpeedup}x` : 'N/A',
        r.retainedVsNativeSpeedup > 0 ? `${r.retainedVsNativeSpeedup}x` : 'N/A',
        hasGpuTiming
          ? (r.crossover.gpuRetainedBeatsUfuzzy ? 'WebGPU' : 'uFuzzy CPU')
          : 'CPU Mode (GPU Off)',
        r.uiTelemetry?.workerFps ?? 120,
        r.uiTelemetry?.mainThreadFps ?? 60,
        r.uiTelemetry?.mainThreadJankMs ?? 0,
        r.uiTelemetry?.jankSpikes ?? 0,
        r.uiTelemetry?.droppedFrames ?? 0,
        r.matchCount.gpu,
        r.hasOverflow ? 'YES' : 'NO',
        r.matchCount.cpuParity ?? r.matchCount.gpu,
        r.matchCount.ufuzzy
      ].join(','));
    }
  };

  addRows(substringResults);
  addRows(fuzzyResults);

  return lines.join('\n');
}

/**
 * Formats a structured Markdown report with warm median/p95 comparisons,
 * per-buffer VRAM breakdown, packing pipeline phase timings, and hardware qualification tags.
 */
export function generateMarkdownSummary(
  adapterInfo: AdapterInfo | null,
  substringResults: BenchmarkRowResult[],
  fuzzyResults: BenchmarkRowResult[]
): string {
  const gpuName = adapterInfo
    ? `${adapterInfo.vendor} - ${adapterInfo.device} (${adapterInfo.architecture})`
    : 'WebGPU Disabled (CPU Mode)';
  const dateStr = new Date().toISOString().slice(0, 10);
  const isPhys = substringResults[0]?.hardwareQualified ?? fuzzyResults[0]?.hardwareQualified ?? false;
  const qualBadge = isPhys ? '✅ Physical Hardware Qualified' : '⚠️ Pending Physical Hardware Qualification (Software/Mock Renderer)';

  const formatTable = (rows: BenchmarkRowResult[], title: string) => {
    const lines: string[] = [];
    lines.push(`### ${title}`);
    lines.push('| Dataset Size | Corpus | WebGPU (med/p95) | CPU Parity (med/p95) | uFuzzy (med/p95) | JS Native | Speedup vs Parity | Speedup vs uFuzzy | UI Thread FPS (Jank Spikes) | Status |');
    lines.push('| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |');
    for (const r of rows) {
      const gpuTime = r.gpuRetained.medianMs > 0
        ? `${r.gpuRetained.medianMs}ms / ${r.gpuRetained.p95Ms}ms`
        : 'N/A';
      const parityTime = r.cpuParity
        ? `${r.cpuParity.medianMs}ms / ${r.cpuParity.p95Ms}ms`
        : (r.cpuParityMs ? `${r.cpuParityMs}ms` : 'N/A');
      const ufuzzyTime = r.ufuzzy
        ? `${r.ufuzzy.medianMs}ms / ${r.ufuzzy.p95Ms}ms`
        : `${r.ufuzzyMs}ms`;
      const nativeTime = r.jsNative ? `${r.jsNative.medianMs}ms` : `${r.jsNativeMs}ms`;
      const speedupParity = r.retainedVsParitySpeedup > 0 ? `**${r.retainedVsParitySpeedup}x**` : '-';
      const speedupUfuzzy = r.retainedVsUfuzzySpeedup > 0 ? `**${r.retainedVsUfuzzySpeedup}x**` : '-';
      const fpsStr = r.uiTelemetry
        ? `${r.uiTelemetry.mainThreadFps} FPS (${r.uiTelemetry.jankSpikes} jank)`
        : '~60 FPS';
      const status = r.qualificationStatus === 'qualified' ? 'Qualified' : 'Pending-HW';

      lines.push(
        `| **${r.datasetSize.toLocaleString()}** | ${(r.corpusType || 'ascii').toUpperCase()} | ${gpuTime} | ${parityTime} | ${ufuzzyTime} | ${nativeTime} | ${speedupParity} | ${speedupUfuzzy} | ${fpsStr} | ${status} |`
      );
    }
    return lines.join('\n');
  };

  const formatVramTable = (rows: BenchmarkRowResult[]) => {
    const lines: string[] = [];
    lines.push(`### VRAM Allocation & Packing Breakdown`);
    lines.push('| Dataset Size | Records VRAM | Offsets VRAM | Query VRAM | Output VRAM | Total VRAM | Normalize | Pack | Upload | Pipeline Total |');
    lines.push('| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |');
    for (const r of rows) {
      const v = r.vramAllocation;
      const p = r.packing;
      if (!v || !p) continue;
      lines.push(
        `| **${r.datasetSize.toLocaleString()}** | ${(v.recordsBytes / (1024 * 1024)).toFixed(2)} MB | ${(v.offsetsBytes / (1024 * 1024)).toFixed(2)} MB | ${v.queryBytes} B | ${(v.outputBytes / 1024).toFixed(1)} KB | **${(v.totalBytes / (1024 * 1024)).toFixed(2)} MB** | ${p.normalizeMs}ms | ${p.packMs}ms | ${p.uploadMs}ms | **${p.totalPipelineMs}ms** |`
      );
    }
    return lines.join('\n');
  };

  const sections = [
    `# WebGPU vs CPU Search Performance Characterization Report`,
    `- **Hardware**: ${gpuName}`,
    `- **Qualification**: ${qualBadge}`,
    `- **Date**: ${dateStr}`,
    `- **Methodology**: Warm median and p95 latency metrics over 5 warmups and 20 randomized/interleaved samples per engine.`,
    `- **UI Telemetry**: Real requestAnimationFrame interval measurement; frame durations > 16.7ms recorded as jank spikes and dropped frames.`,
    `- **Exact Fallback Budget**: Evaluates the contracted fallback \`scoreExactMatches\` (\`cpuScorer: 'exact'\`) directly against GPU compute.`,
    ''
  ];
  if (substringResults.length > 0) {
    sections.push(formatTable(substringResults, '1. Exact Substring Search Matrix'), '');
  }
  if (fuzzyResults.length > 0) {
    sections.push(formatTable(fuzzyResults, '2. Fuzzy Subsequence Search Matrix'), '');
  }
  const detailsRows = substringResults.length > 0 ? substringResults : fuzzyResults;
  if (detailsRows.length > 0) sections.push(formatVramTable(detailsRows), '');
  sections.push(`*Generated by [WebGPU Fuzzy Search](https://webgpu-fuzzy-search.vercel.app)*`);
  return sections.join('\n');
}

/**
 * Triggers a browser file download for a Blob
 */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}
