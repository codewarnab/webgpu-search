import {
  CPUEngine,
  normalizeText,
  packDataset,
  scoreExactMatches,
  WebGPUEngine,
  type SearchMode,
  type SearchResultItem
} from 'webgpu-search';
import { generateCorpus } from './corpus-gen';

export interface ExternalEngineResult {
  name: string;
  medianMs: number;
  p95Ms: number;
  matches: number;
  ran: boolean;
}

export interface QuickBenchmarkResult {
  gpuRan: boolean;
  gpuFailure: string;
  gpuMedianMs: number;
  gpuP95Ms: number;
  gpuMatches: number;
  gpuOverflow: boolean;
  cpuMedianMs: number;
  cpuP95Ms: number;
  cpuMatches: number;
  agree: boolean;
  deviceLabel: string;
  hits: SearchResultItem[];
  externals: ExternalEngineResult[];
  corpusSize: number;
  mode: SearchMode;
  query: string;
}

export { generateCorpus } from './corpus-gen';
export { formatMs } from './format';

export function median(samples: number[]): number {
  const sorted = [...samples].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] ?? 0;
}

export function p95(samples: number[]): number {
  const sorted = [...samples].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * 0.95) - 1)] ?? 0;
}

function resultsAgree(gpu: SearchResultItem[], cpu: SearchResultItem[], gpuTotal: number, cpuTotal: number): boolean {
  return gpuTotal === cpuTotal && gpu.length === cpu.length &&
    gpu.every((item, i) => item.index === cpu[i]?.index && item.score === cpu[i]?.score);
}

/**
 * Run the same generated corpus and query through WebGPU and the exact CPU
 * scorer. Setup (normalize/pack/upload) is excluded from warm-search timing.
 * Never fabricates a GPU timing: when WebGPU is unavailable or fails,
 * `gpuRan` is false and only CPU results are reported.
 */
export async function runQuickCompare(
  size: number,
  mode: SearchMode,
  query: string,
  onStatus: (message: string) => void
): Promise<QuickBenchmarkResult> {
  const WARMUPS = 2;
  const MEASUREMENTS = 7;
  const engine = new WebGPUEngine();
  try {
    onStatus('Preparing the local corpus…');
    const strings = generateCorpus(size);
    const tokens = strings.map(value => normalizeText(value, true).tokens);
    const queryTokens = normalizeText(query, true).tokens;
    const packed = packDataset(tokens, { normalized: true });

    let gpuRan = false;
    let gpuFailure = '';
    try {
      onStatus('Checking WebGPU support…');
      gpuRan = await engine.init();
      if (gpuRan) {
        onStatus('Uploading the corpus to WebGPU…');
        await engine.loadDataset(packed);
      }
    } catch (error) {
      gpuRan = false;
      gpuFailure = error instanceof Error ? error.message : String(error);
    }
    const adapter = engine.adapterInfo;

    onStatus('Warming up, then collecting timed searches…');
    const gpuSamples: number[] = [];
    let gpuLast: SearchResultItem[] = [];
    let gpuTotal = 0;
    let gpuOverflow = false;
    if (gpuRan) {
      try {
        for (let i = 0; i < WARMUPS; i++) {
          await engine.search(query, { mode, maxResults: 100 });
        }
        for (let i = 0; i < MEASUREMENTS; i++) {
          const start = performance.now();
          const response = await engine.search(query, { mode, maxResults: 100 });
          gpuSamples.push(performance.now() - start);
          gpuLast = response.results.map(item => ({ ...item, text: strings[item.index] ?? item.text }));
          gpuTotal = response.totalMatches;
          gpuOverflow = response.hasOverflow;
        }
      } catch (error) {
        gpuRan = false;
        gpuFailure = error instanceof Error ? error.message : String(error);
        gpuSamples.length = 0;
        gpuLast = [];
        gpuTotal = 0;
      }
    }

    onStatus('Measuring the exact CPU route…');
    const cpuRun = () => scoreExactMatches(tokens, queryTokens, mode, 100, strings);
    for (let i = 0; i < WARMUPS; i++) cpuRun();
    const cpuSamples: number[] = [];
    let cpuLast: SearchResultItem[] = [];
    let cpuTotal = 0;
    for (let i = 0; i < MEASUREMENTS; i++) {
      const start = performance.now();
      const response = cpuRun();
      cpuSamples.push(performance.now() - start);
      cpuLast = response.results;
      cpuTotal = response.totalMatches;
    }

    onStatus('Measuring the comparison libraries…');
    const externals: ExternalEngineResult[] = [];
    try {
      const cpuEngine = new CPUEngine();
      const timeIt = (run: () => unknown): number[] => {
        for (let i = 0; i < WARMUPS; i++) run();
        const samples: number[] = [];
        for (let i = 0; i < MEASUREMENTS; i++) {
          const start = performance.now();
          run();
          samples.push(performance.now() - start);
        }
        return samples;
      };
      let ufuzzyMatches = 0;
      const ufuzzySamples = timeIt(() => {
        ufuzzyMatches = cpuEngine.searchWithUFuzzy(strings, query, 100).totalMatches;
      });
      externals.push({
        name: 'uFuzzy',
        medianMs: median(ufuzzySamples),
        p95Ms: p95(ufuzzySamples),
        matches: ufuzzyMatches,
        ran: true
      });
    } catch {
      externals.push({ name: 'uFuzzy', medianMs: 0, p95Ms: 0, matches: 0, ran: false });
    }
    try {
      const { default: Fuse } = await import('fuse.js');
      const fuse = new Fuse(strings, { threshold: 0.4, ignoreLocation: true });
      let fuseMatches = 0;
      const fuseSamples: number[] = [];
      for (let i = 0; i < WARMUPS; i++) fuse.search(query);
      for (let i = 0; i < MEASUREMENTS; i++) {
        const start = performance.now();
        fuseMatches = fuse.search(query).length;
        fuseSamples.push(performance.now() - start);
      }
      externals.push({
        name: 'Fuse.js',
        medianMs: median(fuseSamples),
        p95Ms: p95(fuseSamples),
        matches: fuseMatches,
        ran: true
      });
    } catch {
      externals.push({ name: 'Fuse.js', medianMs: 0, p95Ms: 0, matches: 0, ran: false });
    }

    const deviceLabel = gpuRan
      ? `WebGPU ran on ${adapter ? [adapter.vendor, adapter.device].filter(Boolean).join(' · ') : 'this device'}.`
      : gpuFailure
        ? `WebGPU search failed (${gpuFailure}); CPU-only results shown.`
        : 'WebGPU did not initialize; CPU-only results shown.';

    return {
      gpuRan,
      gpuFailure,
      gpuMedianMs: median(gpuSamples),
      gpuP95Ms: p95(gpuSamples),
      gpuMatches: gpuTotal,
      gpuOverflow,
      cpuMedianMs: median(cpuSamples),
      cpuP95Ms: p95(cpuSamples),
      cpuMatches: cpuTotal,
      agree: gpuRan && resultsAgree(gpuLast, cpuLast, gpuTotal, cpuTotal),
      deviceLabel,
      hits: (gpuRan ? gpuLast : cpuLast).slice(0, 6),
      externals,
      corpusSize: size,
      mode,
      query
    };
  } finally {
    engine.destroy();
  }
}
