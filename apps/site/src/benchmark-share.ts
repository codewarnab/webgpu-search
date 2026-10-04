import type { AdapterInfo, GpuClassification } from 'webgpu-search';
import type { QuickBenchmarkResult } from './quick-benchmark';

export interface SharePayload {
  fingerprint: string;
  adapter: {
    vendor: string;
    device: string;
    architecture: string;
    renderer: string;
    adapterType?: string;
  };
  gpuTier: GpuClassification;
  browser: string;
  os: string;
  corpusSize: number;
  mode: string;
  query: string;
  gpuMedianMs: number;
  gpuP95Ms: number;
  gpuMatches: number;
  cpuMedianMs: number;
  cpuP95Ms: number;
  cpuMatches: number;
  externals: Array<{ name: string; medianMs: number; p95Ms: number; matches: number; ran: boolean }>;
  createdAt: string;
}

const SHARED_PREFIX = 'bench:shared:';

export function fnv1aHex(input: string): string {
  let hash = 0x811c9dc5;
  for (let i = 0; i < input.length; i++) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return (`0000000${(hash >>> 0).toString(16)}`).slice(-8);
}

export function browserFamily(ua: string): string {
  const s = ua.toLowerCase();
  if (s.includes('edg/') || s.includes('edge/')) return 'edge';
  if (s.includes('opr/') || s.includes('opera')) return 'opera';
  if (s.includes('chrome/') && !s.includes('chromium')) return 'chrome';
  if (s.includes('chromium')) return 'chromium';
  if (s.includes('firefox/') || s.includes('fxios')) return 'firefox';
  if (s.includes('safari/') && s.includes('version/')) return 'safari';
  return 'other';
}

export function osFamily(ua: string, platform = ''): string {
  const s = `${ua} ${platform}`.toLowerCase();
  if (/android/.test(s)) return 'android';
  if (/iphone|ipad|ios/.test(s)) return 'ios';
  if (/windows/.test(s)) return 'windows';
  if (/mac os|macintosh/.test(s)) return 'macos';
  if (/cros/.test(s)) return 'chromeos';
  if (/linux/.test(s)) return 'linux';
  return 'other';
}

function normalizeAdapter(a: AdapterInfo | null): string {
  if (!a) return 'no-adapter';
  return [a.vendor, a.device, a.architecture, a.renderer, a.adapterType ?? '']
    .map((v) => v.trim().toLowerCase().replace(/\s+/g, ' '))
    .join('|');
}

/** Stable dedup key: adapter + browser/OS family + workload shape. */
export function buildFingerprint(
  adapter: AdapterInfo | null,
  corpusSize: number,
  mode: string,
  ua = typeof navigator !== 'undefined' ? navigator.userAgent : 'headless'
): string {
  const key = `${normalizeAdapter(adapter)}#${browserFamily(ua)}#${osFamily(ua, typeof navigator !== 'undefined' ? (navigator as unknown as { platform?: string }).platform ?? '' : '')}#${corpusSize}#${mode}`;
  return `webgpu-${fnv1aHex(key)}`;
}

/** Server-side limits (api/submit-benchmark.ts); clamp so a valid run is never rejected. */
const LIMITS = { adapterField: 300, adapterType: 64, tierReason: 300, family: 32, query: 120, externalName: 32, externals: 8, ms: 60000, count: 2000000 } as const;

const clampStr = (value: string, max: number): string => value.slice(0, max);
const clampMs = (value: number): number => (Number.isFinite(value) ? Math.min(LIMITS.ms, Math.max(0, value)) : 0);
const clampCount = (value: number): number =>
  (Number.isFinite(value) ? Math.min(LIMITS.count, Math.max(0, Math.floor(value))) : 0);

export function buildSharePayload(result: QuickBenchmarkResult): SharePayload {
  const ua = typeof navigator !== 'undefined' ? navigator.userAgent : 'headless';
  const field = (value: string | undefined): string => clampStr(value ?? '', LIMITS.adapterField);
  return {
    fingerprint: buildFingerprint(result.adapter, result.corpusSize, result.mode, ua),
    adapter: {
      vendor: field(result.adapter?.vendor),
      device: field(result.adapter?.device),
      architecture: field(result.adapter?.architecture),
      renderer: field(result.adapter?.renderer),
      ...(result.adapter?.adapterType ? { adapterType: clampStr(result.adapter.adapterType, LIMITS.adapterType) } : {})
    },
    gpuTier: {
      tier: result.gpuTier.tier,
      confidence: result.gpuTier.confidence,
      reason: clampStr(result.gpuTier.reason, LIMITS.tierReason)
    },
    browser: clampStr(browserFamily(ua), LIMITS.family),
    os: clampStr(osFamily(ua), LIMITS.family),
    corpusSize: result.corpusSize,
    mode: result.mode,
    query: clampStr(result.query, LIMITS.query),
    gpuMedianMs: clampMs(result.gpuMedianMs),
    gpuP95Ms: clampMs(result.gpuP95Ms),
    gpuMatches: clampCount(result.gpuMatches),
    cpuMedianMs: clampMs(result.cpuMedianMs),
    cpuP95Ms: clampMs(result.cpuP95Ms),
    cpuMatches: clampCount(result.cpuMatches),
    externals: result.externals.slice(0, LIMITS.externals).map(external => ({
      name: clampStr(external.name, LIMITS.externalName),
      medianMs: clampMs(external.medianMs),
      p95Ms: clampMs(external.p95Ms),
      matches: clampCount(external.matches),
      ran: external.ran
    })),
    createdAt: new Date().toISOString()
  };
}

export function alreadyShared(fingerprint: string): boolean {
  try {
    return localStorage.getItem(`${SHARED_PREFIX}${fingerprint}`) === '1';
  } catch {
    return false;
  }
}

export function markShared(fingerprint: string): void {
  try {
    localStorage.setItem(`${SHARED_PREFIX}${fingerprint}`, '1');
  } catch {
    // private-mode storage: sharing still works, just not remembered
  }
}

export async function submitSharedResult(
  payload: SharePayload
): Promise<{ ok: boolean; duplicate?: boolean; unconfigured?: boolean; terminal?: boolean; error?: string }> {
  try {
    const res = await fetch('/api/submit-benchmark', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (res.status === 503) return { ok: false, unconfigured: true };
    const data = (await res.json().catch(() => ({}))) as { duplicate?: boolean; error?: string };
    if (res.status === 409 || data.duplicate) return { ok: true, duplicate: true };
    // 400/413 etc. = this payload will never be accepted: terminal, don't retry.
    // 429 (rate limit) and 5xx are transient and may be retried next run.
    const terminal = res.status >= 400 && res.status < 500 && res.status !== 429;
    if (!res.ok) return { ok: false, terminal, error: data.error ?? `HTTP ${res.status}` };
    return { ok: true };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error) };
  }
}
