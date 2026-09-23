import type { AdapterInfo } from './types';

export type GpuTier = 'discrete' | 'integrated' | 'software' | 'unknown';
export type GpuTierConfidence = 'high' | 'medium' | 'low';

export interface GpuClassification {
  tier: GpuTier;
  confidence: GpuTierConfidence;
  reason: string;
}

const SOFTWARE_PATTERNS = [
  'swiftshader',
  'llvmpipe',
  'lavapipe',
  'softpipe',
  'basic render driver',
  'warp',
  'mock'
];

function combinedDescriptor(info: AdapterInfo): string {
  return `${info.vendor} ${info.device} ${info.architecture} ${info.renderer ?? ''}`.toLowerCase();
}

function hasSoftwareMarker(desc: string): boolean {
  if (SOFTWARE_PATTERNS.some((p) => desc.includes(p))) return true;
  if (/\bsoftware\b/.test(desc)) return true;
  if (/\bcpu (rasterizer|renderer|fallback)\b/.test(desc)) return true;
  return false;
}

/**
 * Classify the WebGPU adapter into discrete / integrated / software / unknown.
 *
 * Portable: pure string heuristics over `AdapterInfo`, no DOM access, safe in
 * workers/Node/SSR. Prefers the native `adapterType` passthrough
 * (`GPUAdapterInfo.type`: "discrete GPU" | "integrated GPU" | "CPU" |
 * "unknown") when present, otherwise falls back to vendor/device keywords.
 * WebGPU cannot enumerate adapters, so "integrated" never proves a discrete
 * GPU exists — callers must word flags as "reflects the iGPU" + fix hints.
 */
export function classifyGpuTier(info: AdapterInfo | null): GpuClassification {
  if (!info) {
    return { tier: 'unknown', confidence: 'low', reason: 'no adapter info' };
  }
  const desc = combinedDescriptor(info);
  if (hasSoftwareMarker(desc)) {
    return { tier: 'software', confidence: 'high', reason: 'software renderer marker' };
  }

  const rawType = (info.adapterType ?? '').toLowerCase().trim();
  if (rawType.includes('discrete')) {
    return { tier: 'discrete', confidence: 'high', reason: 'adapter info type' };
  }
  if (rawType.includes('integrated')) {
    return { tier: 'integrated', confidence: 'high', reason: 'adapter info type' };
  }
  if (rawType === 'cpu' || rawType.includes('cpu')) {
    return { tier: 'software', confidence: 'high', reason: 'adapter info type is CPU' };
  }

  // Discrete markers (high confidence).
  if (
    /geforce\s?(rtx|gtx|mx)|quadro|radeon\s?(rx|pro)|intel\s?arc\s?a|\barc\s?a\d/i.test(desc)
  ) {
    // NVIDIA MX parts are entry-level discrete silicon, still discrete.
    return { tier: 'discrete', confidence: 'high', reason: 'discrete product keyword' };
  }
  if (/nvidia/i.test(desc) && !/tegra|jetson|orin/i.test(desc)) {
    return { tier: 'discrete', confidence: 'medium', reason: 'nvidia vendor' };
  }

  // Integrated markers (high confidence).
  if (
    /intel.*(uhd|hd graphics|iris|xe graphics)|iris\s?(xe|plus)|uhd graphics/i.test(desc) ||
    /amd.*(radeon graphics|vega\s?[368]|ryzen.*vega)|radeon\s?vega/i.test(desc) ||
    /adreno|mali|apple\s?m[1-4]|apple\s?gpu/i.test(desc) ||
    /tegra|jetson|orin/i.test(desc)
  ) {
    return { tier: 'integrated', confidence: 'high', reason: 'integrated product keyword' };
  }
  if (/^intel\b|\bintel\b/i.test(desc)) {
    return { tier: 'integrated', confidence: 'medium', reason: 'intel vendor fallback' };
  }

  return { tier: 'unknown', confidence: 'low', reason: 'no tier keyword matched' };
}
