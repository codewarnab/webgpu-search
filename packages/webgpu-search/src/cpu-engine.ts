import uFuzzy from '@leeoniya/ufuzzy';
import type { SearchResultItem } from './types';
import { nowMs } from './guard';
import { WebGPUSearchError } from './errors';
import { normalizeText } from './text-normalization';
import { compareExactResults, scoreSubstringTokens } from './exact-scorer';

export interface CPUSearchResult {
  query: string;
  totalMatches: number;
  results: SearchResultItem[];
  durationMs: number;
}

export class CPUEngine {
  private ufuzzyInstance: any;

  constructor() {
    this.ufuzzyInstance = new (uFuzzy as any)({
      intraMode: 1,
      intraIns: 1
    });
  }

  /**
   * uFuzzy filter + search with normalized descending scores
   */
  searchWithUFuzzy(
    strings: string[],
    query: string,
    maxResults: number = 1000,
    caseSensitive: boolean = false
  ): CPUSearchResult {
    const cleanQuery = query.trim();
    if (!cleanQuery) {
      return { query: '', totalMatches: 0, results: [], durationMs: 0 };
    }

    const t0 = nowMs();
    let idxs: any = null;
    let info: any = null;
    let order: any = null;

    try {
      [idxs, info, order] = this.ufuzzyInstance.search(strings, cleanQuery);
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      const wrapped = new WebGPUSearchError(`[webgpu-search] uFuzzy search failed: ${detail}`);
      (wrapped as unknown as { cause?: unknown }).cause = err;
      throw wrapped;
    }

    const durationMs = nowMs() - t0;
    const results: SearchResultItem[] = [];

    const indices: number[] = (order && info && order.length > 0)
      ? order.map((o: number) => info.idx[o])
      : (idxs ?? []);

    // NOTE: legacy uFuzzy path is explicitly non-conforming
    // (locale-sensitive lowercasing lives here on purpose, rank scores are
    // fabricated). Excluded from the differential matrix; use cpuScorer:'exact'.
    let skippedCaseSensitive = 0;
    for (let i = 0; i < indices.length; i++) {
      const itemIdx = indices[i];
      const text = strings[itemIdx] ?? '';
      if (caseSensitive && !text.includes(cleanQuery)) {
        skippedCaseSensitive++;
        continue;
      }
      if (results.length < maxResults) {
        const score = Math.max(1, 1000 - results.length * 2);
        results.push({
          index: itemIdx,
          score,
          text
        });
      }
    }

    return {
      query,
      totalMatches: (idxs ? idxs.length : 0) - skippedCaseSensitive,
      results,
      durationMs
    };
  }

  /**
   * Native JS substring scan with GPU scoring symmetry. Records and query go
   * through the same `normalizeText` path as the GPU/exact scorer (NFC +
   * default case folding when case-insensitive, NFC-only when
   * case-sensitive) and are scored in code points via
   * `scoreSubstringTokens`, so scores match the WGSL kernel bit-for-bit
   * (no UTF-16 unit offsets, no locale-sensitive `toLowerCase`).
   */
  searchNaiveScan(
    strings: string[],
    query: string,
    maxResults: number = 1000,
    caseSensitive: boolean = false
  ): CPUSearchResult {
    const normalized = !caseSensitive;
    const nq = normalizeText(typeof query === 'string' ? query : '', normalized);
    if (nq.isEmpty) {
      return { query: '', totalMatches: 0, results: [], durationMs: 0 };
    }

    const t0 = nowMs();
    const candidates: SearchResultItem[] = [];
    let totalMatches = 0;

    for (let i = 0; i < strings.length; i++) {
      const s = strings[i] ?? '';
      const rec = normalizeText(typeof s === 'string' ? s : String(s), normalized).tokens;
      const r = scoreSubstringTokens(rec, nq.tokens);
      if (r.matched) {
        totalMatches++;
        candidates.push({
          index: i,
          score: r.score,
          text: s
        });
      }
    }

    candidates.sort(compareExactResults);
    const results = candidates.slice(0, maxResults);
    const durationMs = nowMs() - t0;

    return {
      query,
      totalMatches,
      results,
      durationMs
    };
  }
}
