import uFuzzy from '@leeoniya/ufuzzy';
import type { SearchResultItem } from './types';
import { nowMs } from './guard';

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
      console.warn('uFuzzy search error:', err);
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
   * Native JS substring search with ranking symmetry and case-sensitivity support
   */
  searchNaiveScan(
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
    const queryTerm = caseSensitive ? cleanQuery : cleanQuery.toLowerCase();
    const candidates: SearchResultItem[] = [];
    let totalMatches = 0;

    for (let i = 0; i < strings.length; i++) {
      const s = strings[i] ?? '';
      const target = caseSensitive ? s : s.toLowerCase();
      const matchStart = target.indexOf(queryTerm);
      if (matchStart !== -1) {
        totalMatches++;
        // Formula matching GPU substring scoring: earlier start + shorter string
        const score = 1000 - (matchStart * 10) - (s.length - cleanQuery.length);
        candidates.push({
          index: i,
          score,
          text: s
        });
      }
    }

    candidates.sort((a, b) => b.score - a.score);
    const results = candidates.slice(0, maxResults);
    const durationMs = nowMs() - t0;

    return {
      query,
      totalMatches,
      results,
      durationMs
    };
  }

  /** @deprecated Use searchWithUFuzzy (explicit vendor name). */
  searchUFuzzy(
    strings: string[],
    query: string,
    maxResults: number = 1000,
    caseSensitive: boolean = false
  ): CPUSearchResult {
    return this.searchWithUFuzzy(strings, query, maxResults, caseSensitive);
  }

  /** @deprecated Use searchNaiveScan. */
  searchNative(
    strings: string[],
    query: string,
    maxResults: number = 1000,
    caseSensitive: boolean = false
  ): CPUSearchResult {
    return this.searchNaiveScan(strings, query, maxResults, caseSensitive);
  }
}
