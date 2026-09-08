import uFuzzy from '@leeoniya/ufuzzy';

export interface CPUSearchResult {
    query: string;
    totalMatches: number;
    results: Array<{ index: number; text: string }>;
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
     * uFuzzy filter + search
     */
    searchUFuzzy(strings: string[], query: string, maxResults: number = 1000): CPUSearchResult {
        const cleanQuery = query.trim();
        if (!cleanQuery) {
            return { query: '', totalMatches: 0, results: [], durationMs: 0 };
        }

        const t0 = performance.now();
        let idxs: any = null;
        let info: any = null;
        let order: any = null;

        try {
            [idxs, info, order] = this.ufuzzyInstance.search(strings, cleanQuery);
        } catch (err) {
            console.warn('uFuzzy search error:', err);
        }

        const durationMs = performance.now() - t0;
        const totalMatches = idxs ? idxs.length : 0;
        const results: Array<{ index: number; text: string }> = [];

        if (order && info && order.length > 0) {
            const count = Math.min(order.length, maxResults);
            for (let i = 0; i < count; i++) {
                const itemIdx = info.idx[order[i]];
                results.push({
                    index: itemIdx,
                    text: strings[itemIdx]
                });
            }
        } else if (idxs && idxs.length > 0) {
            const count = Math.min(idxs.length, maxResults);
            for (let i = 0; i < count; i++) {
                results.push({
                    index: idxs[i],
                    text: strings[idxs[i]]
                });
            }
        }

        return {
            query,
            totalMatches,
            results,
            durationMs
        };
    }

    /**
     * Native JS substring search (case-insensitive includes)
     */
    searchNative(strings: string[], query: string, maxResults: number = 1000): CPUSearchResult {
        const cleanQuery = query.trim();
        if (!cleanQuery) {
            return { query: '', totalMatches: 0, results: [], durationMs: 0 };
        }

        const t0 = performance.now();
        const lowerQuery = cleanQuery.toLowerCase();
        const results: Array<{ index: number; text: string }> = [];
        let totalMatches = 0;

        for (let i = 0; i < strings.length; i++) {
            if (strings[i].toLowerCase().includes(lowerQuery)) {
                totalMatches++;
                if (results.length < maxResults) {
                    results.push({
                        index: i,
                        text: strings[i]
                    });
                }
            }
        }

        const durationMs = performance.now() - t0;

        return {
            query,
            totalMatches,
            results,
            durationMs
        };
    }
}
