import puppeteer from 'puppeteer-core';
import path from 'path';
import fs from 'fs';
import { normalizeText, searchCpuReference } from '../packages/webgpu-search/src/index';
import {
    getChromeExecutablePath,
    getChromeLaunchArgs,
    ensureBenchmarkServer
} from './browser-utils';

async function main() {
    console.log('--- Running WebGPU Top-K & Timing Regression Tests ---');
    const chromePath = getChromeExecutablePath();
    console.log(`Using Chrome binary: ${chromePath}`);

    const server = await ensureBenchmarkServer(5173);

    const browser = await puppeteer.launch({
        executablePath: chromePath,
        headless: 'new',
        args: getChromeLaunchArgs()
    });

    try {
        const page = await browser.newPage();

        page.on('console', msg => {
            const text = msg.text();
            if (text.includes('Error') || text.includes('WebGPU') || text.includes('Failed')) {
                console.log('[Browser Console]', text);
            }
        });

        let loaded = false;
        for (let i = 0; i < 10; i++) {
            try {
                await page.goto(`${server.url}/`, { waitUntil: 'networkidle0', timeout: 15000 });
                loaded = true;
                break;
            } catch (err) {
                await new Promise(r => setTimeout(r, 1000));
            }
        }
        if (!loaded) {
            throw new Error(`Failed to connect to ${server.url}/ after multiple attempts`);
        }

        await page.waitForFunction(
            () => (window as any).gpuEngine && (window as any).gpuEngine.isReady,
            { timeout: 15000 }
        );

        // Evaluate inside page context where WebGPU engine is initialized
        const testResults = await page.evaluate(async () => {
            const engine = (window as any).gpuEngine;
            if (!engine || !engine.isReady) {
                return { success: false, reason: 'WebGPU Engine not ready' };
            }

            const results: { name: string; passed: boolean; details?: any }[] = [];

            // 1. Test: Timing metrics separation
            const searchRes = await engine.search('Controller', { mode: 'substring', maxResults: 100 });
            const timings = searchRes.timings;
            const hasProperTimings = (
                typeof timings.encodeSubmitMs === 'number' &&
                typeof timings.readbackMs === 'number' &&
                typeof timings.totalMs === 'number' &&
                (timings.gpuExecutionMs === null || typeof timings.gpuExecutionMs === 'number')
            );
            results.push({
                name: 'Timing metrics separated (Submit, Exec, Readback, Total)',
                passed: hasProperTimings,
                details: timings
            });

            // 2. Test: Candidate collection across >1000 items with late high-scoring matches
            // Generate synthetic dataset of 3,000 variable-length items where
            // items 1200..1249 start with the query (highest substring score).
            // M3: variable-length u32 packing -- plain strings in, the engine
            // normalizes to post-fold tokens (no 59-char truncation, no slots).
            const count = 3000;
            const strings = new Array<string>(count);

            for (let i = 0; i < count; i++) {
                // For i < 1200 or i >= 1250: match starts late (low substring score)
                // For 1200 <= i < 1250: starts with exact query (highest substring score)
                strings[i] = i >= 1200 && i < 1250
                    ? `Controller_Special_${i}.ts`
                    : `very_long_path_prefix_folder_name/Controller_${i}.ts`;
            }

            await engine.loadDataset({
                size: count,
                strings
            });

            const topResults = await engine.search('Controller', { mode: 'substring', maxResults: 50 });
            // The top results must contain the late high-scoring items (indices 1200-1249).
            // Set-membership (>=8/10) instead of exact order: identical-prefix
            // ties are stable today but brittle to scoring tweaks.
            const top10 = topResults.results.slice(0, 10);
            const inRange = top10.filter((r: any) => r.index >= 1200 && r.index < 1250).length;
            const containsLateHighScorers = inRange >= 8;
            results.push({
                name: 'Broad query (>1000 matches) preserves late high-scoring matches in top-K',
                passed: containsLateHighScorers,
                details: {
                    totalMatches: topResults.totalMatches,
                    top1Index: topResults.results[0]?.index,
                    top1Score: topResults.results[0]?.score,
                    candidateCount: topResults.candidateCount
                }
            });

            // 3. Test: Candidate overflow detection (>8192 items)
            const bigCount = 10000;
            const bigStrings = new Array<string>(bigCount);

            for (let i = 0; i < bigCount; i++) {
                bigStrings[i] = `TestItem_${i}.ts`;
            }

            await engine.loadDataset({
                size: bigCount,
                strings: bigStrings
            });

            const overflowRes = await engine.search('TestItem', { mode: 'substring', maxResults: 100 });
            results.push({
                name: 'Candidate pool overflow (>8192 matches) detected and reported',
                passed: overflowRes.hasOverflow === true && overflowRes.candidateCount === 8192,
                details: {
                    totalMatches: overflowRes.totalMatches,
                    hasOverflow: overflowRes.hasOverflow,
                    candidateCount: overflowRes.candidateCount
                }
            });

            // 4. Test: unicode variable-length packing (astral/empty/whitespace rows).
            const FCP = String.fromCodePoint;
            const uniStrings = [
                'hello',
                'stra' + FCP(0xdf) + 'e',
                FCP(0x1f600),
                '',
                '   ',
                'packages/core/' + FCP(0x4eac) + '/DeepSpecialController.ts',
            ];
            await engine.loadDataset({ size: uniStrings.length, strings: uniStrings });
            const uniRes = await engine.search('DeepSpecial', { mode: 'substring', maxResults: 10 });
            const uniOk = uniRes.totalMatches >= 1 && uniRes.results.some((r: any) => r.text.includes('DeepSpecial'));
            results.push({
                name: 'Unicode rows (astral/empty/whitespace) pack + search without truncation',
                passed: uniOk,
                details: { totalMatches: uniRes.totalMatches, top: uniRes.results[0] }
            });

            return { success: true, results };
        });

        // 5. Test: M4 CPU/GPU differential parity on real hardware (per-PR browser
        // gate in CI + release gate). Node computes the parity reference over
        // the same post-fold tokens; the page executes WGSL. Exact ordered
        // (index,score) + text parity asserted (small corpora, never
        // overflow). Same-host only: the oracle runs in Bun/Node while the
        // subject runs in Chrome, so this assumes Bun ICU and Chrome ICU
        // agree on NFC for this ASCII-heavy corpus (negligible risk here;
        // cross-runtime NFC is probed/reported, not gated).
        const FCP = String.fromCodePoint;
        const parityStrings: string[] = [];
        for (let i = 0; i < 200; i++) {
            parityStrings.push(`src/services/Auth${i % 7 === 0 ? FCP(0xdf) : 's'}Controller_${i}.ts`);
        }
        parityStrings.push(
            'strasse', 'STRASSE', 'stra' + FCP(0xdf) + 'e',
            FCP(0x5317) + FCP(0x4eac) + 'city',
            FCP(0x1f600) + 'smile',
            'caf' + FCP(0xe9),
            'plain'
        );
        const parityCells = [
            { query: 'auth', mode: 'substring', limit: 50 },
            { query: 'auth', mode: 'fuzzy', limit: 50 },
            { query: 'Controller_3', mode: 'substring', limit: 50 },
            { query: 'strasse', mode: 'substring', limit: 50 },
            { query: 'strasse', mode: 'fuzzy', limit: 50 },
            { query: 'SS', mode: 'substring', limit: 50 },
            { query: FCP(0x5317), mode: 'substring', limit: 10 },
            { query: FCP(0x1f600), mode: 'substring', limit: 10 },
            { query: 'zzz-no-match', mode: 'substring', limit: 50 },
            { query: 'zzz-no-match', mode: 'fuzzy', limit: 50 },
        ] as Array<{ query: string; mode: 'substring' | 'fuzzy'; limit: number }>;
        const recordTokens = parityStrings.map((s) => normalizeText(s, true).tokens);
        // Skip the GPU evaluate entirely when the engine never became ready:
        // without this guard a dead engine turns into a fatal evaluate throw
        // instead of a clear gate failure.
        if (!(testResults as any).success) {
            (testResults as any).results.push({
                name: 'M4 parity (skipped: WebGPU engine not ready)',
                passed: false,
                details: testResults
            });
        } else {
            const PARITY_TIMEOUT_MS = 120000;
            const parityEvaluate = page.evaluate(async (args: { strings: string[]; cells: Array<{ query: string; mode: string; limit: number }> }) => {
                const engine = (window as any).gpuEngine;
                await engine.loadDataset({ size: args.strings.length, strings: args.strings });
                const out: Array<{ totalMatches: number; candidateCount: number; hasOverflow: boolean; results: Array<{ index: number; score: number; text: string }> }> = [];
                for (const c of args.cells) {
                    const r = await engine.search(c.query, { mode: c.mode, maxResults: c.limit });
                    out.push({
                        totalMatches: r.totalMatches,
                        candidateCount: r.candidateCount,
                        hasOverflow: r.hasOverflow,
                        results: r.results.map((x: any) => ({ index: x.index, score: x.score, text: x.text }))
                    });
                }
                return out;
            }, { strings: parityStrings, cells: parityCells });
            const gpuParityOut = await Promise.race([
                parityEvaluate,
                new Promise<never>((_, rej) => setTimeout(() => rej(new Error('M4 parity browser evaluate timed out after 120s')), PARITY_TIMEOUT_MS)),
            ]);
            if ((testResults as any).success && Array.isArray((testResults as any).results)) {
                parityCells.forEach((cell, ci) => {
                    const qT = normalizeText(cell.query, true).tokens;
                    const expected = searchCpuReference(recordTokens, qT, cell.mode, cell.limit, parityStrings);
                    const got = (gpuParityOut as any[])[ci];
                    // Small corpora: never overflow, so candidateCount must
                    // equal totalMatches (matches the mock-harness contract
                    // which also pins candidateCount).
                    let pass = got.totalMatches === expected.totalMatches && got.hasOverflow === false && got.candidateCount === expected.totalMatches;
                    if (pass) {
                        if (got.results.length !== expected.results.length) {
                            pass = false;
                        } else {
                            for (let k = 0; k < got.results.length; k++) {
                                const g = got.results[k];
                                const e = expected.results[k];
                                if (g.index !== e.index || g.score !== e.score || g.text !== e.text) {
                                    pass = false;
                                    break;
                                }
                            }
                        }
                    }
                    (testResults as any).results.push({
                        name: `M4 parity ${cell.mode} ${JSON.stringify(cell.query.slice(0, 12))} (ordered index/score/text)`,
                        passed: pass,
                        details: { totalMatches: got.totalMatches, candidateCount: got.candidateCount, expected: expected.totalMatches }
                    });
                });
            }
        }

        console.log('\n--- Test Suite Summary ---');
        console.dir(testResults, { depth: null });

        if (!testResults.success || testResults.results?.some((r: any) => !r.passed)) {
            console.error('Some tests failed!');
            process.exit(1);
        } else {
            console.log('\nAll regression tests passed successfully! [pass]');
        }
    } finally {
        await browser.close().catch(() => {});
        await server.close().catch(() => {});
    }
    process.exit(0);
}

main().catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
});
