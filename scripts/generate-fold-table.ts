/**
 * Fold-table generator for Issue #7 M2.
 *
 * Fetches the pinned Unicode CaseFolding file, keeps C+F only (S+T excluded),
 * and emits a sparse range/delta encoded eager table at
 * packages/webgpu-search/src/fold-table.ts.
 *
 * Pinned source:
 *   URL: https://www.unicode.org/Public/16.0.0/ucd/CaseFolding.txt
 *   File: CaseFolding-16.0.0.txt (Unicode 16.0.0, Sept 2024)
 *   Date header: 2024-04-30, 21:48:11 GMT
 *   License: © 2024 Unicode®, Inc. — https://www.unicode.org/terms_of_use.html
 *   (UCD terms; data-file excerpts permitted with copyright notice.)
 *
 * Expected counts (contract §3, frozen):
 *   C+F total = 1,557 (C=1,453 single→single, F=104 multi-char)
 *   F breakdown = 88 × 1→2 + 16 × 1→3
 *
 * Usage:
 *   bun scripts/generate-fold-table.ts
 *   bun scripts/generate-fold-table.ts --check   # verify committed table matches source
 *   bun scripts/generate-fold-table.ts --input /path/to/CaseFolding.txt
 *
 * Portable: Node/Bun only, no DOM refs.
 */

const PINNED_URL = 'https://www.unicode.org/Public/16.0.0/ucd/CaseFolding.txt';
const PINNED_FILE = 'CaseFolding-16.0.0.txt';
const PINNED_DATE = '2024-04-30, 21:48:11 GMT';
const PINNED_COPYRIGHT = '© 2024 Unicode®, Inc.';
const PINNED_TERMS = 'https://www.unicode.org/terms_of_use.html';
const UNICODE_VERSION = '16.0.0';

const OUT_PATH = new URL('../packages/webgpu-search/src/fold-table.ts', import.meta.url);

interface FoldEntry {
  cp: number;
  status: string;
  mapping: number[];
}

function parseCaseFolding(text: string): FoldEntry[] {
  const entries: FoldEntry[] = [];
  for (const rawLine of text.split('\n')) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;
    const parts = line.split(';');
    if (parts.length < 3) continue;
    const cp = parseInt(parts[0].trim(), 16);
    const status = parts[1].trim();
    const mappingStr = parts[2].split('#')[0].trim();
    if (!mappingStr) continue;
    const mapping = mappingStr.split(/\s+/).map((h) => parseInt(h, 16));
    if (Number.isNaN(cp) || mapping.some((m) => Number.isNaN(m))) continue;
    entries.push({ cp, status, mapping });
  }
  return entries;
}

function buildRanges(sortedC: Array<[number, number]>): Array<[number, number, number]> {
  // sortedC: [cp, mapped][] sorted by cp. Group consecutive cp with constant delta.
  const ranges: Array<[number, number, number]> = [];
  if (sortedC.length === 0) return ranges;
  let rangeStart = sortedC[0][0];
  let prevCp = sortedC[0][0];
  let curDelta = sortedC[0][1] - sortedC[0][0];
  for (let i = 1; i < sortedC.length; i++) {
    const cp = sortedC[i][0];
    const mp = sortedC[i][1];
    const delta = mp - cp;
    if (cp === prevCp + 1 && delta === curDelta) {
      prevCp = cp;
    } else {
      ranges.push([rangeStart, prevCp, curDelta]);
      rangeStart = cp;
      prevCp = cp;
      curDelta = delta;
    }
  }
  ranges.push([rangeStart, prevCp, curDelta]);
  return ranges;
}

function emitTable(ranges: Array<[number, number, number]>, fEntries: FoldEntry[]): string {
  const flatRanges = ranges.map(([s, e, d]) => `${s},${e},${d}`).join(',');
  // F expansions flat: [cp, len, m1, m2(, m3), ...]
  const sortedF = [...fEntries].sort((a, b) => a.cp - b.cp);
  const flatF: number[] = [];
  for (const e of sortedF) {
    flatF.push(e.cp, e.mapping.length, ...e.mapping);
  }
  const flatFStr = flatF.join(',');

  const header = `/**
 * Generated Unicode default case-fold table (C+F only) — DO NOT EDIT BY HAND.
 *
 * Source: ${PINNED_URL}
 * File: ${PINNED_FILE} (Unicode ${UNICODE_VERSION}, Sept 2024)
 * Date: ${PINNED_DATE}
 * License: ${PINNED_COPYRIGHT} — For terms of use see ${PINNED_TERMS}
 *   (Unicode Character Database terms; excerpts permitted with copyright notice.)
 *
 * Generator: scripts/generate-fold-table.ts
 * Status filter: C (common) + F (full) only; S (simple) + T (turkic) EXCLUDED.
 * Counts: C=${ranges.reduce((n, [s, e]) => n + (e - s + 1), 0)} singles in ${
    ranges.length
  } ranges; F=${sortedF.length} expansions (${
    sortedF.filter((e) => e.mapping.length === 2).length
  }×1→2 + ${sortedF.filter((e) => e.mapping.length === 3).length}×1→3).
 *
 * Encoding: sparse range/delta — FOLD_RANGES is flat [start,end,delta,…] where
 *   folded(cp) = cp + delta for start ≤ cp ≤ end. FOLD_EXPANSIONS is flat
 *   [cp,len,m1,m2(,m3),…] for 1→2/3 full mappings. Decoded eagerly into Maps
 *   at module load (no lazy import; delta budgeted ≤8 KB gzip over baseline).
 * Portable: no DOM refs. No charCodeAt/toLowerCase/toUpperCase/indexOf here.
 */
`;

  return `${header}
export const FOLD_UNICODE_VERSION = '${UNICODE_VERSION}' as const;

/** Flat [start,end,delta,…] for C (single→single) mappings. */
export const FOLD_RANGES: readonly number[] = [${flatRanges}];

/** Flat [cp,len,m1,m2(,m3),…] for F (1→2/3) mappings. */
export const FOLD_EXPANSIONS: readonly number[] = [${flatFStr}];

const singleMap: Map<number, readonly number[]> = new Map();
for (let i = 0; i < FOLD_RANGES.length; i += 3) {
  const start: number = FOLD_RANGES[i] as number;
  const end: number = FOLD_RANGES[i + 1] as number;
  const delta: number = FOLD_RANGES[i + 2] as number;
  for (let cp = start; cp <= end; cp++) {
    singleMap.set(cp, Object.freeze([cp + delta]) as readonly number[]);
  }
}

const expansionMap: Map<number, readonly number[]> = new Map();
for (let i = 0; i < FOLD_EXPANSIONS.length; ) {
  const cp: number = FOLD_EXPANSIONS[i] as number;
  const len: number = FOLD_EXPANSIONS[i + 1] as number;
  expansionMap.set(
    cp,
    Object.freeze(FOLD_EXPANSIONS.slice(i + 2, i + 2 + len)) as readonly number[],
  );
  i += 2 + len;
}

Object.freeze(FOLD_RANGES);
Object.freeze(FOLD_EXPANSIONS);

/**
 * Full default case fold for one scalar (C+F only, S+T excluded).
 * Returns the shared frozen 1–3 scalar array, or null if cp maps to itself.
 * Eager Maps built at module load; pure function of cp. Returned arrays are
 * frozen — callers must not mutate them (zero per-fold allocation).
 */
export function foldCodePoint(cp: number): readonly number[] | null {
  const multi = expansionMap.get(cp);
  if (multi !== undefined) return multi;
  const single = singleMap.get(cp);
  if (single !== undefined) return single;
  return null;
}

/** Number of C singles encoded (for tests). */
export const FOLD_C_COUNT: number = singleMap.size;
/** Number of F expansions encoded (for tests). */
export const FOLD_F_COUNT: number = expansionMap.size;
`;
}

async function loadSource(inputPath?: string): Promise<{ text: string; sha256: string }> {
  const { readFile } = await import('node:fs/promises');
  const { createHash } = await import('node:crypto');
  if (inputPath) {
    const text = await readFile(inputPath, 'utf8');
    const sha256 = createHash('sha256').update(text, 'utf8').digest('hex');
    return { text, sha256 };
  }
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(new Error('fetch timeout 30s')), 30_000);
  try {
    const res = await fetch(PINNED_URL, { signal: ctrl.signal });
    if (!res.ok) throw new Error(`fetch ${PINNED_URL} failed: ${res.status}`);
    const text = await res.text();
    const sha256 = createHash('sha256').update(text, 'utf8').digest('hex');
    return { text, sha256 };
  } finally {
    clearTimeout(timer);
  }
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const checkOnly = args.includes('--check');
  const inputIdx = args.indexOf('--input');
  if (inputIdx >= 0 && (args[inputIdx + 1] === undefined || (args[inputIdx + 1] as string).startsWith('--'))) {
    console.error('FAIL --input requires a file path (got flag or nothing). Refusing network fallback.');
    process.exit(1);
  }
  const inputPath = inputIdx >= 0 ? args[inputIdx + 1] : undefined;
  const shaIdx = args.indexOf('--expected-sha256');
  const expectedSha = shaIdx >= 0 ? args[shaIdx + 1] : undefined;

  const { text, sha256 } = await loadSource(inputPath);
  console.log(`CaseFolding sha256: ${sha256}`);
  if (expectedSha !== undefined && sha256 !== expectedSha) {
    throw new Error(`sha256 mismatch: expected ${expectedSha}, got ${sha256}`);
  }
  const entries = parseCaseFolding(text);
  const cEntries = entries.filter((e) => e.status === 'C');
  const fEntries = entries.filter((e) => e.status === 'F');

  const f2 = fEntries.filter((e) => e.mapping.length === 2).length;
  const f3 = fEntries.filter((e) => e.mapping.length === 3).length;
  const cBad = cEntries.filter((e) => e.mapping.length !== 1).length;
  const fBad = fEntries.filter((e) => e.mapping.length < 2 || e.mapping.length > 3).length;

  console.log(`Parsed: C=${cEntries.length} F=${fEntries.length} (F 1→2=${f2} 1→3=${f3})`);
  if (cEntries.length + fEntries.length !== 1557) {
    throw new Error(`C+F count mismatch: expected 1557, got ${cEntries.length + fEntries.length}`);
  }
  if (fEntries.length !== 104 || f2 !== 88 || f3 !== 16) {
    throw new Error(`F breakdown mismatch: expected 104 (88×1→2 + 16×1→3), got ${fEntries.length} (${f2}×1→2 + ${f3}×1→3)`);
  }
  if (cBad !== 0 || fBad !== 0) {
    throw new Error(`Unexpected mapping arities: C bad=${cBad} F bad=${fBad}`);
  }

  const sortedC: Array<[number, number]> = cEntries
    .map((e) => [e.cp, e.mapping[0]] as [number, number])
    .sort((a, b) => a[0] - b[0]);
  const ranges = buildRanges(sortedC);
  console.log(`Ranges: ${ranges.length} (singles=${ranges.filter(([s, e]) => s === e).length})`);

  // Spot-check contracted fixtures.
  const lookup = new Map<number, number[]>();
  for (const [s, e, d] of ranges) for (let cp = s; cp <= e; cp++) lookup.set(cp, [cp + d]);
  for (const e of fEntries) lookup.set(e.cp, e.mapping);
  const must: Array<[number, number[]]> = [
    [0x0041, [0x0061]], // A→a (C)
    [0x00df, [0x0073, 0x0073]], // ß→ss (F)
    [0x0130, [0x0069, 0x0307]], // İ→i+◌̇ (F)
    [0x03a3, [0x03c3]], // Σ→σ (C)
    [0xfb00, [0x0066, 0x0066]], // ﬀ→ff (F)
  ];
  for (const [cp, want] of must) {
    const got = lookup.get(cp);
    const ok = got !== undefined && got.length === want.length && got.every((v, i) => v === want[i]);
    if (!ok) throw new Error(`Fixture U+${cp.toString(16).toUpperCase()} expected [${want}] got [${got}]`);
  }
  // T-only must be absent: Turkic mappings excluded (0049 T→0131, 0130 T→0069).
  // I (U+0049) keeps C→i, never T→ı; İ (U+0130) keeps F→i+◌̇, never T→i.
  // NOTE: contract text labels "ϴ/θ (T-only)" is inaccurate for 16.0.0:
  // U+03F4 has C→03B8 in CaseFolding-16.0.0.txt, so C+F folds it (verified).
  const iFold = lookup.get(0x0049);
  if (iFold === undefined || iFold.length !== 1 || iFold[0] !== 0x0069) {
    throw new Error(`U+0049 must fold to U+0069 via C (got [${iFold}])`);
  }
  const dotFold = lookup.get(0x0130);
  if (dotFold === undefined || dotFold.length !== 2 || dotFold[0] !== 0x0069 || dotFold[1] !== 0x0307) {
    throw new Error(`U+0130 must fold to U+0069 U+0307 via F, not bare U+0069 via T (got [${dotFold}])`);
  }
  // S mapping excluded: U+1E9E keeps F→ss, never S→ß.
  const sharpFold = lookup.get(0x1e9e);
  if (
    sharpFold === undefined ||
    sharpFold.length !== 2 ||
    sharpFold[0] !== 0x0073 ||
    sharpFold[1] !== 0x0073
  ) {
    throw new Error(`U+1E9E must fold to U+0073 U+0073 via F, not U+00DF via S (got [${sharpFold}])`);
  }
  // U+03F4 has C→03B8 in 16.0.0, so it folds (contract "T-only" label is stale).
  const thetaFold = lookup.get(0x03f4);
  if (thetaFold === undefined || thetaFold.length !== 1 || thetaFold[0] !== 0x03b8) {
    throw new Error(`U+03F4 must fold to U+03B8 via C in 16.0.0 (got [${thetaFold}])`);
  }
  // S-only spot: U+0131 (ı) has no C/F? It has no fold entry → identity. Ensure absent.
  if (lookup.has(0x0131)) throw new Error('U+0131 (ı) must have no C/F mapping (identity)');

  const out = emitTable(ranges, fEntries);
  const { readFile, writeFile, rename } = await import('node:fs/promises');
  if (checkOnly) {
    let current = '';
    try {
      current = await readFile(OUT_PATH, 'utf8');
    } catch {
      current = '';
    }
    const norm = (s: string): string => s.replace(/\r\n/g, '\n').trimEnd() + '\n';
    if (norm(current) !== norm(out)) {
      console.error('fold-table.ts out of date (whitespace-normalized compare). Run: bun scripts/generate-fold-table.ts --input <CaseFolding.txt>');
      process.exit(1);
    }
    console.log('fold-table.ts up to date.');
    return;
  }
  // Atomic write: tmp + rename (no half-written table on crash).
  const tmpUrl = new URL('../packages/webgpu-search/src/fold-table.ts.tmp', import.meta.url);
  await writeFile(tmpUrl, out, 'utf8');
  await rename(tmpUrl, OUT_PATH);
  console.log(`Wrote ${OUT_PATH.pathname} (${out.length} chars)`);
}

await main();
