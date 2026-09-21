/**
 * parity-path lint: bans locale/UTF-16 helpers in the parity path.
 *
 * Strict targets (fail on hit): text-normalization.ts, exact-scorer.ts,
 * guard.ts, case-fold-table.ts, dataset-packing.ts, webgpu-engine.ts. Extended
 * bans: charCodeAt (except the vetted ASCII/surrogate-scan contexts below),
 * toLowerCase, toUpperCase, toLocaleLowerCase, toLocaleUpperCase,
 * localeCompare, indexOf, `.includes(` (as indexOf substitute), charAt.
 * Rationale: locale-sensitive folding breaks tr-TR (I->i with dot); UTF-16
 * unit access splits surrogates; substring search must use scalar ===.
 * `codePointAt` / `fromCodePoint` are the approved replacements.
 *
 * promotion: dataset-packing.ts + webgpu-engine.ts moved from info-only to strict —
 * the representation swap makes them parity code (u32 scalars, pure `==`
 * shaders, `normalizeText` queries). Legacy `sanitizeStringForSlot` /
 * `packStringsToGPUBuffer` stay exported behind a `@deprecated` shim (
 * migration) and were rewritten to use `codePointAt` so the
 * strict gate holds without deleting them.
 *
 * Legacy-quarantined files (cpu-engine.ts) use banned helpers behind the
 * legacy uFuzzy/native path on purpose; reported as info, not failures.
 * Comments and string literals are stripped before matching to avoid false
 * positives (e.g. docs mentioning the words).
 *
 * Portable: node:fs only (runs on Bun and Node).
 * Run: bun scripts/check-parity-lint.ts (or: node scripts/check-parity-lint.ts)
 */
import { readFile } from 'node:fs/promises';

const strictTargets = [
  '../packages/webgpu-search/src/text-normalization.ts',
  '../packages/webgpu-search/src/exact-scorer.ts',
  '../packages/webgpu-search/src/search/typo-tolerance.ts',
  '../packages/webgpu-search/src/search/token-search.ts',
  '../packages/webgpu-search/src/search/prefix-search.ts',
  '../packages/webgpu-search/src/guard.ts',
  '../packages/webgpu-search/src/case-fold-table.ts',
  '../packages/webgpu-search/src/dataset-packing.ts',
  '../packages/webgpu-search/src/webgpu-engine.ts',
  '../packages/webgpu-search/src/document-index.ts',
  '../packages/webgpu-search/src/highlight.ts',
];
const legacyInfoTargets = [
  '../packages/webgpu-search/src/cpu-engine.ts',
];
// charCodeAt is allowed only in these vetted ASCII-only fast paths.
const charCodeAtAllowlist: Record<string, number[]> = {
  'text-normalization.ts': [],
};
const banned = [
  'toLowerCase',
  'toUpperCase',
  'toLocaleLowerCase',
  'toLocaleUpperCase',
  'localeCompare',
  'indexOf',
  'charAt',
];

function stripCommentsAndStrings(src: string): string[] {
  // Remove block comments, line comments; blank strings (keep line numbers).
  const noBlock = src.replace(/\/\*[\s\S]*?\*\//g, (m) => '\n'.repeat((m.match(/\n/g) || []).length));
  return noBlock.split('\n').map((line) => {
    let out = '';
    let inStr: string | null = null;
    for (let i = 0; i < line.length; i++) {
      const c = line[i];
      if (inStr) {
        if (c === '\\') {
          i++;
          continue;
        }
        if (c === inStr) inStr = null;
        continue;
      }
      if (c === '"' || c === "'" || c === '`') {
        inStr = c;
        continue;
      }
      if (c === '/' && line[i + 1] === '/') break;
      out += c;
    }
    return out;
  });
}

let fail = false;
for (const rel of strictTargets) {
  const url = new URL(rel, import.meta.url);
  const text = await readFile(url, 'utf8');
  const lines = stripCommentsAndStrings(text);
  const name = rel.split('/').pop() as string;
  for (const token of banned) {
    const hits: number[] = [];
    lines.forEach((line, i) => {
      if (line.includes(token)) hits.push(i + 1);
    });
    if (hits.length > 0) {
      console.error(`FAIL ${name}: banned "${token}" at lines ${hits.join(',')}`);
      fail = true;
    }
  }
  // charCodeAt: only ASCII fast-path lines (guarded by 0x7f comparison nearby).
  const ccHits: number[] = [];
  lines.forEach((line, i) => {
    if (line.includes('charCodeAt')) ccHits.push(i + 1);
  });
  if (ccHits.length > 0) {
    const allowed = new Set(charCodeAtAllowlist[name] ?? []);
    // Auto-allow vetted unit accesses: ASCII fast path (0x7f/0x41) and the
    // lone-surrogate last-resort scan, which must inspect UTF-16 units
    // (0xd800/0xdbff/0xdc00) to detect lone surrogates on pre-2020 engines.
    const autoOk = ccHits.filter((ln) => {
      const ctx = lines.slice(Math.max(0, ln - 6), ln + 2).join('\n');
      return (
        ctx.includes('0x7f') ||
        ctx.includes('0x41') ||
        ctx.includes('isAscii') ||
        ctx.includes('asciiToTokens') ||
        ctx.includes('0xd800') ||
        ctx.includes('0xdbff') ||
        ctx.includes('0xdc00')
      );
    });
    const bad = ccHits.filter((ln) => !allowed.has(ln) && !autoOk.includes(ln));
    if (bad.length > 0) {
      console.error(`FAIL ${name}: banned "charCodeAt" at lines ${bad.join(',')} (outside ASCII fast path)`);
      fail = true;
    }
  }
  // `includes` is banned except the quarantined caseSensitive filter note.
  const incHits: number[] = [];
  lines.forEach((line, i) => {
    if (line.includes('.includes(')) incHits.push(i + 1);
  });
  if (incHits.length > 0) {
    console.error(`FAIL ${name}: banned ".includes(" at lines ${incHits.join(',')} (use scalar ===)`);
    fail = true;
  }
}
for (const rel of legacyInfoTargets) {
  const url = new URL(rel, import.meta.url);
  const text = await readFile(url, 'utf8');
  const n = (text.match(/charCodeAt|toLowerCase|toUpperCase|indexOf/g) || []).length;
  if (n > 0) console.log(`info ${rel.split('/').pop()}: ${n} legacy-path helper uses (allowlisted, excluded from parity matrix)`);
}
if (fail) {
  console.error('Parity lint failed.');
  process.exit(1);
}
console.log('parity lint clean (no banned helpers in parity path)');
