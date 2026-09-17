/**
 * M2 parity-path lint: bans locale/UTF-16 helpers in the parity path.
 *
 * Banned substrings in unicode-preprocess.ts + cpu-reference.ts:
 *   charCodeAt | toLowerCase | toUpperCase | indexOf
 * Rationale: locale-sensitive folding breaks tr-TR (I→ı); UTF-16
 * unit access splits surrogates; substring search must use scalar ===.
 * `codePointAt` / `fromCodePoint` are the approved replacements.
 *
 * Run: bun scripts/check-parity-lint.ts
 */
const targets = [
  new URL('../packages/webgpu-search/src/unicode-preprocess.ts', import.meta.url),
  new URL('../packages/webgpu-search/src/cpu-reference.ts', import.meta.url),
];
const banned = ['charCodeAt', 'toLowerCase', 'toUpperCase', 'indexOf'];
let fail = false;
for (const url of targets) {
  const text = await Bun.file(url).text();
  const lines = text.split('\n');
  for (const token of banned) {
    const hits: number[] = [];
    lines.forEach((line, i) => {
      if (line.includes(token)) hits.push(i + 1);
    });
    if (hits.length > 0) {
      console.error(`❌ ${url.pathname.split('/').pop()}: banned "${token}" at lines ${hits.join(',')}`);
      fail = true;
    }
  }
}
if (fail) {
  console.error('Parity lint failed.');
  process.exit(1);
}
console.log('✅ parity lint clean (no banned helpers in parity path)');
