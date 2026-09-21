/**
 * Naming hygiene gate: fails on temporal/codename leaks in user-visible
 * identifiers (file names, exports, script names, comments, error strings).
 *
 * Excludes:
 * - `docs/archive/**` (historical records)
 * - `@deprecated` alias lines + wire-magic comments (allowed per plan)
 * - generated `dist/**`
 */
const BAN = /\bM[1-8]\b|\bv0\.[1234]\b|\bU2D[34]\b|\bU2F2\b|FORMAT_VERSION_4|\bV2\b|\bphase [123]\b/;
const ALLOW = /@deprecated|0x55324|docs\/archive|U2F2.*wire|wire.*U2|legacy snapshot|canonical.*snapshot|SHARD|deprecated|alias/i;

const targets = [
  "packages/webgpu-search/src",
  "apps/benchmark/src",
  "apps/monaco-palette/src",
  "apps/log-viewer/src",
  "examples",
  "scripts",
];
const exts = [".ts", ".tsx", ".js", ".mjs"];

let hits = 0;
for (const root of targets) {
  const glob = new Bun.Glob("**/*");
  const dir = Bun.file(root);
  let entries: string[] = [];
  try {
    for await (const e of glob.scan({ cwd: root, absolute: false })) {
      if (exts.some((x) => e.endsWith(x))) entries.push(`${root}/${e}`);
    }
  } catch {
    continue;
  }
  void dir;
  for (const f of entries) {
    if (f.endsWith("check-naming.ts")) continue;
    const text = await Bun.file(f).text();
    const lines = text.split("\n");
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i] as string;
      if (!BAN.test(line)) continue;
      if (ALLOW.test(line)) continue;
      const prev = i > 0 ? (lines[i - 1] as string) : "";
      if (/deprecated/i.test(prev)) continue;
      // case-fold math vars (m1/m2 outputs) are not milestones
      if (/out1|out2|1→2|→/.test(line) && /\[cp,len/.test(line)) continue;
      console.error(`${f}:${i + 1}: ${line.trim().slice(0, 160)}`);
      hits++;
    }
  }
}

// File-name gate: no milestone/version tags in durable names
const nameBan = /(^|[-_])(m[1-8]|v0[234]|v04|u2d[34]|u2f2)($|[-_.])/i;
const nameAllow = /check-naming|parity/i;
{
  const glob = new Bun.Glob("**/*");
  for (const root of ["packages/webgpu-search/src", "scripts"]) {
    try {
      for await (const e of glob.scan({ cwd: root })) {
        const base = e.split("/").pop() as string;
        if (nameBan.test(base) && !nameAllow.test(base)) {
          console.error(`filename leak: ${root}/${e}`);
          hits++;
        }
      }
    } catch { /* missing dir */ }
  }
}

if (hits > 0) {
  console.error(`\n[lint:naming] ${hits} naming leak(s). See docs/naming-conventions.md.`);
  process.exit(1);
}
console.log("[lint:naming] clean.");
