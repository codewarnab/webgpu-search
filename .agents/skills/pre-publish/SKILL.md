---
name: pre-publish
description: Automated pre-flight validation checklist and publishing pipeline for webgpu-search releases to npm. Use whenever preparing, bumping, verifying, or publishing a new package version.
---

# Pre-Publish Verification & Release Checklist

This skill defines the mandatory pre-flight validation pipeline and safe release procedure for publishing the `webgpu-search` package to npm.

Whenever a user requests a release, version bump, or package publication, execute this checklist sequentially. **Never skip any phase.**

---

## Pipeline Overview

```
[Phase 1: Git Audit] ──> [Phase 2: Shader Check] ──> [Phase 3: Typecheck & Build]
                                                                  │
[Phase 6: SemVer Bump] <── [Phase 5: Runtime Audit] <── [Phase 4: Tarball Audit]
          │
          └──> [Phase 7: NPM Publish & 2FA] ──> [Phase 8: Git Tag & Push]
```

---

## Phase 1: Working Tree Cleanliness Audit

Ensure no uncommitted changes, stray scratch files, or unfinished edits exist:

```bash
git status --short
```

- **Pass Criteria**: Working tree is completely clean (`nothing to commit, working tree clean`) or contains only the deliberate changes intended for the new release.
- **Fail Action**: Commit or stash uncommitted edits before proceeding.

---

## Phase 2: Offline Shader AST & Memory Layout Check

Validate all WGSL compute shaders offline using `vgpu check`:

```bash
bun run check:shaders
```

- **Pass Criteria**: Both `src/shaders/fuzzy.wgsl` and `src/shaders/substring.wgsl` pass AST parsing, struct member alignment rules, and 16-byte uniform alignment.
- **Fail Action**: Fix syntax or alignment errors in `.wgsl` files before building.

---

## Phase 3: Monorepo Typecheck & Build

Ensure complete TypeScript type safety and fresh distribution bundles across all workspaces:

```bash
bun run typecheck
bun run build
```

- **Pass Criteria**:
  - `tsc --noEmit` exits with code 0 across both `packages/webgpu-search` and `apps/benchmark`.
  - `tsup` generates dual outputs:
    - ESM: `dist/index.js` + `dist/index.d.ts` + sourcemaps
    - CJS: `dist/index.cjs` + `dist/index.d.cts` + sourcemaps
  - Gzipped ESM bundle size is $\le$ 20 kB (currently ~8.3 kB gzip).

---

## Phase 4: Headless Unit & Regression Testing

Run the full headless GPU mock test suite (0ms hardware execution):

```bash
bun run test:mock
bun run test
```

- **Pass Criteria**: All 10/10 test assertions pass:
  1. `WebGPUEngine` initialization with mock device.
  2. Buffer allocations (uniform, candidate pool, staging).
  3. `packStringsToGPUBuffer` 64-byte row layout and alignment.
  4. Query bounds and empty-query edge cases.
  5. `SearchIndex` creation and GPU routing.
  6. CPU auto-routing fallback (`uFuzzy` / native).
  7. Cold search pipeline (`searchCold`).
  8. Memory budget validation (`checkMemoryBudget`) and string sanitization (`sanitizeStringForSlot`).
  9. Empty dataset zero-allocation protection.
  10. Case sensitivity parity.

---

## Phase 5: Tarball Content & Leak Audit

Simulate npm packaging to ensure zero private keys, tests, or unnecessary source files are published:

```bash
cd packages/webgpu-search
npm pack --dry-run
```

- **Pass Criteria**:
  - Exactly **9 files** packaged:
    - `LICENSE`
    - `README.md`
    - `package.json`
    - `dist/index.js`
    - `dist/index.js.map`
    - `dist/index.d.ts`
    - `dist/index.cjs`
    - `dist/index.cjs.map`
    - `dist/index.d.cts`
  - Tarball size $\le$ 70 kB (unpacked $\le$ 300 kB).
  - **Zero leaks**: No `.env`, test scripts, `.wgsl` raw files (they are inlined into `dist/`), or benchmarks.

---

## Phase 6: Dual Runtime & Portability Audit

Verify both module systems load cleanly in a standalone Node.js process:

```bash
# ESM import check
node --input-type=module -e "import { SearchIndex } from './packages/webgpu-search/dist/index.js'; console.log('ESM OK:', typeof SearchIndex.create);"

# CJS require check
node -e "const { SearchIndex } = require('./packages/webgpu-search/dist/index.cjs'); console.log('CJS OK:', typeof SearchIndex.create);"
```

- **Pass Criteria**: Both commands print `function` and exit with code 0.
- **Portability Check**: Confirm zero unguarded `window` or `document` references in `src/` to guarantee Web Worker, Node.js, and SSR safety.

---

## Phase 7: Version Bumping & SemVer

1. Determine the appropriate SemVer increment based on changes:
   - **Patch (`0.1.x`)**: Bug fixes, performance optimizations, documentation.
   - **Minor (`0.x.0`)**: New backward-compatible features, new search modes.
   - **Major (`x.0.0`)**: Breaking API changes, altered struct layouts, dropped Node/browser versions.
2. Update version in `packages/webgpu-search/package.json`.
3. Synchronize lockfile:
   ```bash
   bun install
   ```

---

## Phase 8: Safe Publishing & Git Tagging

1. Confirm npm authentication:
   ```bash
   npm whoami
   ```
2. Publish with public access:
   ```powershell
   cd packages/webgpu-search
   npm publish --access public
   ```
   *(If 2FA/OTP is required, prompt the user for the 6-digit code or append `--otp=<code>`)*.
3. Commit and tag the release:
   ```powershell
   git add .
   git commit -m "chore(release): webgpu-search@<version>"
   git tag v<version>
   git push origin main --tags
   ```
4. Verify live package on registry:
   ```bash
   npm view webgpu-search version
   ```
