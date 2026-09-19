---
name: milestone-workflow
description: Autonomous milestone execution workflow for webgpu-fuzzy-search. Use whenever the user provides a milestone, issue, or feature task to research, implement, verify, and raise a PR end-to-end without requiring repetitive prompting.
---

# Milestone Workflow

An autonomous, end-to-end runbook for executing development milestones in `webgpu-fuzzy-search`. When given a milestone prompt (e.g., "Issue #9 M3: ..."), execute all phases sequentially without pausing for user confirmation unless blocked.

---

## Workflow Pipeline

```
[Phase 1: Deep Research] ──> [Phase 2: Branch Setup] ──> [Phase 3: Implementation]
                                                                  │
[Phase 6: PR Creation] <── [Phase 5: Subagent Verification] <── [Phase 4: Test & Build]
```

---

## Phase 1: Deep Research (Code-First & Web)

1. **Codebase Inspection**:
   - Inspect existing architecture, types, and plan specifications:
     - Milestone plan docs: `PLAN.md`, `ISSUE-*-PLAN.md`.
     - Core types and public contracts: `packages/webgpu-search/src/types.ts`.
     - Engine and index implementations: `packages/webgpu-search/src/`.
     - WGSL compute shaders: `packages/webgpu-search/src/shaders/` if modifying GPU compute.
     - Relevant test suites: `scripts/` and `packages/webgpu-search/test/`.
2. **Web Research (as needed)**:
   - Perform targeted web research if external WebGPU/WGSL specifications, Unicode standards, or browser compatibility constraints are required.

---

## Phase 2: Milestone Branch Setup

1. Check current branch and working tree status:
   ```bash
   git status --short
   ```
2. If starting a new milestone, create and checkout a feature branch off `origin/main`:
   ```bash
   git checkout -b feat/issue-<id>-m<milestone>-<slug> origin/main
   ```
   *(If already on the intended milestone branch, verify the working tree state).*

---

## Phase 3: Implementation Standards

1. **Turborepo & Packaging Constraints**:
   - `packages/webgpu-search`: Zero runtime dependencies (`dependencies: {}`).
   - Cross-platform portability: Guard against direct DOM globals (`window`, `document`) to maintain 100% Web Worker, Node.js, and SSR safety.
2. **WebGPU Context & Safety**:
   - Route device acquisition through `WebGPUContextManager` or injected custom devices. Never call unmanaged `requestDevice()` inside index instances.
3. **WGSL Shader Synchronization**:
   - When modifying `.wgsl` shaders, keep any inlined shader exports in TypeScript code synchronized.
4. **Scoring Symmetry**:
   - Maintain strict parity between WebGPU compute readbacks and CPU reference scorers (`SearchResultItem` with descending normalized integer scores).

---

## Phase 4: Validation & Quality Gates

Run the project verification commands:

```bash
# 1. Offline WGSL shader check
bun run check:shaders

# 2. Workspace TypeScript typecheck
bun run typecheck

# 3. Fresh build across packages
bun run build

# 4. Headless unit & mock tests
bun run test:mock

# 5. Milestone-specific test script (if applicable)
bun run test:m<n>
```

All commands must exit with code 0. Fix any errors before proceeding.

---

## Phase 5: Mandatory Subagent Verification

Per repository guidelines (`AGENTS.md`), spawn a verification subagent before concluding:
- Verify test suites pass (`check:shaders`, `typecheck`, `build`, `test:mock`).
- Audit contracts, type safety, and error handling.
- Audit cross-platform safety (zero unguarded DOM globals).
- Verify clean git diff (no lingering debug console logs or temporary files).
- Resolve all findings flagged by the subagent before sign-off.

---

## Phase 6: Commit, Push & Pull Request

1. **Stage and Commit** (Conventional Commits):
   ```bash
   git add <modified-files>
   git commit -m "feat(<scope>): issue #<id> M<milestone> <concise description>"
   ```
2. **Push to Remote**:
   ```bash
   git push -u origin <branch-name>
   ```
3. **Create Pull Request**:
   Use GitHub CLI targeting `main`:
   ```bash
   gh pr create --base main --head <branch-name> \
     --title "feat(<scope>): issue #<id> M<milestone> <title>" \
     --body "$(cat <<'EOF'
   ## Summary
   Brief summary of milestone implementation.

   ### Key Deliverables
   - Bulleted list of implemented features and architectural details

   ### Testing & Verification
   - Confirmation of passing check:shaders, typecheck, build, test:mock, and milestone tests.
   EOF
   )"
   ```
4. Output the created PR URL and completion summary.
