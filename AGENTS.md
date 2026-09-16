# Repository Rules & Operating Guidelines

## 1. Mandatory Post-Work Subagent Verification Rule

> **CRITICAL RULE**: After finishing any work (feature implementation, refactoring, bugfix, or migration), the agent **MUST ALWAYS spawn a subagent to verify and review the work** before concluding the task.

### Verification Protocol:
1. **Never finalize work without subagent review**: Do not present a task as complete until a verification subagent has executed and confirmed correctness.
2. **Subagent Verification Responsibilities**:
   - **Automated Validation**: Execute `bun run check:shaders`, `bun run typecheck`, `bun run build`, and `bun run test:mock`.
   - **Contract & Regression Check**: Inspect modified and newly created files to verify public contracts, type symmetry, error handling, and edge cases.
   - **Cross-Platform & Portability Audit**: Ensure library code contains no unguarded DOM references (`document`, `window`) to maintain 100% Web Worker, Node.js, and SSR safety.
   - **Clean Diff Audit**: Ensure no temporary test files, debug console logs, or dead scripts remain.
3. **Resolve Findings Before Sign-off**: Any actionable defects or regressions flagged by the verification subagent must be resolved before presenting the final response to the user.

---

## 2. Architecture & Monorepo Standards

1. **Turborepo Workspace Isolation**:
   - `packages/webgpu-search`: The core, zero-dependency library published to npm. Must remain portable across browser main thread, Web Workers, Node.js, and SSR.
   - `apps/benchmark`: Interactive evaluation dashboard. Consumes `webgpu-search` via workspace linking (`workspace:*`) and Vite dev source aliasing.
2. **Shader Integrity**:
   - Every WGSL compute shader edit must be validated offline via `bun run check:shaders` (`vgpu check`).
   - Keep `.wgsl` shaders and their inlined export strings synchronized.
3. **Device & Context Safety**:
   - Always route WebGPU device acquisition through `WebGPUContextManager` or accept injected custom devices. Never call `navigator.gpu.requestDevice()` directly inside repeated index allocations to avoid browser context exhaustion.
4. **Scoring Symmetry**:
   - Both WebGPU compute passes and CPU fallbacks (`uFuzzy` / native) must conform to the unified `SearchResultItem` interface with descending normalized integer scores.
