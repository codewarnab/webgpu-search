---
name: plan-phase-breakdown
description: Deeply breaks down a designated phase of an architectural or implementation plan into granular, engineering-ready milestones with concrete deliverables, contracts, and quality gates, then surgically updates only that target section of the plan in place.
---

# Plan Phase Breakdown & In-Place Refiner

A specialized agent runbook for taking a high-level phase within an architectural or implementation plan (e.g. in `PLAN.md` or `ISSUE-*-PLAN.md`), elaborating it into granular, production-grade milestones with explicit deliverables, types, and verification gates, and **surgically updating only that target section** while leaving the rest of the plan document 100% intact.

---

## When to Use

Use this skill whenever:
- A user asks to break down, expand, detail, or refine a specific phase of a project plan (e.g., "break down Phase 1 of PLAN.md", "elaborate Phase 3 into milestones").
- A phase in an architectural plan contains high-level bullet points that need concrete implementation runbooks, file paths, types, algorithms, and validation commands before engineering begins.

---

## Core Principles

1. **Surgical In-Place Update (Zero Collateral Damage)**:
   - Only modify the target phase within the plan document.
   - Surrounding sections (prior phases, subsequent phases, architecture diagrams, executive summaries, design tokens) must remain completely untouched.
   - Never regenerate or overwrite the entire document if you only need to rewrite one section.

2. **Engineering-Grade Granularity**:
   - Transform high-level aspirations into concrete, actionable steps.
   - For every milestone, specify target files, contracts/types, invariants, and exact verification commands.

3. **Contextual Coherence**:
   - Align terminology, directory structures, and tech stack with the overall plan's architecture.
   - Ensure the output numbering and formatting match the existing document's Markdown hierarchy.

---

## 4-Step Execution Protocol

```
[1. Target & Context Research] ──> [2. Granular Milestone Design]
                                             │
[4. Diff Audit & Verification] <── [3. Surgical In-Place Update]
```

---

### Step 1: Target Identification & Context Research

1. **Locate the Plan Document**:
   - Determine which plan file the user is referencing (e.g., `PLAN.md`, `ISSUE-9-PLAN.md`, or the project's roadmap).
2. **Read the Full Architectural Context**:
   - Read the plan's architectural principles, tech stack, directory layouts, and design tokens to ensure technical consistency.
3. **Inspect the Target Phase**:
   - Note the exact start line and end line of the phase to be expanded.
   - Note the exact heading style (e.g. `### Phase 1: ...` or `1. **Phase 1: ...**`).
   - Identify preceding and succeeding section headers to establish strict replacement boundaries.

---

### Step 2: Granular Milestone Formulation

Break down the target phase into 2 to 5 sequential, self-contained sub-milestones (e.g., `M<Phase>.1`, `M<Phase>.2` or `Milestone 1`, `Milestone 2`).

For each sub-milestone, produce:
1. **Title & Objective**: Concise description of what is built and why.
2. **Target Files**: Explicit relative file paths to be created, modified, or deleted.
3. **Data Structures & API Contracts**:
   - TypeScript interfaces, Rust structs/enums, WGSL layouts, or function signatures.
   - Public exports vs. internal helpers.
4. **Implementation Specifications**:
   - Algorithms, edge cases, error handling strategies, and boundary invariants.
   - Backpressure, concurrency, or memory management requirements (if applicable).
5. **Quality Gates & Automated Verification**:
   - Concrete test commands (e.g., `cargo test`, `vitest run`, `bun run test:mock`, `bun run typecheck`).
   - Specific assertions and edge cases to test.
6. **Definition of Done (DoD)**: Verifiable acceptance checklist.

---

### Step 3: Surgical In-Place Update

1. **Prepare the Replacement Block**:
   - Format the newly elaborated milestones using the document's established heading hierarchy, code block styles, and callout patterns.
   - Include an ASCII or Mermaid sub-pipeline diagram if the phase involves non-trivial dataflow or dependencies.
2. **Execute In-Place Edit**:
   - Use `replace_file_content` targeting ONLY the lines from the start of the target phase heading to the line preceding the next section heading.
   - Ensure leading indentation and newlines are preserved.
3. **Verify Document Integrity**:
   - Read back the modified file around the edit boundaries using `view_file`.
   - Confirm that preceding sections and subsequent sections are completely intact and uncorrupted.

---

### Step 4: Verification & Git Diff Audit

1. **Check Git Status & Diff**:
   ```bash
   git diff <plan-file>
   ```
2. Verify that:
   - Only the intended phase lines were changed (`-` old phase lines, `+` new detailed milestones).
   - No unrelated lines, empty lines at EOF, or global headers were modified.
   - All markdown tables, code fences, and links in the updated section render cleanly.
3. Report completion to the user with a summary of the newly detailed milestones and the updated section diff.
