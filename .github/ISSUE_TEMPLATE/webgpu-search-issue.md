---
name: webgpu-search issue
description: Report a search, snapshot, worker, or packaging defect
labels: [bug]
---

## Environment

- Package version + entry (`webgpu-search` / `webgpu-search/worker`):
- Browser + version / Node/Bun version:
- OS + GPU/driver (or `webgpu-unsupported` path):

## Repro

- Corpus size (docCount/rowCount), query (verbatim), mode + full search options:
- `preferGpu`, `cpuScorer`, `tokenMatch`/`prefixMatch`/`typoTolerance`, `budget`, `filter`/`facets` shape (no PII record text):

## Observed contract

- `engine`, `fallbackReason`, `profileId`, `scoringVersion`, `cpuScorer`:
- `totalMatches` / `candidateCount` / `hasOverflow`:
- `timings` + (`diagnostics: true`) `diagnostics` JSON incl. `warnings`:
- `getStats()` JSON:

## Snapshot (if restore/IDB related)

- `decodeSnapshotHeader(buf)` JSON:
- Full error `name` / `message` / `details` (`IncompatibleIndexError`, `ProfileMismatchError`, `CostBudgetExceededError`, …):

## Expected

- What the CPU baseline (`preferGpu: false`, `cpuScorer: 'exact'`) returns for the same corpus/query, and why the observed result diverges:
