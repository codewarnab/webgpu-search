---
'webgpu-search': major
---

Issue #7 M0/M1: Unicode contract + profiled API (breaking v0.2 preview).

- Freezes the v0.2 text contract (`UNICODE_VERSION 16.0.0`, `SCORING_VERSION parity-v1`, `FORMAT_VERSION 2`, `QUERY_TOKENS_MAX 128`, `RESULT_LIMIT_MAX 8192`, `TextProfileId 'unicode-default'`).
- `SearchResponse` gains required `profileId`/`scoringVersion`/`cpuAlgorithm`; `IndexStats` gains required `profileId`/`unicodeVersion`/`scoringVersion`/`tokenCount`/`folded`/`formatVersion`.
- `IndexOptions.caseSensitive` is now pack-time; per-query mismatch throws `ProfileMismatchError` (build one index per mode).
- `IndexOptions.slotBytes` is throw-on-use (`IncompatibleOptionError`; removal in v0.3).
- New `SearchOptions.cpuAlgorithm` (`'parity'` default, `'ufuzzy'` explicit CPU-only) and `onQueryTooLong` (`'throw'` default, `'cpu-fallback'` forces CPU). M1 enforces query length on a pre-fold code-point approximation and the `preferGpu + ufuzzy` conflict; exact post-fold parity scoring lands in M2 (`cpu-reference.ts`/`unicode-preprocess.ts`). See `docs/unicode-contract.md` and `ISSUE-7-PLAN.md`.
