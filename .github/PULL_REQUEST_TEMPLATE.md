## Summary

<!-- What does this PR change, and why? Link related issues. -->

## Tests run

<!-- Check all that apply; paste commands + results. -->

- [ ] `bun run check:shaders`
- [ ] `bun run typecheck`
- [ ] `bun run build`
- [ ] `bun run test:mock`
- [ ] Targeted suite(s): <!-- e.g. test:contracts / test:parity / test:snapshot -->
- [ ] `bun run test:browser` (Chrome required; note `CHROME_BIN` if relevant)

## Contract impact

- [ ] No public-surface change (`docs/public-api.md` + `scripts/check-public-api.ts` untouched)
- [ ] Public-surface change: `docs/public-api.md` and `scripts/check-public-api.ts` updated in this PR
- [ ] Snapshot/wire change: `docs/snapshot-format.md` updated, legacy + fail-closed cases covered
- [ ] Benchmark fixtures changed: `BENCHMARK_FIXTURE_VERSION` bumped, matrix artifacts + `docs/benchmarks.md` regenerated

## Checklist

- [ ] No unguarded `window` / `document` / `navigator` in library code (Worker/Node/SSR safe)
- [ ] Device acquisition goes through `GpuDevicePool` or injected `options.device`
- [ ] WGSL sources and inlined shader strings stay in sync (if touched)
- [ ] No temporary files, debug logs, or dead scripts left in the diff
- [ ] Issue reports (if any) use header/stats/timings metadata, never PII record text
