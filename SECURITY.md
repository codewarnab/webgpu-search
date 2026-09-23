# Security Policy

## Supported versions

Only the current `1.x` line of `webgpu-search` is supported with security
fixes. Scoring, storage, ordering, or echo changes require a major per
`docs/public-api.md` §7; security or correctness hazards may shorten the
normal deprecation window with a documented migration.

| Version | Supported |
| ------- | --------- |
| `1.x`   | Yes       |
| `< 1.0` | No (upgrade to `1.x`) |

## Reporting a vulnerability

Do **not** open a public GitHub issue for a suspected vulnerability. Use
GitHub's **Private vulnerability reporting** (Security tab → Report a
vulnerability) on
[codewarnab/webgpu-search](https://github.com/codewarnab/webgpu-search),
or contact the maintainers through a private channel.

Include, where possible:

- Package version + entry (`webgpu-search` / `webgpu-search/worker`)
- Browser/Node/Bun version, OS, GPU path if relevant
- Repro steps and full error `name` / `message` / `details`
- Snapshot header / `getStats()` / timings metadata (see
  `docs/diagnostics.md` §5) — never paste PII record text

We will acknowledge receipt, investigate, and coordinate a fix and disclosure
timeline with you.

## Scope and trust boundaries

- The library is offline-capable and zero-network: WGSL compute + CPU
  TypeScript, single runtime dependency `@leeoniya/ufuzzy`. No WASM, no
  `eval`, no remote fetch, no telemetry exfiltration.
- Snapshots and IndexedDB persist **raw record bytes unencrypted**,
  origin-scoped. Treat persisted corpora (logs, docs, messages) as sensitive
  as the source records. Prefer `serialize({ decoupled: true })` when the
  host already guards docs elsewhere.
- CRC32 is corruption detection, not authenticity. Treat snapshots from
  untrusted origins as untrusted input and restore only through the
  fail-closed path (`IncompatibleIndexError` / `IncompatibleHookError` /
  `ProfileMismatchError`; taxonomy in `docs/snapshot-format.md` §3).
- Render `highlightedText` only through an escaping sanitizer that preserves
  `<mark>`; pass `escapeHtml: true` for untrusted corpora.
- Browser/WebGPU numbers from software adapters (SwiftShader/LLVMpipe) are
  labeled `pending-hardware` and are never hardware claims.

Full boundaries: `docs/security-privacy.md`.
