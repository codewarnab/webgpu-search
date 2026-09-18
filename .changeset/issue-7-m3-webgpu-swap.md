---
'webgpu-search': major
---

Issue #7 M3: WebGPU representation swap (breaking v0.2 preview).

- New `packUnicodeToGPUBuffer()` packs post-fold u32 scalars with a zero-renorm `Uint32Array[]` fast path; `slotBytes` is throw-on-use. New `serializeUnicodeDataset()` / `deserializeUnicodeDataset()` persist a 36-byte `U2F2` header (magic, `FORMAT_VERSION 2`, profile/unicode/scoring enums, counts, CRC32); legacy v0.1 buffers without magic fail closed with `IncompatibleIndexError`.
- WGSL substring/fuzzy shaders rewritten as pure-`==` scalar comparators mirroring `cpu-reference.ts` integer formulas (32 B uniform header with `flagsAndProfile`, 512 B persistent storage query buffer, 5 bindings, `@workgroup_size(128)`, chunked multi-dispatch past `maxComputeWorkgroupsPerDimension`).
- `WebGPUEngine` overhaul: token-only datasets (`text: ''`, enriched by `SearchIndex`), 0-row zero-dispatch, defensive `QueryTooLongError` / `ProfileMismatchError` gates, single-mutex `searchCold`, generation-epoch abort for in-flight `mapAsync`, mutex-safe `destroy()`, device-loss handling, authoritative per-buffer limit checks post-`init()`.
- `SearchIndex` routes all valid queries (up to 128 post-fold tokens, any script) to WebGPU when ready — the M2 ASCII/59-token gate is deleted — with parity-CPU fallback; `vramAllocatedBytes` is now actual packed bytes.
- New `profileEnum` / `unicodeEnum` / `scoringEnum` maps in `text-profile.ts` (`1 = 'unicode-default' / '16.0.0' / 'parity-v1'`).
