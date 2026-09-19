---
'webgpu-search': major
---

Release v0.2.0: Breaking Unicode code-point-safe CPU/GPU search engine with U2F2 format, deterministic tie-breaking, and multi-corpus benchmark harness.

- **Breaking Unicode Migration**: Replaced 8-bit ASCII byte packing with 32-bit (`u32`) Unicode scalar values and token offsets. Non-Latin scripts (CJK, Arabic, Indic, Cyrillic) and emojis survive without surrogate splitting or destructive `?` replacement.
- **Spec-Compliant Pipeline**: Full `trim` → `toWellFormed()` (lone-surrogate repair) → `NFC` → `CaseFolding-16.0.0` (`C+F`) → `NFC` normalization pipeline shared byte-for-byte between WebGPU shaders and CPU parity reference scorer.
- **U2F2 Binary Serialization**: New `0x55324632` magic, versioned header (`formatVersion: 2`), CRC32 checksum, and metadata enums. Legacy v0.1 buffers fail closed with `IncompatibleIndexError`.
- **Bit-Exact Scoring & Deterministic Tie-Breaking**: Integer subsequence parity scorer guarantees identical results and rankings `(score DESC, index ASC)` between WebGPU compute passes and CPU fallback when `hasOverflow === false`.
- **Buffer & Limit Hardening**: Authoritative per-buffer limit validation against `device.limits`, 128 post-fold code point query bounds (`QueryTooLongError`), and throw-on-use `slotBytes` deprecation (`IncompatibleOptionError`).
- **Comprehensive Documentation**: Complete migration guide in `docs/migration-v0.2.md` and updated multi-corpus benchmarks in `README.md`.
