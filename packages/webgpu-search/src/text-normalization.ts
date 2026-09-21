/**
 * Shared text normalization.
 *
 * Single pipeline for records + queries:
 * trim -> toWellFormed (or lone-surrogate regex fallback -> U+FFFD)
 * -> NFC -> default case folding + NFC (when normalized) -> u32 scalar stream.
 *
 * Size and empty checks apply to the post-normalization token count.
 * Scores, spans and penalties are measured in post-normalization code points;
 * normalized coordinates must NOT be used to slice the original string
 * (see "Strasse" caveat in docs/text-normalization.md).
 *
 * Portable: no DOM refs. The exact path avoids locale-sensitive case
 * conversion and UTF-16 unit access (see lint script).
 */

import { foldCaseScalar } from './case-fold-table';

/**
 * Lone-surrogate fallback source (no flags, lookbehind-free so Safari <16.4
 * and old Hermes parse it). Group 1 preserves valid surrogate pairs;
 * bare surrogates are replaced with U+FFFD. Exported as a string so
 * consumers cannot inherit `/g` lastIndex state.
 */
export const LONE_SURROGATE_SOURCE: string =
  '([\\uD800-\\uDBFF][\\uDC00-\\uDFFF])|[\\uD800-\\uDFFF]';

/**
 * Legacy exported pattern (kept for compat).
 *
 * Breaking: non-global (no `/g`) and pair-preserving semantics — valid
 * surrogate pairs match as group 1 (preserved), lone surrogates match
 * individually (replaced). Previously `/…/g` with lookbehind assertions;
 * changed for Safari <16.4 / old Hermes parse safety (lookbehind-free) and
 * to end shared `/g` lastIndex hazards. `str.replace(
 * LONE_SURROGATE_PATTERN, …)` now replaces only the FIRST lone surrogate;
 * use `new RegExp(LONE_SURROGATE_SOURCE, 'g')` (or split/join) for global
 * replacement. Prefer `LONE_SURROGATE_SOURCE` for new code.
 */
export const LONE_SURROGATE_PATTERN: RegExp = new RegExp(
  LONE_SURROGATE_SOURCE,
);

function getFallbackRegExp(): RegExp {
  return new RegExp(LONE_SURROGATE_SOURCE, 'g');
}

const REPLACEMENT_CHAR = '\uFFFD';

const nativeToWellFormed: ((this: string) => string) | null =
  typeof String.prototype !== 'undefined' &&
  typeof (String.prototype as unknown as { toWellFormed?: unknown })
    .toWellFormed === 'function'
    ? (String.prototype as unknown as { toWellFormed: () => string })
        .toWellFormed
    : null;

/**
 * ES2024 `toWellFormed` where available, else per-surrogate regex fallback.
 * Semantics: high+high = 2 FFFDs, high+EOF = 1, low-without-high = 1.
 * Genuine U+FFFD is indistinguishable by design (documented).
 * Intrinsic is captured at module load so prototype monkey-patching
 * cannot silently redefine the pinned matrix.
 */
export function toWellFormedSafe(input: string): string {
  const s: string = typeof input === 'string' ? input : String(input ?? '');
  if (nativeToWellFormed !== null) {
    try {
      return nativeToWellFormed.call(s);
    } catch {
      // Fall through to regex fallback below.
    }
  }
  try {
    return s.replace(getFallbackRegExp(), (m: string, pair: string | undefined) =>
      pair !== undefined ? m : REPLACEMENT_CHAR,
    );
  } catch {
    // Last-resort scan without lookbehind (pre-2020 engines).
    let out = '';
    for (let i = 0; i < s.length; i++) {
      const cu: number = s.charCodeAt(i);
      if (cu >= 0xd800 && cu <= 0xdbff) {
        const next: number = i + 1 < s.length ? s.charCodeAt(i + 1) : -1;
        if (next >= 0xdc00 && next <= 0xdfff) {
          out += s[i];
          out += s[i + 1] as string;
          i++;
        } else {
          out += REPLACEMENT_CHAR;
        }
      } else if (cu >= 0xdc00 && cu <= 0xdfff) {
        out += REPLACEMENT_CHAR;
      } else {
        out += s[i];
      }
    }
    return out;
  }
}

export interface NormalizedText {
  /** Post-normalization NFC scalar stream (u32 code points). */
  tokens: Uint32Array;
  /** Post-normalization token count (tokens.length). */
  tokenCount: number;
  /** True when tokenCount === 0 (degenerate post-processing query). */
  isEmpty: boolean;
  /** Whether default case folding was applied (false = NFC-only). */
  normalized: boolean;
  /** @deprecated Use normalized. */
  folded: boolean;
}

/** Append one scalar to a growing number array without banned helpers. */
function pushScalar(out: number[], cp: number): void {
  out[out.length] = cp;
}

/**
 * Normalize one record or query to post-normalization u32 tokens.
 *
 * @param raw raw JS string (records and queries share this path).
 * @param normalized true = NFC + default case folding + NFC; false = NFC-only.
 */
export function normalizeText(raw: string, normalized: boolean): NormalizedText {
  if (typeof raw !== 'string') {
    throw new TypeError(
      `[webgpu-search] normalizeText expects a string, got ${typeof raw}`,
    );
  }
  const trimmed: string = (raw ?? '').trim();
  if (trimmed.length === 0) {
    return { tokens: new Uint32Array(0), tokenCount: 0, isEmpty: true, normalized, folded: normalized };
  }
  // ASCII fast path: NFC is a no-op, lone-surrogate handling is a no-op,
  // fold is A-Z -> a-z only. Avoids two normalize() scans on bulk ASCII.
  if (isAsciiOnly(trimmed)) {
    if (!normalized) {
      return {
        tokens: asciiToTokens(trimmed),
        tokenCount: trimmed.length,
        isEmpty: false,
        normalized,
        folded: normalized,
      };
    }
    const out = new Uint32Array(trimmed.length);
    for (let i = 0; i < trimmed.length; i++) {
      const cu: number = trimmed.charCodeAt(i);
      out[i] = cu >= 0x41 && cu <= 0x5a ? cu + 32 : cu;
    }
    return { tokens: out, tokenCount: out.length, isEmpty: out.length === 0, normalized, folded: normalized };
  }
  const wellFormed: string = toWellFormedSafe(trimmed);
  const nfc1: string = wellFormed.normalize('NFC');

  let foldedScalars: number[];
  if (normalized) {
    foldedScalars = [];
    // Iterate by code point (astral counts as one). No UTF-16 unit access.
    for (const ch of nfc1) {
      const cp: number = ch.codePointAt(0) as number;
      const mapped: readonly number[] | null = foldCaseScalar(cp);
      if (mapped === null) {
        pushScalar(foldedScalars, cp);
      } else {
        for (let k = 0; k < mapped.length; k++) {
          pushScalar(foldedScalars, mapped[k] as number);
        }
      }
    }
    // Re-canonicalize: fold is not NFC-closed (e.g. U+0130 → i + combining dot).
    const foldedStr: string = fromCodePointsChunked(foldedScalars);
    const nfc2: string = foldedStr.normalize('NFC');
    const tokens: Uint32Array = stringToTokens(nfc2);
    return { tokens, tokenCount: tokens.length, isEmpty: tokens.length === 0, normalized, folded: normalized };
  }

  const tokens: Uint32Array = stringToTokens(nfc1);
  return { tokens, tokenCount: tokens.length, isEmpty: tokens.length === 0, normalized, folded: normalized };
}

function isAsciiOnly(s: string): boolean {
  for (let i = 0; i < s.length; i++) {
    if ((s.charCodeAt(i) as number) > 0x7f) return false;
  }
  return true;
}

function asciiToTokens(s: string): Uint32Array {
  const out = new Uint32Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
  return out;
}

/** Build a JS string from scalars in bounded chunks (huge-vector safe). */
function fromCodePointsChunked(scalars: number[]): string {
  const CHUNK = 8192;
  let out = '';
  for (let i = 0; i < scalars.length; i += CHUNK) {
    const end: number = i + CHUNK < scalars.length ? i + CHUNK : scalars.length;
    // Bounded spread: stack-safe (<=8192 args), ~2.7x faster than per-scalar +=.
    out += String.fromCodePoint(...scalars.slice(i, end));
  }
  return out;
}

/** Convert an NFC string to u32 code-point tokens (single pass). */
function stringToTokens(s: string): Uint32Array {
  const buf: number[] = [];
  for (const ch of s) {
    buf[buf.length] = ch.codePointAt(0) as number;
  }
  return new Uint32Array(buf);
}

/**
 * Compare two token streams for equality (manual loop, no banned helpers).
 * Used by tests to pin canonically-equivalent inputs to identical streams.
 */
export function tokensEqual(a: Uint32Array, b: Uint32Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}
