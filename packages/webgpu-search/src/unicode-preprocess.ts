/**
 * v0.2 shared Unicode preprocessing (Issue #7 M2).
 *
 * Single pipeline for records + queries (contract §1):
 *   trim → toWellFormed (or lone-surrogate regex fallback → U+FFFD)
 *   → NFC → C+F fold (when folded) → NFC → u32 scalar stream.
 *
 * Size and empty checks apply to the post-fold token count.
 * Scores, spans and penalties are measured in post-fold code points;
 * folded coordinates must NOT be used to slice the original string
 * (see "Strasse" caveat in docs/unicode-contract.md).
 *
 * Portable: no DOM refs. Parity path avoids locale-sensitive case
 * conversion and UTF-16 unit access (see lint script).
 */

import { foldCodePoint } from './fold-table';

/** Lone-surrogate fallback: each lone surrogate → exactly one U+FFFD. */
export const LONE_SURROGATE_PATTERN: RegExp =
  /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g;

/** Internal fallback instance (not exported, so external lastIndex use cannot pollute the path). */
const FALLBACK_RE_G: RegExp =
  /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/g;

const REPLACEMENT_CHAR = '�';

/**
 * ES2024 `toWellFormed` where available, else per-surrogate regex fallback.
 * Semantics: high+high = 2 FFFDs, high+EOF = 1, low-without-high = 1.
 * Genuine U+FFFD is indistinguishable by design (documented).
 */
export function toWellFormedSafe(input: string): string {
  const s: string = input ?? '';
  const maybe = s as unknown as { toWellFormed?: unknown };
  if (typeof maybe.toWellFormed === 'function') {
    return (maybe.toWellFormed as () => string).call(s);
  }
  FALLBACK_RE_G.lastIndex = 0;
  return s.replace(FALLBACK_RE_G, REPLACEMENT_CHAR);
}

export interface NormalizedText {
  /** Post-fold NFC scalar stream (u32 code points). */
  tokens: Uint32Array;
  /** Post-fold token count (tokens.length). */
  tokenCount: number;
  /** True when tokenCount === 0 (degenerate post-processing query). */
  isEmpty: boolean;
  /** Whether C+F folding was applied (false = NFC-only). */
  folded: boolean;
}

/** Append one scalar to a growing number array without banned helpers. */
function pushScalar(out: number[], cp: number): void {
  out[out.length] = cp;
}

/**
 * Normalize one record or query to post-fold u32 tokens.
 *
 * @param raw raw JS string (records and queries share this path).
 * @param folded true = NFC + C+F fold + NFC; false = NFC-only.
 */
export function normalizeText(raw: string, folded: boolean): NormalizedText {
  const trimmed: string = (raw ?? '').trim();
  if (trimmed.length === 0) {
    return { tokens: new Uint32Array(0), tokenCount: 0, isEmpty: true, folded };
  }
  const wellFormed: string = toWellFormedSafe(trimmed);
  const nfc1: string = wellFormed.normalize('NFC');

  let foldedScalars: number[];
  if (folded) {
    foldedScalars = [];
    // Iterate by code point (astral counts as one). No UTF-16 unit access.
    for (const ch of nfc1) {
      const cp: number = ch.codePointAt(0) as number;
      const mapped: readonly number[] | null = foldCodePoint(cp);
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
    return { tokens, tokenCount: tokens.length, isEmpty: tokens.length === 0, folded };
  }

  const tokens: Uint32Array = stringToTokens(nfc1);
  return { tokens, tokenCount: tokens.length, isEmpty: tokens.length === 0, folded };
}

/** Build a JS string from scalars in bounded chunks (huge-vector safe). */
function fromCodePointsChunked(scalars: number[]): string {
  const CHUNK = 8192;
  let out = '';
  for (let i = 0; i < scalars.length; i += CHUNK) {
    const end: number = i + CHUNK < scalars.length ? i + CHUNK : scalars.length;
    let part = '';
    for (let j = i; j < end; j++) {
      part += String.fromCodePoint(scalars[j] as number);
    }
    out += part;
  }
  return out;
}

/** Convert an NFC string to u32 code-point tokens. */
function stringToTokens(s: string): Uint32Array {
  // First pass: count (avoids growth realloc for large vectors).
  let count = 0;
  for (const _ch of s) count++;
  const out = new Uint32Array(count);
  let i = 0;
  for (const ch of s) {
    out[i++] = ch.codePointAt(0) as number;
  }
  return out;
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
