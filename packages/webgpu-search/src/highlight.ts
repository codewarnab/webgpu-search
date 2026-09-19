import type { HighlightRange, SearchMode } from './types';
import { toWellFormedSafe, normalizeText } from './unicode-preprocess';
import { foldCodePoint } from './fold-table';
import { scoreSubstringTokens } from './cpu-reference';
import { IncompatibleOptionError } from './text-profile';

export interface SourceMappedText {
  /** Post-fold NFC scalar stream (u32 code points). */
  tokens: Uint32Array;
  /** Post-fold token count. */
  tokenCount: number;
  /** True when tokenCount === 0. */
  isEmpty: boolean;
  /** Whether C+F folding was applied. */
  folded: boolean;
  /** Start index in UTF-16 code units in the original JS string for each token. */
  starts: Uint32Array;
  /** End index (exclusive) in UTF-16 code units in the original JS string for each token. */
  ends: Uint32Array;
  /** Number of leading whitespace code units trimmed from raw string. */
  leadingTrimOffset: number;
}

export interface AlignHighlightOptions {
  /** Search mode ('fuzzy' or 'substring', default: 'fuzzy') */
  mode?: SearchMode;
  /** If true, folding is skipped (caseSensitive). Default: false */
  caseSensitive?: boolean;
  /** Alternative to caseSensitive (folded = !caseSensitive). */
  folded?: boolean;
  /** Optional pre-computed source map for raw string. */
  sourceMap?: SourceMappedText;
  /** Optional pre-computed query tokens. */
  queryTokens?: Uint32Array;
}

export interface RenderHighlightOptions {
  /** HTML/formatting tag for snippets (default: 'mark') */
  tag?: string;
  /** Whether to HTML-escape special characters in raw string slices (default: false) */
  escapeHtml?: boolean;
}

const COMBINING_MARK_REGEX = /^\p{M}/u;
const EXTEND_OR_JOINER_REGEX = /^[\p{M}\p{Sk}\u200D\uFE0F]/u;
const JAMO_V_OR_T_REGEX = /^[\u1160-\u11FF\uD7B0-\uD7FB]/u;

let cachedSegmenter: Intl.Segmenter | null = null;
function getSegmenter(): Intl.Segmenter | null {
  if (cachedSegmenter !== null) return cachedSegmenter;
  if (typeof Intl !== 'undefined' && typeof (Intl as any).Segmenter === 'function') {
    try {
      cachedSegmenter = new Intl.Segmenter('en', { granularity: 'grapheme' });
      return cachedSegmenter;
    } catch {
      // Fall through to manual fallback
    }
  }
  return null;
}

function isAsciiOnly(s: string): boolean {
  for (let i = 0; i < s.length; i++) {
    if ((s.charCodeAt(i) as number) > 0x7f) return false;
  }
  return true;
}

function fromCodePointsChunked(scalars: number[]): string {
  const CHUNK = 8192;
  let out = '';
  for (let i = 0; i < scalars.length; i += CHUNK) {
    const end = i + CHUNK < scalars.length ? i + CHUNK : scalars.length;
    out += String.fromCodePoint(...scalars.slice(i, end));
  }
  return out;
}

function appendSegmentTokens(
  segStr: string,
  folded: boolean,
  spanStart: number,
  spanEnd: number,
  allTokens: number[],
  allStarts: number[],
  allEnds: number[]
): void {
  const nfc1 = segStr.normalize('NFC');
  if (!folded) {
    for (const ch of nfc1) {
      allTokens.push(ch.codePointAt(0) as number);
      allStarts.push(spanStart);
      allEnds.push(spanEnd);
    }
    return;
  }
  const foldedScalars: number[] = [];
  for (const ch of nfc1) {
    const cp = ch.codePointAt(0) as number;
    const mapped = foldCodePoint(cp);
    if (mapped === null) {
      foldedScalars.push(cp);
    } else {
      for (let k = 0; k < mapped.length; k++) {
        foldedScalars.push(mapped[k] as number);
      }
    }
  }
  const foldedStr = fromCodePointsChunked(foldedScalars);
  const nfc2 = foldedStr.normalize('NFC');
  for (const ch of nfc2) {
    allTokens.push(ch.codePointAt(0) as number);
    allStarts.push(spanStart);
    allEnds.push(spanEnd);
  }
}

/**
 * Normalizes raw string to post-fold tokens with an exact coordinate source map
 * linking every token to its UTF-16 code unit span in the original raw string.
 *
 * Accounts for:
 * 1. Leading indentation whitespace (`leadingTrimOffset`).
 * 2. Astral surrogate pairs (emojis) mapped atomically to 2 UTF-16 units.
 * 3. Lone surrogates replaced safely with U+FFFD (1 code unit).
 * 4. NFC composition (combining marks compose into base starter).
 * 5. Full C+F case folding expansions ('ß' -> 'ss', 'İ' -> 'i' + dot, 'ﬁ' -> 'fi').
 *    Every expanded token inherits the full original span of the atomic character.
 * 6. Grapheme cluster boundary protection (no broken multi-code-point sequences).
 */
export function normalizeWithSourceMap(raw: string, folded: boolean): SourceMappedText {
  if (typeof raw !== 'string') {
    throw new TypeError(`[webgpu-search] normalizeWithSourceMap expects a string, got ${typeof raw}`);
  }

  const leadingTrimOffset = raw.length - raw.trimStart().length;
  const trimmed = raw.trim();

  if (trimmed.length === 0) {
    return {
      tokens: new Uint32Array(0),
      tokenCount: 0,
      isEmpty: true,
      folded,
      starts: new Uint32Array(0),
      ends: new Uint32Array(0),
      leadingTrimOffset
    };
  }

  // ASCII fast path: 1 token per char, 1:1 UTF-16 span
  if (isAsciiOnly(trimmed)) {
    const len = trimmed.length;
    const tokens = new Uint32Array(len);
    const starts = new Uint32Array(len);
    const ends = new Uint32Array(len);

    for (let i = 0; i < len; i++) {
      const cu = trimmed.charCodeAt(i);
      tokens[i] = folded && cu >= 0x41 && cu <= 0x5a ? cu + 32 : cu;
      starts[i] = leadingTrimOffset + i;
      ends[i] = leadingTrimOffset + i + 1;
    }

    return {
      tokens,
      tokenCount: len,
      isEmpty: false,
      folded,
      starts,
      ends,
      leadingTrimOffset
    };
  }

  const wellFormed = toWellFormedSafe(trimmed);
  const allTokens: number[] = [];
  const allStarts: number[] = [];
  const allEnds: number[] = [];

  const segmenter = getSegmenter();

  if (segmenter !== null) {
    for (const seg of segmenter.segment(wellFormed)) {
      const segStr = seg.segment;
      const spanStart = leadingTrimOffset + seg.index;
      const spanEnd = spanStart + segStr.length;
      appendSegmentTokens(segStr, folded, spanStart, spanEnd, allTokens, allStarts, allEnds);
    }
  } else {
    // Fallback grapheme iteration for environments without Intl.Segmenter
    let i = 0;
    const wfLen = wellFormed.length;
    while (i < wfLen) {
      const segStart = i;
      const cp = wellFormed.codePointAt(i) as number;
      i += cp > 0xffff ? 2 : 1;

      // Extend across following combining marks, modifiers, joiners, and Jamo
      while (i < wfLen) {
        const nextCp = wellFormed.codePointAt(i) as number;
        const nextLen = nextCp > 0xffff ? 2 : 1;
        const slice = wellFormed.slice(i, i + nextLen);
        if (
          COMBINING_MARK_REGEX.test(slice) ||
          EXTEND_OR_JOINER_REGEX.test(slice) ||
          JAMO_V_OR_T_REGEX.test(slice)
        ) {
          i += nextLen;
        } else {
          break;
        }
      }

      const segStr = wellFormed.slice(segStart, i);
      const spanStart = leadingTrimOffset + segStart;
      const spanEnd = leadingTrimOffset + i;
      appendSegmentTokens(segStr, folded, spanStart, spanEnd, allTokens, allStarts, allEnds);
    }
  }

  const tokenCount = allTokens.length;
  return {
    tokens: new Uint32Array(allTokens),
    tokenCount,
    isEmpty: tokenCount === 0,
    folded,
    starts: new Uint32Array(allStarts),
    ends: new Uint32Array(allEnds),
    leadingTrimOffset
  };
}

/**
 * Extends end offset across any trailing combining marks, skin tone modifiers,
 * or cluster joiners so highlight ranges never truncate a grapheme cluster.
 */
export function guardClusterBoundary(raw: string, end: number): number {
  const len = raw.length;
  if (end <= 0) return 0;
  if (end >= len) return len;

  let cur = end;

  // Prevent splitting surrogate pairs (0xd800..0xdbff followed by 0xdc00..0xdfff)
  const prevCu = raw.charCodeAt(cur - 1);
  const curCu = raw.charCodeAt(cur);
  if (prevCu >= 0xd800 && prevCu <= 0xdbff && curCu >= 0xdc00 && curCu <= 0xdfff) {
    cur++;
  }

  const segmenter = getSegmenter();
  if (segmenter !== null) {
    for (const seg of segmenter.segment(raw)) {
      const segStart = seg.index;
      const segEnd = segStart + seg.segment.length;
      if (cur > segStart && cur < segEnd) {
        return segEnd;
      }
      if (segStart >= cur) break;
    }
    return cur;
  }

  // Regex fallback for environments without Intl.Segmenter
  while (cur < len) {
    const cp = raw.codePointAt(cur);
    if (cp === undefined || cp < 0x0300) break;
    const charLen = cp > 0xffff ? 2 : 1;
    const slice = raw.slice(cur, cur + charLen);
    if (EXTEND_OR_JOINER_REGEX.test(slice) || JAMO_V_OR_T_REGEX.test(slice)) {
      cur += charLen;
    } else {
      break;
    }
  }
  return cur;
}

/**
 * Merges overlapping or contiguous highlight ranges into minimal non-overlapping slices.
 * Clamps negative numbers and normalizes inverted ranges.
 */
export function mergeHighlightRanges(ranges: readonly HighlightRange[]): HighlightRange[] {
  if (ranges.length === 0) return [];

  const normalized: HighlightRange[] = [];
  for (let i = 0; i < ranges.length; i++) {
    const r = ranges[i];
    if (r && typeof r.start === 'number' && typeof r.end === 'number') {
      const s = Math.max(0, Math.min(r.start, r.end));
      const e = Math.max(0, Math.max(r.start, r.end));
      if (e > s) {
        normalized.push({ start: s, end: e });
      }
    }
  }
  if (normalized.length === 0) return [];
  if (normalized.length === 1) return [{ start: normalized[0].start, end: normalized[0].end }];

  normalized.sort((a, b) => {
    if (a.start !== b.start) return a.start - b.start;
    return a.end - b.end;
  });

  const merged: HighlightRange[] = [];
  let curStart = normalized[0].start;
  let curEnd = normalized[0].end;

  for (let i = 1; i < normalized.length; i++) {
    const next = normalized[i];
    if (next.start <= curEnd) {
      if (next.end > curEnd) {
        curEnd = next.end;
      }
    } else {
      merged.push({ start: curStart, end: curEnd });
      curStart = next.start;
      curEnd = next.end;
    }
  }
  merged.push({ start: curStart, end: curEnd });
  return merged;
}

/**
 * Computes exact highlight ranges in UTF-16 code units of raw JS string.
 * Highlighting is top-K demand-driven and preserves score-highlight symmetry.
 */
export function alignHighlights(
  raw: string,
  query: string,
  options: AlignHighlightOptions = {}
): HighlightRange[] {
  if (typeof raw !== 'string' || typeof query !== 'string') return [];
  if (raw.length === 0 || query.length === 0) return [];

  const folded = options.folded !== undefined
    ? options.folded
    : options.caseSensitive !== undefined
      ? !options.caseSensitive
      : (options.sourceMap ? options.sourceMap.folded : true);

  if (options.sourceMap && options.folded !== undefined && options.sourceMap.folded !== options.folded) {
    throw new IncompatibleOptionError(
      'folded',
      `[webgpu-search] alignHighlights sourceMap.folded (${options.sourceMap.folded}) diverges from requested folded mode (${options.folded}).`
    );
  }

  const sourceMap = options.sourceMap ?? normalizeWithSourceMap(raw, folded);
  if (sourceMap.isEmpty) return [];

  const queryTokens = options.queryTokens ?? normalizeText(query, folded).tokens;
  const qLen = queryTokens.length;
  if (qLen === 0 || sourceMap.tokenCount < qLen) return [];

  const mode = options.mode ?? 'fuzzy';

  if (mode === 'substring') {
    const match = scoreSubstringTokens(sourceMap.tokens, queryTokens);
    if (!match.matched || match.matchStart < 0) return [];

    const start = sourceMap.starts[match.matchStart];
    const rawEnd = sourceMap.ends[match.matchStart + qLen - 1];
    const end = guardClusterBoundary(raw, rawEnd);
    return [{ start, end }];
  }

  // Fuzzy mode: greedy forward subsequence matching (identical to scoreFuzzyTokens & fuzzy.wgsl)
  let q = 0;
  const strLen = sourceMap.tokenCount;
  const rawRanges: HighlightRange[] = [];

  for (let i = 0; i < strLen; i++) {
    if (sourceMap.tokens[i] === queryTokens[q]) {
      const s = sourceMap.starts[i];
      const e = guardClusterBoundary(raw, sourceMap.ends[i]);
      rawRanges.push({ start: s, end: e });
      q++;
      if (q === qLen) break;
    }
  }

  if (q !== qLen) return [];
  return mergeHighlightRanges(rawRanges);
}

function escapeHtmlEntities(str: string): string {
  return str.replace(/[&<>"']/g, (m) => {
    switch (m) {
      case '&': return '&amp;';
      case '<': return '&lt;';
      case '>': return '&gt;';
      case '"': return '&quot;';
      case "'": return '&#39;';
      default: return m;
    }
  });
}

/**
 * Formats a raw string with HTML tags injected at the specified highlight ranges.
 * If ranges is empty or raw is empty, returns raw (or escaped raw if escapeHtml is true).
 */
export function renderHighlightedText(
  raw: string,
  ranges: readonly HighlightRange[],
  tagOrOptions: string | RenderHighlightOptions = 'mark',
  escapeHtmlOption?: boolean
): string {
  if (!raw) return '';

  const tag = typeof tagOrOptions === 'string' ? tagOrOptions : (tagOrOptions?.tag ?? 'mark');
  const shouldEscape = typeof tagOrOptions === 'object' && tagOrOptions !== null
    ? (tagOrOptions.escapeHtml ?? escapeHtmlOption ?? false)
    : (escapeHtmlOption ?? false);

  if (ranges.length === 0) {
    return shouldEscape ? escapeHtmlEntities(raw) : raw;
  }

  const trimmedTag = tag.trim();
  const tagMatch = /^([a-zA-Z][a-zA-Z0-9_-]*)/.exec(trimmedTag);
  if (!tagMatch) {
    throw new TypeError(`[webgpu-search] Invalid HTML tag: "${tag}". Must be a valid HTML element tag.`);
  }
  const tagName = tagMatch[1];
  const openTag = `<${trimmedTag}>`;
  const closeTag = `</${tagName}>`;

  const merged = mergeHighlightRanges(ranges);
  let out = '';
  let lastIdx = 0;

  for (let i = 0; i < merged.length; i++) {
    const { start, end } = merged[i];
    const clampedStart = Math.max(0, Math.min(start, raw.length));
    const clampedEnd = Math.max(clampedStart, Math.min(end, raw.length));

    if (clampedStart > lastIdx) {
      const slice = raw.slice(lastIdx, clampedStart);
      out += shouldEscape ? escapeHtmlEntities(slice) : slice;
    }
    if (clampedEnd > clampedStart) {
      const slice = raw.slice(clampedStart, clampedEnd);
      out += openTag + (shouldEscape ? escapeHtmlEntities(slice) : slice) + closeTag;
    }
    lastIdx = clampedEnd;
  }

  if (lastIdx < raw.length) {
    const slice = raw.slice(lastIdx);
    out += shouldEscape ? escapeHtmlEntities(slice) : slice;
  }

  return out;
}
