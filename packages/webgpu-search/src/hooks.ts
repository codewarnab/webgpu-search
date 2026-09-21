/**
 * Extensibility pipeline.
 *
 * Type-safe extension hooks for host applications:
 * - `tokenizer`: custom query term splitting for `'token'` mode (code
 * symbols `_`, `-`, `camelCase`). CPU-only (`'token'` is CPU-by-design).
 * Runs once per query; must be pure and deterministic.
 * - `scoringHook`: post-match boost over surviving Top-K candidates only.
 * Must return a finite integer (floats throw `TypeError`); must be pure.
 * - `filterPredicate`: conjunctive post-match predicate (truthiness-coerced).
 * Invocation order is engine-dependent; must be pure.
 * - `postProcess`: final result transformation (results-only: counts/facets/
 * suggestions unaffected; skipped on empty no-hit paths).
 *
 * Persistence safety: closures are never serialized. `serialize()` records
 * only declarative `ExtensionHookIds` (stable `hookId` property when set and
 * non-blank, else `function.name`, else `'anonymous'`); `restore` requires
 * matching handlers via `options.options.extensions` or throws
 * `IncompatibleHookError`. Name-derived IDs can collide; assign explicit
 * `hookId` for persisted hooks. Hooks cannot cross the Web Worker boundary
 * (`SearchWorkerClient` init/search/restore reject `extensions` fail-closed;
 * empty `{}` is a no-op).
 *
 * Portable: no DOM refs. Zero runtime dependencies.
 */

import { IncompatibleHookError } from './errors';
import { normalizeText } from './text-normalization';
import { isTokenDelimiter, splitQueryTerms } from './search/token-search';
import type {
  DocumentSearchResultItem,
  ExtensionHookIds,
  MatchInfo,
  SearchHooks
} from './types';

const KNOWN_HOOK_KEYS = ['tokenizer', 'scoringHook', 'filterPredicate', 'postProcess'] as const;
type HookKey = (typeof KNOWN_HOOK_KEYS)[number];

/** Options for the built-in code-aware tokenizer. */
export interface CodeTokenizerOptions {
  /**
   * Split letter<->digit boundaries (`'auth2'` -> `['auth', '2']`).
   * Default: true.
   */
  splitOnDigitBoundaries?: boolean;
  /**
   * Minimum token length to keep (shorter tokens are dropped).
   * Must be an integer >= 0. Default: 0 (keep all).
   */
  minTokenLength?: number;
}

/**
 * Default string-level tokenizer mirroring the `token` mode delimiter set.
 * Splits on the shared ASCII delimiters (`isTokenDelimiter`); `camelCase`
 * stays intact. Empty segments are dropped.
 */
export function defaultTokenizer(text: string): string[] {
  if (typeof text !== 'string') {
    throw new TypeError(`[webgpu-search] tokenizer expects text: string, got ${typeof text}.`);
  }
  const out: string[] = [];
  let start = -1;
  const chars = Array.from(text);
  const cps: number[] = chars.map((ch) => (ch.codePointAt(0) as number));
  for (let i = 0; i <= cps.length; i++) {
    const cp: number = i < cps.length ? (cps[i] as number) : -1;
    if (i === cps.length || isTokenDelimiter(cp)) {
      if (start >= 0) {
        const piece = chars.slice(start, i).join('');
        if (piece.length > 0) out.push(piece);
        start = -1;
      }
    } else if (start < 0) {
      start = i;
    }
  }
  return out;
}

function isAsciiUpper(cp: number): boolean {
  return cp >= 65 && cp <= 90;
}

function isAsciiLower(cp: number): boolean {
  return cp >= 97 && cp <= 122;
}

function isAsciiDigit(cp: number): boolean {
  return cp >= 48 && cp <= 57;
}

function charKind(ch: string, cp: number): 'upper' | 'lower' | 'digit' | 'other' {
  if (isAsciiUpper(cp)) return 'upper';
  if (isAsciiLower(cp)) return 'lower';
  if (isAsciiDigit(cp)) return 'digit';
  // Unicode-aware fallback (locale-independent, deterministic).
  // Uncased scripts (CJK, emoji) have lower === upper === ch and fall through
  // to 'other' below; cased letters land in exactly one of the branches.
  const lower = ch.toLowerCase();
  const upper = ch.toUpperCase();
  if (ch !== lower && ch === upper) return 'upper';
  if (ch !== upper && ch === lower) return 'lower';
  if (ch >= '0' && ch <= '9') return 'digit';
  return 'other';
}

/**
 * Code-aware tokenizer for symbol navigation and IDE palettes.
 *
 * Splits on the shared delimiters (`_`, `-`, whitespace, punctuation) and
 * additionally on `camelCase` / `PascalCase` / acronym boundaries:
 * - `lower->Upper`: `'UserAuth'` -> `['User', 'Auth']`.
 * - `ACRONYM->Word`: `'HTTPResponse'` -> `['HTTP', 'Response']`,
 * `'getHTTP'` -> `['get', 'HTTP']`.
 * - letter<->digit (when `splitOnDigitBoundaries`, default true):
 * `'auth2Login'` -> `['auth', '2', 'Login']`.
 *
 * Case is preserved (the index folds anyway); empty pieces are dropped.
 */
export function codeTokenizer(text: string, options?: CodeTokenizerOptions): string[] {
  if (typeof text !== 'string') {
    throw new TypeError(`[webgpu-search] codeTokenizer expects text: string, got ${typeof text}.`);
  }
  if (options !== undefined && (typeof options !== 'object' || options === null || Array.isArray(options))) {
    throw new TypeError('[webgpu-search] codeTokenizer options must be an object.');
  }
  const splitDigits = options?.splitOnDigitBoundaries ?? true;
  if (typeof splitDigits !== 'boolean') {
    throw new TypeError('[webgpu-search] codeTokenizer.splitOnDigitBoundaries must be a boolean.');
  }
  const minLen = options?.minTokenLength ?? 0;
  if (!Number.isInteger(minLen) || (minLen as number) < 0) {
    throw new RangeError(`[webgpu-search] codeTokenizer.minTokenLength must be an integer >= 0, got ${String(minLen)}.`);
  }

  const segments = defaultTokenizer(text);
  const out: string[] = [];
  for (let s = 0; s < segments.length; s++) {
    const seg = segments[s] as string;
    const chars = Array.from(seg);
    if (chars.length === 0) continue;
    const cps: number[] = chars.map((ch) => (ch.codePointAt(0) as number));
    let start = 0;
    const flush = (end: number): void => {
      if (end > start) {
        // Codepoint-aware length gate (UTF-16 `String.length` miscounts
        // astral symbols, e.g. 'a😀b'.length === 4 for 3 codepoints).
        const pieceChars = chars.slice(start, end);
        if (pieceChars.length >= (minLen as number)) out.push(pieceChars.join(''));
      }
      start = end;
    };
    for (let i = 1; i < chars.length; i++) {
      const prevCh = chars[i - 1] as string;
      const curCh = chars[i] as string;
      const prevCp = cps[i - 1] as number;
      const curCp = cps[i] as number;
      const prevK = charKind(prevCh, prevCp);
      const curK = charKind(curCh, curCp);
      let boundary = false;
      if ((prevK === 'lower' || prevK === 'digit') && curK === 'upper') {
        boundary = true;
      } else if (prevK === 'upper' && curK === 'upper' && i + 1 < chars.length) {
        // Acronym boundary: 'HTTPResponse' splits before 'R' (upper followed by lower).
        const nextCh = chars[i + 1] as string;
        const nextCp = cps[i + 1] as number;
        if (charKind(nextCh, nextCp) === 'lower') boundary = true;
      } else if (splitDigits) {
        if ((prevK === 'digit' && (curK === 'lower' || curK === 'upper')) ||
          ((prevK === 'lower' || prevK === 'upper') && curK === 'digit')) {
          boundary = true;
        }
      }
      if (boundary) flush(i);
    }
    flush(chars.length);
  }
  return out;
}

/**
 * Validate extension hooks fail-closed.
 * Returns a defensive copy with only defined hooks, or `undefined` when
 * empty. Throws `TypeError` on non-object shapes, unknown keys (typo guard),
 * or non-function hook values.
 */
export function normalizeSearchHooks<TDoc>(
  raw: SearchHooks<TDoc> | undefined
): SearchHooks<TDoc> | undefined {
  if (raw === undefined) return undefined;
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) {
    throw new TypeError('[webgpu-search] extensions must be an object.');
  }
  const record = raw as Record<string, unknown>;
  for (const key of Object.keys(record)) {
    if ((KNOWN_HOOK_KEYS as readonly string[]).indexOf(key) < 0) {
      throw new TypeError(
        `[webgpu-search] Unknown extension hook '${key}'. Expected one of 'tokenizer', 'scoringHook', 'filterPredicate', 'postProcess'.`
      );
    }
  }
  const out: SearchHooks<TDoc> = {};
  let count = 0;
  for (let i = 0; i < KNOWN_HOOK_KEYS.length; i++) {
    const key = KNOWN_HOOK_KEYS[i] as HookKey;
    const val = (raw as Record<string, unknown>)[key];
    if (val === undefined) continue;
    if (typeof val !== 'function') {
      throw new TypeError(`[webgpu-search] extensions.${key} must be a function.`);
    }
    (out as Record<string, unknown>)[key] = val;
    count++;
  }
  return count === 0 ? undefined : out;
}

/**
 * Merge index-level and per-query hooks. Per-query hooks win per key when
 * defined (documented override semantics). Both inputs are validated
 * fail-closed. Returns `undefined` when the merged set is empty.
 */
export function resolveEffectiveHooks<TDoc>(
  indexHooks: SearchHooks<TDoc> | undefined,
  queryHooks: SearchHooks<TDoc> | undefined
): SearchHooks<TDoc> | undefined {
  const normIndex = normalizeSearchHooks(indexHooks);
  const normQuery = normalizeSearchHooks(queryHooks);
  if (!normIndex && !normQuery) return undefined;
  const out: SearchHooks<TDoc> = {};
  let count = 0;
  for (let i = 0; i < KNOWN_HOOK_KEYS.length; i++) {
    const key = KNOWN_HOOK_KEYS[i] as HookKey;
    const qv = normQuery ? (normQuery as Record<string, unknown>)[key] : undefined;
    const iv = normIndex ? (normIndex as Record<string, unknown>)[key] : undefined;
    const winner = qv !== undefined ? qv : iv;
    if (winner !== undefined) {
      (out as Record<string, unknown>)[key] = winner;
      count++;
    }
  }
  return count === 0 ? undefined : out;
}

/** True when at least one hook is present. */
export function hasAnyHook<TDoc>(hooks: SearchHooks<TDoc> | undefined): boolean {
  if (!hooks) return false;
  return (
    hooks.tokenizer !== undefined ||
    hooks.scoringHook !== undefined ||
    hooks.filterPredicate !== undefined ||
    hooks.postProcess !== undefined
  );
}

/**
 * Stable hook identifier: explicit `hookId` property when set (hosts should
 * assign `myScorer.hookId = 'recency-v1'` for restore stability), else
 * `function.name`, else `'anonymous'`.
 *
 * Name-derived IDs collide across distinct functions sharing an inferred name
 * (e.g. two modules each exporting `const tokenizer = ...`, both inferring
 * `name === 'tokenizer'`) and all truly anonymous closures map to
 * `'anonymous'`. Treat `'anonymous'` as fail-open: assign an explicit
 * `hookId` whenever the hook is persisted.
 */
export function getHookId(fn: unknown): string {
  if (typeof fn !== 'function') {
    throw new TypeError('[webgpu-search] hook must be a function.');
  }
  const f = fn as { hookId?: unknown; name?: unknown };
  if (typeof f.hookId === 'string' && f.hookId.trim().length > 0) return f.hookId;
  if (typeof f.name === 'string' && f.name.length > 0) return f.name;
  return 'anonymous';
}

/** Collect declarative hook IDs for snapshot persistence (no closures). */
export function collectHookIds<TDoc>(
  hooks: SearchHooks<TDoc> | undefined
): ExtensionHookIds | undefined {
  if (!hooks || !hasAnyHook(hooks)) return undefined;
  const out: ExtensionHookIds = {};
  if (hooks.tokenizer !== undefined) out.tokenizer = getHookId(hooks.tokenizer);
  if (hooks.scoringHook !== undefined) out.scoringHook = getHookId(hooks.scoringHook);
  if (hooks.filterPredicate !== undefined) out.filterPredicate = getHookId(hooks.filterPredicate);
  if (hooks.postProcess !== undefined) out.postProcess = getHookId(hooks.postProcess);
  return Object.keys(out).length === 0 ? undefined : out;
}

/**
 * Fail-closed restore guard: snapshots recording `required` hook IDs need
 * matching handlers in `provided`. Missing handlers or ID mismatches throw
 * `IncompatibleHookError`. `undefined`/empty `required` is a no-op (backward
 * compatible with older snapshots).
 */
export function assertHooksSatisfied<TDoc>(
  required: ExtensionHookIds | undefined,
  provided: SearchHooks<TDoc> | undefined
): void {
  if (!required || Object.keys(required).length === 0) return;
  const providedIds = collectHookIds(provided);
  for (let i = 0; i < KNOWN_HOOK_KEYS.length; i++) {
    const key = KNOWN_HOOK_KEYS[i] as HookKey;
    const need = (required as Record<string, unknown>)[key] as string | undefined;
    if (need === undefined) continue;
    const haveFn = provided ? (provided as Record<string, unknown>)[key] : undefined;
    if (typeof haveFn !== 'function') {
      throw new IncompatibleHookError(
        String(need),
        `Snapshot requires "${key}" hook "${String(need)}" but no matching handler was supplied (pass via options.options.extensions).`
      );
    }
    const haveId = providedIds ? (providedIds as Record<string, unknown>)[key] as string | undefined : undefined;
    if (haveId !== undefined && haveId !== need) {
      throw new IncompatibleHookError(
        String(need),
        `Snapshot requires "${key}" hook "${String(need)}" but handler "${haveId}" was supplied (mismatched hookId).`
      );
    }
  }
}

/**
 * Invoke a custom tokenizer fail-closed. Returns filtered non-empty strings.
 * Throws `TypeError` when the hook returns a non-array or non-string terms.
 * Hook exceptions propagate to the caller.
 */
export function tokenizeWithHook(
  text: string,
  tokenizer: (text: string) => string[]
): string[] {
  if (typeof tokenizer !== 'function') {
    throw new TypeError('[webgpu-search] tokenizer must be a function.');
  }
  const raw = tokenizer(text);
  if (!Array.isArray(raw)) {
    throw new TypeError('[webgpu-search] tokenizer must return a string[].');
  }
  const out: string[] = [];
  for (let i = 0; i < raw.length; i++) {
    const term = raw[i];
    if (typeof term !== 'string') {
      throw new TypeError('[webgpu-search] tokenizer must return a string[].');
    }
    if (term.length > 0) out.push(term);
  }
  return out;
}

/**
 * Resolve `'token'` mode query terms: custom tokenizer path (string-level
 * hook over the raw query, each term normalized post-normalization) or the default
 * `splitQueryTerms` path over post-normalization code points. Empty hook output yields
 * zero terms (no-match, mirroring all-delimiter queries).
 */
export function getTokenTermsForQuery(
  queryString: string,
  queryTokens: Uint32Array,
  normalized: boolean,
  tokenizer?: (text: string) => string[]
): Uint32Array[] {
  if (tokenizer === undefined) return splitQueryTerms(queryTokens);
  const terms = tokenizeWithHook(queryString, tokenizer);
  const out: Uint32Array[] = [];
  for (let i = 0; i < terms.length; i++) {
    const norm = normalizeText(terms[i] as string, normalized);
    if (!norm.isEmpty && norm.tokens.length > 0) out.push(norm.tokens);
  }
  return out;
}

/**
 * Apply a scoring hook post-match over surviving Top-K results only.
 * Mutates `results` scores in place. Each call receives
 * `(doc, baseScore, { query, matchedField, rawScore, normalizedScore })`
 * with `rawScore === normalizedScore === baseScore`. Returns must be finite
 * integers (the unified `SearchResultItem` contract is descending normalized
 * integer scores; floats throw `TypeError` fail-closed). All hooks must be
 * pure and deterministic: `filterPredicate` runs in engine-dependent order
 * (GPU result order vs exact row order), so non-deterministic hooks diverge
 * across engines.
 * Hook exceptions propagate; non-finite / non-integer returns throw `TypeError`.
 */
export function applyScoringHook<TDoc>(
  results: DocumentSearchResultItem<TDoc>[],
  hook: ((doc: TDoc, baseScore: number, matchInfo: MatchInfo) => number) | undefined,
  query: string
): void {
  if (hook === undefined) return;
  if (typeof hook !== 'function') {
    throw new TypeError('[webgpu-search] extensions.scoringHook must be a function.');
  }
  for (let i = 0; i < results.length; i++) {
    const item = results[i] as DocumentSearchResultItem<TDoc>;
    const baseScore = item.score;
    const matchInfo: MatchInfo = {
      query,
      matchedField: item.matchedField,
      rawScore: baseScore,
      normalizedScore: baseScore
    };
    const next = (hook as (doc: TDoc, baseScore: number, matchInfo: MatchInfo) => number)(
      item.doc,
      baseScore,
      matchInfo
    );
    if (typeof next !== 'number' || !Number.isFinite(next) || !Number.isInteger(next)) {
      throw new TypeError('[webgpu-search] extensions.scoringHook must return a finite integer.');
    }
    item.score = next;
  }
}

/**
 * Apply a post-processing hook as the final result transformation.
 * Returns the hook output (may reorder, filter, or augment). Throws
 * `TypeError` when the hook returns a non-array. Hook exceptions propagate.
 *
 * `postProcess` affects only `results`: `totalMatches` / `candidateCount` /
 * `hasOverflow` are snapshotted pre-pipeline, and `facets` / `suggestions`
 * ignore it (suggestions are index-wide by design). On empty no-hit paths
 * (empty query / corpus / empty filter) `postProcess` never runs.
 */
export function applyPostProcess<TDoc>(
  results: DocumentSearchResultItem<TDoc>[],
  hook:
    | ((results: DocumentSearchResultItem<TDoc>[]) => DocumentSearchResultItem<TDoc>[])
    | undefined
): DocumentSearchResultItem<TDoc>[] {
  if (hook === undefined) return results;
  if (typeof hook !== 'function') {
    throw new TypeError('[webgpu-search] extensions.postProcess must be a function.');
  }
  const next = (
    hook as (results: DocumentSearchResultItem<TDoc>[]) => DocumentSearchResultItem<TDoc>[]
  )(results);
  if (!Array.isArray(next)) {
    throw new TypeError('[webgpu-search] extensions.postProcess must return an array.');
  }
  return next as DocumentSearchResultItem<TDoc>[];
}

/** @deprecated Use normalizeSearchHooks. */
export const normalizeSearchExtensionHooks = normalizeSearchHooks;
