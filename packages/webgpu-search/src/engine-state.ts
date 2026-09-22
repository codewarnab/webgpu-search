import type { EngineType, FallbackReason } from './types';

/**
 * Explicit engine state machine.
 *
 * Replaces the `engineType` + `fallbackReason` string soup previously
 * scattered across `SearchIndex` and `DocumentIndex` with a single
 * discriminated union plus a validated transition helper.
 *
 * States:
 * - `{ engine: 'cpu' }` — initial / destroyed (no reason yet).
 * - `{ engine: 'cpu', fallbackReason }` — CPU fallback with an explicit
 *   reason (`below-threshold`, `prefer-cpu`, `memory-budget-exceeded`,
 *   `webgpu-unsupported`, `device-request-failed`, `device-lost`,
 *   `gpu-execution-error`).
 * - `{ engine: 'webgpu' }` — GPU active (never carries a fallbackReason).
 *
 * All `device-lost`, `gpu-error`, and `below-threshold` paths must route
 * through {@link transitionEngineState} so invalid combinations
 * (e.g. `webgpu` + a fallback reason, unknown reason strings) fail closed.
 *
 * Portable: no DOM / `navigator` / `window` references — safe in
 * Web Workers, Node.js, and SSR.
 */
export type EngineState =
  | { engine: 'webgpu'; fallbackReason?: undefined }
  | { engine: 'cpu'; fallbackReason?: FallbackReason };

const VALID_FALLBACK_REASONS: ReadonlySet<FallbackReason> = new Set<FallbackReason>([
  'webgpu-unsupported',
  'device-request-failed',
  'memory-budget-exceeded',
  'device-lost',
  'below-threshold',
  'prefer-cpu',
  'query-too-long',
  'cpu-algorithm-requested',
  'unsupported-mode',
  'gpu-execution-error',
]);

/** Initial state for freshly constructed indexes (CPU, no reason yet). */
export function initialEngineState(): EngineState {
  return { engine: 'cpu' };
}

/**
 * Validated engine-state transition.
 *
 * @param current - current state (validated fail-closed).
 * @param engine - target engine (`'webgpu'` | `'cpu'`).
 * @param fallbackReason - required for most CPU fallbacks; must be
 *   `undefined` when targeting `'webgpu'`.
 * @returns the next immutable {@link EngineState}.
 * @throws on unknown engines, unknown reasons, or `webgpu` + reason.
 */
export function transitionEngineState(
  current: EngineState,
  engine: EngineType,
  fallbackReason?: FallbackReason,
): EngineState {
  if (current === null || typeof current !== 'object') {
    throw new Error('[webgpu-search] Invalid engine state: expected an EngineState object.');
  }
  if (current.engine !== 'webgpu' && current.engine !== 'cpu') {
    throw new Error(
      `[webgpu-search] Invalid current engine state: "${String((current as { engine?: unknown }).engine)}".`,
    );
  }
  if (engine !== 'webgpu' && engine !== 'cpu') {
    throw new Error(`[webgpu-search] Invalid target engine: "${String(engine)}".`);
  }
  if (fallbackReason !== undefined && !VALID_FALLBACK_REASONS.has(fallbackReason)) {
    throw new Error(`[webgpu-search] Invalid fallback reason: "${String(fallbackReason)}".`);
  }
  if (engine === 'webgpu') {
    if (fallbackReason !== undefined) {
      throw new Error('[webgpu-search] Invalid engine transition: webgpu must not carry a fallbackReason.');
    }
    return { engine: 'webgpu' };
  }
  if (fallbackReason === undefined) {
    return { engine: 'cpu' };
  }
  return { engine: 'cpu', fallbackReason };
}

/** Type-narrowing helper: true when the state is GPU-active. */
export function isWebGpuState(state: EngineState): boolean {
  return state.engine === 'webgpu';
}
