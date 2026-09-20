/**
 * High-speed structured filter compiler and bitset evaluator.
 * Evaluates FilterExpression AST into DocumentBitset via ColumnarStore.
 * 100% portable across browser main thread, Web Workers, Node.js, and SSR (zero DOM references).
 */

import { DocumentBitset } from './bitset';
import { ColumnarStore } from './columnar-store';
import type { FilterExpression, FieldComparison } from '../types';
import { InvalidFilterError } from '../errors';

const VALID_COMPARISON_OPERATORS = new Set([
  'eq',
  'neq',
  'gt',
  'gte',
  'lt',
  'lte',
  'in',
  'nin',
  'exists'
]);

/**
 * Compiles and evaluates a structured FilterExpression against a ColumnarStore.
 * Returns a DocumentBitset where set bits correspond to matching active document indices.
 */
export function compileFilter<TDoc = any>(
  expression: FilterExpression,
  store: ColumnarStore<TDoc>,
  activeMask?: DocumentBitset
): DocumentBitset {
  if (expression === null || typeof expression !== 'object') {
    throw new InvalidFilterError('FilterExpression must be a non-null object');
  }

  const keys = Object.keys(expression);

  // Vacuous truth: empty filter expression {} matches all active documents
  if (keys.length === 0) {
    return (activeMask ?? store.getActiveDocs()).clone();
  }

  let accumulatedResult: DocumentBitset | null = null;

  for (let k = 0; k < keys.length; k++) {
    const key = keys[k];
    const val = (expression as any)[key];
    const currentMask = accumulatedResult ?? activeMask;
    let keyBitset: DocumentBitset;

    if (key === 'and') {
      if (!Array.isArray(val)) {
        throw new InvalidFilterError('"and" operator expects an array of FilterExpressions');
      }
      // Vacuous truth for { and: [] }
      if (val.length === 0) {
        keyBitset = (currentMask ?? store.getActiveDocs()).clone();
      } else {
        keyBitset = compileFilter(val[0], store, currentMask);
        for (let i = 1; i < val.length; i++) {
          if (keyBitset.isEmpty()) break;
          keyBitset = compileFilter(val[i], store, keyBitset);
        }
      }
    } else if (key === 'or') {
      if (!Array.isArray(val)) {
        throw new InvalidFilterError('"or" operator expects an array of FilterExpressions');
      }
      // Vacuous falsehood for { or: [] }
      if (val.length === 0) {
        keyBitset = DocumentBitset.none(store.capacity);
      } else {
        keyBitset = compileFilter(val[0], store);
        for (let i = 1; i < val.length; i++) {
          const next = compileFilter(val[i], store);
          keyBitset.orInPlace(next);
        }
        if (currentMask) {
          keyBitset.andInPlace(currentMask);
        }
      }
    } else if (key === 'not') {
      if (val === null || typeof val !== 'object' || Array.isArray(val)) {
        throw new InvalidFilterError('"not" operator expects a FilterExpression object');
      }
      const inner = compileFilter(val, store);
      keyBitset = (currentMask ?? store.getActiveDocs()).andNot(inner);
    } else {
      // Field filter condition
      const fieldName = key;
      if (!store.hasField(fieldName)) {
        const available = store.getFieldNames();
        const availableDesc = available.length > 0
          ? `Configured fields: ${available.map((f) => `"${f}"`).join(', ')}.`
          : 'No filterFields were configured on this index.';
        throw new InvalidFilterError(
          `Unknown filter field "${fieldName}". ${availableDesc}`,
          fieldName
        );
      }

      keyBitset = evaluateFieldCondition(fieldName, val, store, currentMask);
    }

    accumulatedResult = keyBitset;
    if (accumulatedResult.isEmpty()) {
      break;
    }
  }

  return accumulatedResult ?? (activeMask ?? store.getActiveDocs()).clone();
}

function evaluateFieldCondition<TDoc = any>(
  fieldName: string,
  condition: unknown,
  store: ColumnarStore<TDoc>,
  activeMask?: DocumentBitset
): DocumentBitset {
  const active = activeMask ?? store.getActiveDocs();

  // Case A: Direct literal primitive (string, number, boolean, null)
  if (
    typeof condition === 'string' ||
    typeof condition === 'number' ||
    typeof condition === 'boolean' ||
    condition === null
  ) {
    return store.evaluateEquality(fieldName, condition, active);
  }

  // Case B: Direct array of values -> implicit 'in'
  if (Array.isArray(condition)) {
    return store.evaluateIn(fieldName, condition, active);
  }

  // Case C: FieldComparison object
  if (typeof condition === 'object') {
    const comp = condition as FieldComparison;
    const compKeys = Object.keys(comp);

    if (compKeys.length === 0) {
      return active.clone();
    }

    // Validate comparison operators
    for (let i = 0; i < compKeys.length; i++) {
      const op = compKeys[i];
      if (!VALID_COMPARISON_OPERATORS.has(op)) {
        throw new InvalidFilterError(
          `Unsupported filter operator: "${op}" on field "${fieldName}". Supported operators: ${Array.from(VALID_COMPARISON_OPERATORS).join(', ')}.`,
          fieldName
        );
      }
    }

    let compResult: DocumentBitset | null = null;
    const currentActive = () => compResult ?? active;

    // 1. exists
    if ('exists' in comp) {
      const existsVal = Boolean(comp.exists);
      compResult = store.evaluateExists(fieldName, existsVal, currentActive());
    }

    // 2. eq
    if ('eq' in comp) {
      compResult = store.evaluateEquality(fieldName, comp.eq!, currentActive());
    }

    // 3. neq
    if ('neq' in comp) {
      const eqBs = store.evaluateEquality(fieldName, comp.neq!);
      compResult = currentActive().andNot(eqBs);
    }

    // 4. in
    if ('in' in comp) {
      if (!Array.isArray(comp.in)) {
        throw new InvalidFilterError('"in" operator expects an array of values', fieldName);
      }
      compResult = store.evaluateIn(fieldName, comp.in, currentActive());
    }

    // 5. nin
    if ('nin' in comp) {
      if (!Array.isArray(comp.nin)) {
        throw new InvalidFilterError('"nin" operator expects an array of values', fieldName);
      }
      const inBs = store.evaluateIn(fieldName, comp.nin);
      compResult = currentActive().andNot(inBs);
    }

    // 6. Range operators (gt, gte, lt, lte)
    if ('gt' in comp || 'gte' in comp || 'lt' in comp || 'lte' in comp) {
      compResult = store.evaluateRange(fieldName, comp, currentActive());
    }

    return compResult ?? active.clone();
  }

  throw new InvalidFilterError(
    `Invalid filter condition on field "${fieldName}": expected literal value, array, or FieldComparison object`,
    fieldName
  );
}
