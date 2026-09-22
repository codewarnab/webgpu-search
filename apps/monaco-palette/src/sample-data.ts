import type { MonacoFileRecord, SymbolKind } from './types';

export const CORE_FILES: MonacoFileRecord[] = [
  {
    id: 'f-001',
    path: 'packages/webgpu-search/src/document-index.ts',
    filename: 'document-index.ts',
    symbols: 'DocumentIndex, packStratifiedRows, applyBatch, search, compactTombstones',
    type: 'class',
    language: 'typescript',
    description: 'High-level multi-field document indexing engine with field-stratified packing and dynamic mutations',
    sizeBytes: 34200,
    lineCount: 940
  },
  {
    id: 'f-002',
    path: 'packages/webgpu-search/src/webgpu-engine.ts',
    filename: 'webgpu-engine.ts',
    symbols: 'WebGPUEngine, setupPipelinesAndBuffers, searchCold, executeComputePass, dispatchWorkgroups',
    type: 'class',
    language: 'typescript',
    description: 'Low-level WebGPU compute pipeline manager, uniform binding, and candidate readback',
    sizeBytes: 28150,
    lineCount: 780
  },
  {
    id: 'f-003',
    path: 'packages/webgpu-search/src/shaders/fuzzy.wgsl',
    filename: 'fuzzy.wgsl',
    symbols: 'FuzzyUniforms, computeFuzzyScore, main, Match, OutBuf',
    type: 'shader',
    language: 'wgsl',
    description: 'WGSL compute shader for subsequence matching, leading bonus, consecutive bonus, and distance penalties',
    sizeBytes: 4210,
    lineCount: 140
  },
  {
    id: 'f-004',
    path: 'packages/webgpu-search/src/shaders/substring.wgsl',
    filename: 'substring.wgsl',
    symbols: 'QueryUniforms, OutBuf, main, Match',
    type: 'shader',
    language: 'wgsl',
    description: 'WGSL compute shader for parallel exact substring search with position penalties',
    sizeBytes: 2950,
    lineCount: 95
  },
  {
    id: 'f-005',
    path: 'packages/webgpu-search/src/highlight.ts',
    filename: 'highlight.ts',
    symbols: 'alignHighlights, normalizeWithSourceMap, guardClusterBoundary, renderHighlightedText',
    type: 'function',
    language: 'typescript',
    description: 'Unicode-exact highlight coordinate projection mapping UTF-16 code units with indentation trim compensation',
    sizeBytes: 19800,
    lineCount: 520
  },
  {
    id: 'f-006',
    path: 'packages/webgpu-search/src/snapshot-codec.ts',
    filename: 'snapshot-codec.ts',
    symbols: 'encodeSnapshot, restoreSnapshot, decodeSnapshot, crc32Parts',
    type: 'function',
    language: 'typescript',
    description: 'Versioned Little-Endian snapshot binary persistence with circular-dependency-free CRC32',
    sizeBytes: 21400,
    lineCount: 560
  },
  {
    id: 'f-007',
    path: 'packages/webgpu-search/src/snapshot-idb.ts',
    filename: 'snapshot-idb.ts',
    symbols: 'saveIndexToIDB, restoreIndexFromIDB, openSearchDatabase, deleteIndexFromIDB',
    type: 'function',
    language: 'typescript',
    description: 'Transaction-safe IndexedDB storage helpers with pre-serialization and clean finally closing',
    sizeBytes: 14200,
    lineCount: 390
  },
  {
    id: 'f-008',
    path: 'packages/webgpu-search/src/worker/worker-client.ts',
    filename: 'worker-client.ts',
    symbols: 'SearchWorkerClient, monotonicQueryId, handleAbort, rehydrateErrors',
    type: 'class',
    language: 'typescript',
    description: 'First-party worker client with immediate AbortError rejection and string-isolated enrichment',
    sizeBytes: 23300,
    lineCount: 690
  },
  {
    id: 'f-009',
    path: 'packages/webgpu-search/src/worker/search-worker.ts',
    filename: 'search-worker.ts',
    symbols: 'startSearchWorker, isDedicatedWorker, handleWorkerMessage',
    type: 'function',
    language: 'typescript',
    description: 'SSR-safe dedicated worker entrypoint executing DocumentIndex off the main UI thread',
    sizeBytes: 8400,
    lineCount: 280
  },
  {
    id: 'f-010',
    path: 'packages/webgpu-search/src/types.ts',
    filename: 'types.ts',
    symbols: 'DocumentIndexOptions, DocumentSearchResponse, HighlightRange, MutationBatch, MutationResult',
    type: 'interface',
    language: 'typescript',
    description: 'Public API contract types, telemetry stats interfaces, and mutation types',
    sizeBytes: 12800,
    lineCount: 340
  },
  {
    id: 'f-011',
    path: 'packages/webgpu-search/src/exact-scorer.ts',
    filename: 'exact-scorer.ts',
    symbols: 'scoreExactMatches, scoreExactMatchesMultiField, scoreFuzzyTokens, scoreSubstringTokens',
    type: 'function',
    language: 'typescript',
    description: 'Deterministic CPU reference matching engine asserting byte-for-byte ranking parity with WGSL compute',
    sizeBytes: 24500,
    lineCount: 620
  },
  {
    id: 'f-012',
    path: 'packages/webgpu-search/src/text-normalization.ts',
    filename: 'text-normalization.ts',
    symbols: 'normalizeText, toWellFormedSafe, tokensEqual',
    type: 'function',
    language: 'typescript',
    description: 'NFC normalization, full C+F case folding table lookup, and surrogate pair sanitation',
    sizeBytes: 9800,
    lineCount: 290
  },
  {
    id: 'f-013',
    path: 'src/editor/monaco/quick_open/quick_open_model.ts',
    filename: 'quick_open_model.ts',
    symbols: 'QuickOpenModel, filterItems, sortScores, getActiveItem',
    type: 'class',
    language: 'typescript',
    description: 'Monaco QuickOpen palette data model handling keyboard navigation and item selection',
    sizeBytes: 16400,
    lineCount: 410
  },
  {
    id: 'f-014',
    path: 'src/editor/monaco/text_model/unicode_highlighter.ts',
    filename: 'unicode_highlighter.ts',
    symbols: 'UnicodeHighlighter, decorateMatches, computeDecorationRange',
    type: 'class',
    language: 'typescript',
    description: 'Monaco editor text decorations applying highlighted mark ranges to buffer lines',
    sizeBytes: 18200,
    lineCount: 480
  },
  {
    id: 'f-015',
    path: 'src/runtime/vulkan/vulkan_context_manager.rs',
    filename: 'vulkan_context_manager.rs',
    symbols: 'VulkanContextManager, acquire_device, create_pipeline_layout, submit_command_buffer',
    type: 'class',
    language: 'rust',
    description: 'Low-level Vulkan runtime adapter context initialization for headless compute clusters',
    sizeBytes: 31200,
    lineCount: 820
  }
];

const MODULE_NAMES = [
  'auth', 'billing', 'cache', 'database', 'events', 'filesystem', 'gateway',
  'indexer', 'journal', 'kernel', 'logger', 'metrics', 'network', 'optimizer',
  'pipeline', 'query', 'router', 'storage', 'telemetry', 'utility', 'validator',
  'worker', 'xml_parser', 'yaml_parser', 'zone_allocator'
];

const COMPONENT_KINDS: Array<{ kind: SymbolKind; ext: string; lang: string }> = [
  { kind: 'class', ext: '.ts', lang: 'typescript' },
  { kind: 'function', ext: '.ts', lang: 'typescript' },
  { kind: 'interface', ext: '.ts', lang: 'typescript' },
  { kind: 'shader', ext: '.wgsl', lang: 'wgsl' },
  { kind: 'type', ext: '.ts', lang: 'typescript' },
  { kind: 'constant', ext: '.json', lang: 'json' },
  { kind: 'file', ext: '.md', lang: 'markdown' },
  // Rust records so the `rust` language filter resolves to generated docs
  // (previously only a single CORE_FILES seed record matched).
  { kind: 'class', ext: '.rs', lang: 'rust' }
];

export function generateMonacoRecords(count: number = 600): MonacoFileRecord[] {
  const records: MonacoFileRecord[] = [...CORE_FILES];
  let idCounter = records.length + 1;

  for (let i = records.length; i < count; i++) {
    const mod = MODULE_NAMES[i % MODULE_NAMES.length]!;
    const comp = COMPONENT_KINDS[i % COMPONENT_KINDS.length]!;
    const idx = Math.floor(i / MODULE_NAMES.length);
    const capitalizedMod = mod.charAt(0).toUpperCase() + mod.slice(1);
    
    let filename = '';
    let symbols = '';
    let desc = '';
    let path = '';

    switch (comp.kind) {
      case 'class':
        if (comp.lang === 'rust') {
          filename = `${mod}_context_${idx}.rs`;
          path = `src/runtime/${mod}/${filename}`;
          symbols = `${capitalizedMod}Context${idx}, acquire_device, submit_command_buffer`;
          desc = `Rust runtime adapter context for ${mod} headless compute`;
          break;
        }
        filename = `${mod}_service_${idx}.ts`;
        path = `packages/services/${mod}/${filename}`;
        symbols = `${capitalizedMod}Service${idx}, initialize, handleRequest, flushBuffer, shutdown`;
        desc = `Enterprise ${mod} business logic handler with circuit breaker and telemetry tracking`;
        break;
      case 'function':
        filename = `process_${mod}_stream_${idx}.ts`;
        path = `src/pipelines/${mod}/${filename}`;
        symbols = `process${capitalizedMod}Stream${idx}, transformChunk, filterAnomalies, encodePayload`;
        desc = `High-performance async stream transformer for incoming ${mod} events`;
        break;
      case 'interface':
        filename = `${mod}_contracts_${idx}.ts`;
        path = `src/contracts/${mod}/${filename}`;
        symbols = `I${capitalizedMod}Config${idx}, ${capitalizedMod}State, ${capitalizedMod}ResultPayload`;
        desc = `Strict typing schema and protocol definitions for ${mod} module interop`;
        break;
      case 'shader':
        filename = `${mod}_compute_pass_${idx}.wgsl`;
        path = `src/shaders/compute/${filename}`;
        symbols = `${capitalizedMod}Uniforms${idx}, compute${capitalizedMod}Pass, main, ScratchBuffer`;
        desc = `WGSL compute kernel executing vectorized parallel math for ${mod} pipeline`;
        break;
      case 'type':
        filename = `${mod}_types_${idx}.ts`;
        path = `packages/types/${mod}/${filename}`;
        symbols = `${capitalizedMod}Status, ${capitalizedMod}ErrorCode, ${capitalizedMod}OptionFlags`;
        desc = `Union types, literal status enums, and utility mapped types for ${mod}`;
        break;
      case 'constant':
        filename = `${mod}_preset_${idx}.json`;
        path = `config/presets/${filename}`;
        symbols = `default${capitalizedMod}Config, timeoutThresholdMs, retryLimit, backoffMultiplier`;
        desc = `Configuration presets and deployment defaults for ${mod} cluster nodes`;
        break;
      case 'file':
        filename = `${mod}_architecture_guide_${idx}.md`;
        path = `docs/architecture/${filename}`;
        symbols = `${capitalizedMod}Overview, PerformanceTradeoffs, DeploymentTopology`;
        desc = `Technical specification and design manual for ${mod} subsystem architecture`;
        break;
      default:
        filename = `${mod}_util_${idx}.ts`;
        path = `src/utils/${filename}`;
        symbols = `format${capitalizedMod}, parse${capitalizedMod}`;
        desc = `Utility helpers for ${mod}`;
    }

    records.push({
      id: `f-${String(idCounter++).padStart(4, '0')}`,
      path,
      filename,
      symbols,
      type: comp.kind,
      language: comp.lang,
      description: desc,
      sizeBytes: 1024 + (i * 127) % 65536,
      lineCount: 40 + (i * 17) % 1200
    });
  }

  return records;
}
