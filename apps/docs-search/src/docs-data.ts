import type { DocPageRecord, DocSection } from './types';

export const CORE_DOCS: DocPageRecord[] = [
  {
    id: 'd-001',
    path: 'docs/guides/quickstart.md',
    title: 'Quickstart: offline search in five minutes',
    section: 'Guide',
    content: 'Install webgpu-search, create a DocumentIndex over title and body fields, and run your first fuzzy query. Prefer the SearchWorkerClient so indexing never blocks the main thread.',
    tags: 'quickstart, offline, DocumentIndex, SearchWorkerClient',
    version: '1.0',
    readingMinutes: 5
  },
  {
    id: 'd-002',
    path: 'docs/api/document-index.md',
    title: 'DocumentIndex multi-field API reference',
    section: 'API',
    content: 'DocumentIndex.create packs field-stratified rows with per-field weights. Search supports fuzzy, substring, prefix, and token modes with filters, facets, autocomplete, and Unicode-exact highlights.',
    tags: 'DocumentIndex, fields, weights, search modes',
    version: '1.0',
    readingMinutes: 12
  },
  {
    id: 'd-003',
    path: 'docs/api/worker-client.md',
    title: 'SearchWorkerClient off-thread indexing',
    section: 'API',
    content: 'The worker client mirrors DocumentIndex over postMessage with monotonic query sequencing: rapid typing aborts superseded searches and the last query wins. Caller AbortSignal is honored.',
    tags: 'SearchWorkerClient, worker, AbortSignal, sequencing',
    version: '1.0',
    readingMinutes: 8
  },
  {
    id: 'd-004',
    path: 'docs/storage/snapshot-format.md',
    title: 'Snapshot format v4 persistence guide',
    section: 'Storage',
    content: 'Snapshots are Little-Endian binaries with a 56-byte header carrying magic, format version, profile id, scoring version, and a CRC32 checksum. Legacy v3 restores read-only; corrupt or profile-mismatched buffers reject fail-closed.',
    tags: 'snapshot, persistence, CRC32, migration',
    version: '1.0',
    readingMinutes: 10
  },
  {
    id: 'd-005',
    path: 'docs/storage/indexeddb.md',
    title: 'IndexedDB offline snapshot storage',
    section: 'Storage',
    content: 'Save serialized snapshots with saveIndexToIDB and restore them with loadIndexFromIDB or restoreIndexFromIDB. Docs apps persist the bundle once and restore it on next launch without rebuilding.',
    tags: 'IndexedDB, offline, saveIndexToIDB, restore',
    version: '1.0',
    readingMinutes: 7
  },
  {
    id: 'd-006',
    path: 'docs/guides/filters-facets.md',
    title: 'Structured filters and facet navigation',
    section: 'Guide',
    content: 'Declare filterFields for columnar bitset pre-filtering, then query with filter expressions and terms or range facets. Facets power sidebar navigation over sections, versions, and latency buckets.',
    tags: 'filter, facets, columnar, navigation',
    version: '1.0',
    readingMinutes: 9
  },
  {
    id: 'd-007',
    path: 'docs/guides/highlighting.md',
    title: 'Original-text highlighting without XSS',
    section: 'Guide',
    content: 'Request highlight with escapeHtml and a mark tag to project match coordinates onto the original text. Highlighted output escapes hostile markup so docs render safely via innerHTML.',
    tags: 'highlight, XSS, escapeHtml, mark',
    version: '1.0',
    readingMinutes: 6
  },
  {
    id: 'd-008',
    path: 'docs/reference/text-normalization.md',
    title: 'Unicode normalization and case folding',
    section: 'Reference',
    content: 'Queries and documents share NFC normalization with full case folding across astral planes. Case-insensitive search matches across scripts while caseSensitive mode enforces exact-case packing.',
    tags: 'unicode, NFC, case folding, normalization',
    version: '1.0',
    readingMinutes: 8
  },
  {
    id: 'd-009',
    path: 'docs/reliability/device-loss.md',
    title: 'Device loss recovery and rebuildGpu',
    section: 'Reliability',
    content: 'When the GPU device is lost the index falls back to the exact CPU scorer with a device-lost reason and identical match semantics. Call rebuildGpu to re-acquire the device and re-upload retained corpora.',
    tags: 'device-lost, rebuildGpu, fallback, recovery',
    version: '1.0',
    readingMinutes: 7
  },
  {
    id: 'd-010',
    path: 'docs/reliability/support-matrix.md',
    title: 'Browser and CPU baseline support matrix',
    section: 'Reliability',
    content: 'Chrome, Edge, Safari, and Firefox ship WebGPU or the tested CPU-only baseline. Prefer preferGpu false with the exact scorer as the reference contract reproducible without a GPU.',
    tags: 'support matrix, CPU baseline, browsers, preferGpu',
    version: '1.0',
    readingMinutes: 6
  },
  {
    id: 'd-011',
    path: 'docs/reference/public-api.md',
    title: 'Public API freeze and versioning policy',
    section: 'Reference',
    content: 'The 1.x surface freezes DocumentIndex, SearchIndex, and SearchWorkerClient contracts with descending integer scores and frozen tie-breakers. Scoring or storage changes require a major version.',
    tags: 'public API, versioning, ordering, deprecation',
    version: '1.0',
    readingMinutes: 11
  },
  {
    id: 'd-012',
    path: 'docs/guides/autocomplete.md',
    title: 'Prefix autocomplete for docs search boxes',
    section: 'Guide',
    content: 'Autocomplete resolves prefix completions index-wide with per-field rows. Attach it inline to a search request or call the dedicated autocomplete primitive for lightweight suggestion drop-downs.',
    tags: 'autocomplete, prefix, suggestions',
    version: '1.0',
    readingMinutes: 5
  },
  {
    id: 'd-013',
    path: 'docs/guides/mutations.md',
    title: 'Incremental doc updates without rebuilds',
    section: 'Guide',
    content: 'Add, update, and remove individual pages with atomic batched mutations. The mutation epoch advances per batch and tombstones compact automatically during repacking.',
    tags: 'mutations, add, update, remove, epoch',
    version: '1.0',
    readingMinutes: 6
  },
  {
    id: 'd-014',
    path: 'docs/reference/benchmarks.md',
    title: 'Benchmark fixtures and environments',
    section: 'Reference',
    content: 'Reproducible fixtures pair versioned corpora with named queries, modes, and filters. Headless CPU and browser WebGPU reports stay separate with median, p95, memory, and serialize costs.',
    tags: 'benchmarks, fixtures, median, p95',
    version: '1.0',
    readingMinutes: 9
  }
];

const SECTIONS: DocSection[] = ['Guide', 'API', 'Storage', 'Reliability', 'Reference'];

const TOPIC_WORDS = [
  'tokenizer', 'pipeline', 'workgroup', 'bindgroup', 'readback', 'compaction',
  'tombstone', 'bitset', 'facet', 'quaestor', 'dispatch', 'uniform', 'staging',
  'epoch', 'checksum', 'telemetry', 'latency', 'throughput', 'sharding', 'replica'
];

/** Deterministic offline-docs corpus generator (no Math.random: stable across runs). */
export function generateDocsRecords(count: number = 400): DocPageRecord[] {
  const records: DocPageRecord[] = [...CORE_DOCS];
  let counter = records.length + 1;

  for (let i = records.length; i < count; i++) {
    const section = SECTIONS[i % SECTIONS.length]!;
    const topic = TOPIC_WORDS[i % TOPIC_WORDS.length]!;
    const idx = Math.floor(i / SECTIONS.length);
    const slug = `${topic.replace(/[^a-z0-9]+/gi, '-').toLowerCase()}-${idx}`;
    const version = idx % 4 === 3 ? '0.4' : '1.0';

    records.push({
      id: `d-${String(counter++).padStart(4, '0')}`,
      path: `docs/${section.toLowerCase()}/${slug}.md`,
      title: `${topic} ${section.toLowerCase()} notes part ${idx}`,
      section,
      content: `Offline reference page ${idx} describing ${topic} behavior for ${section} readers, including configuration knobs, failure modes, and migration notes across supported releases.`,
      tags: `${topic}, ${section.toLowerCase()}, offline-docs`,
      version,
      readingMinutes: 3 + ((i * 7) % 12)
    });
  }

  return records;
}
