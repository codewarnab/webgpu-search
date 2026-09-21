# Issue #9 — v0.3 Embeddable Document, Mutation, Worker, Highlight, and Persistence APIs: Implementation Plan

> **Issue Reference**: [GitHub Issue #9: v0.3: add embeddable document, mutation, worker, highlight, and persistence APIs](https://github.com/codewarnab/webgpu-fuzzy-search/issues/9)  
> **Target Milestone**: `0.3.0`  
> **Status**: Approved by Multi-Agent Consensus / Ready for Execution  
> **Predecessor**: v0.2.0 (Unicode code-point-safe CPU/GPU matching, U2F2 binary format, zero-dependency core)  
> **Review Board Artifact**: [`multi-agent-review-issue-9.md`](file:///home/ubuntu/.agy-homes/acct1/.gemini/antigravity-cli/brain/9835b7d1-be2f-4dd6-95e1-187c320d36ae/multi-agent-review-issue-9.md)

---

## 0. Executive Summary & Outcome

The objective of v0.3 is to elevate `webgpu-search` from a low-level string-array compute engine into a **production-grade embeddable search infrastructure** for browser IDEs (Monaco file/symbol search), log viewers, data grids, documentation portals, and offline-first client applications.

Host applications will be able to index structured records with stable IDs, weight individual string fields, perform dynamic batched mutations (`add`, `update`, `remove`), obtain Unicode-exact highlight ranges mapping directly to original JavaScript strings, run searches off-thread via a first-party worker client without UI thread blockage, and serialize/restore versioned indexes to IndexedDB with fail-closed integrity guarantees.

---

## 1. Architectural Principles & Boundaries

### 1.1 Non-Negotiable Acceptance Boundaries

1. **Clean Public Surface**: Host applications interact strictly through high-level interfaces (`DocumentIndex`, `SearchWorkerClient`). They never manually allocate WebGPU buffers, calculate alignments, or manage bind group layouts.
2. **Unicode & Display Alignment**: Highlight ranges map directly to UTF-16 code units in the host application's original JavaScript strings, accurately accounting for leading indentation (`leadingTrimOffset`), NFC normalization, full C+F case folding expansions (`ß` $\to$ `ss`, `İ` $\to$ `i + ◌̇`), astral surrogate pairs (e.g. emojis `😀`), and lone surrogates. Partial expansion matches obey the **Atomic Character Expansion Principle**, and highlight ranges never truncate grapheme clusters.
3. **Fail-Closed Persistence**: Serialized indexes store magic constants, schema versions, Unicode versions, scoring versions, and CRC32 checksums. All header and payload integers are strictly 32-bit Little-Endian. CRC32 covers header `[0..44)` plus payload segments (eliminating circular dependencies). Incompatible snapshots fail immediately with `IncompatibleIndexError`. Decoupled document storage (`docsByteLength = 0`) is supported to avoid 100MB+ JSON string allocations for 100k+ record sets.
4. **Non-Blocking UI Thread & Promise Liveness**: The first-party worker client encapsulates all token packing, GPU dispatch, readback, and candidate ranking off the main thread. Superseded or cancelled queries reject immediately with `AbortError` (never hanging Promises). Custom error prototypes are rehydrated seamlessly across the thread boundary.
5. **Zero Framework Runtime Coupling**: `packages/webgpu-search` remains 100% zero-dependency (retaining only `@leeoniya/ufuzzy` for legacy CPU fallback). No React, Vue, Svelte, Monaco, or DOM references exist inside the core package. Framework integrations are published as lightweight examples, and runnable proof applications reside under `apps/`.

### 1.2 Non-Goals

- No hosted crawler or scraping pipeline.
- No managed cloud search service or network server daemon.
- No proprietary monolithic search UI components or design systems.
- No semantic, embedding, or vector retrieval layer.

### 1.3 Multi-Agent Review Board Findings & Consensus Amendments

A formal multi-agent review conducted across 4 specialized architectural domains evaluated the initial v0.3 design and unanimously ratified **6 Mandatory Architectural Amendments**:

| Reviewer Domain | Initial Grade | Key Defect Uncovered | Mandated Resolution |
|---|:---:|---|---|
| **WebGPU Systems & Hardware Architect** | **C-** | Setting `off[r] = off[r+1]` corrupts prefix sums and swallows preceding rows. Large multi-field corpora cause candidate pool inversion. | Prohibit in-place cumulative offset zeroing. Implement CPU readback tombstone filtering. Pack rows in field-stratified priority order and scale `candidateCapacity` to 32,768. |
| **Unicode & Text Processing Specialist** | **B** | Indentation drift from `normalizeText().trim()`. Slicing partial case folds (`ß`) breaks characters. Combining marks split by tags. | Introduce `leadingTrimOffset` coordinate compensation. Enforce Atomic Character Expansion. Guard grapheme cluster boundaries. |
| **API & Monorepo Framework Architect** | **B+** | Incomplete response types. Examples placed in root `examples/` broken in Turborepo. npm `files` missing `dist/worker`. | Fully specify `DocumentSearchResponse<TDoc>` and options. Relocate proofs to `apps/monaco-palette` and `apps/log-viewer`. Guard worker with `isDedicatedWorker`. |
| **Concurrency & Persistence Engineer** | **C+** | Dropped queries deadlock Promises. ArrayBuffer transfers neuter caller/worker state. CRC32 circularly covers its own field. | Monotonic query IDs with immediate `AbortError` rejection. Opt-in transfer on restore. Compute CRC32 over header `[0..44)` + payload. Support decoupled document persistence. |

---

## 2. Target Architecture & Design Contracts

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│ Host Application UI / Worker Layer                                                │
│ (React / Vue / Svelte / Vanilla TS / Monaco QuickOpen / Log Viewer)               │
└────────────────────────────────────────┬──────────────────────────────────────────┘
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
┌──────────────────────────────────────┐     ┌──────────────────────────────────────┐
│       SearchWorkerClient<TDoc>       │     │         DocumentIndex<TDoc>          │
│   - Typed Message Protocol           │     │   - Stable Record IDs                │
│   - Reject on Abort / Monotonic ID   │     │   - Field Weights & Multi-Field      │
│   - Transfer-Safe Snapshot Copies    │     │   - Field-Stratified Row Packing     │
│   - String-Isolated Enrichment       │     │   - Batched Add / Update / Remove    │
│   - Error Class Rehydration          │     │   - Dynamic Clamped Headroom         │
│   - destroy() / [Symbol.dispose]     │     │   - CPU Readback Tombstone Filter    │
└──────────────────┬───────────────────┘     └──────────────────┬───────────────────┘
                   │                                            │
                   │ Web Worker Boundary                        │
                   ▼                                            │
┌──────────────────────────────────────┐                        │
│          Worker Entrypoint           │                        │
│         (webgpu-search/worker)       │                        │
│   - isDedicatedWorker Safe Guard     │                        │
└──────────────────┬───────────────────┘                        │
                   │                                            │
                   └─────────────────────┬──────────────────────┘
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ Core Engine & Data Plane                                                          │
├──────────────────────────────────────┬────────────────────────────────────────────┤
│ Highlighting & Alignment Engine      │ Versioned Persistence (U2D3)               │
│ - Post-fold to UTF-16 code units     │ - Magic: 0x55324433 ('U2D3', Little-Endian)│
│ - Leading trim offset compensation   │ - CRC32 on header [0..44) + payload        │
│ - Atomic character expansion rule    │ - Pre-serialization vacuum/compaction      │
│ - Grapheme cluster boundary guard    │ - Decoupled Document Storage option        │
│ - Non-allocating top-K scan          │ - Transaction-safe IndexedDB Storage       │
├──────────────────────────────────────┴────────────────────────────────────────────┤
│ WebGPU Compute & CPU Parity Engine                                                │
│ - Substring (1000 - pos*10 - len)    - Fuzzy (bonuses, span & len penalty)        │
│ - Dynamic writeBuffer headroom       - Cumulative Offsets Invariant Preserved     │
│ - Scaled candidate pool (up to 32k)  - Bitmask Filter / Monotonic Compaction      │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Component Specifications

### 3.1 Document Model & Multi-Field Weighting

```ts
export type DocumentId = string | number;

export interface FieldDefinition<TDoc = Record<string, unknown>> {
  /** Field identifier / key in TDoc */
  name: (keyof TDoc & string) | string;
  /** Custom property extractor; defaults to (doc) => doc[name] */
  getter?: (doc: TDoc) => string | string[] | undefined | null;
  /** Weight multiplier applied to raw match score (default: 1.0, must be > 0 and finite) */
  weight?: number;
}

export type DocumentField<TDoc> = (keyof TDoc & string) | FieldDefinition<TDoc>;

export interface DocumentIndexOptions<TDoc = Record<string, unknown>> extends IndexOptions {
  /** Property containing unique document identifier (default: 'id') */
  idField?: (keyof TDoc & string) | ((doc: TDoc) => DocumentId);
  /** Configured search fields and weights */
  fields: Array<DocumentField<TDoc>>;
  /** Pre-allocated row capacity for dynamic mutations (default: records.length * 1.5) */
  initialCapacity?: number;
  /** Buffer expansion factor during headroom overflow (default: 1.5) */
  growthFactor?: number;
  /** Candidate pool capacity for multi-field search (default: 8192, up to 32768) */
  candidateCapacity?: number;
}

export interface DocumentSearchOptions<TDoc = any> extends SearchOptions {
  /** Restrict search to specific configured fields */
  fields?: string[];
  /** Whether to compute highlight ranges (default: true) */
  highlight?: boolean;
  /** HTML/formatting tag for snippets (e.g. 'mark', 'b') */
  tag?: string;
  /** Predicate filter applied post-match */
  filter?: (doc: TDoc) => boolean;
}

export interface DocumentSearchResultItem<TDoc = any> {
  id: DocumentId;
  doc: TDoc;
  score: number;
  matchedField: string;
  /** Highlight ranges per field in UTF-16 code units of the original string */
  highlights?: Record<string, HighlightRange[]>;
  /** Formatted HTML strings with injected tags (when options.tag is set) */
  highlightedText?: Record<string, string>;
  /** Auxiliary matches across other indexed fields */
  matches?: Array<{ field: string; score: number; highlights?: HighlightRange[] }>;
}

export interface DocumentSearchResponse<TDoc = any> {
  query: string;
  mode: SearchMode;
  engine: EngineType;
  totalMatches: number;
  candidateCount: number;
  hasOverflow: boolean;
  results: DocumentSearchResultItem<TDoc>[];
  timings: SearchTimings;
  profileId: TextProfileId;
  scoringVersion: typeof SCORING_VERSION;
  cpuAlgorithm: CpuAlgorithm;
  fallbackReason?: FallbackReason;
}
```

#### Row Layout & Candidate Pool Inversion Prevention:
- **Field-Stratified Row Packing**:
  In multi-field indexing, documents are NOT interleaved (`Doc0_F0, Doc0_F1...`). Instead, rows are packed in **descending field-weight priority**:
  - Rows $0 \dots N-1$: Primary fields (e.g. `title`, weight 2.0).
  - Rows $N \dots 2N-1$: Secondary fields (e.g. `tags`, weight 1.0).
  - Rows $2N \dots 3N-1$: Low-weight fields (e.g. `content`, weight 0.2).
  Since WebGPU workgroups execute monotonically ($0 \dots \text{totalWorkgroups}$), high-weight field rows secure slots in the GPU candidate pool (`out.results[oi]`) before low-weight field rows can saturate the pool.
- **Dynamic Candidate Pool Scaling**:
  `candidateCapacity` scales dynamically with field count:
  $$\text{candidateCapacity} = \min(32768, \max(8192, \text{docCount} \times \text{fieldCount} \times 0.1))$$
- **Fixed-Point Scoring Parity**:
  Field-weighted score calculation follows strict integer fixed-point rounding:
  $$S_{\text{weighted}} = \operatorname{Math.round}(S_{\text{raw}} \times W_{\text{field}})$$
  Identical integer rounding is applied across WebGPU readback enrichment and CPU parity reference paths.

---

### 3.2 Dynamic Batched Mutations & Consistency Rules

```ts
export interface AddOptions {
  upsert?: boolean;
}

export interface MutationBatch<TDoc = any> {
  add?: TDoc[];
  update?: TDoc[];
  remove?: DocumentId[];
}

export interface MutationResult {
  added: number;
  updated: number;
  removed: number;
  mutationEpoch: number;
  compacted: boolean;
  durationMs: number;
}
```

#### Consistency Invariants:
1. **Primary Key Uniqueness**: `docId` is strictly unique within the index. Calling `add()` with an existing `id` throws `DuplicateIdError` unless `{ upsert: true }` is specified.
2. **Atomic Two-Phase Execution**: `applyBatch()` validates all document IDs, extractors, and field strings before touching live buffers. If validation fails or VRAM allocation throws, state is unchanged.
3. **Execution Ordering**: Within a single batch, operations execute in deterministic sequence:
   $$\text{remove} \longrightarrow \text{update} \longrightarrow \text{add}$$
4. **Read-Your-Writes & Mutex Protection**: All mutations acquire `searchMutex` (`queued()`) and increment `this.generation++`. In-flight search readbacks from prior epochs are discarded via `AbortError`. Any search initiated after `await index.applyBatch(...)` immediately observes the updated state.
5. **Cumulative Offsets Preservation (No In-Place Offset Zeroing)**:
   - **CRITICAL INVARIANT**: Cumulative offsets (`off[r]`) are **NEVER modified in-place** during deletions (`off[r] = off[r+1]` is strictly forbidden because it causes row $r-1$ to swallow row $r$'s tokens).
   - **Tombstone Strategy**: Deletions and updates record row indices in a host-side `tombstones: Set<number>` (and an optional GPU `@binding(5) var<storage, read> tombstones: array<u32>` bitmask for high-churn datasets).
   - **Candidate Discard**: The candidate readback loop drops matches where `tombstones.has(rid)`.
   - **CPU-Driven Compaction**: When $\frac{\text{tombstones}}{\text{totalRows}} \ge 0.25$ or headroom is exhausted, contiguous buffers are repacked cleanly on the CPU and uploaded via `writeBuffer`.
6. **Clamped Dynamic Headroom**:
   When allocating buffer headroom, the allocation size is strictly clamped to the adapter's limits:
   $$\text{targetBytes} = \min(\text{requiredBytes} \times \text{growthFactor}, \text{device.limits.maxStorageBufferBindingSize})$$
   If headroom overflows the limit but `requiredBytes` fits, the index allocates exact bytes instead of triggering an unnecessary CPU fallback.

---

### 3.3 Unicode-Safe Highlight Alignment Engine

```ts
export interface HighlightRange {
  /** Start index in UTF-16 code units in the original JS string */
  start: number;
  /** End index (exclusive) in UTF-16 code units in the original JS string */
  end: number;
}

export interface HighlightOptions {
  /** Whether to compute highlight ranges (default: true) */
  highlight?: boolean;
  /** HTML/markdown tag for formatting (e.g. 'mark', 'b') */
  tag?: string;
  /** Fields to highlight ('matched-field' | 'all-matched' | 'all-fields' | string[]) */
  fields?: 'matched-field' | 'all-matched' | 'all-fields' | string[];
}
```

#### Alignment Algorithm (Top-K Demand Driven):
- Highlighting is computed **strictly for the top-K returned documents**, keeping indexing memory overhead at **0 extra bytes**.
- Algorithm:
  1. **Leading Whitespace Offset**: Record `leadingTrimOffset = raw.length - raw.trimStart().length` to prevent indentation drift caused by `normalizeText`'s `.trim()`.
  2. **ASCII Fast-Path**: If pure ASCII, map tokens 1:1 to UTF-16 offsets with zero allocations ($< 0.03\text{ ms}$ for 50 records).
  3. **Unicode Coordinate Map**:
     - Map each code point to its UTF-16 span $[u_{\text{start}}, u_{\text{end}})$ (astral surrogate pairs span 2 UTF-16 units).
     - Account for NFC composition (combining marks merge with base starters).
     - Account for full C+F case folding: when a character expands (e.g. `'ß'` $\to$ `115, 115`), every expanded token inherits the full original span $[u, u+1)$.
  4. **Query Subsequence Alignment**:
     - Locate matched query tokens within the post-fold token stream using the exact same greedy forward loop as `fuzzy.wgsl` and `scoreFuzzyTokens` (preserving score-highlight symmetry).
  5. **Atomic Character Expansion Principle**:
     - Partial expansion matches (e.g. query `'s'` matching the first token of `'ß'`) highlight the entire atomic glyph.
  6. **Grapheme Cluster Boundary Guard**:
     - Highlight spans never terminate in the middle of a combining sequence. If `raw[end]` is a combining mark, `end` extends across the cluster.
  7. **Span Merging**:
     - Overlapping or adjacent spans are merged into minimal non-overlapping `HighlightRange[]` slices.

---

### 3.4 First-Party Worker Client (`webgpu-search/worker`)

```ts
export interface WorkerClientOptions {
  /** Optional custom worker instance or factory */
  worker?: Worker | (() => Worker);
  /** Whether to strip document text across thread boundary (default: true) */
  stringIsolated?: boolean;
}

export interface WorkerRequest {
  id: number;
  type: 'INIT' | 'SEARCH' | 'MUTATE' | 'SERIALIZE' | 'RESTORE' | 'STATS' | 'DESTROY';
  payload?: any;
}

export interface WorkerResponse {
  id: number;
  success: boolean;
  result?: any;
  error?: {
    name: string;
    message: string;
    stack?: string;
    details?: Record<string, unknown>;
  };
}

export class SearchWorkerClient<TDoc = Record<string, unknown>> {
  constructor(options?: WorkerClientOptions);
  init(options?: DocumentIndexOptions<TDoc>): Promise<void>;
  search(query: string, options?: DocumentSearchOptions<TDoc>): Promise<DocumentSearchResponse<TDoc>>;
  add(docs: TDoc | TDoc[], options?: AddOptions): Promise<MutationResult>;
  update(docs: TDoc | TDoc[]): Promise<MutationResult>;
  remove(ids: DocumentId | DocumentId[]): Promise<MutationResult>;
  applyBatch(batch: MutationBatch<TDoc>, options?: AddOptions): Promise<MutationResult>;
  serialize(): Promise<ArrayBuffer>;
  restore(buffer: ArrayBuffer, options?: { transfer?: boolean }): Promise<void>;
  getStats(): Promise<DocumentIndexStats>;
  destroy(): Promise<void>;
  [Symbol.asyncDispose](): Promise<void>;
}
```

#### Concurrency & Message Protocol:
- **Monotonic Query Sequencing & Promise Rejection**:
  Every query receives an incremental `queryId`. When a newer query is initiated or an `AbortController` aborts, all prior pending Promises are **immediately rejected with `AbortError`** and an `ABORT` message is posted to the worker. No Promises are ever left hanging.
- **Transferable Buffer Safety**:
  - `serialize()` returns a newly allocated detached snapshot buffer (never transferring the worker's internal active index buffers).
  - `restore(buffer)` clones the input buffer by default (`buffer.slice(0)`), transferring the backing buffer only when `{ transfer: true }` is explicitly passed.
- **Error Class Rehydration**:
  Errors traversing the worker boundary serialize `{ name, message, stack, details }` and rehydrate into exact class instances (`QueryTooLongError`, `IncompatibleIndexError`, `ProfileMismatchError`, `DuplicateIdError`).
- **SSR & Main-Thread Safety**:
  Worker entrypoint guards against Node/SSR environments:
  ```ts
  const isDedicatedWorker =
    typeof self !== 'undefined' &&
    typeof (self as any).importScripts === 'function' &&
    typeof (self as any).postMessage === 'function';
  if (isDedicatedWorker) startSearchWorker();
  ```

---

### 3.5 Versioned Snapshot Persistence (IndexedDB-Friendly U2D3)

#### Binary Header Layout (48 bytes, 12 $\times$ u32 words, Strictly Little-Endian):
```
[0x00..0x03]: MAGIC = 0x55324433 ('U2D3')
[0x04..0x07]: formatVersion = 3
[0x08..0x0B]: profileEnum = 1 (unicode-default)
[0x0C..0x0F]: unicodeVersionEnum = 1 (16.0.0)
[0x10..0x13]: scoringVersionEnum = 1 (parity-v1)
[0x14..0x17]: docCount (u32, LE)
[0x18..0x1B]: rowCount (u32, LE)
[0x1C..0x1F]: tokenCount (u32, LE)
[0x20..0x23]: folded (u32: 0 or 1)
[0x24..0x27]: schemaByteLength (u32, LE)
[0x28..0x2B]: docsByteLength (u32, LE, 0 if decoupled)
[0x2C..0x2F]: checksum (u32: CRC32 of bytes 0..43 + payload segments)
```

#### Payload Segment Memory Layout:
```
┌─────────────────────────────────────────────────────────────┐
│ Header (48 bytes, LE)                                       │
├─────────────────────────────────────────────────────────────┤
│ Schema JSON Segment (schemaByteLength bytes, UTF-8)         │
├─────────────────────────────────────────────────────────────┤
│ Tokens Segment (tokenCount * 4 bytes, array of u32 LE)      │
├─────────────────────────────────────────────────────────────┤
│ Offsets Segment ((rowCount + 1) * 4 bytes, array of u32 LE) │
├─────────────────────────────────────────────────────────────┤
│ Document Records Segment (docsByteLength bytes, UTF-8)      │
│ * Omitted / 0 bytes if decoupled document storage is used.  │
└─────────────────────────────────────────────────────────────┘
```

#### Deterministic Checksum & Invariants:
- **No Circular Dependency**: Word 11 (`[0x2C..0x2F]`) is strictly the checksum destination. CRC32 is calculated over header bytes 0..43 concatenated with payload segments:
  $$\text{CRC32} = \operatorname{crc32}\left( \text{header}[0..44) \mathbin{\Vert} \text{schema} \mathbin{\Vert} \text{tokens} \mathbin{\Vert} \text{offsets} \mathbin{\Vert} \text{docs} \right)$$
- **Compaction Pre-Condition**: `serializeDocumentIndex()` always executes a compaction pass first, guaranteeing 0 tombstones in serialized snapshots.
- **Decoupled Document Records**: Setting `docsByteLength = 0` allows host applications in string-isolated mode to persist search index structures without generating 100MB+ JSON string allocations. IndexedDB snapshot helpers store index binary snapshots in an `index_snapshots` store and decoupled document records in an `id`-keyed document store.
- **IndexedDB Transaction Safety**:
  `saveIndexToIDB` serializes the snapshot completely *before* initiating `db.transaction(...)`, preventing premature auto-commit `TransactionInactiveError`s. Database connections are closed in `finally` blocks to allow future version migrations.

---

### 3.6 Observability & Telemetry Reporting

```ts
export type FallbackReason =
  | 'webgpu-unsupported'
  | 'device-request-failed'
  | 'memory-budget-exceeded'
  | 'device-lost'
  | 'below-threshold'
  | 'prefer-cpu'
  | 'query-too-long'
  | 'cpu-algorithm-requested'
  | 'gpu-execution-error';

export interface DocumentIndexStats extends IndexStats {
  docCount: number;
  rowCount: number;
  tombstoneCount: number;
  tombstoneRatio: number;
  buildTimeMs: number;
  restoreTimeMs?: number;
  lastMutationTimeMs?: number;
  mutationEpoch: number;
  memory: {
    vramBytes: number;
    ramBytes: number;
    totalBytes: number;
  };
  fallbackReason?: FallbackReason;
}
```

---

## 4. Implementation Milestones & Work Breakdown

```
M1: Specifications & Public Contracts
 ├── DocumentIndex, FieldDefinition & SearchWorkerClient types
 ├── Little-Endian U2D3 binary specification & error hierarchy
 └── Turborepo workspace alignment (apps/ vs packages/)
      │
      ▼
M2: Document Record Engine & Multi-Field Indexing
 ├── Field-stratified token packing & field weights
 ├── Candidate pool scaling (up to 32k)
 └── Fixed-point integer scoring parity validation
      │
      ▼
M3: Unicode-Safe Highlighting Engine
 ├── Indentation trim compensation (leadingTrimOffset)
 ├── Atomic character expansion & grapheme cluster guard
 └── Greedy subsequence alignment & ASCII fast-path
      │
      ▼
M4: Batched Dynamic Mutations & Memory Management
 ├── Clamped buffer headroom & partial writeBuffer appends
 ├── Readback tombstone filtering (zero offset corruption)
 └── CPU-driven vacuum / compaction pipeline
      │
      ▼
M5: First-Party Worker Client & Protocol
 ├── Monotonic query sequencing with immediate AbortError rejection
 ├── Safe transferable buffer handling & error rehydration
 └── SSR-safe dedicated worker entrypoint (webgpu-search/worker)
      │
      ▼
M6: Versioned Snapshot Persistence & IndexedDB
 ├── U2D3 serialization (header [0..44) CRC32 + LE)
 ├── Decoupled document payload support (docsByteLength = 0)
 └── Transaction-safe IndexedDB helpers (saveIndexToIDB)
      │
      ▼
M7: Observability & Framework Integration Recipes
 ├── Telemetry instrumentation (fallbackReason & memory)
 └── React (useDocumentSearch), Vue, Svelte, Vanilla recipes
      │
      ▼
M8: Proof Applications & Verification
 ├── apps/monaco-palette (Proof 1: QuickOpen symbol search)
 ├── apps/log-viewer (Proof 2: 100k data-grid quick find)
 └── Mandatory post-work verification subagent audit
```

### Milestone 1: Specifications, Public Contracts & Monorepo Setup
- **Deliverables**:
  - `src/types.ts`: Define `DocumentRecord`, `FieldDefinition`, `DocumentIndexOptions`, `DocumentSearchOptions`, `DocumentSearchResultItem`, `DocumentSearchResponse`, `HighlightRange`, `MutationBatch`, `MutationResult`.
  - `src/text-profile.ts`: Define `DOC_FORMAT_VERSION = 3`, `SERIALIZED_DOC_MAGIC = 0x55324433`, new error classes (`DuplicateIdError`, `DocumentNotFoundError`).
  - `package.json`: Configure `"files": ["dist", "README.md", "LICENSE"]` and `./worker` export.

### Milestone 2: Document Record Engine & Multi-Field Indexing
- **Deliverables**:
  - `src/document-index.ts`: Implementation of `DocumentIndex<TDoc>`.
  - Field-stratified row packing: primary fields packed first to prevent candidate pool inversion.
  - Candidate capacity scaling (up to 32,768).
  - Parity tests asserting identical ranking between WebGPU and CPU reference engines across multi-field corpora.

### Milestone 3: Unicode-Safe Highlighting Engine
- **Deliverables**:
  - `src/highlight.ts`: `alignHighlights()` and `normalizeWithSourceMap()`.
  - Indentation trim compensation (`leadingTrimOffset`).
  - Atomic character expansion (`ß` $\to$ `ss`, `İ` $\to$ `i + ◌̇`, `ﬁ` $\to$ `fi`).
  - Grapheme cluster boundary guard preventing broken HTML `<mark>` tags.
  - Test matrix covering ASCII, NFD composition, emojis, and case expansions.

### Milestone 4: Batched Dynamic Mutations (`add`, `update`, `remove`)
- **Deliverables**:
  - Cumulative offset preservation: tombstone tracking on CPU candidate readback without in-place offset zeroing.
  - Clamped headroom allocation: bounded by `maxStorageBufferBindingSize`.
  - CPU-driven compaction repacking contiguous arrays at $\ge 25\%$ tombstones.
  - Mutex locking (`queued()`) and two-phase validation for atomic execution.

### Milestone 5: First-Party Worker Client & Protocol
- **Deliverables**:
  - `src/worker/protocol.ts`: Type-safe request/response protocol.
  - `src/worker/worker-client.ts`: `SearchWorkerClient` with immediate `AbortError` rejection on superseded queries.
  - `src/worker/search-worker.ts`: Dedicated worker entrypoint with `isDedicatedWorker` SSR guard.
  - Error rehydration protocol preserving custom error types across `postMessage`.

### Milestone 6: Versioned Snapshot Persistence (IndexedDB-Friendly U2D3)
- **Deliverables**:
  - `src/persistence.ts`: `serializeDocumentIndex()` and `restoreDocumentIndex()`.
  - Little-Endian 48-byte header with circular-dependency-free CRC32.
  - Decoupled document storage support (`docsByteLength = 0`).
  - `src/idb-storage.ts`: Transaction-safe IndexedDB helpers closing connections cleanly in `finally`.

### Milestone 7: Observability & Framework Integration Recipes
- **Deliverables**:
  - Telemetry instrumentation: `fallbackReason`, memory breakdown, mutation epochs.
  - Framework integration recipes in `examples/`: React (`useDocumentSearch`), Vue (`useSearch`), Svelte, Vanilla.

### Milestone 8: Proof Applications & Acceptance Validation
- **Deliverables**:
  - `apps/monaco-palette`: Interactive Monaco QuickOpen file & symbol palette under Turborepo workspace.
  - `apps/log-viewer`: High-throughput 100k data-grid with live find and IndexedDB snapshot restore.
  - Monorepo validation: `bun run check:shaders`, `bun run typecheck`, `bun run build`, `bun run test:mock`, `bun run test:parity`, `bun run test:browser`.
  - Audited v0.3 bundle budget check.
  - Mandatory post-work verification subagent audit.

---

## 5. File Structure Changes

```
packages/webgpu-search/
├── src/
│   ├── index.ts                      # Re-export public v0.3 APIs
│   ├── types.ts                      # Extended with document, search, highlight types
│   ├── text-profile.ts               # Version constants, U2D3 magic, error classes
│   ├── document-index.ts             # NEW: High-level DocumentIndex<TDoc>
│   ├── highlight.ts                  # NEW: Unicode-safe highlight projection engine
│   ├── persistence.ts                # NEW: Little-Endian U2D3 serialization / restore
│   ├── idb-storage.ts                # NEW: Transaction-safe IndexedDB persistence
│   ├── worker/
│   │   ├── protocol.ts               # NEW: Typed worker protocol messages & errors
│   │   ├── worker-client.ts          # NEW: First-party SearchWorkerClient
│   │   └── search-worker.ts          # NEW: Dedicated worker entrypoint (SSR-safe)
│   ├── hybrid-index.ts               # Existing SearchIndex (backward-compatible)
│   ├── webgpu-engine.ts              # Candidate scaling & fallback telemetry
│   ├── cpu-reference.ts              # Multi-field scoring & parity logic
│   └── buffer.ts                     # Clamped headroom & CRC32 utilities
├── package.json                      # Add ./worker export, files: ["dist"]
└── tsup.config.ts                    # Add worker bundle entrypoint

apps/
├── benchmark/                        # Existing benchmark suite
├── monaco-palette/                   # Proof 1: Monaco file/symbol search palette
└── log-viewer/                       # Proof 2: 100k line streaming log viewer

examples/
├── vanilla/                          # Framework recipe: HTML/TS
├── react/                            # Framework recipe: React hook
├── vue/                              # Framework recipe: Vue composable
└── svelte/                           # Framework recipe: Svelte store
```

---

## 6. Verification and Regression Checklist

- [ ] `bun run check:shaders`: Offline WGSL validation via `vgpu check`.
- [ ] `bun run typecheck`: Strict TypeScript checking across monorepo packages and apps.
- [ ] `bun run build`: Clean `tsup` build emitting ESM, CJS, and DTS bundles for both `.` and `./worker`.
- [ ] `bun run test:mock`: Extended mock suite testing document indexing, field weighting, mutations, highlighting, persistence, and worker client.
- [ ] `bun run test:parity`: Parity testing verifying identical ranking between WebGPU and CPU reference across multi-field document datasets.
- [ ] `bun run test:browser`: Headless browser regression suite validating WebGPU execution on real hardware/virtual GPU contexts.
- [ ] Cross-Platform Safety Audit: Core library contains zero un-guarded DOM references (`window`, `document`) to maintain full Web Worker, Node.js, and SSR safety.
- [ ] Mandatory Post-Work Subagent Verification: Independent subagent inspection and sign-off prior to task completion.
