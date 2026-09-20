import React from 'react';
import { useDocumentSearch } from './useDocumentSearch';

export interface SampleDoc {
  id: string;
  title: string;
  category: string;
  description: string;
}

const SAMPLE_DATA: SampleDoc[] = [
  { id: '1', title: 'WebGPU Compute Pipelines', category: 'Graphics', description: 'Deep dive into GPU workgroups and bind groups in WGSL' },
  { id: '2', title: 'Unicode Canonical Decomposition', category: 'Text', description: 'NFD vs NFC composition, astral plane surrogate pairs, and case folding' },
  { id: '3', title: 'IndexedDB Snapshot Persistence', category: 'Storage', description: 'Transaction-safe binary caching using Little-Endian U2D3 headers and CRC32' },
  { id: '4', title: 'Monaco QuickOpen Palette', category: 'IDE', description: 'Sub-millisecond fuzzy file navigation for in-browser developer environments' },
  { id: '5', title: 'Off-Thread Search Workers', category: 'Concurrency', description: 'Non-blocking UI architecture with dedicated Web Workers and AbortControllers' },
];

export const SearchComponent: React.FC = () => {
  const {
    query,
    setQuery,
    results,
    totalMatches,
    isSearching,
    isIndexing,
    isReady,
    error,
    stats,
    engine,
    fallbackReason,
    timings,
    mutationEpoch,
    add,
    remove,
  } = useDocumentSearch<SampleDoc>({
    initialDocs: SAMPLE_DATA,
    indexOptions: {
      fields: [
        { name: 'title', weight: 2.0 },
        { name: 'category', weight: 1.5 },
        { name: 'description', weight: 1.0 },
      ],
    },
    searchOptions: {
      mode: 'fuzzy',
      highlight: true,
      tag: 'mark',
      limit: 10,
    },
  });

  const handleAddSample = async () => {
    const newId = `doc-${Date.now()}`;
    await add({
      id: newId,
      title: `Dynamic Document #${mutationEpoch + 1}`,
      category: 'Dynamic',
      description: 'Added at runtime via atomic batched mutation without reloading the page.',
    });
  };

  const handleRemoveLast = async () => {
    if (results.length > 0) {
      await remove(results[results.length - 1].id);
    }
  };

  return (
    <div style={{ maxWidth: 800, margin: '0 auto', fontFamily: 'system-ui, sans-serif', padding: 20 }}>
      <h2>WebGPU Fuzzy Document Search (React Recipe)</h2>

      {/* Search Input Bar */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 15 }}>
        <input
          type="text"
          value={query}
          onChange={(e: any) => setQuery(e.target.value)}
          placeholder="Type to search (e.g. 'webgpu', 'unicode', 'monaco')..."
          style={{
            flex: 1,
            padding: '10px 14px',
            fontSize: 16,
            borderRadius: 6,
            border: '1px solid #ccc',
          }}
        />
        {query && (
          <button onClick={() => setQuery('')} style={{ padding: '0 15px', cursor: 'pointer' }}>
            Clear
          </button>
        )}
      </div>

      {/* Actions */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 15 }}>
        <button onClick={handleAddSample} disabled={isIndexing} style={{ padding: '6px 12px', cursor: 'pointer' }}>
          + Add Document
        </button>
        <button
          onClick={handleRemoveLast}
          disabled={isIndexing || results.length === 0}
          style={{ padding: '6px 12px', cursor: 'pointer' }}
        >
          - Remove Last Result
        </button>
      </div>

      {/* Telemetry & Observability HUD */}
      <div
        style={{
          background: '#f8f9fa',
          border: '1px solid #e9ecef',
          borderRadius: 8,
          padding: 12,
          marginBottom: 20,
          fontSize: 13,
        }}
      >
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 6 }}>
          <strong>Engine:</strong>
          <span
            style={{
              padding: '2px 8px',
              borderRadius: 4,
              color: '#fff',
              background: engine === 'webgpu' ? '#28a745' : '#e0a800',
              fontWeight: 600,
            }}
          >
            {engine ? engine.toUpperCase() : 'INITIALIZING'}
          </span>
          {fallbackReason && (
            <span style={{ background: '#ffeeba', color: '#856404', padding: '2px 6px', borderRadius: 4 }}>
              Fallback: {fallbackReason}
            </span>
          )}
          <span>Status: {isSearching ? 'Searching...' : isIndexing ? 'Indexing...' : isReady ? 'Ready' : 'Booting'}</span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 8 }}>
          <div><strong>Docs:</strong> {stats ? stats.docCount : 0}</div>
          <div><strong>VRAM:</strong> {stats ? `${(stats.memory.vramBytes / 1024).toFixed(1)} KB` : '0 KB'}</div>
          <div><strong>RAM:</strong> {stats ? `${(stats.memory.ramBytes / 1024).toFixed(1)} KB` : '0 KB'}</div>
          <div><strong>Latency:</strong> {timings ? `${timings.totalMs.toFixed(2)} ms` : '0 ms'}</div>
          <div><strong>Epoch:</strong> {mutationEpoch}</div>
        </div>
      </div>

      {/* Error state */}
      {error && (
        <div style={{ color: '#d9534f', background: '#fdf7f7', padding: 10, borderRadius: 6, marginBottom: 15 }}>
          Error: {error.message}
        </div>
      )}

      {/* Results Header */}
      <div style={{ marginBottom: 10, color: '#666', fontSize: 14 }}>
        Showing {results.length} of {totalMatches} matches
      </div>

      {/* Results List */}
      <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
        {results.map((item) => (
          <li
            key={String(item.id)}
            style={{
              padding: 12,
              borderBottom: '1px solid #eee',
              display: 'flex',
              flexDirection: 'column',
              gap: 4,
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              {item.highlightedText?.title ? (
                <span
                  style={{ fontWeight: 600, fontSize: 16 }}
                  dangerouslySetInnerHTML={{ __html: item.highlightedText.title }}
                />
              ) : (
                <span style={{ fontWeight: 600, fontSize: 16 }}>{item.doc.title}</span>
              )}
              <span style={{ fontSize: 12, color: '#888', background: '#eee', padding: '2px 6px', borderRadius: 4 }}>
                Score: {item.score} ({item.matchedField})
              </span>
            </div>

            <div style={{ fontSize: 12, color: '#0066cc' }}>{item.doc.category}</div>

            {item.highlightedText?.description ? (
              <div
                style={{ fontSize: 14, color: '#444' }}
                dangerouslySetInnerHTML={{ __html: item.highlightedText.description }}
              />
            ) : (
              <div style={{ fontSize: 14, color: '#444' }}>{item.doc.description}</div>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
};
