export type DocSection =
  | 'Guide'
  | 'API'
  | 'Storage'
  | 'Reliability'
  | 'Reference';

export interface DocPageRecord {
  id: string;
  path: string;
  title: string;
  section: DocSection;
  content: string;
  tags: string;
  version: string;
  readingMinutes: number;
}

export interface DocsSearchResult {
  id: string;
  score: number;
  matchedField: string;
  doc: DocPageRecord;
  highlightedText?: Record<string, string>;
  highlights?: Record<string, Array<{ start: number; end: number }>>;
}

export interface DocsSearchState {
  records: DocPageRecord[];
  query: string;
  mode: 'fuzzy' | 'substring' | 'prefix' | 'token';
  engine: 'webgpu' | 'cpu';
  useWorker: boolean;
  highlight: boolean;
  activeResultIndex: number;
  totalMatches: number;
  searchDurationMs: number;
  vramBytes: number;
  ramBytes: number;
  mutationEpoch: number;
  tombstoneCount: number;
  isSearching: boolean;
  activeDoc: DocPageRecord | null;
}
