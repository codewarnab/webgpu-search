export type SymbolKind =
  | 'file'
  | 'class'
  | 'function'
  | 'interface'
  | 'method'
  | 'shader'
  | 'type'
  | 'constant';

export interface MonacoFileRecord {
  id: string;
  path: string;
  filename: string;
  symbols: string;
  type: SymbolKind;
  language: string;
  description: string;
  sizeBytes: number;
  lineCount: number;
}

export interface MonacoPaletteSearchResult {
  id: string;
  score: number;
  matchedField: string;
  doc: MonacoFileRecord;
  highlightedText?: Record<string, string>;
  highlights?: Record<string, Array<{ start: number; end: number }>>;
}

export interface MonacoPaletteState {
  records: MonacoFileRecord[];
  query: string;
  mode: 'fuzzy' | 'substring';
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
  activeFile: MonacoFileRecord | null;
}
