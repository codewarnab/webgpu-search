/**
 * Dataset generation, multi-corpus matrix, and U2F2 unicode packing for search benchmarks.
 *
 * M5 (Issue #7): Expands dataset generation beyond ASCII paths to cover three contracted
 * corpora classes:
 * 1. ASCII code paths (src/components/..., 10k to 2M rows): backwards comparison with v0.1.
 * 2. CJK corpus (multi-byte Hanzi/Kana/Hangul terms, 10k to 500k rows): evaluates 32-bit
 *    scalar packing density and multi-workgroup memory access across non-Latin scripts.
 * 3. Emoji & mixed-script corpus (grapheme-heavy, astral scalars, ZWJ sequences, flags,
 *    10k to 200k rows): validates astral code-point throughput and word-boundary penalty absence.
 *
 * Captures packing pipeline phase breakdown: normalizeMs, packMs.
 * Exposes query matrix: short, long, CJK, Emoji, degenerate surviving, and over-limit queries.
 */
import {
  packUnicodeToGPUBuffer,
  serializeUnicodeDataset,
  normalizeText,
} from 'webgpu-search';

export type CorpusType = 'ascii' | 'cjk' | 'emoji';

export interface Dataset {
  size: number;
  corpusType: CorpusType;
  strings: string[];
  /** U2F2 serialized unicode dataset (header + u32 records + u32 offsets). */
  serializedU2F2: ArrayBuffer;
  serializedByteLength: number;
  /** Post-fold code-point total (exact, from the unicode packer). */
  tokenCount: number;
  /** records + offsets bytes actually allocated (no 64 B fiction). */
  packedBytes: number;
  folded: boolean;
  /** Total UTF-16 code units across all strings in the corpus. */
  utf16Units: number;
  /** Total Unicode scalar values (code points) across all strings. */
  codePoints: number;
  /** Pipeline phase breakdown: time in ms spent normalizing strings. */
  normalizeMs: number;
  /** Pipeline phase breakdown: time in ms spent packing tokens and serializing. */
  packMs: number;
  /** Pre-tokenized record tokens (Uint32Array per row) for zero-renorm CPU reference search. */
  recordTokens: Uint32Array[];
}

export interface GenerateDatasetOptions {
  corpusType?: CorpusType;
  onProgress?: (percent: number) => void;
}

export interface QueryMatrixEntry {
  id: string;
  name: string;
  query: string;
  category: 'short' | 'long' | 'cjk' | 'emoji' | 'degenerate' | 'overlimit';
  description: string;
  expectDegenerate?: boolean;
  expectOverLimit?: boolean;
}

export const QUERY_MATRIX: QueryMatrixEntry[] = [
  {
    id: 'ascii-short',
    name: 'Short ASCII ("Auth")',
    query: 'Auth',
    category: 'short',
    description: '3-6 chars common prefix, high match rate across code paths'
  },
  {
    id: 'ascii-medium',
    name: 'Medium ASCII ("Controller")',
    query: 'Controller',
    category: 'short',
    description: 'Standard symbol suffix, baseline comparison for v0.1 parity'
  },
  {
    id: 'ascii-long',
    name: 'Long ASCII ("AuthControllerService")',
    query: 'AuthControllerService',
    category: 'long',
    description: '15-30 chars compound symbol, deeper branch paths in fuzzy match'
  },
  {
    id: 'cjk-short',
    name: 'CJK Short ("ユーザー")',
    query: 'ユーザー',
    category: 'cjk',
    description: 'Kana term ("User"), evaluates multi-byte Hanzi/Kana scalar lookup'
  },
  {
    id: 'cjk-compound',
    name: 'CJK Compound ("注文履歴管理")',
    query: '注文履歴管理',
    category: 'cjk',
    description: 'Hanzi/Kanji compound ("Order History Management"), non-Latin multi-workgroup'
  },
  {
    id: 'emoji-astral',
    name: 'Emoji Astral ("🧑‍💻")',
    query: '🧑‍💻',
    category: 'emoji',
    description: 'Astral scalar (U+1F9D1) + ZWJ + Laptop (U+1F4BB), tests astral scalar packing'
  },
  {
    id: 'emoji-family',
    name: 'Emoji Family ("👨‍👩‍👧‍👦")',
    query: '👨‍👩‍👧‍👦',
    category: 'emoji',
    description: '4 astral scalars + 3 ZWJ sequences, tests multi-scalar grapheme sequence'
  },
  {
    id: 'degenerate-zwj',
    name: 'Degenerate Surviving ZWJ (U+200D)',
    query: '\u200D',
    category: 'degenerate',
    description: 'Surviving single ZWJ token, verifies non-empty search and original echo',
    expectDegenerate: true
  },
  {
    id: 'degenerate-mark',
    name: 'Degenerate Surviving Mark (U+0300)',
    query: '\u0300',
    category: 'degenerate',
    description: 'Surviving single combining grave accent, searches normally as single token',
    expectDegenerate: true
  },
  {
    id: 'degenerate-tatweel',
    name: 'Degenerate Surviving Tatweel (U+0640)',
    query: '\u0640',
    category: 'degenerate',
    description: 'Surviving single Arabic tatweel, searches normally as single token',
    expectDegenerate: true
  },
  {
    id: 'overlimit-135',
    name: 'Over-Limit Query (>128 tokens)',
    query: 'AlphaBetaGammaDeltaEpsilonZetaEtaThetaIotaKappaLambdaMuNuXiOmicronPiRhoSigmaTauUpsilonPhiChiPsiOmega1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzOverLimitCheckBoundaryCapTest',
    category: 'overlimit',
    description: 'Exceeds 128 tokens cap; verifies fast throw or CPU routing per onQueryTooLong',
    expectOverLimit: true
  }
];

// --- ASCII Corpus Vocabularies ---
const ASCII_PREFIXES = [
  'src/components', 'src/views', 'src/utils', 'src/hooks', 'src/store',
  'src/services', 'src/api', 'src/lib', 'crates/core/src', 'crates/engine/src',
  'packages/client', 'packages/server', 'internal/router', 'internal/auth',
  'cmd/server', 'pkg/middleware', 'tests/e2e', 'docs/tutorials', 'scripts/ci',
  'node_modules/@tanstack', 'node_modules/@trpc', 'node_modules/vite', 'vendor/bundle'
];

const ASCII_NOUNS = [
  'User', 'Account', 'Session', 'Profile', 'Auth', 'Token', 'Order', 'Invoice',
  'Payment', 'Billing', 'Product', 'Catalog', 'Item', 'Cart', 'Checkout', 'Delivery',
  'Search', 'Filter', 'Index', 'Query', 'Table', 'Column', 'Database', 'Cache',
  'Router', 'Socket', 'Worker', 'Queue', 'Job', 'Scheduler', 'Metric', 'Telemetry',
  'Log', 'Trace', 'Span', 'Config', 'Setting', 'Theme', 'Color', 'Canvas', 'Shader',
  'Texture', 'Mesh', 'Pipeline', 'Buffer', 'Array', 'Vector', 'Matrix', 'Engine'
];

const ASCII_SUFFIXES = [
  'Controller', 'Service', 'Handler', 'Manager', 'Provider', 'Context', 'Hook',
  'Component', 'View', 'Modal', 'Drawer', 'Dropdown', 'Tooltip', 'Button', 'Input',
  'Validator', 'Serializer', 'Parser', 'Transformer', 'Adapter', 'Repository',
  'Client', 'Gateway', 'Middleware', 'Reducer', 'Dispatcher', 'Observer', 'Factory'
];

const ASCII_EXTENSIONS = ['.ts', '.tsx', '.rs', '.go', '.py', '.js', '.jsx', '.json', '.wgsl', '.css'];

// --- CJK Corpus Vocabularies (Realistic Hanzi, Kana, Hangul terms) ---
const CJK_PREFIXES = [
  'ユーザー管理/認証サービス', '商品カタログ/注文システム', '決済モジュール/請求書管理',
  'データ基盤/パイプライン', '検索エンジン/全文インデックス', '共通基盤/セキュリティ監査',
  '配送追跡/ロジスティクス', 'APIゲートウェイ/ルーティング', 'システム設定/環境変数',
  'リアルタイム分析/イベントストリーム', '用戶中心/權限控制', '訂單中心/交易結算',
  '商品中心/庫存調度', '日誌監控/警報規則', '緩存集群/分散式鎖',
  '공통모듈/보안인증', '결제시스템/전자영수증', '배송조회/물류센터',
  '회원관리/로그인세션', '검색서비스/자동완성'
];

const CJK_NOUNS = [
  'ユーザー情報', 'アカウント', 'プロファイル', 'セッション管理', 'セキュリティ', '認証トークン',
  '請求情報', '支払い明細', '商品在庫', '注文履歴', '買い物カート',
  '検索フィルタ', 'データベース', 'キャッシュ設定', 'メッセージキュー', 'ジョブタスク',
  'メトリクス', 'ログイベント', 'トレース情報', '画面テーマ', 'シェーダー設定',
  '帳戶資訊', '身份驗證', '交易記錄', '發票數據', '商品明細',
  '購物車', '庫存清單', '查詢過濾器', '資料庫連接', '快取策略',
  '사용자정보', '결제내역', '주문목록', '장바구니', '보안토큰',
  '인증세션', '데이터베이스', '캐시저장소', '작업스케줄러', '로그분석'
];

const CJK_SUFFIXES = [
  'コントローラー', 'サービス', 'ハンドラー', 'マネージャー', 'プロバイダー',
  'コンポーネント', 'リポジトリ', 'バリデーター', 'ミドルウェア', 'アダプター',
  '控制器', '業務邏輯', '處理器', '管理員', '數據源',
  '組件視圖', '儲存庫', '校驗器', '中介軟體', '轉接器',
  '컨트롤러', '서비스', '핸들러', '관리자', '프로바이더',
  '컴포넌트', '리포지토리', '유효성검사기', '미들웨어', '어댑터'
];

const CJK_EXTENSIONS = ['.ts', '.tsx', '.json', '.rs', '.go', '.py', '.sql', '.yaml'];

// --- Emoji & Mixed-Script Vocabularies (Astral scalars, ZWJ sequences, Flags, Multi-script) ---
const EMOJI_PREFIXES = [
  '📁projects🚀/src', '📦packages✨/core', '🎨themes🌈/canvas', '🔒security🛡️/auth',
  '⚙️settings🔧/config', '🌐network📡/router', '📊metrics📈/telemetry', '🛒store💳/checkout',
  '👥team🧑‍💻/roles', '💬chat💭/realtime', '🌍geo📍/locator', '⚡fast⚡/accelerator',
  '🎮game🕹️/engine', '🎵audio🎧/synth', '📸media🎬/pipeline', '🔬research🧪/lab'
];

const EMOJI_NOUNS = [
  'User🧑‍💻', 'Admin👩‍💻', 'Family👨‍👩‍👧‍👦', 'SuperHero🦸‍♂️', 'Runner🏃‍♀️',
  'FlagUS🇺🇸', 'FlagJP🇯🇵', 'FlagEU🇪🇺', 'Rocket🚀', 'Fire🔥',
  'Star⭐', 'Sparkles✨', 'Shield🛡️', 'Heart❤️', 'Earth🌍',
  'Lock🔐', 'Key🔑', 'Music🎵', 'Robot🤖', 'Alien👾',
  'Москва_Сервер', 'Ελλάδα_Κώδικας', 'Cairo_القاهرة', 'Tokyo_東京',
  'Seoul_서울', 'Bengaluru_ಬೆಂಗಳೂರು', 'SãoPaulo_Brasil', 'Zürich_Schweiz'
];

const EMOJI_SUFFIXES = [
  'Service⚡', 'Handler🎯', 'Manager💼', 'Provider🌟', 'View📱',
  'Modal💬', 'Controller🎮', 'Database💾', 'Worker👷', 'Router🚦',
  'Validator🔍', 'Transformer🔄', 'Dispatcher📡', 'Factory🏭'
];

const EMOJI_EXTENSIONS = ['.ts', '.tsx', '.rs', '.json', '.go', '.wgsl', '.md'];

function generateSyntheticStrings(
  count: number,
  corpusType: CorpusType,
  onProgress?: (percent: number) => void
): string[] {
  const strings = new Array<string>(count);
  let prefixes: string[];
  let nouns: string[];
  let suffixes: string[];
  let extensions: string[];

  switch (corpusType) {
    case 'cjk':
      prefixes = CJK_PREFIXES;
      nouns = CJK_NOUNS;
      suffixes = CJK_SUFFIXES;
      extensions = CJK_EXTENSIONS;
      break;
    case 'emoji':
      prefixes = EMOJI_PREFIXES;
      nouns = EMOJI_NOUNS;
      suffixes = EMOJI_SUFFIXES;
      extensions = EMOJI_EXTENSIONS;
      break;
    case 'ascii':
    default:
      prefixes = ASCII_PREFIXES;
      nouns = ASCII_NOUNS;
      suffixes = ASCII_SUFFIXES;
      extensions = ASCII_EXTENSIONS;
      break;
  }

  const prefixLen = prefixes.length;
  const nounLen = nouns.length;
  const suffixLen = suffixes.length;
  const extLen = extensions.length;

  const reportInterval = Math.max(10000, Math.floor(count / 10));

  for (let i = 0; i < count; i++) {
    const p = prefixes[i % prefixLen];
    const n1 = nouns[(i * 7 + 3) % nounLen];
    const n2 = nouns[(i * 13 + 11) % nounLen];
    const s = suffixes[(i * 17 + 5) % suffixLen];
    const ext = extensions[(i * 23 + 7) % extLen];

    const path = `${p}/${n1}${n2}${s}_${i}${ext}`;
    strings[i] = path;

    if (onProgress && (i + 1) % reportInterval === 0) {
      onProgress(Math.round(((i + 1) / count) * 50));
    }
  }

  return strings;
}

/**
 * Generate N synthetic strings for the requested corpus class, measure normalization
 * and packing pipeline breakdown, and produce the U2F2 transfer buffer.
 *
 * Overloaded for backwards compatibility:
 * - `generateDataset(count, onProgress)`
 * - `generateDataset(count, options)`
 */
export function generateDataset(
  count: number,
  optionsOrProgress?: GenerateDatasetOptions | ((percent: number) => void)
): Dataset {
  let corpusType: CorpusType = 'ascii';
  let onProgress: ((percent: number) => void) | undefined;

  if (typeof optionsOrProgress === 'function') {
    onProgress = optionsOrProgress;
  } else if (optionsOrProgress && typeof optionsOrProgress === 'object') {
    corpusType = optionsOrProgress.corpusType ?? 'ascii';
    onProgress = optionsOrProgress.onProgress;
  }

  const strings = generateSyntheticStrings(count, corpusType, onProgress);

  const folded = true;
  const tokenRows = new Array<Uint32Array>(count);
  let totalTokens = 0;
  let utf16Units = 0;
  let codePoints = 0;

  // Phase 1: Normalization & Preprocessing timing
  const tNorm0 = performance.now();
  const reportInterval = Math.max(10000, Math.floor(count / 10));

  for (let i = 0; i < count; i++) {
    const s = strings[i]!;
    utf16Units += s.length;
    // Count Unicode code points (astral scalars count as 1, surrogates counted accurately)
    for (let j = 0; j < s.length; ) {
      const cp = s.codePointAt(j)!;
      j += cp > 0xffff ? 2 : 1;
      codePoints++;
    }
    const norm = normalizeText(s, folded);
    tokenRows[i] = norm.tokens;
    totalTokens += norm.tokenCount;

    if (onProgress && (i + 1) % reportInterval === 0) {
      onProgress(50 + Math.round(((i + 1) / count) * 40));
    }
  }
  const normalizeMs = performance.now() - tNorm0;

  // Phase 2: Packing and Serialization timing (zero-renorm path using pre-tokenized rows)
  const tPack0 = performance.now();
  const packed = packUnicodeToGPUBuffer(tokenRows, { folded, totalTokens });
  const serializedU2F2 = serializeUnicodeDataset(packed);
  const packMs = performance.now() - tPack0;

  if (onProgress) {
    onProgress(100);
  }

  return {
    size: count,
    corpusType,
    strings,
    serializedU2F2,
    serializedByteLength: serializedU2F2.byteLength,
    tokenCount: packed.tokenCount,
    packedBytes: packed.combinedByteLength,
    folded,
    utf16Units,
    codePoints,
    normalizeMs,
    packMs,
    recordTokens: tokenRows
  };
}
