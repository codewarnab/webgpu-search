/**
 * POST /api/submit-benchmark — crowdsourced benchmark intake (Vercel + Upstash).
 *
 * - Write token stays server-side (env `UPSTASH_REDIS_REST_URL` /
 *   `UPSTASH_REDIS_REST_TOKEN`); the browser never sees it.
 * - GET  ?fingerprint=<fp> -> { known: boolean } (has any submission been
 *   stored for this hardware/workload fingerprint).
 * - POST { fingerprint, adapter, gpuTier, browser, os, corpusSize, mode,
 *   query, medians... } -> 200 { ok: true } | 409 { ok: true, duplicate: true }.
 *   Submissions are appended to a per-fingerprint capped list
 *   (`bench:results:<fp>`, newest RESULTS_PER_FP kept) so no single
 *   submitter can pre-claim a fingerprint. Dedup is per (fingerprint,
 *   hashed IP) via SET NX with a TTL; fingerprints are indexed in the
 *   `bench:fingerprints` set.
 * - Only whitelisted, validated fields are persisted; bodies whose JSON
 *   encoding exceeds 8 KiB are rejected with 413 (never truncated).
 * - Best-effort IP rate limit (10/min); fail-open on limiter errors so a
 *   Redis hiccup never blocks the benchmark itself. Store errors on the
 *   read/write path return 502.
 * - Missing env -> 503 { error: 'not-configured' }.
 */

// Node global (no @types/node in this workspace).
declare const process: { env: Record<string, string | undefined> };

type VercelReq = {
  method?: string;
  query?: Record<string, string | string[] | undefined>;
  headers?: Record<string, string | string[] | undefined>;
  body?: unknown;
};

type VercelRes = {
  status: (code: number) => VercelRes;
  json: (data: unknown) => void;
  setHeader: (name: string, value: string) => void;
};

type Obj = Record<string, unknown>;
type RedisCommand = Array<string | number>;
type StoreReply = { ok: boolean; result: unknown };

const ALLOWED_SIZES = new Set([100000, 200000, 500000, 1000000, 2000000]);
const GPU_TIERS = ['discrete', 'integrated', 'software', 'unknown'];
const FP_RE = /^[a-z0-9-]{8,64}$/;
const MAX_BODY_BYTES = 8192;
const RESULTS_PER_FP = 50;
const DEDUP_TTL_S = 60 * 60 * 24 * 30;
const RL_WINDOW_S = 60;
const RL_MAX = 10;

function env(name: string): string {
  return (process.env[name] ?? '').trim().replace(/\/$/, '');
}

function clientIp(req: VercelReq): string {
  const h = req.headers ?? {};
  const xff = h['x-forwarded-for'] ?? h['X-Forwarded-For'];
  const first = Array.isArray(xff) ? xff[0] : xff;
  if (typeof first === 'string' && first) return first.split(',')[0]!.trim().slice(0, 64);
  const real = h['x-real-ip'];
  const r = Array.isArray(real) ? real[0] : real;
  return (typeof r === 'string' ? r : 'unknown').slice(0, 64);
}

async function sha256Hex(input: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(input));
  return Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('');
}

function byteLength(s: string): number {
  return new TextEncoder().encode(s).length;
}

/** POSTs a JSON body to Upstash REST; network/HTTP/parse failures -> null. */
async function upstashPost(path: string, body: unknown): Promise<unknown> {
  const url = env('UPSTASH_REDIS_REST_URL');
  const token = env('UPSTASH_REDIS_REST_TOKEN');
  if (!url || !token) return null;
  try {
    const res = await fetch(`${url}${path}`, {
      method: 'POST',
      headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json' },
      body: JSON.stringify(body)
    });
    if (!res.ok) return null;
    return (await res.json().catch(() => null)) as unknown;
  } catch {
    return null;
  }
}

function toReply(data: unknown): StoreReply {
  if (data && typeof data === 'object' && !Array.isArray(data)) {
    const d = data as Obj;
    if ('error' in d) return { ok: false, result: null };
    if ('result' in d) return { ok: true, result: d.result };
  }
  return { ok: false, result: null };
}

async function redis(command: RedisCommand): Promise<StoreReply> {
  return toReply(await upstashPost('', command));
}

async function pipeline(commands: RedisCommand[]): Promise<StoreReply[] | null> {
  const data = await upstashPost('/pipeline', commands);
  if (!Array.isArray(data) || data.length !== commands.length) return null;
  return data.map(toReply);
}

function isObj(v: unknown): v is Obj {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

function isFiniteMs(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 60000;
}

function isCount(v: unknown): v is number {
  return typeof v === 'number' && Number.isInteger(v) && v >= 0 && v <= 2000000;
}

function isShortStr(v: unknown, max = 300): v is string {
  return typeof v === 'string' && v.length <= max;
}

/** Validates and returns a record built ONLY from whitelisted fields. */
function buildRecord(p: Obj): { record: Obj } | { error: string } {
  if (typeof p.fingerprint !== 'string' || !FP_RE.test(p.fingerprint)) return { error: 'bad fingerprint' };
  if (!isObj(p.adapter)) return { error: 'bad adapter' };
  const a = p.adapter;
  for (const k of ['vendor', 'device', 'architecture', 'renderer']) {
    if (!isShortStr(a[k])) return { error: `bad adapter.${k}` };
  }
  if (a.adapterType !== undefined && !isShortStr(a.adapterType, 64)) return { error: 'bad adapter.adapterType' };
  if (!isObj(p.gpuTier) || typeof p.gpuTier.tier !== 'string' || !GPU_TIERS.includes(p.gpuTier.tier)) {
    return { error: 'bad gpuTier' };
  }
  const t = p.gpuTier;
  if (t.confidence !== undefined && !isShortStr(t.confidence, 32)) return { error: 'bad gpuTier.confidence' };
  if (t.reason !== undefined && !isShortStr(t.reason, 300)) return { error: 'bad gpuTier.reason' };
  if (!isShortStr(p.browser, 32) || !isShortStr(p.os, 32)) return { error: 'bad browser/os' };
  if (typeof p.corpusSize !== 'number' || !ALLOWED_SIZES.has(p.corpusSize)) return { error: 'bad corpusSize' };
  if (p.mode !== 'fuzzy' && p.mode !== 'substring') return { error: 'bad mode' };
  if (!isShortStr(p.query, 120)) return { error: 'bad query' };
  for (const k of ['gpuMedianMs', 'gpuP95Ms', 'cpuMedianMs', 'cpuP95Ms']) {
    if (!isFiniteMs(p[k])) return { error: `bad ${k}` };
  }
  for (const k of ['gpuMatches', 'cpuMatches']) {
    if (!isCount(p[k])) return { error: `bad ${k}` };
  }
  const externals: Obj[] = [];
  if (p.externals !== undefined) {
    if (!Array.isArray(p.externals) || p.externals.length > 8) return { error: 'bad externals' };
    for (const e of p.externals as unknown[]) {
      if (!isObj(e) || !isShortStr(e.name, 32) || !isFiniteMs(e.medianMs) || !isFiniteMs(e.p95Ms)) {
        return { error: 'bad external entry' };
      }
      if (e.matches !== undefined && !isCount(e.matches)) return { error: 'bad external entry' };
      if (e.ran !== undefined && typeof e.ran !== 'boolean') return { error: 'bad external entry' };
      externals.push({
        name: e.name,
        medianMs: e.medianMs,
        p95Ms: e.p95Ms,
        ...(e.matches !== undefined ? { matches: e.matches } : {}),
        ...(e.ran !== undefined ? { ran: e.ran } : {})
      });
    }
  }
  return {
    record: {
      fingerprint: p.fingerprint,
      adapter: {
        vendor: a.vendor,
        device: a.device,
        architecture: a.architecture,
        renderer: a.renderer,
        ...(a.adapterType !== undefined ? { adapterType: a.adapterType } : {})
      },
      gpuTier: {
        tier: t.tier,
        ...(t.confidence !== undefined ? { confidence: t.confidence } : {}),
        ...(t.reason !== undefined ? { reason: t.reason } : {})
      },
      browser: p.browser,
      os: p.os,
      corpusSize: p.corpusSize,
      mode: p.mode,
      query: p.query,
      gpuMedianMs: p.gpuMedianMs,
      gpuP95Ms: p.gpuP95Ms,
      gpuMatches: p.gpuMatches,
      cpuMedianMs: p.cpuMedianMs,
      cpuP95Ms: p.cpuP95Ms,
      cpuMatches: p.cpuMatches,
      externals,
      receivedAt: new Date().toISOString()
    }
  };
}

type BodyResult = { payload: Obj } | { status: 400 | 413; error: string };

function readJsonBody(req: VercelReq): BodyResult {
  let parsed: unknown = req.body;
  if (typeof req.body === 'string') {
    if (byteLength(req.body) > MAX_BODY_BYTES) return { status: 413, error: 'body-too-large' };
    try {
      parsed = JSON.parse(req.body);
    } catch {
      return { status: 400, error: 'bad json body' };
    }
  }
  if (!isObj(parsed)) return { status: 400, error: 'bad json body' };
  // Vercel usually hands us an already-parsed object: measure its encoding.
  let encoded: string;
  try {
    encoded = JSON.stringify(parsed);
  } catch {
    return { status: 400, error: 'bad json body' };
  }
  if (byteLength(encoded) > MAX_BODY_BYTES) return { status: 413, error: 'body-too-large' };
  return { payload: parsed };
}

export default async function handler(req: VercelReq, res: VercelRes): Promise<void> {
  res.setHeader('cache-control', 'no-store');
  if (!env('UPSTASH_REDIS_REST_URL') || !env('UPSTASH_REDIS_REST_TOKEN')) {
    res.status(503).json({ error: 'not-configured' });
    return;
  }

  if (req.method === 'GET') {
    const raw = req.query?.['fingerprint'];
    const fp = Array.isArray(raw) ? raw[0] : raw;
    if (!fp || !FP_RE.test(fp)) {
      res.status(400).json({ error: 'bad fingerprint' });
      return;
    }
    const checked = await redis(['SISMEMBER', 'bench:fingerprints', fp]);
    if (!checked.ok) {
      res.status(502).json({ error: 'store-unavailable' });
      return;
    }
    res.status(200).json({ known: checked.result === 1 });
    return;
  }

  if (req.method !== 'POST') {
    res.status(405).json({ error: 'method-not-allowed' });
    return;
  }

  const body = readJsonBody(req);
  if ('error' in body) {
    res.status(body.status).json({ error: body.error });
    return;
  }
  const built = buildRecord(body.payload);
  if ('error' in built) {
    res.status(400).json({ error: built.error });
    return;
  }
  const fp = built.record.fingerprint as string;
  const ipHash = (await sha256Hex(`${env('UPSTASH_REDIS_REST_TOKEN')}|${clientIp(req)}`)).slice(0, 32);

  // Best-effort rate limit: RL_MAX submits/min/IP. EXPIRE NX runs on every
  // request so a key whose TTL was lost still expires. Fail open on errors.
  const rlKey = `bench:rl:${ipHash}`;
  const rl = await pipeline([
    ['INCR', rlKey],
    ['EXPIRE', rlKey, RL_WINDOW_S, 'NX']
  ]);
  if (rl && !rl[1]!.ok) {
    // Store without EXPIRE NX support: plain EXPIRE so the key can't stick.
    await redis(['EXPIRE', rlKey, RL_WINDOW_S]);
  }
  if (rl && rl[0]!.ok && typeof rl[0]!.result === 'number' && rl[0]!.result > RL_MAX) {
    res.status(429).json({ error: 'rate-limited' });
    return;
  }

  // Per-(fingerprint, submitter) dedup: one submitter can't block others.
  const dedupKey = `bench:dedup:${fp}:${ipHash}`;
  const claim = await redis(['SET', dedupKey, '1', 'EX', DEDUP_TTL_S, 'NX']);
  if (!claim.ok) {
    res.status(502).json({ error: 'store-unavailable' });
    return;
  }
  if (claim.result !== 'OK') {
    res.status(409).json({ ok: true, duplicate: true });
    return;
  }

  const listKey = `bench:results:${fp}`;
  const stored = await pipeline([
    ['RPUSH', listKey, JSON.stringify(built.record)],
    ['LTRIM', listKey, -RESULTS_PER_FP, -1],
    ['SADD', 'bench:fingerprints', fp]
  ]);
  if (!stored || !stored.every(r => r.ok)) {
    // Release the dedup claim so the client can retry later.
    await redis(['DEL', dedupKey]);
    res.status(502).json({ error: 'store-unavailable' });
    return;
  }
  res.status(200).json({ ok: true });
}
