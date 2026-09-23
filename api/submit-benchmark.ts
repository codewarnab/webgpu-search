/**
 * POST /api/submit-benchmark — crowdsourced benchmark intake (Vercel + Upstash).
 *
 * - Write token stays server-side (env `UPSTASH_REDIS_REST_URL` /
 *   `UPSTASH_REDIS_REST_TOKEN`); the browser never sees it.
 * - GET  ?fingerprint=<fp> -> { known: boolean } (dedup probe for the
 *   "untested hardware" share button).
 * - POST { fingerprint, adapter, gpuTier, browser, os, corpusSize, mode,
 *   query, medians... } -> { ok: true } | { ok: true, duplicate: true }.
 *   First write wins (SET NX); fingerprints indexed in a Redis set.
 * - Best-effort IP rate limit (10/min); fail-open on limiter errors so a
 *   Redis hiccup never blocks the benchmark itself.
 * - Missing env -> 503 { error: 'not-configured' }; the client hides the
 *   share button in that case.
 */

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

const ALLOWED_SIZES = new Set([100000, 200000, 500000, 1000000, 2000000]);
const FP_RE = /^[a-z0-9-]{8,64}$/;
const MAX_BODY_BYTES = 8192;

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

async function upstash(path: string, init?: RequestInit): Promise<{ ok: boolean; data: unknown }> {
  const url = env('UPSTASH_REDIS_REST_URL');
  const token = env('UPSTASH_REDIS_REST_TOKEN');
  if (!url || !token) return { ok: false, data: null };
  const res = await fetch(`${url}${path}`, {
    ...init,
    headers: { authorization: `Bearer ${token}`, ...(init?.headers ?? {}) }
  });
  if (!res.ok) return { ok: false, data: null };
  return { ok: true, data: (await res.json().catch(() => null)) as unknown };
}

function upstashResult(data: unknown): number | string | null {
  if (data && typeof data === 'object' && 'result' in (data as Record<string, unknown>)) {
    const r = (data as Record<string, unknown>).result;
    if (typeof r === 'number' || typeof r === 'string') return r;
  }
  return null;
}

function isFiniteMs(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 60000;
}

function isShortStr(v: unknown, max = 300): v is string {
  return typeof v === 'string' && v.length <= max;
}

function validatePayload(p: Record<string, unknown>): string | null {
  if (typeof p.fingerprint !== 'string' || !FP_RE.test(p.fingerprint)) return 'bad fingerprint';
  if (typeof p.adapter !== 'object' || p.adapter === null) return 'bad adapter';
  const a = p.adapter as Record<string, unknown>;
  for (const k of ['vendor', 'device', 'architecture', 'renderer']) {
    if (!isShortStr(a[k])) return `bad adapter.${k}`;
  }
  if (a['adapterType'] !== undefined && !isShortStr(a['adapterType'], 64)) return 'bad adapter.adapterType';
  const t = p.gpuTier as Record<string, unknown> | undefined;
  if (!t || !['discrete', 'integrated', 'software', 'unknown'].includes(String(t['tier']))) return 'bad gpuTier';
  if (!isShortStr(p['browser'], 32) || !isShortStr(p['os'], 32)) return 'bad browser/os';
  if (typeof p.corpusSize !== 'number' || !ALLOWED_SIZES.has(p.corpusSize)) return 'bad corpusSize';
  if (p['mode'] !== 'fuzzy' && p['mode'] !== 'substring') return 'bad mode';
  if (!isShortStr(p['query'], 120)) return 'bad query';
  for (const k of ['gpuMedianMs', 'gpuP95Ms', 'cpuMedianMs', 'cpuP95Ms']) {
    if (!isFiniteMs(p[k])) return `bad ${k}`;
  }
  for (const k of ['gpuMatches', 'cpuMatches']) {
    if (typeof p[k] !== 'number' || !Number.isInteger(p[k]) || (p[k] as number) < 0 || (p[k] as number) > 2000000) {
      return `bad ${k}`;
    }
  }
  if (p['externals'] !== undefined) {
    if (!Array.isArray(p['externals']) || (p['externals'] as unknown[]).length > 8) return 'bad externals';
    for (const e of p['externals'] as Array<Record<string, unknown>>) {
      if (!isShortStr(e['name'], 32) || !isFiniteMs(e['medianMs']) || !isFiniteMs(e['p95Ms'])) return 'bad external entry';
    }
  }
  return null;
}

async function readJsonBody(req: VercelReq): Promise<Record<string, unknown> | null> {
  if (req.body && typeof req.body === 'object' && !Array.isArray(req.body)) {
    return req.body as Record<string, unknown>;
  }
  if (typeof req.body === 'string') {
    if (req.body.length > MAX_BODY_BYTES) return null;
    try {
      const parsed: unknown = JSON.parse(req.body);
      if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) return parsed as Record<string, unknown>;
    } catch {
      return null;
    }
  }
  return null;
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
    const checked = await upstash(`/sismember/bench:fingerprints/${encodeURIComponent(fp)}`);
    if (!checked.ok) {
      res.status(502).json({ error: 'store-unavailable' });
      return;
    }
    res.status(200).json({ known: upstashResult(checked.data) === 1 });
    return;
  }

  if (req.method !== 'POST') {
    res.status(405).json({ error: 'method-not-allowed' });
    return;
  }

  const payload = await readJsonBody(req);
  if (!payload) {
    res.status(400).json({ error: 'bad json body' });
    return;
  }
  const invalid = validatePayload(payload);
  if (invalid) {
    res.status(400).json({ error: invalid });
    return;
  }
  const fp = payload['fingerprint'] as string;

  // Best-effort rate limit: 10 submits/min/IP. Fail open on store errors.
  try {
    const rlKey = `bench:rl:${clientIp(req).replace(/[^a-zA-Z0-9.:_-]/g, '').slice(0, 64)}`;
    const incr = await upstash(`/incr/${encodeURIComponent(rlKey)}`);
    const count = Number(upstashResult(incr.data) ?? 1);
    if (incr.ok && count === 1) {
      await upstash(`/expire/${encodeURIComponent(rlKey)}/60`);
    }
    if (incr.ok && count > 10) {
      res.status(429).json({ error: 'rate-limited' });
      return;
    }
  } catch {
    // fail open
  }

  const record = JSON.stringify({ ...payload, receivedAt: new Date().toISOString() }).slice(0, MAX_BODY_BYTES);
  const setRes = await upstash(`/set/bench:result:${encodeURIComponent(fp)}/${encodeURIComponent(record)}/NX`);
  if (!setRes.ok) {
    res.status(502).json({ error: 'store-unavailable' });
    return;
  }
  if (upstashResult(setRes.data) !== 'OK') {
    res.status(409).json({ ok: true, duplicate: true });
    return;
  }
  await upstash(`/sadd/bench:fingerprints/${encodeURIComponent(fp)}`);
  res.status(200).json({ ok: true });
}
