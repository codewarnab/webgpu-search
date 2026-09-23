import './fonts.css';
import './style.css';
import { formatMs } from './format';

const copyButton = document.querySelector<HTMLButtonElement>('#copy-install');

const qbSize = document.querySelector<HTMLSelectElement>('#qb-size');
const qbQuery = document.querySelector<HTMLInputElement>('#qb-query');
const qbRun = document.querySelector<HTMLButtonElement>('#qb-run');
const qbStatus = document.querySelector<HTMLElement>('#qb-status');
const qbPanel = document.querySelector<HTMLElement>('#qb-results');
const qbFields = {
  device: document.querySelector<HTMLElement>('#qb-device'),
  note: document.querySelector<HTMLElement>('#qb-note'),
  chart: document.querySelector<SVGSVGElement>('#qb-chart'),
  dataLink: document.querySelector<HTMLAnchorElement>('#qb-data-link')
};

const qbShareRow = document.querySelector<HTMLElement>('#qb-share-row');
const qbShareStatus = document.querySelector<HTMLElement>('#qb-share-status');
const qbGpuFlag = document.querySelector<HTMLElement>('#qb-gpu-flag');
const qbHistoryWrap = document.querySelector<HTMLElement>('#qb-history-wrap');
const qbHistoryList = document.querySelector<HTMLElement>('#qb-history-list');
const qbHistoryCount = document.querySelector<HTMLElement>('#qb-history-count');
const qbHistoryClear = document.querySelector<HTMLButtonElement>('#qb-history-clear');

function renderHistory(): void {
  if (!qbHistoryWrap || !qbHistoryList || !qbHistoryCount) return;
  qbHistoryList.replaceChildren();
  void import('./benchmark-history').then(history => {
    const entries = history.loadHistory();
    if (entries.length === 0) {
      qbHistoryWrap!.hidden = true;
      return;
    }
    qbHistoryWrap!.hidden = false;
    qbHistoryCount!.textContent = `(${entries.length})`;
    for (const entry of entries) {
      const li = document.createElement('li');
      const when = new Date(entry.at).toLocaleString();
      li.textContent = entry.gpuRan
        ? `${when} · ${(entry.corpusSize / 1000).toLocaleString()}k · GPU ${formatMs(entry.gpuMedianMs)} · CPU ${formatMs(entry.cpuMedianMs)} · ${entry.gpuTier}`
        : `${when} · ${(entry.corpusSize / 1000).toLocaleString()}k · CPU-only ${formatMs(entry.cpuMedianMs)}`;
      qbHistoryList!.append(li);
    }
  }).catch(() => {
    // history chunk failed: benchmark itself is unaffected
  });
}

qbHistoryClear?.addEventListener('click', () => {
  void import('./benchmark-history').then(history => {
    history.clearHistory();
    renderHistory();
  }).catch(() => {});
});

function renderGpuFlag(
  tier: { tier: string; confidence: string },
  deviceLabel: string,
  gpuRan: boolean
): void {
  if (!qbGpuFlag) return;
  qbGpuFlag.hidden = true;
  qbGpuFlag.replaceChildren();
  qbGpuFlag.className = 'gpu-flag';
  if (!gpuRan) return;
  if (tier.tier === 'integrated') {
    qbGpuFlag.hidden = false;
    qbGpuFlag.classList.add('is-warn');
    const title = document.createElement('strong');
    title.textContent = '⚠ Integrated GPU — these numbers reflect the iGPU, not a discrete GPU.';
    const hint = document.createElement('span');
    hint.textContent =
      'This device reported an integrated adapter. If you have a discrete GPU, Chrome likely used the same adapter as page compositing (powerPreference is only a hint).';
    const details = document.createElement('details');
    const summary = document.createElement('summary');
    summary.textContent = 'How to switch to the discrete GPU';
    const list = document.createElement('ol');
    for (const step of [
      'Windows: Settings → System → Display → Graphics → add Chrome → High performance, then relaunch Chrome.',
      'Verify at chrome://gpu that the discrete GPU is listed for WebGPU/Graphics.',
      'Optional: enable chrome://flags/#force-high-performance-gpu and relaunch.',
      'Run on AC power with Battery Saver off, update GPU drivers, then re-run.',
      `Adapter: ${deviceLabel}`
    ]) {
      const li = document.createElement('li');
      li.textContent = step;
      list.append(li);
    }
    details.append(summary, list);
    qbGpuFlag.append(title, hint, details);
  } else if (tier.tier === 'software') {
    qbGpuFlag.hidden = false;
    qbGpuFlag.classList.add('is-warn');
    const title = document.createElement('strong');
    title.textContent = '⚠ Software renderer — not dedicated-GPU performance.';
    qbGpuFlag.append(title);
  } else if (tier.tier === 'unknown') {
    qbGpuFlag.hidden = false;
    const note = document.createElement('span');
    note.textContent = 'GPU type could not be identified — treat this run as unqualified.';
    qbGpuFlag.append(note);
  }
}

async function autoShareResult(result: {
  adapter: unknown;
  corpusSize: number;
  mode: string;
  gpuRan: boolean;
}): Promise<void> {
  if (!qbShareRow || !qbShareStatus) return;
  qbShareRow.hidden = true;
  qbShareStatus.textContent = '';
  if (!result.gpuRan) return; // CPU-only runs are not crowdsourced
  try {
    const share = await import('./benchmark-share');
    const full = share.buildSharePayload(result as never);
    if (share.alreadyShared(full.fingerprint)) return; // sent before from this browser
    const res = await share.submitSharedResult(full);
    if (res.unconfigured) return; // backend not enabled: stay silent
    if (!res.ok) return; // network/rate-limit hiccup: benchmark itself is unaffected
    share.markShared(full.fingerprint);
    qbShareRow.hidden = false;
    qbShareStatus.textContent = res.duplicate
      ? 'Untested hardware — already in the community dataset.'
      : 'Untested hardware — result shared to the community dataset. Thanks!';
  } catch {
    // share chunk failed to load: benchmark itself is unaffected
  }
}

function renderQbChart(
  chart: SVGSVGElement,
  entries: Array<{ label: string; medianMs: number; matches: number; hint?: string }>
): void {
  chart.replaceChildren();
  const width = 900;
  const rowHeight = 26;
  const height = 58 + entries.length * rowHeight;
  chart.setAttribute('viewBox', `0 0 ${width} ${height}`);
  const max = Math.max(0.01, ...entries.map(entry => entry.medianMs));
  const fastest = Math.min(...entries.map(entry => entry.medianMs));
  const scale = (width - 330) / max;
  const svgEl = (name: string, attrs: Record<string, string>, text?: string): SVGElement => {
    const element = document.createElementNS('http://www.w3.org/2000/svg', name);
    for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
    if (text !== undefined) element.textContent = text;
    return element;
  };
  chart.append(svgEl('text', { x: '18', y: '26', fill: '#29292d', 'font-size': '13', 'font-family': 'sans-serif', 'font-weight': '600' }, 'Median search latency'));
  const compactMatches = (count: number): string =>
    count >= 10000 ? `${Math.round(count / 1000)}k` : count.toLocaleString();
  entries.forEach((entry, index) => {
    const y = 40 + index * rowHeight;
    const isFastest = entry.medianMs <= fastest;
    chart.append(svgEl('text', { x: '18', y: `${y + 12}`, fill: isFastest ? '#29292d' : '#68696e', 'font-size': '11', 'font-family': 'sans-serif', 'font-weight': isFastest ? '600' : '400' }, entry.label));
    if (entry.hint) {
      const info = svgEl('text', { x: `${18 + entry.label.length * 6.2 + 4}`, y: `${y + 12}`, fill: '#1769e0', 'font-size': '10', 'font-family': 'sans-serif', cursor: 'help' }, 'ⓘ');
      info.append(svgEl('title', {}, entry.hint));
      chart.append(info);
    }
    const barWidth = Math.max(2, entry.medianMs * scale);
    chart.append(svgEl('rect', { x: '210', y: `${y + 3}`, width: `${barWidth}`, height: '11', rx: '2', fill: isFastest ? '#252529' : '#a5a5aa' }));
    const value = `${formatMs(entry.medianMs)} · ${compactMatches(entry.matches)} matches`;
    const textX = 220 + barWidth;
    const anchor = textX + value.length * 6.6 > width - 8 ? 'end' : 'start';
    chart.append(svgEl('text', { x: `${anchor === 'end' ? width - 8 : textX}`, y: `${y + 12}`, fill: '#55565b', 'font-size': '11', 'font-family': 'monospace', 'text-anchor': anchor }, value));
  });
}

async function runHomepageBenchmark(): Promise<void> {
  if (!qbSize || !qbQuery || !qbRun || !qbStatus || !qbPanel) return;
  const query = qbQuery.value.trim();
  if (!query) {
    qbStatus.textContent = 'Enter a query before running the benchmark.';
    qbQuery.focus();
    return;
  }
  qbRun.disabled = true;
  qbRun.textContent = 'Running…';
  qbPanel.hidden = true;
  qbStatus.classList.add('is-busy');
  try {
    qbStatus.textContent = 'Loading the benchmark engine…';
    const { runQuickCompare } = await import('./quick-benchmark');
    const result = await runQuickCompare(Number(qbSize.value), 'fuzzy', query, message => {
      if (qbStatus) qbStatus.textContent = message;
    });
    if (qbFields.device) qbFields.device.textContent = result.deviceLabel;
    renderGpuFlag(result.gpuTier, result.deviceLabel, result.gpuRan);
    void autoShareResult(result);
    if (qbFields.chart) {
      const entries = [
        ...(result.gpuRan ? [{ label: 'WebGPU', medianMs: result.gpuMedianMs, matches: result.gpuMatches, hint: 'Your GPU running the search in parallel. Matching CPU · exact totals means the result is correct.' }] : []),
        { label: 'CPU · exact', medianMs: result.cpuMedianMs, matches: result.cpuMatches, hint: 'This library\u2019s own CPU reference scorer — the correctness baseline the GPU result is checked against.' },
        ...result.externals.filter(external => external.ran)
          .map(external => ({
            label: external.name,
            medianMs: external.medianMs,
            matches: external.matches,
            hint: external.name === 'uFuzzy'
              ? 'Popular third-party CPU fuzzy library, shown for context. Its scores are not comparable.'
              : 'Popular third-party fuzzy library with its own match definition — different match counts are expected.'
          }))
      ];
      renderQbChart(qbFields.chart, entries);
    }
    if (qbFields.note) qbFields.note.textContent =
      `CPU route: scoreExactMatches (webgpu-search exact scorer); uFuzzy via CPUEngine, Fuse.js defaults · Corpus: ${result.corpusSize.toLocaleString()} generated records · Query: “${result.query}”. Setup is excluded from warm-search timing.`;
    if (qbFields.dataLink) qbFields.dataLink.href =
      `corpus.html?query=${encodeURIComponent(result.query)}&size=${result.corpusSize}`;
    qbPanel.hidden = false;
    qbStatus.textContent = 'Completed 7 timed searches per available engine.';
    void import('./benchmark-history').then(history => {
      history.saveRun(result);
      renderHistory();
    }).catch(() => {
      renderHistory();
    });
  } catch (error) {
    qbStatus.textContent = `Benchmark failed: ${error instanceof Error ? error.message : String(error)}`;
  } finally {
    qbStatus.classList.remove('is-busy');
    qbRun.disabled = false;
    qbRun.textContent = 'Run benchmark';
  }
}

qbRun?.addEventListener('click', () => { void runHomepageBenchmark(); });
renderHistory();

// Prefetch the benchmark chunk when its section scrolls into view so the
// first Run click feels instant. Initial page load stays lean regardless.
const benchSection = document.querySelector('#benchmark');
if (benchSection && 'IntersectionObserver' in window) {
  const benchObserver = new IntersectionObserver(entries => {
    if (entries.some(entry => entry.isIntersecting)) {
      void import('./quick-benchmark');
      benchObserver.disconnect();
    }
  });
  benchObserver.observe(benchSection);
}

const featuresMore = document.querySelector<HTMLButtonElement>('#features-more');
featuresMore?.addEventListener('click', () => {
  const extras = [...document.querySelectorAll<HTMLElement>('.feature.is-extra')];
  const expanding = extras.some(extra => extra.hidden);
  for (const extra of extras) extra.hidden = !expanding;
  if (featuresMore) {
    featuresMore.textContent = expanding ? 'Show less ↑' : 'Show more ↓';
    featuresMore.setAttribute('aria-expanded', String(expanding));
  }
});

copyButton?.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText('npm install webgpu-search');
    copyButton.textContent = 'Copied';
    window.setTimeout(() => { copyButton.textContent = 'Copy'; }, 1500);
  } catch {
    copyButton.textContent = 'Select command';
  }
});
