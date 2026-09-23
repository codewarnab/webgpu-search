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
const qbShareButton = document.querySelector<HTMLButtonElement>('#qb-share');
const qbShareStatus = document.querySelector<HTMLElement>('#qb-share-status');
const qbGpuFlag = document.querySelector<HTMLElement>('#qb-gpu-flag');

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

async function armShareButton(result: {
  adapter: unknown;
  corpusSize: number;
  mode: string;
  gpuRan: boolean;
}): Promise<void> {
  if (!qbShareRow || !qbShareButton || !qbShareStatus) return;
  qbShareRow.hidden = true;
  qbShareStatus.textContent = '';
  qbShareButton.disabled = false;
  qbShareButton.textContent = 'Share this untested result';
  if (!result.gpuRan) return; // CPU-only runs are not crowdsourced
  try {
    const share = await import('./benchmark-share');
    const full = share.buildSharePayload(result as never);
    if (share.alreadyShared(full.fingerprint)) {
      qbShareRow.hidden = false;
      qbShareStatus.textContent = 'Already shared from this browser. Thanks!';
      qbShareButton.disabled = true;
      return;
    }
    const known = await share.checkKnown(full.fingerprint);
    if (known === true) return; // tested config: stay quiet
    // known === false (untested) or null (backend unknown): offer opt-in share
    qbShareRow.hidden = false;
    if (known === false) {
      qbShareStatus.textContent = 'Untested hardware — consider sharing it.';
    }
    qbShareButton.onclick = async () => {
      qbShareButton.disabled = true;
      qbShareStatus.textContent = 'Sharing…';
      const res = await share.submitSharedResult(full);
      if (res.unconfigured) {
        qbShareStatus.textContent = 'Sharing is not enabled on this deployment yet.';
        qbShareButton.disabled = false;
        return;
      }
      if (res.ok && res.duplicate) {
        qbShareStatus.textContent = 'Already in the dataset. Thanks!';
        share.markShared(full.fingerprint);
        return;
      }
      if (res.ok) {
        qbShareStatus.textContent = 'Shared. Thanks!';
        share.markShared(full.fingerprint);
        return;
      }
      qbShareStatus.textContent = `Share failed: ${res.error ?? 'unknown error'}`;
      qbShareButton.disabled = false;
    };
  } catch {
    // share chunk failed to load: benchmark itself is unaffected
  }
}

function renderQbChart(
  chart: SVGSVGElement,
  entries: Array<{ label: string; medianMs: number; matches: number }>
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
    void armShareButton(result);
    if (qbFields.chart) {
      const entries = [
        ...(result.gpuRan ? [{ label: 'WebGPU', medianMs: result.gpuMedianMs, matches: result.gpuMatches }] : []),
        { label: 'CPU · exact', medianMs: result.cpuMedianMs, matches: result.cpuMatches },
        ...result.externals.filter(external => external.ran)
          .map(external => ({ label: external.name, medianMs: external.medianMs, matches: external.matches }))
      ];
      renderQbChart(qbFields.chart, entries);
    }
    if (qbFields.note) qbFields.note.textContent =
      `CPU route: scoreExactMatches (webgpu-search exact scorer); uFuzzy via CPUEngine, Fuse.js defaults · Corpus: ${result.corpusSize.toLocaleString()} generated records · Query: “${result.query}”. Setup is excluded from warm-search timing.`;
    if (qbFields.dataLink) qbFields.dataLink.href =
      `corpus.html?query=${encodeURIComponent(result.query)}&size=${result.corpusSize}`;
    qbPanel.hidden = false;
    qbStatus.textContent = 'Completed 7 timed searches per available engine.';
  } catch (error) {
    qbStatus.textContent = `Benchmark failed: ${error instanceof Error ? error.message : String(error)}`;
  } finally {
    qbStatus.classList.remove('is-busy');
    qbRun.disabled = false;
    qbRun.textContent = 'Run benchmark';
  }
}

qbRun?.addEventListener('click', () => { void runHomepageBenchmark(); });

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
