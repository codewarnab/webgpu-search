import fs from 'fs';
import puppeteer from 'puppeteer-core';

const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    width: 1360px;
    height: 820px;
    background: #08090d;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #e2e8f0;
    overflow: hidden;
    position: relative;
    padding: 40px 48px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }
  
  /* Ambient lighting & radial mesh */
  .bg-glow-1 {
    position: absolute;
    top: -120px;
    left: 15%;
    width: 650px;
    height: 420px;
    background: radial-gradient(circle, rgba(14, 165, 233, 0.16) 0%, rgba(0,0,0,0) 70%);
    pointer-events: none;
  }
  .bg-glow-2 {
    position: absolute;
    bottom: -100px;
    right: 12%;
    width: 600px;
    height: 400px;
    background: radial-gradient(circle, rgba(16, 185, 129, 0.14) 0%, rgba(0,0,0,0) 70%);
    pointer-events: none;
  }

  /* Grid overlay */
  .grid-pattern {
    position: absolute;
    inset: 0;
    background-image: 
      linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
    background-size: 36px 36px;
    pointer-events: none;
  }

  /* Header */
  .header {
    position: relative;
    z-index: 2;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
  }
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 4px 12px;
    border-radius: 9999px;
    background: rgba(14, 165, 233, 0.12);
    border: 1px solid rgba(14, 165, 233, 0.35);
    color: #38bdf8;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 10px;
  }
  .badge .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #38bdf8;
    box-shadow: 0 0 8px #38bdf8;
  }
  h1 {
    font-size: 32px;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #f8fafc;
    line-height: 1.15;
  }
  h1 span {
    background: linear-gradient(135deg, #38bdf8 0%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .subtitle {
    font-size: 13.5px;
    color: #94a3b8;
    margin-top: 5px;
  }

  /* Top Stat cards */
  .top-stats {
    display: flex;
    gap: 12px;
  }
  .stat-pill {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 10px 18px;
    min-width: 130px;
    text-align: right;
  }
  .stat-pill .val {
    font-size: 24px;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    color: #34d399;
    letter-spacing: -0.02em;
  }
  .stat-pill .val.cyan {
    color: #38bdf8;
  }
  .stat-pill .lbl {
    font-size: 10.5px;
    color: #64748b;
    margin-top: 2px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  /* Main Benchmark Section */
  .main-section {
    position: relative;
    z-index: 2;
    background: rgba(14, 18, 27, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 22px 28px;
    backdrop-filter: blur(16px);
  }

  .chart-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 18px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
  }
  .chart-title {
    font-size: 12.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #cbd5e1;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .chart-title span {
    color: #64748b;
    font-weight: 400;
    text-transform: none;
  }
  .legend {
    display: flex;
    gap: 20px;
    font-size: 12px;
    font-weight: 500;
  }
  .legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .legend-box {
    width: 12px;
    height: 12px;
    border-radius: 3px;
  }
  .legend-box.gpu {
    background: linear-gradient(90deg, #0ea5e9, #10b981);
    box-shadow: 0 0 10px rgba(16, 185, 129, 0.4);
  }
  .legend-box.cpu {
    background: #334155;
  }

  /* Benchmark Rows */
  .rows-container {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .benchmark-row {
    display: grid;
    grid-template-columns: 140px 1fr 120px;
    align-items: center;
    gap: 20px;
  }
  .row-label {
    display: flex;
    flex-direction: column;
  }
  .dataset-count {
    font-size: 14px;
    font-weight: 700;
    color: #f1f5f9;
  }
  .dataset-sub {
    font-size: 11px;
    color: #64748b;
  }

  .bars-track {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .bar-wrapper {
    position: relative;
    height: 20px;
    display: flex;
    align-items: center;
  }
  .bar-bg {
    position: absolute;
    inset: 0;
    background: rgba(255, 255, 255, 0.02);
    border-radius: 5px;
  }
  .bar-fill {
    height: 100%;
    border-radius: 5px;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 10px;
    font-size: 11px;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
  }
  .bar-fill.cpu {
    background: #334155;
    color: #cbd5e1;
  }
  .bar-fill.gpu {
    background: linear-gradient(90deg, #0284c7, #10b981);
    color: #ffffff;
    box-shadow: 0 0 14px rgba(16, 185, 129, 0.35);
  }
  .bar-val-outside {
    position: absolute;
    left: calc(100% + 8px);
    font-size: 11px;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    color: #94a3b8;
    white-space: nowrap;
  }

  .speedup-badge-cell {
    display: flex;
    justify-content: flex-end;
  }
  .speedup-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 10px;
    border-radius: 8px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #34d399;
    font-size: 12.5px;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    box-shadow: 0 0 12px rgba(16, 185, 129, 0.15);
  }
  .speedup-pill.highlight {
    background: rgba(56, 189, 248, 0.15);
    border: 1px solid rgba(56, 189, 248, 0.45);
    color: #38bdf8;
    box-shadow: 0 0 16px rgba(56, 189, 248, 0.25);
  }

  /* Bottom Hardware Cards */
  .footer-grid {
    position: relative;
    z-index: 2;
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 14px;
  }
  .hw-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 13px 16px;
  }
  .hw-tag {
    font-size: 9.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748b;
    margin-bottom: 4px;
  }
  .hw-title {
    font-size: 13px;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .hw-desc {
    font-size: 11px;
    color: #94a3b8;
    line-height: 1.45;
  }
  .hw-desc b {
    color: #34d399;
  }

  /* Footer Meta */
  .meta-bar {
    position: relative;
    z-index: 2;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 10.5px;
    color: #475569;
    padding-top: 8px;
    border-top: 1px solid rgba(255, 255, 255, 0.04);
  }
</style>
</head>
<body>
  <div class="bg-glow-1"></div>
  <div class="bg-glow-2"></div>
  <div class="grid-pattern"></div>

  <!-- Header -->
  <div class="header">
    <div>
      <div class="badge"><span class="dot"></span> In-Browser Compute Benchmark</div>
      <h1>WebGPU <span>Fuzzy Search</span></h1>
      <p class="subtitle">Benchmarked across consumer laptops (Discrete RTX 3050 & Integrated Intel Gen-12) vs @leeoniya/ufuzzy</p>
    </div>
    <div class="top-stats">
      <div class="stat-pill">
        <div class="val">9.8×</div>
        <div class="lbl">Peak Speedup</div>
      </div>
      <div class="stat-pill">
        <div class="val cyan">6.9ms</div>
        <div class="lbl">1M Items Latency</div>
      </div>
      <div class="stat-pill">
        <div class="val" style="color:#f8fafc;">60 FPS</div>
        <div class="lbl">Interactive Budget</div>
      </div>
    </div>
  </div>

  <!-- Main Comparison Chart -->
  <div class="main-section">
    <div class="chart-header">
      <div class="chart-title">
        Fuzzy Match Latency (Lower is Better, ms)
        <span>• Tested with query: "AuthController" • Measured across actual hardware</span>
      </div>
      <div class="legend">
        <div class="legend-item">
          <div class="legend-box cpu"></div>
          <span>uFuzzy (Single-thread CPU)</span>
        </div>
        <div class="legend-item">
          <div class="legend-box gpu"></div>
          <span>WebGPU Retained (NVIDIA RTX 3050 Laptop)</span>
        </div>
      </div>
    </div>

    <div class="rows-container">
      <!-- 100K Items -->
      <div class="benchmark-row">
        <div class="row-label">
          <span class="dataset-count">100,000 strings</span>
          <span class="dataset-sub">Command palette / docs</span>
        </div>
        <div class="bars-track">
          <div class="bar-wrapper">
            <div class="bar-bg"></div>
            <div class="bar-fill cpu" style="width: 5.2%;"></div>
            <span class="bar-val-outside">6.70 ms (CPU)</span>
          </div>
          <div class="bar-wrapper">
            <div class="bar-bg"></div>
            <div class="bar-fill gpu" style="width: 2.6%;"></div>
            <span class="bar-val-outside" style="color:#34d399;">3.33 ms (WebGPU)</span>
          </div>
        </div>
        <div class="speedup-badge-cell">
          <span class="speedup-pill">2.01× faster</span>
        </div>
      </div>

      <!-- 500K Items -->
      <div class="benchmark-row">
        <div class="row-label">
          <span class="dataset-count">500,000 strings</span>
          <span class="dataset-sub">Large codebase / catalog</span>
        </div>
        <div class="bars-track">
          <div class="bar-wrapper">
            <div class="bar-bg"></div>
            <div class="bar-fill cpu" style="width: 25.3%;">32.5 ms</div>
          </div>
          <div class="bar-wrapper">
            <div class="bar-bg"></div>
            <div class="bar-fill gpu" style="width: 2.6%;"></div>
            <span class="bar-val-outside" style="color:#34d399;">3.33 ms (WebGPU)</span>
          </div>
        </div>
        <div class="speedup-badge-cell">
          <span class="speedup-pill highlight">9.75× faster ⚡</span>
        </div>
      </div>

      <!-- 1,000,000 Items -->
      <div class="benchmark-row">
        <div class="row-label">
          <span class="dataset-count">1,000,000 strings</span>
          <span class="dataset-sub">Enterprise logs & telemetry</span>
        </div>
        <div class="bars-track">
          <div class="bar-wrapper">
            <div class="bar-bg"></div>
            <div class="bar-fill cpu" style="width: 49.4%;">63.5 ms</div>
          </div>
          <div class="bar-wrapper">
            <div class="bar-bg"></div>
            <div class="bar-fill gpu" style="width: 5.4%;"></div>
            <span class="bar-val-outside" style="color:#34d399;">6.90 ms (WebGPU)</span>
          </div>
        </div>
        <div class="speedup-badge-cell">
          <span class="speedup-pill highlight">9.21× faster ⚡</span>
        </div>
      </div>

      <!-- 2,000,000 Items -->
      <div class="benchmark-row">
        <div class="row-label">
          <span class="dataset-count">2,000,000 strings</span>
          <span class="dataset-sub">Extreme browser stress test</span>
        </div>
        <div class="bars-track">
          <div class="bar-wrapper">
            <div class="bar-bg"></div>
            <div class="bar-fill cpu" style="width: 100%;">128.6 ms</div>
          </div>
          <div class="bar-wrapper">
            <div class="bar-bg"></div>
            <div class="bar-fill gpu" style="width: 19.6%;">25.2 ms</div>
          </div>
        </div>
        <div class="speedup-badge-cell">
          <span class="speedup-pill">5.10× faster</span>
        </div>
      </div>
    </div>
  </div>

  <!-- Bottom Hardware & Architectural Takeaways -->
  <div class="footer-grid">
    <div class="hw-card">
      <div class="hw-tag">Hardware Run 01</div>
      <div class="hw-title">🎮 NVIDIA RTX 3050 Laptop</div>
      <div class="hw-desc">Sub-7ms queries through <b>1,000,000 records</b>. Up to <b>9.8× speedup</b>. Main thread remains entirely free for UI.</div>
    </div>
    <div class="hw-card">
      <div class="hw-tag">Hardware Run 02</div>
      <div class="hw-title">⚡ Intel Iris Xe (Integrated)</div>
      <div class="hw-desc">Tested on friend's ultrabook: <b>8.8× speedup</b> at 2M items (15.1ms vs 133.2ms). Shines on standard laptop GPUs.</div>
    </div>
    <div class="hw-card">
      <div class="hw-tag">Interactive UI Reality</div>
      <div class="hw-title">⏱️ 16ms Frame Budget</div>
      <div class="hw-desc">CPU drops frames at &gt;200k items. WebGPU stays <b>locked under 7ms</b> at 1M items for instant typing feedback.</div>
    </div>
  </div>

  <!-- Meta footer -->
  <div class="meta-bar">
    <span>Chrome 152 • Windows 11 • Data from multi-laptop run exports • WebGPU Compute Shaders</span>
    <span>Zero server latency • Runs 100% on client GPU</span>
  </div>
</body>
</html>`;

async function main() {
  const browser = await puppeteer.launch({
    executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1360, height: 820, deviceScaleFactor: 2 });
  await page.setContent(html, { waitUntil: 'networkidle0' });
  await page.screenshot({ path: 'benchmark-chart.png' });
  await browser.close();
  console.log('CHART_RENDERED_SUCCESSFULLY');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
