import puppeteer from 'puppeteer-core';

const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    width: 1200px;
    height: 680px;
    background: #ffffff;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #0f172a;
    padding: 48px 56px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }

  /* Header Section */
  .header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding-bottom: 24px;
  }
  .title-group {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  h1 {
    font-size: 26px;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.03em;
  }
  .subtitle {
    font-size: 13px;
    color: #64748b;
    font-weight: 400;
  }

  /* Top Right Clean Legend */
  .legend {
    display: flex;
    align-items: center;
    gap: 20px;
    padding-top: 4px;
  }
  .legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12.5px;
    font-weight: 500;
    color: #475569;
  }
  .legend-color {
    width: 10px;
    height: 10px;
    border-radius: 2px;
  }
  .legend-color.cpu {
    background: #e2e8f0;
  }
  .legend-color.gpu {
    background: #2563eb;
  }

  /* Main Chart Area */
  .chart-container {
    display: flex;
    flex-direction: column;
    gap: 24px;
    margin: auto 0;
  }

  .chart-row {
    display: grid;
    grid-template-columns: 140px 1fr 110px;
    align-items: center;
    gap: 24px;
  }

  .row-label {
    font-size: 13.5px;
    font-weight: 600;
    color: #0f172a;
    letter-spacing: -0.01em;
  }

  .bars-group {
    display: flex;
    flex-direction: column;
    gap: 7px;
    width: 100%;
  }

  .bar-line {
    display: flex;
    align-items: center;
    gap: 10px;
    height: 20px;
  }

  .bar-fill {
    height: 100%;
    border-radius: 4px;
    min-width: 4px;
  }
  .bar-fill.cpu {
    background: #e2e8f0;
  }
  .bar-fill.gpu {
    background: #2563eb;
  }

  .bar-value {
    font-size: 12px;
    font-weight: 500;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }
  .bar-value.cpu {
    color: #64748b;
  }
  .bar-value.gpu {
    color: #1d4ed8;
    font-weight: 600;
  }

  .speedup-col {
    display: flex;
    justify-content: flex-end;
  }
  .speedup-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 4px 10px;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 6px;
    color: #2563eb;
    font-size: 12px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.01em;
  }

  /* Grid Scale Line at Bottom */
  .scale-track {
    display: grid;
    grid-template-columns: 140px 1fr 110px;
    gap: 24px;
    padding-top: 14px;
    border-top: 1px solid #f1f5f9;
  }
  .scale-numbers {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: #94a3b8;
    font-weight: 400;
    padding-right: 80px;
  }

  /* Footer */
  .footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 11.5px;
    color: #94a3b8;
    padding-top: 16px;
  }
</style>
</head>
<body>
  <!-- Header -->
  <div class="header">
    <div class="title-group">
      <h1>WebGPU vs CPU Fuzzy Search</h1>
      <div class="subtitle">Query latency across dataset sizes (lower is better) • Tested on laptop hardware</div>
    </div>
    <div class="legend">
      <div class="legend-item">
        <div class="legend-color cpu"></div>
        <span>uFuzzy (CPU)</span>
      </div>
      <div class="legend-item">
        <div class="legend-color gpu"></div>
        <span>WebGPU</span>
      </div>
    </div>
  </div>

  <!-- Chart Rows (140ms max scale = 100%) -->
  <div class="chart-container">
    <!-- 100,000 items -->
    <div class="chart-row">
      <div class="row-label">100,000 items</div>
      <div class="bars-group">
        <div class="bar-line">
          <div class="bar-fill cpu" style="width: 4.8%;"></div>
          <span class="bar-value cpu">6.7 ms</span>
        </div>
        <div class="bar-line">
          <div class="bar-fill gpu" style="width: 2.4%;"></div>
          <span class="bar-value gpu">3.3 ms</span>
        </div>
      </div>
      <div class="speedup-col">
        <span class="speedup-badge">2.0× faster</span>
      </div>
    </div>

    <!-- 500,000 items -->
    <div class="chart-row">
      <div class="row-label">500,000 items</div>
      <div class="bars-group">
        <div class="bar-line">
          <div class="bar-fill cpu" style="width: 23.2%;"></div>
          <span class="bar-value cpu">32.5 ms</span>
        </div>
        <div class="bar-line">
          <div class="bar-fill gpu" style="width: 2.4%;"></div>
          <span class="bar-value gpu">3.3 ms</span>
        </div>
      </div>
      <div class="speedup-col">
        <span class="speedup-badge">9.8× faster</span>
      </div>
    </div>

    <!-- 1,000,000 items -->
    <div class="chart-row">
      <div class="row-label">1,000,000 items</div>
      <div class="bars-group">
        <div class="bar-line">
          <div class="bar-fill cpu" style="width: 45.4%;"></div>
          <span class="bar-value cpu">63.5 ms</span>
        </div>
        <div class="bar-line">
          <div class="bar-fill gpu" style="width: 4.9%;"></div>
          <span class="bar-value gpu">6.9 ms</span>
        </div>
      </div>
      <div class="speedup-col">
        <span class="speedup-badge">9.2× faster</span>
      </div>
    </div>

    <!-- 2,000,000 items -->
    <div class="chart-row">
      <div class="row-label">2,000,000 items</div>
      <div class="bars-group">
        <div class="bar-line">
          <div class="bar-fill cpu" style="width: 91.8%;"></div>
          <span class="bar-value cpu">128.6 ms</span>
        </div>
        <div class="bar-line">
          <div class="bar-fill gpu" style="width: 18.0%;"></div>
          <span class="bar-value gpu">25.2 ms</span>
        </div>
      </div>
      <div class="speedup-col">
        <span class="speedup-badge">5.1× faster</span>
      </div>
    </div>

    <!-- Scale Track -->
    <div class="scale-track">
      <div></div>
      <div class="scale-numbers">
        <span>0 ms</span>
        <span>30 ms</span>
        <span>60 ms</span>
        <span>90 ms</span>
        <span>120 ms</span>
        <span>140 ms</span>
      </div>
      <div></div>
    </div>
  </div>

  <!-- Footer -->
  <div class="footer">
    <span>Query: "AuthController" • Baseline: @leeoniya/ufuzzy • GPU: NVIDIA RTX 3050 Laptop</span>
    <span>Runs 100% locally in-browser</span>
  </div>
</body>
</html>`;

async function main() {
  const browser = await puppeteer.launch({
    executablePath: 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--font-render-hinting=none']
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1200, height: 680, deviceScaleFactor: 2 });
  await page.setContent(html, { waitUntil: 'networkidle0' });
  await page.evaluateHandle('document.fonts.ready');
  
  const outputPath = 'C:/Users/ASUS/code/webgpu-fuzzy-search/clean_minimal_chart.png';
  await page.screenshot({ path: outputPath });
  await browser.close();
  console.log('CLEAN_CHART_RENDERED_OK:', outputPath);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
