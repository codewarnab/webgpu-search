import fs from 'fs';
import path from 'path';
import os from 'os';
import { spawn, type ChildProcess } from 'child_process';

/**
 * Automatically locates a Chrome or Chromium executable across Linux, macOS, and Windows.
 * Checks environment variables (CHROME_BIN, PUPPETEER_EXECUTABLE_PATH),
 * Playwright and Puppeteer cache directories (~/.cache/ms-playwright, ~/.cache/puppeteer),
 * and standard operating system installation paths.
 */
export function getChromeExecutablePath(): string {
    const isExecutable = (p: string): boolean => {
        try {
            if (!fs.existsSync(p) || !fs.statSync(p).isFile()) return false;
            if (process.platform !== 'win32') fs.accessSync(p, fs.constants.X_OK);
            return true;
        } catch {
            return false;
        }
    };

    if (process.env.CHROME_BIN && isExecutable(process.env.CHROME_BIN)) {
        return process.env.CHROME_BIN;
    }
    if (process.env.PUPPETEER_EXECUTABLE_PATH && isExecutable(process.env.PUPPETEER_EXECUTABLE_PATH)) {
        return process.env.PUPPETEER_EXECUTABLE_PATH;
    }

    const homeDir = os.homedir();
    const platform = process.platform;

    // Search Playwright / Puppeteer cache directories across possible user homes
    const candidateHomes = new Set<string>([
        homeDir,
        process.env.HOME || '',
        '/home/ubuntu',
        '/home/runner',
        '/root'
    ].filter(Boolean));

    const cacheDirs: string[] = [];
    for (const h of candidateHomes) {
        if (platform === 'win32') {
            const localAppData = process.env.LOCALAPPDATA || path.join(h, 'AppData', 'Local');
            cacheDirs.push(path.join(localAppData, 'ms-playwright'));
            cacheDirs.push(path.join(localAppData, 'puppeteer'));
            cacheDirs.push(path.join(h, '.cache', 'puppeteer'));
        } else if (platform === 'darwin') {
            cacheDirs.push(path.join(h, 'Library', 'Caches', 'ms-playwright'));
            cacheDirs.push(path.join(h, 'Library', 'Caches', 'puppeteer'));
        } else {
            cacheDirs.push(path.join(h, '.cache', 'ms-playwright'));
            cacheDirs.push(path.join(h, '.cache', 'puppeteer'));
        }
    }

    for (const cacheDir of cacheDirs) {
        if (!fs.existsSync(cacheDir)) continue;
        const targetNames = platform === 'win32'
            ? ['chrome.exe', 'msedge.exe']
            : platform === 'darwin'
                ? ['Google Chrome for Testing', 'Chromium', 'Google Chrome', 'chrome', 'chromium']
                : ['chrome', 'chromium'];
        const candidates = findExecutablesRecursively(cacheDir, targetNames, 8);
        if (candidates.length > 0) {
            // Sort by modification time descending (prefer latest version)
            candidates.sort((a, b) => {
                try {
                    return fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs;
                } catch {
                    return 0;
                }
            });
            return candidates[0]!;
        }
    }

    // Platform-specific system default candidates
    if (platform === 'win32') {
        const progFiles = process.env.PROGRAMFILES || 'C:\\Program Files';
        const progFilesX86 = process.env['PROGRAMFILES(X86)'] || 'C:\\Program Files (x86)';
        const localAppData = process.env.LOCALAPPDATA || '';
        const candidates = [
            path.join(progFiles, 'Google\\Chrome\\Application\\chrome.exe'),
            path.join(progFilesX86, 'Google\\Chrome\\Application\\chrome.exe'),
            path.join(localAppData, 'Google\\Chrome\\Application\\chrome.exe'),
            path.join(progFiles, 'Microsoft\\Edge\\Application\\msedge.exe')
        ];
        for (const p of candidates) {
            if (fs.existsSync(p)) return p;
        }
    } else if (platform === 'darwin') {
        const candidates = [
            '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
            '/Applications/Chromium.app/Contents/MacOS/Chromium'
        ];
        for (const p of candidates) {
            if (fs.existsSync(p)) return p;
        }
    } else {
        const candidates = [
            '/usr/bin/google-chrome-stable',
            '/usr/bin/google-chrome',
            '/usr/bin/chromium-browser',
            '/usr/bin/chromium',
            '/snap/bin/chromium'
        ];
        for (const p of candidates) {
            if (fs.existsSync(p)) return p;
        }
    }

    throw new Error(
        'Could not automatically find Chrome or Chromium executable. ' +
        'Please set CHROME_BIN or PUPPETEER_EXECUTABLE_PATH environment variable.'
    );
}

function findExecutablesRecursively(dir: string, targetNames: string[], maxDepth = 5): string[] {
    const results: string[] = [];
    function walk(currentDir: string, depth: number) {
        if (depth > maxDepth) return;
        let entries: fs.Dirent[];
        try {
            entries = fs.readdirSync(currentDir, { withFileTypes: true });
        } catch {
            return;
        }
        for (const entry of entries) {
            const fullPath = path.join(currentDir, entry.name);
            if (entry.isDirectory()) {
                walk(fullPath, depth + 1);
            } else if (entry.isFile() || entry.isSymbolicLink()) {
                const match = targetNames.some(t =>
                    process.platform === 'win32' ? t.toLowerCase() === entry.name.toLowerCase() : t === entry.name
                );
                if (match) {
                    if (process.platform !== 'win32') {
                        try {
                            fs.accessSync(fullPath, fs.constants.X_OK);
                            results.push(fullPath);
                        } catch {
                            // not executable
                        }
                    } else {
                        results.push(fullPath);
                    }
                }
            }
        }
    }
    walk(dir, 0);
    return results;
}

/**
 * Standard Chrome launch flags required for WebGPU compute operations in headless environments.
 */
export function getChromeLaunchArgs(): string[] {
    const args = [
        '--enable-unsafe-webgpu',
        '--enable-features=Vulkan,WebGPU',
        '--enable-gpu-rasterization',
        '--enable-unsafe-swiftshader',
        '--disable-dev-shm-usage',
        '--ignore-gpu-blocklist',
        '--disable-gpu-sandbox',
        '--window-size=1280,950',
        '--no-sandbox',
        '--disable-setuid-sandbox'
    ];
    if (process.platform === 'win32') {
        args.push('--use-angle=d3d12');
    }
    return args;
}

export interface ServerHandle {
    url: string;
    close: () => Promise<void>;
}

/**
 * Ensures the Vite benchmark development server is running on the specified port.
 * If already active, reuses it. Otherwise, spawns a new server instance and waits for readiness.
 */
export async function ensureBenchmarkServer(port = 5173, timeoutMs = 30000): Promise<ServerHandle> {
    const url = `http://127.0.0.1:${port}`;

    // Test if already running
    try {
        const res = await fetch(`${url}/`, { signal: AbortSignal.timeout(1000) });
        const html = await res.text().catch(() => '');
        if ((res.ok || res.status === 200 || res.status === 304) && html.includes('WebGPU Fuzzy Search')) {
            console.log(`[Server] Detected existing benchmark server running at ${url}`);
            return {
                url,
                close: async () => {}
            };
        }
    } catch {
        // Not running, proceed with spawning
    }

    console.log(`[Server] Spawning Vite benchmark server on ${url}...`);
    const repoRoot = path.resolve(__dirname, '..');
    const benchmarkDir = path.resolve(repoRoot, 'apps/benchmark');
    const child = spawn(
        'bun',
        ['run', 'dev', '--', '--host', '127.0.0.1', '--port', String(port), '--strictPort', '--no-open'],
        {
            cwd: benchmarkDir,
            detached: process.platform !== 'win32',
            stdio: ['ignore', 'pipe', 'pipe'],
            env: { ...process.env, BROWSER: 'none', CI: 'true' }
        }
    );

    let stdoutLog = '';
    let stderrLog = '';
    child.stdout?.on('data', (d) => { stdoutLog += d.toString(); });
    child.stderr?.on('data', (d) => { stderrLog += d.toString(); });

    let killed = false;
    const cleanup = async () => {
        if (killed) return;
        killed = true;
        try {
            if (process.platform !== 'win32' && child.pid) {
                try {
                    process.kill(-child.pid, 'SIGTERM');
                } catch {
                    child.kill('SIGTERM');
                }
            } else {
                child.kill('SIGTERM');
            }
            const killTimer = setTimeout(() => {
                try {
                    if (process.platform !== 'win32' && child.pid) {
                        process.kill(-child.pid, 'SIGKILL');
                    } else if (!child.killed) {
                        child.kill('SIGKILL');
                    }
                } catch {}
            }, 2000);
            if (typeof killTimer.unref === 'function') {
                killTimer.unref();
            }
        } catch {}
    };

    process.on('exit', () => { cleanup(); });
    process.on('SIGINT', () => { cleanup(); process.exit(130); });
    process.on('SIGTERM', () => { cleanup(); process.exit(143); });

    // Poll until server responds
    const startTime = Date.now();
    while (Date.now() - startTime < timeoutMs) {
        if (child.exitCode !== null) {
            throw new Error(
                `Benchmark server exited prematurely with code ${child.exitCode}.\nStderr: ${stderrLog}\nStdout: ${stdoutLog}`
            );
        }
        try {
            const res = await fetch(`${url}/`, { signal: AbortSignal.timeout(500) });
            if (res.ok || res.status === 200) {
                console.log(`[Server] Benchmark server ready at ${url}`);
                return {
                    url,
                    close: cleanup
                };
            }
        } catch {
            await new Promise(r => setTimeout(r, 200));
        }
    }

    await cleanup();
    throw new Error(
        `Timed out after ${timeoutMs}ms waiting for benchmark server at ${url}.\nStdout: ${stdoutLog}\nStderr: ${stderrLog}`
    );
}

/**
 * Formats benchmark results into a structured GitHub-compatible Markdown summary.
 */
export function formatBenchmarkMarkdown(benchmarkData: {
    substring?: any[];
    fuzzy?: any[];
    meta?: any;
}): string {
    const lines: string[] = [];
    lines.push('# WebGPU Fuzzy Search Benchmark Results');
    lines.push('');
    lines.push(`- **Date**: ${new Date().toISOString()}`);
    const qualStatus = benchmarkData.meta?.qualificationStatus || (benchmarkData as any).qualificationStatus;
    if (qualStatus) {
        const isQualified = qualStatus === 'qualified';
        lines.push(`- **Hardware Qualification**: ${isQualified ? '✅ Qualified (Physical GPU)' : '⚠️ pending-hardware (Software Vulkan / Mock Render)'}`);
    }
    const adapter = benchmarkData.meta?.adapterInfo || (benchmarkData as any).adapterInfo;
    if (adapter) {
        lines.push(`- **Adapter**: ${JSON.stringify(adapter)}`);
    }
    lines.push('');

    const renderTable = (title: string, rows: any[]) => {
        lines.push(`## ${title}`);
        lines.push('');
        lines.push('| Dataset Size | Corpus | Packed VRAM | GPU Retained (med / p95) | CPU Parity (med / p95) | uFuzzy (med) | JS Native (med) | Speedup vs uFuzzy | Speedup vs Parity | Qualification |');
        lines.push('|---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|');

        for (const r of rows) {
            const size = (r.datasetSize || 0).toLocaleString();
            const corpus = r.corpusType || 'ascii';
            const vram = r.vramAllocation?.totalBytes ? `${(r.vramAllocation.totalBytes / (1024 * 1024)).toFixed(2)} MB` : 'N/A';
            const gpuRet = r.gpuRetained?.medianMs !== undefined ? `${r.gpuRetained.medianMs.toFixed(2)} / ${r.gpuRetained.p95Ms.toFixed(2)} ms` : 'N/A';
            const cpuPar = r.cpuParity?.medianMs !== undefined ? `${r.cpuParity.medianMs.toFixed(2)} / ${r.cpuParity.p95Ms.toFixed(2)} ms` : `${(r.cpuParityMs || 0).toFixed(2)} ms`;
            const uf = r.ufuzzy?.medianMs !== undefined ? `${r.ufuzzy.medianMs.toFixed(2)} ms` : `${(r.ufuzzyMs || 0).toFixed(2)} ms`;
            const js = r.jsNative?.medianMs !== undefined ? `${r.jsNative.medianMs.toFixed(2)} ms` : `${(r.jsNativeMs || 0).toFixed(2)} ms`;
            const spUf = `${r.retainedVsUfuzzySpeedup || 0}x`;
            const spPar = `${r.retainedVsParitySpeedup || 0}x`;
            const qual = r.qualificationStatus === 'qualified' ? 'qualified' : 'pending-hardware';

            lines.push(`| ${size} | ${corpus} | ${vram} | ${gpuRet} | ${cpuPar} | ${uf} | ${js} | ${spUf} | ${spPar} | ${qual} |`);
        }
        lines.push('');
    };

    if (benchmarkData.substring && benchmarkData.substring.length > 0) {
        renderTable('Exact Substring Benchmark', benchmarkData.substring);
    }
    if (benchmarkData.fuzzy && benchmarkData.fuzzy.length > 0) {
        renderTable('Fuzzy Subsequence Benchmark', benchmarkData.fuzzy);
    }

    return lines.join('\n');
}
