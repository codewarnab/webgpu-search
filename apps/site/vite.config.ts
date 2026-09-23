import { defineConfig } from 'vite';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const currentDir = path.dirname(fileURLToPath(import.meta.url));
const wgslPlugin = {
  name: 'wgsl-loader',
  enforce: 'pre' as const,
  transform(code: string, id: string) {
    if (!id.endsWith('.wgsl')) return undefined;
    return { code: `export default ${JSON.stringify(code)};`, map: { mappings: '' } };
  }
};

export default defineConfig({
  base: '/',
  plugins: [wgslPlugin],
  worker: { format: 'es', plugins: () => [wgslPlugin] },
  resolve: {
    alias: {
      'webgpu-search/worker': path.resolve(currentDir, '../../packages/webgpu-search/src/worker/search-worker.ts'),
      'webgpu-search': path.resolve(currentDir, '../../packages/webgpu-search/src/index.ts')
    }
  },
  server: {
    host: '0.0.0.0',
    port: 5178,
    proxy: {
      '/benchmark': 'http://localhost:5173',
      '/examples/docs-search': 'http://localhost:5176',
      '/examples/code-palette': 'http://localhost:5174',
      '/examples/log-viewer': 'http://localhost:5175'
    }
  },
  build: {
    rollupOptions: {
      input: {
        home: path.resolve(currentDir, 'index.html'),
        compare: path.resolve(currentDir, 'compare/index.html')
      }
    }
  }
});
