import { defineConfig } from 'vite';
import path from 'path';
import { fileURLToPath } from 'url';

const currentDir = path.dirname(fileURLToPath(import.meta.url));

const wgslPlugin = {
  name: 'wgsl-loader',
  enforce: 'pre' as const,
  transform(code: string, id: string) {
    if (id.endsWith('.wgsl')) {
      return {
        code: `export default ${JSON.stringify(code)}; export const SUBSTRING_WGSL = ${JSON.stringify(code)}; export const FUZZY_WGSL = ${JSON.stringify(code)};`,
        map: { mappings: '' }
      };
    }
  }
};

export default defineConfig({
  plugins: [wgslPlugin],
  worker: {
    format: 'es',
    plugins: () => [wgslPlugin]
  },
  resolve: {
    alias: {
      'webgpu-search/worker': path.resolve(currentDir, '../../packages/webgpu-search/src/worker/search-worker.ts'),
      'webgpu-search': path.resolve(currentDir, '../../packages/webgpu-search/src/index.ts')
    }
  },
  server: {
    port: 5175,
    open: false
  }
});
