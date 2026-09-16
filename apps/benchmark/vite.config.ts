import { defineConfig } from 'vite';
import path from 'path';
import { fileURLToPath } from 'url';

const currentDir = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [
    {
      name: 'wgsl-loader',
      transform(code, id) {
        if (id.endsWith('.wgsl')) {
          return {
            code: `export default ${JSON.stringify(code)}; export const SUBSTRING_WGSL = ${JSON.stringify(code)}; export const FUZZY_WGSL = ${JSON.stringify(code)};`,
            map: { mappings: '' }
          };
        }
      }
    }
  ],
  resolve: {
    alias: {
      // In development, resolve webgpu-search directly to TypeScript source for instant HMR
      'webgpu-search': path.resolve(currentDir, '../../packages/webgpu-search/src/index.ts')
    }
  },
  server: {
    port: 5173,
    open: true
  }
});
