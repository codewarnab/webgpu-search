import { defineConfig } from 'tsup';

export default defineConfig({
  entry: {
    index: 'src/index.ts',
    worker: 'src/worker/search-worker.ts'
  },
  format: ['esm', 'cjs'],
  dts: true,
  clean: true,
  sourcemap: true,
  minify: false,
  splitting: false,
  treeshake: true,

  loader: {
    '.wgsl': 'text'
  },
  esbuildOptions(options) {
    options.banner = {
      js: '/* webgpu-search | MIT License */'
    };
  }
});
