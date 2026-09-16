import { defineConfig } from 'tsup';

export default defineConfig({
  entry: ['src/index.ts'],
  format: ['esm', 'cjs'],
  dts: true,
  clean: true,
  sourcemap: true,
  minify: false,
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
