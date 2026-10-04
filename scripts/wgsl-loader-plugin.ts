/**
 * Bun runtime plugin: `import x from './shaders/*.wgsl'` yields the shader
 * SOURCE TEXT (default export), matching tsup (`loader: { '.wgsl': 'text' }`)
 * and the benchmark Vite plugin. Without it Bun resolves unknown extensions
 * to the file path string, so headless tests would compile a path instead of
 * the real WGSL. Registered via `preload` in the root bunfig.toml.
 */
import { plugin } from 'bun';

plugin({
  name: 'wgsl-text-loader',
  setup(build) {
    build.onLoad({ filter: /\.wgsl$/ }, async ({ path }) => ({
      exports: { default: await Bun.file(path).text() },
      loader: 'object',
    }));
  },
});
