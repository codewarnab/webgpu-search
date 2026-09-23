import { cp, mkdir, rm } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const siteDist = path.join(root, 'apps/site/dist');
const mounts = [
  ['apps/benchmark/dist', 'benchmark'],
  ['apps/docs-search/dist', 'examples/docs-search'],
  ['apps/monaco-palette/dist', 'examples/code-palette'],
  ['apps/log-viewer/dist', 'examples/log-viewer']
] as const;

for (const [sourceRelative, destinationRelative] of mounts) {
  const source = path.join(root, sourceRelative);
  const destination = path.join(siteDist, destinationRelative);
  await rm(destination, { recursive: true, force: true });
  await mkdir(path.dirname(destination), { recursive: true });
  await cp(source, destination, { recursive: true });
}
