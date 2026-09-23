const prefixes = ['src/auth', 'src/search', 'src/components', 'packages/engine', 'apps/docs', 'examples/palette'];
const nouns = ['Auth', 'User', 'Profile', 'Order', 'Invoice', 'Search', 'Query', 'Worker', 'Document', 'Session', 'Token', 'Index'];
const suffixes = ['Controller', 'Service', 'Handler', 'Manager', 'Provider', 'Repository', 'View', 'Worker', 'Index'];

/** Deterministic generated corpus. Same input size always yields the same rows. */
export function generateCorpus(size: number): string[] {
  return Array.from({ length: size }, (_, i) =>
    `${prefixes[i % prefixes.length]}/${nouns[(i * 7 + 3) % nouns.length]}${nouns[(i * 13 + 5) % nouns.length]}${suffixes[(i * 17 + 1) % suffixes.length]}_${i}.ts`
  );
}
