# Historical snapshots

`snapshot-1.0.1.bin` was written by the **published npm package
`webgpu-search@1.0.1`**, not by the working tree. Its JSON manifest records
the npm tarball URL and integrity, input documents, index options, snapshot
format and SHA-256, and ordered search results from that released package.

The fixture protects compatibility with an older writer independently of
current serialization code. The snapshot suite loads the bytes directly and
checks result IDs, scores, matched fields, and restored documents. CI already
runs it through `bun run test:snapshot`.

## Preserve the fixture

Do not regenerate or update this pair to make a failing test pass. Investigate
the compatibility regression instead. Add a separately versioned fixture when
covering another release; retain the existing ones.

To reproduce the original generation in a temporary directory:

1. Install `webgpu-search@1.0.1` with
   `npm install --ignore-scripts --no-audit --no-fund --save-exact webgpu-search@1.0.1`.
2. Verify its `package-lock.json` integrity matches `producer.integrity` in
   the manifest. Import `DocumentIndex` from that installation, outside this
   monorepo so workspace aliases cannot select current code.
3. Pass the manifest's `documents` and `options` to `DocumentIndex.create()`.
   Write `new Uint8Array(index.serialize())` to a temporary file.
4. Run each manifest query using its `mode` and `cpuScorer: 'exact'`. Compare
   `totalMatches` and ordered `{ id, score, matchedField }` against `expected`.
   Destroy the index when finished.

The resulting file must match `snapshotSha256`; reproduction must not overwrite
the checked-in artifact. Queries cover weighted multi-field ranking, ASCII,
German and Greek case folding, decomposed accents, CJK, emoji/ZWJ, and no hits.
