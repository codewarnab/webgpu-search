# Changesets

This folder holds changesets generated for releases of `webgpu-search`.

## Adding a Changeset

When you make a change to `packages/webgpu-search`, run:

```bash
bun changeset
```

Follow the prompts to select the change type (`patch`, `minor`, `major`) and enter a summary.

## Release Process

When a PR with changesets is merged to `main`, GitHub Actions will automatically open a **"Version Packages"** release PR. When that PR is merged, GitHub Actions will publish the updated version to npm.
