# webgpu-search

## 1.0.1

### Patch Changes

- adf41c9: Harden correctness and safety from a repo audit: shared GPU device pool no longer double-acquires or destroys a live device under concurrent creation; GPU buffers are clamped to `maxStorageBufferBindingSize` (oversized corpora now fall back to CPU instead of silently returning 0 results); batch mutations are atomic when a filter getter throws; high-cardinality string filters use bounded memory; worker INIT/RESTORE keep the previous index on failure and handle messages in order; snapshots reject duplicate/invalid doc IDs and accept typed-array buffers; custom IndexedDB store names are created on existing databases.
  
  Behaviour changes: boolean filter fields accept only `true`/`false` (or `"true"`/`"false"`); blank number filter bounds throw `InvalidFilterError`; numeric IDs sort before string IDs on score ties; field getters ignore `Object.prototype` properties.
