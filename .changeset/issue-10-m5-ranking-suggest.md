---
'webgpu-search': minor
---

Issue #10 M5: deterministic ranking and autocomplete primitives. `DocumentIndex.search()` default order changes from `(score DESC, docIndex ASC)` to the 5-tier `(score DESC, weight DESC, exact DESC, length ASC, id ASC)` on all paths; pass `ranking: { tieBreakers: ['score'] }` to approximate the legacy order. Adds `suggest()` plus inline `search({ suggest })` (index-wide, `suggest.field` beats `search.fields`, `false` disables inline), `SuggestOptions.tieBreakers` (inherits search ranking when omitted), fail-closed NaN guards preserving the total-order contract, and `QUERY_TOKENS_MAX` enforcement on suggest.
