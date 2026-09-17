# Roadmap

## v0.1

Shipped on `0.1.x`: a small, typed library you can drop into a RAG stack.

- Public facade: `HybridKit`, `HybridConfig`, `FusionMethod`, `Hit`, `SearchResult`
- Index mapping (`knn_vector` + BM25 text) and search-pipeline upsert (RRF or weighted)
- `ensure_index` / `exists_index` / `delete_index` / `upsert_pipeline` / `index_documents` / `hybrid_search`
- `HybridConfig.from_env()` for `OPENSEARCH_URL` / `OPENSEARCH_HOSTS`, `OPENSEARCH_INDEX`, `OPENSEARCH_DIM`
- Builders for mapping, pipeline, and hybrid query JSON
- Thin Dify external-knowledge adapter + VDB stub
- Docker Compose demo (OpenSearch 2.19.6) and unit tests

## v0.2 prove

Shipped on `0.2.0`: the kit against a real OpenSearch, still the same product.

- Live OpenSearch integration tests in CI (`docker compose` + `pytest -m integration`)
- Documented filter / `_source` examples on the demo corpus
- A short BM25 vs kNN vs hybrid ranking example (`examples/compare_retrievers.py`)
- Thin `lexical_search` / `knn_search` on `HybridKit` (no pipeline; hybrid stays the main path)
- Packaging notes (`pip install` from git; no PyPI release yet)

## v0.3 resume highlight

Shipped on `0.3.0`: freeze the facade and write-up so the repo is CV-citable. Same product; no breaking renames.

- Facade frozen: `HybridKit`, `HybridConfig`, `FusionMethod`, `Hit`, `SearchResult`
- Methods frozen: `ensure_index`, `exists_index`, `delete_index`, `upsert_pipeline`, `index_documents`, `lexical_search`, `knn_search`, `hybrid_search`
- [CHANGELOG.md](CHANGELOG.md)
- [docs/PRODUCTION.md](docs/PRODUCTION.md) — dimension, pipeline version, RRF vs score threshold, `index.knn`, auth, refresh
- [docs/NOTES.md](docs/NOTES.md) — when hybrid beats BM25/kNN, how to measure latency, filter / `_source`
- `EmbedFn` protocol for query embedders (Dify adapter uses it)
- README Stable API (0.3) note

## v0.4 extras

Shipped on `0.4.0`: optional retriever wrappers. Same frozen facade; extras are not required.

- LangChain `HybridKitRetriever` (`pip install os-hybrid-kit[langchain]`)
- LlamaIndex `HybridKitRetriever` (`pip install os-hybrid-kit[llama-index]`)
- [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md)

## v0.5 ops + prove

Shipped on `0.5.0`: alias helpers and a light retriever timing script. Same frozen facade.

- `HybridKit.put_alias` / `delete_alias` / `get_alias` / `swap_alias`
- `HybridKit.with_index` so `config.index` can be an alias for zero-downtime reads
- `examples/benchmark_retrievers.py` — BM25 vs kNN vs hybrid wall times on a fixed corpus (not an eval harness)

Still optional / unshipped:

- Dify marketplace plugin packaging
- Neural Search `neural` query (model hosted in OpenSearch)

## Non-goals

These stay out of this repository unless the scope is rewritten:

- A RAG UI, chat app, or document-ingestion product
- An MCP skills bridge
- An evaluation harness or RAGFlow eval kit
- A LiteLLM / model gateway
- ColQwen, late-interaction, or multimodal retrieval as core
- Guardrails or policy enforcement
- A packaged Dify marketplace plugin as the primary artifact
