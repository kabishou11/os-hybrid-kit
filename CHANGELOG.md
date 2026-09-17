# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.2] - 2026-09-17

Close remaining audit follow-ups on the frozen v0.5 facade. No breaking renames.

### Fixed

- Retriever `search_kwargs` raises `ValueError` when both `filter` and `filter_query` are set (`filter` is the LangChain-style alias)
- Dify VDB `search_by_vector` / `search_by_full_text` call `knn_search` / `lexical_search`, so query-vector dimension checks apply
- `docs/PRODUCTION.md` cites `adapters.dify.retrieval` / `adapters/dify/retrieval.py`

### Changed

- Package version `0.5.2`
- README Stable API heading: frozen since 0.3 / current 0.5.x
- Python 3.13 classifier

### Added

- Unit tests for the dual-filter raise, VDB dimension, `index_documents` bulk actions, and `filter_query` on kit search methods

## [0.5.1] - 2026-09-17

Validation, docs, and DX polish on the frozen v0.5 facade. No breaking renames.

### Fixed

- `put_alias` / `delete_alias` / `get_alias` / `swap_alias` / `with_index` reject empty or whitespace-only alias and index names with `ValueError` (they used to reach OpenSearch)
- `lexical_search` / `hybrid_search` reject empty or whitespace-only `query` before calling OpenSearch
- Empty embedding sequences raise `ValueError` (`knn_search` / `hybrid_search`); wrong-length vectors still raise as in 0.1.1

### Changed

- Package version `0.5.1`
- `Hit.raw` / `SearchResult.raw` default to `{}` so you can construct them without the OpenSearch payload; `parse_search_response` still fills `raw`
- `HybridConfig` / `HybridKit` `__repr__` omit credentials
- Example scripts share `embed_documents` / `wait_for_opensearch` and exit with a `docker compose up -d` hint if the cluster is down
- README, CONTRIBUTING, and ROADMAP aligned to the real facade (`0.5.x` is polish/harden)

### Added

- Thin `Makefile`: `make test`, `make lint`, `make demo-up`, `make demo-down`
- Optional `pytest-cov` on the `dev` extra

## [0.5.0] - 2026-09-17

Ops helpers and a light local timing script around the frozen v0.3 facade. No breaking renames.

### Added

- `HybridKit.put_alias` / `delete_alias` / `get_alias` / `swap_alias` — thin OpenSearch alias helpers (cutover, not a full index-management product)
- `HybridKit.with_index(name)` — clone a kit onto another index or alias, sharing the client
- `examples/benchmark_retrievers.py` — BM25 vs kNN vs hybrid hit ids, scores, and wall time (ms) on a fixed demo corpus
- README aliases / cutover and benchmark notes

### Changed

- Package version `0.5.0`
- `ROADMAP.md` marks alias helpers and the light benchmark as shipped; Neural Search and Dify marketplace stay unshipped

## [0.4.0] - 2026-09-17

Optional retriever extras around the frozen v0.3 facade. No breaking renames.

### Added

- Optional extra `langchain` (`langchain-core>=0.2`) and `llama-index` (`llama-index-core>=0.10`)
- `os_hybrid_kit.integrations.langchain.HybridKitRetriever` — LangChain `BaseRetriever` over `hybrid_search` / `lexical_search` / `knn_search`
- `os_hybrid_kit.integrations.llama_index.HybridKitRetriever` — LlamaIndex retriever returning `NodeWithScore`
- `InstallError` when an extra is missing at import
- [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md)

### Changed

- Package version `0.4.0`
- `ROADMAP.md` marks LangChain / LlamaIndex extras as shipped

## [0.3.0] - 2026-09-17

Resume-ready freeze of the v0.1 facade. No breaking renames.

### Added

- `CHANGELOG.md`
- `docs/PRODUCTION.md` — production checklist (dimension, OpenSearch version vs fusion processors, RRF vs score thresholds, `index.knn` / engine, pipeline names, HTTPS auth, bulk refresh)
- `docs/NOTES.md` — when hybrid beats BM25-only / kNN-only, how to read latency, filter / `_source` tips
- `EmbedFn` (`typing.Protocol`) for query embedders; used by the Dify adapter; exported from `os_hybrid_kit`
- README Stable API (0.3) note and links to changelog, production checklist, notes, and roadmap

### Changed

- Package version `0.3.0`
- `ROADMAP.md` marks v0.3 shipped

## [0.2.0] - 2026-09-17

Prove the kit against a live OpenSearch without expanding product scope.

### Added

- CI `integration` job: `docker compose` OpenSearch 2.19.6, then `pytest -m integration`
- `HybridKit.lexical_search` (BM25 `match`, no search pipeline)
- `HybridKit.knn_search` (raw `knn`, no search pipeline)
- `examples/compare_retrievers.py` — BM25 vs kNN vs hybrid top-k, plus a `term` filter and `_source` includes
- Git install note (`pip install git+https://github.com/kabishou11/os-hybrid-kit.git`)

### Changed

- Live tests cover lexical / kNN / hybrid on the same index
- README "Prove it" section and CI badge

## [0.1.1] - 2026-09-17

Public API and README aligned to one facade.

### Added

- `HybridConfig.from_env()` — `OPENSEARCH_HOSTS` / `OPENSEARCH_URL`, `OPENSEARCH_INDEX`, `OPENSEARCH_DIM`
- `HybridKit.exists_index` and `HybridKit.delete_index`
- `ROADMAP.md`

### Changed

- Clearer `ValueError` when a query vector length does not match `config.dimension`
- README rewritten around `HybridKit` / `HybridConfig` / `FusionMethod` / `Hit` / `SearchResult`

## [0.1.0] - 2026-09-17

Initial library.

### Added

- `HybridKit` facade: index mapping (`knn_vector` + BM25 text), search-pipeline upsert, bulk index, `hybrid_search`
- Fusion: RRF (`score-ranker-processor`, OpenSearch 2.19+) and weighted normalization (`normalization-processor`, 2.11+)
- Builders: `build_index_body`, `build_pipeline_body`, `build_hybrid_query`, and related helpers
- Thin Dify adapter (`adapters/dify/`): external-knowledge `/retrieval` plus a VDB stub
- Docker Compose demo (OpenSearch 2.19.6, security plugin off) and `examples/seed_and_search.py`
- Unit tests and GitHub Actions lint/unit CI
