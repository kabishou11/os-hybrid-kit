# os-hybrid-kit

Portable Python library for **OpenSearch hybrid search**: BM25 lexical match plus kNN vectors, fused with **RRF** or **weighted score normalization**.

[![CI](https://github.com/kabishou11/os-hybrid-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/kabishou11/os-hybrid-kit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**v0.3.0** — facade frozen. You bring the embedding model; the kit builds the mapping, upserts the search pipeline, and runs hybrid search.

[CHANGELOG](CHANGELOG.md) · [Production checklist](docs/PRODUCTION.md) · [Relevance and latency notes](docs/NOTES.md) · [Roadmap](ROADMAP.md)

## Why

Keyword scores (BM25) and vector similarity live on different scales. Adding them naively ranks the wrong documents. OpenSearch 2.x solves this with a `hybrid` query and a **search pipeline** that fuses the two lists after the query phase.

## Install and 5-minute quickstart

Python 3.10+. Docker for the demo cluster. No PyPI release yet.

```bash
python -m pip install git+https://github.com/kabishou11/os-hybrid-kit.git
# or from a clone:
python -m pip install -e .
docker compose up -d
python examples/seed_and_search.py
```

The demo uses OpenSearch **2.19.6** (single node, security plugin off, `http://localhost:9200`) so both RRF and weighted fusion work. It hashes tokens into a tiny vector so you do not need an embedding model.

```bash
python examples/seed_and_search.py --fusion weighted
python examples/seed_and_search.py --query "waterproof boots for muddy paths"
python examples/compare_retrievers.py   # BM25 vs kNN vs hybrid
```

If OpenSearch fails to boot on Linux: `sudo sysctl -w vm.max_map_count=262144`. Wait until `curl http://localhost:9200` returns cluster info. Tear down with `docker compose down`.

## Library usage

**Primary facade:** `HybridKit`, `HybridConfig`, `FusionMethod`, `Hit`, `SearchResult`.

**Methods:** `ensure_index`, `exists_index`, `delete_index`, `upsert_pipeline`, `index_documents`, `lexical_search`, `knn_search`, `hybrid_search`.

```python
from os_hybrid_kit import FusionMethod, HybridConfig, HybridKit

config = HybridConfig(
    hosts=["http://localhost:9200"],
    index="docs",
    dimension=384,            # must match your embedding model
    text_field="content",
    vector_field="embedding",
    fusion=FusionMethod.RRF,  # or FusionMethod.WEIGHTED
    lexical_weight=0.3,       # weighted fusion; clause order is lexical then kNN
    vector_weight=0.7,
)
# Or: HybridConfig.from_env()  # OPENSEARCH_URL / OPENSEARCH_HOSTS,
#                              # OPENSEARCH_INDEX, OPENSEARCH_DIM

kit = HybridKit(config)
kit.delete_index()            # no-op if the index is missing
kit.ensure_index(extra_properties={"title": {"type": "text"}})
kit.upsert_pipeline()

kit.index_documents(
    [
        {
            "_id": "1",
            "title": "Trail shoes",
            "content": "Lightweight trail running shoes.",
            "embedding": query_vector,  # list[float], length == dimension
        }
    ]
)

result = kit.hybrid_search("running shoes", query_vector, size=10)
for hit in result.hits:       # Hit: id, score, source, index
    print(hit.score, hit.source["content"])
# result.texts("content") -> list[str]
```

`hybrid_search` sends a `hybrid` query with two clauses — `match` on the text field, then `knn` on the vector field — and sets `search_pipeline` so OpenSearch fuses the lists. That is the main path. `lexical_search(query)` and `knn_search(embedding)` hit the same index without a pipeline (pure BM25 `match`, raw `knn`). All three accept `filter_query` and `_source` includes/excludes.

Embeddings are **yours**; the kit does not call a model or the Neural Search `neural` query. Pass an `EmbedFn` (`(text: str) -> Sequence[float]`) wherever you need to embed a query (the Dify adapter does). A vector whose length does not match `config.dimension` raises `ValueError`.

`exists_index` / `ensure_index` / `delete_index` operate on `config.index`. `from_env()` reads `OPENSEARCH_HOSTS` (comma-separated, takes precedence) or `OPENSEARCH_URL`, plus `OPENSEARCH_INDEX` and `OPENSEARCH_DIM`. Keyword arguments override the environment.

**Builders** (advanced — use these if you already have an OpenSearch client):

```python
from os_hybrid_kit import (
    EmbedFn,
    build_hybrid_query,
    build_index_body,
    build_knn_query,
    build_knn_vector_property,
    build_lexical_query,
    build_opensearch_client,
    build_pipeline_body,
    build_rrf_pipeline_body,
    build_weighted_pipeline_body,
    parse_search_response,
    upsert_search_pipeline,
)
```

## Stable API (0.3)

These names are frozen for 0.3.x. Additive changes may appear; breaking renames will not.

| Kind | Names |
| --- | --- |
| Types | `HybridKit`, `HybridConfig`, `FusionMethod`, `Hit`, `SearchResult` |
| Methods | `ensure_index`, `exists_index`, `delete_index`, `upsert_pipeline`, `index_documents`, `lexical_search`, `knn_search`, `hybrid_search` |
| Fusion values | `FusionMethod.RRF` (`"rrf"`), `FusionMethod.WEIGHTED` (`"weighted"`) |
| Typing | `EmbedFn` (optional protocol for query embedders) |
| Advanced | `build_*` helpers, `parse_search_response`, `upsert_search_pipeline`, `build_opensearch_client` |

`Hit` fields: `id`, `score`, `source`, `index`. `SearchResult` fields: `hits`, `total`, `max_score`, plus `texts(field)`.

## Fusion

| Fusion | Processor | Combines | OpenSearch |
| --- | --- | --- | --- |
| Reciprocal rank fusion (`FusionMethod.RRF`) | `score-ranker-processor` | ranks, not raw scores | 2.19+ |
| Weighted normalization (`FusionMethod.WEIGHTED`) | `normalization-processor` | min-max / L2 / z-score, then a weighted mean | 2.11+ |

Default weighted weights are `lexical_weight=0.3`, `vector_weight=0.7` and must sum to 1.0. Clause order is lexical then kNN, so those weights line up with the hybrid query.

## Dify adapter

`adapters/dify/` is a stub, not a marketplace plugin.

- **External knowledge:** `retrieve(kit, request, embed_fn)` accepts Dify's `POST /retrieval` JSON and returns `{ "records": [ { content, score, title, metadata } ] }`. Dify only sends text, so you pass an `EmbedFn`.
- **VDB stub:** `HybridVectorStore` exposes `add_texts`, `hybrid_search`, `search_by_vector`, and `search_by_full_text` for a plugin you maintain.

RRF scores are small (they are sums of `1 / (rank_constant + rank)`). If Dify's score threshold is `0.5`, RRF hits will be filtered out. Use threshold `0` with RRF, or switch to weighted fusion (scores are typically in `[0, 1]`). Details: [docs/PRODUCTION.md](docs/PRODUCTION.md).

## Tests

Unit tests run on every push. An `integration` job starts OpenSearch 2.19.6 with `docker compose` and runs `pytest -m integration`. You do not need Docker for unit tests.

```bash
python -m pip install -e ".[dev]"
pytest -m "not integration"   # unit tests; no Docker / live cluster
pytest -m integration         # live cluster; CI sets RUN_INTEGRATION=1
```

Live tests skip unless `RUN_INTEGRATION=1`. Default URL is `OPENSEARCH_URL` or `http://localhost:9200`.

## What this is not

- A RAG UI, chat app, or document-ingestion product
- An MCP skills bridge
- An evaluation harness or RAGFlow eval kit
- A LiteLLM / model gateway
- ColQwen, late-interaction, or multimodal retrieval
- Guardrails or policy enforcement
- A packaged Dify marketplace plugin (the adapter is the retrieval contract plus a VDB stub)

## License

MIT. See [LICENSE](LICENSE).
