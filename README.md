# os-hybrid-kit

Portable Python library for **OpenSearch hybrid search**: BM25 lexical match plus kNN vectors, fused with **RRF** or **weighted score normalization**.

[![CI](https://github.com/kabishou11/os-hybrid-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/kabishou11/os-hybrid-kit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**v0.5.1** — facade frozen; 0.5.x is polish and hardening. You bring the embedding model; the kit builds the mapping, upserts the search pipeline, and runs hybrid search.

[CHANGELOG](CHANGELOG.md) · [Production checklist](docs/PRODUCTION.md) · [Relevance and latency notes](docs/NOTES.md) · [Integrations](docs/INTEGRATIONS.md) · [Roadmap](ROADMAP.md)

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
python examples/compare_retrievers.py   # BM25 vs kNN vs hybrid ranking
python examples/benchmark_retrievers.py # same three modes, wall time (ms)
```

If OpenSearch fails to boot on Linux: `sudo sysctl -w vm.max_map_count=262144`. Wait until `curl http://localhost:9200` returns cluster info. Tear down with `docker compose down` (or `make demo-down`). Example scripts exit with a `docker compose up -d` hint if the cluster is down.

## Library usage

**Primary facade:** `HybridKit`, `HybridConfig`, `FusionMethod`, `Hit`, `SearchResult`.

**Methods:** `ensure_index`, `exists_index`, `delete_index`, `upsert_pipeline`, `index_documents`, `lexical_search`, `knn_search`, `hybrid_search`. Alias helpers: `put_alias`, `delete_alias`, `get_alias`, `swap_alias`, `with_index`.

```python
from os_hybrid_kit import FusionMethod, HybridConfig, HybridKit

config = HybridConfig(
    hosts=["http://localhost:9200"],
    index="docs",
    dimension=384,            # must match your embedding model
    text_field="content",
    vector_field="embedding",
    fusion=FusionMethod.RRF,  # or FusionMethod.WEIGHTED
    pipeline_name="hybrid-search-pipeline",
    lexical_weight=0.3,       # weighted fusion; clause order is lexical then kNN
    vector_weight=0.7,
    exclude_vector=True,      # drop the vector field from _source on search
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
for hit in result.hits:       # Hit: id, score, source, index, raw
    print(hit.score, hit.source["content"])
# result.texts("content") -> list[str]
```

`hybrid_search` sends a `hybrid` query with two clauses — `match` on the text field, then `knn` on the vector field — and sets `search_pipeline` so OpenSearch fuses the lists. That is the main path. `lexical_search(query)` and `knn_search(embedding)` hit the same index without a pipeline (pure BM25 `match`, raw `knn`). All three accept `filter_query` and `_source` includes/excludes.

Embeddings are **yours**; the kit does not call a model or the Neural Search `neural` query. Pass an `EmbedFn` (`(text: str) -> Sequence[float]`) wherever you need to embed a query (the Dify adapter does). An empty embedding, or a vector whose length does not match `config.dimension`, raises `ValueError`. Empty or whitespace-only `query`, alias, and index names on `lexical_search` / `hybrid_search` / alias helpers / `with_index` also raise `ValueError` before OpenSearch is called.

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

## Aliases and cutover

`config.index` is the OpenSearch target for mapping, bulk, and search. It may be a **concrete index** or an **alias**. Pointing reads at an alias lets you rebuild a new index and cut over without changing query code.

```python
kit = HybridKit(config)  # config.index == "docs-v1"
kit.ensure_index()
kit.put_alias("docs-read")  # alias -> docs-v1
kit.with_index("docs-read").hybrid_search(query, embedding)

blue = kit.with_index("docs-v2")
blue.ensure_index()
blue.index_documents([...])
kit.swap_alias("docs-read", "docs-v2", old_index="docs-v1")
```

`swap_alias` issues one `_aliases` request (remove old + add new) when `old_index` is passed. If `old_index` is omitted, the kit looks up current targets first, then updates. This is not a full index-management product: no `is_write_index` policy, no ILM.

## Benchmark

`examples/benchmark_retrievers.py` seeds a fixed demo corpus and prints BM25 vs kNN vs hybrid hit ids, scores, and wall time in milliseconds. It is a stopwatch on your machine, not an eval harness and not a quality claim. `examples/compare_retrievers.py` is the ranking side-by-side (no timings).

```bash
docker compose up -d
python examples/benchmark_retrievers.py
python examples/benchmark_retrievers.py --url http://localhost:9200 --n-queries 10 --fusion weighted
```

## Stable API (0.3)

These names are frozen for 0.3.x+. Additive extras (0.4 LangChain / LlamaIndex retrievers, 0.5 alias helpers) do not rename this facade. Breaking renames will not.

| Kind | Names |
| --- | --- |
| Types | `HybridKit`, `HybridConfig`, `FusionMethod`, `Hit`, `SearchResult` |
| Methods | `ensure_index`, `exists_index`, `delete_index`, `upsert_pipeline`, `index_documents`, `lexical_search`, `knn_search`, `hybrid_search` |
| Additive (0.5) | `put_alias`, `delete_alias`, `get_alias`, `swap_alias`, `with_index` |
| Fusion values | `FusionMethod.RRF` (`"rrf"`), `FusionMethod.WEIGHTED` (`"weighted"`) |
| Typing | `EmbedFn` (optional protocol for query embedders) |
| Advanced | `build_*` helpers, `parse_search_response`, `upsert_search_pipeline`, `build_opensearch_client` |

`Hit` fields: `id`, `score`, `source`, `index`, and `raw` (the original OpenSearch hit dict; optional if you construct a `Hit` yourself — `parse_search_response` always fills it). `SearchResult` fields: `hits`, `total`, `max_score`, `raw`, plus `texts(field)` which returns `str(hit.source.get(field, ""))` for each hit.

## Fusion

| Fusion | Processor | Combines | OpenSearch |
| --- | --- | --- | --- |
| Reciprocal rank fusion (`FusionMethod.RRF`) | `score-ranker-processor` | ranks, not raw scores | 2.19+ |
| Weighted normalization (`FusionMethod.WEIGHTED`) | `normalization-processor` | min-max / L2 / z-score, then a weighted mean | 2.11+ |

Default weighted weights are `lexical_weight=0.3`, `vector_weight=0.7` and must sum to 1.0. Clause order is lexical then kNN, so those weights line up with the hybrid query.

## Integrations (optional)

Thin retrievers around the frozen `HybridKit` facade. Core install does not pull these in.

```bash
pip install "os-hybrid-kit[langchain]"
pip install "os-hybrid-kit[llama-index]"
```

```python
from os_hybrid_kit import HybridConfig, HybridKit
from os_hybrid_kit.integrations.langchain import HybridKitRetriever

kit = HybridKit(HybridConfig(dimension=384, index="docs"))
retriever = HybridKitRetriever(
    kit,
    embed_fn,  # (text: str) -> Sequence[float]
    search_kwargs={"size": 10, "mode": "hybrid"},  # hybrid | lexical | knn
)
docs = retriever.invoke("waterproof trail shoes")
```

```python
from os_hybrid_kit.integrations.llama_index import HybridKitRetriever

retriever = HybridKitRetriever(kit, embed_fn, search_kwargs={"size": 10})
nodes = retriever.retrieve("waterproof trail shoes")
```

Importing a submodule without its extra raises `InstallError`. Details: [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md).

## Dify adapter

`adapters/dify/` is a stub, not a marketplace plugin.

- **External knowledge:** `retrieve(kit, request, embed_fn)` accepts Dify's `POST /retrieval` JSON and returns `{ "records": [ { content, score, title, metadata } ] }`. Dify only sends text, so you pass an `EmbedFn`.
- **VDB stub:** `HybridVectorStore` exposes `add_texts`, `hybrid_search`, `search_by_vector`, and `search_by_full_text` for a plugin you maintain.

RRF scores are small (they are sums of `1 / (rank_constant + rank)`). If Dify's score threshold is `0.5`, RRF hits will be filtered out. Use threshold `0` with RRF, or switch to weighted fusion (scores are typically in `[0, 1]`). Details: [docs/PRODUCTION.md](docs/PRODUCTION.md).

## Tests

Unit tests run on every push. An `integration` job starts OpenSearch 2.19.6 with `docker compose` and runs `pytest -m integration`. You do not need Docker for unit tests. Retriever tests skip unless the matching extra is installed.

```bash
python -m pip install -e ".[dev]"
pytest -m "not integration"   # unit tests; no Docker / live cluster
# or: make test && make lint
pytest -m integration         # live cluster; CI sets RUN_INTEGRATION=1
# optional retrievers:
python -m pip install -e ".[dev,langchain,llama-index]"
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
