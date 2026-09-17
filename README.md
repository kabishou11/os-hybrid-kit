# os-hybrid-kit

Portable English Python library for **OpenSearch hybrid search**: BM25 lexical match plus kNN vectors, fused with **RRF** or **weighted score normalization**. Includes a thin **Dify** external-knowledge / VDB stub.

## Problem

Keyword scores (BM25) and vector similarity live on different scales. Adding them naively ranks the wrong documents. OpenSearch 2.x solves this with a `hybrid` query and a **search pipeline** that runs between the query and fetch phases:

| Fusion | Processor | Combines | OpenSearch |
| --- | --- | --- | --- |
| Reciprocal rank fusion (RRF) | `score-ranker-processor` | ranks, not raw scores | 2.19+ |
| Weighted normalization | `normalization-processor` | min-max / L2 / z-score, then a weighted mean | 2.11+ |

This kit builds the index mapping, upserts the pipeline, and runs `hybrid_search(query, embedding, ...)` through `opensearch-py`. You bring the embedding model.

## Quickstart

Python 3.10+. Docker for the demo cluster.

```bash
python -m pip install -e .
docker compose up -d
python examples/seed_and_search.py
```

The demo uses OpenSearch **2.19.6** (single node, security plugin off, `http://localhost:9200`) so both RRF and weighted fusion work. It hashes tokens into a tiny vector so you do not need an embedding model.

```bash
# Weighted min-max instead of RRF
python examples/seed_and_search.py --fusion weighted

# Custom query
python examples/seed_and_search.py --query "waterproof boots for muddy paths"
```

If OpenSearch fails to boot on Linux, raise the mmap limit once:

```bash
sudo sysctl -w vm.max_map_count=262144
```

Wait until `curl http://localhost:9200` returns cluster info, then rerun the example. Tear down with `docker compose down`.

## Library usage

```python
from os_hybrid_kit import FusionMethod, HybridConfig, HybridKit

config = HybridConfig(
    hosts=["http://localhost:9200"],
    index="docs",
    dimension=384,          # must match your embedding model
    text_field="content",
    vector_field="embedding",
    fusion=FusionMethod.RRF,  # or FusionMethod.WEIGHTED
    lexical_weight=0.3,       # used by weighted fusion; clause order is lexical then kNN
    vector_weight=0.7,
)
kit = HybridKit(config)
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
for hit in result.hits:
    print(hit.score, hit.source["content"])
```

Builders are also public if you already have an OpenSearch client:

```python
from os_hybrid_kit import (
    build_hybrid_query,
    build_index_body,
    build_rrf_pipeline_body,
    build_weighted_pipeline_body,
    upsert_search_pipeline,
)
```

`hybrid_search` sends a `hybrid` query with two clauses — `match` on the text field, then `knn` on the vector field — and sets `search_pipeline` so OpenSearch fuses the lists. Embeddings are **yours**; the kit does not call a model or the Neural Search `neural` query.

## Dify adapter

`adapters/dify/` is a stub, not a marketplace plugin.

- **External knowledge:** `retrieve(kit, request, embed_fn)` accepts Dify's `POST /retrieval` JSON and returns `{ "records": [ { content, score, title, metadata } ] }`. Dify only sends text, so you pass an `embed_fn`.
- **VDB stub:** `HybridVectorStore` exposes `add_texts`, `hybrid_search`, `search_by_vector`, and `search_by_full_text` for a plugin you maintain.

RRF scores are small (they are sums of `1 / (rank_constant + rank)`). If Dify's score threshold is `0.5`, RRF hits will be filtered out. Use threshold `0` with RRF, or switch to weighted fusion (scores are typically in `[0, 1]`).

## Tests

```bash
python -m pip install -e ".[dev]"
pytest                 # unit tests; mapping/pipeline builders do not need OpenSearch
pytest -m integration  # live cluster at OPENSEARCH_URL (default http://localhost:9200)
```

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
