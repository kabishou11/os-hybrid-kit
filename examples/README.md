# Examples

## Docker demo

From the repository root:

```bash
docker compose up -d
python examples/seed_and_search.py
python examples/compare_retrievers.py
```

`seed_and_search.py` waits for OpenSearch, deletes and recreates index `hybrid-demo` via `HybridKit.delete_index` / `ensure_index`, upserts an RRF (default) or weighted search pipeline, indexes six short documents with hashed embeddings, and prints a hybrid ranking. Pass `--host` or set `OPENSEARCH_URL` to point at another cluster.

`compare_retrievers.py` seeds a smaller corpus where BM25-top and kNN-top differ, then prints side-by-side top-k for `lexical_search` (BM25), `knn_search` (raw kNN), and `hybrid_search` (RRF or `--fusion weighted`). It also runs a `term` filter on `category` and limits `_source` to `title`, `content`, and `category`.

OpenSearch 2.19.6 is pinned so RRF (`score-ranker-processor`) is available. Weighted fusion (`--fusion weighted`) also runs on 2.11+.

## Filters and `_source`

Both search helpers and `hybrid_search` take `filter_query` (an OpenSearch query clause) and `source_includes` / `source_excludes`. Example:

```python
kit.hybrid_search(
    query,
    query_vector,
    filter_query={"term": {"category": "footwear"}},
    source_includes=["title", "content", "category"],
)
```

`HybridConfig.exclude_vector` defaults to `True`, so the vector field is dropped from `_source` unless you pass `source_excludes=[]`.

## Embeddings

`embed.py` is a hashing trick for the demo only. In production, pass vectors from the same model you used at index time.
