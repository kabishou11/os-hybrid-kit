# Relevance and latency notes

Short observations for operators. This is not a benchmark report and not an evaluation harness.

## When hybrid beats BM25-only or kNN-only

Run the side-by-side example against a local cluster:

```bash
docker compose up -d
python examples/compare_retrievers.py
python examples/compare_retrievers.py --fusion weighted
```

The script seeds a tiny corpus where the three retrievers **disagree** on purpose (see `examples/compare_retrievers.py` and `examples/README.md`):

- **BM25 (`lexical_search`)** prefers exact query terms. It wins on SKUs, error codes, rare tokens, and short “coverage” copy that repeats the query.
- **kNN (`knn_search`)** prefers vector proximity. It wins on paraphrase and synonyms, and can be fooled by keyword-stuffed text whose hash/embedding is dominated by one token.
- **Hybrid (`hybrid_search`)** fuses the two ranked lists (RRF or weighted min-max). A document that is decent on both can outrank a keyword-only hit and a vector-only hit.

Use hybrid when queries mix precise tokens with natural language. Stay on BM25 when the query is an identifier. Stay on kNN when you only have a vector (no text query).

The demo embedder is a bag-of-words hash (`examples/embed.py`), not a production model. Treat the ranking *pattern* as the lesson, not the particular scores.

## Latency (order of magnitude)

This environment has no Docker, so there are **no measured numbers here**. Do not copy invented p50/p99 values.

Qualitative, for a hybrid request:

1. You embed the query (outside the kit; often the slowest step with a real model).
2. OpenSearch runs two subqueries: BM25 `match` and `knn`, then a search-pipeline processor (RRF or normalization).
3. The Python client parses hits.

Hybrid is typically slower than BM25-only because of the kNN clause and the pipeline. Versus kNN-only, the extra BM25 clause is usually cheap; fusion itself is a small post-query step. HNSW walk cost (`k`, `ef_search`, dimension, engine) dominates on larger indexes.

The Compose demo is a single node with 512 MB heap. Production latency depends on shard count, segment count, `knn_k`, filters, and `_source` size.

**How to measure** (query time inside OpenSearch, not embedding):

```python
result = kit.hybrid_search(query, embedding, size=10)
print(result.raw.get("took"))  # milliseconds; OpenSearch `took`
```

Time `embed_fn(query)` separately if you care about end-to-end. `examples/compare_retrievers.py` is for ranking, not load testing. For a fair comparison, warm the index, drop the vector from `_source` (default), and record `took` for `lexical_search`, `knn_search`, and `hybrid_search` on the same query set.

## Filters and `_source`

All three search methods accept `filter_query` (an OpenSearch query clause) and `source_includes` / `source_excludes`.

```python
kit.hybrid_search(
    query,
    query_vector,
    filter_query={"term": {"category": "footwear"}},
    source_includes=["title", "content", "category"],
)
```

- Map filter fields as `keyword` (`ensure_index(extra_properties={"category": {"type": "keyword"}})`). A `term` query on analyzed `text` will miss.
- Hybrid applies `filter` on the `hybrid` clause; kNN applies it inside the `knn` field; lexical wraps `match` in `bool`/`filter`. See `build_hybrid_query` / `build_knn_query` / `build_lexical_query`.
- Filters cut the candidate set (good for latency and correctness) but cannot recover a document the subqueries never saw.
- `HybridConfig.exclude_vector` defaults to `True`: the vector field is omitted from `_source` unless you pass `source_excludes=[]`. Returning dense vectors inflates payload size.
- `source_includes` is a whitelist; combine it with `exclude_vector` only if you still need to drop other large fields.

## Related

- [Production checklist](PRODUCTION.md)
- [examples/README.md](../examples/README.md)
