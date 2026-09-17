# Production checklist

Use this before pointing `HybridKit` at a real cluster. It is not an OpenSearch operations runbook.

The public facade is `HybridKit`, `HybridConfig`, `FusionMethod`, `Hit`, and `SearchResult`. See [Stable API (0.3)](../README.md#stable-api-03).

## Embedding dimension

- [ ] `HybridConfig.dimension` equals the embedding model size (for example 384, 768, 1024).
- [ ] Every indexed document's vector field has that same length.
- [ ] Query vectors passed to `hybrid_search` / `knn_search` come from the **same model** as index time.

`HybridKit` raises `ValueError` when a **query** vector length does not match `config.dimension`. Bulk `index_documents` does not pre-check vectors; OpenSearch rejects a `knn_vector` whose length does not match the mapping.

Changing dimension requires a new index. `ensure_index` will not alter an existing mapping.

## OpenSearch version vs fusion

| `FusionMethod` | Processor | Minimum OpenSearch |
| --- | --- | --- |
| `RRF` | `score-ranker-processor` | **2.19+** |
| `WEIGHTED` | `normalization-processor` | **2.11+** |

- [ ] Cluster version supports the processor you configured.
- [ ] You called `upsert_pipeline()` after choosing `config.fusion`.

The demo image is OpenSearch **2.19.6** so both paths work. On 2.11–2.18 use `FusionMethod.WEIGHTED` only. A missing processor typically fails at pipeline upsert or search time, not at import.

Default weighted weights are `lexical_weight=0.3`, `vector_weight=0.7` and must sum to 1.0. Hybrid query clause order is lexical then kNN, so those weights line up.

## RRF scores vs score thresholds (Dify and similar)

RRF combines **ranks**, not raw BM25/kNN scores:

```text
score(d) = sum_q  weight_q / (rank_constant + rank_q(d))
```

Default `rank_constant` is 60. A document ranked first on both lists scores about `2 / 61 ≈ 0.033`. That is far below a threshold of `0.5`.

- [ ] If you use RRF, set any downstream score threshold to `0` (or a value on the order of `0.01`), not `0.5`.
- [ ] If a product UI defaults to `0.5` (Dify's external-knowledge setting often does), either lower it or switch to `FusionMethod.WEIGHTED` (min-max scores are typically in `[0, 1]`).

The Dify adapter (`adapters/dify/retrieve`) applies `retrieval_setting.score_threshold` to `Hit.score` as returned by OpenSearch.

## `index.knn` and the mapping engine

`ensure_index` / `build_index_body` set `index.knn: true` (required by the k-NN plugin) and map:

- `text_field` → `text` (BM25)
- `vector_field` → `knn_vector` with `dimension`, `space_type`, and `method` (`hnsw` + `engine`)

Defaults: `engine="lucene"`, `method_name="hnsw"`, `space_type="l2"`. Lucene HNSW is the usual choice on OpenSearch 2.x. `nmslib` / `faiss` need a cluster that actually has those engines.

- [ ] Engine and `space_type` match how you built the vectors (L2 vs cosine).
- [ ] Filter fields used with `term` queries are mapped as `keyword` via `ensure_index(extra_properties=...)`, not `text`.
- [ ] You are not trying to change `knn_vector` dimension or engine in place — recreate the index.

Optional HNSW knobs go in `HybridConfig.hnsw_parameters` (for example `m`, `ef_construction`). They affect recall and latency, not the Python API.

## Pipeline name consistency

- [ ] `upsert_pipeline()` has been called on this cluster for `config.pipeline_name` (default `hybrid-search-pipeline`).
- [ ] `hybrid_search` uses that same name (`config.pipeline_name`, or the `pipeline_name=` override).
- [ ] After changing `fusion`, `rank_constant`, or weights, you upsert again. PUT replaces the pipeline body.

`lexical_search` and `knn_search` do **not** send `search_pipeline`. If hybrid results look like an unfused concatenation, the usual cause is a missing or mistyped pipeline name.

Use distinct pipeline names if RRF and weighted configs share a cluster (the examples do: `hybrid-rrf-pipeline` vs `hybrid-weighted-pipeline`).

## Security plugin, HTTPS, and auth

The Compose demo sets `DISABLE_SECURITY_PLUGIN=true` and uses `http://localhost:9200`. That is not a production posture.

`build_opensearch_client` reads from `HybridConfig`:

| Field | Role |
| --- | --- |
| `hosts` | URLs, including `https://...` |
| `use_ssl` | TLS for the Python client |
| `verify_certs` | Certificate verification (default `False` for local demos) |
| `ssl_show_warn` | Warning when verification is off |
| `username` / `password` | HTTP basic auth, sent only when **both** are set |
| `request_timeout` | Client timeout in seconds (default 30) |

- [ ] Production: `use_ssl=True`, `verify_certs=True`, credentials set, HTTPS hosts.
- [ ] `from_env()` loads `OPENSEARCH_HOSTS` / `OPENSEARCH_URL`, `OPENSEARCH_INDEX`, and `OPENSEARCH_DIM` only. Pass `username`, `password`, `use_ssl`, and `verify_certs` as keyword overrides.

Example:

```python
config = HybridConfig.from_env(
    username="admin",
    password=secret,
    use_ssl=True,
    verify_certs=True,
)
```

## Refresh and bulk indexing

`index_documents(..., refresh=True)` (the default) waits until the bulk is searchable. That is correct for tests and the demo; it is a bottleneck on large ingest.

- [ ] Production bulk: `refresh=False`, then refresh once per batch (`kit.client.indices.refresh(index=config.index)`) or rely on `refresh_interval` (OpenSearch default 1s).
- [ ] Document `_id` is popped from each payload and used as the document id; it is not stored in `_source` unless you also set another field.
- [ ] Vectors in `_source` are large. `HybridConfig.exclude_vector` defaults to `True`, so search responses drop the vector field unless you pass `source_excludes=[]`.

`index_documents` returns the `opensearchpy.helpers.bulk` tuple `(success_count, errors)`. Check `errors` on large loads.

## Related

- [Relevance and latency notes](NOTES.md)
- [CHANGELOG](../CHANGELOG.md)
