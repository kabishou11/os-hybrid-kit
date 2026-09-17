# Examples

## Docker demo

From the repository root:

```bash
docker compose up -d
python examples/seed_and_search.py
```

`seed_and_search.py` waits for OpenSearch, recreates index `hybrid-demo`, upserts an RRF (default) or weighted search pipeline, indexes six short documents with hashed embeddings, and prints a hybrid ranking.

OpenSearch 2.19.6 is pinned so RRF (`score-ranker-processor`) is available. Weighted fusion (`--fusion weighted`) also runs on 2.11+.

## Embeddings

`embed.py` is a hashing trick for the demo only. In production, pass vectors from the same model you used at index time.
