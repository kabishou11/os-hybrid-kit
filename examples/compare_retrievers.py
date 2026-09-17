#!/usr/bin/env python3
"""Compare BM25, kNN, and hybrid rankings on a small corpus.

The hashing embedder is bag-of-words, so BM25 and kNN usually agree. This
corpus is built so they do not:

* ``coverage`` — short field that contains every query term (BM25-top)
* ``stuffed`` — repeats ``boots`` (hash vector dominated by that token, kNN-top)
* ``product`` — natural copy; hybrid (RRF or weighted) blends the two lists

Filters and ``_source`` includes (also used below)::

    kit.lexical_search(
        query,
        filter_query={"term": {"category": "footwear"}},
        source_includes=["title", "content", "category"],
    )

Usage (from the repo root, with the cluster up):

    python -m pip install -e .
    docker compose up -d
    python examples/compare_retrievers.py
    python examples/compare_retrievers.py --fusion weighted
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from embed import embed_documents, embed_text, wait_for_opensearch  # noqa: E402
from os_hybrid_kit import FusionMethod, HybridConfig, HybridKit, SearchResult  # noqa: E402

DEMO_INDEX = "hybrid-compare"
DIMENSION = 32
DEFAULT_QUERY = "hiking boots"

DOCUMENTS = [
    {
        "_id": "coverage",
        "title": "Query-shaped trail copy",
        "content": "Waterproof hiking boots for muddy trails.",
        "category": "footwear",
    },
    {
        "_id": "stuffed",
        "title": "Warehouse boots (keyword-stuffed)",
        "content": "boots boots boots boots boots boots boots for warehouse staff",
        "category": "footwear",
    },
    {
        "_id": "product",
        "title": "Waterproof hiking boots",
        "content": "Waterproof hiking boots with a grippy outsole for rocky wet trails.",
        "category": "footwear",
    },
    {
        "_id": "synonym",
        "title": "Gore-Tex trail footwear",
        "content": "Gore-Tex trail footwear with aggressive lugs for muddy paths.",
        "category": "footwear",
    },
    {
        "_id": "chair",
        "title": "Office chair",
        "content": "Ergonomic office chair with lumbar support for long workdays at a desk.",
        "category": "furniture",
    },
    {
        "_id": "espresso",
        "title": "Espresso machine",
        "content": "Compact espresso machine with a steam wand for home coffee drinks.",
        "category": "kitchen",
    },
]


def _cell(hit_id: str, title: str, score: float, width: int) -> str:
    label = f"{hit_id}: {title}"
    if len(label) > 22:
        label = label[:21] + "…"
    return f"{score:7.4f} {label}".ljust(width)


def print_side_by_side(columns: list[tuple[str, SearchResult]], *, size: int) -> None:
    width = 32
    header = f"{'rank':<4} " + "  ".join(name.ljust(width) for name, _ in columns)
    print(header)
    print("-" * len(header))
    for rank in range(size):
        cells = []
        for _, result in columns:
            if rank < len(result.hits):
                hit = result.hits[rank]
                title = str(hit.source.get("title", hit.id))
                cells.append(_cell(hit.id, title, hit.score, width))
            else:
                cells.append("".ljust(width))
        print(f"{rank + 1:<4} " + "  ".join(cells))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare BM25, kNN, and hybrid top-k on a small corpus."
    )
    parser.add_argument(
        "--host",
        "--url",
        default=os.environ.get("OPENSEARCH_URL", "http://localhost:9200"),
        help="OpenSearch URL (or OPENSEARCH_URL).",
    )
    parser.add_argument(
        "--fusion",
        choices=["rrf", "weighted"],
        default="rrf",
        help="RRF needs OpenSearch 2.19+. Weighted min-max works on 2.11+.",
    )
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for OpenSearch to accept connections.",
    )
    args = parser.parse_args()

    config = HybridConfig(
        hosts=[args.host],
        index=DEMO_INDEX,
        dimension=DIMENSION,
        fusion=FusionMethod(args.fusion),
        pipeline_name=f"hybrid-compare-{args.fusion}-pipeline",
        lexical_weight=0.5,
        vector_weight=0.5,
        size=args.size,
        knn_k=max(10, args.size),
        use_ssl=args.host.startswith("https://"),
    )
    kit = HybridKit(config)
    wait_for_opensearch(kit, url=args.host, timeout=args.timeout)

    kit.delete_index()
    kit.ensure_index(
        extra_properties={
            "title": {"type": "text"},
            "category": {"type": "keyword"},
        }
    )
    kit.upsert_pipeline()
    kit.index_documents(embed_documents(DOCUMENTS, DIMENSION))

    query_vector = embed_text(args.query, DIMENSION)
    # Restrict _source to title/content/category; the vector is dropped either way
    # because config.exclude_vector defaults to True.
    source_includes = ["title", "content", "category"]
    lexical = kit.lexical_search(
        args.query, size=args.size, source_includes=source_includes
    )
    knn = kit.knn_search(
        query_vector, size=args.size, source_includes=source_includes
    )
    hybrid = kit.hybrid_search(
        args.query, query_vector, size=args.size, source_includes=source_includes
    )

    print(f"fusion={config.fusion.value} pipeline={config.pipeline_name}")
    print(f"query={args.query!r}")
    print()
    print_side_by_side(
        [
            ("BM25 (lexical_search)", lexical),
            ("kNN (knn_search)", knn),
            (f"hybrid ({config.fusion.value})", hybrid),
        ],
        size=args.size,
    )

    filtered = kit.hybrid_search(
        args.query,
        query_vector,
        size=args.size,
        filter_query={"term": {"category": "footwear"}},
        source_includes=source_includes,
    )
    print()
    print("hybrid + filter term(category=footwear), _source includes title/content/category:")
    for rank, hit in enumerate(filtered.hits, start=1):
        title = hit.source.get("title", hit.id)
        category = hit.source.get("category", "")
        print(f"{rank}. {hit.score:.4f}  {title}  [{category}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
