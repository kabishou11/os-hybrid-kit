#!/usr/bin/env python3
"""Time BM25, kNN, and hybrid on a fixed local corpus.

Prints hit ids, scores, and wall-clock milliseconds for each (mode, query).
These are stopwatch numbers on this machine and this tiny index — not a
published result, not nDCG, and not a quality claim.

Usage (from the repo root, with the cluster up):

    python -m pip install -e .
    docker compose up -d
    python examples/benchmark_retrievers.py
    python examples/benchmark_retrievers.py --url http://localhost:9200 --n-queries 10 --fusion weighted
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from embed import embed_documents, embed_text, wait_for_opensearch  # noqa: E402
from os_hybrid_kit import FusionMethod, HybridConfig, HybridKit, SearchResult  # noqa: E402

DEMO_INDEX = "hybrid-bench"
DIMENSION = 32
SIZE = 3

DOCUMENTS = [
    {
        "_id": "trail-shoes",
        "title": "Trail running shoes",
        "content": "Lightweight trail running shoes with aggressive lugs for muddy wet paths.",
    },
    {
        "_id": "hiking-boots",
        "title": "Waterproof hiking boots",
        "content": "Waterproof hiking boots with a grippy outsole for rocky wet trails.",
    },
    {
        "_id": "jacket",
        "title": "Running jacket",
        "content": "Packable running jacket that sheds light rain on early morning runs.",
    },
    {
        "_id": "helmet",
        "title": "Road cycling helmet",
        "content": "Ventilated cycling helmet designed for road bikes and summer heat.",
    },
    {
        "_id": "chair",
        "title": "Office chair",
        "content": "Ergonomic office chair with lumbar support for long workdays at a desk.",
    },
    {
        "_id": "espresso",
        "title": "Espresso machine",
        "content": "Compact espresso machine with a steam wand for home coffee drinks.",
    },
    {
        "_id": "coverage",
        "title": "Query-shaped trail copy",
        "content": "Waterproof hiking boots for muddy trails.",
    },
    {
        "_id": "stuffed",
        "title": "Warehouse boots (keyword-stuffed)",
        "content": "boots boots boots boots boots boots boots for warehouse staff",
    },
]

QUERIES = [
    "hiking boots",
    "waterproof trail shoes",
    "office chair lumbar",
    "espresso machine",
    "running jacket rain",
    "cycling helmet",
    "muddy wet paths",
    "grippy outsole trails",
    "warehouse boots",
    "home coffee drinks",
    "packable jacket morning",
    "road bike helmet heat",
]


def _hits_cell(result: SearchResult) -> str:
    return ", ".join(f"{hit.id}:{hit.score:.4f}" for hit in result.hits)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Time BM25, kNN, and hybrid on a fixed local corpus (wall-clock ms)."
    )
    parser.add_argument(
        "--host",
        "--url",
        default=os.environ.get("OPENSEARCH_URL", "http://localhost:9200"),
        help="OpenSearch URL (or OPENSEARCH_URL).",
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=8,
        help="How many queries from the fixed list to run (cycles if larger).",
    )
    parser.add_argument(
        "--fusion",
        choices=["rrf", "weighted"],
        default="rrf",
        help="RRF needs OpenSearch 2.19+. Weighted min-max works on 2.11+.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for OpenSearch to accept connections.",
    )
    args = parser.parse_args()
    if args.n_queries < 1:
        raise SystemExit("--n-queries must be >= 1")

    config = HybridConfig(
        hosts=[args.host],
        index=DEMO_INDEX,
        dimension=DIMENSION,
        fusion=FusionMethod(args.fusion),
        pipeline_name=f"hybrid-bench-{args.fusion}-pipeline",
        lexical_weight=0.5,
        vector_weight=0.5,
        size=SIZE,
        knn_k=max(10, SIZE),
        use_ssl=args.host.startswith("https://"),
    )
    kit = HybridKit(config)
    wait_for_opensearch(kit, url=args.host, timeout=args.timeout)

    kit.delete_index()
    kit.ensure_index(extra_properties={"title": {"type": "text"}})
    kit.upsert_pipeline()
    kit.index_documents(embed_documents(DOCUMENTS, DIMENSION))

    queries = [QUERIES[i % len(QUERIES)] for i in range(args.n_queries)]
    vectors = [embed_text(query, DIMENSION) for query in queries]
    modes = ("lexical", "knn", "hybrid")

    def search(mode: str, query: str, vector: list[float]) -> SearchResult:
        if mode == "lexical":
            return kit.lexical_search(query, size=SIZE)
        if mode == "knn":
            return kit.knn_search(vector, size=SIZE)
        return kit.hybrid_search(query, vector, size=SIZE)

    # One discarded search per mode so the first timed call is not a cold connection.
    for mode in modes:
        search(mode, queries[0], vectors[0])

    header = f"{'mode':<8} {'query':<28} {'wall_ms':>8}  hits"
    print(f"fusion={config.fusion.value} index={config.index} n_queries={len(queries)}")
    print("wall_ms is time.perf_counter around the search call (hash embed excluded).")
    print("Warm-up search discarded. Tiny local index; not a published result.")
    print()
    print(header)
    print("-" * len(header))

    times: dict[str, list[float]] = {mode: [] for mode in modes}
    for query, vector in zip(queries, vectors, strict=True):
        for mode in modes:
            started = time.perf_counter()
            result = search(mode, query, vector)
            wall_ms = (time.perf_counter() - started) * 1000.0
            times[mode].append(wall_ms)
            label = query if len(query) <= 28 else query[:27] + "…"
            print(f"{mode:<8} {label:<28} {wall_ms:8.2f}  {_hits_cell(result)}")

    print()
    print(
        "mean wall_ms  "
        + "  ".join(f"{mode}={_mean(times[mode]):.2f}" for mode in modes)
        + f"  (n={len(queries)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
