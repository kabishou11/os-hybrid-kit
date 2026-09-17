#!/usr/bin/env python3
"""Seed a few docs and run a hybrid query against local OpenSearch.

Usage (from the repo root, with the cluster up):

    python -m pip install -e .
    docker compose up -d
    python examples/seed_and_search.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from embed import embed_text  # noqa: E402
from os_hybrid_kit import FusionMethod, HybridConfig, HybridKit  # noqa: E402

DEMO_INDEX = "hybrid-demo"
DIMENSION = 32

DOCUMENTS = [
    {
        "_id": "1",
        "title": "Trail running shoes",
        "content": "Lightweight trail running shoes with aggressive lugs for muddy wet paths.",
    },
    {
        "_id": "2",
        "title": "Office chair",
        "content": "Ergonomic office chair with lumbar support for long workdays at a desk.",
    },
    {
        "_id": "3",
        "title": "Road cycling helmet",
        "content": "Ventilated cycling helmet designed for road bikes and summer heat.",
    },
    {
        "_id": "4",
        "title": "Waterproof hiking boots",
        "content": "Waterproof hiking boots with a grippy outsole for rocky wet trails.",
    },
    {
        "_id": "5",
        "title": "Espresso machine",
        "content": "Compact espresso machine with a steam wand for home coffee drinks.",
    },
    {
        "_id": "6",
        "title": "Running jacket",
        "content": "Packable running jacket that sheds light rain on early morning runs.",
    },
]


def wait_for_cluster(kit: HybridKit, timeout: float) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            if kit.client.ping():
                return
        except Exception as exc:
            last_error = exc
        time.sleep(2)
    raise SystemExit(f"OpenSearch did not become ready: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed docs and run a hybrid query.")
    parser.add_argument(
        "--host",
        default=os.environ.get("OPENSEARCH_URL", "http://localhost:9200"),
    )
    parser.add_argument(
        "--fusion",
        choices=["rrf", "weighted"],
        default="rrf",
        help="RRF needs OpenSearch 2.19+. Weighted min-max works on 2.11+.",
    )
    parser.add_argument("--query", default="running shoes for wet trails")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    config = HybridConfig(
        hosts=[args.host],
        index=DEMO_INDEX,
        dimension=DIMENSION,
        fusion=FusionMethod(args.fusion),
        pipeline_name=f"hybrid-{args.fusion}-pipeline",
        size=5,
        knn_k=10,
        use_ssl=args.host.startswith("https://"),
    )
    kit = HybridKit(config)
    wait_for_cluster(kit, args.timeout)

    kit.delete_index()
    kit.ensure_index(extra_properties={"title": {"type": "text"}})
    kit.upsert_pipeline()

    seeded = []
    for doc in DOCUMENTS:
        body = dict(doc)
        body["embedding"] = embed_text(f"{body['title']} {body['content']}", DIMENSION)
        seeded.append(body)
    kit.index_documents(seeded)

    query_vector = embed_text(args.query, DIMENSION)
    result = kit.hybrid_search(args.query, query_vector, size=5)

    print(f"fusion={config.fusion.value} pipeline={config.pipeline_name}")
    print(f"query={args.query!r} total={result.total}")
    for rank, hit in enumerate(result.hits, start=1):
        title = hit.source.get("title", hit.id)
        content = hit.source.get("content", "")
        print(f"{rank}. {hit.score:.4f}  {title}  — {content}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
