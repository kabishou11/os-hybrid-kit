from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from os_hybrid_kit import FusionMethod, HybridConfig, HybridKit

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1",
        reason="set RUN_INTEGRATION=1 against a live OpenSearch (see docker compose)",
    ),
]

OPENSEARCH_URL = os.environ.get("OPENSEARCH_URL", "http://localhost:9200")


@pytest.fixture
def live_kit() -> Iterator[HybridKit]:
    config = HybridConfig(
        hosts=[OPENSEARCH_URL],
        index="os-hybrid-kit-itest",
        dimension=2,
        fusion=FusionMethod.WEIGHTED,
        pipeline_name="os-hybrid-kit-itest-pipeline",
        lexical_weight=0.5,
        vector_weight=0.5,
        size=3,
        knn_k=3,
        use_ssl=OPENSEARCH_URL.startswith("https://"),
        verify_certs=False,
    )
    kit = HybridKit(config)
    kit.delete_index()
    try:
        kit.ensure_index(
            extra_properties={
                "title": {"type": "text"},
                "category": {"type": "keyword"},
            }
        )
        kit.upsert_pipeline()
        kit.index_documents(
            [
                {
                    "_id": "1",
                    "title": "red",
                    "content": "red running shoes",
                    "category": "footwear",
                    "embedding": [1.0, 0.0],
                },
                {
                    "_id": "2",
                    "title": "blue",
                    "content": "blue office chair",
                    "category": "furniture",
                    "embedding": [0.0, 1.0],
                },
            ]
        )
        yield kit
    finally:
        kit.delete_index()


def test_hybrid_search_against_live_opensearch(live_kit: HybridKit):
    result = live_kit.hybrid_search("running shoes", [1.0, 0.0], size=2)
    assert result.hits
    assert result.hits[0].id == "1"


def test_lexical_knn_and_hybrid_return_hits(live_kit: HybridKit):
    lexical = live_kit.lexical_search("running shoes", size=2)
    knn = live_kit.knn_search([0.0, 1.0], size=2)
    hybrid = live_kit.hybrid_search("running shoes", [0.0, 1.0], size=2)

    assert lexical.hits
    assert knn.hits
    assert hybrid.hits
    assert lexical.hits[0].id == "1"
    assert knn.hits[0].id == "2"
    assert {hit.id for hit in hybrid.hits} >= {"1", "2"}


def test_put_and_get_alias_then_search_via_alias(live_kit: HybridKit):
    alias = "os-hybrid-kit-itest-alias"
    live_kit.put_alias(alias)
    mapping = live_kit.get_alias(alias)
    assert live_kit.config.index in mapping
    aliased = live_kit.with_index(alias)
    result = aliased.lexical_search("running shoes", size=2)
    assert result.hits
    assert result.hits[0].id == "1"
    live_kit.delete_alias(alias)
