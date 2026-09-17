from __future__ import annotations

import os

import pytest

from os_hybrid_kit import FusionMethod, HybridConfig, HybridKit

pytestmark = pytest.mark.integration

OPENSEARCH_URL = os.environ.get("OPENSEARCH_URL", "http://localhost:9200")


@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="set RUN_INTEGRATION=1 against a live OpenSearch (see docker compose)",
)
def test_hybrid_search_against_live_opensearch():
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
    )
    kit = HybridKit(config)
    if kit.client.indices.exists(index=config.index):
        kit.client.indices.delete(index=config.index)
    kit.ensure_index(extra_properties={"title": {"type": "text"}})
    kit.upsert_pipeline()
    kit.index_documents(
        [
            {
                "_id": "1",
                "title": "red",
                "content": "red running shoes",
                "embedding": [1.0, 0.0],
            },
            {
                "_id": "2",
                "title": "blue",
                "content": "blue office chair",
                "embedding": [0.0, 1.0],
            },
        ]
    )
    result = kit.hybrid_search("running shoes", [1.0, 0.0], size=2)
    assert result.hits
    assert result.hits[0].id == "1"
