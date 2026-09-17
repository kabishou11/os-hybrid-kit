from __future__ import annotations

from typing import Any

import pytest

from os_hybrid_kit import HybridConfig, HybridKit

pytest.importorskip("llama_index.core")

from os_hybrid_kit.integrations.llama_index import HybridKitRetriever  # noqa: E402


class FakeClient:
    def __init__(self) -> None:
        self.searches: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> dict[str, Any]:
        self.searches.append(kwargs)
        return {
            "hits": {
                "total": {"value": 1, "relation": "eq"},
                "max_score": 0.6,
                "hits": [
                    {
                        "_id": "n-1",
                        "_index": "hybrid-test",
                        "_score": 0.6,
                        "_source": {
                            "content": "lightweight trail running shoes",
                            "brand": "Acme",
                            "embedding": [0.4, 0.3, 0.2, 0.1],
                        },
                    }
                ],
            }
        }


def test_llama_index_retriever_hybrid_maps_nodes() -> None:
    kit = HybridKit(
        HybridConfig(dimension=4, index="hybrid-test"),
        client=FakeClient(),  # type: ignore[arg-type]
    )
    retriever = HybridKitRetriever(
        kit,
        lambda _q: [0.4, 0.3, 0.2, 0.1],
        search_kwargs={"size": 8, "mode": "hybrid"},
    )
    nodes = retriever.retrieve("running shoes")
    assert len(nodes) == 1
    assert nodes[0].get_text() == "lightweight trail running shoes"
    assert nodes[0].score == 0.6
    assert nodes[0].node.id_ == "n-1"
    assert nodes[0].metadata["brand"] == "Acme"
    assert nodes[0].metadata["id"] == "n-1"
    assert "embedding" not in nodes[0].metadata
    search = kit.client.searches[0]  # type: ignore[attr-defined]
    assert search["body"]["size"] == 8
    assert "hybrid" in search["body"]["query"]


def test_llama_index_retriever_knn_mode() -> None:
    kit = HybridKit(
        HybridConfig(dimension=4, index="hybrid-test"),
        client=FakeClient(),  # type: ignore[arg-type]
    )
    retriever = HybridKitRetriever(
        kit,
        lambda _q: [0.4, 0.3, 0.2, 0.1],
        search_kwargs={"mode": "knn"},
    )
    nodes = retriever.retrieve("running shoes")
    assert nodes[0].score == 0.6
    search = kit.client.searches[0]  # type: ignore[attr-defined]
    assert "knn" in search["body"]["query"]
    assert "params" not in search
