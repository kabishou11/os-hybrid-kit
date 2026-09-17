from __future__ import annotations

from typing import Any

import pytest

from os_hybrid_kit import FusionMethod, HybridConfig, HybridKit


class FakeIndices:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.exists_index = False

    def exists(self, index: str) -> bool:
        return self.exists_index

    def create(self, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.created.append({"index": index, "body": body})
        self.exists_index = True
        return {"acknowledged": True}


class FakeTransport:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def perform_request(self, method: str, path: str, body: dict[str, Any] | None = None):
        self.calls.append({"method": method, "path": path, "body": body})
        return {"acknowledged": True}


class FakeClient:
    def __init__(self) -> None:
        self.indices = FakeIndices()
        self.transport = FakeTransport()
        self.searches: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> dict[str, Any]:
        self.searches.append(kwargs)
        return {
            "hits": {
                "total": {"value": 1, "relation": "eq"},
                "max_score": 0.5,
                "hits": [
                    {
                        "_id": "1",
                        "_index": "hybrid-test",
                        "_score": 0.5,
                        "_source": {"content": "trail running shoes"},
                    }
                ],
            }
        }


def test_ensure_index_creates_once():
    client = FakeClient()
    kit = HybridKit(HybridConfig(dimension=4, index="hybrid-test"), client=client)  # type: ignore[arg-type]
    assert kit.ensure_index(extra_properties={"title": {"type": "text"}}) is True
    assert kit.ensure_index() is False
    created = client.indices.created[0]
    assert created["index"] == "hybrid-test"
    assert created["body"]["settings"]["index.knn"] is True
    assert "title" in created["body"]["mappings"]["properties"]


def test_upsert_pipeline_puts_rrf_body():
    client = FakeClient()
    kit = HybridKit(
        HybridConfig(dimension=4, fusion=FusionMethod.RRF, pipeline_name="rrf-pipeline"),
        client=client,  # type: ignore[arg-type]
    )
    kit.upsert_pipeline()
    call = client.transport.calls[0]
    assert call["method"] == "PUT"
    assert call["path"] == "/_search/pipeline/rrf-pipeline"
    assert "score-ranker-processor" in call["body"]["phase_results_processors"][0]


def test_hybrid_search_sends_pipeline_and_hybrid_body():
    client = FakeClient()
    kit = HybridKit(HybridConfig(dimension=3, size=5, knn_k=10), client=client)  # type: ignore[arg-type]
    result = kit.hybrid_search("running shoes", [0.1, 0.2, 0.3], size=4)
    request = client.searches[0]
    assert request["index"] == "hybrid-index"
    assert request["params"] == {"search_pipeline": "hybrid-search-pipeline"}
    body = request["body"]
    assert body["size"] == 4
    knn = body["query"]["hybrid"]["queries"][1]["knn"]["embedding"]
    assert knn["k"] == 10
    assert knn["vector"] == [0.1, 0.2, 0.3]
    assert body["_source"]["excludes"] == ["embedding"]
    assert result.hits[0].id == "1"
    assert result.hits[0].source["content"] == "trail running shoes"


def test_hybrid_search_rejects_wrong_embedding_length():
    kit = HybridKit(HybridConfig(dimension=3), client=FakeClient())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="embedding length"):
        kit.hybrid_search("q", [0.1, 0.2])
