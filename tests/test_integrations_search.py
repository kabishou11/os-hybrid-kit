from __future__ import annotations

from typing import Any

import pytest

from os_hybrid_kit import HybridConfig, HybridKit, parse_search_response
from os_hybrid_kit.integrations._search import hit_text_and_metadata, run_kit_search


class FakeClient:
    def __init__(self) -> None:
        self.searches: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> dict[str, Any]:
        self.searches.append(kwargs)
        return {
            "hits": {
                "total": {"value": 1, "relation": "eq"},
                "max_score": 0.8,
                "hits": [
                    {
                        "_id": "doc-1",
                        "_index": "hybrid-test",
                        "_score": 0.8,
                        "_source": {
                            "content": "waterproof trail shoes",
                            "title": "trail.txt",
                            "embedding": [0.1, 0.2, 0.3, 0.4],
                        },
                    }
                ],
            }
        }


def _kit() -> HybridKit:
    return HybridKit(
        HybridConfig(dimension=4, index="hybrid-test"),
        client=FakeClient(),  # type: ignore[arg-type]
    )


def test_run_kit_search_hybrid_embeds_query() -> None:
    kit = _kit()
    calls: list[str] = []

    def embed(text: str) -> list[float]:
        calls.append(text)
        return [0.1, 0.2, 0.3, 0.4]

    result = run_kit_search(kit, "trail shoes", embed, {"size": 5, "mode": "hybrid"})
    search = kit.client.searches[0]  # type: ignore[attr-defined]
    assert calls == ["trail shoes"]
    assert search["body"]["size"] == 5
    assert "hybrid" in search["body"]["query"]
    assert search["params"]["search_pipeline"] == kit.config.pipeline_name
    assert result.hits[0].id == "doc-1"


def test_run_kit_search_lexical_skips_embed_fn() -> None:
    kit = _kit()

    def embed(_text: str) -> list[float]:
        raise AssertionError("lexical mode must not embed")

    run_kit_search(kit, "trail shoes", embed, {"mode": "lexical", "size": 3})
    search = kit.client.searches[0]  # type: ignore[attr-defined]
    assert "match" in search["body"]["query"]
    assert "params" not in search


def test_run_kit_search_knn_and_filter() -> None:
    kit = _kit()
    run_kit_search(
        kit,
        "trail shoes",
        lambda _q: [0.1, 0.2, 0.3, 0.4],
        {"mode": "knn", "filter": {"term": {"category": "footwear"}}, "size": 2},
    )
    search = kit.client.searches[0]  # type: ignore[attr-defined]
    knn = search["body"]["query"]["knn"]["embedding"]
    assert knn["vector"] == [0.1, 0.2, 0.3, 0.4]
    assert knn["filter"] == {"term": {"category": "footwear"}}
    assert "params" not in search


def test_run_kit_search_rejects_unknown_mode() -> None:
    kit = _kit()
    with pytest.raises(ValueError, match="unknown search mode"):
        run_kit_search(kit, "q", lambda _q: [0.1, 0.2, 0.3, 0.4], {"mode": "neural"})


def test_hit_text_and_metadata_drops_vector() -> None:
    result = parse_search_response(FakeClient().search())
    text, metadata = hit_text_and_metadata(
        result.hits[0], text_field="content", vector_field="embedding"
    )
    assert text == "waterproof trail shoes"
    assert metadata["id"] == "doc-1"
    assert metadata["score"] == 0.8
    assert metadata["title"] == "trail.txt"
    assert "content" not in metadata
    assert "embedding" not in metadata
