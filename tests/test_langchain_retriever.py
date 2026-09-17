from __future__ import annotations

from typing import Any

import pytest

from os_hybrid_kit import HybridConfig, HybridKit
from os_hybrid_kit.types import EmbedFn

pytest.importorskip("langchain_core")

from os_hybrid_kit.integrations.langchain import HybridKitRetriever  # noqa: E402


class FakeClient:
    def __init__(self) -> None:
        self.searches: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> dict[str, Any]:
        self.searches.append(kwargs)
        return {
            "hits": {
                "total": {"value": 1, "relation": "eq"},
                "max_score": 0.7,
                "hits": [
                    {
                        "_id": "42",
                        "_index": "hybrid-test",
                        "_score": 0.7,
                        "_source": {
                            "content": "muddy path boots",
                            "sku": "BOOT-1",
                            "embedding": [0.0, 0.1, 0.2, 0.3],
                        },
                    }
                ],
            }
        }


def test_langchain_retriever_kit_and_embed_fn_are_typed() -> None:
    fields = HybridKitRetriever.model_fields
    assert fields["kit"].annotation is HybridKit
    assert fields["embed_fn"].annotation is EmbedFn


def test_langchain_retriever_hybrid_maps_documents() -> None:
    kit = HybridKit(
        HybridConfig(dimension=4, index="hybrid-test"),
        client=FakeClient(),  # type: ignore[arg-type]
    )
    retriever = HybridKitRetriever(
        kit,
        lambda _q: [0.0, 0.1, 0.2, 0.3],
        search_kwargs={"size": 4, "mode": "hybrid"},
    )
    docs = retriever.invoke("boots")
    assert len(docs) == 1
    assert docs[0].page_content == "muddy path boots"
    assert docs[0].metadata["id"] == "42"
    assert docs[0].metadata["score"] == 0.7
    assert docs[0].metadata["sku"] == "BOOT-1"
    assert "embedding" not in docs[0].metadata
    search = kit.client.searches[0]  # type: ignore[attr-defined]
    assert search["body"]["size"] == 4
    assert "hybrid" in search["body"]["query"]


def test_langchain_retriever_lexical_mode() -> None:
    kit = HybridKit(
        HybridConfig(dimension=4, index="hybrid-test"),
        client=FakeClient(),  # type: ignore[arg-type]
    )

    def embed(_query: str) -> list[float]:
        raise AssertionError("lexical mode must not embed")

    retriever = HybridKitRetriever(kit, embed, search_kwargs={"mode": "lexical"})
    docs = retriever.invoke("boots")
    assert docs[0].page_content == "muddy path boots"
    search = kit.client.searches[0]  # type: ignore[attr-defined]
    assert "match" in search["body"]["query"]
