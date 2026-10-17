from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

from os_hybrid_kit import HybridConfig, HybridKit, parse_search_response
from os_hybrid_kit.adapters.dify import (
    HybridVectorStore,
    RetrievalRequest,
    RetrievalSetting,
    hits_to_records,
    retrieve,
)


class FakeClient:
    def __init__(self) -> None:
        self.searches: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> dict[str, Any]:
        self.searches.append(kwargs)
        return {
            "hits": {
                "total": {"value": 2, "relation": "eq"},
                "max_score": 0.9,
                "hits": [
                    {
                        "_id": "doc-1",
                        "_index": "kb",
                        "_score": 0.9,
                        "_source": {
                            "content": "Dify retrieves external chunks.",
                            "title": "intro.txt",
                            "path": "s3://bucket/intro.txt",
                            "embedding": [0.1, 0.2],
                        },
                    },
                    {
                        "_id": "doc-2",
                        "_index": "kb",
                        "_score": 0.2,
                        "_source": {"content": "Unrelated.", "title": "other.txt"},
                    },
                ],
            }
        }


def test_hits_to_records_maps_dify_fields_and_strips_vector():
    result = parse_search_response(FakeClient().search())
    records = hits_to_records(
        result,
        text_field="content",
        vector_field="embedding",
        score_threshold=0.5,
    )
    assert len(records) == 1
    record = records[0]
    assert record.content == "Dify retrieves external chunks."
    assert record.title == "intro.txt"
    assert record.score == 0.9
    assert record.metadata["path"] == "s3://bucket/intro.txt"
    assert record.metadata["document_id"] == "doc-1"
    assert "embedding" not in record.metadata
    assert "content" not in record.metadata


def test_retrieve_embeds_query_and_uses_top_k():
    kit = HybridKit(
        HybridConfig(dimension=2, index="kb", size=10),
        client=FakeClient(),  # type: ignore[arg-type]
    )
    request = RetrievalRequest(
        knowledge_id="kb-1",
        query="What is Dify?",
        retrieval_setting=RetrievalSetting(top_k=3, score_threshold=0.0),
    )
    response = retrieve(kit, request, embed_fn=lambda _q: [0.5, 0.5])
    search = kit.client.searches[0]  # type: ignore[attr-defined]
    assert search["body"]["size"] == 3
    assert search["body"]["query"]["hybrid"]["queries"][1]["knn"]["embedding"]["vector"] == [
        0.5,
        0.5,
    ]
    assert len(response.records) == 2
    assert response.model_dump()["records"][0]["title"] == "intro.txt"


def test_vdb_stub_hybrid_search_returns_documents():
    kit = HybridKit(HybridConfig(dimension=2, index="kb"), client=FakeClient())  # type: ignore[arg-type]
    store = HybridVectorStore(kit)
    docs = store.hybrid_search("What is Dify?", [0.5, 0.5], top_k=2)
    assert docs[0].page_content == "Dify retrieves external chunks."
    assert docs[0].metadata["title"] == "intro.txt"
    assert docs[0].score == 0.9


def test_vdb_search_by_vector_rejects_wrong_dimension():
    kit = HybridKit(HybridConfig(dimension=2, index="kb"), client=FakeClient())  # type: ignore[arg-type]
    store = HybridVectorStore(kit)
    with pytest.raises(ValueError, match="embedding length"):
        store.search_by_vector([0.5], top_k=2)
    assert kit.client.searches == []  # type: ignore[attr-defined]


def test_vdb_search_by_vector_uses_knn_search():
    kit = HybridKit(HybridConfig(dimension=2, index="kb"), client=FakeClient())  # type: ignore[arg-type]
    store = HybridVectorStore(kit)
    docs = store.search_by_vector([0.5, 0.5], top_k=2)
    search = kit.client.searches[0]  # type: ignore[attr-defined]
    assert "params" not in search
    knn = search["body"]["query"]["knn"]["embedding"]
    assert knn["vector"] == [0.5, 0.5]
    assert search["body"]["size"] == 2
    assert docs[0].page_content == "Dify retrieves external chunks."


def test_legacy_adapters_dify_import_warns() -> None:
    # Pytest puts tests/ on sys.path, not the checkout root. The deprecated
    # shim is a source-tree path, so add that root for this import only.
    root = Path(__file__).resolve().parents[1]
    shim = root / "adapters" / "dify" / "__init__.py"
    if not shim.is_file():
        pytest.importorskip("adapters.dify")

    root_s = str(root)
    added = root_s not in sys.path
    if added:
        sys.path.insert(0, root_s)
    sys.modules.pop("adapters.dify", None)
    sys.modules.pop("adapters", None)
    try:
        with pytest.warns(DeprecationWarning, match="os_hybrid_kit.adapters.dify"):
            legacy = importlib.import_module("adapters.dify")
    finally:
        if added:
            sys.path.remove(root_s)

    assert Path(legacy.__file__).resolve() == shim.resolve()
    assert legacy.retrieve is retrieve
    assert legacy.HybridVectorStore is HybridVectorStore


def test_vdb_search_by_full_text_uses_lexical_search():
    kit = HybridKit(HybridConfig(dimension=2, index="kb"), client=FakeClient())  # type: ignore[arg-type]
    store = HybridVectorStore(kit)
    docs = store.search_by_full_text("What is Dify?", top_k=2)
    search = kit.client.searches[0]  # type: ignore[attr-defined]
    assert "params" not in search
    assert search["body"]["query"] == {"match": {"content": {"query": "What is Dify?"}}}
    assert search["body"]["size"] == 2
    assert docs[0].page_content == "Dify retrieves external chunks."
