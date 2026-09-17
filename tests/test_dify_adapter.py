from __future__ import annotations

from typing import Any

from adapters.dify import (
    HybridVectorStore,
    RetrievalRequest,
    RetrievalSetting,
    hits_to_records,
    retrieve,
)

from os_hybrid_kit import HybridConfig, HybridKit, parse_search_response


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
