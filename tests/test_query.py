from __future__ import annotations

from os_hybrid_kit import (
    build_hybrid_query,
    build_knn_query,
    build_lexical_query,
    parse_search_response,
)


def test_hybrid_query_clause_order_is_lexical_then_knn():
    body = build_hybrid_query(
        "running shoes",
        [0.1, 0.2, 0.3],
        text_field="content",
        vector_field="embedding",
        size=5,
        knn_k=12,
    )
    assert body["size"] == 5
    hybrid = body["query"]["hybrid"]
    assert hybrid["queries"][0] == {"match": {"content": {"query": "running shoes"}}}
    assert hybrid["queries"][1] == {
        "knn": {"embedding": {"vector": [0.1, 0.2, 0.3], "k": 12}}
    }
    assert "filter" not in hybrid


def test_hybrid_query_defaults_knn_k_to_size():
    body = build_hybrid_query(
        "q",
        [1.0],
        text_field="content",
        vector_field="embedding",
        size=7,
    )
    knn = body["query"]["hybrid"]["queries"][1]["knn"]["embedding"]
    assert knn["k"] == 7


def test_lexical_query_is_match_without_pipeline_fields():
    body = build_lexical_query("running shoes", text_field="content", size=3)
    assert body == {
        "size": 3,
        "query": {"match": {"content": {"query": "running shoes"}}},
    }


def test_lexical_query_wraps_filter_in_bool():
    body = build_lexical_query(
        "q",
        text_field="content",
        filter_query={"term": {"category": "shoes"}},
        source_includes=["title", "content"],
    )
    assert body["query"] == {
        "bool": {
            "must": [{"match": {"content": {"query": "q"}}}],
            "filter": [{"term": {"category": "shoes"}}],
        }
    }
    assert body["_source"]["includes"] == ["title", "content"]


def test_knn_query_is_raw_knn():
    body = build_knn_query(
        [0.1, 0.2],
        vector_field="embedding",
        size=4,
        knn_k=8,
        source_excludes=["embedding"],
    )
    assert body["size"] == 4
    assert body["query"] == {"knn": {"embedding": {"vector": [0.1, 0.2], "k": 8}}}
    assert body["_source"]["excludes"] == ["embedding"]


def test_knn_query_puts_filter_on_knn_clause():
    body = build_knn_query(
        [1.0],
        vector_field="embedding",
        filter_query={"term": {"category": "shoes"}},
    )
    knn = body["query"]["knn"]["embedding"]
    assert knn["filter"] == {"term": {"category": "shoes"}}
    assert knn["k"] == 10


def test_hybrid_query_optional_filter_and_source_excludes():
    body = build_hybrid_query(
        "q",
        [1.0, 0.0],
        text_field="content",
        vector_field="embedding",
        filter_query={"term": {"category": "shoes"}},
        source_excludes=["embedding"],
        extra_body={"timeout": "2s"},
    )
    assert body["query"]["hybrid"]["filter"] == {"term": {"category": "shoes"}}
    assert body["_source"]["excludes"] == ["embedding"]
    assert body["timeout"] == "2s"


def test_parse_search_response_accepts_total_object():
    result = parse_search_response(
        {
            "hits": {
                "total": {"value": 2, "relation": "eq"},
                "max_score": 1.2,
                "hits": [
                    {
                        "_id": "1",
                        "_index": "hybrid-test",
                        "_score": 1.2,
                        "_source": {"content": "hello"},
                    },
                    {
                        "_id": "2",
                        "_index": "hybrid-test",
                        "_score": 0.4,
                        "_source": {"content": "world"},
                    },
                ],
            }
        }
    )
    assert result.total == 2
    assert result.max_score == 1.2
    assert [hit.id for hit in result.hits] == ["1", "2"]
    assert result.texts("content") == ["hello", "world"]


def test_parse_search_response_accepts_integer_total():
    result = parse_search_response({"hits": {"total": 0, "hits": []}})
    assert result.total == 0
    assert result.hits == []
