from __future__ import annotations

import pytest

from os_hybrid_kit import HybridConfig, build_index_body, build_knn_vector_property


def test_knn_vector_property_lucene_hnsw():
    prop = build_knn_vector_property(8, space_type="l2", engine="lucene")
    assert prop["type"] == "knn_vector"
    assert prop["dimension"] == 8
    assert prop["space_type"] == "l2"
    assert prop["method"]["name"] == "hnsw"
    assert prop["method"]["engine"] == "lucene"
    assert "parameters" not in prop["method"]


def test_knn_vector_property_includes_hnsw_parameters():
    prop = build_knn_vector_property(
        3, hnsw_parameters={"ef_construction": 128, "m": 16}
    )
    assert prop["method"]["parameters"] == {"ef_construction": 128, "m": 16}


def test_index_body_enables_knn_and_maps_text_plus_vector(config: HybridConfig):
    body = build_index_body(config)
    assert body["settings"]["index.knn"] is True
    assert body["settings"]["number_of_shards"] == 1
    properties = body["mappings"]["properties"]
    assert properties["content"] == {"type": "text"}
    assert properties["embedding"]["type"] == "knn_vector"
    assert properties["embedding"]["dimension"] == 4


def test_index_body_merges_extra_properties(config: HybridConfig):
    body = build_index_body(
        config,
        extra_properties={"title": {"type": "text"}, "category": {"type": "keyword"}},
    )
    properties = body["mappings"]["properties"]
    assert properties["title"] == {"type": "text"}
    assert properties["category"] == {"type": "keyword"}
    assert "content" in properties
    assert "embedding" in properties


def test_index_body_rejects_colliding_extra_property(config: HybridConfig):
    with pytest.raises(ValueError, match="collides"):
        build_index_body(config, extra_properties={"content": {"type": "keyword"}})
