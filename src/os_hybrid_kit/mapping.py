from __future__ import annotations

from typing import Any

from os_hybrid_kit.config import HybridConfig


def build_knn_vector_property(
    dimension: int,
    *,
    space_type: str = "l2",
    engine: str = "lucene",
    method_name: str = "hnsw",
    hnsw_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """OpenSearch 2.x ``knn_vector`` field mapping (Lucene HNSW by default)."""
    method: dict[str, Any] = {"name": method_name, "engine": engine}
    if hnsw_parameters:
        method["parameters"] = dict(hnsw_parameters)
    return {
        "type": "knn_vector",
        "dimension": dimension,
        "space_type": space_type,
        "method": method,
    }


def build_index_body(
    config: HybridConfig,
    *,
    extra_properties: dict[str, Any] | None = None,
    extra_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Index settings + mappings for BM25 text plus a kNN vector field.

    Sets ``index.knn: true`` as required by the k-NN plugin. Extra properties
    (title, metadata, ids) are merged into ``mappings.properties``.
    """
    properties: dict[str, Any] = {
        config.text_field: {"type": "text"},
        config.vector_field: build_knn_vector_property(
            config.dimension,
            space_type=config.space_type,
            engine=config.engine,
            method_name=config.method_name,
            hnsw_parameters=config.hnsw_parameters,
        ),
    }
    if extra_properties:
        for name, spec in extra_properties.items():
            if name in properties:
                raise ValueError(f"extra property {name!r} collides with a reserved field")
            properties[name] = spec

    settings: dict[str, Any] = {
        "index.knn": True,
        "number_of_shards": config.number_of_shards,
    }
    if extra_settings:
        settings.update(extra_settings)

    return {
        "settings": settings,
        "mappings": {"properties": properties},
    }
