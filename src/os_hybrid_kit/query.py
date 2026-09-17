from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def build_lexical_clause(query: str, text_field: str) -> dict[str, Any]:
    return {"match": {text_field: {"query": query}}}


def build_knn_clause(
    embedding: Sequence[float],
    vector_field: str,
    *,
    k: int,
) -> dict[str, Any]:
    return {"knn": {vector_field: {"vector": list(embedding), "k": k}}}


def build_hybrid_query(
    query: str,
    embedding: Sequence[float],
    *,
    text_field: str,
    vector_field: str,
    size: int = 10,
    knn_k: int | None = None,
    filter_query: Mapping[str, Any] | None = None,
    source_excludes: Sequence[str] | None = None,
    source_includes: Sequence[str] | None = None,
    extra_body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an OpenSearch ``hybrid`` query body (BM25 match + kNN).

    Clause order is lexical then vector so pipeline ``weights`` align with
    ``[lexical_weight, vector_weight]``.
    """
    k = knn_k if knn_k is not None else size
    hybrid: dict[str, Any] = {
        "queries": [
            build_lexical_clause(query, text_field),
            build_knn_clause(embedding, vector_field, k=k),
        ]
    }
    if filter_query is not None:
        hybrid["filter"] = dict(filter_query)

    body: dict[str, Any] = {
        "size": size,
        "query": {"hybrid": hybrid},
    }

    if source_excludes or source_includes:
        source: dict[str, Any] = {}
        if source_excludes:
            source["excludes"] = list(source_excludes)
        if source_includes:
            source["includes"] = list(source_includes)
        body["_source"] = source

    if extra_body:
        body.update(dict(extra_body))
    return body
