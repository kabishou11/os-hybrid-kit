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


def _with_source_and_extra(
    body: dict[str, Any],
    *,
    source_excludes: Sequence[str] | None = None,
    source_includes: Sequence[str] | None = None,
    extra_body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
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


def build_lexical_query(
    query: str,
    *,
    text_field: str,
    size: int = 10,
    filter_query: Mapping[str, Any] | None = None,
    source_excludes: Sequence[str] | None = None,
    source_includes: Sequence[str] | None = None,
    extra_body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a BM25 ``match`` query body. No search pipeline is required."""
    clause: dict[str, Any] = build_lexical_clause(query, text_field)
    if filter_query is not None:
        clause = {"bool": {"must": [clause], "filter": [dict(filter_query)]}}
    body: dict[str, Any] = {"size": size, "query": clause}
    return _with_source_and_extra(
        body,
        source_excludes=source_excludes,
        source_includes=source_includes,
        extra_body=extra_body,
    )


def build_knn_query(
    embedding: Sequence[float],
    *,
    vector_field: str,
    size: int = 10,
    knn_k: int | None = None,
    filter_query: Mapping[str, Any] | None = None,
    source_excludes: Sequence[str] | None = None,
    source_includes: Sequence[str] | None = None,
    extra_body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a raw ``knn`` query body. No search pipeline is required."""
    k = knn_k if knn_k is not None else size
    clause = build_knn_clause(embedding, vector_field, k=k)
    if filter_query is not None:
        clause["knn"][vector_field]["filter"] = dict(filter_query)
    body: dict[str, Any] = {"size": size, "query": clause}
    return _with_source_and_extra(
        body,
        source_excludes=source_excludes,
        source_includes=source_includes,
        extra_body=extra_body,
    )


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
    return _with_source_and_extra(
        body,
        source_excludes=source_excludes,
        source_includes=source_includes,
        extra_body=extra_body,
    )
