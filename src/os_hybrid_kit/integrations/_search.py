"""Shared search dispatch for optional retriever wrappers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from os_hybrid_kit.client import HybridKit
from os_hybrid_kit.results import Hit, SearchResult
from os_hybrid_kit.types import EmbedFn

_MODES = {"hybrid", "lexical", "knn"}


def run_kit_search(
    kit: HybridKit,
    query: str,
    embed_fn: EmbedFn,
    search_kwargs: Mapping[str, Any] | None = None,
) -> SearchResult:
    """Run hybrid, lexical, or kNN search on ``kit``.

    ``search_kwargs`` may include ``size``, ``filter`` (LangChain-style alias
    of ``filter_query``), ``mode`` (``hybrid`` / ``lexical`` / ``knn``), and
    other HybridKit search keyword arguments (``knn_k``, ``source_includes``,
    ``extra_body``, ...). Setting both ``filter`` and ``filter_query`` raises
    ``ValueError``. Lexical mode does not call ``embed_fn``.
    """
    kwargs = dict(search_kwargs or {})
    mode = kwargs.pop("mode", "hybrid")
    if mode not in _MODES:
        raise ValueError(
            f"unknown search mode {mode!r}; expected 'hybrid', 'lexical', or 'knn'"
        )
    if "filter" in kwargs and "filter_query" in kwargs:
        raise ValueError(
            "search_kwargs cannot set both 'filter' and 'filter_query'; "
            "'filter' is the LangChain-style alias of 'filter_query'"
        )
    filter_query = kwargs.pop("filter", None)
    if filter_query is None:
        filter_query = kwargs.pop("filter_query", None)
    if filter_query is not None:
        kwargs["filter_query"] = filter_query

    if mode == "lexical":
        return kit.lexical_search(query, **kwargs)
    embedding = list(embed_fn(query))
    if mode == "knn":
        return kit.knn_search(embedding, **kwargs)
    return kit.hybrid_search(query, embedding, **kwargs)


def hit_text_and_metadata(
    hit: Hit,
    *,
    text_field: str,
    vector_field: str,
) -> tuple[str, dict[str, Any]]:
    """Page text plus metadata ``{id, score, ...source}`` (vector field dropped)."""
    source = dict(hit.source)
    text = str(source.pop(text_field, ""))
    source.pop(vector_field, None)
    metadata: dict[str, Any] = {"id": hit.id, "score": hit.score, **source}
    return text, metadata
