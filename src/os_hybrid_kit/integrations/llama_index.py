"""LlamaIndex retriever wrapping HybridKit.

Requires the ``llama-index`` extra: ``pip install os-hybrid-kit[llama-index]``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from os_hybrid_kit.client import HybridKit
from os_hybrid_kit.integrations import InstallError
from os_hybrid_kit.integrations._search import hit_text_and_metadata, run_kit_search
from os_hybrid_kit.types import EmbedFn

try:
    from llama_index.core.base.base_retriever import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
except ImportError as exc:
    raise InstallError(
        "LlamaIndex extra is not installed. "
        'Install with: pip install "os-hybrid-kit[llama-index]"'
    ) from exc


class HybridKitRetriever(BaseRetriever):
    """Thin LlamaIndex retriever around :class:`~os_hybrid_kit.client.HybridKit`.

    ``search_kwargs`` accepts ``size``, ``filter`` (OpenSearch clause), and
    ``mode``: ``hybrid`` (default), ``lexical``, or ``knn``.
    """

    def __init__(
        self,
        kit: HybridKit,
        embed_fn: EmbedFn,
        *,
        search_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.kit = kit
        self.embed_fn = embed_fn
        self.search_kwargs = dict(search_kwargs or {})

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        result = run_kit_search(
            self.kit, query_bundle.query_str, self.embed_fn, self.search_kwargs
        )
        text_field = self.kit.config.text_field
        vector_field = self.kit.config.vector_field
        nodes: list[NodeWithScore] = []
        for hit in result.hits:
            text, metadata = hit_text_and_metadata(
                hit, text_field=text_field, vector_field=vector_field
            )
            node = TextNode(text=text, id_=hit.id, metadata=metadata)
            nodes.append(NodeWithScore(node=node, score=hit.score))
        return nodes
