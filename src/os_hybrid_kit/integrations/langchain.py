"""LangChain retriever wrapping HybridKit.

Requires the ``langchain`` extra: ``pip install os-hybrid-kit[langchain]``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from os_hybrid_kit.client import HybridKit
from os_hybrid_kit.integrations import InstallError
from os_hybrid_kit.integrations._search import hit_text_and_metadata, run_kit_search
from os_hybrid_kit.types import EmbedFn

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
except ImportError as exc:
    raise InstallError(
        'LangChain extra is not installed. Install with: pip install "os-hybrid-kit[langchain]"'
    ) from exc


class HybridKitRetriever(BaseRetriever):
    """Thin LangChain retriever around :class:`~os_hybrid_kit.client.HybridKit`.

    ``search_kwargs`` accepts ``size``, ``filter`` (OpenSearch clause), and
    ``mode``: ``hybrid`` (default), ``lexical``, or ``knn``.
    """

    kit: Any
    embed_fn: Any
    search_kwargs: dict[str, Any]

    def __init__(
        self,
        kit: HybridKit,
        embed_fn: EmbedFn,
        search_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            kit=kit,
            embed_fn=embed_fn,
            search_kwargs=dict(search_kwargs or {}),
            **kwargs,
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any = None,
    ) -> list[Document]:
        result = run_kit_search(self.kit, query, self.embed_fn, self.search_kwargs)
        text_field = self.kit.config.text_field
        vector_field = self.kit.config.vector_field
        documents: list[Document] = []
        for hit in result.hits:
            text, metadata = hit_text_and_metadata(
                hit, text_field=text_field, vector_field=vector_field
            )
            documents.append(Document(page_content=text, metadata=metadata))
        return documents
