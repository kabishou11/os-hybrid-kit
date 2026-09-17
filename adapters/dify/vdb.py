from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from os_hybrid_kit.client import HybridKit


@dataclass
class Document:
    """Minimal stand-in for Dify/LangChain-style documents."""

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    vector: list[float] | None = None
    score: float | None = None


class HybridVectorStore:
    """Thin Dify VDB stub. Not a marketplace plugin.

    Call this from a real Dify vector-store plugin, or from the external
    knowledge adapter. Index/search go through :class:`HybridKit`.
    """

    def __init__(self, kit: HybridKit, *, title_field: str = "title") -> None:
        self.kit = kit
        self.title_field = title_field

    def add_texts(
        self,
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        *,
        ids: Sequence[str] | None = None,
        metadatas: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[str]:
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings must have the same length")
        documents: list[dict[str, Any]] = []
        assigned_ids: list[str] = []
        for i, text in enumerate(texts):
            doc_id = str(ids[i]) if ids is not None else str(i)
            assigned_ids.append(doc_id)
            metadata = dict(metadatas[i]) if metadatas is not None else {}
            body: dict[str, Any] = {
                "_id": doc_id,
                self.kit.config.text_field: text,
                self.kit.config.vector_field: list(embeddings[i]),
            }
            if self.title_field in metadata:
                body[self.title_field] = metadata.pop(self.title_field)
            body.update(metadata)
            documents.append(body)
        self.kit.index_documents(documents)
        return assigned_ids

    def hybrid_search(
        self,
        query: str,
        embedding: Sequence[float],
        *,
        top_k: int = 4,
    ) -> list[Document]:
        result = self.kit.hybrid_search(query, embedding, size=top_k)
        return [_hit_to_document(self.kit, hit, self.title_field) for hit in result.hits]

    def search_by_vector(
        self,
        query_vector: Sequence[float],
        *,
        top_k: int = 4,
    ) -> list[Document]:
        result = self.kit.knn_search(query_vector, size=top_k)
        return [_hit_to_document(self.kit, hit, self.title_field) for hit in result.hits]

    def search_by_full_text(self, query: str, *, top_k: int = 4) -> list[Document]:
        result = self.kit.lexical_search(query, size=top_k)
        return [_hit_to_document(self.kit, hit, self.title_field) for hit in result.hits]


def _hit_to_document(kit: HybridKit, hit: Any, title_field: str) -> Document:
    source = dict(hit.source)
    text = str(source.pop(kit.config.text_field, ""))
    source.pop(kit.config.vector_field, None)
    metadata = dict(source)
    metadata.setdefault("document_id", hit.id)
    if title_field not in metadata:
        metadata[title_field] = hit.id
    return Document(page_content=text, metadata=metadata, score=hit.score)
