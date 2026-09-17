"""Thin Dify adapter: External Knowledge ``/retrieval`` + a VDB stub.

This is not a Dify marketplace plugin. Wire ``retrieve()`` to any HTTP
framework, or call :class:`HybridVectorStore` from a plugin you maintain.
"""

from adapters.dify.retrieval import (
    KnowledgeRecord,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalSetting,
    hits_to_records,
    retrieve,
)
from adapters.dify.vdb import Document, HybridVectorStore

__all__ = [
    "Document",
    "HybridVectorStore",
    "KnowledgeRecord",
    "RetrievalRequest",
    "RetrievalResponse",
    "RetrievalSetting",
    "hits_to_records",
    "retrieve",
]
