"""Deprecated import path. Use ``os_hybrid_kit.adapters.dify``.

This shim exists for source checkouts only. It is not included in the wheel.
"""

from __future__ import annotations

import warnings

from os_hybrid_kit.adapters.dify import (
    Document,
    HybridVectorStore,
    KnowledgeRecord,
    RetrievalRequest,
    RetrievalResponse,
    RetrievalSetting,
    hits_to_records,
    retrieve,
)

warnings.warn(
    "adapters is deprecated; import from os_hybrid_kit.adapters.dify",
    DeprecationWarning,
    stacklevel=2,
)

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
