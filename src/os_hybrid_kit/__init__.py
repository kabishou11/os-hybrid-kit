"""Portable OpenSearch hybrid search kit (BM25 + kNN + RRF / weighted fusion).

Primary facade: ``HybridKit``, ``HybridConfig``, ``FusionMethod``, ``Hit``,
``SearchResult``. Mapping, pipeline, and query ``build_*`` helpers are advanced.
"""

from os_hybrid_kit.client import HybridKit, build_opensearch_client
from os_hybrid_kit.config import FusionMethod, HybridConfig
from os_hybrid_kit.mapping import build_index_body, build_knn_vector_property
from os_hybrid_kit.pipeline import (
    build_pipeline_body,
    build_rrf_pipeline_body,
    build_weighted_pipeline_body,
    upsert_search_pipeline,
)
from os_hybrid_kit.query import build_hybrid_query, build_knn_query, build_lexical_query
from os_hybrid_kit.results import Hit, SearchResult, parse_search_response
from os_hybrid_kit.types import EmbedFn

__version__ = "0.3.0"

__all__ = [
    "EmbedFn",
    "FusionMethod",
    "Hit",
    "HybridConfig",
    "HybridKit",
    "SearchResult",
    "__version__",
    "build_hybrid_query",
    "build_index_body",
    "build_knn_query",
    "build_knn_vector_property",
    "build_lexical_query",
    "build_opensearch_client",
    "build_pipeline_body",
    "build_rrf_pipeline_body",
    "build_weighted_pipeline_body",
    "parse_search_response",
    "upsert_search_pipeline",
]
