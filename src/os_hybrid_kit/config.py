from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FusionMethod(str, Enum):
    """How OpenSearch combines BM25 and kNN subquery results."""

    RRF = "rrf"
    WEIGHTED = "weighted"


class HybridConfig(BaseModel):
    """Connection, mapping, pipeline, and search defaults for hybrid search.

    OpenSearch versions:
      * hybrid query + ``normalization-processor``: 2.11+
      * RRF via ``score-ranker-processor``: 2.19+
    """

    model_config = ConfigDict(extra="forbid")

    hosts: list[str] = Field(default_factory=lambda: ["http://localhost:9200"])
    username: str | None = None
    password: str | None = None
    use_ssl: bool = False
    verify_certs: bool = False
    ssl_show_warn: bool = False
    request_timeout: float = 30.0

    index: str = "hybrid-index"
    text_field: str = "content"
    vector_field: str = "embedding"
    dimension: int = Field(..., gt=0)
    space_type: Literal["l2", "cosinesimil", "innerproduct"] = "l2"
    engine: Literal["lucene", "nmslib", "faiss"] = "lucene"
    method_name: str = "hnsw"
    hnsw_parameters: dict[str, Any] | None = None
    number_of_shards: int = Field(default=1, ge=1)

    fusion: FusionMethod = FusionMethod.RRF
    pipeline_name: str = "hybrid-search-pipeline"
    pipeline_description: str | None = None
    rank_constant: int = Field(default=60, ge=1)
    rrf_weights: tuple[float, float] | None = None
    normalization: Literal["min_max", "l2", "z_score"] = "min_max"
    combination: Literal["arithmetic_mean", "geometric_mean", "harmonic_mean"] = (
        "arithmetic_mean"
    )
    lexical_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0)

    size: int = Field(default=10, ge=1)
    knn_k: int = Field(default=10, ge=1)
    exclude_vector: bool = True

    @model_validator(mode="after")
    def _validate_fusion_weights(self) -> HybridConfig:
        if self.fusion is FusionMethod.WEIGHTED:
            total = self.lexical_weight + self.vector_weight
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    "lexical_weight + vector_weight must equal 1.0 for weighted fusion"
                )
        if self.rrf_weights is not None:
            total = self.rrf_weights[0] + self.rrf_weights[1]
            if abs(total - 1.0) > 1e-6:
                raise ValueError("rrf_weights must sum to 1.0")
            if any(w < 0.0 or w > 1.0 for w in self.rrf_weights):
                raise ValueError("rrf_weights values must be in [0.0, 1.0]")
        return self

    @property
    def weighted_weights(self) -> list[float]:
        return [self.lexical_weight, self.vector_weight]
