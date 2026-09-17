from __future__ import annotations

from typing import Any

from os_hybrid_kit.config import FusionMethod, HybridConfig


def build_rrf_pipeline_body(
    *,
    rank_constant: int = 60,
    weights: list[float] | tuple[float, float] | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Search pipeline using the rank-based ``score-ranker-processor`` (OpenSearch 2.19+).

    RRF combines subquery *ranks*, not raw scores::

        score(d) = sum_q  weight_q / (rank_constant + rank_q(d))
    """
    combination: dict[str, Any] = {
        "technique": "rrf",
        "rank_constant": rank_constant,
    }
    if weights is not None:
        combination["parameters"] = {"weights": list(weights)}
    return {
        "description": description or "Hybrid search RRF fusion",
        "phase_results_processors": [
            {"score-ranker-processor": {"combination": combination}}
        ],
    }


def build_weighted_pipeline_body(
    *,
    lexical_weight: float = 0.3,
    vector_weight: float = 0.7,
    normalization: str = "min_max",
    combination: str = "arithmetic_mean",
    description: str | None = None,
) -> dict[str, Any]:
    """Search pipeline using score-based ``normalization-processor`` (OpenSearch 2.11+).

    Default is min-max per subquery, then a weighted arithmetic mean.
    ``weights`` follow hybrid query clause order: lexical first, kNN second.
    They must sum to 1.0.
    """
    weights = [lexical_weight, vector_weight]
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("lexical_weight + vector_weight must equal 1.0")
    return {
        "description": description or "Hybrid search weighted score fusion",
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {"technique": normalization},
                    "combination": {
                        "technique": combination,
                        "parameters": {"weights": weights},
                    },
                }
            }
        ],
    }


def build_pipeline_body(config: HybridConfig) -> dict[str, Any]:
    """Build the search-pipeline body selected by ``config.fusion``."""
    if config.fusion is FusionMethod.RRF:
        weights = list(config.rrf_weights) if config.rrf_weights is not None else None
        return build_rrf_pipeline_body(
            rank_constant=config.rank_constant,
            weights=weights,
            description=config.pipeline_description,
        )
    return build_weighted_pipeline_body(
        lexical_weight=config.lexical_weight,
        vector_weight=config.vector_weight,
        normalization=config.normalization,
        combination=config.combination,
        description=config.pipeline_description,
    )


def upsert_search_pipeline(client: Any, name: str, body: dict[str, Any]) -> Any:
    """PUT ``/_search/pipeline/{name}``. Idempotent; replaces an existing pipeline."""
    return client.transport.perform_request(
        "PUT",
        f"/_search/pipeline/{name}",
        body=body,
    )
