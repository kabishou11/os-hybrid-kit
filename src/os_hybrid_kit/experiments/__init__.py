"""Compact offline fusion experiment module (fixed scores + qrels only).

No network, OpenSearch, GPU, or live index is used. See :mod:`cli` for the
``python -m os_hybrid_kit.experiments`` entry point.
"""

from os_hybrid_kit.experiments.cli import main
from os_hybrid_kit.experiments.experiment import (
    ExperimentResult,
    FusionConfig,
    default_configs,
    load_fixture,
    run_config,
    sweep_configs,
)
from os_hybrid_kit.experiments.fusion import (
    EPSILON,
    dense_only,
    lexical_only,
    min_max_scores,
    rank_score_mix,
    rrf,
    weighted_fusion,
)
from os_hybrid_kit.experiments.metrics import evaluate, mrr_at_k, ndcg_at_k, recall_at_k
from os_hybrid_kit.experiments.tax import TaxReport, fusion_tax
from os_hybrid_kit.experiments.types import Fixture, FusionQuery, RunList, ScoredDoc, build_run_list

__all__ = [
    "EPSILON",
    "ExperimentResult",
    "Fixture",
    "FusionConfig",
    "FusionQuery",
    "RunList",
    "ScoredDoc",
    "TaxReport",
    "build_run_list",
    "default_configs",
    "dense_only",
    "evaluate",
    "fusion_tax",
    "lexical_only",
    "load_fixture",
    "main",
    "min_max_scores",
    "mrr_at_k",
    "ndcg_at_k",
    "rank_score_mix",
    "recall_at_k",
    "rrf",
    "run_config",
    "sweep_configs",
    "weighted_fusion",
]
