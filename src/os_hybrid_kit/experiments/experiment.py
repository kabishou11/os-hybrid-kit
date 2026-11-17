"""Fixture loading, configuration definitions, and experiment orchestration."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from os_hybrid_kit.experiments.fusion import dense_only, lexical_only, score_query
from os_hybrid_kit.experiments.metrics import MetricSummary, evaluate, rank_by_score
from os_hybrid_kit.experiments.tax import TaxReport, fusion_tax
from os_hybrid_kit.experiments.types import Fixture, query_from_pairs, validate_depth


@dataclass(frozen=True)
class FusionConfig:
    name: str
    label: str
    method: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentResult:
    config: FusionConfig
    metrics: MetricSummary
    tax: TaxReport


def load_fixture(path: str | Path) -> Fixture:
    """Load a JSON fixture; refuse anything not explicitly marked synthetic."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("synthetic") is not True:
        raise ValueError("fixture must set 'synthetic': true; only synthetic fixtures are accepted")
    k = int(data["list_depth_k"])
    queries = tuple(
        query_from_pairs(
            item["query_id"],
            [(entry["doc_id"], entry["score"]) for entry in item["dense"]],
            [(entry["doc_id"], entry["score"]) for entry in item["lexical"]],
            item["qrels"],
        )
        for item in data["queries"]
    )
    for query in queries:
        validate_depth(k, query)
    return Fixture(
        name=str(data["name"]),
        synthetic=True,
        list_depth_k=k,
        metric_cutoff=int(data["metric_cutoff"]),
        queries=queries,
        note=str(data.get("note", "")),
        sweep=bool(data.get("sweep", False)),
    )


def default_configs() -> list[FusionConfig]:
    """Named methods, including Eq. (3) special cases and one named example.

    The SciFact point is a dataset-specific train-selected example from
    arXiv:2609.22770v2; it is not a default and is not a measured gain here.
    """
    return [
        FusionConfig("lexical_only", "lexical-only single list (BM42-style)", "lexical"),
        FusionConfig("dense_only", "dense/vector-only single list", "dense"),
        FusionConfig(
            "rrf_equal",
            "RRF, equal weights (in-tree FusionMethod.RRF math, offline replay)",
            "rrf",
            {"rank_constant": 2, "weights": (0.5, 0.5)},
        ),
        FusionConfig(
            "weighted_minmax",
            "weighted min-max (in-tree FusionMethod.WEIGHTED math, offline replay)",
            "weighted",
            {"lexical_weight": 0.3, "vector_weight": 0.7, "normalization": "min_max"},
        ),
        FusionConfig(
            "mix_alpha1_lambda1",
            "Eq.3 alpha=1, lambda=1 -> dense min-max",
            "rank_score",
            {"alpha": 1.0, "lam": 1.0, "kappa": 2.0},
        ),
        FusionConfig(
            "mix_alpha0_lambda1",
            "Eq.3 alpha=0, lambda=1 -> lexical min-max",
            "rank_score",
            {"alpha": 0.0, "lam": 1.0, "kappa": 2.0},
        ),
        FusionConfig(
            "mix_alpha1",
            "Eq.3 alpha=1 -> dense-only utility (lambda=0.75)",
            "rank_score",
            {"alpha": 1.0, "lam": 0.75, "kappa": 2.0},
        ),
        FusionConfig(
            "mix_alpha0",
            "Eq.3 alpha=0 -> lexical-only utility (lambda=0.75)",
            "rank_score",
            {"alpha": 0.0, "lam": 0.75, "kappa": 2.0},
        ),
        FusionConfig(
            "mix_lambda0",
            "Eq.3 lambda=0 -> weighted RRF (alpha=0.8)",
            "rank_score",
            {"alpha": 0.8, "lam": 0.0, "kappa": 2.0},
        ),
        FusionConfig(
            "mix_lambda1",
            "Eq.3 lambda=1 -> weighted min-max CombSUM (alpha=0.8)",
            "rank_score",
            {"alpha": 0.8, "lam": 1.0, "kappa": 2.0},
        ),
        FusionConfig(
            "mix_scifact_example",
            "SciFact train-selected example alpha=0.8/lambda=0.75/kappa=20 (dataset-specific)",
            "rank_score",
            {"alpha": 0.8, "lam": 0.75, "kappa": 20.0},
        ),
    ]


def sweep_configs(
    alphas: Sequence[float] = (0.0, 0.5, 0.8, 1.0),
    lambdas: Sequence[float] = (0.0, 1.0),
    kappas: Sequence[float] = (2.0, 20.0),
) -> list[FusionConfig]:
    configs = []
    for alpha in alphas:
        for lam in lambdas:
            for kappa in kappas:
                configs.append(
                    FusionConfig(
                        name=f"mix_a{alpha:g}_l{lam:g}_kap{kappa:g}",
                        label=f"Eq.3 alpha={alpha:g} lambda={lam:g} kappa={kappa:g}",
                        method="rank_score",
                        params={"alpha": alpha, "lam": lam, "kappa": kappa},
                    )
                )
    return configs


def run_config(fixture: Fixture, config: FusionConfig, k: int | None = None) -> ExperimentResult:
    depth = fixture.list_depth_k if k is None else k
    ranked: dict[str, list[str]] = {}
    for query in fixture.queries:
        scores = score_query(config.method, query, k=depth, params=config.params)
        ranked[query.query_id] = rank_by_score(scores)
    metrics = evaluate(ranked, fixture.queries, fixture.metric_cutoff)

    lexical_ranked = {
        query.query_id: rank_by_score(lexical_only(query, k=depth)) for query in fixture.queries
    }
    dense_ranked = {
        query.query_id: rank_by_score(dense_only(query, k=depth)) for query in fixture.queries
    }
    lexical_metrics = evaluate(lexical_ranked, fixture.queries, fixture.metric_cutoff)
    dense_metrics = evaluate(dense_ranked, fixture.queries, fixture.metric_cutoff)
    tax = fusion_tax(
        fixture.queries,
        {qid: row["ndcg"] for qid, row in metrics.per_query.items()},
        {qid: row["ndcg"] for qid, row in lexical_metrics.per_query.items()},
        {qid: row["ndcg"] for qid, row in dense_metrics.per_query.items()},
    )
    return ExperimentResult(config=config, metrics=metrics, tax=tax)


def format_results(fixture: Fixture, results: Sequence[ExperimentResult]) -> str:
    cutoff = fixture.metric_cutoff
    header = (
        f"{'method':<22} {'nDCG@' + str(cutoff):>8} {'MRR@' + str(cutoff):>8} "
        f"{'R@' + str(cutoff):>8} {'mean tau':>9} {'oracle tau':>10} "
        f"{'d-lex':>8} {'d-dense':>8}"
    )
    lines = [
        f"synthetic fixture: {fixture.name}  synthetic={fixture.synthetic} "
        f"list_depth_k={fixture.list_depth_k} queries={len(fixture.queries)}",
        fixture.note,
        header,
        "-" * len(header),
    ]
    for result in results:
        lines.append(
            f"{result.config.name:<22} {result.metrics.mean_ndcg:>8.4f} "
            f"{result.metrics.mean_mrr:>8.4f} {result.metrics.mean_recall:>8.4f} "
            f"{result.tax.mean_tax:>9.4f} {result.tax.oracle_tax_rate:>10.4f} "
            f"{result.tax.fused_minus_lexical:>8.4f} {result.tax.fused_minus_dense:>8.4f}"
        )
    return "\n".join(lines)


def format_sweep(
    fixture: Fixture,
    results: Sequence[ExperimentResult],
    reference: ExperimentResult,
    best_single_ndcg: float,
) -> str:
    cutoff = fixture.metric_cutoff
    header = (
        f"{'configuration':<27} {'nDCG@' + str(cutoff):>8} "
        f"{'vs RRF':>9} {'vs single':>10} {'flag':>6}"
    )
    lines = [
        f"parameter sweep on synthetic fixture: {fixture.name}",
        header,
        "-" * len(header),
    ]
    for result in results:
        delta_rrf = result.metrics.mean_ndcg - reference.metrics.mean_ndcg
        delta_single = result.metrics.mean_ndcg - best_single_ndcg
        flag = "LOSS" if delta_rrf < -1e-12 or delta_single < -1e-12 else "ok"
        lines.append(
            f"{result.config.name:<27} {result.metrics.mean_ndcg:>8.4f} "
            f"{delta_rrf:>+9.4f} {delta_single:>+10.4f} {flag:>6}"
        )
    return "\n".join(lines)
