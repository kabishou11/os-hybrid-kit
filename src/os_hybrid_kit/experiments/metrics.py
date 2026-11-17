"""Deterministic ranking metrics for fixed run lists: nDCG@k, MRR@k, recall@k."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import log2
from statistics import fmean

from os_hybrid_kit.experiments.types import FusionQuery


def rank_by_score(scores: Mapping[str, float]) -> list[str]:
    """Sort document ids by descending score, ties by descending doc id.

    The tie rule matches :func:`build_run_list` and the trec_eval-style
    convention noted in the fusion-tax paper, so RRF's frequent exact ties
    are resolved deterministically.
    """
    ordered = sorted(scores, reverse=True)
    ordered.sort(key=lambda doc_id: -scores[doc_id])
    return ordered


def dcg(gains: Sequence[float]) -> float:
    return sum(gain / log2(position + 2) for position, gain in enumerate(gains))


def ndcg_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    """nDCG@k with graded qrels; 0.0 when the query has no relevant document."""
    actual = dcg([qrels.get(doc_id, 0) for doc_id in ranked[:k]])
    ideal = dcg(sorted((grade for grade in qrels.values() if grade > 0), reverse=True)[:k])
    return actual / ideal if ideal > 0.0 else 0.0


def mrr_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    """Reciprocal rank of the first relevant document, 0.0 if none in top k."""
    for position, doc_id in enumerate(ranked[:k], start=1):
        if qrels.get(doc_id, 0) > 0:
            return 1.0 / position
    return 0.0


def recall_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    """Fraction of all relevant documents retrieved in top k; 0.0 if none exist."""
    total_relevant = sum(1 for grade in qrels.values() if grade > 0)
    if total_relevant == 0:
        return 0.0
    found = sum(1 for doc_id in ranked[:k] if qrels.get(doc_id, 0) > 0)
    return found / total_relevant


@dataclass(frozen=True)
class MetricSummary:
    cutoff: int
    per_query: Mapping[str, Mapping[str, float]]
    mean_ndcg: float
    mean_mrr: float
    mean_recall: float


def evaluate(
    ranked_by_query: Mapping[str, Sequence[str]], queries: Sequence[FusionQuery], k: int
) -> MetricSummary:
    per_query: dict[str, dict[str, float]] = {}
    for query in queries:
        ranked = ranked_by_query[query.query_id]
        per_query[query.query_id] = {
            "ndcg": ndcg_at_k(ranked, query.qrels, k),
            "mrr": mrr_at_k(ranked, query.qrels, k),
            "recall": recall_at_k(ranked, query.qrels, k),
        }
    return MetricSummary(
        cutoff=k,
        per_query=per_query,
        mean_ndcg=fmean([row["ndcg"] for row in per_query.values()]) if per_query else 0.0,
        mean_mrr=fmean([row["mrr"] for row in per_query.values()]) if per_query else 0.0,
        mean_recall=fmean([row["recall"] for row in per_query.values()]) if per_query else 0.0,
    )
