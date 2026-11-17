"""Fusion-tax diagnostic (Chauhan & Agarwal, arXiv:2610.30612v1, Eq. 1).

``tau(q) = max(NDCG_lexical(q), NDCG_dense(q)) - NDCG_fused(q)``.
Positive tax means fusion is worse than the better single list; negative means
fusion helps. This is a diagnostic on offline fixtures, not a fusion method
and not a claim about real retrieval.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean

from os_hybrid_kit.experiments.types import FusionQuery

_TOLERANCE = 1e-12


@dataclass(frozen=True)
class QueryTax:
    query_id: str
    ndcg_fused: float
    ndcg_lexical: float
    ndcg_dense: float
    tax: float
    hurts: bool


@dataclass(frozen=True)
class TaxReport:
    rows: tuple[QueryTax, ...]
    mean_tax: float
    oracle_tax_rate: float
    mean_ndcg_fused: float
    mean_ndcg_lexical: float
    mean_ndcg_dense: float
    fused_minus_lexical: float
    fused_minus_dense: float
    wins_vs_lexical: int
    wins_vs_dense: int

    @property
    def n_queries(self) -> int:
        return len(self.rows)


def fusion_tax(
    queries: Sequence[FusionQuery],
    ndcg_fused: Mapping[str, float],
    ndcg_lexical: Mapping[str, float],
    ndcg_dense: Mapping[str, float],
) -> TaxReport:
    rows = []
    for query in queries:
        fused = ndcg_fused[query.query_id]
        lexical = ndcg_lexical[query.query_id]
        dense = ndcg_dense[query.query_id]
        tax = max(lexical, dense) - fused
        rows.append(
            QueryTax(
                query_id=query.query_id,
                ndcg_fused=fused,
                ndcg_lexical=lexical,
                ndcg_dense=dense,
                tax=tax,
                hurts=tax > _TOLERANCE,
            )
        )
    fused_values = [row.ndcg_fused for row in rows]
    lexical_values = [row.ndcg_lexical for row in rows]
    dense_values = [row.ndcg_dense for row in rows]
    mean_lexical = fmean(lexical_values) if rows else 0.0
    mean_dense = fmean(dense_values) if rows else 0.0
    mean_fused = fmean(fused_values) if rows else 0.0
    return TaxReport(
        rows=tuple(rows),
        mean_tax=fmean([row.tax for row in rows]) if rows else 0.0,
        oracle_tax_rate=fmean([float(row.hurts) for row in rows]) if rows else 0.0,
        mean_ndcg_fused=mean_fused,
        mean_ndcg_lexical=mean_lexical,
        mean_ndcg_dense=mean_dense,
        fused_minus_lexical=mean_fused - mean_lexical,
        fused_minus_dense=mean_fused - mean_dense,
        wins_vs_lexical=sum(row.ndcg_fused > row.ndcg_lexical + _TOLERANCE for row in rows),
        wins_vs_dense=sum(row.ndcg_fused > row.ndcg_dense + _TOLERANCE for row in rows),
    )
