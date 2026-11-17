"""Fusion math for offline fixture replay.

The functions here reimplement the same math as the in-tree
``FusionMethod.RRF`` (``rank_constant``/kappa, optional subquery weights) and
``FusionMethod.WEIGHTED`` (per-list normalization then weighted mean). This is
fixture replay over fixed scores, not a live OpenSearch pipeline call.

Paper sources, first-hand copies checked 2026-11-17 Asia/Shanghai:

* Pati & Patil, arXiv:2609.22770v2, https://arxiv.org/abs/2609.22770v2
  Eq. (1) min-max, Eq. (2) RRF kernel, Eq. (3) rank-score mix.
* Chauhan & Agarwal, arXiv:2610.30612v1, https://arxiv.org/abs/2610.30612v1
  Eq. (1) fusion tax (diagnostic only, implemented in :mod:`tax`).
"""

from __future__ import annotations

from collections.abc import Mapping
from math import sqrt

from os_hybrid_kit.experiments.types import FusionQuery, RunList, validate_depth

EPSILON = 1e-8


def min_max_scores(run: RunList, *, epsilon: float = EPSILON) -> dict[str, float]:
    """Paper Eq. (1): (s - min s) / (max s - min s + epsilon).

    ``epsilon`` is a numerical stabilizer. The paper writes ``epsilon`` but
    gives no numeric value, so 1e-8 is an explicit module constant, not a
    claimed paper parameter. With constant scores every present document
    normalizes to 0.0.
    """
    _check_positive("epsilon", epsilon)
    scores = run.score_map()
    if not scores:
        return {}
    low = min(scores.values())
    high = max(scores.values())
    denominator = high - low + epsilon
    return {doc_id: (score - low) / denominator for doc_id, score in scores.items()}


def l2_scores(run: RunList) -> dict[str, float]:
    """L2 normalization over one list (the in-tree ``l2`` option)."""
    scores = run.score_map()
    norm = sqrt(sum(value * value for value in scores.values()))
    if norm == 0.0:
        return {doc_id: 0.0 for doc_id in scores}
    return {doc_id: value / norm for doc_id, value in scores.items()}


def z_scores(run: RunList) -> dict[str, float]:
    """Population z-score over one list (the in-tree ``z_score`` option)."""
    scores = run.score_map()
    if not scores:
        return {}
    mean = sum(scores.values()) / len(scores)
    variance = sum((value - mean) ** 2 for value in scores.values()) / len(scores)
    if variance == 0.0:
        return {doc_id: 0.0 for doc_id in scores}
    deviation = sqrt(variance)
    return {doc_id: (value - mean) / deviation for doc_id, value in scores.items()}


def normalize_scores(
    run: RunList, *, normalization: str = "min_max", epsilon: float = EPSILON
) -> dict[str, float]:
    if normalization == "min_max":
        return min_max_scores(run, epsilon=epsilon)
    if normalization == "l2":
        return l2_scores(run)
    if normalization == "z_score":
        return z_scores(run)
    raise ValueError(f"unknown normalization {normalization!r}")


def _check_unit(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number in [0, 1], got {value!r}")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")


def _check_positive(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive number, got {value!r}")
    if not float(value) > 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")


def _rank_kernel(rank: int, kappa: float) -> float:
    return 1.0 / (kappa + rank)


def rrf(
    query: FusionQuery,
    *,
    k: int,
    rank_constant: float,
    weights: tuple[float, float] = (0.5, 0.5),
) -> dict[str, float]:
    """Weighted RRF: w_dense/(kappa + r_dense) + w_lex/(kappa + r_lex).

    Same math as ``FusionMethod.RRF`` with ``rank_constant=kappa``. Weights
    follow (dense, lexical); equal weights are the paper's ``alpha=0.5`` RRF.
    Documents missing from a list use ``r = k + 1``.
    """
    validate_depth(k, query)
    _check_positive("rank_constant", rank_constant)
    _check_unit("dense weight", weights[0])
    _check_unit("lexical weight", weights[1])
    return {
        doc_id: weights[0] * _rank_kernel(query.dense.rank_of(doc_id, k), rank_constant)
        + weights[1] * _rank_kernel(query.lexical.rank_of(doc_id, k), rank_constant)
        for doc_id in query.candidates()
    }


def weighted_fusion(
    query: FusionQuery,
    *,
    k: int,
    lexical_weight: float = 0.3,
    vector_weight: float = 0.7,
    normalization: str = "min_max",
    epsilon: float = EPSILON,
) -> dict[str, float]:
    """Per-list normalization then weighted mean (``FusionMethod.WEIGHTED``).

    Defaults mirror the in-tree config: min-max, lexical 0.3 / vector 0.7.
    Missing documents normalize to 0.0 in either list.
    """
    validate_depth(k, query)
    _check_unit("lexical_weight", lexical_weight)
    _check_unit("vector_weight", vector_weight)
    if abs(lexical_weight + vector_weight - 1.0) > 1e-6:
        raise ValueError("lexical_weight + vector_weight must equal 1.0")
    dense_scores = normalize_scores(query.dense, normalization=normalization, epsilon=epsilon)
    lexical_scores = normalize_scores(query.lexical, normalization=normalization, epsilon=epsilon)
    return {
        doc_id: vector_weight * dense_scores.get(doc_id, 0.0)
        + lexical_weight * lexical_scores.get(doc_id, 0.0)
        for doc_id in query.candidates()
    }


def rank_score_mix(
    query: FusionQuery,
    *,
    k: int,
    alpha: float,
    lam: float,
    kappa: float,
    epsilon: float = EPSILON,
) -> dict[str, float]:
    """Paper Eq. (3), the parameterized rank-score mix::

        u(d) = lam * (alpha*s_dense + (1-alpha)*s_lex)
             + (1-lam) * (alpha*rho(r_dense; kappa) + (1-alpha)*rho(r_lex; kappa))

    ``s`` is Eq. (1) min-max per list (missing -> 0), ``rho`` is Eq. (2), and
    a document missing from a list gets ``r = k + 1``. Special cases:
    ``alpha=1`` dense-only utility, ``alpha=0`` lexical-only, ``lambda=0``
    weighted RRF, ``lambda=1`` weighted min-max CombSUM.
    """
    validate_depth(k, query)
    _check_unit("alpha", alpha)
    _check_unit("lam", lam)
    _check_positive("kappa", kappa)
    dense_scores = min_max_scores(query.dense, epsilon=epsilon)
    lexical_scores = min_max_scores(query.lexical, epsilon=epsilon)
    utilities: dict[str, float] = {}
    for doc_id in query.candidates():
        score_part = alpha * dense_scores.get(doc_id, 0.0) + (1.0 - alpha) * lexical_scores.get(
            doc_id, 0.0
        )
        rank_part = alpha * _rank_kernel(query.dense.rank_of(doc_id, k), kappa) + (
            1.0 - alpha
        ) * _rank_kernel(query.lexical.rank_of(doc_id, k), kappa)
        utilities[doc_id] = lam * score_part + (1.0 - lam) * rank_part
    return utilities


def dense_only(query: FusionQuery, *, k: int, epsilon: float = EPSILON) -> dict[str, float]:
    validate_depth(k, query)
    return min_max_scores(query.dense, epsilon=epsilon)


def lexical_only(query: FusionQuery, *, k: int, epsilon: float = EPSILON) -> dict[str, float]:
    validate_depth(k, query)
    return min_max_scores(query.lexical, epsilon=epsilon)


def score_query(
    method: str, query: FusionQuery, *, k: int, params: Mapping[str, object]
) -> dict[str, float]:
    if method == "dense":
        return dense_only(query, k=k)
    if method == "lexical":
        return lexical_only(query, k=k)
    if method == "rrf":
        return rrf(query, k=k, **params)
    if method == "weighted":
        return weighted_fusion(query, k=k, **params)
    if method == "rank_score":
        return rank_score_mix(query, k=k, **params)
    raise ValueError(f"unknown fusion method {method!r}")
