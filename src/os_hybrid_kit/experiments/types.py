"""Offline replay types for fixed run lists.

No OpenSearch call, network access, GPU, or live index is involved. A
:class:`FusionQuery` holds two already-retrieved lists (dense/vector and
lexical/sparse) plus graded relevance judgments, so fusion math can be
tested deterministically on fixtures.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ScoredDoc:
    doc_id: str
    score: float


@dataclass(frozen=True)
class RunList:
    """One channel's fixed list for one query, in rank order (rank 1 first).

    Construct via :func:`build_run_list` to enforce the documented ordering,
    tie, and duplicate policies. Directly constructed instances are assumed
    already normalized.
    """

    docs: tuple[ScoredDoc, ...]

    def __len__(self) -> int:
        return len(self.docs)

    def __iter__(self):
        return iter(self.docs)

    def score_map(self) -> dict[str, float]:
        return {item.doc_id: item.score for item in self.docs}

    def rank_of(self, doc_id: str, k: int) -> int:
        for position, item in enumerate(self.docs, start=1):
            if item.doc_id == doc_id:
                return position
        return k + 1


@dataclass(frozen=True)
class FusionQuery:
    """One query's dense list, lexical list, and graded qrels.

    ``qrels`` maps document id to a non-negative relevance grade. Judged
    documents absent from both lists are treated as unretrieved (recall
    misses); documents in a list with no judgment have grade 0.
    """

    query_id: str
    dense: RunList
    lexical: RunList
    qrels: Mapping[str, int]

    def candidates(self) -> list[str]:
        return sorted(set(self.dense.score_map()) | set(self.lexical.score_map()), reverse=True)


@dataclass(frozen=True)
class Fixture:
    """A loader-validated collection of fixed queries.

    ``synthetic`` must be true: fixtures are for mathematical correctness and
    boundary behavior only and carry no claim about real retrieval quality.
    """

    name: str
    synthetic: bool
    list_depth_k: int
    metric_cutoff: int
    queries: tuple[FusionQuery, ...]
    note: str = ""
    sweep: bool = False


def build_run_list(pairs: Iterable[tuple[str, float]]) -> RunList:
    """Normalize raw (doc_id, score) pairs into a ranked :class:`RunList`.

    Policies (deterministic, documented):

    * non-finite scores are rejected;
    * exact score ties are broken by document id in descending lexicographic
      order, the trec_eval-style convention noted by the fusion-tax paper;
    * duplicate doc ids keep the highest-scoring occurrence, with the same
      descending-doc-id tie rule.
    """
    items: list[ScoredDoc] = []
    for doc_id, raw_score in pairs:
        score = float(raw_score)
        if not isfinite(score):
            raise ValueError(f"non-finite score for doc {doc_id!r}")
        items.append(ScoredDoc(doc_id=str(doc_id), score=score))
    items.sort(key=lambda item: item.doc_id, reverse=True)
    items.sort(key=lambda item: item.score, reverse=True)

    seen: set[str] = set()
    unique: list[ScoredDoc] = []
    for item in items:
        if item.doc_id in seen:
            continue
        seen.add(item.doc_id)
        unique.append(item)
    return RunList(tuple(unique))


def validate_depth(k: int, query: FusionQuery) -> None:
    """Validate list depth ``k`` for one query.

    ``k`` is the retrieval depth of each input list (the paper's k=100), not a
    hidden default. It must be >= 1, no list may be longer than ``k``, and it
    may not exceed the number of distinct candidate documents (the ``k > n``
    case). The last check is skipped for an empty candidate universe so that
    empty lists remain representable.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"list depth k must be a positive integer, got {k!r}")
    if len(query.dense) > k or len(query.lexical) > k:
        raise ValueError(f"list depth k={k} is smaller than an input list")
    n_candidates = len(query.candidates())
    if n_candidates and k > n_candidates:
        raise ValueError(f"list depth k={k} exceeds candidate count n={n_candidates}")


def validate_qrels(qrels: Mapping[str, int]) -> dict[str, int]:
    clean: dict[str, int] = {}
    for doc_id, raw_grade in qrels.items():
        grade = int(raw_grade)
        if grade < 0:
            raise ValueError(f"qrels grade for {doc_id!r} must be >= 0")
        clean[str(doc_id)] = grade
    return clean


def query_from_pairs(
    query_id: str,
    dense: Sequence[tuple[str, float]],
    lexical: Sequence[tuple[str, float]],
    qrels: Mapping[str, int],
) -> FusionQuery:
    return FusionQuery(
        query_id=str(query_id),
        dense=build_run_list(dense),
        lexical=build_run_list(lexical),
        qrels=validate_qrels(qrels),
    )
