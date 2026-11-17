from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from math import log2

import pytest

from os_hybrid_kit.experiments import (
    EPSILON,
    build_run_list,
    dense_only,
    fusion_tax,
    lexical_only,
    load_fixture,
    min_max_scores,
    rank_score_mix,
    rrf,
    weighted_fusion,
)
from os_hybrid_kit.experiments.cli import DEFAULT_FIXTURES, main
from os_hybrid_kit.experiments.experiment import default_configs, run_config, sweep_configs
from os_hybrid_kit.experiments.fusion import l2_scores, normalize_scores, z_scores
from os_hybrid_kit.experiments.metrics import (
    evaluate,
    mrr_at_k,
    ndcg_at_k,
    rank_by_score,
    recall_at_k,
)
from os_hybrid_kit.experiments.types import FusionQuery, query_from_pairs, validate_depth


def small_query() -> FusionQuery:
    return query_from_pairs(
        "q",
        [("a", 3.0), ("b", 2.0), ("c", 1.0)],
        [("x", 5.0), ("y", 4.0), ("z", 1.0)],
        {"a": 1},
    )


def config(name: str):
    return next(item for item in default_configs() if item.name == name)


def test_min_max_equation_and_constant_score_epsilon_path() -> None:
    run = build_run_list([("a", 3.0), ("b", 2.0), ("c", 1.0)])
    assert min_max_scores(run) == pytest.approx({"a": 1.0, "b": 0.5, "c": 0.0})
    constant = build_run_list([("a", 7.0), ("b", 7.0)])
    assert min_max_scores(constant) == pytest.approx({"a": 0.0, "b": 0.0})


def test_missing_doc_gets_zero_score_and_rank_k_plus_one() -> None:
    query = query_from_pairs(
        "q", [("a", 1.0), ("b", 0.0)], [("c", 1.0), ("b", 0.5), ("d", 0.2)], {}
    )
    assert query.dense.rank_of("c", 4) == 5
    assert query.dense.score_map().get("c", 0.0) == 0.0
    utilities = rank_score_mix(query, k=4, alpha=1.0, lam=0.5, kappa=2.0)
    assert utilities["c"] == pytest.approx(0.5 / 7.0)


def test_duplicate_policy_keeps_best_score_and_ties_break_by_descending_doc_id() -> None:
    deduped = build_run_list([("a", 1.0), ("a", 5.0), ("b", 3.0), ("b", 3.0)])
    assert [(item.doc_id, item.score) for item in deduped] == [("a", 5.0), ("b", 3.0)]
    tied = build_run_list([("p", 1.0), ("q", 1.0), ("p", 1.0)])
    assert [item.doc_id for item in tied] == ["q", "p"]


@pytest.mark.parametrize("bad_k", [-1, 0])
def test_illegal_nonpositive_k(bad_k: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        validate_depth(bad_k, small_query())


def test_illegal_k_above_candidate_count_and_list_longer_than_k() -> None:
    query = small_query()
    with pytest.raises(ValueError, match="exceeds candidate count"):
        validate_depth(7, query)
    with pytest.raises(ValueError, match="smaller than an input list"):
        validate_depth(2, query)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": -0.1, "lam": 0.5, "kappa": 2.0},
        {"alpha": 1.1, "lam": 0.5, "kappa": 2.0},
        {"alpha": 0.5, "lam": -0.1, "kappa": 2.0},
        {"alpha": 0.5, "lam": 1.1, "kappa": 2.0},
        {"alpha": 0.5, "lam": 0.5, "kappa": 0.0},
        {"alpha": 0.5, "lam": 0.5, "kappa": -2.0},
        {"alpha": 0.5, "lam": 0.5, "kappa": 2.0, "epsilon": 0.0},
    ],
)
def test_illegal_fusion_parameters(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        rank_score_mix(small_query(), k=4, **kwargs)


def test_weighted_fusion_rejects_weights_that_do_not_sum_to_one() -> None:
    with pytest.raises(ValueError, match="must equal 1.0"):
        weighted_fusion(small_query(), k=4, lexical_weight=0.4, vector_weight=0.7)


def test_empty_lists_and_no_relevant_documents_are_well_defined() -> None:
    empty = query_from_pairs("q", [], [], {"r": 1})
    assert rank_score_mix(empty, k=1, alpha=0.5, lam=0.5, kappa=2.0) == {}
    summary = evaluate({"q": []}, [empty], 5)
    assert (summary.mean_ndcg, summary.mean_mrr, summary.mean_recall) == (0.0, 0.0, 0.0)
    no_rel = query_from_pairs("q2", [("d", 1.0)], [], {})
    ranked = rank_by_score(dense_only(no_rel, k=1))
    assert ndcg_at_k(ranked, no_rel.qrels, 5) == 0.0
    assert mrr_at_k(ranked, no_rel.qrels, 5) == 0.0
    assert recall_at_k(ranked, no_rel.qrels, 5) == 0.0


def test_equation_three_special_cases_reduce_to_baselines() -> None:
    query = small_query()
    dense = dense_only(query, k=4)
    lexical = lexical_only(query, k=4)
    dense_mix = rank_score_mix(query, k=4, alpha=1.0, lam=1.0, kappa=2.0)
    lexical_mix = rank_score_mix(query, k=4, alpha=0.0, lam=1.0, kappa=2.0)
    assert {key: dense_mix[key] for key in dense} == pytest.approx(dense)
    assert {key: lexical_mix[key] for key in lexical} == pytest.approx(lexical)
    assert rank_score_mix(query, k=4, alpha=0.8, lam=0.0, kappa=2.0) == pytest.approx(
        rrf(query, k=4, rank_constant=2, weights=(0.8, 0.2))
    )
    assert rank_score_mix(query, k=4, alpha=0.3, lam=1.0, kappa=2.0) == pytest.approx(
        weighted_fusion(query, k=4, lexical_weight=0.7, vector_weight=0.3)
    )


def test_alpha_one_ignores_lexical_list_when_lambda_is_below_one() -> None:
    dense = [("a", 3.0), ("b", 1.0), ("z", 2.0)]
    first = query_from_pairs("q", dense, [("x", 9.0)], {})
    second = query_from_pairs("q", dense, [("y", 0.1)], {})
    left = rank_score_mix(first, k=4, alpha=1.0, lam=0.75, kappa=2.0)
    right = rank_score_mix(second, k=4, alpha=1.0, lam=0.75, kappa=2.0)
    for key in ("a", "b", "z"):
        assert left[key] == pytest.approx(right[key])


def test_l2_and_z_score_normalizations() -> None:
    run = build_run_list([("a", 3.0), ("b", 4.0)])
    assert l2_scores(run) == pytest.approx({"a": 3.0 / 5.0, "b": 4.0 / 5.0})
    assert z_scores(run) == pytest.approx({"a": -1.0, "b": 1.0})
    assert z_scores(build_run_list([("a", 5.0), ("b", 5.0)])) == pytest.approx(
        {"a": 0.0, "b": 0.0}
    )
    assert normalize_scores(run, normalization="min_max") == pytest.approx(
        min_max_scores(run)
    )
    with pytest.raises(ValueError, match="unknown normalization"):
        normalize_scores(run, normalization="bogus")


def test_metrics_hand_computed() -> None:
    ranked = ["b", "a", "c"]
    qrels = {"a": 2, "b": 0, "c": 1}
    expected_ndcg = (2.0 / log2(3) + 1.0 / log2(4)) / (2.0 / log2(2) + 1.0 / log2(3))
    assert ndcg_at_k(ranked, qrels, 3) == pytest.approx(expected_ndcg)
    assert mrr_at_k(ranked, qrels, 3) == 0.5
    assert recall_at_k(ranked, qrels, 3) == 1.0
    assert rank_by_score({"a": 1.0, "b": 1.0}) == ["b", "a"]


def test_fusion_tax_sign_and_oracle_rate() -> None:
    queries = (query_from_pairs("q1", [], [], {}), query_from_pairs("q2", [], [], {}))
    report = fusion_tax(
        queries,
        {"q1": 0.5, "q2": 0.9},
        {"q1": 0.8, "q2": 0.7},
        {"q1": 0.6, "q2": 0.8},
    )
    assert report.rows[0].tax == pytest.approx(0.3)
    assert report.rows[0].hurts is True
    assert report.rows[1].tax == pytest.approx(-0.1)
    assert report.rows[1].hurts is False
    assert report.oracle_tax_rate == pytest.approx(0.5)
    assert report.mean_tax == pytest.approx(0.1)
    assert report.n_queries == 2


def test_synthetic_win_fixture_lifts_designed_relevant_doc_over_equal_rrf() -> None:
    fixture = load_fixture(DEFAULT_FIXTURES[0])
    assert fixture.synthetic is True
    query = fixture.queries[0]
    rrf_result = run_config(fixture, config("rrf_equal"))
    mix_result = run_config(fixture, config("mix_scifact_example"))
    rrf_ranking = rank_by_score(
        rrf(query, k=fixture.list_depth_k, rank_constant=2, weights=(0.5, 0.5))
    )
    mix_ranking = rank_by_score(
        rank_score_mix(query, k=fixture.list_depth_k, alpha=0.8, lam=0.75, kappa=20.0)
    )
    assert rrf_ranking.index("X") == 2
    assert mix_ranking.index("X") == 0
    assert mix_result.metrics.per_query["q1"]["ndcg"] > rrf_result.metrics.per_query["q1"]["ndcg"]


def test_synthetic_loss_fixture_and_grid_show_fusion_can_lose() -> None:
    fixture = load_fixture(DEFAULT_FIXTURES[1])
    assert fixture.synthetic is True
    mix = run_config(fixture, config("mix_scifact_example"))
    baseline_rrf = run_config(fixture, config("rrf_equal"))
    lexical = run_config(fixture, config("lexical_only"))
    assert mix.metrics.mean_ndcg < baseline_rrf.metrics.mean_ndcg
    assert mix.metrics.mean_ndcg < lexical.metrics.mean_ndcg
    assert mix.tax.oracle_tax_rate == pytest.approx(0.5)
    assert mix.tax.mean_tax > 0.0
    best_single = max(
        lexical.metrics.mean_ndcg, run_config(fixture, config("dense_only")).metrics.mean_ndcg
    )
    swept = [run_config(fixture, item) for item in sweep_configs()]
    assert any(
        item.metrics.mean_ndcg < baseline_rrf.metrics.mean_ndcg
        and item.metrics.mean_ndcg < best_single
        for item in swept
    )


def test_loader_rejects_fixture_not_marked_synthetic(tmp_path) -> None:
    payload: Mapping[str, object] = {
        "name": "unmarked",
        "synthetic": False,
        "list_depth_k": 1,
        "metric_cutoff": 1,
        "queries": [],
    }
    path = tmp_path / "not_synthetic.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="synthetic"):
        load_fixture(path)


def test_cli_prints_real_numbers_and_synthetic_disclaimer(capsys) -> None:
    assert main([]) == 0
    text = capsys.readouterr().out
    assert "nDCG@3" in text
    assert "synthetic_win" in text
    assert "synthetic_loss" in text
    assert "LOSS" in text
    assert "no real retrieval gain is claimed" in text
    assert main(["--fixture", str(DEFAULT_FIXTURES[0]), "--no-sweep"]) == 0
    assert "parameter sweep" not in capsys.readouterr().out


def test_config_names_are_unique() -> None:
    names: Sequence[str] = [item.name for item in default_configs()]
    assert len(names) == len(set(names))


def test_eq2_rank_kernel_endpoint_values_and_missing_rank() -> None:
    # Eq. (2): rho(r; kappa) = 1/(kappa + r), ranks start at 1.
    # Dense-only RRF (weights (1, 0)) isolates one list: a/b are dense ranks
    # 1/2 -> 1/21 and 1/22; "e" is absent from dense so rank_of yields
    # k + 1 = 5 -> rho = 1/(20 + 5) = 1/25.
    query = query_from_pairs(
        "q",
        [("a", 4.0), ("b", 3.0), ("c", 2.0), ("d", 1.0)],
        [("e", 1.0)],
        {},
    )
    scores = rrf(query, k=4, rank_constant=20, weights=(1.0, 0.0))
    assert scores["a"] == pytest.approx(1.0 / 21.0)
    assert scores["b"] == pytest.approx(1.0 / 22.0)
    assert scores["e"] == pytest.approx(1.0 / 25.0)


def test_eq1_min_max_two_score_list_pins_epsilon() -> None:
    # Eq. (1): s~(d) = (s - min)/(max - min + eps); with {0, 10} the
    # denominator is 10 + 1e-8, so the top score is strictly below 1.
    assert EPSILON == 1e-8
    scores = min_max_scores(build_run_list([("low", 0.0), ("high", 10.0)]))
    assert scores["low"] == 0.0
    assert scores["high"] == 10.0 / (10.0 + EPSILON)
    assert scores["high"] < 1.0


def test_eq3_rank_score_mix_hand_computed_utilities() -> None:
    # Inputs (k=3 -> a missing doc has rank k+1=4; kappa=2; alpha=0.8;
    # lambda=0.75):
    #   dense   : a=3 (rank 1), b=1 (rank 2), c missing
    #   lexical : c=5 (rank 1), b=2 (rank 2), a missing
    # Eq. (1): s~dense(a)=2/(2+eps), s~dense(b)=0,
    #          s~lex(c)=3/(3+eps), s~lex(b)=0; missing docs get 0.
    # Eq. (2): rho(1;2)=1/3, rho(2;2)=1/4, rho(4;2)=1/6.
    # Eq. (3): u = 0.75*(0.8*s~d + 0.2*s~s) + 0.25*(0.8*rho(rd) + 0.2*rho(rs)).
    #   u(a) = 0.75*(0.8*2/(2+eps)) + 0.25*(0.8/3 + 0.2/6) ~= 0.675
    #   u(b) = 0.25*(0.8/4 + 0.2/4) = 0.0625
    #   u(c) = 0.75*(0.2*3/(3+eps)) + 0.25*(0.8/6 + 0.2/3) ~= 0.2
    query = query_from_pairs(
        "q",
        [("a", 3.0), ("b", 1.0)],
        [("c", 5.0), ("b", 2.0)],
        {},
    )
    utilities = rank_score_mix(query, k=3, alpha=0.8, lam=0.75, kappa=2.0)
    expected = {
        "a": 0.75 * (0.8 * (2.0 / (2.0 + EPSILON))) + 0.25 * (0.8 / 3.0 + 0.2 / 6.0),
        "b": 0.25 * (0.8 / 4.0 + 0.2 / 4.0),
        "c": 0.75 * (0.2 * (3.0 / (3.0 + EPSILON))) + 0.25 * (0.8 / 6.0 + 0.2 / 3.0),
    }
    assert utilities == pytest.approx(expected, rel=1e-12)
    assert utilities["b"] == pytest.approx(0.0625)


@pytest.mark.parametrize("alpha", [0.0, 0.8, 1.0])
def test_eq3_lambda_endpoints_equal_weighted_baselines(alpha: float) -> None:
    # lambda=0 drops the score term, leaving alpha*rho_d + (1-alpha)*rho_s,
    # i.e. weighted RRF with weights (alpha, 1-alpha). lambda=1 drops the
    # rank term, leaving weighted min-max CombSUM with vector_weight=alpha.
    query = query_from_pairs(
        "endpoints",
        [("a", 5.0), ("b", 3.0), ("c", 1.0)],
        [("b", 9.0), ("d", 4.0)],
        {},
    )
    assert rank_score_mix(query, k=4, alpha=alpha, lam=0.0, kappa=2.0) == pytest.approx(
        rrf(query, k=4, rank_constant=2, weights=(alpha, 1.0 - alpha))
    )
    assert rank_score_mix(query, k=4, alpha=alpha, lam=1.0, kappa=2.0) == pytest.approx(
        weighted_fusion(query, k=4, lexical_weight=1.0 - alpha, vector_weight=alpha)
    )


def test_graded_ndcg_and_mrr_hand_computed() -> None:
    # Gains 2 ("p") and 1 ("q"); discount is 1/log2(rank+1).
    # Ranking ["z", "q", "p"]: DCG = 1/log2(3) + 2/log2(4) = 1/log2(3) + 1.
    # Ideal DCG = 2/log2(2) + 1/log2(3) = 2 + 1/log2(3).
    qrels = {"p": 2, "q": 1, "z": 0}
    ranked = ["z", "q", "p"]
    expected = (1.0 / log2(3) + 2.0 / log2(4)) / (2.0 / log2(2) + 1.0 / log2(3))
    assert ndcg_at_k(ranked, qrels, 3) == pytest.approx(expected)
    assert mrr_at_k(ranked, qrels, 3) == 0.5
    assert ndcg_at_k(["p", "q", "z"], qrels, 3) == pytest.approx(1.0)


def test_fusion_tax_zero_and_positive_from_real_rankings() -> None:
    match = query_from_pairs(
        "match", [("a", 3.0), ("b", 1.0)], [("b", 2.0), ("a", 1.0)], {"a": 1}
    )
    hurt = query_from_pairs(
        "hurt", [("bad", 100.0), ("a", 1.0)], [("a", 2.0), ("bad", 1.0)], {"a": 1}
    )
    queries = (match, hurt)
    fused: dict[str, float] = {}
    lexical: dict[str, float] = {}
    dense: dict[str, float] = {}
    for query in queries:
        fused[query.query_id] = ndcg_at_k(
            rank_by_score(rank_score_mix(query, k=2, alpha=0.8, lam=0.75, kappa=2.0)),
            query.qrels,
            1,
        )
        lexical[query.query_id] = ndcg_at_k(
            rank_by_score(lexical_only(query, k=2)), query.qrels, 1
        )
        dense[query.query_id] = ndcg_at_k(rank_by_score(dense_only(query, k=2)), query.qrels, 1)
    report = fusion_tax(queries, fused, lexical, dense)
    # "match": dense ranks "a" first and fusion reproduces it -> max(1, 0) - 1 = 0.
    assert report.rows[0].tax == pytest.approx(0.0)
    assert report.rows[0].hurts is False
    # "hurt": dense outlier "bad" (100.0) compresses min-max; fusion ranks it
    # first -> max(0, 1) - 0 = 1. Synthetic only, not a retrieval claim.
    assert report.rows[1].tax == pytest.approx(1.0)
    assert report.rows[1].hurts is True
    assert report.oracle_tax_rate == pytest.approx(0.5)
