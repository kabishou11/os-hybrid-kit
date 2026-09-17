from __future__ import annotations

import pytest

from os_hybrid_kit import (
    FusionMethod,
    HybridConfig,
    build_pipeline_body,
    build_rrf_pipeline_body,
    build_weighted_pipeline_body,
)


def test_rrf_pipeline_uses_score_ranker_processor():
    body = build_rrf_pipeline_body(rank_constant=60)
    processors = body["phase_results_processors"]
    assert len(processors) == 1
    combination = processors[0]["score-ranker-processor"]["combination"]
    assert combination["technique"] == "rrf"
    assert combination["rank_constant"] == 60
    assert "parameters" not in combination


def test_rrf_pipeline_optional_weights():
    body = build_rrf_pipeline_body(rank_constant=40, weights=[0.3, 0.7])
    combination = body["phase_results_processors"][0]["score-ranker-processor"][
        "combination"
    ]
    assert combination["rank_constant"] == 40
    assert combination["parameters"]["weights"] == [0.3, 0.7]


def test_weighted_pipeline_uses_normalization_processor():
    body = build_weighted_pipeline_body(lexical_weight=0.3, vector_weight=0.7)
    processor = body["phase_results_processors"][0]["normalization-processor"]
    assert processor["normalization"]["technique"] == "min_max"
    assert processor["combination"]["technique"] == "arithmetic_mean"
    assert processor["combination"]["parameters"]["weights"] == [0.3, 0.7]


def test_weighted_pipeline_rejects_weights_that_do_not_sum_to_one():
    with pytest.raises(ValueError, match="must equal 1.0"):
        build_weighted_pipeline_body(lexical_weight=0.4, vector_weight=0.4)


def test_build_pipeline_body_dispatches_on_fusion():
    rrf = HybridConfig(dimension=2, fusion=FusionMethod.RRF, rank_constant=20)
    weighted = HybridConfig(
        dimension=2,
        fusion=FusionMethod.WEIGHTED,
        lexical_weight=0.4,
        vector_weight=0.6,
        normalization="l2",
        combination="harmonic_mean",
    )
    rrf_body = build_pipeline_body(rrf)
    weighted_body = build_pipeline_body(weighted)
    assert "score-ranker-processor" in rrf_body["phase_results_processors"][0]
    norm = weighted_body["phase_results_processors"][0]["normalization-processor"]
    assert norm["normalization"]["technique"] == "l2"
    assert norm["combination"]["technique"] == "harmonic_mean"
    assert norm["combination"]["parameters"]["weights"] == [0.4, 0.6]


def test_rrf_weights_from_config():
    config = HybridConfig(dimension=2, fusion=FusionMethod.RRF, rrf_weights=(0.2, 0.8))
    body = build_pipeline_body(config)
    combination = body["phase_results_processors"][0]["score-ranker-processor"][
        "combination"
    ]
    assert combination["parameters"]["weights"] == [0.2, 0.8]
