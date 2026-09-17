from __future__ import annotations

import pytest
from pydantic import ValidationError

from os_hybrid_kit import FusionMethod, HybridConfig


def test_weighted_fusion_requires_weights_to_sum_to_one():
    with pytest.raises(ValidationError):
        HybridConfig(
            dimension=2,
            fusion=FusionMethod.WEIGHTED,
            lexical_weight=0.2,
            vector_weight=0.2,
        )


def test_rrf_does_not_require_lexical_vector_weights_to_sum():
    config = HybridConfig(
        dimension=2,
        fusion=FusionMethod.RRF,
        lexical_weight=0.2,
        vector_weight=0.2,
    )
    assert config.fusion is FusionMethod.RRF


def test_rrf_weights_must_sum_to_one():
    with pytest.raises(ValidationError):
        HybridConfig(dimension=2, rrf_weights=(0.1, 0.2))


def test_dimension_must_be_positive():
    with pytest.raises(ValidationError):
        HybridConfig(dimension=0)
