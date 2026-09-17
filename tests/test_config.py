from __future__ import annotations

import pytest
from pydantic import ValidationError

import os_hybrid_kit
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


def test_from_env_reads_url_index_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENSEARCH_URL", "http://os:9200")
    monkeypatch.setenv("OPENSEARCH_INDEX", "docs")
    monkeypatch.setenv("OPENSEARCH_DIM", "384")
    config = HybridConfig.from_env()
    assert config.hosts == ["http://os:9200"]
    assert config.index == "docs"
    assert config.dimension == 384


def test_from_env_hosts_take_precedence_over_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENSEARCH_URL", "http://os:9200")
    monkeypatch.setenv("OPENSEARCH_HOSTS", "http://a:9200, http://b:9200")
    monkeypatch.setenv("OPENSEARCH_DIM", "8")
    config = HybridConfig.from_env()
    assert config.hosts == ["http://a:9200", "http://b:9200"]


def test_from_env_kwargs_override_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENSEARCH_DIM", "8")
    monkeypatch.setenv("OPENSEARCH_INDEX", "from-env")
    config = HybridConfig.from_env(index="override", dimension=16)
    assert config.index == "override"
    assert config.dimension == 16


def test_from_env_rejects_non_integer_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENSEARCH_DIM", "abc")
    with pytest.raises(ValueError, match="OPENSEARCH_DIM must be an integer"):
        HybridConfig.from_env()


def test_from_env_requires_dimension_from_env_or_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENSEARCH_DIM", raising=False)
    with pytest.raises(ValidationError):
        HybridConfig.from_env()


def test_version_and_public_exports() -> None:
    assert os_hybrid_kit.__version__ == "0.4.0"
    for name in os_hybrid_kit.__all__:
        assert hasattr(os_hybrid_kit, name)
