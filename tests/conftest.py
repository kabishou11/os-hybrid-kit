from __future__ import annotations

import pytest

from os_hybrid_kit import HybridConfig


@pytest.fixture
def config() -> HybridConfig:
    return HybridConfig(
        hosts=["http://localhost:9200"],
        index="hybrid-test",
        dimension=4,
        text_field="content",
        vector_field="embedding",
    )
