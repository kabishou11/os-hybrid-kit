from __future__ import annotations

from os_hybrid_kit import EmbedFn


def test_embed_fn_protocol_accepts_plain_callables() -> None:
    def embed(text: str) -> list[float]:
        return [float(len(text))]

    assert isinstance(embed, EmbedFn)
    assert list(embed("ab")) == [2.0]
    assert not isinstance(object(), EmbedFn)
