"""Shared typing helpers for the public facade."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbedFn(Protocol):
    """Map query text to a dense vector.

    The vector length must match ``HybridConfig.dimension`` (same model as
    at index time). The kit does not call an embedding model itself.
    """

    def __call__(self, text: str) -> Sequence[float]: ...
