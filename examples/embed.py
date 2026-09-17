"""Deterministic bag-of-words hashing so the demo needs no embedding model."""

from __future__ import annotations

import hashlib
import math
import re

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def embed_text(text: str, dimension: int) -> list[float]:
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    vector = [0.0] * dimension
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:8], "big") % dimension
        vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]
