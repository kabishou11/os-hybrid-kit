"""Deterministic bag-of-words hashing so the demo needs no embedding model."""

from __future__ import annotations

import hashlib
import math
import re
import socket
import time
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

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


def embed_documents(
    documents: Sequence[Mapping[str, Any]],
    dimension: int,
    *,
    text_fields: tuple[str, ...] = ("title", "content"),
) -> list[dict[str, Any]]:
    """Copy docs and add a hashed ``embedding`` from ``title`` + ``content``."""
    seeded: list[dict[str, Any]] = []
    for document in documents:
        body = dict(document)
        text = " ".join(str(body.get(field, "")) for field in text_fields)
        body["embedding"] = embed_text(text, dimension)
        seeded.append(body)
    return seeded


def _cluster_url_parts(url: str) -> tuple[str, int]:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = parsed.hostname or "localhost"
    if parsed.port is not None:
        return host, parsed.port
    return host, 443 if parsed.scheme == "https" else 80


def wait_for_opensearch(kit: Any, *, url: str, timeout: float) -> None:
    """Ping until OpenSearch responds, or exit with a docker compose hint.

    Port checks use a short socket timeout so a down cluster fails without
    waiting on the HTTP client timeout each loop.
    """
    host, port = _cluster_url_parts(url)
    deadline = time.time() + timeout
    last_error: Exception | None = None
    print(f"Waiting for OpenSearch at {url} (timeout {timeout:.0f}s)...")
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.5):
                pass
        except OSError as exc:
            last_error = exc
            time.sleep(1)
            continue
        try:
            if kit.client.ping():
                return
        except Exception as exc:
            last_error = exc
        time.sleep(1)
    detail = f" Last error: {last_error}" if last_error else ""
    raise SystemExit(
        f"Cannot reach OpenSearch at {url}.{detail}\n"
        "Start the demo cluster with: docker compose up -d\n"
        "Then check: curl http://localhost:9200"
    )
