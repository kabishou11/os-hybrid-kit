from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Hit:
    id: str
    score: float
    source: dict[str, Any]
    index: str
    raw: dict[str, Any] = field(repr=False, compare=False)


@dataclass(frozen=True)
class SearchResult:
    hits: list[Hit]
    total: int
    max_score: float | None
    raw: dict[str, Any] = field(repr=False, compare=False)

    def texts(self, field: str) -> list[str]:
        return [str(hit.source.get(field, "")) for hit in self.hits]


def parse_total(payload: Any) -> int:
    if payload is None:
        return 0
    if isinstance(payload, int):
        return payload
    if isinstance(payload, dict) and "value" in payload:
        return int(payload["value"])
    return 0


def parse_search_response(response: Mapping[str, Any]) -> SearchResult:
    hits_block = response.get("hits") or {}
    raw_hits = hits_block.get("hits") or []
    hits = [
        Hit(
            id=str(item.get("_id", "")),
            score=float(item.get("_score") or 0.0),
            source=dict(item.get("_source") or {}),
            index=str(item.get("_index", "")),
            raw=dict(item),
        )
        for item in raw_hits
    ]
    max_score = hits_block.get("max_score")
    return SearchResult(
        hits=hits,
        total=parse_total(hits_block.get("total")),
        max_score=None if max_score is None else float(max_score),
        raw=dict(response),
    )
