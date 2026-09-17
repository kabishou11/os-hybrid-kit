from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from os_hybrid_kit.client import HybridKit
from os_hybrid_kit.results import SearchResult

EmbedFn = Callable[[str], Sequence[float]]


class RetrievalSetting(BaseModel):
    model_config = ConfigDict(extra="ignore")

    top_k: int = Field(default=3, ge=1)
    score_threshold: float = Field(default=0.0)


class RetrievalRequest(BaseModel):
    """Dify External Knowledge API request body."""

    model_config = ConfigDict(extra="ignore")

    knowledge_id: str
    query: str
    retrieval_setting: RetrievalSetting
    metadata_condition: dict[str, Any] | None = None


class KnowledgeRecord(BaseModel):
    content: str
    score: float
    title: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    records: list[KnowledgeRecord]


def hits_to_records(
    result: SearchResult,
    *,
    text_field: str,
    vector_field: str,
    score_threshold: float = 0.0,
    title_field: str = "title",
) -> list[KnowledgeRecord]:
    records: list[KnowledgeRecord] = []
    skip = {text_field, vector_field, title_field}
    for hit in result.hits:
        if hit.score < score_threshold:
            continue
        source = hit.source
        metadata = {
            key: value
            for key, value in source.items()
            if key not in skip and not key.startswith("_")
        }
        metadata.setdefault("document_id", hit.id)
        title = source.get(title_field)
        records.append(
            KnowledgeRecord(
                content=str(source.get(text_field, "")),
                score=hit.score,
                title=str(title) if title else hit.id,
                metadata=metadata,
            )
        )
    return records


def retrieve(
    kit: HybridKit,
    request: RetrievalRequest | Mapping[str, Any],
    embed_fn: EmbedFn,
    *,
    title_field: str = "title",
) -> RetrievalResponse:
    """Run hybrid search and shape hits as a Dify ``/retrieval`` response.

    Dify only sends text. ``embed_fn`` must produce a vector that matches
    ``kit.config.dimension``. RRF scores are typically much smaller than 1.0;
    set Dify's score threshold to ``0`` unless you use weighted fusion.
    """
    payload = (
        request
        if isinstance(request, RetrievalRequest)
        else RetrievalRequest.model_validate(request)
    )
    embedding = embed_fn(payload.query)
    result = kit.hybrid_search(
        payload.query,
        embedding,
        size=payload.retrieval_setting.top_k,
    )
    records = hits_to_records(
        result,
        text_field=kit.config.text_field,
        vector_field=kit.config.vector_field,
        score_threshold=payload.retrieval_setting.score_threshold,
        title_field=title_field,
    )
    return RetrievalResponse(records=records)
