from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk

from os_hybrid_kit.config import HybridConfig
from os_hybrid_kit.mapping import build_index_body
from os_hybrid_kit.pipeline import build_pipeline_body, upsert_search_pipeline
from os_hybrid_kit.query import build_hybrid_query
from os_hybrid_kit.results import SearchResult, parse_search_response


def build_opensearch_client(config: HybridConfig) -> OpenSearch:
    kwargs: dict[str, Any] = {
        "hosts": list(config.hosts),
        "use_ssl": config.use_ssl,
        "verify_certs": config.verify_certs,
        "ssl_show_warn": config.ssl_show_warn,
        "timeout": config.request_timeout,
    }
    if config.username is not None and config.password is not None:
        kwargs["http_auth"] = (config.username, config.password)
    return OpenSearch(**kwargs)


class HybridKit:
    """Primary facade: index lifecycle, pipeline upsert, bulk index, hybrid search.

    Typical flow::

        kit = HybridKit(config)
        kit.ensure_index()
        kit.upsert_pipeline()
        kit.index_documents([...])
        result = kit.hybrid_search(query, embedding)
    """

    def __init__(
        self,
        config: HybridConfig,
        *,
        client: OpenSearch | None = None,
    ) -> None:
        self.config = config
        self.client = client if client is not None else build_opensearch_client(config)

    def exists_index(self) -> bool:
        """Return True if the configured index exists."""
        return bool(self.client.indices.exists(index=self.config.index))

    def ensure_index(
        self,
        *,
        extra_properties: dict[str, Any] | None = None,
        extra_settings: dict[str, Any] | None = None,
    ) -> bool:
        """Create the index if it does not exist. Returns True when created."""
        if self.exists_index():
            return False
        body = build_index_body(
            self.config,
            extra_properties=extra_properties,
            extra_settings=extra_settings,
        )
        self.client.indices.create(index=self.config.index, body=body)
        return True

    def delete_index(self) -> bool:
        """Delete the configured index if it exists. Returns True when deleted."""
        if not self.exists_index():
            return False
        self.client.indices.delete(index=self.config.index)
        return True

    def upsert_pipeline(self) -> Any:
        """Create or replace the hybrid search pipeline from config."""
        return upsert_search_pipeline(
            self.client,
            self.config.pipeline_name,
            build_pipeline_body(self.config),
        )

    def index_documents(
        self,
        documents: Sequence[Mapping[str, Any]],
        *,
        id_field: str = "_id",
        refresh: bool = True,
    ) -> tuple[int, list[Any]]:
        """Bulk-index documents. Each doc may include ``_id`` (popped, not stored)."""
        actions: list[dict[str, Any]] = []
        for document in documents:
            payload = dict(document)
            doc_id = payload.pop(id_field, None)
            action: dict[str, Any] = {"_index": self.config.index, "_source": payload}
            if doc_id is not None:
                action["_id"] = doc_id
            actions.append(action)
        return bulk(self.client, actions, refresh=refresh)

    def hybrid_search(
        self,
        query: str,
        embedding: Sequence[float],
        *,
        size: int | None = None,
        knn_k: int | None = None,
        filter_query: Mapping[str, Any] | None = None,
        source_excludes: Sequence[str] | None = None,
        source_includes: Sequence[str] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        pipeline_name: str | None = None,
    ) -> SearchResult:
        """Run BM25 + kNN hybrid search through the configured search pipeline."""
        if len(embedding) != self.config.dimension:
            raise ValueError(
                f"embedding length {len(embedding)} does not match "
                f"config.dimension {self.config.dimension}; use the same model "
                "as at index time"
            )
        size = size if size is not None else self.config.size
        k = knn_k if knn_k is not None else max(self.config.knn_k, size)
        excludes: Sequence[str] | None = source_excludes
        if excludes is None and self.config.exclude_vector:
            excludes = [self.config.vector_field]
        body = build_hybrid_query(
            query,
            embedding,
            text_field=self.config.text_field,
            vector_field=self.config.vector_field,
            size=size,
            knn_k=k,
            filter_query=filter_query,
            source_excludes=excludes,
            source_includes=source_includes,
            extra_body=extra_body,
        )
        response = self.client.search(
            index=self.config.index,
            body=body,
            params={"search_pipeline": pipeline_name or self.config.pipeline_name},
        )
        return parse_search_response(response)
