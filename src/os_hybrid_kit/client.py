from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from opensearchpy import OpenSearch
from opensearchpy.exceptions import NotFoundError
from opensearchpy.helpers import bulk

from os_hybrid_kit.config import HybridConfig
from os_hybrid_kit.mapping import build_index_body
from os_hybrid_kit.pipeline import build_pipeline_body, upsert_search_pipeline
from os_hybrid_kit.query import build_hybrid_query, build_knn_query, build_lexical_query
from os_hybrid_kit.results import SearchResult, parse_search_response


def _require_nonempty(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


class AliasNotFoundError(ValueError):
    """Raised when ``get_alias`` or ``delete_alias`` targets a missing alias."""

    def __init__(self, alias: str) -> None:
        self.alias = alias
        super().__init__(f"alias {alias!r} was not found")


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

    ``lexical_search`` / ``knn_search`` hit the same index without a pipeline.
    ``config.index`` may be an alias; ``put_alias`` / ``swap_alias`` /
    ``with_index`` cover a thin cutover path (not full index management).
    """

    def __init__(
        self,
        config: HybridConfig,
        *,
        client: OpenSearch | None = None,
    ) -> None:
        self.config = config
        self.client = client if client is not None else build_opensearch_client(config)

    def __repr__(self) -> str:
        return (
            f"HybridKit(index={self.config.index!r}, dimension={self.config.dimension}, "
            f"fusion={self.config.fusion.value!r}, pipeline_name={self.config.pipeline_name!r})"
        )

    def with_index(self, name: str) -> HybridKit:
        """Return a kit that targets ``name`` (index or alias), sharing this client.

        Search and index methods use ``config.index`` as the OpenSearch target, so
        passing an alias here is the zero-downtime read path after ``put_alias``
        or ``swap_alias``.
        """
        _require_nonempty(name, "index")
        return HybridKit(self.config.model_copy(update={"index": name}), client=self.client)

    def _target_index(self, index: str | None) -> str:
        name = self.config.index if index is None else index
        return _require_nonempty(name, "index")

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

    def put_alias(self, alias: str, *, index: str | None = None) -> Any:
        """Point ``alias`` at ``index`` (default ``config.index``).

        Does not remove other targets; use ``swap_alias`` for cutover.
        """
        _require_nonempty(alias, "alias")
        return self.client.indices.put_alias(index=self._target_index(index), name=alias)

    def delete_alias(self, alias: str, *, index: str | None = None) -> Any:
        """Remove ``alias`` from ``index`` (default ``config.index``).

        Raises ``AliasNotFoundError`` if the alias is not on that index.
        """
        _require_nonempty(alias, "alias")
        try:
            return self.client.indices.delete_alias(
                index=self._target_index(index), name=alias
            )
        except NotFoundError as exc:
            raise AliasNotFoundError(alias) from exc

    def get_alias(self, alias: str) -> dict[str, Any]:
        """Resolve ``alias`` to OpenSearch's ``{index: {"aliases": {alias: ...}}}`` mapping.

        Raises ``AliasNotFoundError`` if the alias does not exist.
        """
        _require_nonempty(alias, "alias")
        try:
            return dict(self.client.indices.get_alias(name=alias))
        except NotFoundError as exc:
            raise AliasNotFoundError(alias) from exc

    def swap_alias(
        self,
        alias: str,
        new_index: str,
        *,
        old_index: str | None = None,
    ) -> Any:
        """Point ``alias`` at ``new_index``, removing the previous target in one ``_aliases`` call.

        When ``old_index`` is set, this is a single ``update_aliases`` request
        (remove old + add new). When omitted, current targets are looked up
        first, then updated — still one write, but not a single round-trip.
        """
        _require_nonempty(alias, "alias")
        _require_nonempty(new_index, "index")
        if old_index is not None:
            _require_nonempty(old_index, "index")
        actions: list[dict[str, Any]] = []
        if old_index is not None:
            if old_index != new_index:
                actions.append({"remove": {"index": old_index, "alias": alias}})
        else:
            try:
                mapping = self.get_alias(alias)
            except AliasNotFoundError:
                mapping = {}
            for name in mapping:
                if name != new_index:
                    actions.append({"remove": {"index": name, "alias": alias}})
        actions.append({"add": {"index": new_index, "alias": alias}})
        return self.client.indices.update_aliases(body={"actions": actions})

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

    def lexical_search(
        self,
        query: str,
        *,
        size: int | None = None,
        filter_query: Mapping[str, Any] | None = None,
        source_excludes: Sequence[str] | None = None,
        source_includes: Sequence[str] | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> SearchResult:
        """Run a BM25 ``match`` query. Does not use a search pipeline."""
        _require_nonempty(query, "query")
        size = size if size is not None else self.config.size
        body = build_lexical_query(
            query,
            text_field=self.config.text_field,
            size=size,
            filter_query=filter_query,
            source_excludes=self._source_excludes(source_excludes),
            source_includes=source_includes,
            extra_body=extra_body,
        )
        response = self.client.search(index=self.config.index, body=body)
        return parse_search_response(response)

    def knn_search(
        self,
        embedding: Sequence[float],
        *,
        size: int | None = None,
        knn_k: int | None = None,
        filter_query: Mapping[str, Any] | None = None,
        source_excludes: Sequence[str] | None = None,
        source_includes: Sequence[str] | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> SearchResult:
        """Run a raw ``knn`` query. Does not use a search pipeline."""
        self._require_embedding(embedding)
        size = size if size is not None else self.config.size
        k = knn_k if knn_k is not None else max(self.config.knn_k, size)
        body = build_knn_query(
            embedding,
            vector_field=self.config.vector_field,
            size=size,
            knn_k=k,
            filter_query=filter_query,
            source_excludes=self._source_excludes(source_excludes),
            source_includes=source_includes,
            extra_body=extra_body,
        )
        response = self.client.search(index=self.config.index, body=body)
        return parse_search_response(response)

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
        _require_nonempty(query, "query")
        self._require_embedding(embedding)
        size = size if size is not None else self.config.size
        k = knn_k if knn_k is not None else max(self.config.knn_k, size)
        body = build_hybrid_query(
            query,
            embedding,
            text_field=self.config.text_field,
            vector_field=self.config.vector_field,
            size=size,
            knn_k=k,
            filter_query=filter_query,
            source_excludes=self._source_excludes(source_excludes),
            source_includes=source_includes,
            extra_body=extra_body,
        )
        response = self.client.search(
            index=self.config.index,
            body=body,
            params={"search_pipeline": pipeline_name or self.config.pipeline_name},
        )
        return parse_search_response(response)

    def _require_embedding(self, embedding: Sequence[float]) -> None:
        n = len(embedding)
        if n == 0:
            raise ValueError("embedding must be a non-empty sequence")
        if n != self.config.dimension:
            raise ValueError(
                f"embedding length {n} does not match "
                f"config.dimension {self.config.dimension}; use the same model "
                "as at index time"
            )

    def _source_excludes(
        self, source_excludes: Sequence[str] | None
    ) -> Sequence[str] | None:
        if source_excludes is None and self.config.exclude_vector:
            return [self.config.vector_field]
        return source_excludes
