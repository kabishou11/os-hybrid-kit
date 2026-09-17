from __future__ import annotations

from typing import Any

import pytest
from opensearchpy.exceptions import NotFoundError

from os_hybrid_kit import FusionMethod, HybridConfig, HybridKit


class FakeIndices:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.deleted: list[str] = []
        self.exists_index = False
        self.aliases: dict[str, set[str]] = {}
        self.put_alias_calls: list[dict[str, str]] = []
        self.delete_alias_calls: list[dict[str, str]] = []
        self.get_alias_calls: list[str] = []
        self.update_aliases_calls: list[dict[str, Any]] = []

    def exists(self, index: str) -> bool:
        return self.exists_index

    def create(self, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.created.append({"index": index, "body": body})
        self.exists_index = True
        return {"acknowledged": True}

    def delete(self, index: str) -> dict[str, Any]:
        self.deleted.append(index)
        self.exists_index = False
        return {"acknowledged": True}

    def put_alias(self, *, index: str, name: str) -> dict[str, Any]:
        self.put_alias_calls.append({"index": index, "name": name})
        self.aliases.setdefault(name, set()).add(index)
        return {"acknowledged": True}

    def delete_alias(self, *, index: str, name: str) -> dict[str, Any]:
        self.delete_alias_calls.append({"index": index, "name": name})
        targets = self.aliases.get(name)
        if targets is not None:
            targets.discard(index)
            if not targets:
                del self.aliases[name]
        return {"acknowledged": True}

    def get_alias(self, *, name: str) -> dict[str, Any]:
        self.get_alias_calls.append(name)
        targets = self.aliases.get(name)
        if not targets:
            raise NotFoundError(404, "alias_not_found", {"error": "alias not found"})
        return {index: {"aliases": {name: {}}} for index in sorted(targets)}

    def update_aliases(self, *, body: dict[str, Any]) -> dict[str, Any]:
        self.update_aliases_calls.append(body)
        for action in body.get("actions", []):
            if "add" in action:
                alias = action["add"]["alias"]
                index = action["add"]["index"]
                self.aliases.setdefault(alias, set()).add(index)
            if "remove" in action:
                alias = action["remove"]["alias"]
                index = action["remove"]["index"]
                targets = self.aliases.get(alias)
                if targets is not None:
                    targets.discard(index)
                    if not targets:
                        del self.aliases[alias]
        return {"acknowledged": True}


class FakeTransport:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def perform_request(self, method: str, path: str, body: dict[str, Any] | None = None):
        self.calls.append({"method": method, "path": path, "body": body})
        return {"acknowledged": True}


class FakeClient:
    def __init__(self) -> None:
        self.indices = FakeIndices()
        self.transport = FakeTransport()
        self.searches: list[dict[str, Any]] = []

    def search(self, **kwargs: Any) -> dict[str, Any]:
        self.searches.append(kwargs)
        return {
            "hits": {
                "total": {"value": 1, "relation": "eq"},
                "max_score": 0.5,
                "hits": [
                    {
                        "_id": "1",
                        "_index": "hybrid-test",
                        "_score": 0.5,
                        "_source": {"content": "trail running shoes"},
                    }
                ],
            }
        }


def test_ensure_index_creates_once():
    client = FakeClient()
    kit = HybridKit(HybridConfig(dimension=4, index="hybrid-test"), client=client)  # type: ignore[arg-type]
    assert kit.ensure_index(extra_properties={"title": {"type": "text"}}) is True
    assert kit.ensure_index() is False
    created = client.indices.created[0]
    assert created["index"] == "hybrid-test"
    assert created["body"]["settings"]["index.knn"] is True
    assert "title" in created["body"]["mappings"]["properties"]


def test_upsert_pipeline_puts_rrf_body():
    client = FakeClient()
    kit = HybridKit(
        HybridConfig(dimension=4, fusion=FusionMethod.RRF, pipeline_name="rrf-pipeline"),
        client=client,  # type: ignore[arg-type]
    )
    kit.upsert_pipeline()
    call = client.transport.calls[0]
    assert call["method"] == "PUT"
    assert call["path"] == "/_search/pipeline/rrf-pipeline"
    assert "score-ranker-processor" in call["body"]["phase_results_processors"][0]


def test_hybrid_search_sends_pipeline_and_hybrid_body():
    client = FakeClient()
    kit = HybridKit(HybridConfig(dimension=3, size=5, knn_k=10), client=client)  # type: ignore[arg-type]
    result = kit.hybrid_search("running shoes", [0.1, 0.2, 0.3], size=4)
    request = client.searches[0]
    assert request["index"] == "hybrid-index"
    assert request["params"] == {"search_pipeline": "hybrid-search-pipeline"}
    body = request["body"]
    assert body["size"] == 4
    knn = body["query"]["hybrid"]["queries"][1]["knn"]["embedding"]
    assert knn["k"] == 10
    assert knn["vector"] == [0.1, 0.2, 0.3]
    assert body["_source"]["excludes"] == ["embedding"]
    assert result.hits[0].id == "1"
    assert result.hits[0].source["content"] == "trail running shoes"


def test_lexical_search_sends_match_query_without_pipeline():
    client = FakeClient()
    kit = HybridKit(HybridConfig(dimension=3, size=5), client=client)  # type: ignore[arg-type]
    result = kit.lexical_search("running shoes", size=2)
    request = client.searches[0]
    assert request["index"] == "hybrid-index"
    assert "params" not in request
    assert request["body"]["size"] == 2
    assert request["body"]["query"] == {"match": {"content": {"query": "running shoes"}}}
    assert request["body"]["_source"]["excludes"] == ["embedding"]
    assert result.hits[0].id == "1"


def test_knn_search_sends_raw_knn_without_pipeline():
    client = FakeClient()
    kit = HybridKit(HybridConfig(dimension=3, size=5, knn_k=10), client=client)  # type: ignore[arg-type]
    result = kit.knn_search([0.1, 0.2, 0.3], size=4)
    request = client.searches[0]
    assert "params" not in request
    knn = request["body"]["query"]["knn"]["embedding"]
    assert knn["vector"] == [0.1, 0.2, 0.3]
    assert knn["k"] == 10
    assert request["body"]["size"] == 4
    assert result.hits[0].id == "1"


def test_lexical_search_forwards_filter_query():
    client = FakeClient()
    kit = HybridKit(HybridConfig(dimension=3), client=client)  # type: ignore[arg-type]
    kit.lexical_search("running shoes", filter_query={"term": {"category": "shoes"}})
    query = client.searches[0]["body"]["query"]
    assert query["bool"]["filter"] == [{"term": {"category": "shoes"}}]


def test_knn_search_forwards_filter_query():
    client = FakeClient()
    kit = HybridKit(HybridConfig(dimension=3), client=client)  # type: ignore[arg-type]
    kit.knn_search([0.1, 0.2, 0.3], filter_query={"term": {"category": "shoes"}})
    knn = client.searches[0]["body"]["query"]["knn"]["embedding"]
    assert knn["filter"] == {"term": {"category": "shoes"}}


def test_hybrid_search_forwards_filter_query():
    client = FakeClient()
    kit = HybridKit(HybridConfig(dimension=3), client=client)  # type: ignore[arg-type]
    kit.hybrid_search(
        "running shoes",
        [0.1, 0.2, 0.3],
        filter_query={"term": {"category": "shoes"}},
    )
    hybrid = client.searches[0]["body"]["query"]["hybrid"]
    assert hybrid["filter"] == {"term": {"category": "shoes"}}


def test_index_documents_pops_id_from_source(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_bulk(client: Any, actions: Any, refresh: bool = True) -> tuple[int, list[Any]]:
        captured["client"] = client
        captured["actions"] = list(actions)
        captured["refresh"] = refresh
        return (1, [])

    monkeypatch.setattr("os_hybrid_kit.client.bulk", fake_bulk)
    client = FakeClient()
    kit = HybridKit(HybridConfig(dimension=2, index="docs"), client=client)  # type: ignore[arg-type]
    result = kit.index_documents(
        [{"_id": "doc-1", "content": "hello", "embedding": [0.1, 0.2]}]
    )
    assert result == (1, [])
    assert captured["client"] is client
    assert captured["refresh"] is True
    action = captured["actions"][0]
    assert action["_index"] == "docs"
    assert action["_id"] == "doc-1"
    assert action["_source"] == {"content": "hello", "embedding": [0.1, 0.2]}
    assert "_id" not in action["_source"]


def test_index_documents_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_bulk(client: Any, actions: Any, refresh: bool = True) -> tuple[int, list[Any]]:
        captured["actions"] = list(actions)
        captured["refresh"] = refresh
        return (0, [])

    monkeypatch.setattr("os_hybrid_kit.client.bulk", fake_bulk)
    kit = HybridKit(HybridConfig(dimension=2, index="docs"), client=FakeClient())  # type: ignore[arg-type]
    result = kit.index_documents([])
    assert result == (0, [])
    assert captured["actions"] == []
    assert captured["refresh"] is True


def test_index_documents_omits_id_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_bulk(client: Any, actions: Any, refresh: bool = True) -> tuple[int, list[Any]]:
        captured["actions"] = list(actions)
        return (1, [])

    monkeypatch.setattr("os_hybrid_kit.client.bulk", fake_bulk)
    kit = HybridKit(HybridConfig(dimension=2, index="docs"), client=FakeClient())  # type: ignore[arg-type]
    kit.index_documents([{"content": "hello"}])
    action = captured["actions"][0]
    assert "_id" not in action
    assert action["_source"] == {"content": "hello"}


def test_knn_search_rejects_wrong_embedding_length():
    kit = HybridKit(HybridConfig(dimension=3), client=FakeClient())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="embedding length"):
        kit.knn_search([0.1, 0.2])


def test_hybrid_search_rejects_wrong_embedding_length():
    kit = HybridKit(HybridConfig(dimension=3), client=FakeClient())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="embedding length"):
        kit.hybrid_search("q", [0.1, 0.2])


def test_exists_and_delete_index():
    client = FakeClient()
    kit = HybridKit(HybridConfig(dimension=4, index="hybrid-test"), client=client)  # type: ignore[arg-type]
    assert kit.exists_index() is False
    assert kit.delete_index() is False
    assert kit.ensure_index() is True
    assert kit.exists_index() is True
    assert kit.delete_index() is True
    assert kit.exists_index() is False
    assert client.indices.deleted == ["hybrid-test"]


def test_put_and_get_alias_defaults_to_config_index():
    client = FakeClient()
    kit = HybridKit(HybridConfig(dimension=4, index="docs-v1"), client=client)  # type: ignore[arg-type]
    kit.put_alias("docs-read")
    assert client.indices.put_alias_calls == [{"index": "docs-v1", "name": "docs-read"}]
    mapping = kit.get_alias("docs-read")
    assert mapping == {"docs-v1": {"aliases": {"docs-read": {}}}}


def test_put_alias_accepts_explicit_index():
    client = FakeClient()
    kit = HybridKit(HybridConfig(dimension=4, index="docs-v1"), client=client)  # type: ignore[arg-type]
    kit.put_alias("docs-read", index="docs-v2")
    assert client.indices.put_alias_calls == [{"index": "docs-v2", "name": "docs-read"}]
    assert "docs-v2" in kit.get_alias("docs-read")


def test_delete_alias_defaults_to_config_index():
    client = FakeClient()
    kit = HybridKit(HybridConfig(dimension=4, index="docs-v1"), client=client)  # type: ignore[arg-type]
    kit.put_alias("docs-read")
    kit.delete_alias("docs-read")
    assert client.indices.delete_alias_calls == [{"index": "docs-v1", "name": "docs-read"}]
    with pytest.raises(NotFoundError):
        kit.get_alias("docs-read")


def test_swap_alias_with_old_index_is_one_update_aliases_call():
    client = FakeClient()
    kit = HybridKit(HybridConfig(dimension=4, index="docs-v1"), client=client)  # type: ignore[arg-type]
    kit.swap_alias("docs-read", "docs-v2", old_index="docs-v1")
    assert client.indices.get_alias_calls == []
    assert client.indices.update_aliases_calls == [
        {
            "actions": [
                {"remove": {"index": "docs-v1", "alias": "docs-read"}},
                {"add": {"index": "docs-v2", "alias": "docs-read"}},
            ]
        }
    ]
    assert kit.get_alias("docs-read") == {"docs-v2": {"aliases": {"docs-read": {}}}}


def test_swap_alias_looks_up_current_targets_when_old_index_omitted():
    client = FakeClient()
    kit = HybridKit(HybridConfig(dimension=4, index="docs-v1"), client=client)  # type: ignore[arg-type]
    kit.put_alias("docs-read", index="docs-v1")
    kit.swap_alias("docs-read", "docs-v2")
    assert client.indices.get_alias_calls == ["docs-read"]
    assert client.indices.update_aliases_calls == [
        {
            "actions": [
                {"remove": {"index": "docs-v1", "alias": "docs-read"}},
                {"add": {"index": "docs-v2", "alias": "docs-read"}},
            ]
        }
    ]
    assert list(kit.get_alias("docs-read")) == ["docs-v2"]


def test_swap_alias_missing_alias_only_adds():
    client = FakeClient()
    kit = HybridKit(HybridConfig(dimension=4, index="docs-v1"), client=client)  # type: ignore[arg-type]
    kit.swap_alias("docs-read", "docs-v1")
    assert client.indices.update_aliases_calls == [
        {"actions": [{"add": {"index": "docs-v1", "alias": "docs-read"}}]}
    ]
    assert list(kit.get_alias("docs-read")) == ["docs-v1"]


def test_with_index_shares_client_and_retargets_search():
    client = FakeClient()
    kit = HybridKit(HybridConfig(dimension=3, index="docs-v1"), client=client)  # type: ignore[arg-type]
    aliased = kit.with_index("docs-read")
    assert aliased.client is client
    assert aliased.config.index == "docs-read"
    assert kit.config.index == "docs-v1"
    aliased.lexical_search("running shoes")
    assert client.searches[0]["index"] == "docs-read"


@pytest.mark.parametrize("blank", ["", "  ", "\t"])
def test_put_alias_rejects_blank_name(blank: str):
    client = FakeClient()
    kit = HybridKit(HybridConfig(dimension=4, index="docs-v1"), client=client)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="alias must be a non-empty string"):
        kit.put_alias(blank)
    assert client.indices.put_alias_calls == []


def test_put_alias_rejects_blank_index():
    client = FakeClient()
    kit = HybridKit(HybridConfig(dimension=4, index="docs-v1"), client=client)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="index must be a non-empty string"):
        kit.put_alias("docs-read", index="  ")
    assert client.indices.put_alias_calls == []


@pytest.mark.parametrize("blank", ["", "  "])
def test_delete_alias_rejects_blank_name(blank: str):
    client = FakeClient()
    kit = HybridKit(HybridConfig(dimension=4, index="docs-v1"), client=client)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="alias must be a non-empty string"):
        kit.delete_alias(blank)
    assert client.indices.delete_alias_calls == []


def test_get_alias_rejects_blank_name():
    client = FakeClient()
    kit = HybridKit(HybridConfig(dimension=4, index="docs-v1"), client=client)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="alias must be a non-empty string"):
        kit.get_alias("")
    assert client.indices.get_alias_calls == []


def test_swap_alias_rejects_blank_alias_and_indexes():
    client = FakeClient()
    kit = HybridKit(HybridConfig(dimension=4, index="docs-v1"), client=client)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="alias must be a non-empty string"):
        kit.swap_alias("  ", "docs-v2")
    with pytest.raises(ValueError, match="index must be a non-empty string"):
        kit.swap_alias("docs-read", "")
    with pytest.raises(ValueError, match="index must be a non-empty string"):
        kit.swap_alias("docs-read", "docs-v2", old_index=" ")
    assert client.indices.update_aliases_calls == []
    assert client.indices.get_alias_calls == []


@pytest.mark.parametrize("blank", ["", "  "])
def test_with_index_rejects_blank_name(blank: str):
    client = FakeClient()
    kit = HybridKit(HybridConfig(dimension=3, index="docs-v1"), client=client)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="index must be a non-empty string"):
        kit.with_index(blank)
    assert kit.config.index == "docs-v1"


@pytest.mark.parametrize("blank", ["", "  ", "\n"])
def test_lexical_search_rejects_blank_query_before_opensearch(blank: str):
    client = FakeClient()
    kit = HybridKit(HybridConfig(dimension=3), client=client)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        kit.lexical_search(blank)
    assert client.searches == []


@pytest.mark.parametrize("blank", ["", "  "])
def test_hybrid_search_rejects_blank_query_before_opensearch(blank: str):
    client = FakeClient()
    kit = HybridKit(HybridConfig(dimension=3), client=client)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        kit.hybrid_search(blank, [0.1, 0.2, 0.3])
    assert client.searches == []


def test_knn_search_rejects_empty_embedding():
    client = FakeClient()
    kit = HybridKit(HybridConfig(dimension=3), client=client)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="embedding must be a non-empty sequence"):
        kit.knn_search([])
    assert client.searches == []


def test_hybrid_search_rejects_empty_embedding():
    client = FakeClient()
    kit = HybridKit(HybridConfig(dimension=3), client=client)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="embedding must be a non-empty sequence"):
        kit.hybrid_search("running shoes", [])
    assert client.searches == []


def test_kit_repr_omits_secrets():
    kit = HybridKit(
        HybridConfig(dimension=4, index="docs", username="admin", password="s3cret"),
        client=FakeClient(),  # type: ignore[arg-type]
    )
    text = repr(kit)
    assert "docs" in text
    assert "s3cret" not in text
    assert "admin" not in text
    assert "password" not in text
