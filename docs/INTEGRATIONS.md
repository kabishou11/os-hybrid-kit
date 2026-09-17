# Integrations (optional)

Thin retrievers around the frozen `HybridKit` facade. They do not add chains, query engines, or a plugin marketplace.

Install only the extra you need:

```bash
pip install "os-hybrid-kit[langchain]"
pip install "os-hybrid-kit[llama-index]"
```

Missing extras raise `os_hybrid_kit.integrations.InstallError` on import of the submodule.

`search_kwargs` (both retrievers):

| Key | Meaning |
| --- | --- |
| `mode` | `hybrid` (default), `lexical` (BM25, no embed), or `knn` |
| `size` | result count (otherwise `HybridConfig.size`) |
| `filter` | OpenSearch filter clause (also accepts `filter_query`) |

Other `HybridKit` search keywords (`knn_k`, `source_includes`, `extra_body`, …) pass through.

## LangChain

```python
from os_hybrid_kit import HybridConfig, HybridKit
from os_hybrid_kit.integrations.langchain import HybridKitRetriever

kit = HybridKit(HybridConfig(dimension=384, index="docs"))
retriever = HybridKitRetriever(
    kit,
    embed_fn,  # (text: str) -> Sequence[float]; same model as at index time
    search_kwargs={"size": 10, "mode": "hybrid"},
)
docs = retriever.invoke("waterproof trail shoes")
# docs[0].page_content, docs[0].metadata["id"], docs[0].metadata["score"]
```

`HybridKitRetriever` subclasses `langchain_core.retrievers.BaseRetriever`. Use `.invoke` (or `.get_relevant_documents`). Lexical mode does not call `embed_fn`.

## LlamaIndex

```python
from os_hybrid_kit import HybridConfig, HybridKit
from os_hybrid_kit.integrations.llama_index import HybridKitRetriever

kit = HybridKit(HybridConfig(dimension=384, index="docs"))
retriever = HybridKitRetriever(
    kit,
    embed_fn,
    search_kwargs={"size": 10, "mode": "hybrid"},
)
nodes = retriever.retrieve("waterproof trail shoes")
# nodes[0].get_text(), nodes[0].score, nodes[0].node.id_
```

`HybridKitRetriever` subclasses LlamaIndex `BaseRetriever` and returns `NodeWithScore` (`TextNode` + score).

The core package `__all__` is unchanged. Import retrievers from the integration submodules.
