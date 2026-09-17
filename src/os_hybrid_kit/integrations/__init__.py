"""Optional framework wrappers around the frozen HybridKit facade.

Retrievers live in submodules and import their extra only when loaded:

* ``os_hybrid_kit.integrations.langchain`` — ``pip install os-hybrid-kit[langchain]``
* ``os_hybrid_kit.integrations.llama_index`` — ``pip install os-hybrid-kit[llama-index]``
"""

from __future__ import annotations


class InstallError(ImportError):
    """Raised when an optional integration extra is not installed."""


__all__ = ["InstallError"]
