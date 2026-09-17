from __future__ import annotations

import builtins
import importlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from os_hybrid_kit.integrations import InstallError

_INTEGRATION_MODULES = (
    "os_hybrid_kit.integrations.langchain",
    "os_hybrid_kit.integrations.llama_index",
)


def _matches(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


@contextmanager
def _blocked_import(*prefixes: str) -> Iterator[None]:
    real_import = builtins.__import__
    saved = {
        key: module
        for key, module in sys.modules.items()
        if _matches(key, prefixes) or key in _INTEGRATION_MODULES
    }

    def blocked(
        name: str,
        globals: dict | None = None,
        locals: dict | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ):
        if _matches(name, prefixes):
            raise ImportError(f"blocked {name}")
        return real_import(name, globals, locals, fromlist, level)

    for key in saved:
        sys.modules.pop(key, None)

    builtins.__import__ = blocked  # type: ignore[method-assign]
    try:
        yield
    finally:
        builtins.__import__ = real_import  # type: ignore[method-assign]
        for key in list(sys.modules):
            if _matches(key, prefixes) or key in _INTEGRATION_MODULES:
                sys.modules.pop(key, None)
        sys.modules.update(saved)


def test_langchain_module_raises_install_error_when_extra_missing() -> None:
    with (
        _blocked_import("langchain_core"),
        pytest.raises(InstallError, match=r"os-hybrid-kit\[langchain\]"),
    ):
        importlib.import_module("os_hybrid_kit.integrations.langchain")


def test_llama_index_module_raises_install_error_when_extra_missing() -> None:
    with (
        _blocked_import("llama_index"),
        pytest.raises(InstallError, match=r"os-hybrid-kit\[llama-index\]"),
    ):
        importlib.import_module("os_hybrid_kit.integrations.llama_index")
