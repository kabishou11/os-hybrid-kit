# Contributing

## Setup

Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Checks

```bash
ruff check src adapters tests examples
pytest                 # unit tests only; no live OpenSearch
pytest -m integration  # optional; needs docker compose up
```

Keep unit tests free of a running cluster. Mark live checks with `@pytest.mark.integration`.

## Git

- Branches: `work/*`, `fix/*`, `feature/*` only
- Commit messages in English, first person, as the author of the change
- Do not add `Co-authored-by` or `Signed-off-by` trailers for tools or bots

## Scope

This repository is a small OpenSearch hybrid-search library plus a thin Dify adapter. Please do not add a RAG UI, eval suite, model gateway, or a full marketplace plugin unless that work is agreed first.
