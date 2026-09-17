# Contributing

## Setup

Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
# optional retrievers: pip install -e ".[langchain,llama-index]"
```

## Checks

Unit tests do not need OpenSearch:

```bash
ruff check src adapters tests examples   # or: make lint
pytest -m "not integration"              # or: make test
# optional coverage (pytest-cov is on the dev extra):
pytest -m "not integration" --cov=os_hybrid_kit
```

Live cluster (optional; CI sets `RUN_INTEGRATION=1`):

```bash
docker compose up -d                     # or: make demo-up
RUN_INTEGRATION=1 pytest -m integration
docker compose down                      # or: make demo-down
```

Keep unit tests free of a running cluster. Mark live checks with `@pytest.mark.integration`.

## Git

- Prefer a few long-lived `work/*` branches; delete them after merge. Do not open a new branch for every small polish.
- Branches: `work/*`, `fix/*`, `feature/*` only
- Commit messages in English, first person, as the author of the change
- Do not add `Co-authored-by` or `Signed-off-by` trailers for tools or bots

## Scope

This repository is a small OpenSearch hybrid-search library plus a thin Dify adapter. Please do not add a RAG UI, eval suite, model gateway, or a full marketplace plugin unless that work is agreed first.
