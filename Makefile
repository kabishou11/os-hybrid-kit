.PHONY: test lint test-extras demo-up demo-down

test:
	pytest -m "not integration"

lint:
	ruff check src adapters tests examples

test-extras:
	python -m pip install -e ".[dev,langchain,llama-index]"
	pytest -m "not integration"

demo-up:
	docker compose up -d

demo-down:
	docker compose down
