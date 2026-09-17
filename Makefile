.PHONY: test lint demo-up demo-down

test:
	pytest -m "not integration"

lint:
	ruff check src adapters tests examples

demo-up:
	docker compose up -d

demo-down:
	docker compose down
