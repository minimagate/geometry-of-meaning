.PHONY: help install lint format typecheck test clean run-compress run-analyze

.DEFAULT_GOAL := help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package in dev mode with all optional deps
	pip install -e ".[all]"

lint: ## Lint with ruff
	ruff check src/ tests/ scripts/

format: ## Format code with ruff
	ruff format src/ tests/ scripts/

typecheck: ## Type-check with mypy
	mypy src/

test: ## Run test suite
	pytest --cov=geometry_of_meaning --cov-report=term-missing

clean: ## Remove build artifacts, caches, and egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .mypy_cache/ .ruff_cache/ .pytest_cache/ htmlcov/ .coverage

run-compress: ## Run progressive compression — generate.py
	python experiments/semantic_preservation/progressive_compression/generate.py

run-analyze: ## Run progressive compression — analyze.py
	python experiments/semantic_preservation/progressive_compression/analyze.py