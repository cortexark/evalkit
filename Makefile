.PHONY: install dev test lint format typecheck coverage clean dashboard help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package in production mode
	pip install -e .

dev: ## Install package with all dev dependencies
	pip install -e ".[all,dev]"

test: ## Run test suite
	pytest tests/ -v --tb=short

test-fast: ## Run tests excluding slow markers
	pytest tests/ -v --tb=short -m "not slow"

lint: ## Run linter
	ruff check src/ tests/

format: ## Auto-format code
	ruff format src/ tests/
	ruff check --fix src/ tests/

typecheck: ## Run type checker
	mypy src/evalkit/

coverage: ## Run tests with coverage report
	pytest tests/ --cov=evalkit --cov-report=term-missing --cov-report=html

dashboard: ## Launch the Streamlit dashboard
	streamlit run src/evalkit/dashboard/app.py

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

ci: lint typecheck test ## Run full CI pipeline locally
