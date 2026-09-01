# Open Research Agent (ORA) developer Makefile.
# Auto-manages .venv; every target runs inside the virtual environment.

VENV      := .venv
PYTHON    := $(VENV)/bin/python
PIP       := $(VENV)/bin/pip
ORA       := $(VENV)/bin/ora
RUFF      := $(VENV)/bin/ruff

# Overridable parameters
T         ?= tests
ARGS      ?= -q
QUERY     ?=
INTENSITY ?= 2
TIMESTAMP := $(shell date +%Y-%m-%d-%H%M%S)
OUTPUT    ?= reports/ora-$(TIMESTAMP).md

.PHONY: help setup test lint format build check plan research clean clean-venv

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_.-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

$(VENV)/.stamp: pyproject.toml
	python3 -m venv $(VENV)
	$(PIP) install -e ".[dev]"
	@touch $@

setup: $(VENV)/.stamp ## Create .venv and install dev dependencies
	@echo "Environment ready."

test: $(VENV)/.stamp ## Run the test suite (T=tests/path ARGS="-x -q")
	$(PYTHON) -m pytest $(T) $(ARGS)

lint: $(VENV)/.stamp ## Run ruff checks on ora/ and tests/
	$(RUFF) check ora tests

format: $(VENV)/.stamp ## Auto-format code with ruff
	$(RUFF) format ora tests

build: $(VENV)/.stamp ## Build wheel and sdist into dist/
	$(PYTHON) -m build

check: lint test ## Run lint then tests (pre-PR gate)

plan: $(VENV)/.stamp ## Preview a research plan: make plan QUERY="..."
	@test -n "$(QUERY)" || (echo "QUERY is required. Usage: make plan QUERY=\"your research question\""; exit 1)
	$(ORA) plan "$(QUERY)"

research: $(VENV)/.stamp ## Run research: make research QUERY="..." INTENSITY=3 OUTPUT=reports/x.md
	@test -n "$(QUERY)" || (echo "QUERY is required. Usage: make research QUERY=\"your research question\""; exit 1)
	@mkdir -p reports
	$(ORA) research "$(QUERY)" --intensity $(INTENSITY) -y --output "$(OUTPUT)"

clean: ## Remove build artifacts and caches (keeps .venv)
	rm -rf dist build *.egg-info .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -not -path './.venv/*' -prune -exec rm -rf {} +

clean-venv: ## Remove the virtual environment
	rm -rf $(VENV)
