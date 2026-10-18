# Common tasks. Run `make help` to list them.
# Windows without make: the README lists the equivalent commands.

# Use the project virtualenv when it exists, so a forgotten `activate` cannot
# run the tests with the wrong interpreter.
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)
WEB = web

.PHONY: help setup build serve dev test test-backend test-web lint format check clean

help:          ## Show this list
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-14s %s\n", $$1, $$2}'

setup:         ## Install Python (dev) and web dependencies
	$(PY) -m pip install -r requirements-dev.txt
	$(PY) -m pip install -e . --no-deps
	cd $(WEB) && npm ci

build:         ## Build the web app into web/dist
	cd $(WEB) && npm run build

serve: build   ## Build, then start the server on :8000
	$(PY) -m rltestbed.cli serve --port 8000

dev:           ## Frontend dev server on :5173 (run `make serve` in another terminal)
	cd $(WEB) && npm run dev

test-backend:  ## Python tests
	$(PY) -m pytest tests -q

test-web:      ## Frontend tests
	cd $(WEB) && npm run test

test: test-backend test-web ## All tests

lint:          ## ESLint, Prettier check, and the token/contrast check
	cd $(WEB) && npm run lint && npm run format:check && npm run check:tokens

format:        ## Auto-format the frontend
	cd $(WEB) && npm run format

check: lint test build ## Everything CI runs

clean:         ## Remove build output and caches (keeps runs and models)
	rm -rf $(WEB)/dist .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
