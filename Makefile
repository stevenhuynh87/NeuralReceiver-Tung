# Makefile for nrx-bounded paper reproducibility and benchmarks

PYTHON ?= python

.PHONY: all figures tables test check clean help

help:
	@echo "Available commands:"
	@echo "  make figures    - Regenerate all paper figures (Fig 1 - Fig 4)"
	@echo "  make tables     - Regenerate all tables (Table 1 - Table 2)"
	@echo "  make all        - Regenerate figures, tables, and run checks"
	@echo "  make test       - Run full pytest test suite"
	@echo "  make clean      - Clean temporary cache files"

figures:
	$(PYTHON) -m src.eval.make_figures

tables:
	$(PYTHON) tables/table1_gamma_g.py
	$(PYTHON) tables/table2_complexity.py

test:
	$(PYTHON) -m pytest tests/ -v

all: figures tables test

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
