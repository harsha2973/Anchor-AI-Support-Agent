# ==============================================================================
# AI Customer Support Agent - Automation Makefile
# ==============================================================================
# Usage:
#   make setup         - Create virtual environment and install dependencies
#   make download-data - Download or generate dataset stubs
#   make run           - Run the agent CLI pipeline
#   make eval          - Run the evaluation harness
#   make test          - Run test suite
#   make lint          - Run linter and code formatting checks
#   make clean         - Remove temporary build & cache files

PYTHON ?= python
VENV_DIR ?= .venv

ifeq ($(OS),Windows_NT)
    VENV_PYTHON = $(VENV_DIR)/Scripts/python
    VENV_PIP = $(VENV_DIR)/Scripts/pip
else
    VENV_PYTHON = $(VENV_DIR)/bin/python
    VENV_PIP = $(VENV_DIR)/bin/pip
endif

.PHONY: help setup download-data run eval test lint clean

help:
	@echo "Available commands:"
	@echo "  make setup         - Initialize .venv and install requirements"
	@echo "  make download-data - Download/prepare raw data"
	@echo "  make run           - Run agent CLI (interactive or query)"
	@echo "  make eval          - Run evaluation harness on golden set"
	@echo "  make test          - Run automated tests"
	@echo "  make lint          - Run ruff linter check"
	@echo "  make clean         - Remove pycache and artifacts"

setup:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env from .env.example"; fi

download-data:
	$(VENV_PYTHON) -m data.loader --download

run:
	$(VENV_PYTHON) -m agent.run --interactive

eval:
	$(VENV_PYTHON) -m eval.run --golden-set eval/golden_set/golden_examples.jsonl

test:
	$(VENV_PYTHON) -m pytest tests/ eval/

lint:
	$(VENV_PYTHON) -m ruff check .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
