prj-dir := $(shell pwd)
venv-dir := $(prj-dir)/venv
# Override at the command line, e.g. `make prepare-prod python-native=python3.11`.
python-native ?= python3.12
python := $(venv-dir)/bin/python
pip := $(venv-dir)/bin/pip
pytest := $(venv-dir)/bin/pytest

.PHONY: help prepare-prod install install-llm install-dev clean clean-output run batch test

help:
	@echo "whisper-srt - WhisperX-only SRT generator"
	@echo "========================================="
	@echo "Setup:"
	@echo "  make prepare-prod  - Create venv and install core deps"
	@echo "  make install       - Install core deps in existing venv"
	@echo "  make install-llm   - Add the optional LLM resolver deps"
	@echo "  make install-dev   - Add the dev/test deps"
	@echo ""
	@echo "Running:"
	@echo "  make run FILE=video.mp4 [ARGS=...]   - Process single video"
	@echo "  make batch DIR=/videos [ARGS=...]    - Batch process directory"
	@echo "  make test                            - Run unit tests"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         - Remove venv and __pycache__"
	@echo "  make clean-output  - Remove generated SRT/warnings files"

prepare-prod: create-env install
	@echo "Ready. Activate with: source venv/bin/activate"

create-env:
	$(python-native) -m venv $(venv-dir)
	$(pip) install --upgrade pip

install:
	$(pip) install -e .

install-llm:
	$(pip) install -e ".[llm]"

install-dev:
	$(pip) install -e ".[dev]"

run:
	@if [ -z "$(FILE)" ]; then echo "Usage: make run FILE=video.mp4 [ARGS=...]"; exit 1; fi
	$(python) -m whisper_srt.processor $(FILE) $(ARGS)

batch:
	@if [ -z "$(DIR)" ]; then echo "Usage: make batch DIR=/videos [ARGS=...]"; exit 1; fi
	$(python) -m whisper_srt.batch $(DIR) $(ARGS)

test:
	$(pytest) tests -q

clean:
	rm -rf $(venv-dir)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache pip-wheel-metadata
	@echo "Cleaned."

clean-output:
	rm -f *.srt *.srt.warnings.json *.vtt *.ass *.wav
	@echo "Output cleaned."
