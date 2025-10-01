prj-dir := $(shell pwd)
venv-dir := $(prj-dir)/venv
python-native := python3.12
python := $(venv-dir)/bin/python
pip := $(venv-dir)/bin/pip

.PHONY: help prepare-prod install clean run batch

help:
	@echo "Whisper SRT - Commands"
	@echo "======================"
	@echo "Setup:"
	@echo "  make prepare-prod  - Setup environment"
	@echo "  make install       - Install dependencies"
	@echo ""
	@echo "Running:"
	@echo "  make run FILE=video.mp4    - Process video"
	@echo "  make batch DIR=/videos     - Batch process"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         - Remove venv"
	@echo "  make clean-output  - Remove SRT files"

prepare-prod: create-env install
	@echo "✅ Ready! Run: source venv/bin/activate"

create-env:
	$(python-native) -m venv $(venv-dir)
	$(pip) install --upgrade pip

install:
	$(pip) install -e .

run:
	@if [ -z "$(FILE)" ]; then echo "Usage: make run FILE=video.mp4"; exit 1; fi
	$(python) -m whisper_srt.cli $(FILE) $(ARGS)

batch:
	@if [ -z "$(DIR)" ]; then echo "Usage: make batch DIR=/videos"; exit 1; fi
	$(python) -m whisper_srt.batch $(DIR) $(ARGS)

clean:
	rm -rf $(venv-dir)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleaned"

clean-output:
	rm -f *.srt *.vtt *.ass *.wav
	@echo "✅ Output cleaned"
