.PHONY: setup clean test

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e .[dev]
	@echo "Setup complete. Activate with: source .venv/bin/activate"

clean:
	rm -rf .venv
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

test:
	.venv/bin/pytest
