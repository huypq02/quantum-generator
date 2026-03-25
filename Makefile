.PHONY: clean

help:
	@echo Available commands:
	@echo   make clean - Remove build artifacts and cache

clean:
	rm -rf build/ dist/ *.egg-info htmlcov/ .pytest_cache/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
