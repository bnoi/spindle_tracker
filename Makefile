.PHONY: clean flake8

clean:
	find . -name "*.so" -exec rm -rf {} \;
	find . -name "*.pyc" -exec rm -rf {} \;
	find . -depth -name "__pycache__" -type d -exec rm -rf '{}' \;
	rm -rf build/ dist/ *.egg-info/

flake8:
	flake8 --exclude "test_*" --max-line-length=100 --count --statistics --exit-zero spindle_tracker/
