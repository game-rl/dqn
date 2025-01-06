publish:
	python3 -m build
	python3 -m twine upload dist/*
	rm -rf dist/ *.egg-info