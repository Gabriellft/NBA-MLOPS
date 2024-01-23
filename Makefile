dir=mlops_nba
VENV=venv
PYTHON_VERSION=3
PRECOMMIT=$(VENV)/bin/pre-commit
PYTHON=$(VENV)/bin/python$(PYTHON_VERSION)
SAFETY=$(VENV)/bin/safety

## environnement
clean-venv:
	echo "Have a look on if you're on Apple M2 chips https://stackoverflow.com/a/76264243"
# rm -rf $(VENV)

add-venv:
	python -m venv $(VENV)
	echo "Have a look on if you're on Apple M2 chips https://stackoverflow.com/a/76264243"

install-dev:
	python -m pip install --upgrade pip
	python -m pip install -r requirements-dev.txt

install:
	python -m pip install -r requirements.txt --no-cache-dir


## linters
lint:
	python -m pylint $(dir) --rcfile ./setup.cfg

black:
	python -m black $(dir) --check

flake:
	python -m flake8 $(dir) --config ./setup.cfg

isort:
	python -m isort $(dir) --check-only --settings-file ./setup.cfg

format:
	python -m black $(dir)
	python -m isort $(dir) --settings-file ./setup.cfg


check: black isort flake

format-tests:
	python -m black tests
	python -m isort tests --settings-file ./setup.cfg

## unit tests and coverage
test:
	python -m pytest tests -vv --capture=tee-sys

coverage:
	python -m pytest tests --cov-config=.coveragerc --cov=$(dir)

coverage-html:
	python -m pytest tests --cov-config=.coveragerc --cov=$(dir) --cov-report html


clean-logs:
	rm -rf .pytest_cache
	rm -rf htmlcov
	find . -path '*/.output*' -delete
	find . -path '*/__pycache__*' -delete
	find . -path '*/.ipynb_checkpoints*' -delete

safety:
	$(SAFETY) check

setup: clean-venv add-venv install-dev install

extract-rising-stars:
	python3 -m mlops_nba.potential_stars.extract

run: extract-rising-stars