# SET UP ENVRIONMENT
setup-venv:
	python -m pip install virtualenv & \
	python -m venv venv

install-poetry:
	pipx install poetry & \
	pipx ensurepath

install-dependencies:
	poetry lock & \
	poetry install

install: install-poetry install-dependencies

check-code-quality:
	poetry run isort src/ tests/ &
	poetry run black src/ tests/ &
	poetry run pylint src/ tests/

check-coverage:
	poetry run pytest --cov=src/ tests/

checks: check-code-quality check-coverage

run:
	poetry run graphic-designer-tool
