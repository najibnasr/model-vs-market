.PHONY: install data data-offline test lint

# PYTHONPATH=src : robuste même si l'install éditable n'est pas vue par Python (cas macOS + Python 3.13)
RUN = PYTHONPATH=src uv run

install:
	uv sync --all-extras

# Télécharge les CSV manquants puis reconstruit le schéma canonique et les rapports
data:
	$(RUN) python -m mvm.sports.foot.build --config configs/foot.yaml --download

# Reconstruit uniquement à partir de data/raw (aucun accès réseau)
data-offline:
	$(RUN) python -m mvm.sports.foot.build --config configs/foot.yaml

test:
	$(RUN) pytest

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
