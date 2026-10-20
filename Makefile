.PHONY: install data data-offline test lint

install:
	uv sync --all-extras

# Télécharge les CSV manquants puis reconstruit le schéma canonique et les rapports
data:
	uv run python -m mvm.sports.foot.build --config configs/foot.yaml --download

# Reconstruit uniquement à partir de data/raw (aucun accès réseau)
data-offline:
	uv run python -m mvm.sports.foot.build --config configs/foot.yaml

test:
	uv run pytest

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
