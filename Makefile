.PHONY: install data data-offline test lint market-baseline tune-elo eval-elo test-elo tune-dc eval-dc test-dc eval-ens test-ens

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

# Étape (b) : performance du marché sur la période d'évaluation (la barre à franchir)
market-baseline:
	$(RUN) python -m mvm.sports.foot.market_baseline --config configs/foot.yaml

# Étape (c) : Elo. Le tuning ne voit que 2012-13 à 2016-17.
tune-elo:
	$(RUN) python -m mvm.sports.foot.run_model tune --model elo

eval-elo:
	$(RUN) python -m mvm.sports.foot.run_model eval --model elo --split tuning

# Évaluation sur le TEST : une seule fois par modèle final (journalisée dans reports/test_log.csv)
test-elo:
	$(RUN) python -m mvm.sports.foot.run_model eval --model elo --split test --confirm-test

# Étape (d) : Dixon-Coles (comparé au marché, au naïf ET à Elo)
tune-dc:
	$(RUN) python -m mvm.sports.foot.run_model tune --model dc

eval-dc:
	$(RUN) python -m mvm.sports.foot.run_model eval --model dc --split tuning

test-dc:
	$(RUN) python -m mvm.sports.foot.run_model eval --model dc --split test --confirm-test

# Combinaison Elo + Dixon-Coles (50/50, sans réglage)
eval-ens:
	$(RUN) python -m mvm.sports.foot.run_model eval --model ens --split tuning

test-ens:
	$(RUN) python -m mvm.sports.foot.run_model eval --model ens --split test --confirm-test
