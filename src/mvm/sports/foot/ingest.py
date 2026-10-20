"""Téléchargement des CSV football-data.co.uk vers data/raw, sans aucune transformation.

Principe : la couche raw est immuable. On ne réécrit jamais un fichier déjà présent sauf demande
explicite (force=True), et chaque fichier est tracé dans un manifeste (URL, date, SHA-256).
Si football-data corrige un fichier a posteriori, le hash change et on le voit.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

from mvm.core.io import read_json, sha256_file, write_json

log = logging.getLogger(__name__)


def season_code(start_year: int) -> str:
    """2005 -> '0506' (convention des URL football-data)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    """2005 -> '2005-2006' (convention interne)."""
    return f"{start_year}-{start_year + 1}"


def raw_path(raw_dir: Path, league: str, start_year: int) -> Path:
    return raw_dir / season_label(start_year) / f"{league}.csv"


def download_all(cfg: dict, force: bool = False) -> list[dict]:
    """Télécharge toutes les combinaisons ligue x saison manquantes. Retourne le manifeste."""
    raw_dir = Path(cfg["paths"]["raw"])
    manifest_path = raw_dir / "manifest.json"
    manifest = {m["path"]: m for m in read_json(manifest_path)} if manifest_path.exists() else {}

    src = cfg["source"]
    session = requests.Session()
    for year in range(cfg["seasons"]["first"], cfg["seasons"]["last"] + 1):
        for league in cfg["leagues"]:
            dest = raw_path(raw_dir, league, year)
            if dest.exists() and not force:
                continue
            url = f"{src['base_url']}/{season_code(year)}/{league}.csv"
            try:
                resp = session.get(url, timeout=src["timeout_s"])
            except requests.RequestException as exc:
                log.warning("Échec réseau %s : %s", url, exc)
                continue
            if resp.status_code != 200 or len(resp.content) < 100:
                # Fichier absent : la ligue n'était pas couverte cette saison-là. On le note.
                log.info("Indisponible (%s) : %s", resp.status_code, url)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(resp.content)
            manifest[str(dest)] = {
                "path": str(dest),
                "url": url,
                "league": league,
                "season": season_label(year),
                "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "sha256": sha256_file(dest),
                "bytes": len(resp.content),
            }
            log.info("OK %s", url)
            time.sleep(src["request_delay_s"])

    entries = sorted(manifest.values(), key=lambda m: m["path"])
    write_json(entries, manifest_path)
    return entries
