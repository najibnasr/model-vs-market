"""Lecture et écriture des tables canoniques, plus utilitaires de reproductibilité."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_table(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_json(obj: dict | list, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def read_json(path: str | Path) -> dict | list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
