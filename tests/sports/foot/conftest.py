"""Fixtures : deux faux fichiers football-data qui reproduisent les pièges réels.

- 2005-2006 : ancien format (dates dd/mm/yy, agrégats Betbrain BbAv/BbMx, pas d'heure,
  pas de Pinnacle, pas de clôture), avec une cote erronée et une cote périmée.
- 2020-2021 : nouveau format (dates dd/mm/yyyy, heure, Pinnacle + clôture, Avg/Max),
  encodé en latin-1, avec colonne vide en fin de ligne, ligne vide et match non joué.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest

OLD_TEAMS = ["Arsenal", "Chelsea", "Everton", "Fulham"]
NEW_TEAMS = ["Arsenal", "Chelsea", "Everton", "Münster Town"]  # nom avec accent (latin-1)


def _fixtures(teams):
    return list(itertools.permutations(teams, 2))  # aller-retour : n(n-1) matchs


def _old_file(path: Path) -> None:
    header = (
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A,"
        "BbAvH,BbAvD,BbAvA,BbMxH,BbMxD,BbMxA,BbAH"
    )
    lines = [header]
    for i, (h, a) in enumerate(_fixtures(OLD_TEAMS)):
        hg, ag = i % 3, (i + 1) % 2
        ftr = "H" if hg > ag else "A" if hg < ag else "D"
        b365 = ("2.10", "3.30", "3.60")
        if i == 0:
            b365 = ("0.95", "3.30", "3.60")  # cote <= 1 : faute de saisie
        if i == 1:
            b365 = ("9.00", "3.30", "3.60")  # overround aberrant : cote périmée
        lines.append(
            f"E0,{13 + i:02d}/08/05,{h},{a},{hg},{ag},{ftr},{','.join(b365)},"
            f"2.05,3.25,3.55,2.25,3.45,3.80,20"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _new_file(path: Path) -> None:
    header = (
        "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,PSH,PSD,PSA,AvgH,AvgD,AvgA,"
        "MaxH,MaxD,MaxA,PSCH,PSCD,PSCA,AvgCH,AvgCD,AvgCA,"
    )
    lines = [header]
    for i, (h, a) in enumerate(_fixtures(NEW_TEAMS)):
        hg, ag = (i + 2) % 3, i % 2
        ftr = "H" if hg > ag else "A" if hg < ag else "D"
        if i == 3:
            ftr = "A" if ftr != "A" else "H"  # FTR incohérent : doit être signalé, pas cru
        lines.append(
            f"E0,{12 + i:02d}/09/2020,15:00,{h},{a},{hg},{ag},{ftr},"
            f"2.00,3.50,3.90,1.95,3.40,3.80,2.08,3.60,4.10,1.90,3.60,4.20,1.88,3.50,4.05,"
        )
    lines.append("E0,30/05/2021,15:00,Arsenal,Fulham,,,,2.0,3.5,3.9,,,,,,,,,,,,,")  # non joué
    lines.append(",,,,,,,,,,,,,,,,,,,,,,,")  # ligne vide
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("\n".join(lines) + "\n").encode("latin-1"))


@pytest.fixture
def cfg(tmp_path: Path) -> dict:
    raw = tmp_path / "raw"
    _old_file(raw / "2005-2006" / "E0.csv")
    _new_file(raw / "2020-2021" / "E0.csv")
    return {
        "seasons": {"first": 2005, "last": 2020},
        "leagues": {"E0": {"country": "eng", "tier": 1, "name": "Premier League"}},
        "paths": {
            "raw": str(raw),
            "processed": str(tmp_path / "processed"),
            "reports": str(tmp_path / "reports"),
            "team_aliases": str(tmp_path / "none.yaml"),
        },
        "quality": {"overround_min": 1.0, "overround_max": 1.2, "overround_min_best_price": 0.9},
    }
