"""Parsing des CSV football-data.co.uk vers le schéma canonique.

Les fichiers ont évolué sur 20 ans. Pièges traités explicitement ici :
- dates en dd/mm/yy (anciennes saisons) puis dd/mm/yyyy ;
- encodage parfois latin-1, lignes vides et colonnes "Unnamed" en fin de fichier ;
- noms de colonnes de résultat différents selon les fichiers (HomeTeam/HT, FTHG/HG, FTR/Res) ;
- colonnes de cotes qui apparaissent, disparaissent ou changent de nom (BbAvH -> AvgH en 2019/20) ;
- cotes de clôture (suffixe C : PSCH, AvgCH...) présentes seulement sur les saisons récentes.

On préfère une table de correspondance explicite à une détection "magique" : chaque colonne
de cote reconnue est traçable (source_column), et les colonnes non reconnues sont remontées.
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Préfixe de colonne -> identifiant de bookmaker. Les agrégats (moyenne, max) sont gardés à part
# car ce ne sont pas des books où l'on peut parier : "max" en particulier est un best-odds
# reconstitué a posteriori, donc optimiste pour un backtest.
BOOKMAKERS: dict[str, str] = {
    "B365": "bet365",
    "BW": "bwin",
    "IW": "interwetten",
    "LB": "ladbrokes",
    "PS": "pinnacle",
    "WH": "williamhill",
    "VC": "betvictor",
    "SJ": "stanjames",
    "GB": "gamebookers",
    "BS": "bluesquare",
    "SB": "sportingbet",
    "SO": "sportingodds",
    "SY": "stanleybet",
    "BF": "betfair",
    "BFE": "betfair_exchange",
    "1XB": "1xbet",
    # Agrégats : Betbrain jusqu'en 2018/19, puis un autre agrégateur (rupture de série)
    "BbAv": "market_avg",
    "BbMx": "market_max",
    "Avg": "market_avg",
    "Max": "market_max",
}

# Colonnes qui ressemblent à des cotes 1X2 mais n'en sont pas (ex. BbAH = nombre de books
# utilisés pour la moyenne du handicap asiatique).
NOT_ODDS = {"BbAH"}

_book_alt = "|".join(sorted(map(re.escape, BOOKMAKERS), key=len, reverse=True))
ODDS_COL_RE = re.compile(rf"^(?P<book>{_book_alt})(?P<closing>C?)(?P<outcome>[HDA])$")
LOOKS_LIKE_ODDS_RE = re.compile(r"^[A-Za-z0-9]{1,5}C?[HDA]$")

RESULT_ALIASES: dict[str, tuple[str, ...]] = {
    "home_team": ("HomeTeam", "HT", "Home"),
    "away_team": ("AwayTeam", "AT", "Away"),
    "home_score": ("FTHG", "HG"),
    "away_score": ("FTAG", "AG"),
    "result": ("FTR", "Res"),
    "date": ("Date",),
    "time": ("Time",),
}


@dataclass
class ParsedFile:
    league: str
    season: str
    matches: pd.DataFrame  # résultats, noms d'équipes encore bruts
    odds_wide: pd.DataFrame  # colonnes de cotes reconnues, indexées comme matches
    odds_columns: dict[str, tuple[str, str, str]]  # source -> (bookmaker, timing, outcome)
    unrecognized_columns: list[str] = field(default_factory=list)
    n_rows_raw: int = 0
    n_rows_unplayed: int = 0


def read_csv_robust(path: str | Path) -> pd.DataFrame:
    """Lit un CSV football-data quel que soit l'encodage, en ignorant les colonnes parasites."""
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    df = pd.read_csv(io.StringIO(text), dtype=str, on_bad_lines="skip", skipinitialspace=True)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.columns = df.columns.str.strip()
    return df.dropna(how="all")


def parse_dates(s: pd.Series) -> pd.Series:
    """Accepte dd/mm/yy et dd/mm/yyyy. Toute date non reconnue reste NaT (contrôlée ensuite)."""
    s = s.str.strip()
    long_fmt = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    short_fmt = pd.to_datetime(s, format="%d/%m/%y", errors="coerce")
    return long_fmt.fillna(short_fmt)


def _pick(df: pd.DataFrame, key: str) -> pd.Series | None:
    for col in RESULT_ALIASES[key]:
        if col in df.columns:
            return df[col]
    return None


def classify_odds_columns(columns: list[str]) -> tuple[dict[str, tuple[str, str, str]], list[str]]:
    mapping: dict[str, tuple[str, str, str]] = {}
    unrecognized: list[str] = []
    for col in columns:
        if col in NOT_ODDS:
            continue
        m = ODDS_COL_RE.match(col)
        if m:
            timing = "closing" if m["closing"] else "pre_match"
            mapping[col] = (BOOKMAKERS[m["book"]], timing, m["outcome"])
        elif LOOKS_LIKE_ODDS_RE.match(col) and col not in RESULT_ALIASES["result"]:
            unrecognized.append(col)
    # Deux colonnes sources pour le même (book, timing, issue), cas de la transition
    # Betbrain -> nouvel agrégateur : on garde le nouvel agrégateur et on signale le doublon.
    seen: dict[tuple[str, str, str], str] = {}
    for col in list(mapping):
        key = mapping[col]
        if key not in seen:
            seen[key] = col
            continue
        loser = col if col.startswith("Bb") else seen[key]
        winner = seen[key] if loser == col else col
        mapping.pop(loser)
        seen[key] = winner
        unrecognized.append(f"{loser} (doublon de {winner}, ignoré)")
    return mapping, unrecognized


def parse_file(path: str | Path, league: str, season: str) -> ParsedFile:
    df = read_csv_robust(path)
    n_raw = len(df)

    cols = {k: _pick(df, k) for k in RESULT_ALIASES}
    for required in ("home_team", "away_team", "home_score", "away_score", "date"):
        if cols[required] is None:
            raise ValueError(f"{path}: colonne '{required}' introuvable ({list(df.columns)})")

    matches = pd.DataFrame(
        {
            "league": league,
            "season": season,
            "kickoff_date": parse_dates(cols["date"]),
            "kickoff_time": cols["time"].str.strip() if cols["time"] is not None else None,
            "home_team_raw": cols["home_team"].str.strip(),
            "away_team_raw": cols["away_team"].str.strip(),
            "home_score": pd.to_numeric(cols["home_score"], errors="coerce"),
            "away_score": pd.to_numeric(cols["away_score"], errors="coerce"),
            "result_raw": cols["result"].str.strip() if cols["result"] is not None else None,
        }
    )

    # Lignes sans score : matchs non joués (saison en cours, report). Exclues mais comptées.
    played = matches["home_score"].notna() & matches["away_score"].notna()
    played &= matches["home_team_raw"].notna() & matches["away_team_raw"].notna()
    n_unplayed = int((~played).sum())
    matches = matches.loc[played].copy()
    matches["home_score"] = matches["home_score"].astype("int64")
    matches["away_score"] = matches["away_score"].astype("int64")

    mapping, unrecognized = classify_odds_columns(list(df.columns))
    odds_wide = df.loc[matches.index, list(mapping)].apply(pd.to_numeric, errors="coerce")

    return ParsedFile(
        league=league,
        season=season,
        matches=matches,
        odds_wide=odds_wide,
        odds_columns=mapping,
        unrecognized_columns=unrecognized,
        n_rows_raw=n_raw,
        n_rows_unplayed=n_unplayed,
    )


def derive_result(home: pd.Series, away: pd.Series) -> pd.Series:
    """Le résultat est recalculé depuis les scores : la colonne FTR sert seulement de contrôle."""
    out = pd.Series("D", index=home.index, dtype="string")
    out[home > away] = "H"
    out[home < away] = "A"
    return out


def make_event_id(league: str, date: pd.Timestamp, home_id: str, away_id: str) -> str:
    """Identifiant déterministe : même match => même id, à chaque reconstruction."""
    key = f"{league}|{date:%Y-%m-%d}|{home_id}|{away_id}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def odds_to_long(parsed: ParsedFile, event_ids: pd.Series) -> pd.DataFrame:
    """Passe les cotes du format large (une colonne par book x issue) au format long."""
    if not parsed.odds_columns:
        return pd.DataFrame(
            columns=[
                "event_id",
                "bookmaker",
                "market",
                "timing",
                "outcome",
                "decimal_odds",
                "source_column",
            ]
        )
    wide = parsed.odds_wide.copy()
    wide["event_id"] = event_ids.values
    long = wide.melt(id_vars="event_id", var_name="source_column", value_name="decimal_odds")
    long = long.dropna(subset=["decimal_odds"])
    meta = pd.DataFrame.from_dict(
        parsed.odds_columns, orient="index", columns=["bookmaker", "timing", "outcome"]
    )
    long = long.join(meta, on="source_column")
    long["market"] = "1x2"
    return long
