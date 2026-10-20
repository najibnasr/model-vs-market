"""Contrôles qualité et rapport de couverture.

Deux niveaux :
- erreurs bloquantes (le build s'arrête) : identifiants dupliqués, dates illisibles, équipe qui
  joue contre elle-même. Si ça arrive, le pipeline a un bug, on ne continue pas.
- anomalies tolérées mais tracées : saison incomplète (ex. Ligue 1 2019/20 arrêtée par le Covid),
  FTR incohérent avec le score, cotes aberrantes. Les cotes aberrantes sont mises en
  QUARANTAINE plutôt que corrigées : on ne fabrique jamais une cote.

Pourquoi être strict sur les cotes : dans un backtest de value betting, une cote périmée ou mal
saisie (ex. 9.0 au lieu de 1.9) ressemble exactement à une énorme value. Le modèle "la trouve",
le pari "gagne", et le backtest affiche un edge qui n'est qu'une faute de frappe.
"""

from __future__ import annotations

import pandas as pd

KEY = ["event_id", "bookmaker", "timing"]


class DataQualityError(RuntimeError):
    pass


def hard_checks(events: pd.DataFrame) -> None:
    errors = []
    if events["event_id"].duplicated().any():
        errors.append(f"{int(events['event_id'].duplicated().sum())} event_id dupliqués")
    if events["kickoff_date"].isna().any():
        errors.append(f"{int(events['kickoff_date'].isna().sum())} dates illisibles")
    if (events["home_team_id"] == events["away_team_id"]).any():
        errors.append("équipe à domicile = équipe à l'extérieur")
    if errors:
        raise DataQualityError(" ; ".join(errors))


def overround(odds: pd.DataFrame) -> pd.DataFrame:
    """Somme des probabilités implicites brutes (1/cote) par (match, book, moment).

    Un overround de 1.05 signifie que le book "vend" 105 % de probabilité : la marge est d'environ
    5 %. Ce n'est PAS encore le retrait de marge (étape b), juste un contrôle de cohérence.
    """
    g = odds.assign(inv=1.0 / odds["decimal_odds"]).groupby(KEY)
    return g.agg(overround=("inv", "sum"), n_outcomes=("outcome", "nunique")).reset_index()


def quarantine_odds(
    odds: pd.DataFrame, n_outcomes: int, lo: float, hi: float, lo_best_price: float = 0.90
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retire les triplets (match, book, moment) inutilisables et renvoie (propres, quarantaine).

    Borne basse : un book seul ne propose jamais un overround < 1 (il perdrait de l'argent à coup
    sûr). Mais market_max prend la meilleure cote de CHAQUE issue chez des books différents :
    un overround < 1 y est possible et réel (arbitrage entre books). On lui applique donc une
    borne basse plus permissive, sinon on jetterait précisément les matchs les plus intéressants.
    """
    bad_value = odds["decimal_odds"] <= 1.0
    odds = odds.assign(reason=pd.Series(pd.NA, index=odds.index, dtype="string"))
    odds.loc[bad_value, "reason"] = "cote <= 1"

    ov = overround(odds.loc[~bad_value])
    ov["reason"] = pd.Series(pd.NA, index=ov.index, dtype="string")
    ov.loc[ov["n_outcomes"] < n_outcomes, "reason"] = "triplet incomplet"
    floor = ov["bookmaker"].map({"market_max": lo_best_price}).fillna(lo)
    out_of_range = (ov["overround"] < floor) | (ov["overround"] > hi)
    ov.loc[ov["reason"].isna() & out_of_range, "reason"] = "overround hors plage"

    odds = odds.merge(ov[KEY + ["reason"]].rename(columns={"reason": "r2"}), on=KEY, how="left")
    odds["reason"] = odds["reason"].fillna(odds["r2"])
    # un triplet avec une seule cote <= 1 part entièrement en quarantaine
    bad_keys = odds.loc[odds["reason"].notna(), KEY].drop_duplicates()
    odds = odds.drop(columns="r2").merge(bad_keys.assign(_bad=True), on=KEY, how="left")
    is_bad = odds["_bad"].fillna(False).astype(bool)
    odds["reason"] = odds["reason"].fillna("triplet contaminé")

    clean = odds.loc[~is_bad].drop(columns=["reason", "_bad"]).reset_index(drop=True)
    quarantined = odds.loc[is_bad].drop(columns="_bad").reset_index(drop=True)
    return clean, quarantined


def soft_checks(events: pd.DataFrame, raw_results: pd.Series) -> list[dict]:
    """Anomalies tracées dans le rapport, sans bloquer."""
    issues: list[dict] = []

    mism = raw_results.notna() & (raw_results.values != events["result"].values)
    if mism.any():
        issues.append({"check": "FTR incohérent avec le score", "n": int(mism.sum())})

    long = pd.concat(
        [
            events[["kickoff_date", "home_team_id"]].rename(columns={"home_team_id": "t"}),
            events[["kickoff_date", "away_team_id"]].rename(columns={"away_team_id": "t"}),
        ]
    )
    dup_day = long.duplicated(["kickoff_date", "t"]).sum()
    if dup_day:
        issues.append({"check": "équipe jouant deux fois le même jour", "n": int(dup_day)})

    for (league, season), g in events.groupby(["league", "season"]):
        n_teams = pd.concat([g.home_team_id, g.away_team_id]).nunique()
        expected = n_teams * (n_teams - 1)
        if len(g) != expected:
            issues.append(
                {
                    "check": "saison incomplète ou format atypique",
                    "league": league,
                    "season": season,
                    "n": len(g),
                    "expected": expected,
                }
            )
        start = int(season[:4])
        outside = ~g.kickoff_date.between(f"{start}-07-01", f"{start + 1}-07-31")
        if outside.any():
            issues.append(
                {
                    "check": "date hors fenêtre de saison",
                    "league": league,
                    "season": season,
                    "n": int(outside.sum()),
                }
            )
    return issues


def coverage_report(events: pd.DataFrame, odds: pd.DataFrame, n_outcomes: int) -> pd.DataFrame:
    """Par ligue x saison : part des matchs ayant un triplet complet pour chaque (book, moment)."""
    ov = overround(odds)
    ov = ov[ov.n_outcomes == n_outcomes].merge(
        events[["event_id", "league", "season"]], on="event_id"
    )
    base = events.groupby(["league", "season", "tier"]).agg(
        n_matches=("event_id", "size"),
        share_with_time=("kickoff_time", lambda s: s.notna().mean()),
    )
    rows = [base]
    for book in ("pinnacle", "market_avg", "market_max", "bet365"):
        for timing in ("pre_match", "closing"):
            sub = ov[(ov.bookmaker == book) & (ov.timing == timing)]
            agg = sub.groupby(["league", "season"]).agg(
                n=("event_id", "nunique"), med_overround=("overround", "median")
            )
            rows.append(agg["n"].rename(f"cov_{book}_{timing}"))
            if book == "pinnacle":
                rows.append(agg["med_overround"].rename(f"overround_{book}_{timing}"))
    cov = base.reset_index().set_index(["league", "season"]).join(rows[1:], how="left")
    for c in [c for c in cov.columns if c.startswith("cov_")]:
        cov[c] = (cov[c].fillna(0) / cov["n_matches"]).round(3)
    return cov.reset_index().sort_values(["league", "season"])
