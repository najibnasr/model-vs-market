"""Modèle de Dixon & Coles (1997) : buts de Poisson, correction des petits scores, décroissance
temporelle.

Le modèle. Pour un match entre h (domicile) et a (extérieur) dans la ligue L :

    buts domicile  X ~ Poisson(lambda),  log lambda = c_L + gamma_L + att_h - def_a
    buts extérieur Y ~ Poisson(mu),      log mu     = c_L           + att_a - def_h

- att_i : force offensive de l'équipe i ; def_i : force défensive (plus grand = meilleure défense).
- c_L : niveau de buts moyen de la ligue ; gamma_L : avantage du terrain de la ligue.

Pourquoi c'est mieux qu'Elo, en théorie :
1. Il modélise les BUTS, pas seulement le résultat : un 3-0 et un 1-0 informent différemment, et
   le nul sort naturellement du modèle (P(X = Y)), sans couche ajoutée.
2. Attaque et défense sont séparées : une équipe qui marque beaucoup ET encaisse beaucoup n'est
   pas résumée par un seul nombre.
3. Décroissance temporelle : chaque match passé pèse exp(-xi * âge en jours). L'avantage du
   terrain est ré-estimé en permanence sur les matchs récents : c'est la réponse directe au biais
   domicile qu'Elo a montré à l'étape (c).

La correction de Dixon-Coles. Deux Poisson indépendants sous-estiment les 0-0 et 1-1 et
sur-estiment les 1-0 et 0-1 (les équipes "jouent le score"). On multiplie la probabilité des
quatre petits scores par un facteur tau qui dépend d'un paramètre rho (rho < 0 en pratique).

Choix de mise en oeuvre (à défendre en entretien) :
- Un modèle PAR PAYS, ses deux divisions ajustées ensemble : un promu garde ses forces, et les
  montées/descentes relient les échelles des deux divisions.
- Pénalité L2 (prior gaussien centré sur 0) sur att et def : les équipes avec peu de matchs
  (promus) sont tirées vers la moyenne de leur ligue au lieu de recevoir des valeurs extrêmes.
- Ré-estimation chaque semaine, en partant de la solution de la semaine précédente (warm start).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10


def time_decay_weights(days_ago: np.ndarray, xi: float) -> np.ndarray:
    """Poids de chaque match passé selon son âge (en jours) : 1 aujourd'hui, puis décroissance
    exponentielle. Demi-vie = ln(2) / xi (xi = 0.0065 -> environ 107 jours).

    À IMPLÉMENTER PAR NAJIB (voir le message de Claude pour la ligne exacte).
    """
    raise NotImplementedError("time_decay_weights() : à implémenter")


def dc_tau(x, y, lam, mu, rho):
    """Facteur de correction de Dixon-Coles, vectorisé. Vaut 1 sauf pour 0-0, 0-1, 1-0, 1-1."""
    tau = np.ones(np.broadcast(x, y, lam, mu).shape)
    tau = np.where((x == 0) & (y == 0), 1.0 - lam * mu * rho, tau)
    tau = np.where((x == 0) & (y == 1), 1.0 + lam * rho, tau)
    tau = np.where((x == 1) & (y == 0), 1.0 + mu * rho, tau)
    tau = np.where((x == 1) & (y == 1), 1.0 - rho, tau)
    return tau


def outcome_probs(lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    """(n, 3) probabilités H, D, A à partir des intensités, avec la correction tau."""
    g = np.arange(MAX_GOALS + 1)
    px = poisson.pmf(g[None, :], lam[:, None])  # (n, G)
    py = poisson.pmf(g[None, :], mu[:, None])
    m = px[:, :, None] * py[:, None, :]  # (n, G, G) : m[i, x, y]
    xs, ys = np.meshgrid(g, g, indexing="ij")
    m = m * dc_tau(xs[None], ys[None], lam[:, None, None], mu[:, None, None], rho)
    m = np.clip(m, 0.0, None)
    m /= m.sum(axis=(1, 2), keepdims=True)
    home = np.tril(np.ones((len(g), len(g))), -1)  # x > y
    away = np.triu(np.ones((len(g), len(g))), 1)  # x < y
    p_h = (m * home).sum(axis=(1, 2))
    p_a = (m * away).sum(axis=(1, 2))
    return np.column_stack([p_h, 1.0 - p_h - p_a, p_a])


class _CountryFit:
    """Paramètres et historique d'un pays (une ou deux divisions)."""

    def __init__(self):
        self.rows: list[tuple] = []  # (date, league, home, away, hg, ag)
        self.att: dict[str, float] = {}
        self.dfn: dict[str, float] = {}
        self.base: dict[str, float] = defaultdict(lambda: 0.3)  # c_L
        self.home: dict[str, float] = defaultdict(lambda: 0.25)  # gamma_L
        self.rho = -0.05
        self.n_at_fit = -1  # nombre de matchs connus lors du dernier ajustement


class DixonColes:
    """Forecaster compatible avec mvm.core.splits.walk_forward."""

    def __init__(self, xi: float = 0.0065, l2: float = 3.0, max_age_days: int = 1460):
        self.xi = xi
        self.l2 = l2
        self.max_age_days = max_age_days
        self.countries: dict[str, _CountryFit] = defaultdict(_CountryFit)

    @staticmethod
    def _country(team_id: str) -> str:
        return team_id.split(":", 1)[0]

    # --- ajustement ---------------------------------------------------------------------------

    def _fit(self, cf: _CountryFit, as_of: pd.Timestamp) -> None:
        df = pd.DataFrame(cf.rows, columns=["date", "league", "home", "away", "hg", "ag"])
        age = (as_of - df["date"]).dt.days.to_numpy()
        keep = (age >= 0) & (age <= self.max_age_days)
        df, age = df[keep], age[keep]
        if len(df) < 30:
            return
        w = time_decay_weights(age.astype(float), self.xi)

        teams = sorted(set(df.home) | set(df.away))
        leagues = sorted(set(df.league))
        ti = {t: i for i, t in enumerate(teams)}
        li = {lg: i for i, lg in enumerate(leagues)}
        h = df.home.map(ti).to_numpy()
        a = df.away.map(ti).to_numpy()
        lg = df.league.map(li).to_numpy()
        x = df.hg.to_numpy(float)
        y = df.ag.to_numpy(float)
        nt, nl = len(teams), len(leagues)

        def unpack(theta):
            return (
                theta[:nt],
                theta[nt : 2 * nt],
                theta[2 * nt : 2 * nt + nl],
                theta[2 * nt + nl : 2 * nt + 2 * nl],
                theta[-1],
            )

        def loss_grad(theta):
            att, dfn, base, home, rho = unpack(theta)
            eh = base[lg] + home[lg] + att[h] - dfn[a]
            ea = base[lg] + att[a] - dfn[h]
            lam, mu = np.exp(eh), np.exp(ea)
            tau = np.clip(dc_tau(x, y, lam, mu, rho), 1e-10, None)
            ll = w * (x * eh - lam + y * ea - mu + np.log(tau))
            # dérivées de log(tau) par rapport à eta_h, eta_a et rho
            z00, z01 = (x == 0) & (y == 0), (x == 0) & (y == 1)
            z10, z11 = (x == 1) & (y == 0), (x == 1) & (y == 1)
            dth = np.where(z00, -lam * mu * rho, 0) + np.where(z01, lam * rho, 0)
            dta = np.where(z00, -lam * mu * rho, 0) + np.where(z10, mu * rho, 0)
            dtr = (
                np.where(z00, -lam * mu, 0)
                + np.where(z01, lam, 0)
                + np.where(z10, mu, 0)
                + np.where(z11, -1.0, 0)
            )
            gh = w * (x - lam + dth / tau)
            ga = w * (y - mu + dta / tau)
            g_att = np.bincount(h, gh, nt) + np.bincount(a, ga, nt)
            g_dfn = -np.bincount(a, gh, nt) - np.bincount(h, ga, nt)
            g_base = np.bincount(lg, gh + ga, nl)
            g_home = np.bincount(lg, gh, nl)
            g_rho = (w * dtr / tau).sum()
            loss = -ll.sum() + 0.5 * self.l2 * (att @ att + dfn @ dfn)
            grad = -np.concatenate([g_att, g_dfn, g_base, g_home, [g_rho]])
            grad[: 2 * nt] += self.l2 * np.concatenate([att, dfn])
            return loss, grad

        theta0 = np.concatenate(
            [
                [cf.att.get(t, 0.0) for t in teams],
                [cf.dfn.get(t, 0.0) for t in teams],
                [cf.base[lg_] for lg_ in leagues],
                [cf.home[lg_] for lg_ in leagues],
                [cf.rho],
            ]
        )
        bounds = [(None, None)] * (len(theta0) - 1) + [(-0.25, 0.1)]
        res = minimize(loss_grad, theta0, jac=True, method="L-BFGS-B", bounds=bounds)
        att, dfn, base, home, rho = unpack(res.x)
        cf.att.update(zip(teams, att, strict=True))
        cf.dfn.update(zip(teams, dfn, strict=True))
        cf.base.update(zip(leagues, base, strict=True))
        cf.home.update(zip(leagues, home, strict=True))
        cf.rho = float(rho)
        cf.n_at_fit = len(cf.rows)

    # --- interface Forecaster -----------------------------------------------------------------

    def update(self, played: pd.DataFrame) -> None:
        for r in played.itertuples(index=False):
            self.countries[self._country(r.home_team_id)].rows.append(
                (
                    r.kickoff_date,
                    r.league,
                    r.home_team_id,
                    r.away_team_id,
                    r.home_score,
                    r.away_score,
                )
            )

    def predict(self, fixtures: pd.DataFrame) -> np.ndarray:
        out = np.empty((len(fixtures), 3))
        as_of = pd.Timestamp(fixtures["kickoff_date"].min())
        country = fixtures["home_team_id"].map(self._country).to_numpy()
        for c in np.unique(country):
            cf = self.countries[c]
            if cf.n_at_fit != len(cf.rows):  # de nouveaux matchs depuis le dernier ajustement
                self._fit(cf, as_of)
            sel = country == c
            fx = fixtures[sel]
            # équipe inconnue : 0 = niveau moyen de sa ligue (le prior de la pénalité L2)
            att_h = fx.home_team_id.map(cf.att).fillna(0.0).to_numpy()
            att_a = fx.away_team_id.map(cf.att).fillna(0.0).to_numpy()
            def_h = fx.home_team_id.map(cf.dfn).fillna(0.0).to_numpy()
            def_a = fx.away_team_id.map(cf.dfn).fillna(0.0).to_numpy()
            base = fx.league.map(cf.base).to_numpy(float)
            home = fx.league.map(cf.home).to_numpy(float)
            lam = np.exp(base + home + att_h - def_a)
            mu = np.exp(base + att_a - def_h)
            out[sel] = outcome_probs(lam, mu, cf.rho)
        return out

    def team_table(self, country: str) -> pd.DataFrame:
        cf = self.countries[country]
        t = pd.DataFrame({"attack": cf.att, "defence": cf.dfn})
        return t.assign(strength=t.attack + t.defence).sort_values("strength", ascending=False)
