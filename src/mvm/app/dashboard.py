"""Dashboard « Model vs Market » (démo d'entretien).

Lancement : make app   (ou : PYTHONPATH=src streamlit run src/mvm/app/dashboard.py)
Pré-requis : make data, make predict, make backtest (et make test-backtest pour la page test).
Variable d'environnement optionnelle MVM_CONFIG pour pointer vers une autre config.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mvm.core import metrics
from mvm.core.io import load_config, read_table
from mvm.core.market import market_probabilities, odds_matrix

OUTCOMES = ("H", "D", "A")
LABELS = {"H": "Victoire domicile", "D": "Nul", "A": "Victoire extérieur"}
PCOLS = [f"p_{o}" for o in OUTCOMES]
# Palette catégorielle validée (ordre fixe) : modèle, marché, résultat / 3e série
C_MODEL, C_MARKET, C_THIRD = "#2a78d6", "#eb6834", "#1baf7a"
GRID = "rgba(128,128,128,0.18)"
MODEL_NAMES = {
    "ens": "Combinaison Elo + DC",
    "elo": "Elo",
    "dc": "Dixon-Coles",
    "base": "Fréquences de base",
}

st.set_page_config(page_title="Model vs Market", layout="wide")


# --- données ----------------------------------------------------------------------------------


@st.cache_resource
def config() -> dict:
    return load_config(os.environ.get("MVM_CONFIG", "configs/foot.yaml"))


@st.cache_data(show_spinner="Chargement des matchs et des cotes…")
def load_data() -> dict[str, pd.DataFrame]:
    cfg = config()
    proc = Path(cfg["paths"]["processed"])
    events = read_table(proc / "events.parquet")
    odds = read_table(proc / "odds.parquet")
    preds = read_table(proc / "predictions.parquet")
    tpath = proc / "teams.parquet"
    if tpath.exists():
        names = read_table(tpath).drop_duplicates("team_id").set_index("team_id")["canonical_name"]
    else:
        names = pd.Series(dtype="string")
    ev = events.assign(
        home=events.home_team_id.map(names).fillna(events.home_team_id),
        away=events.away_team_id.map(names).fillna(events.away_team_id),
    )
    ref = cfg["evaluation"]["reference"]
    close = market_probabilities(odds, ref["bookmaker"], ref["timing"], ref["margin_method"])
    close = close[PCOLS].add_prefix("mkt_")
    pre = odds_matrix(odds, cfg["betting"]["bet_bookmaker"], cfg["betting"]["bet_timing"])
    pre = pre.add_prefix("odds_")
    return {"events": ev.set_index("event_id"), "preds": preds, "close": close, "pre": pre}


def split_seasons(split: str) -> tuple[str, str]:
    ev = config()["evaluation"]
    return {
        "Réglage (2012-2017)": (ev["eval_first"], ev["tuning_last"]),
        "Test (2017-2025)": (ev["test_first"], ev["test_last"]),
    }[split]


def model_frame(model: str) -> pd.DataFrame:
    d = load_data()
    p = d["preds"][d["preds"].model == model].set_index("event_id")[[*PCOLS, "outcome_idx"]]
    df = d["events"].join(p, how="inner").join(d["close"], how="inner").join(d["pre"])
    y = df["outcome_idx"].to_numpy().astype(int)
    df["ll_model"] = metrics.log_loss(df[PCOLS].to_numpy(), y)
    df["ll_market"] = metrics.log_loss(df[[f"mkt_{c}" for c in PCOLS]].to_numpy(), y)
    return df.sort_values("kickoff_date")


def style(fig: go.Figure, height: int = 380, **kw) -> go.Figure:
    layout = dict(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.08, x=0),
        hovermode="x unified",
    )
    fig.update_layout(**{**layout, **kw})
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def reports_dir() -> Path:
    return Path(config()["paths"]["reports"])


# --- pages ------------------------------------------------------------------------------------


def page_matches():
    st.header("Modèle contre marché, match par match")
    st.caption(
        "Probabilités du modèle (prédites avant le match, sans voir le résultat) et du "
        "marché (clôture Pinnacle, marge retirée par la méthode de Shin)."
    )
    c1, c2, c3, c4 = st.columns(4)
    model = c1.selectbox("Modèle", list(MODEL_NAMES)[:3], format_func=MODEL_NAMES.get)
    df = model_frame(model)
    league = c2.selectbox("Ligue", sorted(df.league.unique()))
    seasons = sorted(df[df.league == league].season.unique(), reverse=True)
    season = c3.selectbox("Saison", seasons)
    sub = df[(df.league == league) & (df.season == season)]
    teams = ["Toutes"] + sorted(set(sub.home) | set(sub.away))
    team = c4.selectbox("Équipe", teams)
    if team != "Toutes":
        sub = sub[(sub.home == team) | (sub.away == team)]
    if sub.empty:
        st.info("Aucun match pour cette sélection.")
        return

    k = st.columns(4)
    k[0].metric("Matchs", f"{len(sub)}")
    k[1].metric("Log-loss modèle", f"{sub.ll_model.mean():.4f}")
    k[2].metric("Log-loss marché", f"{sub.ll_market.mean():.4f}")
    diff = sub.ll_model.mean() - sub.ll_market.mean()
    k[3].metric(
        "Écart (modèle − marché)",
        f"{diff:+.4f}",
        help="Négatif = le modèle fait mieux que le marché sur ces matchs.",
    )

    cum = (sub.ll_model - sub.ll_market).cumsum()
    fig = go.Figure(
        go.Scatter(
            x=sub.kickoff_date,
            y=cum,
            mode="lines",
            line=dict(color=C_MODEL, width=2),
            name="Écart cumulé",
            hovertemplate="%{x|%d/%m/%Y} : %{y:+.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_color="gray", line_width=1)
    st.plotly_chart(
        style(
            fig, 260, title="Log-loss cumulée : modèle − marché (au-dessus de 0, le marché gagne)"
        ),
        width="stretch",
    )

    table = sub.assign(
        Date=sub.kickoff_date.dt.strftime("%d/%m/%Y"),
        Match=sub.home + " – " + sub.away,
        Score=sub.home_score.astype(str) + "-" + sub.away_score.astype(str),
        **{f"Modèle {o}": sub[f"p_{o}"] for o in OUTCOMES},
        **{f"Marché {o}": sub[f"mkt_p_{o}"] for o in OUTCOMES},
    )
    show = (
        ["Date", "Match", "Score"]
        + [f"Modèle {o}" for o in OUTCOMES]
        + [f"Marché {o}" for o in OUTCOMES]
    )
    st.dataframe(
        table[show].iloc[::-1],
        hide_index=True,
        width="stretch",
        column_config={c: st.column_config.NumberColumn(format="percent") for c in show[3:]},
    )

    st.subheader("Zoom sur un match")
    labels = sub.kickoff_date.dt.strftime("%d/%m/%Y") + " · " + sub.home + " – " + sub.away
    pick = st.selectbox("Match", sub.index[::-1], format_func=lambda i: labels[i])
    m = sub.loc[pick]
    fig = go.Figure()
    for name, cols, color in (
        ("Modèle", PCOLS, C_MODEL),
        ("Marché", [f"mkt_{c}" for c in PCOLS], C_MARKET),
    ):
        fig.add_bar(
            x=[LABELS[o] for o in OUTCOMES],
            y=m[cols].to_numpy(float),
            name=name,
            marker_color=color,
            texttemplate="%{y:.0%}",
            textposition="outside",
            hovertemplate="%{x} : %{y:.1%}<extra>" + name + "</extra>",
        )
    st.plotly_chart(
        style(
            fig,
            340,
            barmode="group",
            yaxis_tickformat=".0%",
            title=f"{m.home} {m.home_score}-{m.away_score} {m.away} "
            f"(résultat : {LABELS[OUTCOMES[int(m.outcome_idx)]]})",
        ),
        width="stretch",
    )
    if not np.isnan(m.get("odds_H", np.nan)):
        cols = st.columns(3)
        for c, o in zip(cols, OUTCOMES, strict=True):
            ev_ = m[f"p_{o}"] * m[f"odds_{o}"] - 1
            c.metric(f"Cote pré-match {o}", f"{m[f'odds_{o}']:.2f}")
            c.caption(f"Edge selon le modèle : {ev_:+.1%}")


def page_calibration():
    st.header("Calibration")
    st.caption(
        "Quand on annonce 30 %, l'événement arrive-t-il 30 % du temps ? Points sur la "
        "diagonale = probabilités honnêtes. Barres d'erreur : intervalle binomial à 95 %."
    )
    c1, c2, c3 = st.columns(3)
    model = c1.selectbox("Modèle", list(MODEL_NAMES)[:3], format_func=MODEL_NAMES.get)
    split = c2.selectbox("Période", ["Réglage (2012-2017)", "Test (2017-2025)"])
    outcome = c3.selectbox("Issue", OUTCOMES, format_func=LABELS.get)
    first, last = split_seasons(split)
    df = model_frame(model)
    df = df[(df.season >= first) & (df.season <= last)]
    hit = (df.outcome_idx == OUTCOMES.index(outcome)).astype(float)

    fig = go.Figure()
    fig.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        line=dict(color="gray", width=1, dash="dot"),
        name="Calibration parfaite",
        hoverinfo="skip",
    )
    rows = []
    for name, col, color in (
        ("Modèle", f"p_{outcome}", C_MODEL),
        ("Marché", f"mkt_p_{outcome}", C_MARKET),
    ):
        t = metrics.reliability_table(df[col], hit, 10)
        fig.add_scatter(
            x=t.mean_pred,
            y=t.freq,
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=2),
            marker=dict(size=8),
            error_y=dict(
                type="data",
                symmetric=False,
                color=color,
                thickness=1,
                array=t.ci_high - t.freq,
                arrayminus=t.freq - t.ci_low,
            ),
            customdata=t.n,
            hovertemplate="prédit %{x:.1%} · observé %{y:.1%} · %{customdata} matchs"
            "<extra>" + name + "</extra>",
        )
        rows.append(
            {
                "Source": name,
                "ECE": metrics.ece(df[col], hit),
                "Prédit moyen": df[col].mean(),
                "Observé": hit.mean(),
            }
        )
    lim = float(max(df[f"p_{outcome}"].max(), df[f"mkt_p_{outcome}"].max())) + 0.02
    st.plotly_chart(
        style(
            fig,
            460,
            xaxis_title="Probabilité annoncée",
            yaxis_title="Fréquence observée",
            xaxis_tickformat=".0%",
            yaxis_tickformat=".0%",
            xaxis_range=[0, lim],
            yaxis_range=[0, lim],
            hovermode="closest",
        ),
        width="stretch",
    )
    st.dataframe(
        pd.DataFrame(rows).style.format(
            {"ECE": "{:.4f}", "Prédit moyen": "{:.1%}", "Observé": "{:.1%}"}
        ),
        hide_index=True,
    )


def page_backtest():
    st.header("Backtest de value betting")
    st.caption(
        "Paris au prix Pinnacle pré-match quand l'edge selon le modèle dépasse 5 %, mise "
        "quart de Kelly. CLV = cote prise × probabilité de clôture − 1 : l'indicateur qui "
        "sépare le talent de la chance."
    )
    split = st.selectbox(
        "Période",
        ["tuning", "test"],
        format_func={"tuning": "Réglage (2012-2017)", "test": "Test (2017-2025)"}.get,
    )
    rep = reports_dir()
    paths = {k: rep / f"backtest_{split}_{k}.csv" for k in ("summary", "bets", "bankroll")}
    if not all(p.exists() for p in paths.values()):
        target = "backtest" if split == "tuning" else "test-backtest"
        st.warning(f"Résultats absents : lancer `make {target}`.")
        return
    summary = pd.read_csv(paths["summary"])
    bets = pd.read_csv(paths["bets"], parse_dates=["week"])
    bank = pd.read_csv(paths["bankroll"], parse_dates=["week"])

    main = summary.iloc[0]
    k = st.columns(4)
    k[0].metric("Paris", f"{int(main.n_bets):,}".replace(",", " "))
    k[1].metric("ROI (mise fixe)", f"{main.roi_flat:+.1%}")
    k[1].caption(f"IC 95 % [{main.roi_ci_low:+.1%} ; {main.roi_ci_high:+.1%}]")
    k[2].metric("CLV moyenne", f"{main.clv:+.2%}")
    k[2].caption(f"IC 95 % [{main.clv_ci_low:+.2%} ; {main.clv_ci_high:+.2%}]")
    k[3].metric("Capital final (¼ Kelly)", f"{main.kelly_return:+.1%}")

    a, b = st.columns(2)
    bets = bets.sort_values("week")
    x = np.arange(1, len(bets) + 1)
    fig = go.Figure()
    fig.add_scatter(
        x=x,
        y=bets.flat_return.cumsum(),
        mode="lines",
        name="Profit réel (unités)",
        line=dict(color=C_MODEL, width=2),
    )
    fig.add_scatter(
        x=x,
        y=bets.clv.cumsum(),
        mode="lines",
        name="Profit attendu selon la CLV",
        line=dict(color=C_MARKET, width=2),
    )
    fig.add_hline(y=0, line_color="gray", line_width=1)
    a.plotly_chart(
        style(
            fig,
            360,
            title="Profit cumulé à mise fixe : bruité vs CLV",
            xaxis_title="Nombre de paris",
            yaxis_title="Unités",
        ),
        width="stretch",
    )
    fig = go.Figure(
        go.Scatter(
            x=bank.week,
            y=bank.bankroll,
            mode="lines",
            line=dict(color=C_MODEL, width=2),
            name="Capital",
        )
    )
    b.plotly_chart(
        style(
            fig,
            360,
            title="Capital avec ¼ Kelly (échelle log)",
            yaxis_type="log",
            yaxis_title="Capital (départ 1 000)",
        ),
        width="stretch",
    )

    st.subheader("Toutes les stratégies et les contrôles")
    s = summary.copy()
    fig = go.Figure(
        go.Bar(
            y=s.strategy,
            x=s.clv,
            orientation="h",
            marker_color=[
                C_THIRD if "oracle" in t else C_MARKET if "CONTRÔLE" in t else C_MODEL
                for t in s.strategy
            ],
            error_x=dict(
                type="data",
                symmetric=False,
                array=s.clv_ci_high - s.clv,
                arrayminus=s.clv - s.clv_ci_low,
                color="gray",
                thickness=1,
            ),
            hovertemplate="%{y} : CLV %{x:+.2%}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color="gray", line_width=1)
    st.plotly_chart(
        style(
            fig,
            420,
            xaxis_tickformat=".0%",
            title="CLV moyenne par stratégie (vert : contrôle positif, orange : contrôle négatif)",
            yaxis_autorange="reversed",
            hovermode="closest",
        ),
        width="stretch",
    )
    cols = {
        "strategy": "Stratégie",
        "n_bets": "Paris",
        "hit_rate": "Taux de réussite",
        "avg_odds": "Cote moyenne",
        "roi_flat": "ROI",
        "clv": "CLV",
        "kelly_return": "Capital ¼ Kelly",
        "max_drawdown": "Drawdown max",
    }
    st.dataframe(
        s[list(cols)]
        .rename(columns=cols)
        .style.format(
            {
                "Taux de réussite": "{:.1%}",
                "Cote moyenne": "{:.2f}",
                "ROI": "{:+.1%}",
                "CLV": "{:+.2%}",
                "Capital ¼ Kelly": "{:+.1%}",
                "Drawdown max": "{:.0%}",
            }
        ),
        hide_index=True,
        width="stretch",
    )

    if "league" in bets.columns:
        lg = (
            bets.groupby("league")
            .agg(paris=("clv", "size"), clv=("clv", "mean"), roi=("flat_return", "mean"))
            .reset_index()
        )
        fig = go.Figure()
        fig.add_bar(x=lg.league, y=lg.clv, name="CLV", marker_color=C_MARKET)
        fig.add_bar(x=lg.league, y=lg.roi, name="ROI", marker_color=C_MODEL)
        st.plotly_chart(
            style(
                fig,
                340,
                barmode="group",
                yaxis_tickformat=".0%",
                title="Par ligue (exploratoire) : un ROI positif avec une CLV "
                "négative, c'est de la chance",
            ),
            width="stretch",
        )


def page_method():
    st.header("Méthodologie et résultats")
    st.caption(
        "Protocole figé avant toute modélisation, prédictions écrites avant chaque test, "
        "y compris celles qui ont échoué."
    )
    docs = {
        "Protocole": "docs/protocol.md",
        "Elo": "docs/results_elo.md",
        "Dixon-Coles et combinaison": "docs/results_dc.md",
        "Paris": "docs/results_betting.md",
        "Données": "data/README.md",
    }
    tabs = st.tabs(list(docs))
    for tab, path in zip(tabs, docs.values(), strict=True):
        with tab:
            p = Path(path)
            st.markdown(p.read_text(encoding="utf-8") if p.exists() else f"_{path} introuvable_")


PAGES = {
    "Match par match": page_matches,
    "Calibration": page_calibration,
    "Backtest": page_backtest,
    "Méthodologie": page_method,
}

st.sidebar.title("Model vs Market")
st.sidebar.caption(
    "Probabilités de matchs de foot contre les cotes des bookmakers. 12 ligues, 2005-2026."
)
page = st.sidebar.radio("Page", list(PAGES))
PAGES[page]()
