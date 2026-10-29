"""Coulisses : les pages techniques (log-loss, calibration, backtest détaillé, méthodologie).

Textes affichés en anglais."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mvm.app.data import MODEL_NAMES, OUTCOMES, PCOLS, model_frame, reports_dir, split_range
from mvm.app.ui import BOOK, CHALK, MODEL, THIRD
from mvm.core import metrics

LABELS = {"H": "Home win", "D": "Draw", "A": "Away win"}
C_MODEL, C_MARKET, C_THIRD = MODEL, BOOK, THIRD
GRID = "rgba(239,238,230,0.12)"
SPLITS = {"Tuning (2012-2017)": "tuning", "Test (2017-2025)": "test"}


def strategy_en(name: str) -> str:
    """Les libellés du backtest sont en français dans les rapports : traduction à l'affichage."""
    for a, b in (
        ("PRINCIPALE ", "MAIN "),
        ("CONTRÔLE + oracle (clôture)", "Positive control: oracle (closing)"),
        ("CONTRÔLE - fréquences de base", "Negative control: base rates"),
    ):
        name = name.replace(a, b)
    return name


def split_seasons(label: str) -> tuple[str, str]:
    return split_range(SPLITS[label])


def style(fig: go.Figure, height: int = 380, **kw) -> go.Figure:
    layout = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Barlow", color=CHALK),
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.08, x=0),
        hovermode="x unified",
    )
    fig.update_layout(**{**layout, **kw})
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def page_matches():
    st.subheader("Model vs market, match by match")
    st.caption(
        "Model probabilities (predicted before the match, without seeing the result) and "
        "market probabilities (closing price, margin removed with Shin's method): Pinnacle up "
        "to 2024-25, market average for the live season. 2025-26 is hidden (v2 holdout)."
    )
    c1, c2, c3, c4 = st.columns(4)
    model = c1.selectbox(
        "Modèle", list(MODEL_NAMES)[:3], format_func=MODEL_NAMES.get, key="lab_m_model"
    )
    df = model_frame(model)
    league = c2.selectbox("Ligue", sorted(df.league.unique()), key="lab_m_league")
    seasons = sorted(df[df.league == league].season.unique(), reverse=True)
    season = c3.selectbox("Saison", seasons, key="lab_m_season")
    sub = df[(df.league == league) & (df.season == season)]
    teams = ["All"] + sorted(set(sub.home) | set(sub.away))
    team = c4.selectbox("Équipe", teams, key="lab_m_team")
    if team != "All":
        sub = sub[(sub.home == team) | (sub.away == team)]
    if sub.empty:
        st.info("No match for this selection.")
        return

    k = st.columns(4)
    k[0].metric("Matches", f"{len(sub)}")
    k[1].metric("Model log-loss", f"{sub.ll_model.mean():.4f}")
    k[2].metric("Market log-loss", f"{sub.ll_market.mean():.4f}")
    diff = sub.ll_model.mean() - sub.ll_market.mean()
    k[3].metric(
        "Gap (model - market)",
        f"{diff:+.4f}",
        help="Negative = the model beats the market on these matches.",
    )

    cum = (sub.ll_model - sub.ll_market).cumsum()
    fig = go.Figure(
        go.Scatter(
            x=sub.kickoff_date,
            y=cum,
            mode="lines",
            line=dict(color=C_MODEL, width=2),
            name="Cumulative gap",
            hovertemplate="%{x|%d/%m/%Y} : %{y:+.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_color="rgba(239,238,230,.4)", line_width=1)
    st.plotly_chart(
        style(fig, 260, title="Cumulative log-loss, model - market (above 0, the market wins)"),
        width="stretch",
    )

    table = sub.assign(
        Date=sub.kickoff_date.dt.strftime("%d/%m/%Y"),
        Match=sub.home + " – " + sub.away,
        Score=sub.home_score.astype(str) + "-" + sub.away_score.astype(str),
        **{f"Model {o}": sub[f"p_{o}"] for o in OUTCOMES},
        **{f"Market {o}": sub[f"mkt_p_{o}"] for o in OUTCOMES},
    )
    show = (
        ["Date", "Match", "Score"]
        + [f"Model {o}" for o in OUTCOMES]
        + [f"Market {o}" for o in OUTCOMES]
    )
    st.dataframe(
        table[show].iloc[::-1],
        hide_index=True,
        width="stretch",
        column_config={c: st.column_config.NumberColumn(format="percent") for c in show[3:]},
    )

    st.subheader("Zoom on one match")
    labels = sub.kickoff_date.dt.strftime("%d/%m/%Y") + " · " + sub.home + " – " + sub.away
    pick = st.selectbox("Match", sub.index[::-1], format_func=lambda i: labels[i], key="lab_m_pick")
    m = sub.loc[pick]
    fig = go.Figure()
    for name, cols, color in (
        ("Model", PCOLS, C_MODEL),
        ("Market", [f"mkt_{c}" for c in PCOLS], C_MARKET),
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
            f"(result: {LABELS[OUTCOMES[int(m.outcome_idx)]]})",
        ),
        width="stretch",
    )
    if not np.isnan(m.get("odds_H", np.nan)):
        cols = st.columns(3)
        for c, o in zip(cols, OUTCOMES, strict=True):
            ev_ = m[f"p_{o}"] * m[f"odds_{o}"] - 1
            c.metric(f"Pre-match odds {o}", f"{m[f'odds_{o}']:.2f}")
            c.caption(f"Model edge: {ev_:+.1%}")


def page_calibration():
    st.subheader("Calibration")
    st.caption(
        "When we say 30%, does it happen 30% of the time? Points on the diagonal = honest "
        "probabilities. Error bars: 95% binomial interval."
    )
    c1, c2, c3 = st.columns(3)
    model = c1.selectbox(
        "Modèle", list(MODEL_NAMES)[:3], format_func=MODEL_NAMES.get, key="lab_c_model"
    )
    split = c2.selectbox("Période", list(SPLITS), key="lab_c_split")
    outcome = c3.selectbox("Issue", OUTCOMES, format_func=LABELS.get, key="lab_c_outcome")
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
        name="Perfect calibration",
        hoverinfo="skip",
    )
    rows = []
    for name, col, color in (
        ("Model", f"p_{outcome}", C_MODEL),
        ("Market", f"mkt_p_{outcome}", C_MARKET),
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
                "Mean forecast": df[col].mean(),
                "Observed": hit.mean(),
            }
        )
    lim = float(max(df[f"p_{outcome}"].max(), df[f"mkt_p_{outcome}"].max())) + 0.02
    st.plotly_chart(
        style(
            fig,
            460,
            xaxis_title="Forecast probability",
            yaxis_title="Observed frequency",
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
            {"ECE": "{:.4f}", "Mean forecast": "{:.1%}", "Observed": "{:.1%}"}
        ),
        hide_index=True,
    )


def page_backtest():
    st.subheader("Value-betting backtest")
    st.caption(
        "Bets at Pinnacle's pre-match price when the model's edge exceeds 5%, quarter-Kelly "
        "stakes. CLV = odds taken x closing probability - 1: the statistic that separates "
        "skill from luck."
    )
    split = st.selectbox(
        "Period",
        ["tuning", "test"],
        format_func={"tuning": "Tuning (2012-2017)", "test": "Test (2017-2025)"}.get,
        key="lab_b_split",
    )
    rep = reports_dir()
    paths = {k: rep / f"backtest_{split}_{k}.csv" for k in ("summary", "bets", "bankroll")}
    if not all(p.exists() for p in paths.values()):
        target = "backtest" if split == "tuning" else "test-backtest"
        st.warning(f"No results yet: run `make {target}`.")
        return
    summary = pd.read_csv(paths["summary"])
    bets = pd.read_csv(paths["bets"], parse_dates=["week"])
    bank = pd.read_csv(paths["bankroll"], parse_dates=["week"])

    main = summary.iloc[0]
    k = st.columns(4)
    k[0].metric("Bets", f"{int(main.n_bets):,}")
    k[1].metric("ROI (flat stakes)", f"{main.roi_flat:+.1%}")
    k[1].caption(f"95% CI [{main.roi_ci_low:+.1%}, {main.roi_ci_high:+.1%}]")
    k[2].metric("Mean CLV", f"{main.clv:+.2%}")
    k[2].caption(f"95% CI [{main.clv_ci_low:+.2%}, {main.clv_ci_high:+.2%}]")
    k[3].metric("Final bankroll (¼ Kelly)", f"{main.kelly_return:+.1%}")

    a, b = st.columns(2)
    bets = bets.sort_values("week")
    x = np.arange(1, len(bets) + 1)
    fig = go.Figure()
    fig.add_scatter(
        x=x,
        y=bets.flat_return.cumsum(),
        mode="lines",
        name="Actual profit (units)",
        line=dict(color=C_MODEL, width=2),
    )
    fig.add_scatter(
        x=x,
        y=bets.clv.cumsum(),
        mode="lines",
        name="Profit expected from CLV",
        line=dict(color=C_MARKET, width=2),
    )
    fig.add_hline(y=0, line_color="rgba(239,238,230,.4)", line_width=1)
    a.plotly_chart(
        style(
            fig,
            360,
            title="Cumulative flat-stake profit: noisy vs CLV",
            xaxis_title="Number of bets",
            yaxis_title="Units",
        ),
        width="stretch",
    )
    fig = go.Figure(
        go.Scatter(
            x=bank.week,
            y=bank.bankroll,
            mode="lines",
            line=dict(color=C_MODEL, width=2),
            name="Bankroll",
        )
    )
    b.plotly_chart(
        style(
            fig,
            360,
            title="Bankroll with ¼ Kelly (log scale)",
            yaxis_type="log",
            yaxis_title="Bankroll (start 1,000)",
        ),
        width="stretch",
    )

    st.subheader("All strategies and controls")
    s = summary.copy()
    s["strategy"] = s.strategy.map(strategy_en)
    fig = go.Figure(
        go.Bar(
            y=s.strategy,
            x=s.clv,
            orientation="h",
            marker_color=[
                C_THIRD if "oracle" in t else C_MARKET if "control" in t else C_MODEL
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
    fig.add_vline(x=0, line_color="rgba(239,238,230,.4)", line_width=1)
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
        "strategy": "Strategy",
        "n_bets": "Bets",
        "hit_rate": "Hit rate",
        "avg_odds": "Average odds",
        "roi_flat": "ROI",
        "clv": "CLV",
        "kelly_return": "¼ Kelly bankroll",
        "max_drawdown": "Max drawdown",
    }
    st.dataframe(
        s[list(cols)]
        .rename(columns=cols)
        .style.format(
            {
                "Hit rate": "{:.1%}",
                "Average odds": "{:.2f}",
                "ROI": "{:+.1%}",
                "CLV": "{:+.2%}",
                "¼ Kelly bankroll": "{:+.1%}",
                "Max drawdown": "{:.0%}",
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
                title="By league (exploratory): a positive ROI with a negative CLV is luck",
            ),
            width="stretch",
        )


def page_method():
    st.subheader("Method and results")
    st.caption(
        "Protocol frozen before any modelling, expectations written before every test, "
        "including the ones that failed."
    )
    docs = {
        "Protocol": "docs/protocol.md",
        "Elo": "docs/results_elo.md",
        "Dixon-Coles and blend": "docs/results_dc.md",
        "Betting": "docs/results_betting.md",
        "Live season": "docs/live_2026_27.md",
        "Data": "data/README.md",
    }
    tabs = st.tabs(list(docs))
    for tab, path in zip(tabs, docs.values(), strict=True):
        with tab:
            p = Path(path)
            st.markdown(p.read_text(encoding="utf-8") if p.exists() else f"_{path} not found_")


SUBPAGES = {
    "Log-loss match by match": page_matches,
    "Calibration": page_calibration,
    "Detailed backtest": page_backtest,
    "Method": page_method,
}


def page_lab():
    st.html(
        '<p class="lede">The same results, the technical way: log-loss, calibration, '
        "confidence intervals, backtest controls and the pre-registered protocol.</p>"
    )
    with st.container(key="subnav"):
        sub = st.radio(
            "View", list(SUBPAGES), horizontal=True, label_visibility="collapsed", key="lab_view"
        )
    SUBPAGES[sub]()
