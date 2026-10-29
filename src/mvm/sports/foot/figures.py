"""Figures du write-up (docs/research_note.md), calculées sur les vraies données.

    make figures      (pré-requis : make predict, make backtest, make test-backtest)

Écrit des PNG dans docs/figures/. Toutes les figures portent sur la période de TEST
(2017-18 à 2024-25), déjà regardée une fois : rien n'est ré-estimé ici, on ne fait que dessiner.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mvm.core import metrics  # noqa: E402
from mvm.core.io import load_config, read_table  # noqa: E402
from mvm.core.market import market_probabilities  # noqa: E402
from mvm.sports.foot.run_model import load, season_range  # noqa: E402

PCOLS = ["p_H", "p_D", "p_A"]
# Une couleur par entité, la même dans toutes les figures (palette validée, fond clair)
C = {
    "ens": "#2a78d6",
    "market": "#eb6834",
    "elo": "#6250d6",
    "dc": "#1baf7a",
    "base": "#8a8a85",
    "oracle": "#008300",
}
NAMES = {
    "base": "League base rates",
    "elo": "Elo",
    "dc": "Dixon-Coles",
    "ens": "Elo + Dixon-Coles (50/50)",
    "market": "Pinnacle closing price",
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.titlepad": 14,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.size": 10,
            "legend.frameon": False,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def test_frame(cfg: dict) -> pd.DataFrame:
    """Matchs du test, échantillon commun : prédictions des 4 modèles + clôture Pinnacle."""
    events, odds = load(cfg)
    first, last = season_range(cfg["evaluation"], "test")
    ev = events[(events.season >= first) & (events.season <= last)].set_index("event_id")
    ref = cfg["evaluation"]["reference"]
    mkt = market_probabilities(odds, ref["bookmaker"], ref["timing"], ref["margin_method"])
    df = ev[["season", "league", "result"]].join(mkt[PCOLS].add_prefix("market_"), how="inner")
    preds = read_table(Path(cfg["paths"]["processed"]) / "predictions.parquet")
    for m, g in preds.groupby("model"):
        df = df.join(g.set_index("event_id")[PCOLS].add_prefix(f"{m}_"), how="inner")
    df["y"] = df.result.map({"H": 0, "D": 1, "A": 2})
    return df


def fig_gap_closed(df: pd.DataFrame, out: Path) -> dict:
    y = df.y.to_numpy()
    ll = {s: metrics.log_loss(df[[f"{s}_{c}" for c in PCOLS]].to_numpy(), y).mean() for s in NAMES}
    share = {s: (ll["base"] - ll[s]) / (ll["base"] - ll["market"]) for s in NAMES}
    order = ["base", "elo", "dc", "ens", "market"]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ypos = np.arange(len(order))[::-1]
    ax.barh(ypos, [share[s] * 100 for s in order], color=[C[s] for s in order], height=0.62)
    for yy, s in zip(ypos, order, strict=True):
        ax.text(
            share[s] * 100 + 1.5,
            yy,
            f"{share[s]:.0%}   (log-loss {ll[s]:.4f})",
            va="center",
            color=INK,
            fontsize=9,
        )
    ax.set_yticks(ypos, [NAMES[s] for s in order])
    ax.set_xlim(0, 135)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of the information gap closed, from base rates (0%) to the market (100%)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Test seasons 2017-18 to 2024-25, {len(df):,} matches")
    fig.savefig(out / "fig1_gap_closed.png")
    plt.close(fig)
    return {"log_loss": ll, "gap_closed": share}


def _reliability(p: np.ndarray, hit: np.ndarray, n_bins: int = 12) -> pd.DataFrame:
    q = pd.qcut(p, n_bins, labels=False, duplicates="drop")
    return (
        pd.DataFrame({"p": p, "hit": hit, "q": q})
        .groupby("q")
        .agg(pred=("p", "mean"), obs=("hit", "mean"), n=("hit", "size"))
    )


def fig_calibration(df: pd.DataFrame, out: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2))
    y = df.y.to_numpy()
    for s in ("dc", "elo", "market"):
        p = np.concatenate([df[f"{s}_{c}"].to_numpy() for c in PCOLS])
        hit = np.concatenate([(y == k).astype(float) for k in range(3)])
        t = _reliability(p, hit)
        a.plot(t.pred, t.obs - t.pred, marker="o", ms=5, lw=2, color=C[s], label=NAMES[s])
    a.axhline(0, color=MUTED, lw=1)
    a.set_xlabel("Forecast probability (all outcomes pooled, equal-count bins)")
    a.set_ylabel("Observed minus forecast frequency")
    a.yaxis.set_major_formatter(lambda v, _: f"{v * 100:+.0f} pt")
    a.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    a.set_title("Dixon-Coles is too cautious with favourites")
    a.legend(loc="upper left")

    g = df.assign(home=(y == 0).astype(float)).groupby("season")
    seasons = [s[:4] + "-" + s[-2:] for s in g.size().index]
    for s in ("elo", "dc", "market"):
        bias = (g[f"{s}_p_H"].mean() - g["home"].mean()) * 100
        b.plot(seasons, bias, marker="o", ms=5, lw=2, color=C[s], label=NAMES[s])
    b.axhline(0, color=MUTED, lw=1)
    b.set_ylabel("Home wins: forecast minus observed (points)")
    b.set_title("Elo over-predicts home wins")
    b.legend(loc="best")
    b.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out / "fig2_calibration.png")
    plt.close(fig)


STRATEGY_EN = {
    "PRINCIPALE ens @ pinnacle": "Main: blend @ Pinnacle pre-match",
    "ens @ market_avg": "Blend @ market average",
    "ens @ market_max": "Blend @ best price",
    "elo @ pinnacle": "Elo alone @ Pinnacle",
    "dc @ pinnacle": "Dixon-Coles alone @ Pinnacle",
    "CONTRÔLE + oracle (clôture)": "Positive control: oracle",
    "CONTRÔLE - fréquences de base": "Negative control: base rates",
}


def fig_clv(summary: pd.DataFrame, out: Path) -> None:
    s = summary[summary.strategy.isin(STRATEGY_EN)].copy()
    s["label"] = s.strategy.map(STRATEGY_EN)
    s = s.iloc[::-1]
    colors = [
        C["oracle"] if "oracle" in t else C["base"] if "base rates" in t else C["ens"]
        for t in s.label
    ]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    yy = np.arange(len(s))
    ax.barh(yy, s.clv * 100, color=colors, height=0.6)
    ax.errorbar(
        s.clv * 100,
        yy,
        xerr=[(s.clv - s.clv_ci_low) * 100, (s.clv_ci_high - s.clv) * 100],
        fmt="none",
        ecolor=INK,
        elinewidth=1,
        capsize=3,
    )
    for y_, v, lo, hi in zip(yy, s.clv, s.clv_ci_low, s.clv_ci_high, strict=True):
        ax.text(
            (hi if v > 0 else lo) * 100 + (0.4 if v > 0 else -0.4),
            y_,
            f"{v:+.1%}",
            va="center",
            ha="left" if v > 0 else "right",
            fontsize=9,
            color=INK,
        )
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(yy, s.label)
    ax.set_xlim(min(s.clv.min() * 100 - 3, -10), max(s.clv.max() * 100 + 3, 3))
    ax.set_xlabel("Closing line value per bet (odds taken x Pinnacle closing probability - 1)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Every model strategy loses against the close; the controls behave")
    fig.savefig(out / "fig3_clv_strategies.png")
    plt.close(fig)


def fig_profit_vs_clv(bets: pd.DataFrame, out: Path) -> None:
    bets = bets.sort_values(["week", "event_id"]).reset_index(drop=True)
    x = np.arange(1, len(bets) + 1)
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    ax.plot(x, bets.flat_return.cumsum(), color=C["ens"], lw=1.6)
    ax.plot(x, bets.clv.cumsum(), color=C["market"], lw=2)
    ax.annotate(
        "Actual profit",
        (x[-1], bets.flat_return.sum()),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        color=C["ens"],
        fontsize=9,
    )
    ax.annotate(
        "Profit implied by CLV",
        (x[-1], bets.clv.sum()),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        color=C["market"],
        fontsize=9,
    )
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("Bets, in chronological order (1 unit each)")
    ax.set_ylabel("Cumulative units")
    ax.set_title("Results are noisy, CLV is not: same bets, two measures")
    ax.set_xlim(0, x[-1] * 1.18)
    fig.savefig(out / "fig4_profit_vs_clv.png")
    plt.close(fig)


def fig_edge_threshold(summary: pd.DataFrame, main_edge: float, out: Path) -> None:
    rows = []
    for _, r in summary.iterrows():
        if r.strategy.startswith("PRINCIPALE"):
            rows.append((main_edge, r))
        elif "edge > " in r.strategy:
            rows.append((float(r.strategy.split("edge > ")[1]), r))
    rows.sort(key=lambda t: t[0])
    e = np.array([t[0] for t in rows]) * 100
    clv = np.array([t[1].clv for t in rows]) * 100
    base = summary[summary.strategy.str.contains("fréquences de base")].clv.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(e, clv, marker="o", ms=7, lw=2, color=C["ens"])
    for xx, v in zip(e, clv, strict=True):
        ax.annotate(
            f"{v:+.2f}%",
            (xx, v),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=INK,
        )
    ax.axhline(base, color=C["base"], lw=1.5, ls="--")
    ax.annotate(
        "No-information bettor",
        (e[-1], base),
        xytext=(0, -12),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=MUTED,
    )
    ax.set_xticks(e, [f"{v:.0f}%" for v in e])
    lo, hi = min(clv.min(), base), max(clv.max(), base)
    ax.set_ylim(lo - 0.25 * (hi - lo) - 0.1, hi + 0.25 * (hi - lo) + 0.1)
    ax.set_xlabel("Minimum edge the model must see before betting")
    ax.set_ylabel("Mean CLV (%)")
    ax.set_title("The more the model disagrees with the market, the worse the bet")
    fig.savefig(out / "fig5_edge_threshold.png")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/foot.yaml")
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _style()

    df = test_frame(cfg)
    res = fig_gap_closed(df, out)
    fig_calibration(df, out)
    rep = Path(cfg["paths"]["reports"])
    summary_path = rep / "backtest_test_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        fig_clv(summary, out)
        fig_edge_threshold(summary, cfg["betting"]["strategy"]["min_edge"], out)
        fig_profit_vs_clv(pd.read_csv(rep / "backtest_test_bets.csv"), out)
    else:
        print("reports/backtest_test_summary.csv absent : lancer make test-backtest d'abord.")
    print(f"{len(df):,} matchs du test. Part de l'écart comblée :")
    for s, v in res["gap_closed"].items():
        print(f"  {s:7s} {v:6.1%}   log-loss {res['log_loss'][s]:.4f}")
    print(f"Figures écrites dans {out}/")


if __name__ == "__main__":
    main()
