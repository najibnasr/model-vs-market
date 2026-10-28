"""Pages grand public : le duel, la journée, ton club, et si tu avais parié ?"""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mvm.app import data, ui

ALL_LEAGUES = "Toutes les ligues"


def _league_select(df: pd.DataFrame, key: str, allow_all: bool = True):
    leagues = sorted(
        df.league.unique(), key=lambda c: (data.config()["leagues"].get(c, {}).get("tier", 9), c)
    )
    opts = ([ALL_LEAGUES] if allow_all else []) + leagues
    return st.selectbox(
        "Championnat",
        opts,
        key=key,
        format_func=lambda c: c if c == ALL_LEAGUES else data.league_name(c),
    )


def _period_options(df: pd.DataFrame) -> dict[str, tuple[str, str]]:
    seasons = sorted(df.season.unique(), reverse=True)
    opts = {}
    first, last = data.split_range("test")
    if any(first <= s <= last for s in seasons):
        opts[f"Toutes les saisons {first[:4]}-{last[-4:]}"] = (first, last)
    for s in seasons:
        opts[f"Saison {s[:4]}-{s[-2:]}"] = (s, s)
    return opts


# --- le duel ------------------------------------------------------------------------------------


def page_duel():
    st.html(
        '<p class="lede">Avant chaque match, mon modèle annonce ses probabilités, sans connaître '
        "le résultat. Les bookmakers aussi, à travers leurs cotes. Après le coup de sifflet final, "
        "celui qui avait donné le plus de chances au résultat réel gagne le duel.</p>"
    )
    ens = data.model_frame("ens")
    c1, c2 = st.columns(2)
    with c1:
        league = _league_select(ens, "duel_league")
    periods = _period_options(ens)
    with c2:
        period = st.selectbox("Période", list(periods), key="duel_period")
    first, last = periods[period]

    def pick(df):
        df = df[(df.season >= first) & (df.season <= last)]
        return df if league == ALL_LEAGUES else df[df.league == league]

    sub = pick(ens)
    if sub.empty:
        st.info("Aucun match pour cette sélection. Choisis une autre période.")
        return
    n = sub.duel.value_counts()
    st.html(
        ui.scoreboard(
            n.get("model", 0),
            n.get("book", 0),
            n.get("tie", 0),
            f"Duels gagnés sur {ui.num(len(sub))} matchs.",
        )
    )

    # probabilité « moyenne » donnée au bon résultat : moyenne géométrique = exp(-log-loss)
    g_mod, g_mkt = np.exp(-sub.ll_model.mean()), np.exp(-sub.ll_market.mean())
    st.html('<div class="section">Qui voit le plus juste, en moyenne ?</div>')
    st.html(
        ui.versus_bars(
            [("Le modèle", g_mod, ui.MODEL), ("Les bookmakers", g_mkt, ui.BOOK)],
            max(g_mod, g_mkt) * 1.08,
        )
        + '<p class="note">Probabilité moyenne accordée au résultat qui s’est vraiment produit '
        "(moyenne géométrique, l’équivalent exact de la log-loss utilisée dans les coulisses). "
        f"Pour comparaison, deviner au hasard parmi les trois issues donnerait {ui.pct(1 / 3)}.</p>"
    )

    st.html('<div class="section">Le classement des pronostiqueurs</div>')
    rows = [("Les bookmakers", "Cote de clôture de Pinnacle, marge retirée", g_mkt, None, "ref")]
    desc = {
        "ens": ("Mon modèle", "La moyenne d’Elo et de Dixon-Coles"),
        "elo": ("Elo", "Le classement des échecs, adapté au foot"),
        "dc": ("Dixon-Coles", "Modèle des buts marqués et encaissés"),
        "base": ("Le supporter sans avis", "Joue toujours les fréquences moyennes du championnat"),
    }
    for m, (name, sub_txt) in desc.items():
        s = pick(data.model_frame(m))
        if s.empty:
            continue
        won = (s.duel == "model").mean()
        rows.append((name, sub_txt, np.exp(-s.ll_model.mean()), won, "mod" if m == "ens" else ""))
    rows.sort(key=lambda r: -r[2])
    body = []
    for i, (name, sub_txt, g, won, cls) in enumerate(rows, 1):
        duel = "réf." if won is None else ui.pct(won)
        body.append(
            f'<tr class="{cls}"><td class="pos">{i}</td><td class="name">{html.escape(name)}'
            f"<small>{html.escape(sub_txt)}</small></td><td>{ui.pct(g, 1)}</td><td>{duel}</td></tr>"
        )
    st.html(
        '<table class="standings"><thead><tr><th></th><th>Pronostiqueur</th>'
        "<th>Probabilité moyenne sur le bon résultat</th>"
        "<th>Duels gagnés contre les bookmakers</th>"
        f"</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )

    gap = sub.pt_model - sub.pt_market
    for title, idx in (
        ("Les coups que le modèle avait sentis", gap.nlargest(4).index),
        ("Quand les bookmakers avaient raison", gap.nsmallest(4).index),
    ):
        st.html(f'<div class="section">{title}</div>')
        st.html(
            '<div class="cards">'
            + "".join(ui.match_card(r, show_bet=False) for _, r in sub.loc[idx].iterrows())
            + "</div>"
        )


# --- la journée ---------------------------------------------------------------------------------


def page_matchday():
    st.html(
        '<p class="lede">Tous les matchs d’une semaine, avec les pronostics du modèle et des '
        "bookmakers. Le contour blanc marque le résultat réel. Quand le modèle croyait tenir une "
        "bonne affaire, son ticket est affiché.</p>"
    )
    ens = data.model_frame("ens")
    c1, c2, c3 = st.columns([1, 1, 1.3])
    with c1:
        league = _league_select(ens, "md_league", allow_all=False)
    lg = ens[ens.league == league]
    with c2:
        season = st.selectbox(
            "Saison",
            sorted(lg.season.unique(), reverse=True),
            key="md_season",
            format_func=lambda s: f"{s[:4]}-{s[-2:]}",
        )
    se = lg[lg.season == season]
    weeks = sorted(se.kickoff_date.dt.to_period("W-SUN").dt.start_time.unique())
    with c3:
        week = st.select_slider(
            "Semaine",
            options=weeks,
            value=weeks[-1],
            key="md_week",
            format_func=lambda w: f"du {ui.date_fr(w)}",
        )
    games = se[se.kickoff_date.dt.to_period("W-SUN").dt.start_time == week]
    n = games.duel.value_counts()
    bets = games[games.bet_idx >= 0]
    won = (bets.bet_idx == bets.outcome_idx).sum()
    st.html(
        f'<p class="bigline"><span class="m">Modèle {n.get("model", 0)}</span>, '
        f"bookmakers {n.get('book', 0)}, égalités {n.get('tie', 0)}</p>"
        f'<p class="note">{len(bets)} ticket(s) joué(s) par le modèle cette semaine, '
        f"{won} gagné(s).</p>"
    )
    st.html(ui.card_grid(games))


# --- ton club -----------------------------------------------------------------------------------


def page_club():
    ens = data.model_frame("ens")
    clubs = (
        pd.concat(
            [
                ens[["home", "league"]].rename(columns={"home": "team"}),
                ens[["away", "league"]].rename(columns={"away": "team"}),
            ]
        )
        .groupby("team")
        .league.agg(lambda s: s.value_counts().index[0])
    )
    names = sorted(clubs.index)
    default = next(
        (names.index(t) for t in ("Barcelona", "Real Madrid", "Arsenal") if t in names), 0
    )
    c1, c2 = st.columns(2)
    with c1:
        team = st.selectbox("Ton club", names, index=default, key="club_team")
    played = ens[(ens.home == team) | (ens.away == team)]
    with c2:
        season = st.selectbox(
            "Saison",
            sorted(played.season.unique(), reverse=True),
            key="club_season",
            format_func=lambda s: f"{s[:4]}-{s[-2:]}",
        )
    games = played[played.season == season]
    n = games.duel.value_counts()
    leagues = ", ".join(data.league_name(c) for c in games.league.unique())
    st.html(
        f'<div class="clubhero">{ui.badge(team, 64)}<div><h2>{html.escape(team)}</h2>'
        f"<p>{html.escape(leagues)}, saison {season[:4]}-{season[-2:]}, {len(games)} matchs</p>"
        "</div></div>"
    )
    st.html(
        ui.scoreboard(
            n.get("model", 0),
            n.get("book", 0),
            n.get("tie", 0),
            f"Qui a le mieux lu les matchs de {html.escape(team)} cette saison ?",
        )
    )

    chips = []
    for _, r in games.iterrows():
        home = r.home == team
        gf, ga = (r.home_score, r.away_score) if home else (r.away_score, r.home_score)
        letter = "V" if gf > ga else "N" if gf == ga else "D"
        color = {"model": ui.MODEL, "book": ui.BOOK, "tie": "#c9c8bd"}[r.duel]
        opp = r.away if home else r.home
        title = f"{ui.date_fr(r.kickoff_date)} : {'contre' if home else 'chez'} {opp}, {gf}-{ga}"
        chips.append(f'<div style="background:{color}" title="{html.escape(title)}">{letter}</div>')
    st.html(
        '<div class="section">Le fil de la saison</div>'
        '<p class="note">Une case par match : V, N ou D pour le résultat du club, la couleur pour '
        "le vainqueur du duel.</p>"
        f'<div class="strip">{"".join(chips)}</div>'
        f'<div class="legend"><span><i style="background:{ui.MODEL}"></i>'
        "le modèle a mieux vu</span>"
        f'<span><i style="background:{ui.BOOK}"></i>les bookmakers ont mieux vu</span>'
        '<span><i style="background:#c9c8bd"></i>égalité</span></div>'
    )
    recent = games.iloc[::-1]
    st.html('<div class="section">Les derniers matchs</div>')
    st.html(ui.card_grid(recent.head(6)))
    if len(recent) > 6:
        with st.expander(f"Voir les {len(recent) - 6} autres matchs de la saison"):
            st.html(ui.card_grid(recent.iloc[6:]))


# --- et si tu avais parié ? ---------------------------------------------------------------------

STYLES = {
    "Prudent": (0.10, "ne parie que si le modèle voit au moins 10 % d’avantage"),
    "Équilibré": (0.05, "parie dès 5 % d’avantage (la stratégie testée)"),
    "Audacieux": (0.02, "parie dès 2 % d’avantage"),
}
BOOKS = {
    "pinnacle": "Pinnacle, le bookmaker le plus dur à battre",
    "market_max": "La meilleure cote parmi tous les bookmakers",
}


def _bettor(name, who, color, summary, initial):
    final = summary.get("kelly_final_bankroll", initial)
    clv = summary.get("clv", 0.0)
    n = summary.get("n_bets", 0)
    clv_txt = (
        f"cotes en moyenne {ui.pct(abs(clv), 1)} {'meilleures' if clv > 0 else 'moins bonnes'} "
        "que la cote finale"
    )
    return (
        f'<div class="bettor" style="--c:{color}"><h4>{html.escape(name)}</h4>'
        f'<div class="who">{html.escape(who)}</div><div class="amount">{ui.money(final)}</div>'
        f"<ul><li>{ui.num(n)} paris</li><li>{clv_txt}</li></ul></div>"
    )


def page_whatif():
    st.html(
        '<p class="lede">Tu pars avec 1 000 € et tu suis aveuglément le modèle : à chaque fois '
        "qu’il pense que la cote est trop généreuse, tu paries. La mise grandit et rétrécit avec "
        "ta cagnotte (un quart du critère de Kelly, jamais plus de 2 % par pari).</p>"
    )
    seasons = data.available_seasons()
    splits = {}
    tf, tl = data.split_range("test")
    uf, ul = data.split_range("tuning")
    if any(tf <= s <= tl for s in seasons):
        splits["test"] = f"Saisons {tf[:4]} à {tl[-4:]}"
    if any(uf <= s <= ul for s in seasons):
        splits["tuning"] = f"Saisons {uf[:4]} à {ul[-4:]}"
    c1, c2, c3 = st.columns(3)
    with c1:
        style = st.radio(
            "Ton style",
            list(STYLES),
            index=1,
            key="wi_style",
            captions=[v[1] for v in STYLES.values()],
        )
    with c2:
        book = st.radio("Où tu paries", list(BOOKS), format_func=BOOKS.get, key="wi_book")
    with c3:
        split = st.radio("Période", list(splits), format_func=splits.get, key="wi_split")
    edge = STYLES[style][0]
    initial = data.config()["betting"]["initial_bankroll"]

    you, bets, weekly = data.simulate(split, "model", book, edge)
    oracle, _, w_or = data.simulate(split, "oracle", book, edge)
    base, _, w_base = data.simulate(split, "base", book, edge)
    if not len(weekly):
        st.info("Aucun pari avec ces réglages.")
        return
    start, end = weekly.week.iloc[0], weekly.week.iloc[-1]
    final = you["kelly_final_bankroll"]
    verb = "il te reste" if final < initial else "tu as"
    st.html(
        f'<p class="bigline">1 000 € le {ui.date_fr(start)}. Le {ui.date_fr(end)}, '
        f'{verb} <span class="m">{ui.money(final)}</span>.</p>'
    )

    fig = go.Figure()
    for w, name, color, dash in (
        (w_or, "Le voyant", ui.THIRD, "dot"),
        (w_base, "Le supporter sans avis", ui.BOOK, "solid"),
        (weekly, "Toi, avec le modèle", ui.MODEL, "solid"),
    ):
        if len(w):
            fig.add_scatter(
                x=w.week,
                y=w.bankroll,
                name=name,
                mode="lines",
                line=dict(color=color, width=3 if "Toi" in name else 2, dash=dash),
                hovertemplate="%{x|%d/%m/%Y} : %{y:,.0f} €<extra>" + name + "</extra>",
            )
    fig.add_hline(y=initial, line_color="rgba(239,238,230,.35)", line_width=1)
    fig.update_layout(
        template="plotly_dark",
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Barlow", color=ui.CHALK, size=14),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.08, x=0),
        yaxis=dict(
            type="log",
            title="Cagnotte (échelle log)",
            gridcolor="rgba(239,238,230,.1)",
            tickformat=",.0f",
            ticksuffix=" €",
        ),
        xaxis=dict(gridcolor="rgba(239,238,230,.06)"),
        hovermode="x unified",
        separators=", ",
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.html('<div class="section">Trois parieurs, mêmes règles</div>')
    a, b, c = st.columns(3)
    with a:
        st.html(
            _bettor(
                "Toi, avec le modèle", "Tu suis les probabilités du modèle.", ui.MODEL, you, initial
            )
        )
    with b:
        st.html(
            _bettor(
                "Le voyant",
                "Il triche : il connaît déjà la cote de fin de marché.",
                ui.THIRD,
                oracle,
                initial,
            )
        )
    with c:
        st.html(
            _bettor(
                "Le supporter sans avis",
                "Il ne regarde que les moyennes du championnat.",
                ui.BOOK,
                base,
                initial,
            )
        )

    st.html(
        '<div class="section">Pourquoi tu perds (et pourquoi c’est normal)</div>'
        '<div class="note" style="font-size:1rem;color:var(--chalk)">'
        "<p>Les bookmakers gardent une marge sur chaque cote : même un pronostic parfaitement "
        "moyen perd cette marge à la longue. C’est ce qui arrive au supporter sans avis.</p>"
        "<p>Pour gagner, il faut voir plus juste que le marché. Le bon thermomètre n’est pas le "
        "résultat (trop de hasard sur quelques centaines de paris) mais la cote finale, juste "
        "avant le coup d’envoi : elle intègre toutes les infos (blessures, compositions, "
        "argent des "
        "parieurs professionnels). Si tu prends régulièrement de meilleures cotes que celle-là, tu "
        "as un vrai avantage. Le voyant le prouve : il les connaît et sa cagnotte explose.</p>"
        "<p>Mon modèle, lui, ne regarde que les scores passés. Les bookmakers savent tout ce qu’il "
        "sait, et plus encore. Quand il croit voir une bonne affaire, c’est le plus souvent "
        "lui qui "
        "se trompe.</p></div>"
    )

    if len(bets):
        ev = data.load_data()["events"]
        bt = bets.join(
            ev[["home", "away", "home_score", "away_score", "kickoff_date"]], on="event_id"
        )
        best, worst = bt.loc[bt.pnl.idxmax()], bt.loc[bt.pnl.idxmin()]
        st.html('<div class="section">Ton plus beau ticket, ton pire ticket</div>')
        a, b = st.columns(2)
        for col, r, title in ((a, best, "Le plus beau"), (b, worst, "Le pire")):
            pick = {"H": f"victoire de {r.home}", "D": "match nul", "A": f"victoire de {r.away}"}[
                r.outcome
            ]
            col.html(
                '<div class="mcard"><div class="date">'
                f"{title}, le {ui.date_fr(r.kickoff_date)}</div>"
                f"<div><b>{html.escape(r.home)} {int(r.home_score)}-{int(r.away_score)} "
                f'{html.escape(r.away)}</b></div><div class="ticket"><span>Pari sur '
                f"{html.escape(pick)} à {ui.dec(r.odds)}, mise {ui.money(r.stake)}</span>"
                f'<span class="res {"won" if r.won else "lost"}">'
                f"{'+' if r.pnl > 0 else ''}{ui.money(r.pnl)}</span></div></div>"
            )
