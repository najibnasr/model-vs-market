"""Pages grand public : cette semaine, la saison en direct, ton club, l'historique."""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mvm.app import data, ui

ALL = "All leagues"
OUT_LABELS = ("Home", "Draw", "Away")


def _league_pills(codes, key: str, allow_all: bool = True, default: str | None = None) -> str:
    opts = ([ALL] if allow_all else []) + data.league_order(set(codes))
    default = default if default in opts else opts[0]
    choice = st.pills(
        "League",
        opts,
        default=default,
        key=key,
        format_func=lambda c: c if c == ALL else data.league_name(c),
        label_visibility="collapsed",
    )
    return choice or default


def _filter(df: pd.DataFrame, league: str) -> pd.DataFrame:
    return df if league == ALL else df[df.league == league]


def _no_live_data() -> None:
    st.info(
        "No data for the current season yet. Run `make live` in the terminal: it downloads the "
        "latest results and the upcoming fixtures, then predicts every match."
    )


def _chart_layout(fig: go.Figure, height: int, ytitle: str, log: bool = False) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Barlow", color=ui.CHALK, size=14),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.1, x=0),
        yaxis=dict(
            type="log" if log else "linear",
            title=ytitle,
            gridcolor="rgba(239,238,230,.1)",
            tickprefix="€",
            tickformat=",.0f",
        ),
        xaxis=dict(gridcolor="rgba(239,238,230,.06)"),
        hovermode="x unified",
    )
    return fig


def _bankroll_chart(curves, initial: float, log: bool) -> None:
    fig = go.Figure()
    for w, name, color, dash, width in curves:
        if len(w):
            fig.add_scatter(
                x=w.week,
                y=w.bankroll,
                name=name,
                mode="lines",
                line=dict(color=color, width=width, dash=dash),
                hovertemplate="%{x|%d %b %Y}: €%{y:,.0f}<extra>" + name + "</extra>",
            )
    fig.add_hline(y=initial, line_color="rgba(239,238,230,.35)", line_width=1)
    st.plotly_chart(
        _chart_layout(fig, 360, "Bankroll" + (" (log scale)" if log else ""), log),
        width="stretch",
        config={"displayModeBar": False},
    )


def _bettor(name: str, who: str, color: str, summary: dict, initial: float) -> str:
    final = summary.get("kelly_final_bankroll", initial)
    clv = summary.get("clv", 0.0)
    n = summary.get("n_bets", 0)
    better = "better" if clv > 0 else "worse"
    return (
        f'<div class="bettor" style="--c:{color}"><h4>{html.escape(name)}</h4>'
        f'<div class="who">{html.escape(who)}</div><div class="amount">{ui.money(final)}</div>'
        f"<ul><li>{ui.num(n)} bets</li><li>odds {ui.pct(abs(clv), 1)} {better} than the "
        "final price, on average</li></ul></div>"
    )


def _three_bettors(you, oracle, base, initial) -> None:
    a, b, c = st.columns(3)
    with a:
        st.html(_bettor("You, with the model", "You follow the model.", ui.MODEL, you, initial))
    with b:
        st.html(
            _bettor(
                "The clairvoyant",
                "Cheats: already knows the final market price.",
                ui.THIRD,
                oracle,
                initial,
            )
        )
    with c:
        st.html(
            _bettor(
                "The fan with no opinion",
                "Only looks at league averages.",
                ui.BOOK,
                base,
                initial,
            )
        )


def _why_you_lose() -> None:
    st.html(
        '<div class="section">Why you lose (and why that is normal)</div>'
        '<div class="note prose">'
        "<p>Bookmakers keep a margin on every price, so even a perfectly average forecaster "
        "slowly loses it. That is what happens to the fan with no opinion.</p>"
        "<p>To win, you have to see further than the market. The right thermometer is not the "
        "result (far too much luck over a few hundred bets) but the final price just before "
        "kick-off, which has absorbed everything: injuries, line-ups, professional money. Beat "
        "that price regularly and you have a real edge. The clairvoyant proves it: he knows it "
        "in advance and his bankroll explodes.</p>"
        "<p>My model only looks at past scores. The bookmakers know everything it knows, and "
        "more. When it thinks it has found a bargain, it is usually the one that is wrong.</p>"
        "</div>"
    )


# --- cette semaine ------------------------------------------------------------------------------


def _on_pick(event_id: str, key: str) -> None:
    v = st.session_state.get(key)
    data.save_pick(event_id, None if v is None else OUT_LABELS.index(v))


def page_week():
    up = data.upcoming()
    if up is None:
        ui.lede("The next fixtures appear here, with the model's forecast next to the market's.")
        _no_live_data()
        return
    when = data.last_update()
    ui.lede(
        "The next matches, forecast before kick-off by a model frozen before the season. Make "
        "your own call on each match: once results are in, the Season page tells you who read "
        "the games best, you, the model or the bookmakers."
        + (f" Last update: {ui.date_en(when, weekday=True)}." if when else "")
    )
    if when and (up.kickoff_date.max() - pd.Timestamp(when.date())).days <= 1:
        st.info(
            "Only the last midweek matches are listed: football-data.co.uk publishes the next "
            "round's fixtures and prices on Friday. Run `make live` again on Friday or Saturday "
            "morning to get the weekend."
        )
    league = _league_pills(up.league, "week_league", allow_all=True, default=up.league.iloc[0])
    games = _filter(up, league)
    fav_m = games[data.PCOLS].to_numpy().argmax(axis=1)
    fav_b = games[data.MKT].to_numpy().argmax(axis=1)
    n_bets = int((games.bet_idx >= 0).sum())
    st.html(
        f'<p class="bigline">{len(games)} matches. The model disagrees with the bookmakers on '
        f'<span class="m">{int((fav_m != fav_b).sum())}</span> and bets on {n_bets}.</p>'
        '<p class="note">Bars: probability of a home win, a draw and an away win. Bookmakers: '
        "average price across bookmakers, margin removed.</p>"
    )
    picks = data.load_picks().set_index("event_id")["pick"]
    cols = st.columns(2)
    for i, (eid, r) in enumerate(games.iterrows()):
        with cols[i % 2], st.container(key=f"card_{eid}"):
            st.html(ui.upcoming_card(r))
            prev = picks.get(eid, -1)
            key = f"pick_{eid}"
            st.segmented_control(
                "Your call",
                OUT_LABELS,
                default=OUT_LABELS[int(prev)] if prev >= 0 else None,
                key=key,
                on_change=_on_pick,
                args=(eid, key),
                format_func=lambda o, r=r: {
                    "Home": r.home,
                    "Draw": "Draw",
                    "Away": r.away,
                }[o],
            )


# --- la saison en direct ------------------------------------------------------------------------


def _your_record(season: pd.DataFrame) -> None:
    picks = data.load_picks()
    picks = picks[picks.pick >= 0].set_index("event_id")
    done = season.join(picks[["pick"]], how="inner")
    ui.section("You vs the model vs the bookmakers")
    if done.empty:
        n = len(picks)
        st.html(
            '<p class="note">'
            + (
                f"You have {n} call(s) waiting for their result."
                if n
                else "Make your calls on the This week page: they are scored here once played."
            )
            + "</p>"
        )
        return
    y = done.outcome_idx.to_numpy()
    rows = [
        ("You", (done.pick.to_numpy() == y).mean(), ui.CHALK),
        ("The model", (done[data.PCOLS].to_numpy().argmax(axis=1) == y).mean(), ui.MODEL),
        ("The bookmakers", (done[data.MKT].to_numpy().argmax(axis=1) == y).mean(), ui.BOOK),
    ]
    st.html(
        ui.versus_bars(rows, 1.0)
        + f'<p class="note">Share of correct calls on the {len(done)} match(es) you predicted '
        "(the favourite of the model or of the bookmakers counts as their call).</p>"
    )


def page_season():
    season = data.live_season()
    df = data.live_frame("ens")
    if df.empty:
        ui.lede(f"The {ui.season_label(season)} season, match by match, as it happens.")
        _no_live_data()
        return
    ui.lede(
        f"Every {ui.season_label(season)} match so far. The model's settings were frozen before "
        "the season and it never sees a result before predicting. After the final whistle, "
        "whoever gave the higher probability to what actually happened wins the duel."
    )
    league = _league_pills(df.league, "season_league")
    sub = _filter(df, league)
    n = sub.duel.value_counts()
    logged = len(set(sub.index) & data.logged_ids())
    st.html(
        ui.scoreboard(
            n.get("model", 0),
            n.get("book", 0),
            n.get("tie", 0),
            f"Duels won over {ui.num(len(sub))} matches since "
            f"{ui.date_en(sub.kickoff_date.min())}."
            + (
                f" {ui.num(logged)} of these forecasts were published in the public log "
                "before kick-off."
                if logged
                else ""
            ),
        )
    )
    g_mod, g_mkt = np.exp(-sub.ll_model.mean()), np.exp(-sub.ll_market.mean())
    ui.section("Who sees it better, on average?")
    st.html(
        ui.versus_bars(
            [("The model", g_mod, ui.MODEL), ("The bookmakers", g_mkt, ui.BOOK)],
            max(g_mod, g_mkt) * 1.08,
        )
        + '<p class="note">Average probability given to the result that actually happened '
        "(geometric mean, the exact counterpart of the log-loss used in Backstage). Guessing at "
        f"random among the three outcomes would give {ui.pct(1 / 3)}. This season the "
        "bookmakers are the market average: Pinnacle prices are no longer published, and the "
        "average is a slightly easier benchmark.</p>"
    )

    _your_record(sub)

    ui.section(
        "If you had followed the model since the first matchday",
        "€1,000, a bet every time the model sees at least 5% of value at the average price, "
        "stake = a quarter of the Kelly criterion, never more than 2% of the bankroll.",
    )
    initial = data.config()["betting"]["initial_bankroll"]
    book = "market_avg"
    you, bets, weekly = data.simulate_live("model", book, 0.05)
    oracle, _, w_or = data.simulate_live("oracle", book, 0.05)
    base, _, w_base = data.simulate_live("base", book, 0.05)
    if len(weekly):
        _bankroll_chart(
            [
                (w_or, "The clairvoyant", ui.THIRD, "dot", 2),
                (w_base, "The fan with no opinion", ui.BOOK, "solid", 2),
                (weekly, "You, with the model", ui.MODEL, "solid", 3),
            ],
            initial,
            log=False,
        )
        _three_bettors(you, oracle, base, initial)

    last_week = sub[sub.week == sub.week.max()].iloc[::-1]
    ui.section(f"Latest results, week of {ui.date_en(last_week.week.iloc[0])}")
    st.html(ui.card_grid(last_week.head(8)))
    if len(last_week) > 8:
        with st.expander(f"See the other {len(last_week) - 8} matches of the week"):
            st.html(ui.card_grid(last_week.iloc[8:]))


# --- ton club -----------------------------------------------------------------------------------


def _club_confidence(team: str, games: pd.DataFrame) -> None:
    """Probabilité moyenne de victoire du club selon chacun : explique un duel déséquilibré."""
    if games.empty:
        return
    home = (games.home == team).to_numpy()
    p_mod = np.where(home, games.p_H, games.p_A).mean()
    p_mkt = np.where(home, games.mkt_p_H, games.mkt_p_A).mean()
    won = np.where(home, games.outcome_idx == 0, games.outcome_idx == 2).mean()
    gap = p_mod - p_mkt
    if abs(gap) < 0.03:
        verdict = "The model and the bookmakers rate this club about the same."
    elif gap < 0:
        verdict = (
            "The model rates this club lower than the bookmakers do. When a strong team keeps "
            "winning, the more confident forecaster wins the duel almost every week: the model "
            "is known to be too cautious with heavy favourites (it only sees past scores, and "
            "its estimates are pulled towards the league average)."
        )
    else:
        verdict = "The model rates this club higher than the bookmakers do."
    ui.section(
        f"How much does each believe in {team}?",
        f"Average probability of a {html.escape(team)} win in these matches. "
        f"Actual win rate: {ui.pct(won)}.",
    )
    st.html(
        ui.versus_bars(
            [
                ("The model", p_mod, ui.MODEL),
                ("The bookmakers", p_mkt, ui.BOOK),
                ("Actual wins", won, ui.CHALK),
            ],
            1.0,
        )
        + f'<p class="note">{verdict}</p>'
    )


def page_club():
    df = data.model_frame("ens")
    clubs = pd.concat(
        [
            df[["home", "league", "kickoff_date"]].rename(columns={"home": "team"}),
            df[["away", "league", "kickoff_date"]].rename(columns={"away": "team"}),
        ]
    )
    latest_league = clubs.sort_values("kickoff_date").groupby("team").league.last()
    names = sorted(latest_league.index)
    default = next(
        (names.index(t) for t in ("Barcelona", "Real Madrid", "Arsenal") if t in names), 0
    )
    c1, c2 = st.columns(2)
    with c1:
        team = st.selectbox("Your club", names, index=default, key="club_team")
    played = df[(df.home == team) | (df.away == team)]
    seasons = sorted(played.season.unique(), reverse=True)
    with c2:
        season = st.selectbox("Season", seasons, key="club_season", format_func=ui.season_label)
    games = played[played.season == season]
    n = games.duel.value_counts()
    leagues = ", ".join(data.league_name(c) for c in games.league.unique())
    st.html(
        f'<div class="clubhero">{ui.badge(team, 64)}<div><h2>{html.escape(team)}</h2>'
        f"<p>{html.escape(leagues)}, {ui.season_label(season)}, {len(games)} matches played</p>"
        "</div></div>"
    )

    up = data.upcoming() if season == data.live_season() else None
    if up is not None:
        nxt = up[(up.home == team) | (up.away == team)]
        if len(nxt):
            ui.section("Next match")
            st.html('<div class="cards">' + ui.upcoming_card(nxt.iloc[0]) + "</div>")

    st.html(
        ui.scoreboard(
            n.get("model", 0),
            n.get("book", 0),
            n.get("tie", 0),
            f"Who read {html.escape(team)}'s matches better this season?",
        )
    )
    _club_confidence(team, games)

    chips = []
    for _, r in games.iterrows():
        home = r.home == team
        gf, ga = (r.home_score, r.away_score) if home else (r.away_score, r.home_score)
        letter = "W" if gf > ga else "D" if gf == ga else "L"
        color = {"model": ui.MODEL, "book": ui.BOOK, "tie": ui.TIE_GREY}[r.duel]
        opp = r.away if home else r.home
        title = f"{ui.date_en(r.kickoff_date)}: {'vs' if home else 'at'} {opp}, {gf}-{ga}"
        chips.append(f'<div style="background:{color}" title="{html.escape(title)}">{letter}</div>')
    ui.section(
        "The season so far",
        "One square per match: W, D or L for the club's result, the colour for the duel winner.",
    )
    st.html(
        f'<div class="strip">{"".join(chips)}</div>'
        f'<div class="legend"><span><i style="background:{ui.MODEL}"></i>the model saw it '
        f'better</span><span><i style="background:{ui.BOOK}"></i>the bookmakers saw it '
        f'better</span><span><i style="background:{ui.TIE_GREY}"></i>level</span></div>'
    )
    recent = games.iloc[::-1]
    ui.section("Latest matches")
    st.html(ui.card_grid(recent.head(6)))
    if len(recent) > 6:
        with st.expander(f"See the other {len(recent) - 6} matches of the season"):
            st.html(ui.card_grid(recent.iloc[6:]))


# --- l'historique (2012-2025) -------------------------------------------------------------------


def _track_duel():
    df = data.archive_frame("ens")
    t0, t1 = data.split_range("test")
    periods = {}
    if ((df.season >= t0) & (df.season <= t1)).any():
        periods[f"{t0[:4]} to {t1[-4:]} (test seasons)"] = (t0, t1)
    for s in sorted(df.season.unique(), reverse=True):
        periods[ui.season_label(s)] = (s, s)
    c1, c2 = st.columns(2)
    with c1:
        leagues = [ALL, *data.league_order(set(df.league))]
        league = st.selectbox(
            "League",
            leagues,
            key="track_league",
            format_func=lambda c: c if c == ALL else data.league_name(c),
        )
    with c2:
        period = st.selectbox("Period", list(periods), key="track_period")
    first, last = periods[period]

    def pick(d):
        return _filter(d[(d.season >= first) & (d.season <= last)], league)

    sub = pick(df)
    if sub.empty:
        st.info("No match for this selection.")
        return
    n = sub.duel.value_counts()
    st.html(
        ui.scoreboard(
            n.get("model", 0),
            n.get("book", 0),
            n.get("tie", 0),
            f"Duels won over {ui.num(len(sub))} matches, against Pinnacle's closing price.",
        )
    )
    ui.section("The forecasters' table")
    g_mkt = np.exp(-sub.ll_market.mean())
    rows = [("The bookmakers", "Pinnacle closing price, margin removed", g_mkt, None, "ref")]
    desc = {
        "ens": ("My model", "Average of Elo and Dixon-Coles"),
        "elo": ("Elo", "The chess rating, adapted to football"),
        "dc": ("Dixon-Coles", "A model of goals scored and conceded"),
        "base": ("The fan with no opinion", "Always plays the league averages"),
    }
    for m, (name, sub_txt) in desc.items():
        s = pick(data.archive_frame(m))
        if len(s):
            won = (s.duel == "model").mean()
            g = np.exp(-s.ll_model.mean())
            rows.append((name, sub_txt, g, won, "mod" if m == "ens" else ""))
    rows.sort(key=lambda r: -r[2])
    body = "".join(
        f'<tr class="{cls}"><td class="pos">{i}</td><td class="name">{html.escape(name)}'
        f"<small>{html.escape(t)}</small></td><td>{ui.pct(g, 1)}</td>"
        f"<td>{'ref.' if won is None else ui.pct(won)}</td></tr>"
        for i, (name, t, g, won, cls) in enumerate(rows, 1)
    )
    st.html(
        '<table class="standings"><thead><tr><th></th><th>Forecaster</th>'
        "<th>Average probability on the actual result</th>"
        f"<th>Duels won against the bookmakers</th></tr></thead><tbody>{body}</tbody></table>"
    )
    gap = sub.pt_model - sub.pt_market
    for title, idx in (
        ("Upsets the model saw coming", gap.nlargest(4).index),
        ("When the bookmakers knew better", gap.nsmallest(4).index),
    ):
        ui.section(title)
        st.html(
            '<div class="cards">'
            + "".join(ui.match_card(r, show_bet=False) for _, r in sub.loc[idx].iterrows())
            + "</div>"
        )


STYLES = {
    "Cautious": (0.10, "bets only when the model sees at least 10% of value"),
    "Balanced": (0.05, "bets from 5% of value (the tested strategy)"),
    "Bold": (0.02, "bets from 2% of value"),
}
BOOKS = {
    "pinnacle": "Pinnacle, the hardest bookmaker to beat",
    "market_max": "The best price across all bookmakers",
}


def _track_whatif():
    seasons = set(data.archive_frame("ens").season)
    splits = {}
    for key, label in (("test", "test seasons"), ("tuning", "tuning seasons")):
        f, t = data.split_range(key)
        if any(f <= s <= t for s in seasons):
            splits[key] = f"{f[:4]} to {t[-4:]} ({label})"
    c1, c2, c3 = st.columns(3)
    with c1:
        style = st.radio(
            "Your style",
            list(STYLES),
            index=1,
            key="wi_style",
            captions=[v[1] for v in STYLES.values()],
        )
    with c2:
        book = st.radio("Where you bet", list(BOOKS), format_func=BOOKS.get, key="wi_book")
    with c3:
        split = st.radio("Period", list(splits), format_func=splits.get, key="wi_split")
    edge = STYLES[style][0]
    initial = data.config()["betting"]["initial_bankroll"]
    you, bets, weekly = data.simulate(split, "model", book, edge)
    oracle, _, w_or = data.simulate(split, "oracle", book, edge)
    base, _, w_base = data.simulate(split, "base", book, edge)
    if not len(weekly):
        st.info("No bet with these settings.")
        return
    final = you["kelly_final_bankroll"]
    st.html(
        f'<p class="bigline">€1,000 on {ui.date_en(weekly.week.iloc[0])}. On '
        f"{ui.date_en(weekly.week.iloc[-1])}, you "
        f"{'are left with' if final < initial else 'have'} "
        f'<span class="m">{ui.money(final)}</span>.</p>'
    )
    _bankroll_chart(
        [
            (w_or, "The clairvoyant", ui.THIRD, "dot", 2),
            (w_base, "The fan with no opinion", ui.BOOK, "solid", 2),
            (weekly, "You, with the model", ui.MODEL, "solid", 3),
        ],
        initial,
        log=True,
    )
    ui.section("Three bettors, same rules")
    _three_bettors(you, oracle, base, initial)
    _why_you_lose()


TRACK = {"The duel": _track_duel, "What if you had bet?": _track_whatif}


def page_track():
    a0, a1 = data.split_range("archive")
    ui.lede(
        f"The same duel, replayed on {a0[:4]} to {a1[-4:]} against Pinnacle, the sharpest "
        "bookmaker. Thousands of matches: enough to separate skill from luck."
    )
    with st.container(key="subnav"):
        view = st.radio(
            "View", list(TRACK), horizontal=True, label_visibility="collapsed", key="track_view"
        )
    TRACK[view]()
