"""Thème « match de nuit » et composants HTML du dashboard (textes affichés en anglais).

Deux camps, comme deux maillots : le modèle en bleu, les bookmakers en orange. Les équipes ont
un écusson généré (code à 3 lettres façon télé + couleur dérivée du nom) : aucun logo réel.
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from pathlib import Path

import pandas as pd
import streamlit as st

MODEL, BOOK, THIRD = "#4ea1ff", "#ff8a3d", "#3ddc97"
CHALK, DIM, NIGHT, TURF = "#efeee6", "#a9b8ae", "#0c231c", "#12342a"
TIE_GREY = "#c9c8bd"
NBSP = " "

# Couleurs de maillot (fond, texte) : assez variées pour distinguer deux équipes d'un match.
KITS = [
    ("#d7263d", "#ffffff"),
    ("#1d4ed8", "#ffffff"),
    ("#f2c230", "#14231c"),
    ("#7a1f3d", "#ffffff"),
    ("#6cb4ee", "#0c231c"),
    ("#f4f4f0", "#14231c"),
    ("#2f9e55", "#ffffff"),
    ("#f06a1d", "#ffffff"),
    ("#272d3d", "#ffffff"),
    ("#7d4cc2", "#ffffff"),
    ("#0e8f8f", "#ffffff"),
    ("#b9b9b4", "#14231c"),
]
# Préfixes ignorés pour le code à 3 lettres ("FC Koln" -> KOL)
PREFIXES = {
    *("fc", "ac", "as", "sc", "cf", "rc", "us", "sv", "vfb", "vfl", "tsg", "cd", "ud", "sd"),
    *("rcd", "afc", "ogc", "fk", "sporting", "1", "ssc", "ss", "sk", "bv", "bsc", "fsv"),
}
CSS_PATH = Path(__file__).with_name("theme.css")
PITCH_SVG = (
    '<svg class="pitch" viewBox="0 0 1000 300" preserveAspectRatio="xMidYMid slice" '
    'aria-hidden="true"><g fill="none" stroke="rgba(239,238,230,.07)" stroke-width="2">'
    '<line x1="500" y1="0" x2="500" y2="300"/><circle cx="500" cy="150" r="92"/>'
    '<circle cx="500" cy="150" r="4" fill="rgba(239,238,230,.1)"/></g></svg>'
)


def inject_css() -> None:
    st.html(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>")


# --- formats (conventions anglaises) ------------------------------------------------------------


def pct(x: float, d: int = 0) -> str:
    return f"{x * 100:.{d}f}%"


def num(n: float) -> str:
    return f"{int(round(n)):,}"


def money(x: float, signed: bool = False) -> str:
    sign = "-" if x < 0 else ("+" if signed and x > 0 else "")
    return f"{sign}€{num(abs(x))}"


def dec(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}"


def date_en(ts, weekday: bool = False, year: bool = True) -> str:
    ts = pd.Timestamp(ts)
    out = f"{ts.day} {ts:%b}"
    if weekday:
        out = f"{ts:%a} {out}"
    return f"{out} {ts.year}" if year else out


def season_label(s: str) -> str:
    return f"{s[:4]}-{s[-2:]}"


# --- équipes ------------------------------------------------------------------------------------


def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s.replace("ß", "ss")).encode("ascii", "ignore").decode()


def team_code(name: str) -> str:
    words = [w for w in re.split(r"[\s\-\.'’:]+", _ascii(name)) if w]
    sig = [w for w in words if w.lower() not in PREFIXES] or words or ["???"]
    two_words = len(sig) >= 2 and len(sig[0]) <= 4  # "Man United" -> MUN, "Real Madrid" -> RMA
    code = sig[0][0] + sig[1][:2] if two_words else sig[0][:3]
    return code.upper().ljust(3, "X")[:3]


def kit_index(name: str) -> int:
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % len(KITS)


def kits(home: str, away: str) -> tuple[tuple[str, str], tuple[str, str]]:
    h, a = kit_index(home), kit_index(away)
    if a == h:
        a = (a + 5) % len(KITS)
    return KITS[h], KITS[a]


def badge(name: str, size: int = 34, kit: tuple[str, str] | None = None) -> str:
    """Écusson en CSS pur (st.html filtre le SVG) : forme de bouclier, code à 3 lettres."""
    bg, fg = kit or KITS[kit_index(name)]
    return (
        f'<span class="crest" role="img" aria-label="{html.escape(name)}" '
        f'style="--s:{size}px;--bg:{bg};--fg:{fg}"><span>{html.escape(team_code(name))}</span></span>'
    )


# --- composants ---------------------------------------------------------------------------------


def brand(subtitle: str) -> None:
    st.html(
        '<div class="brand"><h1><span class="m">The model</span> vs '
        f'<span class="b">the bookmakers</span></h1><span>{html.escape(subtitle)}</span></div>'
    )


def lede(text: str) -> None:
    st.html(f'<p class="lede">{text}</p>')


def section(title: str, note: str | None = None) -> None:
    st.html(
        f'<div class="section">{html.escape(title)}</div>'
        + (f'<p class="note">{note}</p>' if note else "")
    )


def scoreboard(model_wins: int, book_wins: int, ties: int, foot: str) -> str:
    return (
        f'<div class="board">{PITCH_SVG}<div class="row">'
        f'<div><div class="who m">The model</div><div class="big">{num(model_wins)}</div>'
        f'<div class="kit" style="background:{MODEL}"></div></div>'
        '<div class="mid">–</div>'
        f'<div><div class="who b">The bookmakers</div><div class="big">{num(book_wins)}</div>'
        f'<div class="kit" style="background:{BOOK}"></div></div></div>'
        f'<div class="foot">{foot}<br>{num(ties)} level (less than one point apart).</div>'
        "</div>"
    )


def versus_bars(rows: list[tuple[str, float, str]], vmax: float) -> str:
    out = ['<div class="vs">']
    for label, v, color in rows:
        w = max(0.0, min(1.0, v / vmax)) * 100
        out.append(
            f'<div class="lab">{html.escape(label)}</div><div class="track">'
            f'<div class="fill" style="width:{w:.1f}%;background:{color}"></div></div>'
            f'<div class="val">{pct(v, 1)}</div>'
        )
    out.append("</div>")
    return "".join(out)


def _pbar(p: list[float], hk, ak, hit: int | None) -> str:
    segs = [(p[0], hk[0], hk[1]), (p[1], "rgba(239,238,230,.22)", CHALK), (p[2], ak[0], ak[1])]
    out = ['<div class="pbar">']
    for i, (v, bg, fg) in enumerate(segs):
        txt = pct(v) if v >= 0.09 else ""
        cls = ' class="hit"' if i == hit else ""
        out.append(f'<div{cls} style="width:{v * 100:.2f}%;background:{bg};color:{fg}">{txt}</div>')
    out.append("</div>")
    return "".join(out)


def _bars(r, hk, ak, hit) -> str:
    pm = [r.p_H, r.p_D, r.p_A]
    pk = [r.mkt_p_H, r.mkt_p_D, r.mkt_p_A]
    return (
        f'<div class="prow"><div class="who"><i style="background:{MODEL}"></i>Model</div>'
        f"{_pbar(pm, hk, ak, hit)}</div>"
        f'<div class="prow"><div class="who"><i style="background:{BOOK}"></i>Bookmakers</div>'
        f"{_pbar(pk, hk, ak, hit)}</div>"
    )


def _head(home: str, away: str, centre: str, hk, ak) -> str:
    return (
        '<div class="mhead">'
        f'<div class="team">{badge(home, kit=hk)}<span>{html.escape(home)}</span></div>'
        f'<div class="score">{centre}</div>'
        f'<div class="team away">{badge(away, kit=ak)}<span>{html.escape(away)}</span></div></div>'
    )


def pick_label(k: int, home: str, away: str) -> str:
    return [f"{home} to win", "the draw", f"{away} to win"][k]


def match_card(r, show_bet: bool = True, show_date: bool = True) -> str:
    """Match joué : probabilités du modèle et du marché, vainqueur du duel, ticket éventuel."""
    hk, ak = kits(r.home, r.away)
    y = int(r.outcome_idx)
    home, away = html.escape(r.home), html.escape(r.away)
    res = ["a home win", "a draw", "an away win"][y]
    if r.duel == "model":
        verdict = (
            f'<b class="m">The model saw it better</b>: {pct(r.pt_model)} on {res}, '
            f"against {pct(r.pt_market)} for the bookmakers."
        )
    elif r.duel == "book":
        verdict = (
            f'<b class="b">The bookmakers saw it better</b>: {pct(r.pt_market)} on {res}, '
            f"against {pct(r.pt_model)} for the model."
        )
    else:
        verdict = f'<b class="t">Level</b>: both gave {pct(r.pt_market)} to {res}.'
    out = ['<div class="mcard">']
    if show_date:
        out.append(f'<div class="date">{date_en(r.kickoff_date, weekday=True)}</div>')
    out.append(_head(r.home, r.away, f"{int(r.home_score)}-{int(r.away_score)}", hk, ak))
    out.append(_bars(r, hk, ak, y) + f'<div class="mfoot">{verdict}</div>')
    if show_bet and r.bet_idx >= 0:
        k = int(r.bet_idx)
        won = k == y
        gain = money(10 * (r.bet_odds - 1), signed=True) if won else money(-10)
        out.append(
            f'<div class="ticket"><span>The model bet €10 on '
            f"<b>{html.escape(pick_label(k, home, away))}</b> at {dec(r.bet_odds)}</span>"
            f'<span class="res {"won" if won else "lost"}">{"Won" if won else "Lost"} '
            f"{gain}</span></div>"
        )
    out.append("</div>")
    return "".join(out)


def upcoming_card(r) -> str:
    """Match à venir : probabilités du modèle et des bookmakers, et le pari que fait le modèle."""
    hk, ak = kits(r.home, r.away)
    time = r.kickoff_time if isinstance(r.kickoff_time, str) and r.kickoff_time else "vs"
    fav_m = int(pd.Series([r.p_H, r.p_D, r.p_A]).argmax())
    fav_b = int(pd.Series([r.mkt_p_H, r.mkt_p_D, r.mkt_p_A]).argmax())
    fav = [html.escape(r.home), "a draw", html.escape(r.away)]
    if fav_m != fav_b:
        foot = (
            f'<b class="m">They disagree.</b> The model favours {fav[fav_m]}, '
            f"the bookmakers favour {fav[fav_b]}."
        )
    else:
        foot = f"Both favour {fav[fav_m]}."
    out = [
        '<div class="mcard upcoming">',
        f'<div class="date">{date_en(r.kickoff_date, weekday=True, year=False)}, '
        f"{html.escape(data_league(r))}</div>",
        _head(r.home, r.away, html.escape(time), hk, ak),
        _bars(r, hk, ak, None),
        f'<div class="mfoot">{foot}</div>',
    ]
    if r.bet_idx >= 0:
        k = int(r.bet_idx)
        out.append(
            f'<div class="ticket"><span>The model bets €10 on '
            f"<b>{html.escape(pick_label(k, r.home, r.away))}</b> at {dec(r.bet_odds)}</span>"
            f'<span class="res">edge {pct(r.bet_edge)}</span></div>'
        )
    out.append("</div>")
    return "".join(out)


def data_league(r) -> str:
    return getattr(r, "league_name", r.league)


def card_grid(rows) -> str:
    return '<div class="cards">' + "".join(match_card(r) for _, r in rows.iterrows()) + "</div>"
