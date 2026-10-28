"""Thème « match de nuit » et composants HTML du dashboard.

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
NBSP = "\u00a0"  # espace insécable, avant % et entre milliers

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
PREFIXES = {
    "fc",
    "ac",
    "as",
    "sc",
    "cf",
    "rc",
    "us",
    "sv",
    "vfb",
    "vfl",
    "tsg",
    "cd",
    "ud",
    "sd",
    "rcd",
    "afc",
    "ogc",
    "fk",
    "sporting",
    "1",
    "ssc",
    "ss",
    "sk",
    "bv",
    "bsc",
    "fsv",
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


# --- formats ------------------------------------------------------------------------------------


def pct(x: float, d: int = 0) -> str:
    return f"{x * 100:.{d}f}".replace(".", ",") + f"{NBSP}%"


def num(n: float) -> str:
    return f"{int(round(n)):,}".replace(",", NBSP)


def money(x: float) -> str:
    return f"{num(x)}{NBSP}€"


def dec(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}".replace(".", ",")


MONTHS = [
    "janv.",
    "févr.",
    "mars",
    "avr.",
    "mai",
    "juin",
    "juil.",
    "août",
    "sept.",
    "oct.",
    "nov.",
    "déc.",
]


def date_fr(ts) -> str:
    ts = pd.Timestamp(ts)
    return f"{ts.day} {MONTHS[ts.month - 1]} {ts.year}"


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
        '<div class="brand"><h1><span class="m">Le modèle</span> contre '
        f'<span class="b">les bookmakers</span></h1><span>{html.escape(subtitle)}</span></div>'
    )


def scoreboard(model_wins: int, book_wins: int, ties: int, foot: str) -> str:
    return (
        f'<div class="board">{PITCH_SVG}<div class="row">'
        f'<div><div class="who m">Le modèle</div><div class="big">{num(model_wins)}</div>'
        f'<div class="kit" style="background:{MODEL}"></div></div>'
        '<div class="mid">–</div>'
        f'<div><div class="who b">Les bookmakers</div><div class="big">{num(book_wins)}</div>'
        f'<div class="kit" style="background:{BOOK}"></div></div></div>'
        f'<div class="foot">{foot}<br>{num(ties)} matchs à égalité '
        "(moins d’un point d’écart).</div>"
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


def _pbar(p: list[float], hk, ak, hit: int) -> str:
    segs = [(p[0], hk[0], hk[1]), (p[1], "rgba(239,238,230,.22)", CHALK), (p[2], ak[0], ak[1])]
    out = ['<div class="pbar">']
    for i, (v, bg, fg) in enumerate(segs):
        txt = pct(v) if v >= 0.09 else ""
        cls = ' class="hit"' if i == hit else ""
        out.append(f'<div{cls} style="width:{v * 100:.2f}%;background:{bg};color:{fg}">{txt}</div>')
    out.append("</div>")
    return "".join(out)


def match_card(r, show_bet: bool = True, show_date: bool = True) -> str:
    hk, ak = kits(r.home, r.away)
    y = int(r.outcome_idx)
    pm = [r.p_H, r.p_D, r.p_A]
    pk = [r.mkt_p_H, r.mkt_p_D, r.mkt_p_A]
    home, away = html.escape(r.home), html.escape(r.away)
    res_txt = [f"la victoire de {home}", "le nul", f"la victoire de {away}"][y]
    if r.duel == "model":
        verdict = (
            f'<b class="m">Le modèle a mieux vu</b> : {pct(r.pt_model)} pour {res_txt}, '
            f"contre {pct(r.pt_market)} chez les bookmakers."
        )
    elif r.duel == "book":
        verdict = (
            f'<b class="b">Les bookmakers ont mieux vu</b> : {pct(r.pt_market)} pour '
            f"{res_txt}, contre {pct(r.pt_model)} pour le modèle."
        )
    else:
        verdict = f'<b class="t">Égalité</b> : tous deux à {pct(r.pt_market)} pour {res_txt}.'
    out = ['<div class="mcard">']
    if show_date:
        out.append(f'<div class="date">{date_fr(r.kickoff_date)}</div>')
    out.append(
        '<div class="mhead">'
        f'<div class="team">{badge(r.home, kit=hk)}<span>{home}</span></div>'
        f'<div class="score">{int(r.home_score)}-{int(r.away_score)}</div>'
        f'<div class="team away">{badge(r.away, kit=ak)}<span>{away}</span></div></div>'
        f'<div class="prow"><div class="who"><i style="background:{MODEL}"></i>Modèle</div>'
        f"{_pbar(pm, hk, ak, y)}</div>"
        f'<div class="prow"><div class="who"><i style="background:{BOOK}"></i>Bookmakers</div>'
        f"{_pbar(pk, hk, ak, y)}</div>"
        f'<div class="mfoot">{verdict}</div>'
    )
    if show_bet and r.bet_idx >= 0:
        k = int(r.bet_idx)
        pick = [f"victoire de {home}", "match nul", f"victoire de {away}"][k]
        won = k == y
        gain = f"+{dec(10 * (r.bet_odds - 1))}{NBSP}€" if won else f"-10{NBSP}€"
        out.append(
            f'<div class="ticket"><span>Le modèle aurait misé 10{NBSP}€ sur <b>{pick}</b> à '
            f'{dec(r.bet_odds)}</span><span class="res {"won" if won else "lost"}">'
            f"{'Gagné' if won else 'Perdu'} : {gain}</span></div>"
        )
    out.append("</div>")
    return "".join(out)


def card_grid(rows) -> str:
    return '<div class="cards">' + "".join(match_card(r) for _, r in rows.iterrows()) + "</div>"
