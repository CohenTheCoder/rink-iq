"""Who played well, and why? Report cards, three stars and a team comparison.

Every grade is *relative to the other skaters in the same video*: each stat
becomes a percentile, the strongest and weakest percentiles become plain
sentences ("Top speed 24.1 mph, fastest of 18 skaters"), and a weighted blend
becomes a letter grade. Nothing is a black box: the reasons are the inputs.
"""
from __future__ import annotations

import html
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import identity

from .metrics import is_goalie

MIN_SECONDS = 2.0   # ignore fragments shorter than this


@dataclass
class Stat:
    col: str
    label: str
    higher_is_better: bool
    good: str           # sentence template for a strength
    bad: str            # sentence template for a weakness
    weight: float = 1.0


STATS = [
    Stat("top_speed_mph", "Top speed", True, "Top speed {v:.1f} mph", "Top speed only {v:.1f} mph", 1.2),
    Stat("avg_speed_mph", "Avg speed", True, "Kept moving: {v:.1f} mph on average", "Low tempo: {v:.1f} mph on average"),
    Stat("bursts_per_min", "Bursts / min", True, "Explosive: {n} hard acceleration(s) in {secs:.0f} s",
         "Few hard accelerations ({n} in {secs:.0f} s)"),
    Stat("distance_per_min", "Feet / min", True, "Covered a lot of ice ({dist:.0f} ft in {secs:.0f} s)",
         "Covered little ice ({dist:.0f} ft in {secs:.0f} s)", 0.8),
    Stat("in_position_pct", "In position %", True, "Positionally sound: in position {v:.0f}% of the time",
         "Out of position {inv:.0f}% of the time", 1.3),
    Stat("goal_side_pct", "Goal-side %", True, "Goal-side of the attacker {v:.0f}% of the time when defending",
         "Got beaten to the inside: goal-side only {v:.0f}% when defending", 1.1),
    Stat("gap_ft", "Gap (ft)", False, "Tight gap control ({v:.0f} ft from the nearest attacker)",
         "Loose gap ({v:.0f} ft from the nearest attacker)", 0.8),
    Stat("space_ft", "Space (ft)", True, "Found open ice: {v:.0f} ft of space on the attack",
         "Crowded on the attack ({v:.0f} ft of space)", 0.8),
    Stat("possession_per_min", "Possession s/min", True, "Drove play: {poss:.1f} s with the puck",
         "Rarely had the puck ({poss:.1f} s)", 0.9),
    Stat("shots_per_min", "Shots / min", True, "Shooter: {n} shot(s){hard}", "", 1.0),
    Stat("passes_per_min", "Passes / min", True, "Moved the puck: {n} pass(es)", "", 0.7),
    Stat("turnovers_per_min", "Turnovers / min", False, "", "Gave it away: {n} turnover(s)", 0.9),
]


@dataclass
class Card:
    track_id: int
    name: str
    team: str
    grade: str
    score: float
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    minutes: float = 0.0
    review_clips: int = 0


def _grade(score: float) -> str:
    for cut, g in [(85, "A+"), (75, "A"), (65, "B+"), (55, "B"), (45, "C+"), (35, "C")]:
        if score >= cut:
            return g
    return "D"


def _prepare(summary: pd.DataFrame) -> pd.DataFrame:
    s = summary[summary["team"].isin(["A", "B"]) & ~summary["label"].map(is_goalie)].copy()
    s = s[s["seconds"] >= MIN_SECONDS]
    minutes = (s["seconds"] / 60).clip(lower=1e-6)
    s["minutes"] = minutes
    s["distance_per_min"] = s["distance_ft"] / minutes
    for c in ("possession_s", "shots", "passes", "turnovers"):
        if c not in s:
            s[c] = 0.0
    s["possession_per_min"] = s["possession_s"] / minutes
    s["shots_per_min"] = s["shots"] / minutes
    s["passes_per_min"] = s["passes"] / minutes
    s["turnovers_per_min"] = s["turnovers"] / minutes
    return s


def report_cards(summary: pd.DataFrame, review: pd.DataFrame | None = None,
                 names: dict | None = None) -> list[Card]:
    names = names or {}
    s = _prepare(summary)
    if s.empty:
        return []
    n = len(s)
    pct = {}
    for st in STATS:
        if st.col in s and s[st.col].notna().sum() >= 2 and s[st.col].nunique() > 1:
            r = s[st.col].rank(pct=True, ascending=st.higher_is_better)
            pct[st.col] = r
    cards = []
    clips = review.groupby("track_id").size() if review is not None and not review.empty else pd.Series(dtype=int)
    for idx, row in s.iterrows():
        scores, weights, good, bad = [], [], [], []
        for st in STATS:
            if st.col not in pct or pd.isna(row.get(st.col)):
                continue
            p = float(pct[st.col].loc[idx])
            scores.append(p)
            weights.append(st.weight)
            v = float(row[st.col])
            rank = int(s[st.col].rank(ascending=not st.higher_is_better, method="min").loc[idx])
            fmt = dict(v=v, inv=100 - v if st.col.endswith("_pct") else v,
                       n=int(row.get(st.col.replace("_per_min", ""), 0) or 0), secs=float(row["seconds"]),
                       dist=float(row.get("distance_ft", 0) or 0), poss=float(row.get("possession_s", 0) or 0),
                       hard=(f", hardest {row['hardest_shot_mph']:.0f} mph"
                             if st.col == "shots_per_min" and pd.notna(row.get("hardest_shot_mph")) else ""))
            if p >= 0.75 and st.good and (not st.col.endswith("_per_min") or fmt["n"] > 0 or
                                          st.col in ("bursts_per_min", "distance_per_min", "possession_per_min")):
                good.append((p, st.good.format(**fmt) + (f" · #{rank} of {n}" if rank <= 3 else "")))
            elif p <= 0.25 and st.bad and (st.col != "turnovers_per_min" or fmt["n"] > 0):
                bad.append((p, st.bad.format(**fmt)))
        score = 100 * float(np.average(scores, weights=weights)) if scores else 50.0
        tid = int(row["track_id"])
        k = int(clips.get(tid, 0))
        if k:
            bad.append((0.2, f"{k} out-of-position moment(s) flagged for film review"))
        cards.append(Card(tid, identity.label(tid, names), row["team"], _grade(score), round(score, 1),
                          [t for _, t in sorted(good, reverse=True)[:3]],
                          [t for _, t in sorted(bad)[:3]], round(float(row["minutes"]), 1), k))
    cards.sort(key=lambda c: -c.score)
    return cards


def three_stars(cards: list[Card]) -> list[Card]:
    return cards[:3]


def team_comparison(summary: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    s = summary[summary["team"].isin(["A", "B"]) & ~summary["label"].map(is_goalie)]
    rows = {}
    for team, g in s.groupby("team"):
        ev = events[events["team"] == team] if not events.empty else events
        rows[f"Team {team}"] = {
            "Skaters tracked": len(g),
            "Avg speed (mph)": round(g["avg_speed_mph"].mean(), 1),
            "Top speed (mph)": round(g["top_speed_mph"].max(), 1),
            "Total distance (ft)": round(g["distance_ft"].sum()),
            "Possession (s)": round(g.get("possession_s", pd.Series(dtype=float)).sum(), 1),
            "Shots": int((ev["type"] == "shot").sum()) if not ev.empty else 0,
            "Passes": int((ev["type"] == "pass").sum()) if not ev.empty else 0,
            "Turnovers": int((ev["type"] == "turnover").sum()) if not ev.empty else 0,
            "In position %": round(g["in_position_pct"].mean(), 0) if "in_position_pct" in g else np.nan,
        }
    return pd.DataFrame(rows)


def html_report(cards: list[Card], teams: pd.DataFrame, title: str = "RinkIQ game report") -> str:
    """A self-contained HTML page you can download, print or email."""
    esc = html.escape
    color = {"A": "#2f6fd6", "B": "#d0393e"}

    def card_html(c: Card) -> str:
        li_g = "".join(f"<li>✅ {esc(t)}</li>" for t in c.strengths) or "<li>—</li>"
        li_b = "".join(f"<li>⚠️ {esc(t)}</li>" for t in c.weaknesses) or "<li>—</li>"
        return (f"<div class='card' style='border-top:6px solid {color.get(c.team, '#888')}'>"
                f"<div class='grade'>{c.grade}</div><h3>{esc(c.name)}</h3>"
                f"<div class='sub'>Team {c.team} · score {c.score:.0f}/100 · {c.minutes:.1f} min tracked</div>"
                f"<ul>{li_g}</ul><ul>{li_b}</ul></div>")

    stars = "".join(f"<div class='star'><span>{'⭐' * (3 - i)}</span> <b>{esc(c.name)}</b> "
                    f"(Team {c.team}, {c.grade}): {esc(c.strengths[0]) if c.strengths else ''}</div>"
                    for i, c in enumerate(cards[:3]))
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{esc(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{{font-family:Inter,Segoe UI,Helvetica,Arial,sans-serif;margin:24px;color:#0f1b2d;background:#fff}}
h1{{margin:0 0 4px}} .muted{{color:#5b6b7d}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px;margin-top:14px}}
.card{{background:#f4f8fb;border-radius:12px;padding:14px 16px;position:relative}}
.card h3{{margin:0 0 2px}} .sub{{color:#5b6b7d;font-size:13px}} ul{{padding-left:18px;margin:8px 0}}
.grade{{position:absolute;right:14px;top:10px;font-size:30px;font-weight:800}}
.star{{font-size:17px;margin:4px 0}} table{{border-collapse:collapse;margin-top:8px}}
td,th{{border-bottom:1px solid #dde5ee;padding:6px 12px;text-align:right}} th:first-child,td:first-child{{text-align:left}}
</style></head><body>
<h1>🏒 {esc(title)}</h1><div class="muted">Grades are relative to the other skaters in this video.</div>
<h2>Three stars</h2>{stars}
<h2>Team comparison</h2>{teams.to_html(border=0, na_rep='—')}
<h2>Report cards</h2><div class="grid">{''.join(card_html(c) for c in cards)}</div>
<p class="muted">Generated by RinkIQ: computer-vision tracking → stats. Positioning stats need a calibrated rink.</p>
</body></html>"""
