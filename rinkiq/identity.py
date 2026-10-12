"""Who is that? Jersey numbers → roster → player names.

1. **Read digits.** The Hockey-Vision number model finds single digits on an
   upscaled player crop. Two digits side by side form a two-digit number.
2. **Vote over time.** One frame can misread a "63" as "3" (an arm covers the
   6). Every track collects readings, and a number is confirmed once it wins
   clearly across several frames.
3. **Look it up.** Team + number → name, from the public NHL roster API or a
   roster you paste in (any league: "63 Zack Ostapchuk", one per line).
4. **Re-identify.** If a player's track broke in two (occlusion, leaving the
   frame), both pieces carry the same team + number, so they merge into one player.
"""
from __future__ import annotations

import json
import re
import urllib.request
from collections import Counter, defaultdict
from functools import lru_cache

import cv2
import numpy as np
import pandas as pd

NHL_TEAMS = {
    "ANA": "Anaheim Ducks", "BOS": "Boston Bruins", "BUF": "Buffalo Sabres", "CGY": "Calgary Flames",
    "CAR": "Carolina Hurricanes", "CHI": "Chicago Blackhawks", "COL": "Colorado Avalanche",
    "CBJ": "Columbus Blue Jackets", "DAL": "Dallas Stars", "DET": "Detroit Red Wings", "EDM": "Edmonton Oilers",
    "FLA": "Florida Panthers", "LAK": "Los Angeles Kings", "MIN": "Minnesota Wild", "MTL": "Montréal Canadiens",
    "NSH": "Nashville Predators", "NJD": "New Jersey Devils", "NYI": "New York Islanders",
    "NYR": "New York Rangers", "OTT": "Ottawa Senators", "PHI": "Philadelphia Flyers",
    "PIT": "Pittsburgh Penguins", "SJS": "San Jose Sharks", "SEA": "Seattle Kraken", "STL": "St. Louis Blues",
    "TBL": "Tampa Bay Lightning", "TOR": "Toronto Maple Leafs", "UTA": "Utah Mammoth", "VAN": "Vancouver Canucks",
    "VGK": "Vegas Golden Knights", "WSH": "Washington Capitals", "WPG": "Winnipeg Jets",
}


# ------------------------------------------------------------------ 1. read digits
def read_number(crop: np.ndarray, model, min_conf: float = 0.45) -> tuple[str, float] | None:
    """Best jersey number on one player crop, or None."""
    if crop.size == 0 or min(crop.shape[:2]) < 12:
        return None
    up = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    r = model(up, conf=min_conf, verbose=False)[0]
    if len(r.boxes) == 0:
        return None
    digits = [(b, model.names[int(k)], float(c)) for b, k, c in
              zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())]
    # drop duplicate detections of the same digit position (keep the more confident)
    digits.sort(key=lambda d: -d[2])
    kept = []
    for b, n, c in digits:
        if all(_overlap_x(b, k[0]) < 0.5 for k in kept):
            kept.append((b, n, c))
    best = max(kept, key=lambda d: d[2])
    # look for a neighbour that makes a two-digit number
    pair = None
    for b, n, c in kept:
        if b is best[0]:
            continue
        h = max(best[0][3] - best[0][1], 1)
        same_row = _overlap_y(b, best[0]) > 0.5
        gap = max(b[0], best[0][0]) - min(b[2], best[0][2])
        if same_row and gap < 0.6 * h and (pair is None or c > pair[2]):
            pair = (b, n, c)
    if pair is None:
        return best[1], best[2]
    left, right = sorted([best, pair], key=lambda d: d[0][0])
    if left[1] == "0":  # jerseys don't have leading zeros ("07" is really "7" … or a misread)
        return right[1], right[2]
    return left[1] + right[1], min(left[2], right[2])


def _overlap_x(a, b) -> float:
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    return inter / max(1e-6, min(a[2] - a[0], b[2] - b[0]))


def _overlap_y(a, b) -> float:
    inter = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return inter / max(1e-6, min(a[3] - a[1], b[3] - b[1]))


# ------------------------------------------------------------------ 2. vote
def vote_numbers(readings: list[tuple[int, str, float]], min_votes: int = 2,
                 min_share: float = 0.55) -> dict[int, tuple[str, float]]:
    """readings = [(track_id, number, conf)] → {track_id: (number, share of the vote)}.

    Two-digit readings get a bonus: a partly hidden "63" often reads as "3",
    almost never the other way round.
    """
    by_track: dict[int, Counter] = defaultdict(Counter)
    for tid, num, conf in readings:
        by_track[tid][num] += conf * (1.5 if len(num) == 2 else 1.0)
    out = {}
    for tid, votes in by_track.items():
        num, score = votes.most_common(1)[0]
        n_reads = sum(1 for t, n, _ in readings if t == tid and n == num)
        # fold "3" votes into a winning "63" (a sub-reading of it)
        support = score + sum(v for k, v in votes.items() if k != num and len(k) == 1 and k in num)
        share = support / sum(votes.values())
        if n_reads >= min_votes and share >= min_share:
            out[tid] = (num, round(share, 2))
    return out


# ------------------------------------------------------------------ 3. rosters
@lru_cache(maxsize=64)
def nhl_roster(team: str, season: str) -> dict[str, str]:
    """{"63": "Zack Ostapchuk", …} from the public NHL web API."""
    url = f"https://api-web.nhle.com/v1/roster/{team}/{season}"
    req = urllib.request.Request(url, headers={"User-Agent": "rinkiq/0.1"})
    with urllib.request.urlopen(req, timeout=10) as r:
        d = json.loads(r.read())
    out = {}
    for group in ("forwards", "defensemen", "goalies"):
        for p in d.get(group, []):
            if p.get("sweaterNumber") is not None:
                out[str(p["sweaterNumber"])] = f"{p['firstName']['default']} {p['lastName']['default']}"
    return out


def parse_roster(text: str) -> dict[str, str]:
    """Pasted roster, one player per line: '63 Zack Ostapchuk', '#9, Dmitry Orlov', '9 - Orlov'."""
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s*#?(\d{1,2})\s*[,.\-:|\t ]\s*(.+?)\s*$", line)
        if m:
            out[str(int(m.group(1)))] = m.group(2)
    return out


def label(track_id, names: dict | None = None) -> str:
    """Display name for a track: its player name/number if known, otherwise its tracker ID."""
    names = names or {}
    if track_id in names:
        return names[track_id]
    try:
        return f"ID {int(track_id)}"
    except (TypeError, ValueError):
        return str(track_id)


def short(name: str) -> str:
    first, _, last = name.partition(" ")
    return f"{first[0]}. {last}" if last else name


# ------------------------------------------------------------------ 4. assign + re-identify
def assign_names(numbers: dict[int, tuple[str, float]], team_of_track: dict[int, str],
                 rosters: dict[str, dict[str, str]]) -> dict[int, str]:
    """{track_id: '#63 Z. Ostapchuk'} (or '#63' when the number isn't on the roster)."""
    out = {}
    for tid, (num, _) in numbers.items():
        team = team_of_track.get(tid)
        name = rosters.get(team, {}).get(num)
        out[tid] = f"#{num} {short(name)}" if name else f"#{num}"
    return out


def merge_tracks(players: pd.DataFrame, numbers: dict[int, tuple[str, float]]) -> tuple[pd.DataFrame, dict[int, int]]:
    """Join track pieces that share team + number and never appear at the same time."""
    if players.empty or not numbers:
        return players, {}
    span = players.groupby("track_id").agg(t0=("t", "min"), t1=("t", "max"), label=("label", "first"))
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for tid, (num, _) in numbers.items():
        if tid in span.index:
            groups[(span.loc[tid, "label"], num)].append(tid)
    remap = {}
    for tids in groups.values():
        tids.sort(key=lambda t: span.loc[t, "t0"])
        head, end = tids[0], span.loc[tids[0], "t1"]
        for t in tids[1:]:
            if span.loc[t, "t0"] > end:       # no time overlap → same person, new track piece
                remap[t] = head
                end = max(end, span.loc[t, "t1"])
    if not remap:
        return players, {}
    players = players.copy()
    players["track_id"] = players["track_id"].replace(remap)
    return players, remap
