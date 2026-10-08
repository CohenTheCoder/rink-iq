"""A small synthetic hockey sequence with known ground truth.

Why simulate? Two reasons:
* The analytics layer can be tested with numbers we *know* (e.g. an 88 mph
  shot at t≈10.6 s) instead of trusting the detector.
* The app has an instant demo that works offline, no video needed.

The play: Team A breaks out, enters the zone, works it to the point for a
one-timer; Team B recovers the rebound and counter-attacks for a wrist shot.
Team B's left winger "cheats" up ice while defending — a deliberate
out-of-position moment the analytics should flag.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .pipeline import PLAYER_COLUMNS, PUCK_COLUMNS
from .rink import CENTER_Y, GOAL_LINE_LEFT, GOAL_LINE_RIGHT, RINK_LENGTH, RINK_WIDTH

MPH = 1.46667  # ft/s per mph
ROLES = ["C", "LW", "RW", "LD", "RD"]
# Lateral slot for each role, in the *team's own* frame (+ = its left side when attacking).
LATERAL = {"C": 0.0, "LW": 20.0, "RW": -20.0, "LD": 17.0, "RD": -17.0}

SCRIPT = [
    dict(kind="carry", who=("A", "LD"), to=(38, 28), until=2.5),
    dict(kind="pass", to=("A", "C"), mph=45),
    dict(kind="carry", who=("A", "C"), to=(132, 48), until=7.0),
    dict(kind="pass", to=("A", "RW"), mph=50),
    dict(kind="carry", who=("A", "RW"), to=(160, 18), until=9.2),
    dict(kind="pass", to=("A", "LD"), mph=55),
    dict(kind="carry", who=("A", "LD"), to=(142, 66), until=10.6),
    dict(kind="shot", mph=88, then=("B", "RD")),
    dict(kind="carry", who=("B", "RD"), to=(172, 24), until=13.0),
    dict(kind="pass", to=("B", "C"), mph=48),
    dict(kind="carry", who=("B", "C"), to=(80, 40), until=16.5),
    dict(kind="pass", to=("B", "LW"), mph=50),
    dict(kind="carry", who=("B", "LW"), to=(45, 22), until=18.6),
    dict(kind="shot", mph=72, then=None),
]
CHEATER = ("B", "LW")
CHEAT_WINDOW = (7.0, 10.5)


def _attacks_right(team: str) -> bool:
    return team == "A"


def _net(team: str) -> np.ndarray:  # the net a team defends
    return np.array([GOAL_LINE_LEFT if _attacks_right(team) else GOAL_LINE_RIGHT, CENTER_Y])


def _target(team: str, role: str, puck: np.ndarray, has_puck: bool, t: float) -> np.ndarray:
    fwd = 1.0 if _attacks_right(team) else -1.0
    lat = LATERAL[role] * (1.0 if _attacks_right(team) else -1.0)
    if has_puck:
        ahead = {"C": 6, "LW": 14, "RW": 14, "LD": -28, "RD": -28}[role]
        tgt = puck + np.array([fwd * ahead, lat])
        if role in ("LD", "RD"):  # D stay near the blue line once in the zone
            blue = 125.0 if fwd > 0 else 75.0
            tgt[0] = max(tgt[0], blue + 3) if fwd > 0 and puck[0] > 125 else tgt[0]
            tgt[0] = min(tgt[0], blue - 3) if fwd < 0 and puck[0] < 75 else tgt[0]
    else:
        net = _net(team)
        frac = {"C": 0.8, "LW": 0.62, "RW": 0.62, "LD": 0.32, "RD": 0.32}[role]
        tgt = net + frac * (puck - net) + np.array([0.0, lat * 0.7])
        if (team, role) == CHEATER and CHEAT_WINDOW[0] <= t <= CHEAT_WINDOW[1]:
            tgt = puck + np.array([fwd * 45.0, lat])  # floats up ice, behind the play
    return np.clip(tgt, [4, 4], [RINK_LENGTH - 4, RINK_WIDTH - 4])


def simulate_game(seconds: float = 20.0, fps: float = 30.0, seed: int = 7, pixel_noise_ft: float = 0.3,
                  puck_dropout: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    dt = 1.0 / fps
    keys = [(tm, r) for tm in ("A", "B") for r in ROLES]
    vmax = {k: rng.uniform(29, 35) for k in keys}  # 20-24 mph top speed
    amax = 16.0
    pos = {}
    for tm, r in keys:
        side = 60.0 if tm == "A" else 140.0
        pos[(tm, r)] = np.array([side + rng.uniform(-10, 10), CENTER_Y + LATERAL[r] * (1 if tm == "A" else -1)])
    vel = {k: np.zeros(2) for k in keys}
    speed_log = {k: [] for k in keys}
    goalies = {"A": np.array([GOAL_LINE_LEFT + 4, CENTER_Y]), "B": np.array([GOAL_LINE_RIGHT - 4, CENTER_Y])}
    ref = np.array([100.0, 80.0])
    ids = {k: i + 1 for i, k in enumerate(keys)}
    ids.update({("A", "G"): 11, ("B", "G"): 12, ("REF", "R"): 13})

    step_i = 0
    state = dict(mode="carry", owner=SCRIPT[0]["who"])
    puck = pos[SCRIPT[0]["who"]].copy()
    flight = None
    player_rows, puck_rows = [], []
    n = int(seconds * fps)
    for f in range(n):
        t = f * dt
        step = SCRIPT[step_i] if step_i < len(SCRIPT) else None
        owner = state.get("owner")
        owner_team = owner[0] if owner else None

        # --- puck logic -------------------------------------------------
        if state["mode"] == "carry" and step and step["kind"] == "carry" and t >= step["until"]:
            step_i += 1
            step = SCRIPT[step_i] if step_i < len(SCRIPT) else None
            if step and step["kind"] in ("pass", "shot"):
                if step["kind"] == "pass":
                    dest_key = step["to"]
                    flight = dict(dest_key=dest_key, speed=step["mph"] * MPH)
                else:
                    shooter_team = owner[0]
                    net_x = GOAL_LINE_RIGHT if _attacks_right(shooter_team) else GOAL_LINE_LEFT
                    flight = dict(dest_pt=np.array([net_x, CENTER_Y + 1.5]), speed=step["mph"] * MPH,
                                  then=step["then"])
                state = dict(mode="flight")
                step_i += 1
        if state["mode"] == "flight":
            dest = pos[flight["dest_key"]] if "dest_key" in flight else flight["dest_pt"]
            d = dest - puck
            dist = np.linalg.norm(d)
            move = flight["speed"] * dt
            if dist <= move + 1.0:
                puck = dest.copy()
                if "dest_key" in flight:
                    state = dict(mode="carry", owner=flight["dest_key"])
                elif flight.get("then"):  # rebound off the goalie to a defender
                    flight = dict(dest_key=flight["then"], speed=30 * MPH)
                else:
                    state = dict(mode="dead")
            else:
                puck = puck + d / dist * move
        if state["mode"] == "carry":
            seg = SCRIPT[step_i] if step_i < len(SCRIPT) else None
            ok = seg["to"] if seg and seg["kind"] == "carry" else pos[state["owner"]]
            state["waypoint"] = np.array(ok, dtype=float)

        owner = state.get("owner") if state["mode"] == "carry" else None
        owner_team = owner[0] if owner else (None if state["mode"] != "flight" else "flight")

        # --- skaters ----------------------------------------------------
        for k in keys:
            tm, r = k
            if owner == k:
                tgt = state["waypoint"]
            else:
                has = owner_team == tm or (owner_team == "flight" and flight and "dest_key" in flight
                                           and flight["dest_key"][0] == tm)
                tgt = _target(tm, r, puck, has, t)
            desired = (tgt - pos[k]) * 1.4
            sp = np.linalg.norm(desired)
            if sp > vmax[k]:
                desired *= vmax[k] / sp
            dv = desired - vel[k]
            a = np.linalg.norm(dv) / dt
            if a > amax:
                dv *= amax / a
            vel[k] = vel[k] + dv + rng.normal(0, 0.15, 2)
            pos[k] = np.clip(pos[k] + vel[k] * dt, [2, 2], [RINK_LENGTH - 2, RINK_WIDTH - 2])
            speed_log[k].append(np.linalg.norm(vel[k]) / MPH)
        if owner is not None:
            heading = vel[owner] / (np.linalg.norm(vel[owner]) + 1e-9)
            puck = pos[owner] + heading * 2.0

        for tm in ("A", "B"):  # goalies square up to the puck
            net = _net(tm)
            goalies[tm] = net + np.array([4.0 if tm == "A" else -4.0, np.clip((puck[1] - CENTER_Y) * 0.15, -3, 3)])
        ref = ref + (np.array([np.clip(puck[0], 30, 170), 80.0]) - ref) * 0.05

        def facing(k):
            tm, r = k
            v = vel[k]
            defending = owner_team not in (tm, None, "flight")
            if defending and r in ("LD", "RD"):  # D-men face the play while retreating
                d = puck - pos[k]
                return d / (np.linalg.norm(d) + 1e-9)
            if np.linalg.norm(v) > 3:
                return v / np.linalg.norm(v)
            d = puck - pos[k]
            return d / (np.linalg.norm(d) + 1e-9)

        for k in keys:
            label = f"team_{k[0].lower()}_player"
            p = pos[k] + rng.normal(0, pixel_noise_ft, 2)
            fv = facing(k)
            player_rows.append([f, t, ids[k], label, 0.9, p[0], p[1]] + [np.nan] * 7 + [fv[0], fv[1], 0.8])
        for tm in ("A", "B"):
            g = goalies[tm] + rng.normal(0, pixel_noise_ft, 2)
            player_rows.append([f, t, ids[(tm, "G")], f"goalie_{tm.lower()}", 0.9, g[0], g[1]] + [np.nan] * 7
                               + [1.0 if tm == "A" else -1.0, 0.0, 0.8])
        player_rows.append([f, t, 13, "referee", 0.9, ref[0], ref[1]] + [np.nan] * 7 + [0.0, -1.0, 0.8])
        if state["mode"] != "dead" and rng.random() > puck_dropout:
            pp = puck + rng.normal(0, pixel_noise_ft, 2)
            puck_rows.append([f, t, pp[0], pp[1], np.nan, np.nan, 0.8])

    players = pd.DataFrame(player_rows, columns=PLAYER_COLUMNS)
    puck_df = pd.DataFrame(puck_rows, columns=PUCK_COLUMNS)
    names = {ids[k]: f"{k[0]}-{k[1]}" for k in ids}
    half_s = int(fps / 2)  # same definition as the stats: fastest 0.5 s average
    truth = dict(top_speed_mph={ids[k]: float(pd.Series(speed_log[k]).rolling(half_s).mean().max())
                                for k in keys},
                 shots=[(s["mph"]) for s in SCRIPT if s["kind"] == "shot"],
                 passes=sum(s["kind"] == "pass" for s in SCRIPT), cheater=ids[CHEATER])
    meta = dict(fps=fps, frame_stride=1, calibrated=True, units="ft", synthetic=True, names=names, truth=truth)
    return players, puck_df, meta
