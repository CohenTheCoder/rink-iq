"""Step 4 — Turn x/y tracks into hockey stats.

Input: the `players` and `puck` DataFrames from `pipeline.run_video` (or
`simulate.simulate_game`). Positions are in feet; time `t` is in seconds.

Everything here is plain pandas/numpy so each stat is easy to audit:

| Stat | How it's computed |
|---|---|
| Speed / top speed | glitches dropped → Savitzky–Golay-smoothed position → derivative → mph; top speed = fastest 0.5 s average |
| Distance skated | sum of smoothed step lengths |
| Bursts | stretches ≥ 0.3 s with acceleration > 8 ft/s² |
| High-intensity % | share of time above 15 mph |
| Skating backwards % | body facing points opposite to travel (needs pose) |
| Zone time | OZ / NZ / DZ split using the team's attack direction |
| Possession | nearest skater within 6 ft of the puck |
| Passes / turnovers | possession changes to a teammate / an opponent |
| Shots + shot speed | puck accelerates > 35 mph out of a player's possession, toward a net |
| Goal-side % | when defending, closer to own net than the nearest attacker |
| Structure deviation | distance from the player's usual spot relative to team shape |
| Out of position | not goal-side while defending, or > 25 ft off their usual spot |
| Space | distance to nearest opponent while their team has the puck |
| Skate Score | 0–100 percentile composite of the above (weights below) |
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull, QhullError

from .rink import CENTER_Y, GOAL_LINE_LEFT, GOAL_LINE_RIGHT, zone_of

FTPS_TO_MPH = 0.681818
MAX_SKATER_MPH = 30.0
MAX_PUCK_MPH = 110.0
HIGH_INTENSITY_MPH = 15.0
BURST_ACCEL = 8.0          # ft/s²
BURST_MIN_S = 0.3
POSSESSION_FT = 6.0
SHOT_MIN_MPH = 35.0
OUT_OF_SLOT_FT = 25.0
SKATE_SCORE_WEIGHTS = {"top_speed_mph": 0.25, "avg_speed_mph": 0.15, "bursts_per_min": 0.2,
                       "in_position_pct": 0.2, "space_ft": 0.2}


def team_of(label: str) -> str:
    if label in ("team_a_player", "goalie_a"):
        return "A"
    if label in ("team_b_player", "goalie_b"):
        return "B"
    return "REF" if label == "referee" else "?"


def is_goalie(label: str) -> bool:
    return label.startswith("goalie")


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < 5:
        return values
    w = min(window, len(values) if len(values) % 2 else len(values) - 1)
    w = max(5, w | 1)
    if w > len(values):
        return values
    return savgol_filter(values, w, 2)


def clean_track(t: np.ndarray, x: np.ndarray, y: np.ndarray, max_mph: float = 40.0,
                patience: int = 5) -> np.ndarray:
    """Segment labels for one track (-1 = outlier sample to drop).

    A real skater can't teleport. Any sample that implies > 40 mph from the
    last good one is a glitch (box jitter, a stick, a mis-matched ID) and is
    dropped. If the glitch persists for `patience` samples it was really an ID
    switch, so we start a new segment there instead of drawing a fake sprint.
    """
    vmax = max_mph / FTPS_TO_MPH
    seg = np.full(len(t), -1)
    if len(t) == 0:
        return seg
    seg[0], last, current, misses = 0, 0, 0, 0
    for i in range(1, len(t)):
        dt = max(t[i] - t[last], 1e-6)
        if np.hypot(x[i] - x[last], y[i] - y[last]) / dt <= vmax:
            seg[i], last, misses = current, i, 0
        else:
            misses += 1
            if misses >= patience:
                current += 1
                seg[i], last, misses = current, i, 0
    return seg


def kinematics(players: pd.DataFrame, fps: float, min_track_s: float = 0.5) -> pd.DataFrame:
    """Add smoothed position, velocity, speed (mph) and acceleration per track."""
    if players.empty:
        return players.assign(sx=[], sy=[], vx=[], vy=[], speed_mph=[], accel=[], team=[])
    window = int(round(fps * 0.7)) | 1
    pieces = []
    for tid, g in players.sort_values("t").groupby("track_id"):
        g = g.drop_duplicates("t").copy()
        g["segment"] = clean_track(g["t"].to_numpy(), g["x"].to_numpy(), g["y"].to_numpy())
        pieces += [s for _, s in g[g["segment"] >= 0].groupby("segment")]
    out = []
    for g in pieces:
        if g["t"].iloc[-1] - g["t"].iloc[0] < min_track_s or len(g) < 5:
            continue
        g = g.copy()
        t = g["t"].to_numpy()
        g["sx"] = _smooth(g["x"].to_numpy(), window)
        g["sy"] = _smooth(g["y"].to_numpy(), window)
        g["vx"] = np.gradient(g["sx"].to_numpy(), t)
        g["vy"] = np.gradient(g["sy"].to_numpy(), t)
        speed = np.hypot(g["vx"], g["vy"]) * FTPS_TO_MPH
        g["speed_mph"] = np.where(speed > MAX_SKATER_MPH, np.nan, speed)
        g["accel"] = np.gradient(_smooth(np.nan_to_num(g["speed_mph"].to_numpy() / FTPS_TO_MPH), window), t)
        out.append(g)
    if not out:
        return players.iloc[0:0].assign(sx=[], sy=[], vx=[], vy=[], speed_mph=[], accel=[], team=[])
    df = pd.concat(out, ignore_index=True)
    df["team"] = df["label"].map(team_of)
    return df


def puck_kinematics(puck: pd.DataFrame, fps: float, max_gap_s: float = 0.2) -> pd.DataFrame:
    """Fill short detection gaps, smooth lightly, and compute puck speed."""
    if puck.empty or len(puck) < 3:
        return puck.assign(speed_mph=np.nan, vx=np.nan, vy=np.nan)
    p = puck.sort_values("t").drop_duplicates("t").set_index("t")
    grid = np.round(np.arange(p.index.min(), p.index.max() + 1e-9, 1 / fps), 6)
    p.index = np.round(p.index, 6)
    p = p.reindex(np.union1d(p.index, grid))
    limit = max(1, int(round(max_gap_s * fps)))
    p[["x", "y"]] = p[["x", "y"]].interpolate(limit=limit, limit_area="inside")
    p = p.loc[grid].dropna(subset=["x", "y"]).reset_index().rename(columns={"index": "t"})
    if len(p) < 3:
        return p.assign(speed_mph=np.nan, vx=np.nan, vy=np.nan)
    window = max(3, int(round(fps * 0.1)) | 1)
    sx, sy = _smooth(p["x"].to_numpy(), window), _smooth(p["y"].to_numpy(), window)
    t = p["t"].to_numpy()
    p["vx"], p["vy"] = np.gradient(sx, t), np.gradient(sy, t)
    speed = np.hypot(p["vx"], p["vy"]) * FTPS_TO_MPH
    p["speed_mph"] = np.where(speed > MAX_PUCK_MPH, np.nan, speed)
    return p


def attack_directions(k: pd.DataFrame, default_a_attacks_right: bool = True) -> dict[str, bool]:
    """Which way each team attacks. A goalie's position gives away the net it defends."""
    a_right = default_a_attacks_right
    goalies = k[k["label"].map(is_goalie)]
    if not goalies.empty:
        med = goalies.groupby("label")["sx"].median()
        if "goalie_a" in med:
            a_right = med["goalie_a"] < 100
        elif "goalie_b" in med:
            a_right = med["goalie_b"] > 100
    return {"A": a_right, "B": not a_right}


def own_net(attacks_right: bool) -> np.ndarray:
    return np.array([GOAL_LINE_LEFT if attacks_right else GOAL_LINE_RIGHT, CENTER_Y])


def possession(k: pd.DataFrame, pk: pd.DataFrame) -> pd.DataFrame:
    """Per puck frame: the nearest skater within POSSESSION_FT (or nobody)."""
    if pk.empty or k.empty:
        return pd.DataFrame(columns=["t", "owner", "team", "dist"])
    skaters = k[~k["label"].map(is_goalie) & (k["team"].isin(["A", "B"]))]
    rows = []
    by_t = {t: g for t, g in skaters.groupby(np.round(skaters["t"], 6))}
    for t, x, y in zip(pk["t"], pk["x"], pk["y"]):
        g = by_t.get(round(t, 6))
        if g is None or g.empty:
            rows.append((t, -1, None, np.nan))
            continue
        d = np.hypot(g["sx"] - x, g["sy"] - y).to_numpy()
        i = int(np.argmin(d))
        if d[i] <= POSSESSION_FT:
            rows.append((t, int(g["track_id"].iloc[i]), g["team"].iloc[i], float(d[i])))
        else:
            rows.append((t, -1, None, float(d[i])))
    return pd.DataFrame(rows, columns=["t", "owner", "team", "dist"])


def _runs(owner: pd.Series, min_len: int) -> list[tuple[int, int, int]]:
    """(owner, start_idx, end_idx) for stable possession runs of at least min_len samples."""
    runs, start = [], 0
    vals = owner.to_numpy()
    for i in range(1, len(vals) + 1):
        if i == len(vals) or vals[i] != vals[start]:
            if vals[start] >= 0 and i - start >= min_len:
                runs.append((int(vals[start]), start, i - 1))
            start = i
    return runs


def _release_speed(after: pd.DataFrame) -> float:
    """Robust release speed: median of the samples near the peak (one noisy frame can't inflate it)."""
    s = after["speed_mph"].dropna()
    if s.empty:
        return np.nan
    return float(s[s >= 0.8 * s.max()].median())


def detect_events(poss: pd.DataFrame, pk: pd.DataFrame, k: pd.DataFrame, attack: dict[str, bool],
                  fps: float) -> pd.DataFrame:
    """Passes, turnovers and shots from possession changes and puck speed.

    Every time a stable possession ends we look at the next 0.4 s of puck flight:
    fast (> 35 mph) and heading at the opponent's net → a **shot** (even if the
    other team picks up the rebound). Otherwise the next owner decides it:
    a teammate → **pass**, an opponent → **turnover**.
    """
    cols = ["t", "type", "player", "team", "to_player", "speed_mph", "x", "y"]
    if poss.empty or pk.empty:
        return pd.DataFrame(columns=cols)
    team_of_track = k.drop_duplicates("track_id").set_index("track_id")["team"].to_dict()
    runs = _runs(poss["owner"], min_len=max(2, int(0.15 * fps)))
    events = []
    for i, (o, s, e) in enumerate(runs):
        nxt = runs[i + 1] if i + 1 < len(runs) else None
        if nxt is not None and nxt[0] == o:
            continue
        t_rel = poss["t"].iloc[e]
        team = team_of_track.get(o)
        after = pk[(pk["t"] > t_rel) & (pk["t"] <= t_rel + 0.4)]
        speed = _release_speed(after) if not after.empty else np.nan
        x0, y0 = (float(after["x"].iloc[0]), float(after["y"].iloc[0])) if not after.empty else (np.nan, np.nan)
        to_team = team_of_track.get(nxt[0]) if nxt is not None else None
        quick_teammate = (nxt is not None and to_team == team and poss["t"].iloc[nxt[1]] - t_rel < 1.0)
        is_shot = False
        if team in attack and speed >= SHOT_MIN_MPH and not quick_teammate:
            net_x = GOAL_LINE_RIGHT if attack[team] else GOAL_LINE_LEFT
            vx = after.loc[after["speed_mph"].idxmax(), "vx"]
            is_shot = np.sign(vx) == np.sign(net_x - x0)
        if is_shot:
            events.append((t_rel, "shot", o, team, -1, speed, x0, y0))
        elif nxt is not None:
            kind = "pass" if to_team == team else "turnover"
            events.append((t_rel, kind, o, team, nxt[0], speed, x0, y0))
    return pd.DataFrame(events, columns=cols)


def frame_context(k: pd.DataFrame, poss: pd.DataFrame, attack: dict[str, bool]) -> pd.DataFrame:
    """Per player-frame: nearest opponent, goal-side, structure deviation, in/out of position."""
    if k.empty:
        return k
    df = k.copy()
    df["tk"] = np.round(df["t"], 6)
    teams = df["team"].isin(["A", "B"]) & ~df["label"].map(is_goalie)
    # Which team has the puck at each time (forward-filled briefly through loose-puck moments).
    if not poss.empty:
        pt = poss.assign(tk=np.round(poss["t"], 6)).set_index("tk")["team"]
        pt = pt.ffill(limit=15)
        df["poss_team"] = df["tk"].map(pt)
    else:
        df["poss_team"] = None

    # Team shape: centroid per team per frame, then each player's offset from it.
    sk = df[teams]
    cent = sk.groupby(["tk", "team"])[["sx", "sy"]].mean().rename(columns={"sx": "cx", "sy": "cy"})
    df = df.join(cent, on=["tk", "team"])
    df["off_x"], df["off_y"] = df["sx"] - df["cx"], df["sy"] - df["cy"]
    # A player's "usual spot" differs with and without the puck, so learn one per phase.
    df["phase"] = np.where(df["poss_team"].isna(), "loose", np.where(df["poss_team"] == df["team"], "att", "def"))
    slot = df[teams].groupby(["track_id", "phase"])[["off_x", "off_y"]].median().rename(
        columns={"off_x": "slot_x", "off_y": "slot_y"})
    df = df.join(slot, on=["track_id", "phase"])
    df["slot_dev_ft"] = np.hypot(df["off_x"] - df["slot_x"], df["off_y"] - df["slot_y"])

    nearest, goal_side = np.full(len(df), np.nan), np.full(len(df), np.nan)
    groups = {key: g for key, g in df[teams].groupby(["tk", "team"])}
    pos = {i: p for p, i in enumerate(df.index)}
    for (tk, team), g in groups.items():
        opp_team = "B" if team == "A" else "A"
        opp = groups.get((tk, opp_team))
        if opp is None or opp.empty:
            continue
        net = own_net(attack[team])
        ox, oy = opp["sx"].to_numpy(), opp["sy"].to_numpy()
        for idx, x, y in zip(g.index, g["sx"], g["sy"]):
            d = np.hypot(ox - x, oy - y)
            j = int(np.argmin(d))
            nearest[pos[idx]] = d[j]
            mine = np.hypot(*(np.array([x, y]) - net))
            theirs = np.hypot(ox[j] - net[0], oy[j] - net[1])
            goal_side[pos[idx]] = float(mine <= theirs + 1.0)
    df["nearest_opp_ft"] = nearest
    df["goal_side"] = goal_side
    df["defending"] = df["poss_team"].notna() & (df["poss_team"] != df["team"]) & teams
    df["zone"] = [zone_of(x, attack.get(tm, True)) if tm in attack else None for x, tm in zip(df["sx"], df["team"])]
    df["out_of_position"] = teams & ((df["defending"] & (df["goal_side"] == 0))
                                     | (df["slot_dev_ft"] > OUT_OF_SLOT_FT))
    return df.drop(columns=["tk"])


def _sustained_top_speed(g: pd.DataFrame, seconds: float = 0.5) -> float:
    """Fastest speed held for half a second (a single noisy frame can't set the record)."""
    s = g.set_index(pd.to_timedelta(g["t"], unit="s"))["speed_mph"].dropna()
    if s.empty:
        return np.nan
    rolled = s.rolling(f"{int(seconds * 1000)}ms", min_periods=3).mean()
    return float(rolled.max()) if rolled.notna().any() else float(s.median())


def _bursts(g: pd.DataFrame) -> int:
    hot = (g["accel"] > BURST_ACCEL).to_numpy()
    t = g["t"].to_numpy()
    n, start = 0, None
    for i, h in enumerate(hot):
        if h and start is None:
            start = t[i]
        if (not h or i == len(hot) - 1) and start is not None:
            if t[i] - start >= BURST_MIN_S:
                n += 1
            start = None
    return n


def player_summary(ctx: pd.DataFrame, poss: pd.DataFrame, events: pd.DataFrame, calibrated: bool) -> pd.DataFrame:
    rows = []
    poss_time = poss[poss["owner"] >= 0].groupby("owner").size() if not poss.empty else pd.Series(dtype=float)
    dt_poss = np.median(np.diff(poss["t"])) if len(poss) > 1 else 0
    for tid, g in ctx.groupby("track_id"):
        g = g.sort_values("t")
        dur = g["t"].iloc[-1] - g["t"].iloc[0]
        steps = np.hypot(np.diff(g["sx"]), np.diff(g["sy"]))
        row = dict(
            track_id=tid, label=g["label"].iloc[0], team=g["team"].iloc[0], seconds=round(dur, 1),
            distance_ft=float(np.nansum(steps)),
            avg_speed_mph=float(np.nanmean(g["speed_mph"])),
            top_speed_mph=_sustained_top_speed(g),
            max_accel_ftps2=float(np.nanmax(g["accel"])) if g["accel"].notna().any() else np.nan,
            bursts=_bursts(g),
            high_intensity_pct=float(100 * np.nanmean(g["speed_mph"] > HIGH_INTENSITY_MPH)),
        )
        row["bursts_per_min"] = row["bursts"] / max(dur / 60, 1e-6)
        facing = g[["face_x", "face_y"]].to_numpy()
        vel = g[["vx", "vy"]].to_numpy()
        moving = (g["speed_mph"] > 5).to_numpy() & ~np.isnan(facing).any(axis=1)
        if moving.sum() >= 3:
            v = vel[moving] / (np.linalg.norm(vel[moving], axis=1, keepdims=True) + 1e-9)
            dots = np.sum(v * facing[moving], axis=1)
            row["backward_pct"] = float(100 * np.mean(dots < -0.3))
            row["facing_samples"] = int(moving.sum())
        else:
            row["backward_pct"], row["facing_samples"] = np.nan, int(moving.sum())
        if calibrated:
            zones = g["zone"].value_counts(normalize=True)
            for z in ("OZ", "NZ", "DZ"):
                row[f"{z}_pct"] = float(100 * zones.get(z, 0.0))
            d = g[g["defending"]]
            row["goal_side_pct"] = float(100 * d["goal_side"].mean()) if d["goal_side"].notna().any() else np.nan
            row["gap_ft"] = float(d["nearest_opp_ft"].mean()) if len(d) else np.nan
        row["slot_dev_ft"] = float(g["slot_dev_ft"].mean()) if "slot_dev_ft" in g else np.nan
        skater = row["team"] in ("A", "B") and not is_goalie(row["label"])
        row["in_position_pct"] = float(100 * (1 - g["out_of_position"].mean())) if skater else np.nan
        att = g[g["poss_team"] == g["team"]] if "poss_team" in g else g.iloc[0:0]
        row["space_ft"] = float(att["nearest_opp_ft"].mean()) if len(att) else float(g["nearest_opp_ft"].mean())
        row["possession_s"] = float(poss_time.get(tid, 0) * dt_poss)
        if not events.empty:
            ev = events[events["player"] == tid]
            row["passes"] = int((ev["type"] == "pass").sum())
            row["turnovers"] = int((ev["type"] == "turnover").sum())
            row["shots"] = int((ev["type"] == "shot").sum())
            row["hardest_shot_mph"] = float(ev.loc[ev["type"] == "shot", "speed_mph"].max()) if row["shots"] else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    skaters = summary["team"].isin(["A", "B"]) & ~summary["label"].map(is_goalie)
    score = np.zeros(len(summary))
    weight_used = 0.0
    for col, w in SKATE_SCORE_WEIGHTS.items():
        if col in summary and summary.loc[skaters, col].notna().any():
            score += w * summary[col].where(skaters).rank(pct=True).fillna(0).to_numpy()
            weight_used += w
    summary["skate_score"] = np.where(skaters, np.round(100 * score / max(weight_used, 1e-9)), np.nan)
    return summary.sort_values(["skate_score", "top_speed_mph"], ascending=False).reset_index(drop=True)


def team_shape(ctx: pd.DataFrame) -> pd.DataFrame:
    """Per frame & team: width, depth and area of the skaters' convex hull."""
    rows = []
    sk = ctx[ctx["team"].isin(["A", "B"]) & ~ctx["label"].map(is_goalie)]
    for (t, team), g in sk.groupby([np.round(sk["t"], 6), "team"]):
        pts = g[["sx", "sy"]].to_numpy()
        area = np.nan
        if len(pts) >= 3:
            try:
                area = ConvexHull(pts).volume  # in 2-D, "volume" is the area
            except QhullError:
                area = 0.0
        rows.append(dict(t=t, team=team, width_ft=np.ptp(pts[:, 1]), depth_ft=np.ptp(pts[:, 0]),
                         hull_area_sqft=area, n=len(pts)))
    return pd.DataFrame(rows)


def review_moments(ctx: pd.DataFrame, min_s: float = 0.5) -> pd.DataFrame:
    """Stretches where a skater was out of position — a ready-made film-review list."""
    rows = []
    for tid, g in ctx[ctx["team"].isin(["A", "B"])].groupby("track_id"):
        g = g.sort_values("t")
        flag = g["out_of_position"].to_numpy()
        t = g["t"].to_numpy()
        start = None
        for i, f in enumerate(flag):
            if f and start is None:
                start = i
            if (not f or i == len(flag) - 1) and start is not None:
                end = i if f else i - 1
                if t[end] - t[start] >= min_s:
                    seg = g.iloc[start:end + 1]
                    why = []
                    if (seg["defending"] & (seg["goal_side"] == 0)).mean() > 0.5:
                        why.append("not goal-side of their check")
                    if (seg["slot_dev_ft"] > OUT_OF_SLOT_FT).mean() > 0.5:
                        why.append(f"~{seg['slot_dev_ft'].mean():.0f} ft off their usual spot")
                    rows.append(dict(track_id=tid, team=g["team"].iloc[0], start_s=round(t[start], 2),
                                     end_s=round(t[end], 2), frame=int(seg["frame"].iloc[0]),
                                     reason=" & ".join(why) or "drifted from structure"))
                start = None
    return pd.DataFrame(rows, columns=["track_id", "team", "start_s", "end_s", "frame", "reason"])


def analyze(players: pd.DataFrame, puck: pd.DataFrame, meta: dict, a_attacks_right: bool | None = None) -> dict:
    """One call that runs every step above and returns all tables."""
    fps = meta["fps"] / meta.get("frame_stride", 1)
    k = kinematics(players, fps)
    pk = puck_kinematics(puck, fps)
    if a_attacks_right is None:
        attack = attack_directions(k)
    else:
        attack = {"A": a_attacks_right, "B": not a_attacks_right}
    poss = possession(k, pk)
    ev = detect_events(poss, pk, k, attack, fps)
    ctx = frame_context(k, poss, attack)
    summary = player_summary(ctx, poss, ev, meta.get("calibrated", False))
    return dict(tracks=ctx, puck=pk, possession=poss, events=ev, summary=summary,
                shape=team_shape(ctx), review=review_moments(ctx) if meta.get("calibrated") else pd.DataFrame(),
                attack=attack)
