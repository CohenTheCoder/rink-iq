"""Broadcast-style highlight video: the analysis drawn back onto the film.

    tracks + events ──► per frame: ice rings · motion trails · name/speed tags ·
                        facing arrows · glowing puck · SHOT/PASS banners · mini-map
                    ──► H.264 MP4 (plays in any browser)

Trails stay glued to the ice while the camera pans: a past position is mapped
into the shot's reference pixels with that frame's camera matrix, then back
into the current frame with the inverse of the current one.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
import pandas as pd

from . import identity

from .rink import RINK_LENGTH, RINK_WIDTH

TEAM = {"A": (47, 111, 214), "B": (208, 57, 62), "REF": (150, 150, 150), "?": (200, 200, 200)}  # RGB
GOLD = (242, 169, 0)
TRAIL_FRAMES = 24


# ------------------------------------------------------------------ small drawing helpers
def _blend(img, overlay, alpha):
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _rounded(img, x1, y1, x2, y2, color, r=8, alpha=0.85):
    ov = img.copy()
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    r = max(1, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(ov, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(ov, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(ov, (cx, cy), r, color, -1)
    _blend(img, ov, alpha)


def _text(img, s, org, scale, color=(255, 255, 255), thick=1):
    cv2.putText(img, s, (int(org[0]), int(org[1])), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _speed_color(mph):
    if np.isnan(mph):
        return (180, 180, 180)
    if mph < 10:
        return (90, 200, 120)
    if mph < 18:
        return (245, 200, 60)
    return (240, 80, 60)


def _tag(img, x, y, title, mph, team_rgb, s):
    label = title if np.isnan(mph) else f"{title}  {mph:.0f} mph"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, max(1, int(1.4 * s)))
    pad = int(6 * s)
    x1, y1 = int(x - tw / 2 - pad - 8 * s), int(y - th - 2 * pad)
    _rounded(img, x1, y1, x1 + tw + 2 * pad + 12 * s, y, (15, 27, 45), r=int(7 * s), alpha=0.78)
    cv2.rectangle(img, (x1, y1), (x1 + int(4 * s), int(y)), team_rgb, -1)
    if not np.isnan(mph):
        cv2.circle(img, (int(x1 + 12 * s), int(y - pad - th / 2)), int(3.5 * s), _speed_color(mph), -1)
    _text(img, label, (x1 + 18 * s, y - pad), 0.5 * s, thick=max(1, int(1.4 * s)))


def _ring(img, cx, cy, w, color, s, spotlight=False):
    ov = img.copy()
    axes = (int(max(10, w * 0.75)), int(max(4, w * 0.22)))
    cv2.ellipse(ov, (int(cx), int(cy)), axes, 0, 0, 360, color, -1)
    _blend(img, ov, 0.35)
    cv2.ellipse(img, (int(cx), int(cy)), axes, 0, 0, 360, color, max(2, int(2 * s)), cv2.LINE_AA)
    if spotlight:
        for k in range(3):
            cv2.ellipse(img, (int(cx), int(cy)), (axes[0] + int((4 + 4 * k) * s), axes[1] + int((2 + 2 * k) * s)),
                        0, 0, 360, GOLD, max(1, int(1.5 * s)), cv2.LINE_AA)


def _trail(img, pts, color, s):
    if len(pts) < 2:
        return
    ov = img.copy()
    for i in range(1, len(pts)):
        cv2.line(ov, tuple(map(int, pts[i - 1])), tuple(map(int, pts[i])), color,
                 max(1, int((1 + 4 * i / len(pts)) * s)), cv2.LINE_AA)
    _blend(img, ov, 0.7)


def _banner(img, text, sub, color, progress, s):
    """Slides in from the top, holds, fades. progress 0..1 over the banner's life."""
    h, w = img.shape[:2]
    alpha = min(1.0, progress * 6, (1 - progress) * 4)
    if alpha <= 0:
        return
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.1 * s, max(2, int(2 * s)))
    bw = tw + int(60 * s)
    y = int(70 * s - (1 - min(1, progress * 6)) * 40 * s)
    ov = img.copy()
    _rounded(ov, w / 2 - bw / 2, y, w / 2 + bw / 2, y + th + int(46 * s), color, r=int(12 * s), alpha=1.0)
    cv2.putText(ov, text, (int(w / 2 - tw / 2), y + th + int(12 * s)), cv2.FONT_HERSHEY_DUPLEX, 1.1 * s,
                (255, 255, 255), max(2, int(2 * s)), cv2.LINE_AA)
    (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, 1)
    _text(ov, sub, (w / 2 - sw / 2, y + th + int(34 * s)), 0.5 * s)
    _blend(img, ov, alpha)


def _minimap(img, rows, puck_xy, s, names, attack=None):
    h, w = img.shape[:2]
    mw = int(300 * s)
    mh = int(mw * RINK_WIDTH / RINK_LENGTH)
    x0, y0 = w - mw - int(16 * s), h - mh - int(16 * s)
    _rounded(img, x0 - 6, y0 - 6, x0 + mw + 6, y0 + mh + 6, (15, 27, 45), r=int(14 * s), alpha=0.65)

    def P(x, y):
        return int(x0 + x / RINK_LENGTH * mw), int(y0 + (1 - y / RINK_WIDTH) * mh)

    cv2.rectangle(img, P(0, 85), P(200, 0), (235, 242, 248), -1)
    for xl, c, t in [(11, (208, 57, 62), 1), (189, (208, 57, 62), 1), (75, (47, 111, 214), 3), (125, (47, 111, 214), 3),
                     (100, (208, 57, 62), 3)]:
        cv2.line(img, P(xl, 1), P(xl, 84), c, max(1, int(t * s * 0.7)))
    cv2.circle(img, P(100, 42.5), int(15 / RINK_LENGTH * mw), (47, 111, 214), 1, cv2.LINE_AA)
    for _, r in rows.iterrows():
        if np.isnan(r.get("sx", np.nan)):
            continue
        cv2.circle(img, P(r["sx"], r["sy"]), max(3, int(4.5 * s)), TEAM.get(r["team"], (200, 200, 200)), -1, cv2.LINE_AA)
        cv2.circle(img, P(r["sx"], r["sy"]), max(3, int(4.5 * s)), (255, 255, 255), 1, cv2.LINE_AA)
    if puck_xy is not None:
        cv2.circle(img, P(*puck_xy), max(2, int(3 * s)), (10, 10, 10), -1, cv2.LINE_AA)


def _hud(img, t, fastest, s, title="RinkIQ", corner="bottom-left"):
    """Title, clock and fastest skater. Bottom-left by default: broadcasts put their score bug top-left."""
    h = img.shape[0]
    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, max(2, int(2 * s)))
    clock = f"{int(t // 60)}:{t % 60:04.1f}"
    (cw, _), _ = cv2.getTextSize(clock, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, max(1, int(1.5 * s)))
    y0 = 14 * s if corner == "top-left" else h - 64 * s - (24 * s if fastest else 0)
    _rounded(img, 14 * s, y0, 26 * s + tw + 20 * s + cw + 16 * s, y0 + 50 * s, (15, 27, 45), r=int(10 * s), alpha=0.72)
    _text(img, title, (26 * s, y0 + 32 * s), 0.8 * s, thick=max(2, int(2 * s)))
    _text(img, clock, (26 * s + tw + 20 * s, y0 + 32 * s), 0.6 * s, (190, 205, 220), max(1, int(1.5 * s)))
    if fastest:
        _text(img, f"fastest: {fastest}", (18 * s, y0 + 70 * s), 0.5 * s, GOLD, max(1, int(1.4 * s)))


# ------------------------------------------------------------------ main renderer
class _Writer:
    def __init__(self, path, size, fps):
        import imageio_ffmpeg
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        self.gen = imageio_ffmpeg.write_frames(str(path), size, fps=fps, codec="libx264", quality=7,
                                               pix_fmt_in="rgb24", macro_block_size=2,
                                               output_params=["-movflags", "+faststart"])
        self.gen.send(None)

    def write(self, rgb):
        self.gen.send(np.ascontiguousarray(rgb))

    def close(self):
        self.gen.close()


def _event_overlays(img, t, events, names, s, window=1.4):
    if events is None or events.empty:
        return
    for e in events.itertuples():
        age = t - e.t
        if 0 <= age <= window:
            who = identity.label(e.player, names)
            if e.type == "shot":
                _banner(img, f"SHOT  {e.speed_mph:.0f} MPH", f"{who}  |  Team {e.team}", (208, 57, 62), age / window, s)
            elif e.type == "pass" and age <= 0.9:
                to = identity.label(e.to_player, names)
                _banner(img, "PASS", f"{who}  ->  {to}", (47, 111, 214), age / 0.9, s * 0.7)
            elif e.type == "turnover" and age <= 0.9:
                _banner(img, "TURNOVER", who, (60, 60, 60), age / 0.9, s * 0.7)


def render_broadcast(video_path: str, res: dict, puck_raw: pd.DataFrame, meta: dict, out_path: str,
                     names: dict | None = None, start_s: float | None = None, end_s: float | None = None,
                     out_width: int = 1280, progress: Callable[[float, str], None] | None = None) -> str:
    """Draw the analysis onto the original film. Returns out_path."""
    names = names or {}
    tr = res["tracks"]
    frames = meta.get("frames")
    if frames is None or len(frames) == 0:
        frames = pd.DataFrame({"frame": sorted(tr["frame"].unique()), "shot": 0})
        for i in range(3):
            for j in range(3):
                frames[f"m{i}{j}"] = float(i == j)
    fps = meta["fps"]
    t0 = (start_s or 0) + 0.0
    fr = frames.copy()
    fr["t"] = fr["frame"] / fps
    fr = fr[(fr["t"] >= t0) & ((fr["t"] <= end_s) if end_s is not None else True)]
    if fr.empty:
        raise ValueError("No analysed frames in that time range.")
    mats = {int(r.frame): (int(r.shot), np.array([getattr(r, f"m{i}{j}") for i in range(3) for j in range(3)])
                           .reshape(3, 3)) for r in fr.itertuples()}
    by_frame = {f: g for f, g in tr.groupby("frame")}
    puck_by_frame = {int(r.frame): (r.px, r.py, r.x, r.y) for r in puck_raw.itertuples()} if not puck_raw.empty else {}
    order = sorted(mats)
    pos_in_order = {f: i for i, f in enumerate(order)}
    calibrated = meta.get("calibrated", False)
    events = res.get("events")

    cap = cv2.VideoCapture(video_path)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale_out = min(1.0, out_width / W)
    size = (int(W * scale_out) // 2 * 2, int(H * scale_out) // 2 * 2)
    out_fps = fps / meta.get("frame_stride", 1)
    writer = _Writer(out_path, size, out_fps)
    s = W / 1280
    cap.set(cv2.CAP_PROP_POS_FRAMES, order[0])
    cur = order[0]
    for k, fno in enumerate(order):
        while cur < fno:  # skip frames that weren't analysed
            cap.grab()
            cur += 1
        ok, frame = cap.read()
        cur += 1
        if not ok:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        shot, M_now = mats[fno]
        M_inv = np.linalg.inv(M_now)
        rows = by_frame.get(fno, pd.DataFrame())
        t = fno / fps
        # trails (same shot only), mapped into today's pixels
        past = order[max(0, pos_in_order[fno] - TRAIL_FRAMES):pos_in_order[fno] + 1]
        fastest_id, fastest_v = None, 0
        for _, r in rows.iterrows():
            if r["team"] not in ("A", "B"):
                continue
            pts = []
            for pf in past:
                if mats[pf][0] != shot:
                    continue
                g = by_frame.get(pf)
                if g is None:
                    continue
                q = g[g["track_id"] == r["track_id"]]
                if len(q):
                    p = np.array([q["px"].iloc[0], q["py"].iloc[0], 1.0])
                    p = M_inv @ (mats[pf][1] @ p)
                    pts.append((p[0] / p[2], p[1] / p[2]))
            _trail(img, pts, TEAM[r["team"]], s)
            v = r.get("speed_mph", np.nan)
            if not np.isnan(v) and v > fastest_v and not str(r["label"]).startswith("goalie"):
                fastest_id, fastest_v = r["track_id"], v
        # puck trail + glow
        ptr = []
        for pf in past[-10:]:
            if pf in puck_by_frame and mats[pf][0] == shot:
                p = M_inv @ (mats[pf][1] @ np.array([puck_by_frame[pf][0], puck_by_frame[pf][1], 1.0]))
                ptr.append((p[0] / p[2], p[1] / p[2]))
        if len(ptr) >= 2:
            _trail(img, ptr, (255, 235, 120), s)
        for _, r in rows.iterrows():
            color = TEAM.get(r["team"], (200, 200, 200))
            w = r["x2"] - r["x1"]
            _ring(img, r["px"], r["py"], w, color, s, spotlight=r["track_id"] == fastest_id and fastest_v > 12)
        for _, r in rows.iterrows():
            fx, fy = r.get("face_x", np.nan), r.get("face_y", np.nan)
            if not (np.isnan(fx) or np.isnan(fy)):
                L = (r["y2"] - r["y1"]) * 0.55
                cv2.arrowedLine(img, (int(r["px"]), int(r["py"])), (int(r["px"] + fx * L), int(r["py"] - fy * L * 0.5)),
                                GOLD, max(2, int(2 * s)), cv2.LINE_AA, tipLength=0.3)
            if r["team"] in ("A", "B", "REF"):
                title = identity.label(r["track_id"], names) if r["team"] != "REF" else "REF"
                _tag(img, r["px"], r["y1"] - 4 * s, title, r.get("speed_mph", np.nan) if r["team"] != "REF" else np.nan,
                     color, s)
        if fno in puck_by_frame:
            px_, py_ = puck_by_frame[fno][:2]
            for rad, a in ((18, 0.15), (12, 0.25), (7, 0.5)):
                ov = img.copy()
                cv2.circle(ov, (int(px_), int(py_)), int(rad * s), (255, 240, 150), -1, cv2.LINE_AA)
                _blend(img, ov, a)
            cv2.circle(img, (int(px_), int(py_)), int(5 * s), (20, 20, 20), -1, cv2.LINE_AA)
        _event_overlays(img, t, events, names, s)
        fastest_txt = (f"{identity.label(fastest_id, names)} {fastest_v:.0f} mph" if fastest_id is not None
                       else "")
        _hud(img, t, fastest_txt, s)
        if calibrated and len(rows):
            pk = puck_by_frame.get(fno)
            _minimap(img, rows, (pk[2], pk[3]) if pk else None, s, names)
        if scale_out < 1:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        elif img.shape[1] != size[0] or img.shape[0] != size[1]:
            img = cv2.resize(img, size)
        writer.write(img)
        if progress and k % 5 == 0:
            progress((k + 1) / len(order), f"Rendering {k + 1}/{len(order)}")
    writer.close()
    cap.release()
    return out_path


def render_rink_replay(res: dict, meta: dict, out_path: str, names: dict | None = None,
                       progress: Callable[[float, str], None] | None = None, out_fps: float = 30) -> str:
    """Top-down animated replay (for the synthetic demo, or any calibrated run)."""
    names = names or {}
    tr = res["tracks"]
    pk = res["puck"]
    events = res.get("events")
    W, Hh = 1280, 600
    S = (W - 80) / RINK_LENGTH
    ox, oy = 40, (Hh - RINK_WIDTH * S) / 2 + 20

    def P(x, y):
        return int(ox + x * S), int(oy + (RINK_WIDTH - y) * S)

    base = np.full((Hh, W, 3), (232, 238, 244), np.uint8)
    cv2.rectangle(base, P(0, 85), P(200, 0), (248, 251, 253), -1)
    cv2.rectangle(base, P(0, 85), P(200, 0), (154, 167, 180), 2)
    for xl, c, t in [(11, (208, 57, 62), 2), (189, (208, 57, 62), 2), (75, (47, 111, 214), 6), (125, (47, 111, 214), 6),
                     (100, (208, 57, 62), 6)]:
        cv2.line(base, P(xl, 1), P(xl, 84), c, t)
    for cx in (31, 169):
        for cy in (20.5, 64.5):
            cv2.circle(base, P(cx, cy), int(15 * S), (208, 57, 62), 2, cv2.LINE_AA)
            cv2.circle(base, P(cx, cy), 4, (208, 57, 62), -1)
    cv2.circle(base, P(100, 42.5), int(15 * S), (47, 111, 214), 2, cv2.LINE_AA)
    writer = _Writer(out_path, (W, Hh), out_fps)
    times = np.sort(tr["t"].round(4).unique())
    step = max(1, int(round((1 / out_fps) / max(np.median(np.diff(times)), 1e-6))))
    times = times[::step]
    by_t = {t: g for t, g in tr.groupby(tr["t"].round(4))}
    pk_t = pk.assign(tk=pk["t"].round(4)).set_index("tk") if not pk.empty else None
    for k, t in enumerate(times):
        img = base.copy()
        rows = by_t.get(t, pd.DataFrame())
        recent = tr[(tr["t"] > t - 1.0) & (tr["t"] <= t)]
        for tid, g in recent.groupby("track_id"):
            team = g["team"].iloc[0]
            _trail(img, [P(x, y) for x, y in zip(g["sx"], g["sy"])], TEAM.get(team, (150, 150, 150)), 1.2)
        if pk_t is not None:
            pr = pk[(pk["t"] > t - 0.35) & (pk["t"] <= t)]
            _trail(img, [P(x, y) for x, y in zip(pr["x"], pr["y"])], (255, 200, 40), 1.4)
        sk = rows[rows["team"].isin(["A", "B"]) & ~rows["label"].str.startswith("goalie")] if len(rows) else rows
        fastest = sk.loc[sk["speed_mph"].idxmax()] if len(sk) and sk["speed_mph"].notna().any() else None
        for _, r in rows.iterrows():
            c = TEAM.get(r["team"], (150, 150, 150))
            p = P(r["sx"], r["sy"])
            if fastest is not None and r["track_id"] == fastest["track_id"]:
                cv2.circle(img, p, 20, GOLD, 3, cv2.LINE_AA)
            cv2.circle(img, p, 12, c, -1, cv2.LINE_AA)
            cv2.circle(img, p, 12, (255, 255, 255), 2, cv2.LINE_AA)
            if not np.isnan(r.get("face_x", np.nan)):
                cv2.arrowedLine(img, p, (int(p[0] + r["face_x"] * 26), int(p[1] - r["face_y"] * 26)), GOLD, 2,
                                cv2.LINE_AA, tipLength=0.35)
            if r["team"] in ("A", "B"):
                _tag(img, p[0], p[1] - 16, identity.label(r["track_id"], names), r["speed_mph"], c, 0.85)
        if pk_t is not None and t in pk_t.index:
            q = pk_t.loc[t]
            q = q.iloc[0] if isinstance(q, pd.DataFrame) else q
            cv2.circle(img, P(q["x"], q["y"]), 9, (255, 240, 150), -1, cv2.LINE_AA)
            cv2.circle(img, P(q["x"], q["y"]), 5, (20, 20, 20), -1, cv2.LINE_AA)
        _event_overlays(img, t, events, names, 1.0)
        f_txt = f"{identity.label(fastest['track_id'], names)} {fastest['speed_mph']:.0f} mph" \
            if fastest is not None else ""
        _hud(img, t, f_txt, 1.0, "RinkIQ replay", corner="top-left")
        writer.write(img)
        if progress and k % 10 == 0:
            progress((k + 1) / len(times), f"Rendering {k + 1}/{len(times)}")
    writer.close()
    return out_path


def live_preview(frame_bgr: np.ndarray, live: list[dict], history: dict, names: dict | None = None) -> np.ndarray:
    """Quick overlay shown *while* a video is being processed (boxes, IDs, short trails)."""
    names = names or {}
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    s = img.shape[1] / 1280
    for d in live:
        if d["label"] == "puck":
            cv2.circle(img, tuple(map(int, d["foot"])), int(10 * s), (255, 235, 120), 2, cv2.LINE_AA)
            continue
        team = "A" if d["label"].endswith("_a") or "team_a" in d["label"] else \
            "B" if d["label"].endswith("_b") or "team_b" in d["label"] else "REF"
        hist = history.setdefault(d["track_id"], [])
        hist.append(d["foot"])
        del hist[:-15]
        _trail(img, hist, TEAM[team], s)
        x1, y1, x2, y2 = d["box"]
        _ring(img, (x1 + x2) / 2, y2, x2 - x1, TEAM[team], s)
        _tag(img, (x1 + x2) / 2, y1 - 4 * s, identity.label(d["track_id"], names), np.nan, TEAM[team], s)
    return img
