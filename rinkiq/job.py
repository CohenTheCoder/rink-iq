"""One video in, one JSON result + output files out. The function the online worker runs for every job.

    run_analysis(video, options, workdir)
        ├─ find the rink automatically          (rinkiq.autocal)
        ├─ detect + track + read numbers        (rinkiq.pipeline)
        ├─ stats, names, report cards           (rinkiq.metrics / identity / report)
        └─ broadcast-style video + HTML report + CSV zip   (rinkiq.render / report)

It knows nothing about Supabase or the web, so it is easy to test and to reuse (a CLI, a notebook, a queue).
"""
from __future__ import annotations

import io
import math
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import identity, metrics, pipeline
from .autocal import auto_calibrate
from .render import render_broadcast
from .report import html_report, report_cards, team_comparison

QUALITY = {  # analysed fps, pose, number-reading interval (frames)
    "fast": dict(fps=10, pose=False, numbers=3),
    "balanced": dict(fps=15, pose=True, numbers=3),
    "detailed": dict(fps=30, pose=True, numbers=3),
}
RESULT_VERSION = 1


class JobCanceled(Exception):
    """Raised from the progress callback when the job was canceled while running."""


@dataclass
class JobOptions:
    start_s: float = 0.0
    seconds: float = 20.0              # how much of the video to analyse
    quality: str = "fast"
    render: bool = True
    render_seconds: float = 30.0       # rendering costs time, so the video covers at most this much
    team_a: str = ""                   # NHL code of the dark-jersey team, e.g. "SJS"
    team_b: str = ""
    roster_a: str = ""                 # or pasted "63 Firstname Lastname" lines (wins over team_a)
    roster_b: str = ""
    season: str = "20252026"
    max_seconds: float | None = None   # hard cap set by the worker's owner, not by the user


def parse_options(raw: dict | None, max_seconds: float | None = None) -> JobOptions:
    """Turn the untrusted JSON stored with a job into safe, bounded options."""
    raw = raw or {}

    def num(key, default, lo, hi):
        try:
            v = float(raw.get(key, default))
        except (TypeError, ValueError):
            return default
        return min(max(v, lo), hi) if math.isfinite(v) else default

    def text(key, limit):
        return str(raw.get(key) or "")[:limit].strip()

    quality = text("quality", 12).lower()
    return JobOptions(
        start_s=num("start_s", 0.0, 0.0, 6 * 3600.0),
        seconds=num("seconds", 20.0, 2.0, 3 * 3600.0),
        quality=quality if quality in QUALITY else "fast",
        render=bool(raw.get("render", True)),
        render_seconds=num("render_seconds", 30.0, 2.0, 60.0),
        team_a=text("team_a", 3).upper(), team_b=text("team_b", 3).upper(),
        roster_a=text("roster_a", 5000), roster_b=text("roster_b", 5000),
        season=text("season", 8) or "20252026",
        max_seconds=max_seconds,
    )


@dataclass
class Analysis:
    result: dict
    files: dict[str, Path] = field(default_factory=dict)   # artifact name → local path


# ------------------------------------------------------------------ helpers
def clean(obj):
    """Make anything numpy/pandas-flavoured JSON-safe (NaN and ∞ become null)."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else round(obj, 4)
    return obj


def _rosters(options: JobOptions) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for team, code, pasted in (("A", options.team_a, options.roster_a), ("B", options.team_b, options.roster_b)):
        try:
            if pasted:
                out[team] = identity.parse_roster(pasted)
            elif code:
                out[team] = identity.nhl_roster(code, options.season)
        except Exception:  # noqa: BLE001 — offline or unknown team: numbers still show, just without names
            pass
    return out


def _stats_zip(res: dict, names: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for key in ("summary", "events", "review", "tracks"):
            df = res.get(key)
            if df is None or df.empty:
                continue
            df = df.copy()
            if "track_id" in df:
                df.insert(1, "player", df["track_id"].map(lambda t: identity.label(t, names)))
            z.writestr(f"{key}.csv", df.to_csv(index=False))
    return buf.getvalue()


def build_result(res: dict, meta: dict, names: dict, info: dict, options: JobOptions, auto, start: float,
                 end: float) -> dict:
    summ = res["summary"]
    skaters = summ[summ["team"].isin(["A", "B"]) & ~summ["label"].str.startswith("goalie")]
    reliable = skaters[skaters["seconds"] >= 2]
    ev = res["events"]
    cards = report_cards(summ, res["review"], names)
    teams = team_comparison(summ, ev)
    fastest = skaters.loc[skaters["top_speed_mph"].idxmax()] if skaters["top_speed_mph"].notna().any() else None
    counts = ev["type"].value_counts().to_dict() if not ev.empty else {}
    return clean({
        "version": RESULT_VERSION,
        "video": {"fps": info["fps"], "duration_s": info["n_frames"] / info["fps"],
                  "width": info["width"], "height": info["height"]},
        "range": {"start_s": start, "end_s": end},
        "calibration": {
            "mode": "rink feet" if meta.get("calibrated") else "speed only",
            "auto": None if auto is None else {"frame": auto.frame, "seconds": auto.frame / info["fps"],
                                               "markings": len(auto.marks), "lines": len(auto.lines),
                                               "fit_error_ft": auto.error_ft},
        },
        "tiles": {
            "skaters": len(reliable),
            "named": sum(1 for v in names.values() if " " in v),
            "numbers_read": len(meta.get("numbers", {})),
            "top_speed_mph": None if fastest is None else fastest["top_speed_mph"],
            "fastest": None if fastest is None else identity.label(fastest["track_id"], names),
            "shots": counts.get("shot", 0), "passes": counts.get("pass", 0), "turnovers": counts.get("turnover", 0),
        },
        "stars": [asdict(c) for c in cards[:3]],
        "cards": [asdict(c) for c in cards],
        "teams": {col: {k: v for k, v in teams[col].items()} for col in teams.columns} if not teams.empty else {},
        "events": [] if ev.empty else [
            {"t": r.t, "type": r.type, "player": identity.label(r.player, names),
             "to": identity.label(r.to_player, names) if r.to_player >= 0 else None,
             "team": r.team, "speed_mph": r.speed_mph} for r in ev.head(200).itertuples()],
        "camera_shots": [{"start_s": s["start_frame"] / info["fps"], "end_s": s["end_frame"] / info["fps"],
                          "status": s["status"]} for s in meta.get("shots", [])],
        "files": {},
    })


# ------------------------------------------------------------------ the job
def run_analysis(video_path: str | Path, options: JobOptions, workdir: str | Path,
                 progress: Callable[[float, str], None] | None = None) -> Analysis:
    """Run everything. progress(fraction 0..1, message) may raise (e.g. JobCanceled) to stop the job."""
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda f, m: None)
    path = str(video_path)
    info = pipeline.video_info(path)
    if info["n_frames"] <= 0:
        raise ValueError("The video could not be read (is it a valid mp4/mov?).")
    dur = info["n_frames"] / info["fps"]

    seconds = options.seconds if options.max_seconds is None else min(options.seconds, options.max_seconds)
    start = min(options.start_s, max(0.0, dur - 2.0))
    end = min(dur, start + seconds)
    fps_target, pose = QUALITY[options.quality]["fps"], QUALITY[options.quality]["pose"]
    stride = max(1, round(info["fps"] / fps_target))

    # 1 · find the rink (0 – 8 %)
    say(0.0, "Looking for the rink…")
    auto = auto_calibrate(path, start_s=start, max_s=min(end - start, 60.0), step_s=0.5,
                          progress=lambda f, m: say(0.08 * f, m))
    cal, ref = (auto.calibration, auto.frame) if auto is not None else (None, int(start * info["fps"]))
    say(0.08, "Rink found, tracking players…" if auto is not None else
        "No rink markings found: speed-only mode. Tracking players…")

    # 2 · detect, track, read numbers (8 – 70 %)
    cfg = pipeline.RunConfig(frame_stride=stride, use_pose=pose, read_numbers=True,
                             number_every=QUALITY[options.quality]["numbers"], start_s=start, end_s=end)
    players, puck, meta = pipeline.run_video(path, calibration=cal, reference_frame=ref, config=cfg,
                                             progress=lambda f, m: say(0.08 + 0.62 * f, m))
    if players.empty:
        raise ValueError("No skaters were detected. RinkIQ needs hockey film from an elevated camera.")

    # 3 · stats, names, report (70 – 74 %)
    say(0.70, "Computing stats and report cards…")
    res = metrics.analyze(players, puck, meta)
    team_of = res["tracks"].drop_duplicates("track_id").set_index("track_id")["team"].to_dict()
    names = identity.assign_names(meta.get("numbers", {}), team_of, _rosters(options))
    result = build_result(res, meta, names, info, options, auto, start, end)
    files: dict[str, Path] = {}
    cards = report_cards(res["summary"], res["review"], names)
    report_path = work / "report.html"
    report_path.write_text(html_report(cards, team_comparison(res["summary"], res["events"])), encoding="utf-8")
    files["report"] = report_path
    zip_path = work / "stats.zip"
    zip_path.write_bytes(_stats_zip(res, names))
    files["stats"] = zip_path

    # 4 · broadcast video (74 – 97 %)
    if options.render:
        say(0.74, "Rendering the broadcast video…")
        video_out = work / "broadcast.mp4"
        render_end = min(end, start + options.render_seconds)
        render_broadcast(path, res, puck, meta, str(video_out), names, start_s=start, end_s=render_end,
                         out_width=960, progress=lambda f, m: say(0.74 + 0.23 * f, m))
        files["video"] = video_out
    result["files"] = {k: v.name for k, v in files.items()}
    say(0.97, "Uploading results…")
    return Analysis(result=result, files=files)
