"""🚀 Analyze a video: the one-button flow. Upload → auto-calibrate → track (live) → stats → broadcast video."""
from __future__ import annotations

import io
import os
import tempfile
import zipfile
from pathlib import Path

import cv2
import streamlit as st

from rinkiq import identity, metrics, models, pipeline
from rinkiq.autocal import auto_calibrate
from rinkiq.render import live_preview, render_broadcast
from rinkiq.rink import draw_rink_overlay
from rinkiq.report import html_report, report_cards, team_comparison

UPLOAD_DIR = Path(tempfile.gettempdir()) / "rinkiq_uploads"
# On a shared public server, cap how much of each upload gets analysed (set by the Space's Dockerfile).
MAX_SECONDS = float(os.environ.get("RINKIQ_MAX_SECONDS", "0")) or None

CSS = """
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@600;800&display=swap" rel="stylesheet">
<style>
.rq-jumbo{background:radial-gradient(circle at 20% 0%,#16243d 0%,#0a1220 70%);color:#eef3fa;border-radius:16px;
  padding:22px 26px;border:3px solid #c8102e;box-shadow:inset 0 0 0 3px #0b3d91,0 6px 24px rgba(0,0,0,.25)}
.rq-jumbo h1{font-family:'Barlow Condensed',sans-serif;font-weight:800;letter-spacing:1px;font-size:2.6rem;margin:0;
  color:#fff;text-transform:uppercase}
.rq-jumbo p{margin:.3rem 0 0;color:#b9c6d8;font-size:1.02rem}
.rq-led{font-family:'Share Tech Mono',monospace;color:#ffb81c;text-shadow:0 0 8px rgba(255,184,28,.55)}
.rq-steps{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}
.rq-step{background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.14);border-radius:999px;padding:4px 12px;
  font-family:'Barlow Condensed',sans-serif;font-size:1.05rem;letter-spacing:.5px}
.rq-tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:8px 0 4px}
.rq-tile{background:#0a1220;border-radius:12px;padding:12px 14px;border-bottom:4px solid #c8102e}
.rq-tile .k{color:#9fb0c6;font-family:'Barlow Condensed',sans-serif;text-transform:uppercase;letter-spacing:1px;font-size:.9rem}
.rq-tile .v{font-family:'Share Tech Mono',monospace;color:#ffb81c;font-size:1.7rem}
.rq-star{background:#fff;border-radius:14px;padding:14px 16px;border-top:6px solid #c8102e;
  box-shadow:0 2px 10px rgba(10,18,32,.12);height:100%}
.rq-star .s{font-size:1.1rem}.rq-star h3{font-family:'Barlow Condensed',sans-serif;font-weight:800;font-size:1.6rem;margin:.1rem 0}
.rq-star .g{float:right;font-family:'Share Tech Mono',monospace;font-size:1.8rem;color:#0b3d91}
.rq-star ul{padding-left:1.1rem;margin:.4rem 0 0}.rq-star li{margin:.15rem 0}
</style>"""


def _tiles(pairs):
    cells = "".join(f"<div class='rq-tile'><div class='k'>{k}</div><div class='v'>{v}</div></div>" for k, v in pairs)
    st.markdown(f"<div class='rq-tiles'>{cells}</div>", unsafe_allow_html=True)


def _roster_picker(key: str, default: str | None):
    teams = ["(none)", "Paste a roster"] + [f"{k} · {v}" for k, v in identity.NHL_TEAMS.items()]
    idx = next((i for i, t in enumerate(teams) if default and t.startswith(default)), 0)
    pick = st.selectbox(key, teams, index=idx, key=f"quick_{key}")
    if pick == "Paste a roster":
        return ("paste", st.text_area(f"{key}: one '63 Firstname Lastname' per line", key=f"quick_paste_{key}", height=100))
    return ("nhl", pick.split(" · ")[0]) if pick != "(none)" else None


def _rosters(choices: dict, season: str) -> dict:
    out = {}
    for team, ch in choices.items():
        if not ch:
            continue
        kind, val = ch
        try:
            out[team] = identity.parse_roster(val) if kind == "paste" else identity.nhl_roster(val, season)
        except Exception as e:  # noqa: BLE001
            st.warning(f"Couldn't load Team {team}'s roster ({e}). Numbers will show without names.")
    return out


def quick_tab(ss, set_result):
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("""<div class='rq-jumbo'><h1>Drop in game film. Get the <span class='rq-led'>stats</span>.</h1>
<p>Upload any elevated-camera hockey video: a TV broadcast, a rink camera, a phone from the stands. RinkIQ finds the
rink by itself, tracks every skater, reads jersey numbers and grades every player.</p>
<div class='rq-steps'><span class='rq-step'>1 · Find the rink</span><span class='rq-step'>2 · Track players</span>
<span class='rq-step'>3 · Read numbers</span><span class='rq-step'>4 · Stats + report cards</span>
<span class='rq-step'>5 · Broadcast video</span></div></div>""", unsafe_allow_html=True)
    st.write("")
    c1, c2 = st.columns([3, 2])
    with c1:
        up = st.file_uploader("Your video", type=["mp4", "mov", "m4v", "avi", "mkv", "webm"],
                              help="Longer videos work. You choose how much to analyse below.")
        use_sample = st.toggle("No video? Use the sample NHL clip (San Jose vs Montréal)", value=up is None)
    path, default_teams = None, (None, None)
    if up is not None:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        path = UPLOAD_DIR / up.name
        if not path.exists() or path.stat().st_size != up.size:
            path.write_bytes(up.getbuffer())
        path = str(path)
    elif use_sample:
        path = str(models.download_sample_clip(list(models.SAMPLE_CLIPS)[1]))
        default_teams = ("SJS", "MTL")
    with c2:
        st.markdown("**Teams** (optional, for player names)")
        ra = _roster_picker("Team A (dark jerseys)", default_teams[0])
        rb = _roster_picker("Team B (light jerseys)", default_teams[1])
        season = st.text_input("NHL season", "20252026")
    if not path:
        return
    info = pipeline.video_info(path)
    dur = info["n_frames"] / info["fps"]
    limit = min(dur, MAX_SECONDS) if MAX_SECONDS else dur
    c1, c2, c3 = st.columns(3)
    span = c1.slider("Seconds to analyse", 2.0, float(max(2.0, limit)), float(min(limit, 20.0)), 1.0,
                     help=("On this free demo server analysis is capped at "
                           f"{MAX_SECONDS:.0f} s per video." if MAX_SECONDS else
                           "A full game works too. It just takes longer (about real-time×3 at 10 fps on a laptop)."))
    start = c2.number_input("Starting at (s)", 0.0, max(0.0, dur - span), 0.0, 5.0)
    quality = c3.select_slider("Speed vs. detail", ["Fast", "Balanced", "Detailed"], value="Fast",
                               help="Fast: 10 fps, no pose. Balanced: 15 fps + pose. Detailed: 30 fps + pose.")
    render_it = st.toggle("Render the broadcast-style video when done", value=True)
    if not st.button("🚀 Analyze", type="primary", use_container_width=True):
        _show_results(ss)
        return

    fps_target, pose = {"Fast": (10, False), "Balanced": (15, True), "Detailed": (30, True)}[quality]
    stride = max(1, round(info["fps"] / fps_target))
    with st.status("Finding the rink…", expanded=True) as status:
        bar = st.progress(0.0)
        auto = auto_calibrate(path, start_s=start, max_s=min(span, 60.0), step_s=0.5,
                              progress=lambda f, m: bar.progress(f, m))
        if auto is None:
            st.write("⚠️ Not enough rink markings found, so this runs in **speed-only mode** (box height as a "
                     "ruler). Calibrate by hand in **② Calibrate** for positions and positional stats.")
            cal, ref = None, int(start * info["fps"])
        else:
            cal, ref = auto.calibration, auto.frame
            st.write(f"✅ Rink found at {auto.frame / info['fps']:.1f}s from {len(auto.marks)} markings + "
                     f"{len(auto.lines)} painted line(s) · fit error {auto.error_ft:.1f} ft")
            ss.quick_overlay = cv2.cvtColor(draw_rink_overlay(pipeline.read_frame(path, auto.frame), cal.H),
                                            cv2.COLOR_BGR2RGB)
            st.image(ss.quick_overlay, width=560,
                     caption="Where RinkIQ thinks the rink is. The drawn lines should sit on the painted ones.")
        status.update(label="Tracking players…")
        live, history, n = st.empty(), {}, {"i": 0}

        def show(frame, rows):
            n["i"] += 1
            if n["i"] % 4 == 1:
                live.image(live_preview(frame, rows, history), use_container_width=True)

        cfg = pipeline.RunConfig(frame_stride=stride, use_pose=pose, read_numbers=True, number_every=3,
                                 start_s=start, end_s=start + span)
        players, puck, meta = pipeline.run_video(path, calibration=cal, reference_frame=ref, config=cfg,
                                                 progress=lambda f, m: bar.progress(f, m), on_frame=show)
        live.empty()
        if players.empty:
            status.update(label="No players found", state="error")
            st.error("No skaters were detected. RinkIQ needs hockey film from an elevated camera.")
            return
        status.update(label="Computing stats…")
        ss.names = {}
        ss.video, ss.info, ss.cal, ss.source = path, info, cal, "video"
        set_result(players, puck, meta)
        team_of = ss.result["tracks"].drop_duplicates("track_id").set_index("track_id")["team"].to_dict()
        ss.names = identity.assign_names(meta.get("numbers", {}), team_of, _rosters({"A": ra, "B": rb}, season))
        ss.render_path = None
        if render_it:
            status.update(label="Rendering the broadcast video…")
            out = UPLOAD_DIR / f"rinkiq_{Path(path).stem}_{start:.0f}_{start + min(span, 30):.0f}.mp4"
            render_broadcast(path, ss.result, puck, meta, str(out), ss.names, start_s=start,
                             end_s=start + min(span, 30.0), out_width=960,
                             progress=lambda f, m: bar.progress(f, m))
            ss.render_path = str(out)
        status.update(label="Done!", state="complete", expanded=False)
    _show_results(ss)


def _show_results(ss):
    res = ss.get("result")
    if res is None or ss.get("source") != "video":
        return
    meta = ss.raw[2]
    summ = res["summary"]
    skaters = summ[summ["team"].isin(["A", "B"]) & ~summ["label"].str.startswith("goalie")]
    ev = res["events"]
    cards = report_cards(summ, res["review"], ss.names)
    fastest = skaters.loc[skaters["top_speed_mph"].idxmax()] if skaters["top_speed_mph"].notna().any() else None
    _tiles([("Skaters tracked", len(skaters[skaters["seconds"] >= 2])),
            ("Players named", sum(1 for v in ss.names.values() if " " in v)),
            ("Top speed", f"{fastest['top_speed_mph']:.1f}" if fastest is not None else "–"),
            ("Shots · passes", f"{int((ev['type'] == 'shot').sum()) if not ev.empty else 0} · "
                               f"{int((ev['type'] == 'pass').sum()) if not ev.empty else 0}"),
            ("Mode", "rink feet" if meta.get("calibrated") else "speed only")])
    if ss.get("render_path") and Path(ss.render_path).exists():
        st.video(ss.render_path)
    if cards:
        st.markdown("#### ⭐ Three stars")
        cols = st.columns(3)
        for col, c, stars in zip(cols, cards[:3], ["⭐⭐⭐", "⭐⭐", "⭐"]):
            items = "".join(f"<li>{s}</li>" for s in c.strengths[:3]) or "<li>Solid all-round shift</li>"
            col.markdown(f"<div class='rq-star'><span class='g'>{c.grade}</span><div class='s'>{stars}</div>"
                         f"<h3>{c.name}</h3><div style='color:#5b6b7d'>Team {c.team} · {c.minutes:.1f} min tracked"
                         f"</div><ul>{items}</ul></div>", unsafe_allow_html=True)
        st.write("")
        teams = team_comparison(summ, ev)
        st.dataframe(teams, use_container_width=True)
        report = html_report(cards, teams)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for key in ("summary", "events", "review", "tracks"):
                if not res[key].empty:
                    df = res[key].copy()
                    if "track_id" in df:
                        df.insert(1, "player", df["track_id"].map(lambda t: identity.label(t, ss.names)))
                    z.writestr(f"{key}.csv", df.to_csv(index=False))
        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇ Game report (HTML)", report, "rinkiq_report.html", "text/html", use_container_width=True)
        d2.download_button("⬇ All stats (CSV zip)", buf.getvalue(), "rinkiq_stats.zip", "application/zip",
                           use_container_width=True)
        if ss.get("render_path") and Path(ss.render_path).exists():
            d3.download_button("⬇ Broadcast video", Path(ss.render_path).read_bytes(), Path(ss.render_path).name,
                               "video/mp4", use_container_width=True)
    st.caption("Want more? **④ Stats** has every number and chart, **⑤ Film review** lists every out-of-position "
               "moment, and **🎬 Broadcast** renders any other stretch of the video.")

