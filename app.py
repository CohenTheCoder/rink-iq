"""RinkIQ — hockey game-film analytics.  Run:  streamlit run app.py"""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from learn import learn_tab
from quick import quick_tab
from rinkiq import identity, metrics, models, pipeline, simulate
from rinkiq.autocal import auto_calibrate
from rinkiq.camera import shots
from rinkiq.draw import annotate
from rinkiq.render import live_preview, render_broadcast, render_rink_replay
from rinkiq.report import html_report, report_cards, team_comparison
from rinkiq.rink import (LANDMARKS, RINK_LENGTH, RINK_WIDTH, Calibration, draw_rink_overlay, rink_layout,
                         suggest_landmarks)

st.set_page_config(page_title="RinkIQ · hockey film analytics", page_icon="🏒", layout="wide")

TEAM_COLORS = {"A": "#2f6fd6", "B": "#d0393e", "REF": "#8a96a3"}
UPLOAD_DIR = Path(tempfile.gettempdir()) / "rinkiq_uploads"
ss = st.session_state
for key, default in dict(video=None, info=None, cal=None, raw=None, result=None, source=None,
                         names={}).items():
    ss.setdefault(key, default)


# ---------------------------------------------------------------- helpers
@st.cache_data(show_spinner=False)
def camera_shots(path: str, n_frames: int, fps: float) -> list[dict]:
    return shots(path, n_frames, fps)


def step_status() -> list[tuple[str, bool]]:
    return [
        ("Import models from Hugging Face", True),
        ("Load game film", ss.video is not None or ss.source == "demo"),
        ("Calibrate the rink (optional)", ss.cal is not None or ss.source == "demo"),
        ("Detect & track every player", ss.raw is not None),
        ("Compute advanced stats", ss.result is not None),
        ("Present & review", ss.result is not None),
    ]


def set_result(players, puck, meta):
    ss.raw = (players, puck, meta)
    ss.result = metrics.analyze(players, puck, meta)
    ss.names = meta.get("names", {})


def name(tid) -> str:
    return identity.label(tid, ss.names)


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def cached_roster(team: str, season: str) -> dict:
    return identity.nhl_roster(team, season)


def roster_panel(meta: dict):
    """Jersey numbers → names. NHL rosters come from the public NHL API; anything else can be pasted."""
    nums = meta.get("numbers", {})
    with st.expander(f"👤 Player names: {len(nums)} jersey number(s) read", expanded=bool(nums)):
        if not nums:
            st.caption("No jersey numbers were confirmed. Turn on 'Read jersey numbers' and use more of the "
                       "video. Numbers need a few clear looks at a player's back or front.")
            return
        teams = ["(none)", "Paste a roster"] + [f"{k} · {v}" for k, v in identity.NHL_TEAMS.items()]
        sample = models.HF_REPO.replace("/", "--") in str(ss.video)  # the bundled clips are SJ (dark) vs MTL
        default = {"A": "SJS", "B": "MTL"} if sample else {}
        c1, c2, c3 = st.columns([2, 2, 1])
        season = c3.text_input("NHL season", "20252026", help="e.g. 20252026 for 2025–26")
        rosters = {}
        for col, team in ((c1, "A"), (c2, "B")):
            idx = next((i for i, t in enumerate(teams) if t.startswith(default.get(team, "~"))), 0)
            pick = col.selectbox(f"Team {team} {'(dark jerseys in the sample)' if team == 'A' and default else ''}",
                                 teams, index=idx, key=f"roster_{team}")
            if pick == "Paste a roster":
                txt = col.text_area(f"Team {team} roster: one '63 Firstname Lastname' per line", key=f"paste_{team}",
                                    height=120)
                rosters[team] = identity.parse_roster(txt)
            elif pick != "(none)":
                try:
                    rosters[team] = cached_roster(pick.split(" · ")[0], season)
                except Exception as e:  # noqa: BLE001
                    col.warning(f"Couldn't fetch the roster ({e}).")
        team_of = ss.result["tracks"].drop_duplicates("track_id").set_index("track_id")["team"].to_dict()
        ss.names = identity.assign_names({int(k): tuple(v) for k, v in nums.items()}, team_of, rosters)
        rows = [dict(track=k, team=team_of.get(int(k), "?"), number=v[0], vote_share=v[1], name=ss.names.get(int(k), ""))
                for k, v in nums.items()]
        st.dataframe(pd.DataFrame(rows).sort_values(["team", "number"]), hide_index=True, use_container_width=True)
        if meta.get("merged_tracks"):
            st.caption(f"Re-identification: {len(meta['merged_tracks'])} broken track piece(s) were joined to their "
                       "player because they share a team and a jersey number.")


def rink_figure(height=420, title=None) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**rink_layout(height), showlegend=True, legend=dict(orientation="h", y=-0.02))
    if title:
        fig.update_layout(title=title)
    return fig


def snapshot_figure(res: dict, t: float, highlight: int | None = None) -> go.Figure:
    tr = res["tracks"]
    near = tr.loc[(tr["t"] - t).abs() < 0.02]
    fig = rink_figure(380)
    if highlight is not None:
        h = near[near["track_id"] == highlight]
        fig.add_trace(go.Scatter(x=h["sx"], y=h["sy"], mode="markers", name="selected", hoverinfo="skip",
                                 marker=dict(size=34, color="rgba(242,169,0,0.25)", line=dict(color="#f2a900", width=2))))
    for team, g in near.groupby("team"):
        fig.add_trace(go.Scatter(x=g["sx"], y=g["sy"], mode="markers+text", name=f"Team {team}",
                                 text=[name(i) for i in g["track_id"]], textposition="top center",
                                 marker=dict(size=14, color=TEAM_COLORS.get(team, "#888"),
                                             line=dict(color="white", width=1.5))))
        for _, r in g.iterrows():  # facing arrows
            if not np.isnan(r.get("face_x", np.nan)):
                fig.add_annotation(x=r["sx"] + 7 * r["face_x"], y=r["sy"] + 7 * r["face_y"], ax=r["sx"],
                                   ay=r["sy"], xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                                   arrowhead=2, arrowwidth=1.5, arrowcolor="#f2a900", text="")
    pk = res["puck"]
    if not pk.empty:
        p = pk.loc[(pk["t"] - t).abs() < 0.03]
        if not p.empty:
            fig.add_trace(go.Scatter(x=p["x"].iloc[:1], y=p["y"].iloc[:1], mode="markers", name="Puck",
                                     marker=dict(size=9, color="#111", symbol="circle")))
    return fig


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🏒 RinkIQ")
    st.caption("Computer vision → x/y tracking → advanced hockey stats")
    st.subheader("The process")
    for i, (label, done) in enumerate(step_status(), 1):
        st.markdown(f"{'✅' if done else '⬜️'} **{i}.** {label}")
    st.divider()
    st.markdown("**Models** (Hugging Face)\n\n"
                "- `AlexRHaigh/Hockey-Vision` — players+teams, puck, rink lines, faceoff dots\n"
                "- `yolov8n-pose` — body keypoints → facing")
    if ss.result is not None and st.button("Start over", use_container_width=True):
        for k in ("video", "info", "cal", "raw", "result", "source"):
            ss[k] = None
        ss.names = {}
        st.rerun()

st.title("Hockey game-film analytics")
st.caption("New here? **🚀 Analyze a video** does everything in one click. The numbered tabs are the same pipeline "
           "step by step, with full control.")
tab_quick, tab_film, tab_cal, tab_track, tab_broadcast, tab_stats, tab_review, tab_learn, tab_how = st.tabs(
    ["🚀 Analyze a video", "① Film", "② Calibrate", "③ Detect & track", "🎬 Broadcast", "④ Stats", "⑤ Film review",
     "🎓 Learn CV", "How it works"])

# ---------------------------------------------------------------- 🚀 one-button flow
with tab_quick:
    quick_tab(ss, set_result)

# ---------------------------------------------------------------- 1. film
with tab_film:
    st.markdown("Pick where the film comes from. **No video handy?** The synthetic game runs the whole "
                "stats layer instantly with known ground truth.")
    src = st.radio("Source", ["Synthetic demo game (instant)", "Pre-analysed NHL clip (instant)",
                              "Sample NHL broadcast clip", "Upload my own film"], horizontal=True)
    if src.startswith("Pre-analysed"):
        st.info("A real 10-second NHL broadcast (San Jose vs Montréal) that has already been through the full "
                "pipeline: calibration, tracking, pose, jersey numbers. Loads instantly. Then open "
                "**🎬 Broadcast**, **④ Stats** or **⑤ Film review**. To watch the models run live, pick "
                "*Sample NHL broadcast clip* instead.")
        if st.button("▶ Load pre-analysed clip", type="primary"):
            with st.spinner("Downloading the clip from Hugging Face…"):
                ss.video = str(models.download_sample_clip(list(models.SAMPLE_CLIPS)[0]))
            ss.info = pipeline.video_info(ss.video)
            first = list(models.SAMPLE_CALIBRATION)[0]
            ss.cal = Calibration.from_pairs([((x, y), n) for x, y, n in models.SAMPLE_CALIBRATION[first]])
            ss.cal.reference_frame = 0
            ss.source, ss.names, ss.render_path = "video", {}, None
            set_result(*pipeline.load_saved_run(Path(__file__).with_name("samples") / "sample_run"))
            ss.run_msg = "Pre-analysed clip loaded: add the rosters below (pre-filled for SJ vs MTL)."
            st.rerun()
    if src.startswith("Synthetic"):
        st.info("A 20-second scripted sequence: breakout → zone entry → point shot → counter-attack. "
                "Team B's left winger deliberately cheats up ice — watch the analytics catch it.")
        if st.button("▶ Load demo game", type="primary"):
            ss.source, ss.video, ss.cal = "demo", None, None
            set_result(*simulate.simulate_game())
            st.rerun()
    if ss.source == "demo":
        st.success("Demo game loaded: open **🎬 Broadcast**, **④ Stats** and **⑤ Film review**.")
    elif src.startswith("Pre-analysed") or src.startswith("Synthetic"):
        if ss.result is not None and ss.source == "video":
            st.success("Loaded. Open **③ Detect & track** for player names, then **🎬 Broadcast** or **④ Stats**.")
    else:
        if src.startswith("Sample"):
            clip_name = st.selectbox("Clip", list(models.SAMPLE_CLIPS))
            st.caption("10-second clips from the model author's examples (they carry a few light overlays).")
            if st.button("Download clip from Hugging Face", type="primary"):
                with st.spinner("Downloading…"):
                    ss.video = str(models.download_sample_clip(clip_name))
                ss.source, ss.cal, ss.raw, ss.result = "video", None, None, None
                st.rerun()
        else:
            up = st.file_uploader("Game film (mp4 / mov / avi)", type=["mp4", "mov", "avi", "mkv"])
            if up is not None and (ss.video is None or Path(ss.video).name != up.name):
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                dest = UPLOAD_DIR / up.name
                dest.write_bytes(up.getbuffer())
                ss.video, ss.source, ss.cal, ss.raw, ss.result = str(dest), "video", None, None, None
                st.rerun()
        if ss.video:
            info = pipeline.video_info(ss.video)
            ss.info = info
            c1, c2 = st.columns([3, 2])
            with c1:
                st.video(ss.video)
            with c2:
                st.metric("Length", f"{info['n_frames'] / info['fps']:.1f} s")
                st.metric("Frame rate", f"{info['fps']:.0f} fps")
                st.metric("Resolution", f"{info['width']}×{info['height']}")
                with st.spinner("Looking for camera cuts…"):
                    segs = camera_shots(ss.video, info["n_frames"], info["fps"])
                if len(segs) > 1:
                    st.warning(f"{len(segs)} camera shots found. A calibration only holds within one shot — "
                               "pick the shot's first frame as the reference in ② Calibrate.")
                    st.dataframe(pd.DataFrame(segs), hide_index=True)
            st.success("Next: **② Calibrate** (recommended) or jump to **③ Detect & track**.")

# ---------------------------------------------------------------- 2. calibrate
with tab_cal:
    if ss.source == "demo":
        st.info("The demo game is already in rink coordinates — nothing to calibrate.")
    elif not ss.video:
        st.info("Load a film in ① first.")
    else:
        st.markdown(
            "To turn pixels into **feet on the ice**, match ≥ 4 points in a frame to known rink landmarks. "
            "The faceoff-dot and rink-line models pre-fill what they can — check the names, fix any that are "
            "wrong, and add points (e.g. where the center line meets the far boards) by reading pixel "
            "coordinates off the gridded frame.")
        ref_idx = st.slider("Reference frame", 0, max(0, ss.info["n_frames"] - 1), 0,
                            help="Choose a frame where several dots / line ends are visible.")
        frame = pipeline.read_frame(ss.video, ref_idx)
        preset = next((pts for n, pts in models.SAMPLE_CALIBRATION.items()
                       if ss.video and ss.video.endswith(models.SAMPLE_CLIPS[n])), None)
        b1, b2 = st.columns([1, 2])
        if preset and b2.button("✨ Use hand-checked points for this sample clip (frame 0)"):
            ss.cal_rows = pd.DataFrame([dict(use=True, px=x, py=y, landmark=n) for x, y, n in preset])
            ss.cal_lines = []
            ss.cal_ver = ss.get("cal_ver", 0) + 1
        if st.button("🤖 Calibrate automatically (search the video for rink markings)", type="primary"):
            bar = st.progress(0.0)
            auto = auto_calibrate(ss.video, max_s=60, progress=lambda f, m: bar.progress(f, m))
            bar.empty()
            if auto is None:
                st.warning("Not enough rink markings found. Use the manual steps below.")
            else:
                ss.cal = auto.calibration
                st.success(f"Found the rink at {auto.frame / ss.info['fps']:.1f}s ({len(auto.marks)} markings + "
                           f"{len(auto.lines)} lines, fit error {auto.error_ft:.1f} ft). Move the slider to "
                           f"frame {auto.frame} to see the overlay.")
        if b1.button("🔎 Auto-detect faceoff dots & lines"):
            with st.spinner("Running dots + rink models…"):
                feat = pipeline.detect_rink_features(frame)
            cl = feat["lines"].get("Center_Line")
            guess = suggest_landmarks(feat["dots"], float(np.median(cl)) if cl else None,
                                      feat["lines"].get("Blue_Line", []), frame.shape[0])
            ss.cal_rows = pd.DataFrame([dict(use=g is not None, px=round(x), py=round(y), landmark=g)
                                        for (x, y), g in zip(feat["dots"], guess)])
            ss.cal_lines = feat["rink_boxes"]
            ss.cal_ver = ss.get("cal_ver", 0) + 1
        rows = ss.get("cal_rows", pd.DataFrame(columns=["use", "px", "py", "landmark"]))
        show_grid = st.toggle("Show pixel grid", value=True)
        vis = frame.copy()
        if show_grid:
            for x in range(0, vis.shape[1], 100):
                cv2.line(vis, (x, 0), (x, vis.shape[0]), (80, 200, 80), 1)
                cv2.putText(vis, str(x), (x + 3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 200, 80), 1)
            for y in range(0, vis.shape[0], 100):
                cv2.line(vis, (0, y), (vis.shape[1], y), (80, 200, 80), 1)
                cv2.putText(vis, str(y), (3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 200, 80), 1)
        for name_, b in ss.get("cal_lines", []):
            cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 160, 60), 1)
            cv2.putText(vis, name_, (int(b[0]), int(b[1]) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 160, 60), 1)
        for i, r in rows.reset_index(drop=True).iterrows():
            if pd.notna(r["px"]):
                cv2.circle(vis, (int(r["px"]), int(r["py"])), 8, (0, 215, 255), 2)
                cv2.putText(vis, str(i), (int(r["px"]) + 10, int(r["py"]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 215, 255), 2)
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)
        edited = st.data_editor(
            rows, num_rows="dynamic", use_container_width=True, key=f"cal_editor_{ss.get('cal_ver', 0)}",
            column_config={
                "use": st.column_config.CheckboxColumn("Use", default=True),
                "px": st.column_config.NumberColumn("Pixel x", step=1),
                "py": st.column_config.NumberColumn("Pixel y", step=1),
                "landmark": st.column_config.SelectboxColumn("Rink landmark", options=list(LANDMARKS)),
            })
        c1, c2 = st.columns(2)
        if c1.button("📐 Solve calibration", type="primary"):
            use = edited[edited["use"].fillna(False) & edited["landmark"].notna() & edited["px"].notna()]
            try:
                cal = Calibration.from_pairs([((float(r.px), float(r.py)), r.landmark) for r in use.itertuples()])
                cal.reference_frame = ref_idx
                ss.cal = cal
                st.rerun()
            except ValueError as e:
                st.error(str(e))
        if c2.button("Skip — speed-only mode"):
            ss.cal = None
            st.info("Without calibration, speeds use each player's height as a ruler; zone, goal-side and "
                    "positioning stats are disabled.")
        if ss.cal is not None:
            err = ss.cal.reprojection_error_ft()
            (st.success if err < 3 else st.warning)(
                f"Calibrated with {len(ss.cal.pixel_points)} points · mean error {err:.1f} ft "
                f"{'(good)' if err < 3 else '(double-check the landmark names)'}")
            S = 4  # px per foot in the preview
            T = np.array([[S, 0, 0], [0, -S, RINK_WIDTH * S], [0, 0, 1]])  # rink ft → preview px (y up)
            top = cv2.warpPerspective(frame, T @ ss.cal.H, (int(RINK_LENGTH * S), int(RINK_WIDTH * S)))
            for xl, col in [(75, (214, 111, 47)), (125, (214, 111, 47)), (100, (62, 57, 208))]:
                cv2.line(top, (xl * S, 0), (xl * S, int(RINK_WIDTH * S)), col, 2)
            c1, c2 = st.columns(2)
            c1.image(cv2.cvtColor(draw_rink_overlay(frame, ss.cal.H), cv2.COLOR_BGR2RGB), use_container_width=True,
                     caption="The calibrated rink drawn onto the frame: it should sit on the painted lines.")
            c2.image(cv2.cvtColor(top, cv2.COLOR_BGR2RGB), use_container_width=True,
                     caption="Top-down check: the frame warped onto the rink. Painted lines should sit on the "
                             "drawn blue/red lines.")

# ---------------------------------------------------------------- 3. detect & track
with tab_track:
    if ss.source == "demo":
        st.info("The demo game skips detection (the tracks are simulated). See ④ Stats and 🎬 Broadcast.")
    elif not ss.video:
        st.info("Load a film in ① first.")
    else:
        dur = ss.info["n_frames"] / ss.info["fps"]
        c1, c2 = st.columns([3, 2])
        t_range = c1.slider("Part of the video to analyse (seconds)", 0.0, float(max(1.0, dur)),
                            (0.0, float(min(dur, 10.0))), 0.5,
                            help="Full games work too. Use the 'Long game' preset and leave it running.")
        speed = c2.select_slider("Speed vs. detail", ["Long game", "Fast", "Balanced", "Detailed"], value="Balanced",
                                 help="Long game: 10 analysed fps, no pose, about real-time×3 on a laptop. "
                                      "Detailed: 30 fps with pose on every other frame.")
        c3, c4, c5 = st.columns(3)
        use_pose = c3.toggle("Body orientation (pose)", value=speed in ("Balanced", "Detailed"))
        read_numbers = c4.toggle("Read jersey numbers → names", value=True)
        save_job = c5.toggle("Save progress to disk (resumable)", value=(t_range[1] - t_range[0]) > 120,
                             help="Results are written in chunks. If the app or laptop stops, run again to "
                                  "continue where it left off.")
        target_fps = {"Long game": 10, "Fast": 15, "Balanced": 30, "Detailed": 30}[speed]
        stride = max(1, round(ss.info["fps"] / target_fps))
        ref = getattr(ss.cal, "reference_frame", 0) if ss.cal else 0
        cfg = pipeline.RunConfig(frame_stride=stride, use_pose=use_pose, read_numbers=read_numbers,
                                 pose_every={"Long game": 6, "Fast": 6, "Balanced": 3, "Detailed": 2}[speed],
                                 number_every={"Long game": 10, "Fast": 8, "Balanced": 8, "Detailed": 4}[speed],
                                 start_s=max(t_range[0], ref / ss.info["fps"]), end_s=t_range[1])
        n_an = int((t_range[1] - t_range[0]) * ss.info["fps"] / stride)
        st.caption(f"Mode: **{'rink feet (calibrated)' if ss.cal else 'speed-only (uncalibrated)'}** · "
                   f"~{n_an:,} frames to analyse · roughly {n_an * 0.45 / 60:.0f}–{n_an * 0.8 / 60:.0f} min on a "
                   f"laptop CPU (much faster on a GPU).")
        job_dir = (UPLOAD_DIR / "jobs" / f"{Path(ss.video).stem}_{int(t_range[0])}_{int(t_range[1])}_{stride}"
                   if save_job else None)
        if job_dir is not None and (job_dir / "state.json").exists():
            st.info("💾 A saved run for this video and range exists. Running again **resumes** it. "
                    "Delete it to start fresh.")
            if st.button("Delete saved progress"):
                import shutil

                shutil.rmtree(job_dir, ignore_errors=True)
                st.rerun()
        if st.button("▶ Run detection + tracking", type="primary"):
            bar = st.progress(0.0, text="Loading models from Hugging Face…")
            live_box = st.empty()
            history, counter = {}, {"n": 0}

            def show_live(frame, live):
                counter["n"] += 1
                if counter["n"] % 3 == 1:
                    live_box.image(live_preview(frame, live, history), use_container_width=True,
                                   caption="Live: what the models see right now (stats are computed at the end)")

            players, puck, meta = pipeline.run_video(ss.video, calibration=ss.cal, reference_frame=ref, config=cfg,
                                                     progress=lambda f, m: bar.progress(f, text=m),
                                                     on_frame=show_live, job_dir=job_dir)
            bar.empty()
            live_box.empty()
            if players.empty:
                st.error("No players detected. Is this hockey film from an elevated camera?")
            else:
                ss.names = {}
                set_result(players, puck, meta)
                ss.render_path = None
                skipped = sum(s["status"].startswith("skipped") for s in meta["shots"])
                ss.run_msg = (f"Tracked {players['track_id'].nunique()} people across "
                              f"{players['frame'].nunique():,} frames in {len(meta['shots'])} camera shot(s)"
                              f"{f' ({skipped} skipped: close-ups/replays)' if skipped else ''} · puck seen in "
                              f"{len(puck):,} frames · {len(meta['numbers'])} jersey number(s) read. "
                              f"Add rosters below, then open **🎬 Broadcast** or **④ Stats**.")
                st.rerun()
        if ss.get("run_msg") and ss.result is not None:
            st.success(ss.run_msg)
        if ss.result is not None and ss.raw is not None:
            meta = ss.raw[2]
            if len(meta.get("shots", [])) > 1:
                with st.expander(f"🎥 Camera shots ({len(meta['shots'])})"):
                    sh = pd.DataFrame(meta["shots"]).drop(columns=["H"], errors="ignore")
                    sh["start_s"] = (sh["start_frame"] / meta["fps"]).round(1)
                    sh["end_s"] = (sh["end_frame"] / meta["fps"]).round(1)
                    st.dataframe(sh[["start_s", "end_s", "status"]], hide_index=True, use_container_width=True)
                    st.caption("After each cut the tracker restarts. Calibrated runs re-locate the new view on the "
                               "rink by matching it to views already calibrated. Views that don't match "
                               "(close-ups, replays) are skipped.")
            roster_panel(meta)
            res = ss.result
            tr = res["tracks"]
            frames = sorted(tr["frame"].unique())
            if frames:
                fno = st.select_slider("Scrub through the film", options=frames, value=frames[len(frames) // 2])
                rows = tr[tr["frame"] == fno]
                img = pipeline.read_frame(ss.video, int(fno))
                pk = ss.raw[1]
                pp = pk[pk["frame"] == fno]
                img = annotate(img, rows, (pp["px"].iloc[0], pp["py"].iloc[0]) if len(pp) else None, ss.names)
                c1, c2 = st.columns([3, 2])
                c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True,
                         caption="Boxes = team colour · label = live speed · yellow arrow = body facing")
                if meta.get("calibrated"):
                    c2.plotly_chart(snapshot_figure(res, float(rows["t"].iloc[0]) if len(rows) else 0.0),
                                    use_container_width=True)
                else:
                    c2.info("Calibrate to see the top-down rink view.")

# ---------------------------------------------------------------- 🎬 broadcast
with tab_broadcast:
    res = ss.result
    if res is None:
        st.info("Load the demo game in ① or run ③ first. Then this tab draws everything back onto the film.")
    else:
        meta = ss.raw[2]
        st.markdown("**The analysis, drawn back onto the film:** team rings under every skater, motion trails "
                    "glued to the ice, live name/speed tags, body-facing arrows, a glowing puck, SHOT/PASS "
                    "pop-ups, a gold ring on the fastest skater, and a mini-map.")
        t_all = float(res["tracks"]["t"].max())
        if ss.source == "demo":
            rng = (0.0, t_all)
        else:
            rng = st.slider("Clip to render (seconds)", 0.0, max(0.5, t_all), (0.0, min(t_all, 20.0)), 0.5)
            st.caption("Rendering costs ~0.05–0.1 s per analysed frame. For a full game, render the moments "
                       "you care about (the shot list is in ④ Stats).")
        if st.button("🎬 Render video", type="primary"):
            out = UPLOAD_DIR / f"rinkiq_{'demo' if ss.source == 'demo' else Path(ss.video).stem}_{rng[0]:.0f}_{rng[1]:.0f}.mp4"
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            bar = st.progress(0.0, "Rendering…")
            if ss.source == "demo":
                render_rink_replay(res, meta, str(out), ss.names, progress=lambda f, m: bar.progress(f, m))
            else:
                render_broadcast(ss.video, res, ss.raw[1], meta, str(out), ss.names, start_s=rng[0], end_s=rng[1],
                                 progress=lambda f, m: bar.progress(f, m))
            bar.empty()
            ss.render_path = str(out)
        if ss.get("render_path") and Path(ss.render_path).exists():
            st.video(ss.render_path)
            st.download_button("⬇ Download video", Path(ss.render_path).read_bytes(), Path(ss.render_path).name,
                               mime="video/mp4")

# ---------------------------------------------------------------- 4. stats
with tab_stats:
    res = ss.result
    if res is None:
        st.info("Run ③ (or load the demo game in ①) to see stats.")
    else:
        meta = ss.raw[2]
        calibrated = meta.get("calibrated", False)
        summ = res["summary"].copy()
        summ.insert(0, "player", summ["track_id"].map(name))
        skaters = summ[summ["team"].isin(["A", "B"]) & ~summ["label"].str.startswith("goalie")]
        reliable = skaters[skaters["seconds"] >= 1.0]
        ev = res["events"]

        k = st.columns(5)
        k[0].metric("Skaters tracked", len(reliable))
        if len(reliable) and reliable["top_speed_mph"].notna().any():
            fastest = reliable.loc[reliable["top_speed_mph"].idxmax()]
            k[1].metric(f"Fastest · {fastest['player']}", f"{fastest['top_speed_mph']:.1f} mph")
        shots_ = ev[ev["type"] == "shot"] if not ev.empty else ev
        k[2].metric("Hardest shot", f"{shots_['speed_mph'].max():.0f} mph" if len(shots_) else "—")
        k[3].metric("Passes", int((ev["type"] == "pass").sum()) if not ev.empty else 0)
        k[4].metric("Turnovers", int((ev["type"] == "turnover").sum()) if not ev.empty else 0)
        if meta.get("synthetic"):
            truth = meta["truth"]
            st.caption(f"Ground truth for the demo: shots at {truth['shots']} mph, {truth['passes']} passes, "
                       f"top speeds {min(truth['top_speed_mph'].values()):.1f}–"
                       f"{max(truth['top_speed_mph'].values()):.1f} mph, and {name(truth['cheater'])} cheats.")

        st.subheader("🏅 Who played well, and why")
        cards = report_cards(res["summary"], res["review"], ss.names)
        if cards:
            stars = st.columns(3)
            for col, c, medal in zip(stars, cards[:3], ["⭐⭐⭐ First star", "⭐⭐ Second star", "⭐ Third star"]):
                with col.container(border=True):
                    st.caption(medal)
                    st.markdown(f"### {c.name} · {c.grade}")
                    st.caption(f"Team {c.team} · {c.score:.0f}/100 · {c.minutes:.1f} min tracked")
                    for s_ in c.strengths:
                        st.markdown(f"✅ {s_}")
            with st.expander(f"📋 All {len(cards)} report cards + team comparison"):
                st.dataframe(team_comparison(res["summary"], ev), use_container_width=True)
                cols = st.columns(3)
                for i, c in enumerate(cards):
                    with cols[i % 3].container(border=True):
                        st.markdown(f"**{c.name}** · Team {c.team} · **{c.grade}** ({c.score:.0f})")
                        for s_ in c.strengths:
                            st.markdown(f"✅ {s_}")
                        for w_ in c.weaknesses:
                            st.markdown(f"⚠️ {w_}")
            st.download_button("⬇ Download the game report (HTML)",
                               html_report(cards, team_comparison(res["summary"], ev)), "rinkiq_report.html",
                               mime="text/html")
            st.caption("Grades compare each skater with the others in this video, using percentiles of every stat "
                       "below (tracks under 2 s are left out). The reasons are the stats that pushed the grade "
                       "up or down.")

        st.subheader("Skater leaderboard")
        st.caption("Skate Score = percentile blend of top speed (25%), avg speed (15%), bursts/min (20%), "
                   "in-position % (20%) and space created (20%). Tracks shorter than 1 s are hidden.")
        cols = ["player", "team", "skate_score", "top_speed_mph", "avg_speed_mph", "distance_ft", "bursts",
                "high_intensity_pct", "backward_pct"]
        if calibrated:
            cols += ["in_position_pct", "goal_side_pct", "gap_ft", "space_ft", "OZ_pct", "NZ_pct", "DZ_pct"]
        cols += [c for c in ["possession_s", "passes", "shots", "hardest_shot_mph", "turnovers"] if c in reliable]
        st.dataframe(
            reliable[cols].round(1), hide_index=True, use_container_width=True,
            column_config={
                "skate_score": st.column_config.ProgressColumn("Skate Score", min_value=0, max_value=100,
                                                               format="%d"),
                "top_speed_mph": st.column_config.NumberColumn("Top mph", format="%.1f"),
                "avg_speed_mph": st.column_config.NumberColumn("Avg mph", format="%.1f"),
                "distance_ft": st.column_config.NumberColumn("Distance ft", format="%.0f"),
                "high_intensity_pct": st.column_config.NumberColumn(">15 mph %", format="%.0f"),
                "backward_pct": st.column_config.NumberColumn("Backward %", format="%.0f",
                                                              help="Share of skating time with body facing "
                                                                   "opposite the direction of travel (pose)"),
                "in_position_pct": st.column_config.NumberColumn("In position %", format="%.0f"),
                "goal_side_pct": st.column_config.NumberColumn("Goal-side %", format="%.0f"),
                "gap_ft": st.column_config.NumberColumn("Gap ft", format="%.0f"),
                "space_ft": st.column_config.NumberColumn("Space ft", format="%.0f"),
            })
        st.download_button("Download all stats (CSV)", reliable.to_csv(index=False), "rinkiq_stats.csv")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Speed over time")
            tr = res["tracks"]
            pick = st.multiselect("Players", reliable["player"].tolist(), default=reliable["player"].tolist()[:3])
            ids = reliable.loc[reliable["player"].isin(pick), "track_id"]
            sub = tr[tr["track_id"].isin(ids)].assign(player=lambda d: d["track_id"].map(name))
            fig = px.line(sub, x="t", y="speed_mph", color="player", labels={"t": "seconds", "speed_mph": "mph"})
            fig.add_hline(y=metrics.HIGH_INTENSITY_MPH, line_dash="dot", annotation_text="high intensity")
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Where they skated")
            if calibrated:
                who = st.selectbox("Heatmap for", ["Team A", "Team B"] + reliable["player"].tolist())
                tr = res["tracks"]
                if who.startswith("Team"):
                    sel = tr[(tr["team"] == who[-1]) & ~tr["label"].str.startswith("goalie")]
                else:
                    sel = tr[tr["track_id"] == reliable.loc[reliable["player"] == who, "track_id"].iloc[0]]
                fig = rink_figure(360)
                fig.add_trace(go.Histogram2dContour(x=sel["sx"], y=sel["sy"], colorscale="Blues", showscale=False,
                                                    contours=dict(coloring="fill", showlines=False), opacity=0.75,
                                                    xbins=dict(start=0, end=200, size=8),
                                                    ybins=dict(start=0, end=85, size=8), name="time spent"))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Calibrate the rink to unlock heatmaps, zone time and positioning.")

        if calibrated:
            st.subheader("Rink replay")
            t_max = float(res["tracks"]["t"].max())
            t = st.slider("Time (s)", 0.0, t_max, min(10.6, t_max), 0.1)
            st.plotly_chart(snapshot_figure(res, t), use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Shots, passes & turnovers")
                fig = rink_figure(360)
                if not ev.empty:
                    style = {"shot": ("star", 16), "pass": ("circle", 9), "turnover": ("x", 11)}
                    for kind, g in ev.groupby("type"):
                        sym, size = style.get(kind, ("circle", 8))
                        fig.add_trace(go.Scatter(
                            x=g["x"], y=g["y"], mode="markers", name=kind,
                            marker=dict(symbol=sym, size=size, color=[TEAM_COLORS.get(tm, "#888") for tm in g["team"]],
                                        line=dict(color="white", width=1)),
                            text=[f"{name(p)} · {s:.0f} mph" for p, s in zip(g["player"], g["speed_mph"])],
                            hovertemplate="%{text}<extra></extra>"))
                st.plotly_chart(fig, use_container_width=True)
                if not ev.empty:
                    show = ev.assign(player=ev["player"].map(name),
                                     to=ev["to_player"].map(lambda i: name(i) if i >= 0 else ""))
                    st.dataframe(show[["t", "type", "player", "to", "speed_mph"]].round(1), hide_index=True,
                                 use_container_width=True)
            with c2:
                st.subheader("Team shape")
                shape = res["shape"]
                if not shape.empty:
                    shape = shape.sort_values("t").copy()
                    for c in ("width_ft", "depth_ft"):  # light smoothing for readability
                        shape[c] = shape.groupby("team")[c].transform(lambda s: s.rolling(9, center=True,
                                                                                          min_periods=1).median())
                    m = shape.melt(id_vars=["t", "team"], value_vars=["width_ft", "depth_ft"])
                    fig = px.line(m, x="t", y="value", color="team", line_dash="variable",
                                  color_discrete_map=TEAM_COLORS, labels={"value": "feet", "t": "seconds"})
                    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Width = sideline-to-sideline spread, depth = end-to-end spread of the 5 skaters. "
                               "A team that stays compact while defending is hard to play through.")

# ---------------------------------------------------------------- 5. review
with tab_review:
    res = ss.result
    if res is None:
        st.info("Nothing to review yet.")
    elif res["review"].empty:
        st.info("Positioning review needs a calibrated rink (or no out-of-position moments were found).")
    else:
        st.markdown("Every stretch (≥ 0.5 s) where a skater was **out of position**: not goal-side of the "
                    "nearest attacker while defending, or > 25 ft away from their usual spot in the team's "
                    "shape. Click one to see the moment.")
        rv = res["review"].copy()
        rv.insert(0, "player", rv["track_id"].map(name))
        sel = st.dataframe(rv[["player", "team", "start_s", "end_s", "reason"]], hide_index=True,
                           use_container_width=True, on_select="rerun", selection_mode="single-row")
        chosen = sel.selection.rows[0] if sel and sel.selection.rows else 0
        r = rv.iloc[chosen]
        st.markdown(f"**{r['player']}** · {r['start_s']:.1f}–{r['end_s']:.1f} s · {r['reason']}")
        c1, c2 = st.columns([3, 2]) if ss.video else (None, st.container())
        if ss.video and c1 is not None:
            tr = res["tracks"]
            img = pipeline.read_frame(ss.video, int(r["frame"]))
            img = annotate(img, tr[tr["frame"] == r["frame"]], None, ss.names)
            c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
        c2.plotly_chart(snapshot_figure(res, float(r["start_s"]), highlight=int(r["track_id"])),
                        use_container_width=True)

# ---------------------------------------------------------------- learn
with tab_learn:
    learn_tab()

# ---------------------------------------------------------------- how it works
with tab_how:
    st.markdown(Path(__file__).with_name("docs").joinpath("HOW_IT_WORKS.md").read_text())
