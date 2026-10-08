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

from rinkiq import metrics, models, pipeline, simulate
from rinkiq.camera import shots
from rinkiq.draw import annotate
from rinkiq.rink import LANDMARKS, RINK_LENGTH, RINK_WIDTH, Calibration, rink_layout, suggest_landmarks

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
    return ss.names.get(tid, f"#{int(tid)}")


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
tab_film, tab_cal, tab_track, tab_stats, tab_review, tab_how = st.tabs(
    ["① Film", "② Calibrate", "③ Detect & track", "④ Stats", "⑤ Film review", "How it works"])

# ---------------------------------------------------------------- 1. film
with tab_film:
    st.markdown("Pick where the film comes from. **No video handy?** The synthetic game runs the whole "
                "stats layer instantly with known ground truth.")
    src = st.radio("Source", ["Synthetic demo game (instant)", "Sample NHL broadcast clip", "Upload my own film"],
                   horizontal=True)
    if src.startswith("Synthetic"):
        st.info("A 20-second scripted sequence: breakout → zone entry → point shot → counter-attack. "
                "Team B's left winger deliberately cheats up ice — watch the analytics catch it.")
        if st.button("▶ Load demo game", type="primary"):
            ss.source, ss.video, ss.cal = "demo", None, None
            set_result(*simulate.simulate_game())
            st.rerun()
    if ss.source == "demo":
        st.success("Demo game loaded — open **④ Stats** and **⑤ Film review**.")
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
            st.image(cv2.cvtColor(top, cv2.COLOR_BGR2RGB), use_container_width=True,
                     caption="Top-down check: the frame warped onto the rink. Painted lines should sit on the "
                             "drawn blue/red lines.")

# ---------------------------------------------------------------- 3. detect & track
with tab_track:
    if ss.source == "demo":
        st.info("The demo game skips detection (the tracks are simulated). See ④ Stats.")
    elif not ss.video:
        st.info("Load a film in ① first.")
    else:
        dur = ss.info["n_frames"] / ss.info["fps"]
        c1, c2, c3 = st.columns(3)
        seconds = c1.slider("Seconds to analyse", 1.0, float(max(1.0, dur)), float(min(dur, 10.0)), 0.5)
        speed = c2.select_slider("Speed vs. detail", ["Fast", "Balanced", "Detailed"], value="Balanced",
                                 help="Fast analyses ~15 fps without pose; Detailed analyses ~30 fps with pose "
                                      "on every other frame.")
        use_pose = c3.toggle("Body orientation (pose model)", value=speed != "Fast")
        target_fps = {"Fast": 15, "Balanced": 30, "Detailed": 30}[speed]
        cfg = pipeline.RunConfig(frame_stride=max(1, round(ss.info["fps"] / target_fps)),
                                 max_frames=int(seconds * ss.info["fps"]), use_pose=use_pose,
                                 pose_every={"Fast": 6, "Balanced": 3, "Detailed": 2}[speed])
        st.caption(f"Mode: **{'rink feet (calibrated)' if ss.cal else 'speed-only (uncalibrated)'}** · "
                   f"~{cfg.max_frames // cfg.frame_stride} frames to analyse. On a laptop CPU expect "
                   f"~0.3–0.8 s per frame.")
        if st.button("▶ Run detection + tracking", type="primary"):
            bar = st.progress(0.0, text="Loading models from Hugging Face…")
            ref = getattr(ss.cal, "reference_frame", 0) if ss.cal else 0
            players, puck, meta = pipeline.run_video(ss.video, calibration=ss.cal, reference_frame=ref, config=cfg,
                                                     progress=lambda f, m: bar.progress(f, text=m))
            bar.empty()
            if players.empty:
                st.error("No players detected — is this hockey film from an elevated camera?")
            else:
                set_result(players, puck, meta)
                ss.run_msg = (f"Tracked {players['track_id'].nunique()} people across "
                              f"{players['frame'].nunique()} frames · puck seen in {len(puck)} frames. "
                              f"Open **④ Stats**.")
                st.rerun()
        if ss.get("run_msg") and ss.result is not None:
            st.success(ss.run_msg)
        if ss.result is not None and ss.raw[2].get("calibrated") is not None:
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
                if ss.raw[2].get("calibrated"):
                    c2.plotly_chart(snapshot_figure(res, float(rows["t"].iloc[0]) if len(rows) else 0.0),
                                    use_container_width=True)
                else:
                    c2.info("Calibrate to see the top-down rink view.")

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

# ---------------------------------------------------------------- how it works
with tab_how:
    st.markdown(Path(__file__).with_name("docs").joinpath("HOW_IT_WORKS.md").read_text())
