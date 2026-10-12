"""🎓 Learn tab: an interactive, step-by-step tour of the computer-vision pipeline.

Each step runs the *real* code from `rinkiq/` on a real frame, with sliders,
so you can see what every stage does to the data. Read along in docs/LEARN.md.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import savgol_filter

from rinkiq import models, pipeline, simulate
from rinkiq.pose import facing_from_keypoints
from rinkiq.rink import LANDMARKS, RINK_LENGTH, RINK_WIDTH, Calibration, pixels_to_rink

CLIP = "Broadcast clip · neutral zone (MTL vs SJ)"
STEPS = ["1 · Pixels", "2 · Detection", "3 · Tracking", "4 · Camera motion", "5 · Homography",
         "6 · Speed & smoothing", "7 · Body pose"]


@st.cache_data(show_spinner=False)
def _frames(idx: tuple[int, ...]) -> list[np.ndarray]:
    clip = str(models.download_sample_clip(CLIP))
    return [pipeline.read_frame(clip, i) for i in idx]


@st.cache_data(show_spinner=False)
def _detect(frame: np.ndarray) -> pd.DataFrame:
    r = models.load("player")(frame, conf=0.05, imgsz=960, verbose=False)[0]
    return pd.DataFrame([dict(x1=b[0], y1=b[1], x2=b[2], y2=b[3], conf=float(c), label=r.names[int(k)])
                         for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())])


def _rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _boxes(img, df, color=(255, 200, 0), label=True):
    out = img.copy()
    for r in df.itertuples():
        cv2.rectangle(out, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), color, 2)
        if label:
            cv2.putText(out, f"{r.label.replace('_player', '')} {r.conf:.2f}", (int(r.x1), int(r.y1) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def learn_tab():
    st.markdown("### 🎓 Learn computer vision, one step at a time")
    st.caption("Every step below runs the project's real code on a real broadcast frame. Move the sliders and "
               "watch what changes. The written guide is in `docs/LEARN.md`.")
    step = st.radio("Step", STEPS, horizontal=True, label_visibility="collapsed")
    try:
        with st.spinner("Fetching the sample clip from Hugging Face…"):
            f0, f6, f30 = _frames((0, 6, 30))
    except Exception as e:  # noqa: BLE001
        st.error(f"Couldn't download the sample clip ({e}). Step 6 still works offline.")
        f0 = f6 = f30 = None

    if step == STEPS[0] and f0 is not None:
        st.markdown("**A video is a stack of images, and an image is a grid of numbers.** Each pixel stores "
                    "three numbers (blue, green, red in OpenCV's order) from 0 to 255. Computer vision is maths "
                    "on that grid.")
        c1, c2 = st.columns([3, 2])
        x = c2.slider("Zoom x", 0, f0.shape[1] - 16, 368, 8)
        y = c2.slider("Zoom y", 0, f0.shape[0] - 16, 160, 8)
        vis = f0.copy()
        cv2.rectangle(vis, (x, y), (x + 16, y + 16), (0, 255, 255), 2)
        c1.image(_rgb(vis), caption=f"Frame shape = {f0.shape} → height × width × colour channels",
                 use_container_width=True)
        patch = f0[y:y + 16, x:x + 16]
        c2.image(cv2.resize(_rgb(patch), (256, 256), interpolation=cv2.INTER_NEAREST),
                 caption="That 16×16 patch, blown up. Each square is one pixel.")
        c2.dataframe(pd.DataFrame(patch[:6, :6].mean(axis=2).astype(int)), use_container_width=True)
        c2.caption("Brightness (average of B, G, R) of the top-left 6×6 pixels. White ice ≈ 200+, a dark "
                   "jersey ≈ 30–80.")

    elif step == STEPS[1] and f0 is not None:
        st.markdown("**Object detection** draws a *bounding box* around each thing it recognises and gives a "
                    "*class* (team A player, referee…) and a *confidence* (0–1). The model is a YOLO network "
                    "downloaded from Hugging Face. It looks at the whole image once and predicts all boxes at "
                    "the same time.")
        thr = st.slider("Confidence threshold: only keep boxes above this", 0.05, 0.95, 0.30, 0.05)
        with st.spinner("Running the player model…"):
            det = _detect(f0)
        keep = det[det["conf"] >= thr]
        c1, c2 = st.columns([3, 1])
        c1.image(_rgb(_boxes(f0, keep)), use_container_width=True)
        c2.metric("Boxes kept", len(keep), f"{len(det) - len(keep)} filtered out", delta_color="off")
        c2.markdown("Too **low** a threshold gives false alarms (crowd, reflections). Too **high** misses "
                    "real players. 0.3 is this project's default. This is the classic **precision vs. "
                    "recall** trade-off.")
        c2.dataframe(keep[["label", "conf"]].round(2), hide_index=True, use_container_width=True)

    elif step == STEPS[2] and f0 is not None:
        st.markdown("**Tracking** answers *\"is this box the same player as that box a moment ago?\"* The core "
                    "idea is **IoU (intersection over union)**: how much two boxes overlap, from 0 (none) to 1 "
                    "(identical). Boxes in consecutive frames with high IoU are probably the same player. "
                    "ByteTrack adds motion prediction and a second pass for low-confidence boxes.")
        gap = st.select_slider("Frames apart", [6, 30], value=6,
                               help="At 60 fps, 6 frames = 0.1 s and 30 frames = 0.5 s")
        later = f6 if gap == 6 else f30
        a = _detect(f0).query("conf >= 0.3").reset_index(drop=True)
        b = _detect(later).query("conf >= 0.3").reset_index(drop=True)
        rows = []
        for i, ra in a.iterrows():
            scores = [_iou(ra[["x1", "y1", "x2", "y2"]].to_numpy(), rb[["x1", "y1", "x2", "y2"]].to_numpy())
                      for _, rb in b.iterrows()]
            j = int(np.argmax(scores)) if scores else -1
            rows.append(dict(box_then=i, best_match_now=j if j >= 0 and scores[j] > 0.3 else "—",
                             IoU=round(max(scores), 2) if scores else 0))
        vis = cv2.addWeighted(_boxes(f0, a, (255, 120, 0), False), 0.5, _boxes(later, b, (0, 220, 255), False), 0.5, 0)
        c1, c2 = st.columns([3, 2])
        c1.image(_rgb(vis), caption="Orange = earlier boxes, yellow = later boxes (the two frames blended)",
                 use_container_width=True)
        c2.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        c2.caption("At 0.1 s apart most players overlap a lot (IoU > 0.5) and are easy to match. At 0.5 s fast "
                   "skaters have moved off their old box, which is why trackers also predict motion.")

    elif step == STEPS[3] and f0 is not None:
        st.markdown("**The camera moves too.** To separate player motion from camera motion, we find "
                    "*keypoints* (distinctive corners: ad logos, glass edges) with **ORB** and match them "
                    "between frames. Lines that all shift the same way tell us how the camera panned.")
        later = st.radio("Compare frame 0 with", ["frame 6", "frame 30"], horizontal=True)
        g = f6 if later == "frame 6" else f30
        orb = cv2.ORB_create(800)
        k1, d1 = orb.detectAndCompute(cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY), None)
        k2, d2 = orb.detectAndCompute(cv2.cvtColor(g, cv2.COLOR_BGR2GRAY), None)
        m = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2), key=lambda m: m.distance)[:60]
        src = np.float32([k1[x.queryIdx].pt for x in m])
        dst = np.float32([k2[x.trainIdx].pt for x in m])
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        vis = cv2.drawMatches(f0, k1, g, k2, [x for x, ok in zip(m, inl.ravel()) if ok], None,
                              matchColor=(0, 220, 120), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        st.image(_rgb(vis), use_container_width=True,
                 caption=f"{int(inl.sum())} of {len(m)} matches agree on one camera motion (RANSAC inliers, "
                         f"green). Shift ≈ {H[0, 2]:.0f} px sideways.")
        st.caption("**RANSAC** repeatedly guesses a motion from a few random matches and keeps the guess most "
                   "matches agree with, so wrong matches (on moving players) are ignored.")

    elif step == STEPS[4] and f0 is not None:
        st.markdown("**A homography turns camera pixels into rink feet.** Because the ice is flat, one 3×3 "
                    "matrix maps every ice pixel to a top-down rink position. Four known points (faceoff dots, "
                    "line/boards corners) are enough to solve it. Try making one point slightly wrong and watch "
                    "the error grow far away from it.")
        pts = models.SAMPLE_CALIBRATION[CLIP]
        nudge = st.slider("Mis-click the 'far-left neutral dot' by … pixels", 0, 30, 0, 2)
        pairs = [((x + (nudge if i == 0 else 0), y), n) for i, (x, y, n) in enumerate(pts)]
        cal = Calibration.from_pairs(pairs)
        S = 4
        T = np.array([[S, 0, 0], [0, -S, RINK_WIDTH * S], [0, 0, 1]])
        top = cv2.warpPerspective(f0, T @ cal.H, (int(RINK_LENGTH * S), int(RINK_WIDTH * S)))
        for xl, col in [(75, (214, 111, 47)), (125, (214, 111, 47)), (100, (62, 57, 208))]:
            cv2.line(top, (xl * S, 0), (xl * S, int(RINK_WIDTH * S)), col, 2)
        c1, c2 = st.columns([3, 2])
        c1.image(_rgb(top), caption="The frame warped to a top-down view (drawn lines = true rink lines)",
                 use_container_width=True)
        truth = np.array(LANDMARKS["End-zone dot · left · far"])
        ref = Calibration.from_pairs([((x, y), n) for x, y, n in pts])
        far_px = pixels_to_rink(np.array([truth]), np.linalg.inv(ref.H))
        moved = pixels_to_rink(far_px, cal.H)[0]
        c2.metric("Mean error at the clicked points", f"{cal.reprojection_error_ft():.1f} ft")
        c2.metric("Error at the far end-zone dot (≈ 40 ft away)", f"{np.linalg.norm(moved - truth):.1f} ft")
        with c2.expander("The matrix itself"):
            st.code(np.array2string(cal.H, precision=4, suppress_small=True))
        c2.caption("Small clicking errors near the points become big errors far away. That's why RinkIQ "
                   "keeps re-anchoring on every faceoff dot it sees during a pan.")

    elif step == STEPS[5]:
        st.markdown("**From positions to speed.** Speed = distance ÷ time between frames. But detections jitter by "
                    "a few pixels, and at 30 fps a 0.3 ft wobble already looks like 9 ft/s! So we **smooth** the "
                    "path first (a Savitzky–Golay filter fits little curves through a sliding window).")
        c1, c2 = st.columns(2)
        noise = c1.slider("Detection jitter (ft)", 0.0, 1.5, 0.4, 0.1)
        window = c2.slider("Smoothing window (frames)", 3, 41, 21, 2)
        clean, _, _ = simulate.simulate_game(seconds=8, pixel_noise_ft=0.0, puck_dropout=0)
        noisy, _, _ = simulate.simulate_game(seconds=8, pixel_noise_ft=noise, puck_dropout=0)
        g0 = clean[clean["track_id"] == 1].sort_values("t")
        g = noisy[noisy["track_id"] == 1].sort_values("t")
        t = g["t"].to_numpy()

        def speed(x, y):
            return np.hypot(np.gradient(x, t), np.gradient(y, t)) * 0.6818

        raw = speed(g["x"].to_numpy(), g["y"].to_numpy())
        sm = speed(savgol_filter(g["x"], window, 2), savgol_filter(g["y"], window, 2))
        true = speed(g0["x"].to_numpy(), g0["y"].to_numpy())
        fig = go.Figure([go.Scatter(x=t, y=raw, name="raw (no smoothing)", line=dict(color="#d0393e", width=1)),
                         go.Scatter(x=t, y=sm, name=f"smoothed ({window} frames)", line=dict(color="#2f6fd6", width=2.5)),
                         go.Scatter(x=t, y=true, name="truth", line=dict(color="#0f1b2d", dash="dot"))])
        fig.update_layout(height=380, yaxis_title="mph", xaxis_title="seconds", margin=dict(l=10, r=10, t=10, b=10),
                          legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
        err = np.mean(np.abs(sm - true))
        st.metric("Average error of the smoothed speed", f"{err:.2f} mph",
                  help="Too little smoothing keeps the jitter; too much flattens real accelerations.")
        st.caption("Raise the jitter and the red line goes wild. Raise the window and blue calms down, but "
                   "too wide and it starts to miss real stops and sprints. Picking this trade-off is everyday "
                   "data-science work.")

    elif step == STEPS[6] and f0 is not None:
        st.markdown("**Pose estimation** finds 17 body *keypoints* (nose, shoulders, hips, knees…). From the "
                    "shoulders and hips we estimate which way a player faces. If the right shoulder appears on "
                    "the left of the image, they face the camera.")
        det = _detect(f0).query("conf >= 0.3").sort_values("conf", ascending=False).reset_index(drop=True)
        pick = st.slider("Player", 0, max(0, len(det) - 1), 0)
        r = det.iloc[pick]
        with st.spinner("Running the pose model…"):
            out = pipeline._pose_on_crops(f0, np.array([[r.x1, r.y1, r.x2, r.y2]]))[0]
        pad = 20
        x1, y1 = int(max(0, r.x1 - pad)), int(max(0, r.y1 - pad))
        crop = f0[y1:int(r.y2 + pad), x1:int(r.x2 + pad)].copy()
        if out is None:
            st.info("The pose model didn't find a body in this crop. Try another player.")
        else:
            k, c = out
            k = k - np.array([x1, y1])
            for a_, b_ in [(5, 6), (5, 11), (6, 12), (11, 12), (5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15),
                           (12, 14), (14, 16), (0, 5), (0, 6)]:
                if c[a_] > 0.3 and c[b_] > 0.3:
                    cv2.line(crop, tuple(k[a_].astype(int)), tuple(k[b_].astype(int)), (0, 255, 120), 1)
            for i, (px_, py_) in enumerate(k):
                if c[i] > 0.3:
                    cv2.circle(crop, (int(px_), int(py_)), 2, (0, 200, 255), -1)
            est = facing_from_keypoints(k, c)
            big = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            c1, c2 = st.columns([1, 2])
            c1.image(_rgb(big), caption=f"{r.label}, confidence {r.conf:.2f}")
            if est is None:
                c2.info("Not enough confident shoulder/hip points to estimate facing.")
            else:
                v, conf = est
                facing = "toward the camera" if v[1] > 0.3 else "away from the camera" if v[1] < -0.3 else \
                    ("to the right" if v[0] > 0 else "to the left")
                c2.metric("Estimated facing", facing)
                c2.write(f"Facing vector (image ground plane): x = {v[0]:+.2f}, y = {v[1]:+.2f} · keypoint "
                         f"confidence {conf:.2f}")
                kp = pd.DataFrame(dict(point=["nose", "L shoulder", "R shoulder", "L hip", "R hip"],
                                       x=k[[0, 5, 6, 11, 12], 0].round(0), y=k[[0, 5, 6, 11, 12], 1].round(0),
                                       conf=c[[0, 5, 6, 11, 12]].round(2)))
                c2.dataframe(kp, hide_index=True, use_container_width=True)
        st.caption("Broadcast players are only ~60–100 px tall, so keypoints are rough. That's why RinkIQ "
                   "uses facing only for a coarse stat: *skating backwards %*.")
    st.divider()
    st.markdown("➡️ Now go back to **① Film** and run the full pipeline. Every tab there is one of these "
                "steps at full scale.")

