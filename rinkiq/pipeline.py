"""Step 2 — Detect, track and map every player, frame by frame.

For each frame of the film:

1. **Detect** players / goalies / refs (+ team) with the Hockey-Vision player model.
2. **Track** them across frames with ByteTrack so each skater keeps one ID.
3. **Detect the puck** with the dedicated puck model.
4. **Pose** (optional) — keypoints on each player crop → body-facing direction.
5. **Compensate camera motion** so pans/zooms don't look like skating.
6. **Map to the rink** — feet (bottom-centre of the box) → rink feet via homography.
   Without a calibration we fall back to *stabilised pixels* and use each
   player's box height as a ruler (a skater in stance ≈ 5.5 ft tall).

Output: two tidy DataFrames — `players` and `puck` — which are the contract
for everything in `metrics.py`. You can also feed in tracking data from any
other source (or `simulate.py`) as long as it has the same columns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from . import models
from .camera import CameraMotion, static_overlay_mask
from .pose import facing_from_keypoints
from .rink import Calibration, direction_to_rink, on_rink, pixels_to_rink, snap_correction

SKATER_HEIGHT_FT = 5.5
PLAYER_COLUMNS = ["frame", "t", "track_id", "label", "conf", "x", "y", "px", "py", "box_h",
                  "x1", "y1", "x2", "y2", "face_x", "face_y", "face_conf"]
PUCK_COLUMNS = ["frame", "t", "x", "y", "px", "py", "conf"]


@dataclass
class RunConfig:
    frame_stride: int | None = None  # analyse every Nth frame; None = auto (~30 analysed fps)
    max_frames: int | None = None    # stop early (handy for quick previews)
    player_conf: float = 0.3
    puck_conf: float = 0.35
    use_pose: bool = True
    pose_every: int = 3              # run pose on every Nth analysed frame
    player_imgsz: int = 960          # players are small in broadcast video — upsample a bit
    puck_imgsz: int = 640            # the puck model was trained at 640 and finds more there
    anchor_every: int = 4            # re-check faceoff dots every Nth analysed frame (drift fix)


def video_info(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    info = dict(fps=cap.get(cv2.CAP_PROP_FPS) or 30.0, n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return info


def read_frame(path: str, index: int = 0) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read frame {index} of {path}")
    return frame


def detect_rink_features(frame: np.ndarray) -> dict:
    """Faceoff dots + line positions on one frame (used to pre-fill calibration)."""
    dots = models.load("dots")(frame, conf=0.2, verbose=False)[0]
    rink = models.load("rink")(frame, conf=0.3, verbose=False)[0]
    dot_pts = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, float(c))
               for b, c in zip(dots.boxes.xyxy.tolist(), dots.boxes.conf.tolist())]
    dot_pts.sort(key=lambda d: -d[2])
    lines: dict[str, list[float]] = {}
    for b, c in zip(rink.boxes.xyxy.tolist(), rink.boxes.cls.tolist()):
        name = rink.names[int(c)]
        lines.setdefault(name, []).append((b[0] + b[2]) / 2)
    return {"dots": [(x, y) for x, y, _ in dot_pts], "dot_conf": [c for *_, c in dot_pts], "lines": lines,
            "rink_boxes": [(rink.names[int(c)], b) for b, c in zip(rink.boxes.xyxy.tolist(), rink.boxes.cls.tolist())]}


def dot_observations(frame: np.ndarray) -> tuple[np.ndarray, list[bool]]:
    """Faceoff-dot centres in pixels, and whether each sits inside a detected faceoff circle."""
    dots = models.load("dots")(frame, conf=0.35, verbose=False)[0].boxes.xyxy.cpu().numpy()
    rink = models.load("rink")(frame, conf=0.3, verbose=False)[0]
    circles = [b for b, c in zip(rink.boxes.xyxy.cpu().numpy(), rink.boxes.cls.cpu().numpy().astype(int))
               if rink.names[c] == "Circle"]
    centres = np.stack([(dots[:, 0] + dots[:, 2]) / 2, (dots[:, 1] + dots[:, 3]) / 2], axis=1) if len(dots) \
        else np.zeros((0, 2))
    inside = [any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in circles) for x, y in centres]
    return centres, inside


def _pose_on_crops(frame: np.ndarray, boxes: np.ndarray) -> list[tuple[np.ndarray, np.ndarray] | None]:
    """Run the pose model on each player crop (players are tiny in full frames)."""
    pose = models.load("pose")
    out: list[tuple[np.ndarray, np.ndarray] | None] = []
    crops, offsets = [], []
    H, W = frame.shape[:2]
    for x1, y1, x2, y2 in boxes:
        pw, ph = 0.25 * (x2 - x1), 0.1 * (y2 - y1)
        cx1, cy1 = int(max(0, x1 - pw)), int(max(0, y1 - ph))
        cx2, cy2 = int(min(W, x2 + pw)), int(min(H, y2 + ph))
        crop = frame[cy1:cy2, cx1:cx2]
        scale = 256 / max(1, crop.shape[0])
        crops.append(cv2.resize(crop, None, fx=scale, fy=scale) if crop.size else np.zeros((256, 128, 3), np.uint8))
        offsets.append((cx1, cy1, scale))
    if not crops:
        return out
    results = pose(crops, imgsz=256, conf=0.25, verbose=False)
    for r, (ox, oy, s) in zip(results, offsets):
        if r.keypoints is None or len(r.keypoints) == 0:
            out.append(None)
            continue
        best = int(np.argmax(r.boxes.conf.cpu().numpy()))
        k = r.keypoints.xy[best].cpu().numpy() / s + np.array([ox, oy])
        c = r.keypoints.conf[best].cpu().numpy() if r.keypoints.conf is not None else np.ones(17)
        out.append((k, c))
    return out


def run_video(path: str, calibration: Calibration | None = None, reference_frame: int = 0,
              config: RunConfig | None = None,
              progress: Callable[[float, str], None] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cfg = config or RunConfig()
    info = video_info(path)
    fps = info["fps"]
    stride = cfg.frame_stride or max(1, round(fps / 30))
    player_model = models.load("player")
    puck_model = models.load("puck")
    names = player_model.names

    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, reference_frame)
    ok, ref = cap.read()
    if not ok:
        raise ValueError("Could not read the video.")
    overlay, fixed_camera = static_overlay_mask(path)
    camera = None if fixed_camera else CameraMotion(ref, ignore=overlay)
    cap.set(cv2.CAP_PROP_POS_FRAMES, reference_frame)

    player_rows, puck_rows = [], []
    total = info["n_frames"] - reference_frame
    if cfg.max_frames:
        total = min(total, cfg.max_frames)
    idx, analysed = 0, 0
    while idx < total:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride:
            idx += 1
            continue
        fno, t = reference_frame + idx, idx / fps

        res = player_model.track(frame, persist=True, tracker="bytetrack.yaml", conf=cfg.player_conf,
                                 imgsz=cfg.player_imgsz, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
        cls = res.boxes.cls.cpu().numpy().astype(int) if len(res.boxes) else np.zeros(0, int)
        confs = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.zeros(0)
        ids = (res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else np.full(len(boxes), -1))

        to_ref = camera.update(frame, boxes) if camera is not None else np.eye(3)
        H = calibration.H @ to_ref if calibration is not None and calibration.H is not None else None
        if H is not None and camera is not None and cfg.anchor_every and analysed % cfg.anchor_every == 0:
            centres, inside = dot_observations(frame)
            C = snap_correction(pixels_to_rink(centres, H), inside) if len(centres) else None
            if C is not None:
                C = np.eye(3) + 0.35 * (C - np.eye(3))  # ease in: no sudden jumps in position
                H = C @ H
                to_ref = np.linalg.inv(calibration.H) @ H
                camera.prev_to_ref = to_ref  # later frames chain from the corrected view

        keep = np.array([names[c] != "puck" and i >= 0 for c, i in zip(cls, ids)], dtype=bool)
        poses = (_pose_on_crops(frame, boxes[keep]) if cfg.use_pose and analysed % cfg.pose_every == 0
                 else [None] * int(keep.sum()))
        feet = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], axis=1) if len(boxes) else np.zeros((0, 2))
        feet_ref = pixels_to_rink(feet, to_ref)  # stabilised pixels in the reference frame
        feet_rink = pixels_to_rink(feet, H) if H is not None else None

        pi = 0
        for j in range(len(boxes)):
            label = names[cls[j]]
            if label == "puck" or ids[j] < 0:
                continue
            x1, y1, x2, y2 = boxes[j]
            fx = fy = fc = np.nan
            pose = poses[pi]
            pi += 1
            if pose is not None:
                est = facing_from_keypoints(*pose)
                if est is not None:
                    v, fc = est
                    if H is not None:
                        v = direction_to_rink(tuple(feet[j]), (v[0], v[1]), H)
                    else:  # pixel mode: image y points down, rink y points up
                        v = np.array([v[0], -v[1]])
                    fx, fy = v
            if feet_rink is not None:
                x, y = feet_rink[j]
            else:
                x, y = feet_ref[j][0], -feet_ref[j][1]  # flip so "up" in the image is +y
            player_rows.append([fno, t, int(ids[j]), label, float(confs[j]), float(x), float(y),
                                float(feet[j][0]), float(feet[j][1]), float(y2 - y1),
                                float(x1), float(y1), float(x2), float(y2), fx, fy, fc])

        pr = puck_model(frame, conf=cfg.puck_conf, imgsz=cfg.puck_imgsz, verbose=False)[0]
        if len(pr.boxes):
            b = pr.boxes.xyxy[int(pr.boxes.conf.argmax())].cpu().numpy()
            c = float(pr.boxes.conf.max())
            p = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]])
            p_ref = pixels_to_rink(p, to_ref)[0]
            if H is not None:
                # The puck sits on the ice, so its centre maps straight through H.
                x, y = pixels_to_rink(p, H)[0]
            else:
                x, y = p_ref[0], -p_ref[1]
            puck_rows.append([fno, t, float(x), float(y), float(p[0, 0]), float(p[0, 1]), c])

        analysed += 1
        idx += 1
        if progress:
            progress(min(1.0, idx / max(1, total)), f"Frame {idx}/{total}")
    cap.release()
    player_model.predictor = None  # reset tracker state for the next run

    players = pd.DataFrame(player_rows, columns=PLAYER_COLUMNS)
    puck = pd.DataFrame(puck_rows, columns=PUCK_COLUMNS)
    if calibration is not None:  # drop people on the bench / in the crowd
        players = players[on_rink(players["x"], players["y"])].reset_index(drop=True)
        puck = puck[on_rink(puck["x"], puck["y"], margin=0)].reset_index(drop=True)
    players = consolidate_labels(players)
    meta = dict(fps=fps, frame_stride=stride, calibrated=calibration is not None,
                units="ft" if calibration is not None else "px", width=info["width"], height=info["height"])
    if calibration is None:
        players, puck = pixels_to_feet(players, puck)
    return players, puck, meta


def consolidate_labels(players: pd.DataFrame) -> pd.DataFrame:
    """A track keeps one label: the confidence-weighted majority over its life."""
    if players.empty:
        return players
    votes = players.groupby(["track_id", "label"])["conf"].sum().reset_index()
    best = votes.sort_values("conf").groupby("track_id").tail(1).set_index("track_id")["label"]
    players = players.copy()
    players["label"] = players["track_id"].map(best)
    return players


def pixels_to_feet(players: pd.DataFrame, puck: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """No calibration? Use box height as a ruler: ft-per-pixel = 5.5 ft / box height.

    Each track gets its own (median) scale, so near and far players are both
    roughly right. Positions are relative, not true rink coordinates.
    """
    if players.empty:
        return players, puck
    players = players.copy()
    scale = SKATER_HEIGHT_FT / players.groupby("track_id")["box_h"].transform("median")
    players["x"] = players["x"] * scale
    players["y"] = players["y"] * scale
    if not puck.empty:
        puck = puck.copy()
        global_scale = SKATER_HEIGHT_FT / players["box_h"].median()
        puck["x"] = puck["x"] * global_scale
        puck["y"] = puck["y"] * global_scale
    return players, puck
