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

Long videos: cuts start new shots (tracker reset, rink re-located from
keyframes), and `job_dir` makes the run resumable. See `run_video`.

Output: two tidy DataFrames — `players` and `puck` — which are the contract
for everything in `metrics.py`. You can also feed in tracking data from any
other source (or `simulate.py`) as long as it has the same columns.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from . import identity, models
from .camera import CameraMotion, static_overlay_mask
from .pose import facing_from_keypoints
from .rink import Calibration, direction_to_rink, on_rink, pixels_to_rink, snap_correction

SKATER_HEIGHT_FT = 5.5
PLAYER_COLUMNS = ["frame", "t", "track_id", "label", "conf", "x", "y", "px", "py", "box_h",
                  "x1", "y1", "x2", "y2", "face_x", "face_y", "face_conf"]
PUCK_COLUMNS = ["frame", "t", "x", "y", "px", "py", "conf"]  # px/py = pixel position in that frame
# Per analysed frame: which camera shot it belongs to and the 3×3 matrix mapping its pixels to the
# shot's reference pixels (used to draw motion trails that stay glued to the ice while the camera pans).
FRAME_COLUMNS = ["frame", "t", "shot"] + [f"m{i}{j}" for i in range(3) for j in range(3)]


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
    anchor_every: int | None = None  # re-check faceoff dots every Nth analysed frame; None = every ~0.13 s
    read_numbers: bool = True        # read jersey numbers (→ names with a roster)
    number_every: int = 8            # … on every Nth analysed frame
    start_s: float = 0.0             # analyse only part of the video
    end_s: float | None = None
    keyframe_every: int = 45         # remember a calibrated view every N analysed frames (for re-location)
    reloc_every: int = 15            # while a shot is unplaceable, retry re-location every N frames
    chunk_s: float = 60.0            # with a job_dir: flush results to disk every N seconds of video


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


class _Keyframes:
    """Calibrated views we've seen, so we can find our place again after a camera cut."""

    def __init__(self, cap: int = 300):
        self.pts: list[np.ndarray] = []
        self.des: list[np.ndarray] = []
        self.H: list[np.ndarray] = []
        self.cap = cap
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def add(self, kp, des, H: np.ndarray) -> None:
        if des is None or len(kp) < 50:
            return
        self.pts.append(np.float32([k.pt for k in kp]))
        self.des.append(des)
        self.H.append(H.copy())
        if len(self.H) > self.cap:  # keep the bank bounded: drop every other old keyframe
            for lst in (self.pts, self.des, self.H):
                del lst[0:len(lst) // 2:2]

    def locate(self, kp, des, min_inliers: int = 60) -> np.ndarray | None:
        """Pixel → rink homography for a new view, by matching it to the best keyframe."""
        if des is None or not self.H:
            return None
        pts = np.float32([k.pt for k in kp])
        best, best_n = None, 0
        for kpts, kdes, kH in zip(self.pts, self.des, self.H):
            m = self.matcher.match(des, kdes)
            if len(m) < min_inliers:
                continue
            src = pts[[x.queryIdx for x in m]].reshape(-1, 1, 2)
            dst = kpts[[x.trainIdx for x in m]].reshape(-1, 1, 2)
            M, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
            n = int(inl.sum()) if inl is not None else 0
            if M is not None and n > best_n:
                best, best_n = kH @ M, n
        return best if best_n >= min_inliers else None

    def save(self, path) -> None:
        np.savez_compressed(path, pts=np.array(self.pts, dtype=object), des=np.array(self.des, dtype=object),
                            H=np.array(self.H))

    def load(self, path) -> None:
        z = np.load(path, allow_pickle=True)
        self.pts, self.des, self.H = list(z["pts"]), list(z["des"]), list(z["H"])


def _plausible(H: np.ndarray, shape) -> bool:
    """Does the lower half of the picture (where the ice is) land on or near the rink?"""
    h, w = shape[:2]
    pts = pixels_to_rink(np.float32([[w * 0.25, h * 0.7], [w * 0.5, h * 0.6], [w * 0.75, h * 0.7]]), H)
    return bool(np.all(np.isfinite(pts)) and np.all((pts[:, 0] > -40) & (pts[:, 0] < 240)
                                                    & (pts[:, 1] > -40) & (pts[:, 1] < 125)))


def _blend_h(H_from: np.ndarray, H_to: np.ndarray, a: float) -> np.ndarray:
    """Move part of the way from one pixel→rink homography to another (normalised entries)."""
    A, B = H_from / H_from[2, 2], H_to / H_to[2, 2]
    return (1 - a) * A + a * B


def _histogram(frame: np.ndarray) -> np.ndarray:
    small = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([small], [0, 1], None, [30, 32], [0, 180, 0, 256])
    return cv2.normalize(h, h)


def run_video(path: str, calibration: Calibration | None = None, reference_frame: int = 0,
              config: RunConfig | None = None,
              progress: Callable[[float, str], None] | None = None,
              on_frame: Callable[[np.ndarray, list[dict]], None] | None = None,
              job_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Stream through a video of any length. Returns (players, puck, meta).

    * **Camera cuts** start a new *shot*: the tracker resets (IDs stay unique)
      and, when calibrated, the new view is re-located on the rink by matching it
      against keyframes of views we already calibrated. Shots that can't be
      placed (close-ups, replays from odd angles) are skipped and listed in
      meta["shots"].
    * **job_dir** makes long runs resumable: results are flushed to CSV chunks
      every `chunk_s` seconds of video; rerunning with the same job_dir continues.
    * **on_frame(frame, rows)** is called for every analysed frame, for live previews.
    """
    cfg = config or RunConfig()
    info = video_info(path)
    fps = info["fps"]
    stride = cfg.frame_stride or max(1, round(fps / 30))
    anchor_every = cfg.anchor_every or max(1, round(0.13 * fps / stride))
    ease = 0.8  # how far each dot snap moves toward the real dots (with snaps every ~0.13 s)
    player_model = models.load("player")
    puck_model = models.load("puck")
    number_model = models.load("numbers") if cfg.read_numbers else None
    names = player_model.names
    job = Path(job_dir) if job_dir else None

    start = int(cfg.start_s * fps) if cfg.start_s else reference_frame
    stop = info["n_frames"] if cfg.end_s is None else min(info["n_frames"], int(cfg.end_s * fps))
    if cfg.max_frames:
        stop = min(stop, start + cfg.max_frames)

    player_rows, puck_rows, frame_rows, readings, shots_log = [], [], [], [], []
    keyframes = _Keyframes()
    id_offset, max_gid = 0, 0
    if job is not None:
        job.mkdir(parents=True, exist_ok=True)
        state_file = job / "state.json"
        if state_file.exists():  # resume
            st = json.loads(state_file.read_text())
            start, id_offset, max_gid = st["next_frame"], st["max_gid"] + 1, st["max_gid"]
            shots_log = st["shots"]
            if (job / "keyframes.npz").exists():
                keyframes.load(job / "keyframes.npz")

    overlay, fixed_camera = static_overlay_mask(path)
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, reference_frame)
    ok, ref = cap.read()
    if not ok:
        raise ValueError("Could not read the video.")
    probe = CameraMotion(ref, ignore=overlay)
    if calibration is not None and calibration.H is not None:
        keyframes.add(probe.ref_kp, probe.ref_des, calibration.H)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    camera, shot_H, shot = None, None, None
    prev_hist, last_snap, analysed = None, -999, 0
    chunk_frames, last_flush = int(cfg.chunk_s * fps), start

    def new_shot(frame, fno):
        nonlocal camera, shot_H, shot, id_offset
        player_model.predictor = None              # fresh tracker for the new camera shot
        id_offset = max_gid + 1
        camera = None if fixed_camera else CameraMotion(frame, ignore=overlay)
        if calibration is None:
            shot_H, status = None, "pixels"
        elif fno == reference_frame and not shots_log:
            shot_H, status = calibration.H.copy(), "calibrated"
        else:
            kp, des = (camera or probe)._features(frame, None)
            shot_H = keyframes.locate(kp, des)
            status = "re-located" if shot_H is not None else "skipped (no rink match)"
        shot = dict(start_frame=fno, end_frame=fno, status=status,
                    H=None if shot_H is None else np.asarray(shot_H).round(8).tolist())
        shots_log.append(shot)

    def flush(final=False):
        nonlocal player_rows, puck_rows, frame_rows, readings, last_flush
        if job is None:
            return
        n = len(list(job.glob("players_*.csv")))
        pd.DataFrame(player_rows, columns=PLAYER_COLUMNS).to_csv(job / f"players_{n:05d}.csv", index=False)
        pd.DataFrame(puck_rows, columns=PUCK_COLUMNS).to_csv(job / f"puck_{n:05d}.csv", index=False)
        pd.DataFrame(frame_rows, columns=FRAME_COLUMNS).to_csv(job / f"frames_{n:05d}.csv", index=False)
        pd.DataFrame(readings, columns=["track_id", "number", "conf"]).to_csv(job / f"numbers_{n:05d}.csv", index=False)
        keyframes.save(job / "keyframes.npz")
        (job / "state.json").write_text(json.dumps(dict(next_frame=int(fno + 1), max_gid=int(max_gid),
                                                         shots=shots_log, done=final)))
        player_rows, puck_rows, frame_rows, readings = [], [], [], []
        last_flush = fno

    fno = start - 1
    while True:
        fno += 1
        if fno >= stop:
            break
        if (fno - start) % stride:
            if not cap.grab():                     # skip without decoding into a numpy array
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        t = (fno - reference_frame) / fps

        hist = _histogram(frame)
        cut = prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < 0.6
        prev_hist = hist
        if shot is None or cut:
            new_shot(frame, fno)
        shot["end_frame"] = fno
        if shot["status"].startswith("skipped"):
            # keep trying: a camera often zooms out of a close-up back to the wide view
            if analysed % cfg.reloc_every == 0:
                kp, des = probe._features(frame, None)
                if keyframes.locate(kp, des) is not None:
                    new_shot(frame, fno)  # found our place again: start a fresh, re-located shot
            if shot["status"].startswith("skipped"):
                analysed += 1
                if progress:
                    progress(min(1.0, (fno - start) / max(1, stop - start)), f"{t:,.0f}s · skipping close-up/replay")
                continue

        res = player_model.track(frame, persist=True, tracker="bytetrack.yaml", conf=cfg.player_conf,
                                 imgsz=cfg.player_imgsz, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
        cls = res.boxes.cls.cpu().numpy().astype(int) if len(res.boxes) else np.zeros(0, int)
        confs = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.zeros(0)
        ids = (res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else np.full(len(boxes), -1))

        to_ref = camera.update(frame, boxes) if camera is not None else np.eye(3)
        H = shot_H @ to_ref if shot_H is not None else None
        if H is not None and camera is not None and analysed % anchor_every == 0:
            centres, inside = dot_observations(frame)
            C = snap_correction(pixels_to_rink(centres, H), inside) if len(centres) else None
            if C is not None:
                C = np.eye(3) + ease * (C - np.eye(3))  # ease in: no sudden jumps in position
                if _plausible(C @ H, frame.shape):
                    H = C @ H
                    to_ref = np.linalg.inv(shot_H) @ H
                    camera.prev_to_ref = to_ref  # later frames chain from the corrected view
                    if len(centres) >= 2:
                        last_snap = analysed  # two+ dots pin rotation and scale too: a trustworthy view
        if H is not None and camera is not None and analysed % cfg.reloc_every == cfg.reloc_every // 2:
            # Drift check without dots: does this view match a calibrated keyframe?
            found = keyframes.locate(camera.prev_kp, camera.prev_des, min_inliers=80)
            if found is not None:
                corners = np.float32([[0, 0], [frame.shape[1], 0], [frame.shape[1], frame.shape[0]], [0, frame.shape[0]]])
                gap = np.abs(pixels_to_rink(corners, found) - pixels_to_rink(corners, H)).max()
                if gap < 40 and _plausible(found, frame.shape):  # a plausible correction, not a wrong match
                    H = found if gap < 1 else _blend_h(H, found, 0.6)
                    to_ref = np.linalg.inv(shot_H) @ H
                    camera.prev_to_ref = to_ref
        if H is not None and camera is not None and analysed - last_snap == 0 and analysed % cfg.keyframe_every < anchor_every:
            keyframes.add(camera.prev_kp, camera.prev_des, H)
        frame_rows.append([fno, t, len(shots_log) - 1, *to_ref.ravel().tolist()])

        keep = np.array([names[c] != "puck" and i >= 0 for c, i in zip(cls, ids)], dtype=bool)
        poses = (_pose_on_crops(frame, boxes[keep]) if cfg.use_pose and analysed % cfg.pose_every == 0
                 else [None] * int(keep.sum()))
        feet = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], axis=1) if len(boxes) else np.zeros((0, 2))
        feet_ref = pixels_to_rink(feet, to_ref)  # stabilised pixels in the shot's reference frame
        feet_rink = pixels_to_rink(feet, H) if H is not None else None
        read_now = number_model is not None and analysed % cfg.number_every == 0

        live, pi = [], 0
        for j in range(len(boxes)):
            label = names[cls[j]]
            if label == "puck" or ids[j] < 0:
                continue
            gid = int(ids[j]) + id_offset
            max_gid = max(max_gid, gid)
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
            if read_now and label != "referee" and (y2 - y1) >= 35:
                got = identity.read_number(frame[int(max(0, y1)):int(y2), int(max(0, x1)):int(x2)], number_model)
                if got:
                    readings.append((gid, got[0], got[1]))
            player_rows.append([fno, t, gid, label, float(confs[j]), float(x), float(y),
                                float(feet[j][0]), float(feet[j][1]), float(y2 - y1),
                                float(x1), float(y1), float(x2), float(y2), fx, fy, fc])
            live.append(dict(track_id=gid, label=label, box=(x1, y1, x2, y2), foot=tuple(feet[j])))

        pr = puck_model(frame, conf=cfg.puck_conf, imgsz=cfg.puck_imgsz, verbose=False)[0]
        if len(pr.boxes):
            b = pr.boxes.xyxy[int(pr.boxes.conf.argmax())].cpu().numpy()
            c = float(pr.boxes.conf.max())
            p = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]])
            if H is not None:
                x, y = pixels_to_rink(p, H)[0]  # the puck sits on the ice, so its centre maps straight through H
            else:
                p_ref = pixels_to_rink(p, to_ref)[0]
                x, y = p_ref[0], -p_ref[1]
            puck_rows.append([fno, t, float(x), float(y), float(p[0, 0]), float(p[0, 1]), c])
            live.append(dict(track_id=-1, label="puck", box=tuple(b), foot=(p[0, 0], p[0, 1])))

        if on_frame is not None:
            on_frame(frame, live)
        analysed += 1
        if job is not None and fno - last_flush >= chunk_frames:
            flush()
        if progress:
            progress(min(1.0, (fno - start) / max(1, stop - start)), f"{t:,.1f}s / {(stop - reference_frame) / fps:,.1f}s")
    cap.release()
    player_model.predictor = None  # reset tracker state for the next run

    if job is not None:
        flush(final=True)
        load = lambda pat, cols: pd.concat([pd.read_csv(p) for p in sorted(job.glob(pat))] or  # noqa: E731
                                           [pd.DataFrame(columns=cols)], ignore_index=True)
        players = load("players_*.csv", PLAYER_COLUMNS)
        puck = load("puck_*.csv", PUCK_COLUMNS)
        frames = load("frames_*.csv", FRAME_COLUMNS)
        reads = load("numbers_*.csv", ["track_id", "number", "conf"])
        readings = [(int(a), str(b), float(c)) for a, b, c in reads.itertuples(index=False)]
    else:
        players = pd.DataFrame(player_rows, columns=PLAYER_COLUMNS)
        puck = pd.DataFrame(puck_rows, columns=PUCK_COLUMNS)
        frames = pd.DataFrame(frame_rows, columns=FRAME_COLUMNS)
    if calibration is not None:  # drop people on the bench / in the crowd
        players = players[on_rink(players["x"], players["y"])].reset_index(drop=True)
        puck = puck[on_rink(puck["x"], puck["y"], margin=0)].reset_index(drop=True)
    players = consolidate_labels(players)
    numbers = identity.vote_numbers(readings)
    players, merged = identity.merge_tracks(players, numbers)
    numbers = {merged.get(k, k): v for k, v in numbers.items()}
    meta = dict(fps=fps, frame_stride=stride, calibrated=calibration is not None,
                units="ft" if calibration is not None else "px", width=info["width"], height=info["height"],
                shots=shots_log, numbers={int(k): v for k, v in numbers.items()}, merged_tracks=merged,
                frames=frames, reference_frame=reference_frame)
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


def load_saved_run(folder: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load a run saved by scripts/export_sample_run.py (players.csv, puck.csv, frames.csv, meta.json)."""
    folder = Path(folder)
    meta = json.loads((folder / "meta.json").read_text())
    meta["numbers"] = {int(k): tuple(v) for k, v in meta.get("numbers", {}).items()}
    meta["merged_tracks"] = {int(k): v for k, v in meta.get("merged_tracks", {}).items()}
    meta["frames"] = pd.read_csv(folder / "frames.csv")
    return pd.read_csv(folder / "players.csv"), pd.read_csv(folder / "puck.csv"), meta
