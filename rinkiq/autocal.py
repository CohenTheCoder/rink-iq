"""Automatic rink calibration: no clicking needed.

The rink and dots models already find painted markings. Each one is a
*candidate* for a few known rink positions:

| Seen in the frame | Could be (rink feet) |
|---|---|
| faceoff dot inside a faceoff circle | one of the 4 end-zone dots |
| faceoff dot without a circle | one of the 4 neutral-zone dots, or center ice |
| the center-ice circle | center ice (100, 42.5) |
| a goal frame | the middle of the left or right goal line |

We try every assignment that respects simple facts about a broadcast view
(things further left in the picture are further left on the rink, higher up
is further away, and a painted center line splits left from right), fit a
homography to each, and keep the one whose points line up best. Searching a
few dozen frames finds a view with enough markings.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from . import models
from .rink import (CENTER_X, CENTER_Y, DOT_OFFSET_Y, GOAL_LINE_LEFT, GOAL_LINE_RIGHT, NEUTRAL_DOT_X, Calibration,
                   pixels_to_rink)

NEUTRAL = [(x, CENTER_Y + d) for x in NEUTRAL_DOT_X for d in (DOT_OFFSET_Y, -DOT_OFFSET_Y)]
END_ZONE = [(31.0, CENTER_Y + DOT_OFFSET_Y), (31.0, CENTER_Y - DOT_OFFSET_Y),
            (169.0, CENTER_Y + DOT_OFFSET_Y), (169.0, CENTER_Y - DOT_OFFSET_Y)]
CENTER = (CENTER_X, CENTER_Y)
GOALS = [(GOAL_LINE_LEFT, CENTER_Y), (GOAL_LINE_RIGHT, CENTER_Y)]


@dataclass
class Mark:
    kind: str                    # "dot" | "ez_dot" | "center_circle" | "goal"
    px: tuple[float, float]
    conf: float
    options: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class Evidence:
    """Markings not used in the fit, plus the picture itself, to judge a candidate homography."""
    boxes: list                                    # raw rink-model detections
    lines: list[tuple[str, tuple[float, float]]]   # (class, box centre) of painted lines
    circles: list[tuple[float, float]]
    hsv: np.ndarray


@dataclass
class AutoCal:
    calibration: Calibration
    frame: int
    error_ft: float
    marks: list[Mark]
    assignment: list[tuple[float, float]]
    line_error_ft: float = 0.0
    ice_share: float = 0.0
    score: float = 0.0
    lines: list = field(default_factory=list)   # (kind, rink x) of painted lines used in the fit

    @property
    def n_points(self) -> int:
        return len(self.marks) + len(self.lines)


def find_marks(frame: np.ndarray) -> tuple[list[Mark], float | None, Evidence]:
    """Markings in one frame, the x position of the painted center line if seen, and extra evidence."""
    dots = models.load("dots")(frame, conf=0.3, verbose=False)[0]
    rink = models.load("rink")(frame, conf=0.35, verbose=False)[0]
    boxes = [(rink.names[int(c)], b, float(p)) for b, c, p in
             zip(rink.boxes.xyxy.cpu().numpy(), rink.boxes.cls.cpu().numpy(), rink.boxes.conf.cpu().numpy())]
    circles = [b for n, b, _ in boxes if n == "Circle"]
    marks: list[Mark] = []
    for b, p in zip(dots.boxes.xyxy.cpu().numpy(), dots.boxes.conf.cpu().numpy()):
        x, y = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        in_circle = any(c[0] <= x <= c[2] and c[1] <= y <= c[3] for c in circles)
        marks.append(Mark("ez_dot" if in_circle else "dot", (float(x), float(y)), float(p),
                          END_ZONE if in_circle else NEUTRAL + [CENTER]))
    cc = [(b, p) for n, b, p in boxes if n == "Center_Circle"]
    if cc:
        b, p = max(cc, key=lambda t: t[1])
        marks.append(Mark("center_circle", (float((b[0] + b[2]) / 2), float((b[1] + b[3]) / 2)), p, [CENTER]))
    for n, b, p in boxes:
        if n == "Goal_Posts" and p >= 0.45:
            marks.append(Mark("goal", (float((b[0] + b[2]) / 2), float(b[3])), p, GOALS))
    lines = [(b[0] + b[2]) / 2 for n, b, _ in boxes if n == "Center_Line"]
    ev = Evidence(boxes=boxes, lines=[(n, ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)) for n, b, _ in boxes
                         if n in ("Blue_Line", "Center_Line", "Goal_Back_Line")],
                  circles=[((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in circles],
                  hsv=cv2.cvtColor(cv2.resize(frame, None, fx=0.25, fy=0.25), cv2.COLOR_BGR2HSV))
    return marks, (float(np.median(lines)) if lines else None), ev


LINE_X = {"Blue_Line": (75.0, 125.0), "Center_Line": (CENTER_X,), "Goal_Back_Line": (GOAL_LINE_LEFT, GOAL_LINE_RIGHT)}


@dataclass
class PaintedLine:
    kind: str                         # "Blue_Line" | "Center_Line" | "Goal_Back_Line"
    line: np.ndarray                  # image line (a, b, c) with a·x + b·y + c = 0, (a, b) unit length
    mid: tuple[float, float]
    length: float
    options: tuple[float, ...] = ()   # rink x positions it could be


def painted_lines(frame: np.ndarray, boxes) -> list[PaintedLine]:
    """Fit a straight line to the blue/red paint inside each detected line box."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    out = []
    for name, b, p in boxes:
        if name not in LINE_X or p < 0.4:
            continue
        x1, y1, x2, y2 = (int(v) for v in b)
        roi = hsv[max(0, y1):y2, max(0, x1):x2]
        if roi.size == 0:
            continue
        h, s_, v = roi[..., 0], roi[..., 1], roi[..., 2]
        # Broadcast paint under the ice is faint: saturation ~40 vs ~10 for bare ice. So the test is
        # "noticeably more colourful than ice, in the right hue range", not "strongly coloured".
        ice_s = float(np.median(s_[v > 150])) if (v > 150).any() else 15.0
        if name == "Blue_Line":
            paint = (h > 98) & (h < 135) & (s_ > ice_s + 18) & (v > 60)
        else:  # red center / goal line (often faded, checkered, or magenta-ish on camera)
            paint = ((h < 12) | (h > 150)) & (s_ > ice_s + 15) & (v > 70)
        paint = paint.astype(np.uint8)
        # The longest straight stroke is the painted line. Blobs such as logos, jerseys or
        # burned-in labels don't make long thin segments.
        segs = cv2.HoughLinesP(paint * 255, 1, np.pi / 180, threshold=40, minLineLength=int(0.15 * frame.shape[0]),
                               maxLineGap=12)
        if segs is None:
            continue
        sx1, sy1, sx2, sy2 = max(segs[:, 0], key=lambda q: np.hypot(q[2] - q[0], q[3] - q[1]))
        d = np.array([sx2 - sx1, sy2 - sy1], float)
        d /= np.linalg.norm(d) + 1e-9
        n_ = np.array([d[1], -d[0]])
        ys, xs = np.nonzero(paint)
        pts = np.stack([xs, ys], 1).astype(np.float32)
        near = np.abs((pts - [sx1, sy1]) @ n_) < 6           # paint pixels on that stroke only
        if near.sum() < 30:
            continue
        pts = pts[near] + [max(0, x1), max(0, y1)]
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        a, b_, c = vy, -vx, vx * y0 - vy * x0  # normal form through (x0, y0) along (vx, vy)
        proj = (pts - [x0, y0]) @ np.array([vx, vy])
        length = float(np.ptp(proj))
        if length < 0.15 * frame.shape[0]:
            continue
        out.append(PaintedLine(name, np.array([a, b_, c], float), (float(x0), float(y0)), length, LINE_X[name]))
    # one line per painted rink line: merge duplicates of the same kind that lie on top of each other
    merged: list[PaintedLine] = []
    for L in sorted(out, key=lambda l: -l.length):
        if all(not (m.kind == L.kind and abs(L.line[:2] @ np.array(m.mid) + L.line[2]) < 0.03 * frame.shape[1])
               for m in merged):
            merged.append(L)
    return merged
_RINK_GRID = np.float64([(x, y) for x in np.arange(8, 193, 8) for y in np.arange(6, 80, 6)])


def judge(H: np.ndarray, ev: Evidence, w: int, h: int) -> tuple[float, float]:
    """(line error in ft, share of in-picture rink points that land on ice-coloured pixels)."""
    errs = []
    for name, (cx, cy) in ev.lines:
        x, _ = pixels_to_rink(np.float64([[cx, cy]]), H)[0]
        errs.append(min(abs(x - lx) for lx in LINE_X[name]))
    for cx, cy in ev.circles:
        p = pixels_to_rink(np.float64([[cx, cy]]), H)[0]
        errs.append(min(np.hypot(*(p - np.array(d))) for d in END_ZONE))
    line_err = float(np.mean(np.minimum(errs, 30))) if errs else 0.0
    px = pixels_to_rink(_RINK_GRID, np.linalg.inv(H))
    inside = (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
    if inside.sum() < 10:
        return line_err, 0.0
    q = (px[inside] / 4).astype(int)
    hh, ww = ev.hsv.shape[:2]
    q[:, 0], q[:, 1] = np.clip(q[:, 0], 0, ww - 1), np.clip(q[:, 1], 0, hh - 1)
    s, v = ev.hsv[q[:, 1], q[:, 0], 1], ev.hsv[q[:, 1], q[:, 0], 2]
    ice = float(np.mean((v > 140) & (s < 70)))
    return line_err, ice


def _normaliser(pts: np.ndarray) -> np.ndarray:
    c = pts.mean(0)
    d = np.mean(np.linalg.norm(pts - c, axis=1)) or 1.0
    k = np.sqrt(2) / d
    return np.array([[k, 0, -k * c[0]], [0, k, -k * c[1]], [0, 0, 1]])


def fit_points_lines(img_pts, rink_pts, img_lines, rink_xs) -> np.ndarray | None:
    """Homography image→rink from point pairs AND line pairs (image line ↔ rink line x = const).

    A point pair gives 2 equations (q × Hp = 0). A line pair gives 2 more: lines map the other way,
    l_img ∝ Hᵀ·L_rink. Careful: rink lines are all parallel (x = const), so two of them share a vanishing
    point and add less than it seems. Use ≥ 3 non-collinear points + ≥ 1 line, or ≥ 4 points.
    Coordinates are normalised first (Hartley) so the equations are well conditioned.
    """
    ip = np.asarray(img_pts, float).reshape(-1, 2)
    rp = np.asarray(rink_pts, float).reshape(-1, 2)
    # normalisers from everything we know about each plane
    i_ref = ip if len(ip) >= 2 else np.vstack([ip, np.array([[0, 0], [1280, 720]])])
    r_ref = rp if len(rp) >= 2 else np.vstack([rp, np.array([[0, 0], [200, 85]])])
    Ti, Tr = _normaliser(i_ref), _normaliser(r_ref)
    rows = []
    for p, q in zip(ip, rp):
        p = Ti @ np.array([p[0], p[1], 1.0])
        q = Tr @ np.array([q[0], q[1], 1.0])
        x, y, w = p
        u, v, s = q
        rows.append([0, 0, 0, -s * x, -s * y, -s * w, v * x, v * y, v * w])
        rows.append([s * x, s * y, s * w, 0, 0, 0, -u * x, -u * y, -u * w])
    for l_img, X in zip(img_lines, rink_xs):
        li = np.linalg.inv(Ti).T @ np.asarray(l_img, float)      # image line in normalised coords
        L = np.linalg.inv(Tr).T @ np.array([1.0, 0.0, -X])        # rink line x = X, normalised
        # li ∝ Hᵀ L  →  li × (Hᵀ L) = 0 ; write Hᵀ L as a linear function of h (row-major H)
        A = np.zeros((3, 9))
        for col in range(3):           # (Hᵀ L)_col = Σ_r H[r, col] · L[r]
            for r in range(3):
                A[col, r * 3 + col] = L[r]
        cx = np.array([[0, -li[2], li[1]], [li[2], 0, -li[0]], [-li[1], li[0], 0]])
        rows.extend((cx @ A)[:2])
    if len(rows) < 8:
        return None
    _, _, vt = np.linalg.svd(np.asarray(rows))
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Tr) @ Hn @ Ti
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else None


def _consistent(marks: list[Mark], assign, center_x: float | None, w: int, h: int) -> bool:
    for i, j in itertools.combinations(range(len(marks)), 2):
        (xi, yi), (xj, yj) = marks[i].px, marks[j].px
        (ri, si), (rj, sj) = assign[i], assign[j]
        if (ri, si) == (rj, sj):
            return False
        # left/right order is kept (a broadcast camera looks across the rink from the side)
        if xi < xj - 0.06 * w and ri > rj:
            return False
        if xj < xi - 0.06 * w and rj > ri:
            return False
        # same column on the rink: the higher one in the picture is the far one
        if ri == rj and abs(yi - yj) > 0.04 * h and (yi < yj) != (si > sj):
            return False
    if center_x is not None:
        for m, (r, _) in zip(marks, assign):
            if r != CENTER_X and abs(m.px[0] - center_x) > 0.04 * w and (m.px[0] < center_x) != (r < CENTER_X):
                return False
    return True


def not_mirrored(H: np.ndarray, w: int, h: int) -> bool:
    """True if H keeps handedness, as every real camera above the ice does.

    The rink is symmetric, so a left↔right flipped assignment of markings can fit almost as well as
    the right one. But a camera can't film a mirror image: going right then down in the picture must
    turn the same way on the rink as it does on screen. With image y pointing down and rink y pointing
    up, that means the local Jacobian of pixel → rink has a negative determinant.
    """
    p0 = np.float64([[w / 2, h * 0.6], [w / 2 + 5, h * 0.6], [w / 2, h * 0.6 + 5]])
    a, b, c = pixels_to_rink(p0, H)
    J = np.column_stack([(b - a) / 5, (c - a) / 5])
    return bool(np.linalg.det(J) < 0)


def _line_x_at(L: PaintedLine, y: float) -> float:
    a, b, c = L.line
    return float(-(b * y + c) / a) if abs(a) > 1e-6 else float("inf")


def _line_residual(H: np.ndarray, L: PaintedLine, X: float, h: int) -> float:
    """Mean |rink x − X| (ft) along the painted line's visible stretch."""
    a, b, c = L.line
    d = np.array([b, -a])  # direction along the line
    t = np.linspace(-L.length / 2, L.length / 2, 7)
    pts = np.array(L.mid) + t[:, None] * d
    return float(np.mean(np.abs(pixels_to_rink(pts, H)[:, 0] - X)))


def solve_frame(frame: np.ndarray, frame_index: int = 0, max_error_ft: float = 3.0) -> AutoCal | None:
    """Best rink fit for one frame, or None.

    Points (dots, center circle, goals) and painted lines (blue, center, goal lines) are matched to the
    rink in every consistent way. Each candidate is fitted, then judged on how well it explains
    *everything* seen, and on whether the projected rink lands on white ice.
    """
    marks, center_x, ev = find_marks(frame)
    lines = painted_lines(frame, ev.boxes)
    # Arena logos at center ice fool the "center circle" detector, so a circle only counts as center ice
    # when the painted red center line actually runs through it.
    centre_lines = [L for L in lines if L.kind == "Center_Line"]
    marks = [m for m in marks if m.kind != "center_circle"
             or any(abs(_line_x_at(L, m.px[1]) - m.px[0]) < 0.03 * frame.shape[1] for L in centre_lines)]
    marks = sorted(marks, key=lambda m: -m.conf)[:6]
    # Every painted line on a rink runs the same way (x = constant), so lines pin down left/right but
    # say little about near/far. That takes ≥ 3 points that aren't in a row (checked per candidate).
    if len(marks) < 3 or len(marks) + len(lines) < 4:
        return None
    h, w = frame.shape[:2]
    y_ref = h * 0.6
    best = None
    for assign in itertools.product(*[m.options for m in marks]):
        if not _consistent(marks, assign, center_x, w, h):
            continue
        for lx in itertools.product(*[L.options for L in lines]):
            if len(set(lx)) < len(lx):
                continue
            # lines keep their left-to-right order, and marks sit on the correct side of each line
            xs_img = [_line_x_at(L, y_ref) for L in lines]
            if any((xs_img[i] < xs_img[j]) != (lx[i] < lx[j]) for i, j in itertools.combinations(range(len(lines)), 2)):
                continue
            ok = True
            for m, (rx, _) in zip(marks, assign):
                for L, X in zip(lines, lx):
                    lxi = _line_x_at(L, m.px[1])
                    if abs(m.px[0] - lxi) > 0.03 * w and (m.px[0] < lxi) != (rx < X):
                        ok = False
            if not ok:
                continue
            dst = np.float64(assign)
            if cv2.contourArea(cv2.convexHull(dst.astype(np.float32))) < 300:  # points (nearly) in a row
                continue
            H = fit_points_lines([m.px for m in marks], dst, [L.line for L in lines], lx)
            if H is None or not np.all(np.isfinite(H)):
                continue
            src = np.float64([m.px for m in marks])
            res = list(np.linalg.norm(pixels_to_rink(src, H) - dst, axis=1))
            res += [_line_residual(H, L, X, h) for L, X in zip(lines, lx)]
            err = float(np.mean(res))
            if err > max_error_ft:
                continue
            if not not_mirrored(H, w, h):
                continue
            b_, m_ = pixels_to_rink(np.float64([[w / 2, h * 0.95], [w / 2, h * 0.5]]), H)
            if not (np.all(np.isfinite([b_, m_])) and b_[1] < m_[1] and -30 < m_[0] < 230 and -20 < m_[1] < 105):
                continue
            line_err, ice = judge(H, ev, w, h)
            score = err + 0.5 * line_err + 20 * (1 - ice)
            if best is None or score < best[0]:
                best = (score, err, assign, H, line_err, ice, lx)
    if best is None or best[5] < 0.6 or best[4] > 8:
        return None
    score, err, assign, H, line_err, ice, lx = best
    if 2 * (len(marks) + len(lx)) < 10:
        # Exactly enough constraints always fit perfectly, so the fit can't vouch for itself. Demand a
        # spare one (or strong outside evidence from the painted-line boxes) before trusting it.
        if not (line_err < 1.5 and ice > 0.9 and len(ev.lines) >= 2):
            return None
    cal = Calibration(pixel_points=[m.px for m in marks], rink_points=list(assign), H=H)
    cal.reference_frame = frame_index
    res = AutoCal(cal, frame_index, err, marks, list(assign), line_err, ice, score)
    res.lines = [(L.kind, X) for L, X in zip(lines, lx)]
    return res


def auto_calibrate(path: str, start_s: float = 0.0, max_s: float = 90.0, step_s: float = 0.5,
                   progress: Callable[[float, str], None] | None = None) -> AutoCal | None:
    """Search frames for the view with the most markings that fit an NHL rink well."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first, last = int(start_s * fps), min(n - 1, int((start_s + max_s) * fps))
    step = max(1, int(step_s * fps))
    best: AutoCal | None = None
    idxs = list(range(first, last + 1, step))
    for k, i in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        sol = solve_frame(frame, i)
        if sol is not None and (best is None or (sol.n_points, -sol.score) > (best.n_points, -best.score)):
            best = sol
        if progress:
            progress((k + 1) / len(idxs), f"Looking for rink markings… {i / fps:.0f}s"
                     + (f" · best so far: {best.n_points} marks, {best.error_ft:.1f} ft" if best else ""))
        # The earliest clearly-good view wins: analysis can then start calibrated, not wait for it.
        if sol is not None and sol.n_points >= 5 and sol.error_ft < 0.6 and sol.line_error_ft < 3 and sol.ice_share > 0.85:
            best = sol
            break
    cap.release()
    return best
