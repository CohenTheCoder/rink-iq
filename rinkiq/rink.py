"""Step 3 — Rink geometry and camera-to-rink calibration.

Everything downstream is measured in *rink feet*, using an NHL sheet:

    x: 0 → 200 ft  (left end boards → right end boards)
    y: 0 → 85 ft   (near boards, closest to the camera → far boards)

A homography (3x3 matrix) maps a pixel on the ice in a video frame to a point
on that top-down rink. Four known landmarks are enough to solve it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

RINK_LENGTH = 200.0
RINK_WIDTH = 85.0
CENTER_X = 100.0
CENTER_Y = 42.5
GOAL_LINE_LEFT = 11.0
GOAL_LINE_RIGHT = 189.0
BLUE_LINE_LEFT = 75.0
BLUE_LINE_RIGHT = 125.0
CORNER_RADIUS = 28.0
CIRCLE_RADIUS = 15.0
DOT_OFFSET_Y = 22.0  # faceoff dots sit 22 ft either side of the long axis

# Named points a user (or the auto-suggester) can match to pixels.
LANDMARKS: dict[str, tuple[float, float]] = {
    "Center ice dot": (CENTER_X, CENTER_Y),
    "Neutral dot · left · far": (70.0, CENTER_Y + DOT_OFFSET_Y),
    "Neutral dot · left · near": (70.0, CENTER_Y - DOT_OFFSET_Y),
    "Neutral dot · right · far": (130.0, CENTER_Y + DOT_OFFSET_Y),
    "Neutral dot · right · near": (130.0, CENTER_Y - DOT_OFFSET_Y),
    "End-zone dot · left · far": (31.0, CENTER_Y + DOT_OFFSET_Y),
    "End-zone dot · left · near": (31.0, CENTER_Y - DOT_OFFSET_Y),
    "End-zone dot · right · far": (169.0, CENTER_Y + DOT_OFFSET_Y),
    "End-zone dot · right · near": (169.0, CENTER_Y - DOT_OFFSET_Y),
    "Center line × far boards": (CENTER_X, RINK_WIDTH),
    "Center line × near boards": (CENTER_X, 0.0),
    "Left blue line × far boards": (BLUE_LINE_LEFT, RINK_WIDTH),
    "Left blue line × near boards": (BLUE_LINE_LEFT, 0.0),
    "Right blue line × far boards": (BLUE_LINE_RIGHT, RINK_WIDTH),
    "Right blue line × near boards": (BLUE_LINE_RIGHT, 0.0),
}


@dataclass
class Calibration:
    """Pixel ↔ rink mapping for one reference frame."""

    pixel_points: list[tuple[float, float]] = field(default_factory=list)
    rink_points: list[tuple[float, float]] = field(default_factory=list)
    H: np.ndarray | None = None  # pixel -> rink

    @classmethod
    def from_pairs(cls, pairs: list[tuple[tuple[float, float], str]]) -> "Calibration":
        """pairs = [((px, py), landmark_name), ...] — needs 4+ non-collinear points."""
        px = [p for p, _ in pairs]
        rk = [LANDMARKS[name] for _, name in pairs]
        cal = cls(pixel_points=px, rink_points=rk)
        cal.solve()
        return cal

    def solve(self) -> None:
        if len(self.pixel_points) < 4:
            raise ValueError("Need at least 4 landmark points to calibrate.")
        src = np.asarray(self.pixel_points, dtype=np.float64)
        dst = np.asarray(self.rink_points, dtype=np.float64)
        method = cv2.RANSAC if len(src) > 4 else 0
        H, _ = cv2.findHomography(src, dst, method, 3.0)
        if H is None:
            raise ValueError("Points are degenerate (collinear?) — pick landmarks that form a quadrilateral.")
        self.H = H

    def reprojection_error_ft(self) -> float:
        """Mean distance (ft) between where landmarks are and where the fit puts them."""
        if self.H is None:
            return float("nan")
        proj = pixels_to_rink(np.asarray(self.pixel_points), self.H)
        return float(np.mean(np.linalg.norm(proj - np.asarray(self.rink_points), axis=1)))


def pixels_to_rink(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    if len(pts) == 0:
        return np.zeros((0, 2))
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def rink_to_pixels(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    return pixels_to_rink(points, np.linalg.inv(H))


def direction_to_rink(foot_px: tuple[float, float], direction_px: tuple[float, float], H: np.ndarray,
                      step: float = 10.0) -> np.ndarray:
    """Push a unit pixel-direction through the homography at one spot (finite difference)."""
    p0 = np.array(foot_px, dtype=np.float64)
    d = np.array(direction_px, dtype=np.float64)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return np.zeros(2)
    p1 = p0 + d / n * step
    a, b = pixels_to_rink(np.stack([p0, p1]), H)
    v = b - a
    nv = np.linalg.norm(v)
    return v / nv if nv > 1e-9 else np.zeros(2)


def suggest_landmarks(dots_px: list[tuple[float, float]], center_line_x: float | None,
                      blue_line_xs: list[float], frame_h: int) -> list[str | None]:
    """Best-guess landmark name for each detected faceoff dot (user confirms in the app).

    Rules of thumb for a broadcast camera sitting at center ice:
      * dots between a blue line and the center line are neutral-zone dots
      * dots on the far side of a blue line are end-zone dots
      * left/right comes from which side of the center line the dot is on
      * far/near comes from how high the dot is in the frame
    """
    if not dots_px:
        return []
    names: list[str | None] = []
    ys = np.array([p[1] for p in dots_px])
    for (x, y) in dots_px:
        if center_line_x is not None:
            side = "left" if x < center_line_x else "right"
        else:
            side = "left" if x < np.median([p[0] for p in dots_px]) else "right"
        zone = "Neutral"
        for bx in blue_line_xs:
            if center_line_x is not None and ((side == "left" and x < bx < center_line_x)
                                               or (side == "right" and center_line_x < bx < x)):
                zone = "End-zone"
        if len(dots_px) > 1 and ys.max() - ys.min() > frame_h * 0.15:
            depth = "far" if y < (ys.max() + ys.min()) / 2 else "near"
        else:
            depth = "far" if y < frame_h / 2 else "near"
        names.append(f"{zone} dot · {side} · {depth}")
    # Never suggest the same landmark twice — keep the more confident (first) one.
    seen: set[str] = set()
    for i, n in enumerate(names):
        if n in seen:
            names[i] = None
        elif n is not None:
            seen.add(n)
    return names


def zone_of(x: float, attacks_right: bool) -> str:
    """Offensive / neutral / defensive zone for a team, given which way it attacks."""
    if BLUE_LINE_LEFT <= x <= BLUE_LINE_RIGHT:
        return "NZ"
    in_right_end = x > BLUE_LINE_RIGHT
    return "OZ" if in_right_end == attacks_right else "DZ"


def rink_shapes(color: str = "#9aa7b4") -> list[dict]:
    """Plotly layout shapes that draw an NHL rink (boards, lines, circles, dots)."""
    red, blue = "#d0393e", "#2f6fd6"
    shapes: list[dict] = []
    r = CORNER_RADIUS
    path = (f"M {r},0 L {RINK_LENGTH - r},0 Q {RINK_LENGTH},0 {RINK_LENGTH},{r} "
            f"L {RINK_LENGTH},{RINK_WIDTH - r} Q {RINK_LENGTH},{RINK_WIDTH} {RINK_LENGTH - r},{RINK_WIDTH} "
            f"L {r},{RINK_WIDTH} Q 0,{RINK_WIDTH} 0,{RINK_WIDTH - r} L 0,{r} Q 0,0 {r},0 Z")
    shapes.append(dict(type="path", path=path, line=dict(color=color, width=2), layer="below"))
    for x, c, w in [(GOAL_LINE_LEFT, red, 1), (GOAL_LINE_RIGHT, red, 1), (BLUE_LINE_LEFT, blue, 4),
                    (BLUE_LINE_RIGHT, blue, 4), (CENTER_X, red, 4)]:
        shapes.append(dict(type="line", x0=x, x1=x, y0=1, y1=RINK_WIDTH - 1, line=dict(color=c, width=w),
                           layer="below"))
    for cx in (31.0, 169.0):
        for cy in (CENTER_Y - DOT_OFFSET_Y, CENTER_Y + DOT_OFFSET_Y):
            shapes.append(dict(type="circle", x0=cx - CIRCLE_RADIUS, x1=cx + CIRCLE_RADIUS,
                               y0=cy - CIRCLE_RADIUS, y1=cy + CIRCLE_RADIUS, line=dict(color=red, width=1),
                               layer="below"))
    shapes.append(dict(type="circle", x0=CENTER_X - CIRCLE_RADIUS, x1=CENTER_X + CIRCLE_RADIUS,
                       y0=CENTER_Y - CIRCLE_RADIUS, y1=CENTER_Y + CIRCLE_RADIUS, line=dict(color=blue, width=1),
                       layer="below"))
    for name, (x, y) in LANDMARKS.items():
        if "dot" in name:
            shapes.append(dict(type="circle", x0=x - 1, x1=x + 1, y0=y - 1, y1=y + 1, fillcolor=red,
                               line=dict(width=0), layer="below"))
    for gx in (GOAL_LINE_LEFT, GOAL_LINE_RIGHT):  # creases
        sgn = 1 if gx < CENTER_X else -1
        shapes.append(dict(type="path", path=f"M {gx},{CENTER_Y - 4} L {gx + sgn * 6},{CENTER_Y - 4} "
                                             f"Q {gx + sgn * 7},{CENTER_Y} {gx + sgn * 6},{CENTER_Y + 4} "
                                             f"L {gx},{CENTER_Y + 4}",
                           line=dict(color=red, width=1), fillcolor="rgba(47,111,214,0.15)", layer="below"))
    return shapes


def rink_layout(height: int = 420) -> dict:
    return dict(
        shapes=rink_shapes(),
        xaxis=dict(range=[-3, RINK_LENGTH + 3], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-3, RINK_WIDTH + 3], showgrid=False, zeroline=False, visible=False,
                   scaleanchor="x", scaleratio=1),
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="#f4f8fb", paper_bgcolor="rgba(0,0,0,0)",
    )


def _assign_dot(x: float, y: float, in_circle: bool, y_rank: str | None) -> tuple[float, float]:
    """Which real faceoff dot is this? Uses coarse facts that survive large drift:
    circle → end-zone dot; side of center ice → left/right; vertical order → far/near."""
    left = x < CENTER_X
    if in_circle:
        dx = 31.0 if left else 169.0
    elif abs(x - CENTER_X) < 12:
        return (CENTER_X, CENTER_Y)
    else:
        dx = 70.0 if left else 130.0
    far = (y_rank == "far") if y_rank else y > CENTER_Y
    return (dx, CENTER_Y + DOT_OFFSET_Y if far else CENTER_Y - DOT_OFFSET_Y)


def snap_correction(dots_rink: np.ndarray, in_circle: list[bool], max_snap_ft: float = 60.0) -> np.ndarray | None:
    """Drift fix: nudge the homography so detected faceoff dots land on real dots.

    Camera tracking drifts over a long pan, and a calibration made in one part
    of the rink extrapolates poorly to the far end. Faceoff dots are fixed,
    known points, so whenever the dots model sees them we work out which real
    dot each one is (see `_assign_dot`) and fit a small rink-space correction:
    a shift for one dot, a shift + rotation + scale for two or more.

    Returns a 3x3 correction (rink → rink), or None if nothing could be matched.
    """
    pts = np.asarray(dots_rink, dtype=float).reshape(-1, 2)
    if len(pts) == 0:
        return None
    src, dst = [], []
    for i, ((x, y), circ) in enumerate(zip(pts, in_circle)):
        # Two dots in the same zone → the higher one (larger y) is the far dot.
        partners = [j for j in range(len(pts)) if j != i and in_circle[j] == circ
                    and (pts[j][0] < CENTER_X) == (x < CENTER_X)]
        rank = None
        if partners:
            rank = "far" if y > pts[partners[0]][1] else "near"
        target = _assign_dot(x, y, circ, rank)
        if np.hypot(target[0] - x, target[1] - y) <= max_snap_ft and target not in dst:
            src.append((x, y))
            dst.append(target)
    if not src:
        return None
    src, dst = np.asarray(src), np.asarray(dst)
    if len(src) == 1:
        C = np.eye(3)
        C[:2, 2] = dst[0] - src[0]
        return C
    A, _ = cv2.estimateAffinePartial2D(src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2))
    if A is None:
        return None
    scale = np.sqrt(abs(np.linalg.det(A[:, :2])))
    if not 0.7 <= scale <= 1.4:  # implausible jump — ignore this frame's dots
        return None
    return np.vstack([A, [0, 0, 1]])


def on_rink(x: np.ndarray, y: np.ndarray, margin: float = 5.0) -> np.ndarray:
    return (x > -margin) & (x < RINK_LENGTH + margin) & (y > -margin) & (y < RINK_WIDTH + margin)
