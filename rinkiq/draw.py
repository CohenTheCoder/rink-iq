"""Overlay drawing: boxes, IDs, live speed and facing arrows on a video frame."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

TEAM_BGR = {"A": (214, 111, 47), "B": (62, 57, 208), "REF": (150, 150, 150), "?": (200, 200, 200)}


def annotate(frame: np.ndarray, rows: pd.DataFrame, puck_px: tuple[float, float] | None = None,
             names: dict | None = None) -> np.ndarray:
    """rows: the analysed track rows for this frame (needs x1..y2, team, speed_mph, face_x/face_y)."""
    img = frame.copy()
    scale = img.shape[1] / 1280
    for _, r in rows.iterrows():
        if np.isnan(r.get("x1", np.nan)):
            continue
        color = TEAM_BGR.get(r.get("team", "?"), (200, 200, 200))
        x1, y1, x2, y2 = map(int, (r["x1"], r["y1"], r["x2"], r["y2"]))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, max(1, int(2 * scale)))
        cx, fy = (x1 + x2) // 2, y2
        cv2.ellipse(img, (cx, fy), (int((x2 - x1) * 0.6), int(6 * scale)), 0, 0, 360, color, max(1, int(2 * scale)))
        label = names.get(r["track_id"], f"#{int(r['track_id'])}") if names else f"#{int(r['track_id'])}"
        speed = r.get("speed_mph", np.nan)
        if not np.isnan(speed):
            label += f"  {speed:.0f} mph"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, (255, 255, 255), 1,
                    cv2.LINE_AA)
        fx, fyv = r.get("face_x", np.nan), r.get("face_y", np.nan)
        if not (np.isnan(fx) or np.isnan(fyv)):
            # Rink +y points to the far boards = up the image, so flip y for drawing.
            L = int((y2 - y1) * 0.5)
            tip = (int(cx + fx * L), int(fy - fyv * L * 0.5))
            cv2.arrowedLine(img, (cx, fy), tip, (0, 220, 255), max(1, int(2 * scale)), tipLength=0.35)
    if puck_px is not None and not np.isnan(puck_px[0]):
        cv2.circle(img, (int(puck_px[0]), int(puck_px[1])), int(9 * scale), (0, 255, 255), 2)
    return img
