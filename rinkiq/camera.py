"""Camera-motion compensation.

Broadcast cameras pan and zoom, so a player standing still can still "move" in
pixels. Because the ice is a flat plane and the camera mostly rotates in place,
any two frames are related by a homography. We estimate it with ORB feature
matching + RANSAC, ignoring the players (they move on their own), and use it
to express every frame in the pixel space of one *reference frame*.

Broadcasts also burn in graphics (score bug, network logo) that never move
on screen. If we let ORB match those, every frame "agrees" the camera is
still. `static_overlay_mask` finds them by looking for pixels that barely
change across the whole clip, and we ignore them.
"""
from __future__ import annotations

import cv2
import numpy as np


def static_overlay_mask(path: str, samples: int = 30, std_threshold: float = 3.0) -> tuple[np.ndarray | None, bool]:
    """Return (mask of pixels that are burned-in graphics, camera_is_fixed).

    Pixels whose brightness hardly changes over the clip are overlays — unless
    almost the whole frame is static, which simply means a fixed camera.
    """
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames, size = [], None
    for i in np.linspace(0, max(0, n - 1), min(samples, max(1, n))).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            size = (f.shape[1], f.shape[0])
            frames.append(cv2.cvtColor(cv2.resize(f, None, fx=0.25, fy=0.25), cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    if len(frames) < 5:
        return None, False
    std = np.std(np.stack(frames), axis=0)
    static = (std < std_threshold).astype(np.uint8)
    if static.mean() > 0.6:
        return None, True  # most of the picture never changes → the camera doesn't move
    static = cv2.dilate(static, np.ones((5, 5), np.uint8))
    full = cv2.resize(static, size, interpolation=cv2.INTER_NEAREST)
    return full.astype(bool), False


class CameraMotion:
    def __init__(self, reference: np.ndarray, n_features: int = 2000, min_inliers: int = 40,
                 ignore: np.ndarray | None = None):
        self.ignore = ignore
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.min_inliers = min_inliers
        self.ref_kp, self.ref_des = self._features(reference, None)
        self.prev_kp, self.prev_des = self.ref_kp, self.ref_des
        self.prev_to_ref = np.eye(3)

    def _features(self, frame: np.ndarray, boxes: np.ndarray | None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        mask = np.full(gray.shape, 255, np.uint8)
        if self.ignore is not None and self.ignore.shape == mask.shape:
            mask[self.ignore] = 0
        if boxes is not None:
            for x1, y1, x2, y2 in np.asarray(boxes, dtype=int):
                pad = int(0.15 * (y2 - y1))
                mask[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad] = 0
        return self.orb.detectAndCompute(gray, mask)

    def _match(self, kp, des, kp2, des2) -> tuple[np.ndarray | None, int]:
        if des is None or des2 is None or len(kp) < 8 or len(kp2) < 8:
            return None, 0
        matches = self.matcher.match(des, des2)
        if len(matches) < 8:
            return None, 0
        src = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        return H, int(inl.sum()) if inl is not None else 0

    def update(self, frame: np.ndarray, player_boxes: np.ndarray | None = None) -> np.ndarray:
        """Return the 3x3 matrix mapping this frame's pixels to reference-frame pixels.

        Primary path: match against the previous frame (big overlap, ~1000
        inliers) and chain. A direct match to the reference frame is only used
        when it *agrees* with the chain — then it wipes out accumulated drift.
        (Repetitive crowds and ads can produce confident-looking but wrong
        matches once the reference view is off screen.)
        """
        kp, des = self._features(frame, player_boxes)
        H_prev, n_prev = self._match(kp, des, self.prev_kp, self.prev_des)
        if H_prev is not None and n_prev >= self.min_inliers // 2:
            to_ref = self.prev_to_ref @ H_prev
        else:
            to_ref = self.prev_to_ref  # lost track: assume the camera held still
        H_ref, n_ref = self._match(kp, des, self.ref_kp, self.ref_des)
        if H_ref is not None and n_ref >= max(self.min_inliers, 0.3 * n_prev):
            h, w = frame.shape[:2]
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            gap = np.abs(cv2.perspectiveTransform(corners, H_ref) - cv2.perspectiveTransform(corners, to_ref)).max()
            if gap < 0.03 * w:
                to_ref = H_ref
        self.prev_kp, self.prev_des, self.prev_to_ref = kp, des, to_ref
        return to_ref


def find_cuts(path: str, threshold: float = 0.6) -> list[int]:
    """Frame indices where the broadcast cuts to a different camera.

    Compares colour histograms of consecutive (downscaled) frames; a cut makes
    the correlation drop sharply. A calibration is only valid within one shot.
    """
    cap = cv2.VideoCapture(path)
    cuts, prev, i = [], None, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        small = cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([small], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        if prev is not None and cv2.compareHist(prev, hist, cv2.HISTCMP_CORREL) < threshold:
            cuts.append(i)
        prev, i = hist, i + 1
    cap.release()
    return cuts


def shots(path: str, n_frames: int, fps: float) -> list[dict]:
    """Split a clip into camera shots: [{start, end, seconds}, ...] (end exclusive)."""
    bounds = [0] + find_cuts(path) + [n_frames]
    return [dict(start=a, end=b, seconds=round((b - a) / fps, 1)) for a, b in zip(bounds, bounds[1:]) if b - a > 5]
