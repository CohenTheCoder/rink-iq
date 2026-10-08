"""Body orientation from pose keypoints (which way is a player facing?).

From a single broadcast camera we only see a 2-D skeleton, so we estimate the
facing direction on the ice with three cues:

1. **Shoulder order** — if a player's *right* shoulder appears on the *left*
   of the image, they are facing the camera; otherwise they face away.
2. **Shoulder width** — shoulders look widest when square to the camera and
   collapse to a point when side-on. width / expected-width ≈ |cos(angle)|.
3. **Nose offset** — when side-on, the nose sits on the side they face.

The result is a unit vector in *camera ground coordinates*
(x → image right, y → toward the camera / image bottom). `rink.direction_to_rink`
turns it into rink coordinates. This is a heuristic — treat it as "experimental".
"""
from __future__ import annotations

import numpy as np

NOSE, L_SH, R_SH, L_HIP, R_HIP = 0, 5, 6, 11, 12
SHOULDER_TO_TORSO = 0.85  # shoulder width ≈ 0.85 × shoulder-to-hip length (with pads)


def facing_from_keypoints(kpts: np.ndarray, conf: np.ndarray, min_conf: float = 0.4,
                          fallback_lateral_sign: float = 1.0) -> tuple[np.ndarray, float] | None:
    """kpts: (17, 2) pixel coords, conf: (17,). Returns (unit vector, confidence) or None."""
    if conf[L_SH] < min_conf or conf[R_SH] < min_conf:
        return None
    ls, rs = kpts[L_SH], kpts[R_SH]
    sh_mid = (ls + rs) / 2
    hips_ok = conf[L_HIP] >= min_conf and conf[R_HIP] >= min_conf
    torso = np.linalg.norm(sh_mid - (kpts[L_HIP] + kpts[R_HIP]) / 2) if hips_ok else None
    if not torso or torso < 1:
        return None
    dx = rs[0] - ls[0]  # person's right minus left, in image x
    ratio = float(np.clip(abs(dx) / (SHOULDER_TO_TORSO * torso), 0.0, 1.0))
    toward_camera = 1.0 if dx < 0 else -1.0
    depth = toward_camera * ratio
    if conf[NOSE] >= min_conf and abs(kpts[NOSE][0] - sh_mid[0]) > 0.05 * torso:
        lateral_sign = float(np.sign(kpts[NOSE][0] - sh_mid[0]))
    else:
        lateral_sign = fallback_lateral_sign
    lateral = lateral_sign * np.sqrt(max(0.0, 1.0 - ratio ** 2))
    v = np.array([lateral, depth])
    v /= np.linalg.norm(v) + 1e-9
    confidence = float(np.mean(conf[[L_SH, R_SH, L_HIP, R_HIP]]))
    return v, confidence


def compass(vx: float, vy: float, attacks_right: bool | None = None) -> str:
    """Human label for a rink-space direction. With a team's attack direction, use hockey words."""
    angle = np.degrees(np.arctan2(vy, vx)) % 360
    if attacks_right is None:
        labels = ["→ right end", "↗", "↑ far boards", "↖", "← left end", "↙", "↓ near boards", "↘"]
        return labels[int(((angle + 22.5) % 360) // 45)]
    forward = vx if attacks_right else -vx
    if forward > 0.7:
        return "up-ice (toward attack)"
    if forward < -0.7:
        return "back toward own net"
    return "toward the boards"
