"""Step 1 — Import the models from Hugging Face.

We don't train anything from scratch. Two pretrained pieces do the "seeing":

* **Hockey-Vision** (huggingface.co/AlexRHaigh/Hockey-Vision, Apache-2.0) — small
  YOLO detectors trained on NHL broadcast video. We use:
    - `new_player_model.pt` → skaters, goalies, referees *and which team* (by jersey colour)
    - `new_nano_puck.pt`    → the puck
    - `new_rink_model.pt`   → painted lines (blue / center lines) for calibration
    - `new_dots.pt`         → faceoff dots for calibration
* **YOLOv8n-pose** (Ultralytics) — 17 body keypoints per person; we use the
  shoulders, hips and nose to estimate which way a player's body is facing.

`hf_hub_download` caches files in ~/.cache/huggingface, so each one downloads once.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

HF_REPO = "AlexRHaigh/Hockey-Vision"
WEIGHTS = {
    "player": "new_player_model.pt",
    "puck": "new_nano_puck.pt",
    "rink": "new_rink_model.pt",
    "dots": "new_dots.pt",
}
POSE_WEIGHTS = "yolov8n-pose.pt"  # Ultralytics fetches this on first use
SAMPLE_CLIPS = {
    "Broadcast clip · neutral zone (MTL vs SJ)": "Examples/dots/clip2.mp4",
    "Broadcast clip · zone entry (MTL vs SJ)": "Examples/puck/clip2.mp4",
    "Broadcast clip · rink markings (MTL vs SJ)": "Examples/rink/clip2.mp4",
}

# Hand-checked landmark points (frame 0) so the sample clip can be calibrated in one click.
SAMPLE_CALIBRATION = {
    "Broadcast clip · neutral zone (MTL vs SJ)": [
        (332, 262, "Neutral dot · left · far"), (158, 700, "Neutral dot · left · near"),
        (1188, 265, "Neutral dot · right · far"), (743, 135, "Center line × far boards"),
        (272, 135, "Left blue line × far boards"),
    ],
}


def weight_path(kind: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(HF_REPO, WEIGHTS[kind])


@lru_cache(maxsize=None)
def load(kind: str):
    """Load one model. kind ∈ {player, puck, rink, dots, pose}."""
    from ultralytics import YOLO

    if kind == "pose":
        return YOLO(POSE_WEIGHTS)
    return YOLO(weight_path(kind))


def download_sample_clip(name: str) -> Path:
    """Grab one of the short example clips that ship with the Hockey-Vision repo.

    These are the model author's demo videos (some have light overlays drawn on
    them). They are downloaded on demand and are not redistributed in this repo.
    """
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(HF_REPO, SAMPLE_CLIPS[name]))
