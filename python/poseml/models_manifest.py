"""Registry of the official MediaPipe BlazePose .tflite models.

These are the source-of-truth weights we port to PyTorch and then convert to CoreML.
URLs point at Google's public `mediapipe-assets` bucket. SHA-256 hashes are filled in
after the first verified download (run `download_models.py`, which prints them) so that
future downloads are integrity-checked and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TFLITE_DIR = REPO_ROOT / "models" / "tflite"

BUCKET = "https://storage.googleapis.com/mediapipe-assets"


@dataclass(frozen=True)
class ModelSpec:
    name: str  # local filename (under models/tflite/)
    url: str
    sha256: str | None  # None until pinned; download script prints the computed value
    role: str  # "detector" | "landmark"
    note: str = ""


MODELS: dict[str, ModelSpec] = {
    "pose_detection": ModelSpec(
        name="pose_detection.tflite",
        url=f"{BUCKET}/pose_detection.tflite",
        sha256="9ba9dd3d42efaaba86b4ff0122b06f29c4122e756b329d89dca1e297fd8f866c",
        role="detector",
        note="SSD-style person detector; runs on the full frame to find the ROI.",
    ),
    "pose_landmark_lite": ModelSpec(
        name="pose_landmark_lite.tflite",
        url=f"{BUCKET}/pose_landmark_lite.tflite",
        sha256="1150dc68a713b80660b90ef46ce4e85c1c781bb88b6e3512cc64e6a685ba5588",
        role="landmark",
        note="Fastest landmark variant. Lowest accuracy.",
    ),
    "pose_landmark_full": ModelSpec(
        name="pose_landmark_full.tflite",
        url=f"{BUCKET}/pose_landmark_full.tflite",
        sha256="e9a5c5cb17f736fafd4c2ec1da3b3d331d6edbe8a0d32395855aeb2cdfd64b9f",
        role="landmark",
        note="Balanced default; start here.",
    ),
    "pose_landmark_heavy": ModelSpec(
        name="pose_landmark_heavy.tflite",
        url=f"{BUCKET}/pose_landmark_heavy.tflite",
        sha256="59e42d71bcd44cbdbabc419f0ff76686595fd265419566bd4009ef703ea8e1fe",
        role="landmark",
        note="Most accurate, heaviest.",
    ),
}
