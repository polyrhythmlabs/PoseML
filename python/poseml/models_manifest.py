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


# Exported-model I/O contract: tflite output tensor name -> Swift-friendly Core ML output
# name, keyed by role. Shared by the converter (which names the outputs) and the verifier
# (which reads them back). Kept role-keyed so the 3 landmark variants share one map.
DETECTOR_OUTPUTS = [
    ("Identity", "box_coords"),    # [1, 2254, 12] per-anchor box + keypoint regressions
    ("Identity_1", "box_scores"),  # [1, 2254, 1] per-anchor logits
]
LANDMARK_OUTPUTS = [
    ("Identity", "landmarks"),          # [1, 195] 39 x (x,y,z,visibility,presence)
    ("Identity_1", "pose_flag"),        # [1, 1] pose-presence logit
    ("Identity_4", "world_landmarks"),  # [1, 117] 39 x metric (x,y,z)
]
LANDMARK_EXTRA_OUTPUTS = [
    ("Identity_2", "segmentation"),  # [1, 256, 256, 1] mask
    ("Identity_3", "heatmap"),       # [1, 64, 64, 39]
]


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


def spec_for(path) -> ModelSpec:
    """Look up the ModelSpec for a .tflite path (or filename). Raises ValueError if unknown."""
    name = Path(path).name
    for spec in MODELS.values():
        if spec.name == name:
            return spec
    raise ValueError(f"{name} is not in the model manifest")


def output_map(role: str, *, all_outputs: bool = False) -> list[tuple[str, str]]:
    """Ordered (tflite name, Core ML name) outputs to export for a model role.

    Landmark exports only the ship heads by default; all_outputs re-adds mask + heatmap.
    """
    if role == "detector":
        return list(DETECTOR_OUTPUTS)
    outs = list(LANDMARK_OUTPUTS)
    if all_outputs:
        outs += LANDMARK_EXTRA_OUTPUTS
    return outs
