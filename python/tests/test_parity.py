"""Regression gate: every ported BlazePose model matches its .tflite reference.

Skips gracefully if the .tflite files haven't been downloaded (`make models`).
"""

from __future__ import annotations

import pytest

from poseml.models_manifest import MODELS, TFLITE_DIR
from poseml.verify.parity import check


@pytest.mark.parametrize("name", [spec.name for spec in MODELS.values()])
def test_parity(name: str) -> None:
    path = TFLITE_DIR / name
    if not path.exists():
        pytest.skip(f"{name} not downloaded (run `make models`)")
    ok, rows = check(str(path), seed=0, abs_tol=5e-3, k=3.0)
    failed = [r for r in rows if not r[5]]
    assert ok, f"{name} parity failed for outputs: {[f[0] for f in failed]}\n{rows}"
