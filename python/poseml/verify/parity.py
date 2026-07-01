"""Numerical parity: PyTorch port vs the original .tflite.

Principle: the .tflite weights are fp16, so bit-exact agreement is impossible — and some
outputs (e.g. BlazePose's segmentation mask) are numerically ill-conditioned, where even
TFLite's own two backends (XNNPACK vs reference kernels) disagree substantially.

So we don't gate against a fixed tolerance alone. For each output we compute a **noise
floor** = |XNNPACK - reference| (TFLite disagreeing with itself), and require:

    error(torch vs reference) <= max(abs_tol, K * noise_floor)

i.e. the port must match TFLite at least about as well as TFLite matches itself. This keeps
a tight gate on well-conditioned outputs (landmarks: floor ~1e-3) while not flagging
ill-conditioned ones (mask: floor ~0.4) as failures.

Usage:
    uv run --group reference python -m poseml.verify.parity models/tflite/pose_landmark_full.tflite
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from poseml.litert import make_interpreter, run_outputs
from poseml.tflite_port.parser import parse
from poseml.tflite_port.torch_graph import TfliteModule

Row = tuple[str, tuple[int, ...], float, float, float, bool]


def _maxabs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())


class Harness:
    """Parse + build once; compare per seed. All the heavy setup is input-independent."""

    def __init__(self, model: str):
        self.g = parse(model)
        self.mod = TfliteModule(self.g).eval()
        self.ref = make_interpreter(model, resolver="ref")  # golden (no delegate fusion)
        self.alt = make_interpreter(model, resolver="auto")  # XNNPACK, for the noise floor
        self.in_shape = self.g.t(self.g.inputs[0]).shape

    def compare(self, seed: int, abs_tol: float, k: float) -> tuple[bool, list[Row]]:
        x = np.random.default_rng(seed).random(self.in_shape, dtype=np.float32)
        ref = run_outputs(self.ref, x)
        alt = run_outputs(self.alt, x)
        with torch.no_grad():
            got = {n: v.detach().cpu().numpy() for n, v in self.mod(torch.from_numpy(x)).items()}

        rows: list[Row] = []
        ok = True
        for name, r in ref.items():
            gv = got.get(name)
            if gv is None or gv.shape != r.shape:
                rows.append((name, r.shape, float("inf"), 0.0, float("inf"), False))
                ok = False
                continue
            err = _maxabs(gv, r)
            floor = _maxabs(alt[name], r)
            gate = max(abs_tol, k * floor)
            passed = err <= gate
            ok = ok and passed
            rows.append((name, r.shape, err, floor, gate, passed))
        return ok, rows


def check(model: str, seed: int = 0, abs_tol: float = 5e-3, k: float = 3.0) -> tuple[bool, list[Row]]:
    return Harness(model).compare(seed, abs_tol, k)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--abs-tol", type=float, default=5e-3)
    ap.add_argument("--k", type=float, default=3.0, help="noise-floor multiplier")
    args = ap.parse_args()

    print(f"model: {args.model}")
    print(f"gate: err(torch vs ref) <= max({args.abs_tol:.0e}, {args.k}x |xnnpack-ref|)\n")
    print(f"{'output':13} {'shape':17} {'torch_err':>11} {'noise_flr':>11} {'gate':>11}  st")

    harness = Harness(args.model)
    all_ok = True
    for seed in range(args.seeds):
        ok, rows = harness.compare(seed, args.abs_tol, args.k)
        all_ok = all_ok and ok
        print(f"[seed {seed}]")
        for name, shape, err, floor, gate, passed in rows:
            print(f"  {name:11} {str(tuple(shape)):17} {err:11.3e} {floor:11.3e} "
                  f"{gate:11.3e}  {'ok' if passed else 'FAIL'}")

    print(f"\n{'PASS' if all_ok else 'FAIL'} over {args.seeds} seed(s)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
