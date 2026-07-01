"""Numerical parity: the exported Core ML model vs its .tflite reference.

The Core ML model bakes ``pixel/255`` into an image input, so we can feed *bit-identical*
values to both sides: a random uint8 image goes to Core ML (which divides by 255 internally)
and the same ``uint8/255`` float goes to the tflite interpreter. That isolates the FP16
conversion error from any input-quantization noise.

Two things get validated, at two tolerances:
  * **Conversion fidelity** — export with ``--precision fp32`` and this passes at ~1e-3:
    proof the tflite -> coremltools graph translation is exact.
  * **FP16 budget** — the shipping fp16 model diverges more (fp16 *activation* rounding,
    which is what runs on the ANE). The default gate ``max(atol, rtol * max|ref|)`` is the
    empirical worst-case budget measured on **random-noise** input (adversarial: the net is
    out of distribution, so activations are large and fp16 error is amplified). On real
    frames the delta is far smaller; end-to-end accuracy is re-checked in Phase 3.

Usage:
    uv run --group reference python -m poseml.verify.coreml_parity \
        --model models/tflite/pose_landmark_full.tflite \
        --coreml models/coreml/pose_landmark_full.mlpackage
"""

from __future__ import annotations

import argparse

import coremltools as ct
import numpy as np
from PIL import Image

from poseml.litert import make_interpreter, run_outputs
from poseml.models_manifest import output_map, spec_for


def compare(model_path: str, coreml_path: str, *, seeds: int, atol: float, rtol: float) -> bool:
    out_map = output_map(spec_for(model_path).role)  # ship-outputs only
    mlmodel = ct.models.MLModel(coreml_path)
    itype = mlmodel.get_spec().description.input[0].type.imageType
    h, w = itype.height, itype.width
    interp = make_interpreter(model_path, resolver="auto")  # input-independent; build once

    print(f"model: {coreml_path}")
    print(f"gate: max_abs_err <= max({atol:.0e}, {rtol:.0e} * max|ref|)\n")
    print(f"{'output':16} {'shape':16} {'max_abs':>11} {'mean_abs':>11} {'gate':>11}  st")

    all_ok = True
    for seed in range(seeds):
        img_u8 = np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)
        ref = run_outputs(interp, (img_u8.astype(np.float32) / 255.0)[None])  # matches baked scale
        pred = mlmodel.predict({"image": Image.fromarray(img_u8, "RGB")})

        print(f"[seed {seed}]")
        for tname, cname in out_map:
            r = ref[tname].astype(np.float64)
            g = np.asarray(pred[cname], dtype=np.float64).reshape(r.shape)
            diff = np.abs(g - r)
            max_abs, mean_abs = float(diff.max()), float(diff.mean())
            gate = max(atol, rtol * float(np.abs(r).max()))
            ok = max_abs <= gate
            all_ok = all_ok and ok
            print(f"  {cname:14} {str(tuple(r.shape)):16} {max_abs:11.3e} {mean_abs:11.3e} "
                  f"{gate:11.3e}  {'ok' if ok else 'FAIL'}")

    print(f"\n{'PASS' if all_ok else 'FAIL'} over {seeds} seed(s)")
    return all_ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="reference .tflite")
    ap.add_argument("--coreml", required=True, help="exported .mlpackage")
    ap.add_argument("--seeds", type=int, default=3)
    # Defaults = the fp16 activation-precision budget (worst-case on noise input).
    # For a strict conversion-fidelity check on an fp32 export, pass --atol 5e-3 --rtol 5e-3.
    ap.add_argument("--atol", type=float, default=3e-2)
    ap.add_argument("--rtol", type=float, default=1.5e-2)
    args = ap.parse_args()
    ok = compare(args.model, args.coreml, seeds=args.seeds, atol=args.atol, rtol=args.rtol)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
