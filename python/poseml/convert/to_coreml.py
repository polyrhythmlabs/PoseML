"""Convert a ported BlazePose model to a Core ML ML Program (.mlpackage).

Pipeline: parse .tflite -> `TfliteModule` (NHWC) -> wrap in an NCHW **image** front end
-> `torch.jit.trace` -> `coremltools` ML Program.

Performance choices (see PLAN.md "Overriding priority"):
  * ML Program + FLOAT16 + ComputeUnit.ALL  -> ANE-resident, FP16 execution.
  * `ct.ImageType` input  -> the model takes a CVPixelBuffer directly (zero-copy from the
    camera); the [0,255] -> [0,1] normalization is baked into `scale`, so Swift does no
    per-pixel math. (MediaPipe's ImageToTensorCalculator feeds these nets in [0,1].)
  * Outputs are renamed to something Swift-friendly, and the segmentation mask + heatmap
    heads are pruned by default (dead-code-eliminated by coremltools) since the
    latency build only needs the landmark/score/world-landmark regressions.

Usage:
    uv run --group reference python -m poseml.convert.to_coreml \
        --model models/tflite/pose_landmark_full.tflite
    uv run --group reference python -m poseml.convert.to_coreml \
        --model models/tflite/pose_detection.tflite
    # keep every head (adds segmentation + heatmap outputs):
    uv run ... --model models/tflite/pose_landmark_full.tflite --all-outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import torch
from torch import nn

from poseml.models_manifest import REPO_ROOT, output_map, spec_for
from poseml.tflite_port import build

COREML_DIR = REPO_ROOT / "models" / "coreml"


class ImageFrontEnd(nn.Module):
    """Adapt the NHWC core module to a channel-first image input.

    Core ML `ImageType` feeds pixels as NCHW ``[1, 3, H, W]``; the ported graph is
    NHWC-native. The leading transpose is folded away by coremltools' layout passes, so it
    costs nothing at runtime. Selecting a subset of outputs lets coremltools prune the
    unreferenced heads (mask/heatmap) from the graph entirely.
    """

    def __init__(self, core: nn.Module, out_names: list[str]):
        super().__init__()
        self.core = core
        self.out_names = out_names

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = image.permute(0, 2, 3, 1)  # NCHW -> NHWC
        out = self.core(x)
        return tuple(out[n] for n in self.out_names)


_PRECISION = {"fp16": ct.precision.FLOAT16, "fp32": ct.precision.FLOAT32}


def convert(model_path: Path, out_path: Path, *, all_outputs: bool = False,
            precision: str = "fp16") -> Path:
    spec = spec_for(model_path)
    core = build(str(model_path))
    graph = core.g  # reuse the module's parsed graph (avoid a second parse)
    _, h, w, _ = graph.t(graph.inputs[0]).shape  # NHWC

    out_map = output_map(spec.role, all_outputs=all_outputs)
    tflite_names = [t for t, _ in out_map]
    coreml_names = [c for _, c in out_map]

    wrapper = ImageFrontEnd(core, tflite_names).eval()

    example = torch.rand(1, 3, h, w)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, h, w),
                scale=1.0 / 255.0,   # [0,255] uint8 pixels -> [0,1], baked in
                bias=[0.0, 0.0, 0.0],
                color_layout=ct.colorlayout.RGB,
            )
        ],
        outputs=[ct.TensorType(name=n) for n in coreml_names],
        convert_to="mlprogram",
        compute_precision=_PRECISION[precision],  # fp16 for ANE; fp32 to check conversion fidelity
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS17,
    )

    mlmodel.short_description = f"BlazePose {spec.role} ({model_path.stem}) — ported from tflite."
    mlmodel.input_description["image"] = f"RGB image, {w}x{h}."
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))
    print(f"saved {out_path}")
    print(f"  input : image [1,3,{h},{w}] RGB (scale 1/255)")
    shapes = {graph.t(o).name: graph.t(o).shape for o in graph.outputs}
    for tname, cname in out_map:
        print(f"  output: {cname:16} <- {tname} {shapes[tname]}")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="path to a .tflite in the manifest")
    ap.add_argument("--out", help="output .mlpackage path (default: models/coreml/<stem>.mlpackage)")
    ap.add_argument("--all-outputs", action="store_true",
                    help="keep the segmentation mask + heatmap heads (landmark only)")
    ap.add_argument("--precision", choices=("fp16", "fp32"), default="fp16",
                    help="fp16 (default, ANE) or fp32 (conversion-fidelity check)")
    args = ap.parse_args()

    model_path = Path(args.model)
    suffix = "" if args.precision == "fp16" else f".{args.precision}"
    out_path = Path(args.out) if args.out else COREML_DIR / f"{model_path.stem}{suffix}.mlpackage"
    convert(model_path, out_path, all_outputs=args.all_outputs, precision=args.precision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
