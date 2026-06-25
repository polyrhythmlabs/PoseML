"""Dump the I/O contract of each downloaded .tflite model.

This is the "contract" artifact referenced in PLAN.md Phase 0: exact input/output tensor
names, shapes, dtypes, and quantization params — the thing every downstream stage
(PyTorch parity, CoreML conversion, Swift decode) must match.

Uses LiteRT (ai-edge-litert), the modern standalone TFLite interpreter.

Usage:
    uv run --group reference python -m poseml.convert.inspect_tflite
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TFLITE_DIR = REPO_ROOT / "models" / "tflite"
OUT_PATH = REPO_ROOT / "models" / "tflite" / "io_contract.json"


def _load_interpreter(path: Path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "ai-edge-litert not installed. Run: uv sync --group reference"
        ) from e
    interp = Interpreter(model_path=str(path))
    interp.allocate_tensors()
    return interp


def _describe(details: list[dict]) -> list[dict]:
    out = []
    for d in details:
        out.append(
            {
                "name": d["name"],
                "shape": [int(x) for x in d["shape"]],
                "dtype": str(d["dtype"]),
                "quantization": d.get("quantization"),
            }
        )
    return out


def main() -> int:
    models = sorted(TFLITE_DIR.glob("*.tflite"))
    if not models:
        raise SystemExit(
            f"No .tflite files in {TFLITE_DIR}. Run download_models first."
        )

    contract: dict[str, dict] = {}
    for path in models:
        interp = _load_interpreter(path)
        info = {
            "inputs": _describe(interp.get_input_details()),
            "outputs": _describe(interp.get_output_details()),
            "num_tensors": len(interp.get_tensor_details()),
        }
        contract[path.name] = info

        print(f"\n=== {path.name} ===")
        for io in ("inputs", "outputs"):
            print(f"  {io}:")
            for t in info[io]:
                print(f"    {t['name']:40} {str(t['shape']):20} {t['dtype']}")
        print(f"  tensors: {info['num_tensors']}")

    OUT_PATH.write_text(json.dumps(contract, indent=2))
    print(f"\nWrote contract -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
