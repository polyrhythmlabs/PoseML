"""Port BlazePose .tflite models to PyTorch.

The port is generated deterministically from the .tflite file (structure + fp16 weights),
so there's no separate checkpoint to manage — `build(path)` reconstructs the exact module.
"""

from __future__ import annotations

from poseml.tflite_port.parser import Graph, parse
from poseml.tflite_port.torch_graph import TfliteModule


def build(tflite_path: str) -> TfliteModule:
    """Parse a .tflite file and return a traceable PyTorch module equivalent to it."""
    return TfliteModule(parse(tflite_path)).eval()


__all__ = ["Graph", "TfliteModule", "build", "parse"]
