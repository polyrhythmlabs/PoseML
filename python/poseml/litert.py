"""Shared helpers for the LiteRT (ai-edge-litert) interpreter.

One place to build an interpreter (with the "not installed" hint) and to run a model,
so the parser, parity harness, and inspect script don't each re-implement it.
"""

from __future__ import annotations

import numpy as np

_RESOLVERS = {"auto": "AUTO", "ref": "BUILTIN_REF"}


def make_interpreter(model_path, *, resolver: str = "auto", preserve_all: bool = False):
    """Build and allocate a LiteRT interpreter.

    resolver: "auto" (XNNPACK) or "ref" (reference kernels — deterministic, no delegate
    fusion, so intermediate tensors are materializable with preserve_all=True).
    """
    try:
        from ai_edge_litert.interpreter import Interpreter, OpResolverType
    except ImportError as e:  # pragma: no cover
        raise SystemExit("ai-edge-litert not installed. Run: uv sync --group reference") from e

    kwargs = {"experimental_op_resolver_type": getattr(OpResolverType, _RESOLVERS[resolver])}
    if preserve_all:
        kwargs["experimental_preserve_all_tensors"] = True
    it = Interpreter(model_path=str(model_path), **kwargs)
    it.allocate_tensors()
    return it


def run_outputs(interp, x: np.ndarray) -> dict[str, np.ndarray]:
    """Set the single input, invoke, and return {output_name: value}."""
    interp.set_tensor(interp.get_input_details()[0]["index"], x.astype(np.float32))
    interp.invoke()
    return {d["name"]: interp.get_tensor(d["index"]).copy() for d in interp.get_output_details()}
