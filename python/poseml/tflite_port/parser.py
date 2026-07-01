"""Parse a .tflite flatbuffer into a small, explicit graph IR.

We read **structure + op options** from the flatbuffer schema (pure-python `tflite`
package) and **resolved float weight values** from the LiteRT interpreter.

Constant folding: any op that merely reformats a constant — `DEQUANTIZE` (fp16→f32) and
`DENSIFY` (sparse→dense) — is evaluated here and its output recorded as a plain constant,
then the op is dropped from the graph. So the executor sees a single notion, "a tensor with
constant data", and only ever dispatches the real compute ops:
CONV_2D, DEPTHWISE_CONV_2D, ADD, PAD, RESHAPE, RESIZE_BILINEAR, MAX_POOL_2D,
CONCATENATION, DEPTH_TO_SPACE, LOGISTIC.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from tflite.ActivationFunctionType import ActivationFunctionType
from tflite.AddOptions import AddOptions
from tflite.BuiltinOperator import BuiltinOperator
from tflite.BuiltinOptions import BuiltinOptions
from tflite.ConcatenationOptions import ConcatenationOptions
from tflite.Conv2DOptions import Conv2DOptions
from tflite.DepthToSpaceOptions import DepthToSpaceOptions
from tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions
from tflite.Model import Model
from tflite.Padding import Padding
from tflite.Pool2DOptions import Pool2DOptions
from tflite.ReshapeOptions import ReshapeOptions
from tflite.ResizeBilinearOptions import ResizeBilinearOptions

from poseml.litert import make_interpreter

# Ops that only reformat a constant; folded to constants at parse time and dropped.
_FOLD_OPS = {"DEQUANTIZE", "DENSIFY"}

_BUILTIN_NAMES = {v: k for k, v in vars(BuiltinOperator).items() if not k.startswith("_")}


@dataclass
class Tensor:
    name: str
    shape: tuple[int, ...]
    is_const: bool
    data: np.ndarray | None  # float32 values for constants, else None


@dataclass
class Op:
    type: str
    inputs: list[int]
    outputs: list[int]
    opts: dict = field(default_factory=dict)


@dataclass
class Graph:
    tensors: dict[int, Tensor]
    ops: list[Op]
    inputs: list[int]
    outputs: list[int]

    def t(self, idx: int) -> Tensor:
        return self.tensors[idx]


_ACT = {
    ActivationFunctionType.NONE: "none",
    ActivationFunctionType.RELU: "relu",
    ActivationFunctionType.RELU6: "relu6",
    ActivationFunctionType.RELU_N1_TO_1: "relu_n1_to_1",
    ActivationFunctionType.TANH: "tanh",
}
_PAD = {Padding.SAME: "same", Padding.VALID: "valid"}


# Per builtin-options type: (schema class, extractor). Shared prologue lives in _decode_options.
_OPTION_DECODERS = {
    BuiltinOptions.Conv2DOptions: (Conv2DOptions, lambda o: {
        "padding": _PAD[o.Padding()],
        "stride": (o.StrideH(), o.StrideW()),
        "dilation": (o.DilationHFactor(), o.DilationWFactor()),
        "activation": _ACT[o.FusedActivationFunction()],
    }),
    BuiltinOptions.DepthwiseConv2DOptions: (DepthwiseConv2DOptions, lambda o: {
        "padding": _PAD[o.Padding()],
        "stride": (o.StrideH(), o.StrideW()),
        "depth_multiplier": o.DepthMultiplier(),
        "dilation": (o.DilationHFactor(), o.DilationWFactor()),
        "activation": _ACT[o.FusedActivationFunction()],
    }),
    BuiltinOptions.Pool2DOptions: (Pool2DOptions, lambda o: {
        "padding": _PAD[o.Padding()],
        "stride": (o.StrideH(), o.StrideW()),
        "filter": (o.FilterHeight(), o.FilterWidth()),
        "activation": _ACT[o.FusedActivationFunction()],
    }),
    BuiltinOptions.AddOptions: (AddOptions, lambda o: {
        "activation": _ACT[o.FusedActivationFunction()],
    }),
    BuiltinOptions.ConcatenationOptions: (ConcatenationOptions, lambda o: {
        "axis": o.Axis(),
        "activation": _ACT[o.FusedActivationFunction()],
    }),
    BuiltinOptions.ResizeBilinearOptions: (ResizeBilinearOptions, lambda o: {
        "align_corners": bool(o.AlignCorners()),
        "half_pixel_centers": bool(o.HalfPixelCenters()),
    }),
    BuiltinOptions.DepthToSpaceOptions: (DepthToSpaceOptions, lambda o: {
        "block_size": o.BlockSize(),
    }),
    BuiltinOptions.ReshapeOptions: (ReshapeOptions, lambda o: {
        "new_shape": list(o.NewShapeAsNumpy()) if o.NewShapeLength() else None,
    }),
}


def _decode_options(op) -> dict:
    dec = _OPTION_DECODERS.get(op.BuiltinOptionsType())
    tab = op.BuiltinOptions()
    if dec is None or tab is None:
        return {}
    cls, extract = dec
    o = cls()
    o.Init(tab.Bytes, tab.Pos)
    return extract(o)


def parse(tflite_path: str) -> Graph:
    with open(tflite_path, "rb") as f:
        buf = f.read()
    model = Model.GetRootAs(buf, 0)
    sg = model.Subgraphs(0)

    op_names = [_op_name(model.OperatorCodes(i)) for i in range(model.OperatorCodesLength())]
    raw = [sg.Operators(i) for i in range(sg.OperatorsLength())]
    has_densify = any(op_names[op.OpcodeIndex()] == "DENSIFY" for op in raw)

    # A single reference-kernel interpreter serves both jobs: read constant buffers, and
    # (only if the model has sparse weights) materialize DENSIFY outputs after one zero invoke.
    interp = make_interpreter(tflite_path, resolver="ref", preserve_all=has_densify)
    if has_densify:
        inp = interp.get_input_details()[0]
        interp.set_tensor(inp["index"], np.zeros(inp["shape"], dtype=inp["dtype"]))
        interp.invoke()

    tensors: dict[int, Tensor] = {}
    for i in range(sg.TensorsLength()):
        t = sg.Tensors(i)
        shape = tuple(int(x) for x in t.ShapeAsNumpy()) if t.ShapeLength() else ()
        is_const = model.Buffers(t.Buffer()).DataLength() > 0
        data = np.asarray(interp.get_tensor(i)).astype(np.float32) if is_const else None
        tensors[i] = Tensor(t.Name().decode(), shape, is_const, data)

    ops: list[Op] = []
    for op in raw:
        name = op_names[op.OpcodeIndex()]
        ins, outs = _in_out(op)
        if name in _FOLD_OPS:
            _fold_const(name, ins, outs, tensors, interp)
            continue
        ops.append(Op(name, ins, outs, _decode_options(op)))

    return Graph(
        tensors=tensors,
        ops=ops,
        inputs=[int(x) for x in sg.InputsAsNumpy()],
        outputs=[int(x) for x in sg.OutputsAsNumpy()],
    )


def _fold_const(name: str, ins: list[int], outs: list[int], tensors, interp) -> None:
    """Record the reformatter op's output as a constant."""
    out = outs[0]
    if name == "DEQUANTIZE":
        src = tensors[ins[0]]
        assert src.data is not None, "DEQUANTIZE of a non-constant is not supported"
        data = src.data
    else:  # DENSIFY: dense values were materialized by the zero invoke above
        data = np.asarray(interp.get_tensor(out)).astype(np.float32)
    tensors[out] = Tensor(tensors[out].name, tuple(data.shape), is_const=True, data=data)


def _in_out(op) -> tuple[list[int], list[int]]:
    ins = [int(x) for x in op.InputsAsNumpy()] if op.InputsLength() else []
    outs = [int(x) for x in op.OutputsAsNumpy()] if op.OutputsLength() else []
    return ins, outs


def _op_name(code) -> str:
    builtin = max(code.BuiltinCode(), code.DeprecatedBuiltinCode())
    return _BUILTIN_NAMES.get(builtin, f"UNKNOWN_{builtin}")
