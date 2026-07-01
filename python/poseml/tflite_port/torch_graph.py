"""Execute a parsed TFLite Graph as a PyTorch nn.Module.

Design: tensors are kept in **NHWC** (TFLite's native layout) throughout the env, so
elementwise / concat / reshape / pad ops match TFLite axis semantics exactly. Spatial ops
(conv, depthwise, pool, depth_to_space, resize) transpose to NCHW just for the torch call
and back. The extra transposes are free in the final CoreML model — coremltools folds them
during layout canonicalization.

The module is traceable (`torch.jit.trace`) for Phase-2 CoreML conversion.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from poseml.tflite_port.parser import Graph, Op


def _same_pad_1d(i: int, k: int, s: int, d: int) -> tuple[int, int]:
    """TensorFlow 'SAME' padding for one spatial dim. Returns (before, after)."""
    k_eff = (k - 1) * d + 1
    out = math.ceil(i / s)
    total = max(0, (out - 1) * s + k_eff - i)
    return total // 2, total - total // 2


def _act(x: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "none":
        return x
    if kind == "relu":
        return F.relu(x)
    if kind == "relu6":
        return F.relu6(x)
    if kind == "tanh":
        return torch.tanh(x)
    if kind == "relu_n1_to_1":
        return torch.clamp(x, -1.0, 1.0)
    raise NotImplementedError(f"activation {kind}")


class TfliteModule(nn.Module):
    def __init__(self, graph: Graph):
        super().__init__()
        self.g = graph
        # Register only constants actually consumed by an op (the raw fp16/sparse originals
        # are unreferenced once DEQUANTIZE/DENSIFY are folded away in the parser).
        used = {i for op in graph.ops for i in op.inputs}
        self._const_names: dict[int, str] = {}
        for idx in used:
            t = graph.tensors[idx]
            if t.is_const and t.data is not None:
                name = f"c{idx}"
                self.register_buffer(name, torch.from_numpy(t.data.copy()), persistent=False)
                self._const_names[idx] = name

    def _const(self, idx: int) -> torch.Tensor:
        return getattr(self, self._const_names[idx])

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        env = self._eval_env(x)
        return {self.g.t(o).name: env[o] for o in self.g.outputs}

    def _eval_env(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        g = self.g
        env: dict[int, torch.Tensor] = {idx: self._const(idx) for idx in self._const_names}
        env[g.inputs[0]] = x  # NHWC
        for op in g.ops:
            self._run(op, env)
        return env

    # --- op dispatch -----------------------------------------------------------
    def _run(self, op: Op, env: dict[int, torch.Tensor]) -> None:  # noqa: C901
        g = self.g
        ins = op.inputs
        out = op.outputs[0]
        o = op.opts

        if op.type == "CONV_2D":
            env[out] = self._conv(env[ins[0]], env[ins[1]], env[ins[2]], o)

        elif op.type == "DEPTHWISE_CONV_2D":
            env[out] = self._dwconv(env[ins[0]], env[ins[1]], env[ins[2]], o)

        elif op.type == "ADD":
            env[out] = _act(env[ins[0]] + env[ins[1]], o["activation"])

        elif op.type == "PAD":
            env[out] = self._pad(env[ins[0]], g.t(ins[1]).data)

        elif op.type == "RESHAPE":
            env[out] = env[ins[0]].reshape(self._reshape_target(op))

        elif op.type == "RESIZE_BILINEAR":
            size = [int(v) for v in g.t(ins[1]).data.tolist()]
            env[out] = self._resize(env[ins[0]], size, o)

        elif op.type == "MAX_POOL_2D":
            env[out] = self._maxpool(env[ins[0]], o)

        elif op.type == "CONCATENATION":
            env[out] = _act(torch.cat([env[i] for i in ins], dim=o["axis"]), o["activation"])

        elif op.type == "DEPTH_TO_SPACE":
            env[out] = self._depth_to_space(env[ins[0]], o["block_size"])

        elif op.type == "LOGISTIC":
            env[out] = torch.sigmoid(env[ins[0]])

        else:
            raise NotImplementedError(f"op {op.type}")

    # --- op implementations ----------------------------------------------------
    def _conv(self, x, w, b, o) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        w = w.permute(0, 3, 1, 2)  # OHWI -> OIHW
        sh, sw = o["stride"]
        dh, dw = o["dilation"]
        x = self._apply_padding(x, w.shape[2], w.shape[3], sh, sw, dh, dw, o["padding"])
        y = F.conv2d(x, w, b, stride=(sh, sw), dilation=(dh, dw))
        return _act(y, o["activation"]).permute(0, 2, 3, 1)

    def _dwconv(self, x, w, b, o) -> torch.Tensor:
        assert o["depth_multiplier"] == 1, "only depth_multiplier=1 supported"
        c = x.shape[3]
        x = x.permute(0, 3, 1, 2)  # NCHW
        # TFLite depthwise weight: [1, kh, kw, C] -> torch [C, 1, kh, kw]
        w = w[0].permute(2, 0, 1).unsqueeze(1)
        sh, sw = o["stride"]
        dh, dw = o["dilation"]
        x = self._apply_padding(x, w.shape[2], w.shape[3], sh, sw, dh, dw, o["padding"])
        y = F.conv2d(x, w, b, stride=(sh, sw), dilation=(dh, dw), groups=c)
        return _act(y, o["activation"]).permute(0, 2, 3, 1)

    @staticmethod
    def _apply_padding(x, kh, kw, sh, sw, dh, dw, padding) -> torch.Tensor:
        if padding == "valid":
            return x
        ph = _same_pad_1d(x.shape[2], kh, sh, dh)
        pw = _same_pad_1d(x.shape[3], kw, sw, dw)
        # F.pad order for NCHW: (W_before, W_after, H_before, H_after)
        return F.pad(x, (pw[0], pw[1], ph[0], ph[1]))

    @staticmethod
    def _pad(x, paddings) -> torch.Tensor:
        # paddings: [[n0,n1],[h0,h1],[w0,w1],[c0,c1]] for NHWC; F.pad pads last dim first
        p = paddings.astype(int)
        order = (p[3][0], p[3][1], p[2][0], p[2][1], p[1][0], p[1][1], p[0][0], p[0][1])
        return F.pad(x, tuple(int(v) for v in order))

    def _reshape_target(self, op: Op) -> list[int]:
        if len(op.inputs) > 1 and self.g.t(op.inputs[1]).data is not None:
            return [int(v) for v in self.g.t(op.inputs[1]).data.tolist()]
        return [int(v) for v in op.opts["new_shape"]]

    @staticmethod
    def _resize(x, size, o) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        align = o["align_corners"]
        # TF half_pixel_centers -> torch align_corners=False (default sampling grid)
        y = F.interpolate(x, size=tuple(size), mode="bilinear", align_corners=align)
        return y.permute(0, 2, 3, 1)

    def _maxpool(self, x, o) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        fh, fw = o["filter"]
        sh, sw = o["stride"]
        x = self._apply_padding(x, fh, fw, sh, sw, 1, 1, o["padding"])
        y = F.max_pool2d(x, kernel_size=(fh, fw), stride=(sh, sw))
        return y.permute(0, 2, 3, 1)

    @staticmethod
    def _depth_to_space(x, bs) -> torch.Tensor:
        # TFLite DepthToSpace uses DCR ordering: in-channel = (di*bs + dj)*C + k
        # (channel k innermost). This differs from torch pixel_shuffle (CRD). Do it
        # explicitly in NHWC.
        b, h, w, cin = x.shape
        c = cin // (bs * bs)
        x = x.reshape(b, h, w, bs, bs, c)  # (di, dj, k)
        x = x.permute(0, 1, 3, 2, 4, 5)  # [b, h, di, w, dj, k]
        return x.reshape(b, h * bs, w * bs, c)
