# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Packed INT4 expert checkpoints and their required FlatQuant parameters."""

from __future__ import annotations

import math

import torch
from torch import nn


def flatquant_factors(width: int) -> tuple[int, int]:
    """Return the closest ascending factor pair used by FlatQuant checkpoints."""
    if width <= 0:
        raise ValueError("FlatQuant width must be positive")
    for left in range(math.isqrt(width), 0, -1):
        if width % left == 0:
            return left, width // left
    raise AssertionError("positive integers have a factor pair")


class FlatQuantParameters(nn.Module):
    """Required checkpoint tensors; never initialize missing transforms to identity."""

    def __init__(self, shapes: dict[str, tuple[int, ...]], *, dtype: torch.dtype):
        super().__init__()
        self.loaded: set[str] = set()
        for name, shape in shapes.items():
            param = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
            self.register_parameter(name, param)
            param.weight_loader = self._loader(name)

    def _loader(self, name):
        def load(param, value):
            if value.shape != param.shape or not value.is_floating_point():
                raise ValueError(
                    f"Invalid FlatQuant {name}: expected {tuple(param.shape)}"
                )
            if name in self.loaded:
                raise ValueError(f"Duplicate FlatQuant parameter {name}")
            param.data.copy_(value)
            self.loaded.add(name)

        return load

    def validate_loaded(self) -> None:
        missing = set(self._parameters) - self.loaded
        if missing:
            raise ValueError(f"Missing FlatQuant parameters: {sorted(missing)}")


def create_int4_weight_pair(
    spec, layer: nn.Module, *, transposed: bool, scale_strategy: str
) -> int:
    """Create EP-local W4A4 weights, scales and replicated FlatQuant matrices.

    Checkpoint weights pack eight signed nibbles per INT32. With transposed
    GMM they are [E,N,K/8]; otherwise [E,N/8,K], transposed once at post-load.
    FlatQuant-2 transforms the complete intermediate row, so TP splitting is
    unsupported; EP expert sharding does not split either transform.
    """
    if spec.tp_size != 1 or spec.tp_rank != 0:
        raise ValueError("FlatQuant W4A4 requires expert TP=1; use expert parallelism")
    if type(transposed) is not bool or scale_strategy not in {"channel", "tensor"}:
        raise ValueError("Invalid W4A4 layout or scale strategy")
    h, i, e = spec.hidden_size, spec.intermediate_size, spec.num_local_experts
    if min(h, i, e) <= 0 or h % 8 or i % 8:
        raise ValueError("Packed W4A4 requires positive H/I divisible by eight")
    layer.w4a4_gmm_is_transposed = transposed
    layer.w4a4_scale_strategy = scale_strategy
    layer._int4_loaded: set[tuple[str, int, str]] = set()
    layer._int4_plan = None
    shapes = {
        "w13_weight": (e, 2 * i, h // 8) if transposed else (e, 2 * i // 8, h),
        "w2_weight": (e, h, i // 8) if transposed else (e, h // 8, i),
        "w13_weight_scale": (e, 2 * i),
        "w2_weight_scale": (e, h),
    }

    def make_loader(name):
        def load(param, value, shard_id, local_expert_id):
            expected = {"w2"} if name.startswith("w2_") else {"w1", "w3"}
            if shard_id not in expected or not 0 <= local_expert_id < e:
                raise ValueError("Invalid W4A4 expert/shard destination")
            key = (name, local_expert_id, shard_id)
            if key in layer._int4_loaded:
                raise ValueError(f"Duplicate W4A4 checkpoint shard {key}")
            target = param.data[local_expert_id]
            if shard_id != "w2":
                half = target.shape[0] // 2
                target = target.narrow(0, half if shard_id == "w3" else 0, half)
            if name.endswith("_scale"):
                count = 1 if scale_strategy == "tensor" else target.numel()
                if value.numel() != count or not value.is_floating_point():
                    raise ValueError(
                        f"Invalid W4A4 {name}: expected {count} scale values"
                    )
                value = value.reshape(-1).expand_as(target)
            elif value.dtype != torch.int32 or value.shape != target.shape:
                raise ValueError(
                    f"Invalid packed W4A4 {name}: expected INT32 {tuple(target.shape)}"
                )
            target.copy_(value)
            layer._int4_loaded.add(key)

        return load

    for name, shape in shapes.items():
        dtype = torch.float32 if name.endswith("_scale") else torch.int32
        param = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
        layer.register_parameter(name, param)
        param.weight_loader = make_loader(name)
    for stage, width in ((1, h), (2, i)):
        left, right = flatquant_factors(width)
        layer.add_module(
            f"w{stage}_trans",
            FlatQuantParameters(
                {"matrix_left": (left, left), "matrix_right": (right, right)},
                dtype=torch.bfloat16,
            ),
        )
        layer.add_module(
            f"act_quantizer_w{stage}",
            FlatQuantParameters({"clip_factor_a": (1,)}, dtype=torch.float32),
        )
    return i


def validate_int4_weights_loaded(layer: nn.Module) -> None:
    """Reject partial checkpoints before weight preparation or graph capture."""
    expected = {
        (name, expert, shard)
        for name, shards in (
            ("w13_weight", ("w1", "w3")),
            ("w2_weight", ("w2",)),
            ("w13_weight_scale", ("w1", "w3")),
            ("w2_weight_scale", ("w2",)),
        )
        for expert in range(layer.w13_weight.shape[0])
        for shard in shards
    }
    if missing := expected - layer._int4_loaded:
        raise ValueError(f"Missing W4A4 checkpoint shards: {sorted(missing)[:8]}")
    for module in (
        layer.w1_trans,
        layer.w2_trans,
        layer.act_quantizer_w1,
        layer.act_quantizer_w2,
    ):
        module.validate_loaded()
