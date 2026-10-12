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

"""FP8 experts with one FP32 scale per output channel (``fp8_channel``).

The logical layout a per-channel FP8 MoE kernel consumes, as loaded:

- ``w13_weight`` FP8 E4M3 ``[experts, 2 * ispp, hidden]``: gate rows, then up
  rows, of this rank's intermediate partition.
- ``w13_weight_scale`` FP32 ``[experts, 2 * ispp]``: the scale of each w13 row.
- ``w2_weight`` FP8 E4M3 ``[experts, hidden, ispp]``.
- ``w2_weight_scale`` FP32 ``[experts, hidden]``: the scale of each w2 row (all
  hidden channels on every rank).

A row dequantizes to ``weight[e, n, :] * scale[e, n]``. Activations are
quantized per token by the kernel. The scales start as NaN, so a row the
checkpoint did not fill fails ``check_fp8_channel_scales`` after loading
instead of running with a made-up scale; ``--load-format dummy`` sets them to 1
before that check. A kernel's own layout (shuffle, row interleave, padding) is
its post-load step's.
"""

from __future__ import annotations

import torch
from torch import nn

from tokenspeed.runtime.layers.moe.types import MoELayerSpec
from tokenspeed.runtime.layers.moe.weights.loaders import make_channel_scale_loader
from tokenspeed.runtime.layers.moe.weights.unquant import create_dense_weight_pair
from tokenspeed.runtime.utils import set_weight_attrs

_SCALE_NAMES = ("w13_weight_scale", "w2_weight_scale")


def _unit_scales(param: torch.Tensor, generator: torch.Generator) -> None:
    """Dummy loading has no checkpoint to fill the NaN scales from."""
    param.data.fill_(1.0)


def create_fp8_channel_weight_pair(spec: MoELayerSpec, layer: nn.Module) -> None:
    ispp = create_dense_weight_pair(spec, layer, params_dtype=torch.float8_e4m3fn)
    rows = {"w13_weight_scale": 2 * ispp, "w2_weight_scale": spec.hidden_size}
    for name in _SCALE_NAMES:
        scale = torch.nn.Parameter(
            torch.full(
                (spec.num_local_experts, rows[name]),
                float("nan"),
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter(name, scale)
        set_weight_attrs(
            scale,
            {
                "weight_loader": make_channel_scale_loader(spec, name),
                "dummy_initializer": _unit_scales,
            },
        )


def check_fp8_channel_scales(layer: nn.Module, prefix: str) -> None:
    """Refuse expert scales the checkpoint did not fill (still NaN) or filled
    with a non-finite value, before a kernel's post-load step consumes them."""
    for name in _SCALE_NAMES:
        scale = getattr(layer, name)
        bad = ~torch.isfinite(scale)
        if bool(bad.any()):
            expert, row = (int(i) for i in bad.nonzero()[0])
            raise ValueError(
                f"{prefix}: {name} has {int(bad.sum())} rows that the checkpoint "
                f"did not load or loaded as NaN/inf (first: local expert "
                f"{expert}, row {row}). FP8 per-channel experts need one finite "
                f"scale per output row."
            )


__all__ = ["check_fp8_channel_scales", "create_fp8_channel_weight_pair"]
