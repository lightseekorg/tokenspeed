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
quantized per token by the kernel. Rows a padded ``ispp`` adds stay zero with
scale 1. A kernel's own layout (shuffle, row interleave, padding) is its
post-load step's.
"""

from __future__ import annotations

import torch
from torch import nn

from tokenspeed.runtime.layers.moe.types import MoELayerSpec
from tokenspeed.runtime.layers.moe.weights.loaders import make_channel_scale_loader
from tokenspeed.runtime.layers.moe.weights.unquant import create_dense_weight_pair
from tokenspeed.runtime.utils import set_weight_attrs


def create_fp8_channel_weight_pair(spec: MoELayerSpec, layer: nn.Module) -> None:
    ispp = create_dense_weight_pair(spec, layer, params_dtype=torch.float8_e4m3fn)
    w13_weight_scale = torch.nn.Parameter(
        torch.ones(spec.num_local_experts, 2 * ispp, dtype=torch.float32),
        requires_grad=False,
    )
    w2_weight_scale = torch.nn.Parameter(
        torch.ones(spec.num_local_experts, spec.hidden_size, dtype=torch.float32),
        requires_grad=False,
    )
    layer.register_parameter("w13_weight_scale", w13_weight_scale)
    layer.register_parameter("w2_weight_scale", w2_weight_scale)
    scale_loader = make_channel_scale_loader(spec)
    set_weight_attrs(w13_weight_scale, {"weight_loader": scale_loader})
    set_weight_attrs(w2_weight_scale, {"weight_loader": scale_loader})


__all__ = ["create_fp8_channel_weight_pair"]
