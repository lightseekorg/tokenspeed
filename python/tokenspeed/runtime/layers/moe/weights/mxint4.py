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

from __future__ import annotations

import torch
from torch import nn

from tokenspeed.runtime.layers.moe.types import MoELayerSpec
from tokenspeed.runtime.layers.moe.weights.loaders import (
    make_group_scale_loader,
    make_weight_loader,
)
from tokenspeed.runtime.utils import set_weight_attrs

# Eight INT4 values per int32 word (32 // 4); the kernel and its repack are
# INT4-only, so this is a constant, not a config knob.
_PACKED_FACTOR = 8


def _ignore_weight_loader(param, loaded_weight, **kwargs) -> None:
    """No-op loader for checkpoint metadata tensors the kernel does not use."""
    del param, loaded_weight, kwargs


def create_mxint4_weight_pair(
    spec: MoELayerSpec,
    layer: nn.Module,
    *,
    group_size: int,
) -> None:
    """Register per-expert INT4 pack-quantized weights with bf16 group scales.

    Tensors keep the natural ``[out, in // _PACKED_FACTOR]`` checkpoint layout
    (gate/up fused along the output dim), so the shared MoE checkpoint loaders
    fill them without transposition; the trtllm process-weights kernel rewrites
    them into the kernel layout afterwards. Eight INT4 values are packed per
    ``int32`` word, while scales hold one ``bfloat16`` value per ``group_size``
    input elements.
    """
    ispp = spec.intermediate_size // spec.tp_size

    # Fused gate_up_proj (column parallel): [2 * intermediate, hidden // pack].
    w13_weight_packed = torch.nn.Parameter(
        torch.empty(
            spec.num_local_experts,
            2 * ispp,
            spec.hidden_size // _PACKED_FACTOR,
            dtype=torch.int32,
        ),
        requires_grad=False,
    )
    # down_proj (row parallel): [hidden, intermediate // pack].
    w2_weight_packed = torch.nn.Parameter(
        torch.empty(
            spec.num_local_experts,
            spec.hidden_size,
            ispp // _PACKED_FACTOR,
            dtype=torch.int32,
        ),
        requires_grad=False,
    )
    layer.register_parameter("w13_weight_packed", w13_weight_packed)
    layer.register_parameter("w2_weight_packed", w2_weight_packed)

    # Per-group bf16 scales: one scale per ``group_size`` input elements.
    w13_weight_scale = torch.nn.Parameter(
        torch.ones(
            spec.num_local_experts,
            2 * ispp,
            spec.hidden_size // group_size,
            dtype=torch.bfloat16,
        ),
        requires_grad=False,
    )
    w2_weight_scale = torch.nn.Parameter(
        torch.ones(
            spec.num_local_experts,
            spec.hidden_size,
            ispp // group_size,
            dtype=torch.bfloat16,
        ),
        requires_grad=False,
    )
    layer.register_parameter("w13_weight_scale", w13_weight_scale)
    layer.register_parameter("w2_weight_scale", w2_weight_scale)

    weight_loader = make_weight_loader(spec)
    scale_loader = make_group_scale_loader(spec)
    set_weight_attrs(w13_weight_packed, {"weight_loader": weight_loader})
    set_weight_attrs(w2_weight_packed, {"weight_loader": weight_loader})
    set_weight_attrs(w13_weight_scale, {"weight_loader": scale_loader})
    set_weight_attrs(w2_weight_scale, {"weight_loader": scale_loader})

    # compressed-tensors ships a per-proj ``weight_shape`` metadata tensor the
    # kernel ignores. Register absorbers so the loader has a target (it raises
    # on a matched expert tensor with no parameter); dropped in process_weights.
    for shape_name in ("w13_weight_shape", "w2_weight_shape"):
        shape_param = torch.nn.Parameter(
            torch.empty(spec.num_local_experts, 2, dtype=torch.int32),
            requires_grad=False,
        )
        layer.register_parameter(shape_name, shape_param)
        set_weight_attrs(shape_param, {"weight_loader": _ignore_weight_loader})


__all__ = ["create_mxint4_weight_pair"]
