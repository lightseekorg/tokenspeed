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

"""Single-bank N16/K64-byte storage for the MXFP8 prefill pipeline."""

from __future__ import annotations

from functools import partial

import torch
from tokenspeed_kernel_amd._triton import gluon


@gluon.jit
def _n16_weight_offset(n, k_packed, packed_k):
    return (
        (n // 16 * (packed_k // 64) + k_packed // 64) * 1024
        + (k_packed // 16 % 4) * 256
        + n % 16 * 16
        + k_packed % 16
    )


@gluon.jit
def _n32_scale_offset(n, group, groups):
    return (
        n // 32 * (groups * 32)
        + group // 8 * 256
        + group % 4 * 64
        + n % 16 * 4
        + (group % 8) // 4 * 2
        + (n % 32) // 16
    )


def _shuffle_weight(weight: torch.Tensor, *, gate_up: bool) -> torch.Tensor:
    experts, columns, packed_k = weight.shape
    if gate_up:
        weight = weight.reshape(experts, 2, columns // 32, 16, packed_k // 64, 4, 16)
        order = (0, 2, 1, 4, 5, 3, 6)
    else:
        weight = weight.reshape(experts, columns // 16, 16, packed_k // 64, 4, 16)
        order = (0, 1, 3, 4, 2, 5)
    return (
        weight.permute(order)
        .contiguous()
        .reshape(experts, columns // 16, packed_k // 64, 4, 16, 16)
    )


def _shuffle_scale(scale: torch.Tensor, *, gate_up: bool) -> torch.Tensor:
    experts, columns, groups = scale.shape
    if gate_up:
        scale = scale.reshape(experts, 2, columns // 32, 16, groups // 8, 2, 4)
        order = (0, 2, 4, 6, 3, 5, 1)
    else:
        scale = scale.reshape(experts, columns // 32, 2, 16, groups // 8, 2, 4)
        order = (0, 1, 4, 6, 3, 5, 2)
    return (
        scale.permute(order)
        .contiguous()
        .reshape(experts, columns // 32, groups // 8, 4, 16, 4)
    )


def n16_mxfp4_shape(
    w13: torch.Tensor,
    s13: torch.Tensor,
    w2: torch.Tensor,
    s2: torch.Tensor,
) -> tuple[int, int, int]:
    """Validate packed cell axes and return (experts, hidden, intermediate)."""
    tensors = (w13, s13, w2, s2)
    if any(
        t.ndim != 6 or t.dtype != torch.uint8 or not t.is_contiguous() for t in tensors
    ):
        raise ValueError("N16 MXFP4 requires contiguous rank-6 uint8 cells")
    e, d, i = w13.shape[0], w13.shape[2] * 128, w13.shape[1] * 8
    if (
        not e
        or not d
        or not i
        or d % 256
        or i % 256
        or w13.shape != (e, i // 8, d // 128, 4, 16, 16)
        or s13.shape != (e, i // 16, d // 256, 4, 16, 4)
        or w2.shape != (e, d // 16, i // 128, 4, 16, 16)
        or s2.shape != (e, d // 32, i // 256, 4, 16, 4)
        or len({t.device for t in tensors}) != 1
    ):
        raise ValueError("inconsistent N16 weight/scale cell shapes")
    return e, d, i


def _load_n16_expert(
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    shard_id: str,
    local_expert_id: int,
    *,
    loader,
    gate_up: bool,
    is_scale: bool,
    interleaved: bool,
) -> None:
    # Reuse checkpoint sharding/conversion on one temporary linear expert.
    # Copy back into the existing bank so captured pointers remain valid.
    cells = param[local_expert_id : local_expert_id + 1]
    if cells.shape[0] != 1:
        raise ValueError("N16 checkpoint update requires a local expert")
    if is_scale:
        _, n, k, _, _, _ = cells.shape
        cells = cells.reshape(1, n, k, 4, 16, 2, 2)
        order = (0, 6, 1, 4, 2, 5, 3) if gate_up else (0, 1, 6, 4, 2, 5, 3)
        linear = cells.permute(order).contiguous().reshape(1, n * 32, k * 8)
    else:
        _, n, k, _, _, _ = cells.shape
        if gate_up:
            cells = cells.reshape(1, n // 2, 2, k, 4, 16, 16)
            order = (0, 2, 1, 5, 3, 4, 6)
        else:
            order = (0, 1, 4, 2, 3, 5)
        linear = cells.permute(order).contiguous().reshape(1, n * 16, k * 64)
    if interleaved:
        linear = (
            linear.reshape(1, 2, -1, linear.shape[-1])
            .transpose(1, 2)
            .reshape_as(linear)
        )
    linear = torch.nn.Parameter(linear, requires_grad=False)
    linear.__dict__.update(param.__dict__)
    linear.weight_loader = loader
    loader(linear, loaded_weight, shard_id=shard_id, local_expert_id=0)
    if interleaved:
        linear = (
            linear.reshape(1, -1, 2, linear.shape[-1])
            .transpose(1, 2)
            .reshape_as(linear)
        )
    shuffle = _shuffle_scale if is_scale else _shuffle_weight
    param.data[local_expert_id : local_expert_id + 1].copy_(
        shuffle(linear, gate_up=gate_up)
    )


def preprocess_n16_mxfp4_weights(w: torch.nn.Module) -> None:
    """Replace linear expert tensors with one N-major N16 bank in place."""
    names = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
    tensors = tuple(getattr(w, name) for name in names)
    if any(t.ndim == 6 for t in tensors):
        raise ValueError("N16 MXFP4 weights were already prepared")
    if any(
        t.dtype != torch.uint8 or t.ndim != 3 or not t.is_contiguous() for t in tensors
    ):
        raise ValueError(
            "N16 MXFP4 preparation requires contiguous rank-3 uint8 tensors"
        )
    w13, s13, w2, s2 = tensors
    experts, two_i, packed_d = w13.shape
    intermediate, hidden = two_i // 2, packed_d * 2
    if (
        two_i % 2
        or hidden % 256
        or intermediate % 256
        or w2.shape != (experts, hidden, intermediate // 2)
        or s13.shape != (experts, two_i, hidden // 32)
        or s2.shape != (experts, hidden, intermediate // 32)
    ):
        raise ValueError(
            "N16 MXFP4 requires D and I divisible by 256 and matching scales"
        )
    if len({t.device for t in tensors}) != 1:
        raise ValueError("N16 MXFP4 weights and scales must be colocated")
    input_layout = getattr(w, "w13_input_layout", "concatenated")
    if input_layout not in {"concatenated", "stacked", "interleaved"}:
        raise ValueError("unsupported W13 input layout")
    if input_layout == "interleaved":
        w13 = (
            w13.reshape(experts, intermediate, 2, packed_d)
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape_as(w13)
        )
        s13 = (
            s13.reshape(experts, intermediate, 2, hidden // 32)
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape_as(s13)
        )
    prepared = (
        _shuffle_weight(w13, gate_up=True),
        _shuffle_scale(s13, gate_up=True),
        _shuffle_weight(w2, gate_up=False),
        _shuffle_scale(s2, gate_up=False),
    )
    for name, tensor, original in zip(names, prepared, tensors, strict=True):
        parameter = torch.nn.Parameter(tensor, requires_grad=False)
        parameter.__dict__.update(original.__dict__)
        if hasattr(original, "weight_loader"):
            gate_up = name.startswith("w13")
            parameter.weight_loader = partial(
                _load_n16_expert,
                loader=original.weight_loader,
                gate_up=gate_up,
                is_scale=name.endswith("scale"),
                interleaved=gate_up and input_layout == "interleaved",
            )
        setattr(w, name, parameter)
