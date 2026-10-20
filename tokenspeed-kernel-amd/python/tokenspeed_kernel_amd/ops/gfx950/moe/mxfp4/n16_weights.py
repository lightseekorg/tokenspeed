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

"""Single-bank N16/K64-byte MXFP4 expert storage for gfx950 MoE pipelines.

Each 16-row group of a weight stores its K bytes as 64-byte chunks, each a
1 KiB block in MFMA lane order, so one wave reads its fragment as one
contiguous 1 KiB load. W13 alternates 16-row gate and up groups. W2 keeps
its rows' chunks back to back; a K width that is not a multiple of 64
bytes ends in a half chunk holding only the leading lane quads, so W2 is
stored flat (``[E, D / 16, 16 * I / 2]``) without padding. Scales of two
16-row groups and two K groups share a 32-bit word (``_n32_scale_offset``);
W2 scales pad K to whole 8-group cells.

Users: the MXFP8 SiTU prefill/decode pipelines and the dynamic-MXFP4
(A4W4) pipeline in ``mxfp4_gemm.py``.
"""

from __future__ import annotations

from functools import partial

import torch
from tokenspeed_kernel_amd._triton import gluon


@gluon.jit
def _n16_weight_offset(n, k_packed, packed_k):
    """Byte offset of packed K byte ``k_packed`` of row ``n``."""
    # Whole chunks first, then the half-chunk tail of a row group. Keep this
    # form: with a runtime ``packed_k`` the single ``n // 16 * packed_k * 16``
    # product made the SiTU warp decode about 10% slower.
    return (
        (n // 16 * (packed_k // 64) + k_packed // 64) * 1024
        + n // 16 * (packed_k % 64 * 16)
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
        # Flat rows of chunks; a trailing half chunk keeps its two leading
        # lane quads only.
        chunks = (packed_k + 63) // 64
        weight = torch.nn.functional.pad(weight, (0, chunks * 64 - packed_k))
        return (
            weight.reshape(experts, columns // 16, 16, chunks, 4, 16)
            .permute(0, 1, 3, 4, 2, 5)
            .reshape(experts, columns // 16, chunks * 1024)[..., : 16 * packed_k]
            .contiguous()
        )
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
        # Pad K to whole 8-group cells with unit scales.
        cells = (groups + 7) // 8
        scale = torch.nn.functional.pad(scale, (0, cells * 8 - groups), value=127)
        groups = cells * 8
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
    """Validate packed cell axes and return (experts, hidden, intermediate).

    Requires ``hidden % 256 == 0`` and ``intermediate % 64 == 0``; pipelines
    with stricter K tiling check their own multiples.
    """
    tensors = (w13, s13, w2, s2)
    if any(t.dtype != torch.uint8 or not t.is_contiguous() for t in tensors) or tuple(
        t.ndim for t in tensors
    ) != (6, 6, 3, 6):
        raise ValueError(
            "N16 MXFP4 requires contiguous uint8 cells: rank-6 W13 and scales, "
            "rank-3 W2"
        )
    e, d, i = w13.shape[0], w13.shape[2] * 128, w13.shape[1] * 8
    if (
        not e
        or not d
        or not i
        or d % 256
        or i % 64
        or w13.shape != (e, i // 8, d // 128, 4, 16, 16)
        or s13.shape != (e, i // 16, d // 256, 4, 16, 4)
        or w2.shape != (e, d // 16, 8 * i)
        or s2.shape != (e, d // 32, (i + 255) // 256, 4, 16, 4)
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
    linear_k: int,
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
    elif gate_up:
        _, n, k, _, _, _ = cells.shape
        cells = cells.reshape(1, n // 2, 2, k, 4, 16, 16)
        linear = (
            cells.permute(0, 2, 1, 5, 3, 4, 6).contiguous().reshape(1, n * 16, k * 64)
        )
    else:
        _, n, size = cells.shape
        chunks = (linear_k + 63) // 64
        cells = torch.nn.functional.pad(cells, (0, chunks * 1024 - size))
        linear = (
            cells.reshape(1, n, chunks, 4, 16, 16)
            .permute(0, 1, 4, 2, 3, 5)
            .reshape(1, n * 16, chunks * 64)
        )
    # Drop the K padding of the last W2 chunk or scale cell.
    linear = linear[..., :linear_k].contiguous()
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
        or intermediate % 64
        or w2.shape != (experts, hidden, intermediate // 2)
        or s13.shape != (experts, two_i, hidden // 32)
        or s2.shape != (experts, hidden, intermediate // 32)
    ):
        raise ValueError(
            "N16 MXFP4 requires D divisible by 256, I by 64 and matching scales"
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
                linear_k=int(original.shape[-1]),
            )
        setattr(w, name, parameter)
