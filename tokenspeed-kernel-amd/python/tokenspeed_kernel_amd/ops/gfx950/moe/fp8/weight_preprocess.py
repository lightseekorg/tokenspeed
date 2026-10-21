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


"""Prepare one canonical N16/K32 FP8 expert bank for gfx950 MoE kernels."""

from functools import partial

import torch


def pack_fp8_weight_n16k32(weight: torch.Tensor) -> torch.Tensor:
    """Pack ``[expert, output, input]`` FP8 weights into N16/K32 tiles."""
    assert weight.ndim == 3 and weight.dtype == torch.float8_e4m3fn
    experts, output, inputs = weight.shape
    assert output % 16 == 0 and inputs % 32 == 0
    return (
        weight.reshape(experts, output // 16, 16, inputs // 32, 2, 16)
        .permute(0, 1, 3, 4, 2, 5)
        .contiguous()
        .reshape(experts, output, inputs)
    )


def _copy_packed_fp8_shard(dst: torch.Tensor, src: torch.Tensor) -> None:
    # Checkpoint loaders already select the local TP slice and gate/up half.
    # Complete rows compose with N16/K32 tiles; partial input columns do not.
    if not dst.is_contiguous() or dst.shape != src.shape or src.dtype != dst.dtype:
        raise ValueError("packed checkpoint updates require complete local FP8 rows")
    dst.copy_(pack_fp8_weight_n16k32(src.unsqueeze(0))[0])


def preprocess_gluon_fp8_packed_gfx950_moe_weights(
    plan: dict, w: torch.nn.Module
) -> None:
    """Pack loaded experts once, then pack incoming checkpoint shards on copy.

    Weights declare checkpoint loaders; synthetic inputs supply no-op loaders.
    """
    del plan
    assert w.w13_weight is not None and w.w2_weight is not None
    assert w.w13_weight_scale_inv is not None and w.w2_weight_scale_inv is not None
    with torch.no_grad():
        w.w13_weight.copy_(pack_fp8_weight_n16k32(w.w13_weight))
        w.w2_weight.copy_(pack_fp8_weight_n16k32(w.w2_weight))
    # The layer owns the one-time preprocessing guard. Reloads retain the
    # checkpoint loader and graph-captured address without a linear bank.
    w.w13_weight.weight_loader = partial(
        w.w13_weight.weight_loader, weight_copy=_copy_packed_fp8_shard
    )
    w.w2_weight.weight_loader = partial(
        w.w2_weight.weight_loader, weight_copy=_copy_packed_fp8_shard
    )
