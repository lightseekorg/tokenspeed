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
"""CUDA implementation of fused mHC reduction, mixing, and residual apply."""

from __future__ import annotations

import functools
from pathlib import Path

import torch


@functools.cache
def _load_mhc_module():
    import tvm_ffi

    so_path = Path(__file__).parent / "objs" / "mhc_big_fuse" / "mhc_big_fuse.so"
    if not so_path.exists():
        raise RuntimeError(
            f"tokenspeed_kernel mHC library not found at {so_path}. "
            "Run: pip install -e tokenspeed_kernel/python/"
        )
    return tvm_ffi.load_module(str(so_path))


def mhc_big_fuse(
    projection: torch.Tensor,
    square_sum: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    residual: torch.Tensor,
    layer_input: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    hidden_size: int,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    n_splits: int,
    num_tokens: int,
    *,
    norm_weight: torch.Tensor | None,
    norm_eps: float,
    block_size: int,
    enable_pdl: bool,
) -> None:
    """Reduce split-K mHC projections and emit all pre-mapping outputs.

    The input/output tensors are preallocated by the mHC dispatcher.  This
    function replaces the separate pre-mix and residual-apply kernels with one
    CUDA launch. ``block_size`` supports 128, 256, or 512 threads.
    """
    norm_weight_arg = (
        norm_weight
        if norm_weight is not None
        else layer_input.reshape(-1)[:hidden_size]
    )
    _load_mhc_module().mhc_big_fuse(
        projection,
        square_sum,
        hc_scale,
        hc_base,
        residual,
        layer_input,
        post_mix,
        comb_mix,
        int(hidden_size),
        float(rms_eps),
        float(hc_eps),
        int(sinkhorn_iters),
        int(n_splits),
        int(num_tokens),
        norm_weight_arg,
        float(norm_eps),
        norm_weight is not None,
        int(block_size),
        bool(enable_pdl),
    )


# Decode uses 64 splits on GB200. Prefill graph buckets can produce arbitrary
# split counts, so callers must retain the established Triton fallback there.
mhc_big_fuse.supported_n_splits = frozenset({1, 2, 4, 8, 16, 32, 64})
mhc_big_fuse.supports_fused_norm = True
