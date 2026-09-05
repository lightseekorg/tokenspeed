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
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = ["mhc_fused_hc", "mhc_post", "mhc_pre"]


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float | None = None,
    override: str | None = None,
    solution: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the mHC pre-mapping for one residual stream.

    Args:
        residual: BF16 residual streams shaped ``[..., hc_mult, hidden_size]``.
        fn: FP32 mixing projection shaped
            ``[2 * hc_mult + hc_mult**2, hc_mult * hidden_size]``.
        hc_scale: FP32 pre, post, and combine scales shaped ``[3]``.
        hc_base: FP32 mixing biases shaped
            ``[2 * hc_mult + hc_mult**2]``.
        rms_eps: Epsilon used by the residual RMS normalization.
        hc_eps: Epsilon added during pre-mix and Sinkhorn normalization.
        sinkhorn_iters: Number of Sinkhorn row/column normalization iterations.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.

    Returns:
        A tuple of the BF16 layer input ``[..., hidden_size]``, FP32 post mix
        ``[..., hc_mult, 1]``, and FP32 combine mix
        ``[..., hc_mult, hc_mult]``.
    """
    hc_mult = int(residual.shape[-2])
    hidden_size = int(residual.shape[-1])
    num_tokens = int(residual.numel() // (hc_mult * hidden_size))
    traits = {
        "num_tokens": num_tokens,
        "hc_mult": hc_mult,
        "hidden_size": hidden_size,
        "sinkhorn_iters": int(sinkhorn_iters),
    }
    signature = format_signature(
        residual=dense_tensor_format(residual.dtype),
        fn=dense_tensor_format(fn.dtype),
        hc_scale=dense_tensor_format(hc_scale.dtype),
        hc_base=dense_tensor_format(hc_base.dtype),
    )
    kernel = select_kernel(
        "mhc",
        "pre",
        signature,
        traits=traits,
        override=override,
        solution=solution,
    )
    ShapeCapture.get().record("mhc", "pre", kernel.name, residual.dtype, traits)
    with kernel_scope(
        "mhc",
        "pre",
        residual.dtype,
        kernel_name=kernel.name,
        **traits,
    ):
        return kernel(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_eps,
            sinkhorn_iters,
            norm_weight,
            norm_eps,
        )


def mhc_post(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    override: str | None = None,
    solution: str | None = None,
) -> torch.Tensor:
    """Compute the mHC post-mapping and residual-stream update.

    Args:
        hidden_states: BF16 layer output shaped ``[..., hidden_size]``.
        residual: BF16 residual streams shaped
            ``[..., hc_mult, hidden_size]``.
        post: FP32 post mix shaped ``[..., hc_mult, 1]``.
        comb: FP32 combine mix shaped ``[..., hc_mult, hc_mult]``.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.

    Returns:
        Updated BF16 residual streams with the same shape as ``residual``.
    """
    hc_mult = int(residual.shape[-2])
    hidden_size = int(residual.shape[-1])
    num_tokens = int(residual.numel() // (hc_mult * hidden_size))
    traits = {
        "num_tokens": num_tokens,
        "hc_mult": hc_mult,
        "hidden_size": hidden_size,
    }
    signature = format_signature(
        hidden_states=dense_tensor_format(hidden_states.dtype),
        residual=dense_tensor_format(residual.dtype),
        post=dense_tensor_format(post.dtype),
        comb=dense_tensor_format(comb.dtype),
    )
    kernel = select_kernel(
        "mhc",
        "post",
        signature,
        traits=traits,
        override=override,
        solution=solution,
    )
    ShapeCapture.get().record("mhc", "post", kernel.name, residual.dtype, traits)
    with kernel_scope(
        "mhc",
        "post",
        residual.dtype,
        kernel_name=kernel.name,
        **traits,
    ):
        return kernel(hidden_states, residual, post, comb)


def mhc_fused_hc(
    x_prev: torch.Tensor,
    residual_prev: torch.Tensor,
    post_prev: torch.Tensor,
    comb_prev: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compose the registered previous post-mapping and current pre-mapping.

    Args:
        x_prev: BF16 previous layer output shaped ``[..., hidden_size]``.
        residual_prev: BF16 previous residual streams shaped
            ``[..., hc_mult, hidden_size]``.
        post_prev: FP32 previous post mix shaped ``[..., hc_mult, 1]``.
        comb_prev: FP32 previous combine mix shaped
            ``[..., hc_mult, hc_mult]``.
        fn: FP32 current mixing projection shaped
            ``[2 * hc_mult + hc_mult**2, hc_mult * hidden_size]``.
        hc_scale: FP32 current pre, post, and combine scales shaped ``[3]``.
        hc_base: FP32 current mixing biases shaped
            ``[2 * hc_mult + hc_mult**2]``.
        rms_eps: Epsilon used by the residual RMS normalization.
        hc_eps: Epsilon added during pre-mix and Sinkhorn normalization.
        sinkhorn_iters: Number of Sinkhorn row/column normalization iterations.

    Returns:
        A tuple of the current BF16 residual streams, BF16 layer input, FP32
        post mix, and FP32 combine mix.
    """
    residual_cur = mhc_post(x_prev, residual_prev, post_prev, comb_prev)
    layer_input, post_cur, comb_cur = mhc_pre(
        residual_cur,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
        sinkhorn_iters,
        norm_weight,
        norm_eps,
    )
    return residual_cur, layer_input, post_cur, comb_cur


# Registration side effects must run after the public API is defined.
import tokenspeed_kernel.ops.mhc.deep_gemm  # noqa: E402,F401
import tokenspeed_kernel.ops.mhc.gluon  # noqa: E402,F401
import tokenspeed_kernel.ops.mhc.triton  # noqa: E402,F401
