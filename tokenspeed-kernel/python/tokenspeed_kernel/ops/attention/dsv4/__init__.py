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


def _attention_format_signature(**roles: torch.Tensor):
    return format_signature(
        **{role: dense_tensor_format(tensor.dtype) for role, tensor in roles.items()}
    )


# ===-----------------------------------------------------------------------===#
# DSv4 Kernels
# ===-----------------------------------------------------------------------===#


def dsv4_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lens: torch.Tensor,
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    override: str | None = None,
    solution: str | None = None,
    lse: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run DeepSeek V4 selected attention over a dense K/V workspace.

    Args:
        q: BF16 queries shaped ``[tokens, heads, 512]``.
        kv: BF16 selected K/V workspace with rows of width 512.
        indices: Selected workspace row indices shaped ``[tokens, width]``.
            Negative entries are ignored. Nonnegative entries in each active
            prefix must be smaller than the number of rows in ``kv``.
        lens: Valid selected width for each query token.
        attn_sink: One sink logit per head, or None for a no-sink partial.
        softmax_scale: Scale applied to query-key dot products.
        out: Optional output shaped like ``q``.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.
        lse: Optional FP32 [tokens, heads] output for natural-log LSE.
            Partial output requires attn_sink=None; the sink is applied at merge.

    Returns:
        BF16 attention output shaped like ``q``.
    """
    if q.ndim != 3 or q.shape[0] < 1 or q.shape[-1] != 512:
        raise ValueError(
            f"q must have shape [tokens, heads, 512], got {tuple(q.shape)}"
        )
    if kv.ndim < 2 or kv.shape[-1] != 512:
        raise ValueError(f"kv must contain rows of width 512, got {tuple(kv.shape)}")
    tokens = int(q.shape[0])
    if indices.ndim != 2 or indices.shape[0] != tokens:
        raise ValueError("indices must have one row per query token")
    if lens.ndim != 1 or lens.numel() != tokens:
        raise ValueError("lens must have one entry per query token")
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("indices must have dtype int32 or int64")
    if lens.dtype not in (torch.int32, torch.int64):
        raise TypeError("lens must have dtype int32 or int64")
    if attn_sink is not None and attn_sink.numel() < q.shape[1]:
        raise ValueError("attn_sink must provide one value per query head")
    if any(
        tensor is not None and tensor.device != q.device
        for tensor in (kv, indices, lens, attn_sink, lse)
    ):
        raise ValueError("all selected-attention tensors must share a device")
    if out is not None and (
        out.shape != q.shape or out.dtype != q.dtype or out.device != q.device
    ):
        raise ValueError("out must match q shape, dtype, and device")

    if lse is not None:
        if attn_sink is not None:
            raise ValueError("partial attention LSE requires no sink")
        if (
            lse.shape != q.shape[:2]
            or lse.dtype != torch.float32
            or not lse.is_contiguous()
        ):
            raise ValueError("lse must be contiguous FP32 [tokens, heads]")

    signature = _attention_format_signature(q=q, kv=kv)
    traits = {
        "num_q_heads": int(q.shape[1]),
        "head_dim": int(q.shape[-1]),
        "selected_width": int(indices.shape[-1]),
        "cache_layout": "dense_workspace",
        "metadata_dtypes": frozenset({indices.dtype, lens.dtype}),
        "sinks": attn_sink is not None,
    }
    if lse is not None:
        traits["return_lse"] = True
    kernel = select_kernel(
        "attention",
        "dsv4_prefill",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )
    shape_params = {
        "tokens": int(q.shape[0]),
        "num_heads": int(q.shape[1]),
        "head_dim": int(q.shape[2]),
        "selected_width": int(indices.shape[-1]),
        "kv_rows": int(kv.numel() // q.shape[-1]),
    }
    ShapeCapture.get().record(
        "attention",
        "dsv4_prefill",
        kernel.name,
        q.dtype,
        shape_params,
    )
    with kernel_scope(
        "attention",
        "dsv4_prefill",
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            q=q,
            kv=kv,
            indices=indices,
            lens=lens,
            attn_sink=attn_sink,
            softmax_scale=softmax_scale,
            out=out,
            **({"lse": lse} if lse is not None else {}),
        )


# Backend registration (side-effect imports)
# isort: off
import tokenspeed_kernel.ops.attention.dsv4.cuda  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.dsv4.triton  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.dsv4.gluon  # noqa: E402,F401

# isort: on

__all__ = [
    "dsv4_prefill",
]
