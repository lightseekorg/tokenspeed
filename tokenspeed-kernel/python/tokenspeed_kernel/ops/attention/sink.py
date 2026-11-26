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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Per-head learnable attention sink rescaling from natural-log LSE.

Given standard attention output ``O = softmax(logits) @ V`` and natural-log
``lse = logsumexp(logits, dim=-1)``, adding one virtual sink key with score
``sink[h]`` and value ``0`` scales each row identically by
``sigmoid(lse - sink[h])``. When a row is completely masked (``lse = -inf``),
``sigmoid(-inf - sink) = 0``, keeping empty rows zero.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import tl, triton


@triton.jit(do_not_specialize=["num_tokens"])
def _apply_attention_sink_kernel(
    out_ptr,
    lse_ptr,
    sink_ptr,
    num_tokens,
    stride_ot,
    stride_oh,
    stride_lt,
    stride_lh,
    DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    if token >= num_tokens:
        return

    lse = tl.load(lse_ptr + token * stride_lt + head * stride_lh).to(tl.float32)
    sink = tl.load(sink_ptr + head).to(tl.float32)
    scale = tl.sigmoid(lse - sink)

    offs = tl.arange(0, BLOCK_D)
    base = token * stride_ot + head * stride_oh
    for start in range(0, DIM, BLOCK_D):
        cols = start + offs
        mask = cols < DIM
        val = tl.load(out_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(
            out_ptr + base + cols,
            (val * scale).to(out_ptr.dtype.element_ty),
            mask=mask,
        )


def apply_attention_sink(
    output: torch.Tensor,
    lse: torch.Tensor,
    sink: torch.Tensor,
    *,
    inplace: bool = False,
) -> torch.Tensor:
    """Apply learnable per-head attention sink rescaling ``O * sigmoid(lse - sink)``.

    Args:
        output: Attention output ``[T, H, D]`` (BF16/FP16/FP32).
        lse: Natural-log logsumexp ``[T, H]`` (FP32).
        sink: Per-head sink logits ``[H]`` (any floating dtype; evaluated in FP32).
        inplace: Whether to mutate ``output`` in place when possible.

    Returns:
        Rescaled tensor with the same shape and dtype as ``output``.
    """
    if output.ndim != 3:
        raise ValueError(
            f"apply_attention_sink expects output [T, H, D], got {tuple(output.shape)}"
        )
    num_tokens, num_heads, dim = output.shape
    lse_2d = lse.reshape(num_tokens, num_heads)
    if sink.ndim != 1 or sink.numel() < num_heads:
        raise ValueError(
            f"apply_attention_sink expects sink with >= {num_heads} elements, "
            f"got {tuple(sink.shape)}"
        )
    if num_tokens == 0:
        return output if inplace else output.clone()

    if not output.is_cuda or output.stride(-1) != 1:
        scale = torch.sigmoid(
            lse_2d.float() - sink[:num_heads].float().unsqueeze(0)
        ).unsqueeze(-1)
        rescaled = (output.float() * scale).to(output.dtype)
        if inplace:
            output.copy_(rescaled)
            return output
        return rescaled

    target = output if inplace else output.clone()
    sink_contig = sink[:num_heads].contiguous()
    block_d = min(512, triton.next_power_of_2(dim))
    _apply_attention_sink_kernel[(num_tokens, num_heads)](
        target,
        lse_2d,
        sink_contig,
        num_tokens,
        target.stride(0),
        target.stride(1),
        lse_2d.stride(0),
        lse_2d.stride(1),
        DIM=dim,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return target


__all__ = ["apply_attention_sink"]
