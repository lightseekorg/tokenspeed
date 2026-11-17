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


"""The relative-attention gate/sink epilogue in one kernel."""

from __future__ import annotations

import torch
from tokenspeed_kernel._triton import libdevice, tl, triton
from tokenspeed_kernel.platform import pdl_enabled


@triton.jit
def _sigmoid(x):
    """On NVIDIA the reference kernels' ``rcp.approx(ex2.approx(-x * log2(e))
    + 1)``, subnormals kept; elsewhere PyTorch's ``1 / (1 + expf(-x))``."""
    if tl.target_info.is_cuda():
        exp = tl.inline_asm_elementwise(
            "ex2.approx.f32 $0, $1;",
            "=f,f",
            [-x * 1.4426950408889634],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        y = tl.inline_asm_elementwise(
            "rcp.approx.f32 $0, $1;",
            "=f,f",
            [exp + 1.0],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    else:
        y = tl.div_rn(1.0, 1.0 + libdevice.exp(-x))
    return y


@triton.jit
def _gate_sink_head(
    out_ptr,
    attn_ptr,
    gate_ptr,
    lse_ptr,
    sink_ptr,
    token,
    head,
    stride_out,
    stride_attn,
    stride_gate,
    stride_lse,
    stride_sink,
    HEAD_DIM: tl.constexpr,
):
    col = head * HEAD_DIM + tl.arange(0, HEAD_DIM)
    attn = tl.load(attn_ptr + token * stride_attn + col)
    gate = tl.load(gate_ptr + token * stride_gate + col).to(tl.float32)
    lse = tl.load(lse_ptr + token * stride_lse + head).to(tl.float32)
    sink = tl.load(sink_ptr + token * stride_sink + head).to(tl.float32)
    gated = attn * _sigmoid(gate)
    gated = gated * _sigmoid(lse - sink)
    tl.store(out_ptr + token * stride_out + col, gated.to(out_ptr.dtype.element_ty))


@triton.jit(
    do_not_specialize=[
        "tokens",
        "stride_out",
        "stride_attn",
        "stride_gate",
        "stride_lse",
        "stride_sink",
    ]
)
def _gate_sink_kernel(
    out_ptr,
    attn_ptr,
    gate_ptr,
    lse_ptr,
    sink_ptr,
    tokens,
    stride_out,
    stride_attn,
    stride_gate,
    stride_lse,
    stride_sink,
    HEAD_DIM: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
):
    if ENABLE_PDL:
        # Let the output projection prefetch its weights, then wait for the attention.
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
    # A program walks tokens of one head, so a prefill chunk needs only a few waves.
    head = tl.program_id(1)
    strides = (stride_out, stride_attn, stride_gate, stride_lse, stride_sink)
    _gate_sink_head(
        out_ptr,
        attn_ptr,
        gate_ptr,
        lse_ptr,
        sink_ptr,
        tl.program_id(0).to(tl.int64),
        head,
        *strides,
        HEAD_DIM,
    )
    for token in range(
        tl.program_id(0) + tl.num_programs(0), tokens, tl.num_programs(0)
    ):
        _gate_sink_head(
            out_ptr,
            attn_ptr,
            gate_ptr,
            lse_ptr,
            sink_ptr,
            token.to(tl.int64),
            head,
            *strides,
            HEAD_DIM,
        )


# Token programs per head: decode batches keep one program per token.
_TOKEN_PROGRAMS = 256


def rel_mha_gate_sink(
    attention: torch.Tensor,
    gate: torch.Tensor,
    lse: torch.Tensor,
    sink: torch.Tensor,
    *,
    head_dim: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """``(attention * sigmoid(gate)) * sigmoid(lse - sink)`` in fp32, stored
    once in ``out_dtype``.

    Under PDL the next kernel may start before this one's predecessor
    finishes; it must read only constant operands before its own wait.

    Args:
        attention: FP32 ``[tokens, heads * head_dim]`` attention output.
        gate: ``[tokens, heads * head_dim]`` gate rows; a strided row view is
            fine as long as the columns are contiguous.
        lse: ``[tokens, heads]`` natural-log softmax statistics.
        sink: ``[tokens, heads]`` sink logits, columns contiguous.
        head_dim: Channels per head.
        out_dtype: Dtype of the gated rows.

    Returns:
        ``[tokens, heads * head_dim]`` gated rows in ``out_dtype``.
    """
    tokens, width = attention.shape
    heads = width // head_dim
    if (
        attention.dtype != torch.float32
        or attention.stride(-1) != 1
        or gate.shape != (tokens, width)
        or gate.stride(-1) != 1
        or lse.shape != (tokens, heads)
        or lse.stride(-1) != 1
        or sink.shape != (tokens, heads)
        or sink.stride(-1) != 1
    ):
        raise ValueError("rel_mha_gate_sink needs [T, H*D] rows and [T, H] scalars")
    if not attention.is_cuda:
        gated = attention.view(tokens, heads, head_dim) * torch.sigmoid(
            gate.view(tokens, heads, head_dim).float()
        )
        gated = gated * torch.sigmoid(lse.float() - sink.float()).unsqueeze(-1)
        return gated.flatten(1).to(out_dtype)
    out = torch.empty((tokens, width), dtype=out_dtype, device=attention.device)
    if tokens:
        _gate_sink_kernel[(min(tokens, _TOKEN_PROGRAMS), heads)](
            out,
            attention,
            gate,
            lse,
            sink,
            tokens,
            out.stride(0),
            attention.stride(0),
            gate.stride(0),
            lse.stride(0),
            sink.stride(0),
            HEAD_DIM=head_dim,
            ENABLE_PDL=pdl_enabled(),
            # Each product rounds alone, as in the reference.
            enable_fp_fusion=False,
            **({"launch_pdl": True} if pdl_enabled() else {}),
        )
    return out
