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

import torch
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature


def swa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    softmax_scale: float,
    *,
    window_left: int,
    solution: str | None,
) -> torch.Tensor:
    """Dots3 causal sliding-window prefill with bottom-right aligned queries.

    Args:
        q: Queries shaped [total_q, num_q_heads, 256], in BF16 or FP16.
        k: Keys shaped [total_kv, num_kv_heads, 256], with the same dtype as q.
        v: Values shaped [total_kv, num_kv_heads, 128], with the same dtype as q.
            The query head count must be divisible by the KV head count.
            Q/K/V must have contiguous last dimensions.
        cu_seqlens_q: Contiguous GPU int32/int64 cumulative query lengths,
            shaped [batch + 1], starting at zero and ending at total_q.
        cu_seqlens_kv: Corresponding cumulative KV lengths, ending at total_kv.
        max_seqlen_q: Upper bound on each request's query length (launch grid).
        max_seqlen_kv: Upper bound on each request's KV length. Actual lengths
            are read from cu_seqlens_kv on the GPU.
        softmax_scale: Scale applied to QK logits before softmax.
        window_left: Non-negative number of previous keys visible in addition
            to the current key. Query i has position kv_len - q_len + i.
            Queries without visible keys produce zero. Prefix-only chunks
            whose queries are not a suffix of the keys are not supported.
        solution: Kernel solution to select ("triton"), or None for selection.

    Returns:
        Attention output shaped [total_q, num_q_heads, 128], with q's dtype.
        This operation is always causal and does not return LSE.
    """
    if window_left < 0:
        raise ValueError("window_left must be non-negative for SWA prefill")
    signature = format_signature(
        q=dense_tensor_format(q.dtype),
        k=dense_tensor_format(k.dtype),
        v=dense_tensor_format(v.dtype),
    )
    kernel = select_kernel(
        "attention",
        "dots3_note_swa_prefill",
        signature,
        traits={"head_dim": q.shape[-1], "value_head_dim": v.shape[-1]},
        solution=solution,
    )
    shape_params = {
        "batch_size": cu_seqlens_q.shape[0] - 1,
        "total_q": q.shape[0],
        "total_kv": k.shape[0],
        "num_q_heads": q.shape[1],
        "num_kv_heads": k.shape[1],
        "head_dim": q.shape[-1],
        "v_head_dim": v.shape[-1],
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_kv": max_seqlen_kv,
        "window_left": window_left,
    }
    ShapeCapture.get().record(
        "attention", "dots3_note_swa_prefill", kernel.name, q.dtype, shape_params
    )
    with kernel_scope(
        "attention",
        "dots3_note_swa_prefill",
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            softmax_scale,
            window_left=window_left,
        )


# Backend registration (side-effect import).
import tokenspeed_kernel.ops.attention.dots3_note.triton  # noqa: E402,F401
