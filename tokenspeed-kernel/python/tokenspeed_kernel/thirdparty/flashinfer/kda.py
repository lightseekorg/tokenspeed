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

"""FlashInfer cuDNN KDA integration."""

from __future__ import annotations

import torch


def cudnn_kda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    lower_bound: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scan token-strided Q/K/V and return a separate native V-major state.

    Args:
        q/k/v: Packed BF16 [1, tokens, heads, 128] activations.
        g_raw: BF16/FP32 gate with the same shape and 16-byte-aligned token strides.
        beta_logits: Raw BF16 [1, tokens, heads] beta projection.
        A_log/dt_bias: FP32 per-head decay and per-channel gate bias.
        initial_state: FP32 [sequences, heads, V, K] gathered cache state.
        cu_seqlens: Device boundaries [sequences + 1].
        lower_bound: Safe-gate bound forwarded to cuDNN.

    Returns:
        Packed output and FP32 final state; initial_state is not modified.
    """
    from flashinfer.cudnn import cudnn_recurrent_kda

    if lower_bound is None:
        raise ValueError("cuDNN KDA requires a safe-gate lower bound")
    if g_raw.stride(-3) * g_raw.element_size() % 16:
        raise ValueError("cuDNN KDA requires 16-byte-aligned gate token strides")
    # A separate output avoids FlashInfer's in-place state copy-back.
    final_state = torch.empty_like(initial_state)
    output, _ = cudnn_recurrent_kda(
        q,
        k,
        v,
        g_raw,
        beta_logits,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=q.shape[-1] ** -0.5,
        initial_state=initial_state,
        output_state=final_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        beta_is_logit=True,
    )
    return output, final_state
