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

"""FlashInfer KDA prefill integration."""

from __future__ import annotations

from collections import OrderedDict

import torch

_HEAD_DIM = 128
# Plans of recent boundary sets; every KDA layer of a forward shares one.
_PLAN_CACHE_SIZE = 8
_plans: OrderedDict[tuple, tuple] = OrderedDict()


def _plan(offsets: tuple[int, ...], heads: int, device: torch.device) -> tuple:
    from flashinfer.kda_kernels.persistent.forward import make_plan

    key = (offsets, heads, device)
    plan = _plans.get(key)
    if plan is None:
        plan = make_plan(list(offsets), heads, device)
        _plans[key] = plan
        if len(_plans) > _PLAN_CACHE_SIZE:
            _plans.popitem(last=False)
    else:
        _plans.move_to_end(key)
    return plan


def persistent_kda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    initial_state: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor,
    lower_bound: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scan packed KDA prefill with FlashInfer's persistent SM100/SM103 kernel.

    FlashInfer's public ``recurrent_kda`` plans this kernel from device
    boundaries with a ``tolist()`` per call, a stream-synchronizing D2H on
    every KDA layer. This adapter plans from the host mirror instead and
    calls the kernel's forward entry, so it repeats the public path's input
    contract here.

    Args:
        q/k/v: Packed BF16 [1, tokens, heads, 128] activations; heads must be
            a multiple of eight.
        g_raw: Raw BF16 [1, tokens, heads, 128] gate; the decay gate is applied
            in-kernel.
        beta_logits: Raw BF16 [1, tokens, heads] beta projection.
        A_log/dt_bias: FP32 per-head decay [heads] and gate bias [heads * 128]
            or [heads, 128].
        initial_state: FP32 [sequences, heads, V, K] gathered cache state. The
            kernel advances it in place and returns it as the final state.
        cu_seqlens_cpu: Host int64 boundaries [sequences + 1], strictly
            increasing from 0 to the token count.
        lower_bound: Safe-gate bound in [-5, 0).

    Returns:
        Packed BF16 output and the FP32 final state (``initial_state``).
    """
    from flashinfer.kda_kernels.persistent.forward import fwd

    if q.ndim != 4 or q.shape[0] != 1 or q.dtype != torch.bfloat16:
        raise ValueError(
            "FlashInfer persistent KDA requires packed BF16 [1, T, H, 128] q"
        )
    _, tokens, heads, dim = q.shape
    if dim != _HEAD_DIM or heads % 8:
        raise ValueError(
            f"FlashInfer persistent KDA requires head dim 128 and heads divisible "
            f"by 8; got heads={heads}, dim={dim}"
        )
    if lower_bound is None or not -5.0 <= float(lower_bound) < 0.0:
        raise ValueError("FlashInfer persistent KDA requires a lower bound in [-5, 0)")
    if tokens * heads * dim >= 2**31:
        raise ValueError("FlashInfer persistent KDA requires T*H*128 < 2**31")
    offsets = tuple(cu_seqlens_cpu.tolist())
    sequences = len(offsets) - 1
    if (
        sequences < 1
        or offsets[0] != 0
        or offsets[-1] != tokens
        or any(end <= start for start, end in zip(offsets, offsets[1:]))
    ):
        raise ValueError(
            "KDA boundaries must rise strictly from 0 to the token count, "
            f"got {offsets[:4]}...{offsets[-1:]} for {tokens} tokens"
        )
    if sequences * heads * dim * dim >= 2**31:
        raise ValueError("FlashInfer persistent KDA requires N*H*128*128 < 2**31")
    if (
        initial_state.shape != (sequences, heads, dim, dim)
        or initial_state.dtype != torch.float32
        or not initial_state.is_contiguous()
        or initial_state.data_ptr() % 32
    ):
        raise ValueError(
            "FlashInfer persistent KDA requires a contiguous, 32-byte-aligned FP32 "
            f"[{sequences}, {heads}, 128, 128] initial state"
        )
    if A_log.shape != (heads,) or A_log.dtype != torch.float32:
        raise ValueError("FlashInfer persistent KDA requires FP32 A_log [heads]")
    if dt_bias.dtype != torch.float32 or dt_bias.shape not in (
        (heads * dim,),
        (heads, dim),
    ):
        raise ValueError(
            "FlashInfer persistent KDA requires FP32 dt_bias [heads * 128]"
        )
    # The kernel's descriptors index dense rows; serving's beta is a slice of
    # the merged projection, so it is the input this usually copies.
    q, k, v, g_raw, beta_logits, A_log, dt_bias = (
        t.contiguous() for t in (q, k, v, g_raw, beta_logits, A_log, dt_bias)
    )
    for name, tensor in (("k", k), ("v", v), ("g_raw", g_raw)):
        if tensor.shape != q.shape or tensor.dtype != torch.bfloat16:
            raise ValueError(f"FlashInfer persistent KDA requires BF16 {name} like q")
    if beta_logits.shape != q.shape[:3] or beta_logits.dtype != torch.bfloat16:
        raise ValueError("FlashInfer persistent KDA requires BF16 beta [1, T, H]")
    plan = _plan(offsets, heads, q.device)
    output = torch.empty_like(v)
    with torch.cuda.device(q.device):
        fwd(
            q,
            k,
            v,
            g_raw,
            beta_logits,
            dim**-0.5,
            output,
            A_log,
            dt_bias,
            float(lower_bound),
            initial_state=initial_state,
            final_state=initial_state,
            cu_seqlens=plan[0],
            plan=plan,
        )
    return output, initial_state
