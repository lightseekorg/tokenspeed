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

"""PyTorch references for ``tokenspeed_kernel.ops.attention.mamba2``."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@register_kernel(
    "attention",
    "mamba2_recurrence",
    name="torch_mamba2_recurrence",
    solution="torch",
    signatures=format_signatures(
        "x", "dense", {torch.bfloat16, torch.float16, torch.float32}
    ),
    priority=Priority.REFERENCE,
)
def torch_mamba2_recurrence(
    x: torch.Tensor,
    dt: torch.Tensor,
    A_log: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    *,
    dt_limit: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-by-token Mamba2 SSM recurrence of one sequence in FP64.

    ``step = clamp(softplus(dt + dt_bias), *dt_limit)``; the state decays by
    ``exp(step * -exp(A_log))`` and adds ``step * x`` along ``B``; the output
    reads the state along ``C`` plus ``D * x``. B/C groups are shared by
    consecutive heads.

    Args:
        x: ``[T, heads, P]``; dt: ``[T, heads]``.
        B, C: ``[T, groups, N]``.
        A_log, D, dt_bias: ``[heads]``.
        state: ``[heads, P, N]``.

    Returns:
        FP32 ``(y [T, heads, P], final state [heads, P, N])``.
    """
    heads_per_group = x.shape[1] // B.shape[1]
    A = -torch.exp(A_log.double())
    s = state.double().clone()
    ys = []
    for t in range(x.shape[0]):
        step = F.softplus(dt[t].double() + dt_bias.double()).clamp(*dt_limit)
        b = B[t].double().repeat_interleave(heads_per_group, dim=0)
        c = C[t].double().repeat_interleave(heads_per_group, dim=0)
        xt = x[t].double()
        s = (
            s * torch.exp(step * A)[:, None, None]
            + (step[:, None] * xt)[:, :, None] * b[:, None, :]
        )
        ys.append(torch.einsum("hpn,hn->hp", s, c) + D.double()[:, None] * xt)
    return torch.stack(ys).float(), s.float()
