# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# SPDX-FileCopyrightText: Copyright (c) 2024, Tri Dao
#
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

"""Gated RMSNorm module for the linear-attention blocks: ``norm(x) * gate(z)``
or ``norm(x * gate(z))`` with ``silu`` or ``sigmoid`` as the gate, run by
``tokenspeed_kernel.ops.layernorm.gated_rmsnorm``. Inference only (no
backward)."""

import torch
from tokenspeed_kernel.ops.layernorm import gated_rmsnorm


def rmsnorm_fn(
    x,
    weight,
    z=None,
    eps=1e-6,
    group_size=None,
    norm_before_gate=True,
    sigmoid_gate=False,
    *,
    weights_independent: bool,
    fp8_scale: torch.Tensor | None = None,
):
    """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else
    norm(x * silu(z)); ``sigmoid_gate`` swaps silu(z) for sigmoid(z). With
    ``group_size`` set, each group of that many features is normalized on its
    own (``None`` means one group over the whole last dim).
    A ``weight`` of ``group_size`` features is shared by every group.

    ``weights_independent`` promises the weight is already visible before the
    preceding PDL producer starts, permitting its load before the input wait.
    Pass False for producer-written weights. Returns a tensor shaped like x,
    quantized to static FP8 with ``fp8_scale`` when one is given.
    """
    return gated_rmsnorm(
        x,
        weight,
        z,
        eps=eps,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        sigmoid_gate=sigmoid_gate,
        fp8_scale=fp8_scale,
        weights_independent=weights_independent,
    )


class RMSNorm(torch.nn.Module):

    def __init__(
        self,
        hidden_size,
        eps=1e-5,
        group_size=None,
        norm_before_gate=True,
        device=None,
        dtype=None,
        sigmoid_gate=False,
    ):
        """If group_size is not None, we do GroupNorm with each group having group_size elements.
        group_size=None is equivalent to group_size=hidden_size (i.e. there's only 1 group).
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.sigmoid_gate = sigmoid_gate
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor | None = None,
        fp8_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else
        norm(x * silu(z)); ``sigmoid_gate`` swaps silu(z) for sigmoid(z).
        ``fp8_scale`` quantizes the output for a static-FP8 consumer."""
        return rmsnorm_fn(
            x,
            self.weight,
            z=z,
            eps=self.eps,
            group_size=self.group_size,
            norm_before_gate=self.norm_before_gate,
            sigmoid_gate=self.sigmoid_gate,
            weights_independent=True,
            fp8_scale=fp8_scale,
        )
