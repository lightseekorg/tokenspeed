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

# Mamba2 gated RMSNorm for TokenSpeed tensor-parallel execution.
#
# Original source:
# https://github.com/sgl-project/sglang/blob/03c77dc33d0a051aa15c1235407440d9d107b98f/python/sglang/srt/layers/attention/mamba/mixer2_rms_norm_gated.py

from __future__ import annotations

import torch
import torch.nn as nn

from tokenspeed.runtime.distributed.comm_ops import all_gather, all_reduce
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.model_loader.weight_utils import sharded_weight_loader
from tokenspeed.runtime.utils import set_weight_attrs


class Mixer2RMSNormGated(nn.Module):
    def __init__(
        self,
        full_hidden_size: int,
        full_n_groups: int,
        mapping: Mapping,
        use_rms_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.tp_size = mapping.attn.tp_size
        self.tp_rank = mapping.attn.tp_rank
        self.tp_group = mapping.attn.tp_group
        self.full_hidden_size = full_hidden_size
        self.group_size = full_hidden_size // full_n_groups
        self.per_rank_hidden_size = full_hidden_size // self.tp_size
        self.n_groups = full_hidden_size // self.group_size

        self.variance_epsilon = eps
        self.use_rms_norm = use_rms_norm
        if self.use_rms_norm:
            self.weight = nn.Parameter(torch.ones(self.per_rank_hidden_size))
            set_weight_attrs(
                self.weight,
                {"weight_loader": sharded_weight_loader(0, self.tp_rank)},
            )
        else:
            self.register_parameter("weight", None)
        assert self.full_hidden_size % self.tp_size == 0

    def forward(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        input_dtype = x.dtype
        x = x * torch.nn.functional.silu(gate.to(torch.float32))
        if not self.use_rms_norm:
            return x.to(input_dtype)

        if self.n_groups == 1:
            if self.tp_size > 1:
                local_sums = x.pow(2).sum(dim=-1, keepdim=True)
                global_sums = all_reduce(local_sums, self.tp_group)
                variance = global_sums / (self.tp_size * x.shape[-1])
            else:
                variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.variance_epsilon)
        else:
            redundant_tp = self.n_groups % self.tp_size != 0
            if redundant_tp:
                x = all_gather(x, self.tp_group, dim=-1)

            *prefix_dims, hidden_dim = x.shape
            group_count = hidden_dim // self.group_size
            x_grouped = x.view(*prefix_dims, group_count, self.group_size)
            variance = x_grouped.pow(2).mean(-1, keepdim=True)
            x_grouped = x_grouped * torch.rsqrt(variance + self.variance_epsilon)
            x = x_grouped.view(*prefix_dims, hidden_dim)

            if redundant_tp:
                start = self.per_rank_hidden_size * self.tp_rank
                end = start + self.per_rank_hidden_size
                x = x[..., start:end]

        return self.weight * x.to(input_dtype)
