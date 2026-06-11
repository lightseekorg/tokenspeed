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

# Mamba2 mixer using TokenSpeed runtime linear-attention Triton kernels.
#
# Original source:
# https://github.com/sgl-project/sglang/blob/03c77dc33d0a051aa15c1235407440d9d107b98f/python/sglang/srt/layers/attention/mamba/mamba.py
# The direct Triton Mamba kernels live beside this module in linear attention.

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
from flashinfer.mamba import selective_state_update

from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.distributed.utils import divide
from tokenspeed.runtime.layers.attention.linear.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from tokenspeed.runtime.layers.attention.linear.mamba2_metadata import Mamba2Metadata
from tokenspeed.runtime.layers.attention.linear.mixer2_rms_norm_gated import (
    Mixer2RMSNormGated,
)
from tokenspeed.runtime.layers.attention.linear.ssd_combined import (
    mamba_chunk_scan_combined,
)
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.model_loader.weight_utils import (
    mamba_v2_sharded_weight_loader,
    sharded_weight_loader,
)
from tokenspeed.runtime.utils import add_prefix, set_weight_attrs

LoaderFunction = Callable[[torch.Tensor, torch.Tensor], None]


def _extra_groups_for_head_shards(ngroups: int, tp_size: int) -> int:
    if ngroups % tp_size == 0:
        return 0
    return tp_size - ngroups


def _composed_weight_loader(
    loader: LoaderFunction, transform: Callable[[torch.Tensor], torch.Tensor]
) -> LoaderFunction:
    def composed(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        loader(param, loaded_weight)
        param.data.copy_(transform(param))

    return composed


class MambaMixer2(nn.Module):
    def __init__(
        self,
        config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.mapping = mapping
        self.tp_size = mapping.attn.tp_size
        self.tp_rank = mapping.attn.tp_rank
        self.tp_group = mapping.attn.tp_group
        self.num_heads = config.mamba_num_heads
        self.head_dim = config.mamba_head_dim
        self.ssm_state_size = config.ssm_state_size
        self.activation = config.mamba_hidden_act
        self.intermediate_size = self.num_heads * self.head_dim
        self.n_groups = config.mamba_n_groups
        if self.n_groups % self.tp_size != 0:
            self.n_groups += _extra_groups_for_head_shards(
                config.mamba_n_groups, self.tp_size
            )
        self.groups_ssm_state_size = self.n_groups * self.ssm_state_size
        self.conv_dim = self.intermediate_size + 2 * self.groups_ssm_state_size
        self.prefix = prefix

        assert self.num_heads % self.tp_size == 0
        assert (config.mamba_n_groups % self.tp_size) == 0 or config.mamba_n_groups == 1
        assert (
            (config.mamba_n_groups % self.tp_size == 0)
            or self.tp_size == 1
            or quant_config is None
        )

        conv_kernel_size = config.conv_kernel
        if config.mamba_n_groups % self.tp_size == 0:
            self.conv1d = MergedColumnParallelLinear(
                input_size=conv_kernel_size,
                output_sizes=[
                    self.intermediate_size,
                    self.groups_ssm_state_size,
                    self.groups_ssm_state_size,
                ],
                bias=config.use_conv_bias,
                quant_config=None,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                tp_group=self.tp_group,
                prefix=add_prefix("conv1d", prefix),
            )
            self.in_proj = MergedColumnParallelLinear(
                input_size=config.hidden_size,
                output_sizes=[
                    self.intermediate_size,
                    self.intermediate_size,
                    self.groups_ssm_state_size,
                    self.groups_ssm_state_size,
                    self.num_heads,
                ],
                bias=config.use_bias,
                quant_config=quant_config,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                tp_group=self.tp_group,
                prefix=add_prefix("in_proj", prefix),
            )
        else:
            self.conv1d = ColumnParallelLinear(
                input_size=conv_kernel_size,
                output_size=self.conv_dim,
                bias=config.use_conv_bias,
                quant_config=None,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                tp_group=self.tp_group,
                prefix=add_prefix("conv1d", prefix),
            )
            self.in_proj = ColumnParallelLinear(
                input_size=config.hidden_size,
                output_size=self.intermediate_size + self.conv_dim + self.num_heads,
                bias=config.use_bias,
                quant_config=quant_config,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                tp_group=self.tp_group,
                prefix=add_prefix("in_proj", prefix),
            )
            group_settings = (
                self.groups_ssm_state_size,
                (self.n_groups - config.mamba_n_groups) * self.ssm_state_size,
                config.mamba_n_groups == 1,
            )
            intermediate_settings = (self.intermediate_size, 0, False)
            head_settings = (self.num_heads, 0, False)
            if self.conv1d.bias is not None:
                delattr(self.conv1d.bias, "weight_loader")
                set_weight_attrs(
                    self.conv1d.bias,
                    {
                        "weight_loader": mamba_v2_sharded_weight_loader(
                            [
                                intermediate_settings,
                                group_settings,
                                group_settings,
                            ],
                            self.tp_size,
                            self.tp_rank,
                        )
                    },
                )
            delattr(self.conv1d.weight, "weight_loader")
            set_weight_attrs(
                self.conv1d.weight,
                {
                    "weight_loader": mamba_v2_sharded_weight_loader(
                        [intermediate_settings, group_settings, group_settings],
                        self.tp_size,
                        self.tp_rank,
                    )
                },
            )
            if quant_config is None:
                delattr(self.in_proj.weight, "weight_loader")
                set_weight_attrs(
                    self.in_proj.weight,
                    {
                        "weight_loader": mamba_v2_sharded_weight_loader(
                            [
                                intermediate_settings,
                                intermediate_settings,
                                group_settings,
                                group_settings,
                                head_settings,
                            ],
                            self.tp_size,
                            self.tp_rank,
                        )
                    },
                )

        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)
        self.A = nn.Parameter(
            torch.empty(divide(self.num_heads, self.tp_size), dtype=torch.float32)
        )
        self.D = nn.Parameter(torch.ones(self.num_heads // self.tp_size))
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads // self.tp_size))
        set_weight_attrs(
            self.D, {"weight_loader": sharded_weight_loader(0, self.tp_rank)}
        )
        set_weight_attrs(
            self.A,
            {
                "weight_loader": _composed_weight_loader(
                    sharded_weight_loader(0, self.tp_rank),
                    lambda x: -torch.exp(x.float()),
                )
            },
        )
        set_weight_attrs(
            self.dt_bias, {"weight_loader": sharded_weight_loader(0, self.tp_rank)}
        )

        self.out_proj = RowParallelLinear(
            self.intermediate_size,
            config.hidden_size,
            bias=config.use_bias,
            input_is_parallel=True,
            reduce_results=True,
            quant_config=quant_config,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            tp_group=self.tp_group,
            prefix=add_prefix("out_proj", prefix),
        )
        self.norm = Mixer2RMSNormGated(
            self.intermediate_size,
            config.mamba_n_groups,
            mapping,
            use_rms_norm=True,
            eps=config.layer_norm_epsilon,
        )

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        metadata: Mamba2Metadata,
    ) -> torch.Tensor | None:
        state_indices_tensor = metadata.mamba_cache_indices
        projected_states, _ = self.in_proj(hidden_states)

        gate, hidden_states_B_C, dt = torch.split(
            projected_states,
            [
                self.intermediate_size // self.tp_size,
                self.conv_dim // self.tp_size,
                self.num_heads // self.tp_size,
            ],
            dim=-1,
        )
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        def split_hidden_states_B_C(hidden_states_B_C):
            return torch.split(
                hidden_states_B_C,
                [
                    self.intermediate_size // self.tp_size,
                    self.groups_ssm_state_size // self.tp_size,
                    self.groups_ssm_state_size // self.tp_size,
                ],
                dim=-1,
            )

        num_prefills = metadata.num_prefills
        num_decodes = metadata.num_decodes
        num_decode_tokens = num_decodes
        num_prefill_tokens = metadata.num_prefill_tokens
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0
        num_actual_tokens = num_prefill_tokens + num_decode_tokens
        assert num_actual_tokens == projected_states.shape[0]

        hidden_states_B_C_p, hidden_states_B_C_d = torch.split(
            hidden_states_B_C, [num_prefill_tokens, num_decode_tokens], dim=0
        )
        dt_p, dt_d = torch.split(dt, [num_prefill_tokens, num_decode_tokens], dim=0)
        state_indices_tensor_p, state_indices_tensor_d = torch.split(
            state_indices_tensor, [num_prefills, num_decodes], dim=0
        )
        query_start_loc_p = (
            metadata.query_start_loc[: num_prefills + 1] if has_prefill else None
        )

        preallocated_ssm_out = torch.empty(
            [
                projected_states.shape[0],
                (self.num_heads * self.head_dim) // self.tp_size,
            ],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        preallocated_ssm_out_p, preallocated_ssm_out_d = torch.split(
            preallocated_ssm_out, [num_prefill_tokens, num_decode_tokens], dim=0
        )
        intermediate_states = None

        if has_prefill:
            mixed_metadata = metadata.mixed_metadata
            assert mixed_metadata is not None
            has_initial_states_p = mixed_metadata.has_initial_states
            x = hidden_states_B_C_p.transpose(0, 1)
            if metadata.track_conv_indices is not None:
                x_to_track = x[:, metadata.track_conv_indices].transpose(0, 1)
                assert metadata.track_ssm_h_dst is not None
                conv_state[metadata.track_ssm_h_dst] = x_to_track

            hidden_states_B_C_p = causal_conv1d_fn(
                x,
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_states_p,
                cache_indices=state_indices_tensor_p,
                query_start_loc=query_start_loc_p,
                seq_lens_cpu=mixed_metadata.extend_seq_lens_cpu,
            ).transpose(0, 1)[:num_prefill_tokens]

            hidden_states_p, B_p, C_p = split_hidden_states_B_C(hidden_states_B_C_p)
            initial_states = None
            if mixed_metadata.prep_initial_states:
                initial_states = torch.where(
                    has_initial_states_p[:, None, None, None],
                    ssm_state[state_indices_tensor_p],
                    0,
                )

            intermediate_states, varlen_state = mamba_chunk_scan_combined(
                hidden_states_p.view(
                    1,
                    num_prefill_tokens,
                    self.num_heads // self.tp_size,
                    self.head_dim,
                ),
                dt_p.unsqueeze(0),
                self.A,
                B_p.view(1, num_prefill_tokens, self.n_groups // self.tp_size, -1),
                C_p.view(1, num_prefill_tokens, self.n_groups // self.tp_size, -1),
                chunk_size=mixed_metadata.chunk_size,
                D=self.D,
                z=None,
                dt_bias=self.dt_bias,
                seq_idx=mixed_metadata.seq_idx,
                chunk_indices=mixed_metadata.chunk_indices,
                chunk_offsets=mixed_metadata.chunk_offsets,
                cu_seqlens=query_start_loc_p,
                initial_states=initial_states,
                return_varlen_states=True,
                return_final_states=False,
                return_intermediate_states=True,
                dt_softplus=True,
                dt_limit=(0.0, float("inf")),
                out=preallocated_ssm_out_p.view(
                    1, num_prefill_tokens, -1, self.head_dim
                ),
                state_dtype=ssm_state.dtype,
            )
            ssm_state[state_indices_tensor_p] = varlen_state

        if has_decode:
            hidden_states_B_C_d = causal_conv1d_update(
                hidden_states_B_C_d,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=state_indices_tensor_d,
            )

            hidden_states_d, B_d, C_d = split_hidden_states_B_C(hidden_states_B_C_d)
            n_groups = self.n_groups // self.tp_size
            A_d = (
                self.A[:, None, ...][:, :, None]
                .expand(-1, self.head_dim, self.ssm_state_size)
                .to(dtype=torch.float32)
            )
            dt_d = dt_d[:, :, None].expand(-1, -1, self.head_dim)
            dt_bias = self.dt_bias[:, None, ...].expand(-1, self.head_dim)
            D_d = self.D[:, None, ...].expand(-1, self.head_dim)
            B_d = B_d.view(-1, n_groups, B_d.shape[1] // n_groups)
            C_d = C_d.view(-1, n_groups, C_d.shape[1] // n_groups)
            hidden_states_d = hidden_states_d.view(
                -1, self.num_heads // self.tp_size, self.head_dim
            )

            selective_state_update(
                ssm_state,
                hidden_states_d,
                dt_d,
                A_d,
                B_d,
                C_d,
                D_d,
                z=None,
                dt_bias=dt_bias,
                dt_softplus=True,
                state_batch_indices=state_indices_tensor_d,
                out=preallocated_ssm_out_d.view(num_decodes, -1, self.head_dim),
            )

        hidden_states = self.norm(preallocated_ssm_out, gate[:num_actual_tokens])
        output[:num_actual_tokens], _ = self.out_proj(hidden_states)

        if (
            metadata.track_ssm_h_src is not None
            and metadata.track_ssm_h_dst is not None
        ):
            if intermediate_states is None:
                raise RuntimeError("Missing intermediate Mamba2 states for tracking")
            ssm_state[metadata.track_ssm_h_dst] = intermediate_states.squeeze(0)[
                metadata.track_ssm_h_src
            ].to(ssm_state.dtype, copy=False)

        if (
            metadata.track_ssm_final_src is not None
            and metadata.track_ssm_final_dst is not None
        ):
            conv_state[metadata.track_ssm_final_dst] = conv_state[
                metadata.track_ssm_final_src
            ]
            ssm_state[metadata.track_ssm_final_dst] = ssm_state[
                metadata.track_ssm_final_src
            ]

        return intermediate_states

    @property
    def mamba_type(self) -> str:
        return "mamba2"
