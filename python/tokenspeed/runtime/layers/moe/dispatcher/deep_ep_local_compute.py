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


import tokenspeed_kernel
import torch
from tokenspeed_kernel.ops.activation.flashinfer import silu_and_mul
from tokenspeed_kernel.ops.gemm.deep_gemm import (
    m_grouped_fp8_gemm_nt_contiguous,
    m_grouped_fp8_gemm_nt_masked,
)
from tokenspeed_kernel.ops.gemm.fp8_utils import per_token_group_quant_fp8
from tokenspeed_kernel.ops.moe.deepep import get_tma_aligned_size, tma_align_input_scale

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)


class DeepExecutor:
    def __init__(
        self, w13_weight, w13_weight_scale_inv, w2_weight, w2_weight_scale_inv
    ):
        self.w13_weight = w13_weight
        self.w13_weight_scale_inv = w13_weight_scale_inv
        self.w2_weight = w2_weight
        self.w2_weight_scale_inv = w2_weight_scale_inv

        self.hidden_size = self.w13_weight.size(2)
        self.inter_size_x2 = self.w13_weight.size(1)

        self.fp8_dtype = w13_weight.dtype
        self.scale_block_size = 128
        self.num_experts = w13_weight.size(0)

    def forward_normal(
        self,
        hidden_states,
        input_scales,
        topk_idx,
        topk_weights,
        num_recv_tokens_per_expert: list[int],
    ):
        if num_recv_tokens_per_expert is None:
            return hidden_states.bfloat16()
        all_tokens = sum(num_recv_tokens_per_expert)
        if all_tokens <= 0:
            return hidden_states.bfloat16()
        _, K = hidden_states.size()
        N = self.w13_weight.size(1)
        scale_block_size = 128

        device = hidden_states.device

        input_tensor = [
            torch.empty(
                (all_tokens, K),
                device=device,
                dtype=self.fp8_dtype,
            ),
            torch.empty(
                (all_tokens, K // 128),
                device=device,
                dtype=torch.float32,
            ),
        ]
        m_indices = torch.empty(all_tokens, device=device, dtype=torch.int32)
        output_index = torch.empty_like(topk_idx)
        num_recv_tokens_per_expert_gpu = torch.tensor(
            num_recv_tokens_per_expert,
            dtype=torch.int32,
            pin_memory=True,
            device="cpu",
        ).cuda(non_blocking=True)
        expert_start_loc = torch.empty_like(num_recv_tokens_per_expert_gpu)

        tokenspeed_kernel.moe_dispatch(
            hidden_states,
            input_scales,
            topk_idx,
            num_recv_tokens_per_expert_gpu,
            expert_start_loc,
            input_tensor[0],
            input_tensor[1],
            m_indices,
            output_index,
            dtype=hidden_states.dtype,
            traits={"comm_strategy": "deep_ep"},
            expected_kernel_name="deepep_moe_scatter",
        )

        gate_up_output = torch.empty(
            (all_tokens, N),
            device=device,
            dtype=torch.bfloat16,
        )
        input_tensor[1] = tma_align_input_scale(input_tensor[1])
        m_grouped_fp8_gemm_nt_contiguous(
            input_tensor,
            (self.w13_weight, self.w13_weight_scale_inv),
            gate_up_output,
            m_indices,
        )

        down_input = torch.empty(
            (
                all_tokens,
                N // 2,
            ),
            device=device,
            dtype=torch.bfloat16,
        )
        silu_and_mul(gate_up_output.view(-1, N), down_input)
        del gate_up_output
        down_output = torch.empty(
            (all_tokens, K),
            device=device,
            dtype=torch.bfloat16,
        )
        down_input_fp8, down_input_scale = per_token_group_quant_fp8(
            down_input, scale_block_size
        )
        down_input_scale = tma_align_input_scale(down_input_scale)
        m_grouped_fp8_gemm_nt_contiguous(
            (down_input_fp8, down_input_scale),
            (self.w2_weight, self.w2_weight_scale_inv),
            down_output,
            m_indices,
        )

        gather_out = torch.empty_like(
            hidden_states,
            device=device,
            dtype=torch.bfloat16,
        )
        tokenspeed_kernel.moe_combine(
            down_output,
            topk_idx,
            topk_weights,
            output_index,
            gather_out,
            dtype=torch.bfloat16,
            traits={"comm_strategy": "deep_ep"},
            expected_kernel_name="deepep_moe_gather",
        )

        return gather_out

    def get_col_major_tma_aligned_scale(self, b, m, n, device):
        aligned_m = get_tma_aligned_size(m, 4)
        stride0 = aligned_m * n
        stride1 = 1
        stride2 = aligned_m
        return torch.empty_strided(
            (b, m, n), (stride0, stride1, stride2), dtype=torch.float32, device=device
        )

    def forward_low_latency(
        self,
        hidden_states,
        input_scales,
        masked_m: torch.Tensor,
        # expected_m: a value hint (which is a value on CPU) for the M expectation of each batch,
        # correctly setting this value may lead to better performance.
        expected_m: int,
        num_tokens_hint: int = 0,
    ):
        del num_tokens_hint
        num_groups, M, _ = hidden_states.size()
        device = hidden_states.device

        gate_up_output = torch.empty(
            (num_groups, M, self.inter_size_x2), device=device, dtype=torch.bfloat16
        )
        m_grouped_fp8_gemm_nt_masked(
            (hidden_states, input_scales),
            (self.w13_weight, self.w13_weight_scale_inv),
            gate_up_output,
            masked_m,
            expected_m,
        )

        # Act
        activation = torch.empty(
            (num_groups, M, self.inter_size_x2 // 2),
            device=device,
            dtype=self.fp8_dtype,
        )
        activation_scale = self.get_col_major_tma_aligned_scale(
            num_groups,
            M,
            self.inter_size_x2 // 2 // self.scale_block_size,
            device,
        )

        from tokenspeed_kernel.ops.activation.cuda import silu_and_mul_fuse_block_quant

        activation, activation_scale = silu_and_mul_fuse_block_quant(
            gate_up_output,
            activation_scale,
            activation,
            True,
            masked_m,
            expected_m,
            self.num_experts,
        )

        output = torch.empty(
            (num_groups, M, self.hidden_size), device=device, dtype=torch.bfloat16
        )
        m_grouped_fp8_gemm_nt_masked(
            (activation, activation_scale),
            (self.w2_weight, self.w2_weight_scale_inv),
            output,
            masked_m,
            expected_m,
        )
        return output
