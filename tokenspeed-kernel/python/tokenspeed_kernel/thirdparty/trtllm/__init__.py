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

"""Kernel dispatch for trtllm-kernel quant and MoE helpers.

The implementation uses TRT-LLM CUDA kernels exposed as torch.ops.trtllm /
torch.ops.tensorrt_llm.
"""

import torch
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import error_fn

platform = current_platform()

dsv3_fused_a_gemm = error_fn
fp8_blockwise_scaled_mm = error_fn
per_token_group_quant_8bit = error_fn
per_tensor_quant_fp8 = error_fn
per_token_quant_fp8 = error_fn
moe_align_block_size = error_fn
fast_topk_v2 = error_fn

# deep_ep_cpp MUST be loaded before trtllm_kernel.  libtensorrt_llm.so
# statically links libcudart_static.a, creating a second CUDA runtime in
# the process.  deep_ep_cpp uses CUDA separate compilation (device-linked
# binaries) whose deferred __cudaRegisterLinkedBinary registration relies
# on internal libcudart function-pointer tables.  If the static-linked
# cudart in libtensorrt_llm initializes first, it corrupts this state in
# the global libcudart.so.13, and all 820+ kernel registrations from
# deep_ep_cpp silently fail (cudaFuncGetAttributes returns rc=400).
if platform.is_nvidia:
    deep_ep_cpp_loaded = False
    try:
        import deep_ep_cpp  # noqa: F401 — triggers .init_array CUDA registration
    except ImportError:
        pass
    else:
        deep_ep_cpp_loaded = True

    trtllm_kernel_loaded = False
    if deep_ep_cpp_loaded:
        try:
            import trtllm_kernel  # noqa: F401  — loads .so and registers torch.ops.trtllm.*
        except ImportError:
            pass
        else:
            trtllm_kernel_loaded = True

    if trtllm_kernel_loaded:

        # DSv3 min-latency fused-A projection (hidden→q_a‖kv_a_mqa). On SM≥90
        # with bf16 and shape [1..16, 7168] × [7168, 2112], trtllm fires a
        # hand-rolled warp-specialized kernel; off-shape it falls back to cuBLAS.
        def dsv3_fused_a_gemm(mat_a: torch.Tensor, mat_b: torch.Tensor) -> torch.Tensor:
            return torch.ops.trtllm.dsv3_fused_a_gemm_op(mat_a, mat_b, None, None)

        # FP8 blockwise matmul helper.
        def fp8_blockwise_scaled_mm(
            mat_a: torch.Tensor,
            mat_b: torch.Tensor,
            scales_a: torch.Tensor,
            scales_b: torch.Tensor,
            out_dtype: torch.dtype,
        ) -> torch.Tensor:
            alpha = torch.tensor(1.0, dtype=torch.float32, device=mat_a.device)
            return torch.ops.trtllm.fp8_block_scaling_gemm_impl(
                mat_a, mat_b, alpha, scales_a, scales_b, out_dtype
            )

        def per_token_group_quant_8bit(
            x: torch.Tensor,
            group_size: int = 128,
            use_ue8m0: bool = False,
        ) -> tuple:
            assert (
                group_size == 128
            ), f"trtllm fp8_quantize_1x128 only supports group_size=128, got {group_size}"
            return torch.ops.trtllm.fp8_quantize_1x128(x, use_ue8m0)

        def per_tensor_quant_fp8(
            input: torch.Tensor,
            output: torch.Tensor,
            scale: torch.Tensor,
        ) -> None:
            q, s = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(input)
            output.copy_(q)
            scale.copy_(s.float().squeeze())

        def per_token_quant_fp8(
            input: torch.Tensor,
            output: torch.Tensor,
            scale: torch.Tensor,
        ) -> None:
            q, s = torch.ops.tensorrt_llm.quantize_e4m3_activation(input)
            output.copy_(q)
            scale.copy_(s.float().squeeze(-1))

        def moe_align_block_size(
            topk_ids: torch.Tensor,
            num_experts: int,
            block_size: int,
            sorted_ids: torch.Tensor,
            expert_ids: torch.Tensor,
            num_tokens_post_pad: torch.Tensor,
            cumsum_buffer: torch.Tensor = None,
            pad_sorted_token_ids: bool = False,
        ):
            # Kernel convention: num_experts is actually num_experts + 1.
            actual_num_experts = num_experts - 1
            torch.ops.trtllm.moe_align_block_size(
                topk_ids,
                actual_num_experts,
                block_size,
                sorted_ids,
                expert_ids,
                num_tokens_post_pad,
            )

        def fast_topk_v2(
            values: torch.Tensor,
            seq_lens: torch.Tensor,
            indices: torch.Tensor,
            topk: int,
            next_n: int = 1,
        ):
            seq_lens = seq_lens.to(torch.int32).contiguous()
            if next_n == 1:
                torch.ops.trtllm.indexer_topk_decode(
                    values, seq_lens, indices, next_n, topk
                )
            else:
                row_ends = seq_lens.cumsum(0)
                row_starts = row_ends - seq_lens
                torch.ops.trtllm.indexer_topk_prefill(
                    values, row_starts, row_ends, indices, topk
                )


__all__ = [
    "dsv3_fused_a_gemm",
    "fp8_blockwise_scaled_mm",
    "per_token_group_quant_8bit",
    "per_tensor_quant_fp8",
    "per_token_quant_fp8",
    "moe_align_block_size",
    "fast_topk_v2",
]
