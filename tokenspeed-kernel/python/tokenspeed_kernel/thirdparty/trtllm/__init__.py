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

"""Kernel dispatch for trtllm-kernel quantization, MoE, and mHC helpers.

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
fast_topk_v2 = error_fn
mhc_big_fuse = error_fn
mhc_fused_hc = error_fn
mhc_post_mapping = error_fn


def has_mhc_kernels() -> bool:
    """Return whether the installed TRT-LLM extension exposes all mHC ops."""

    return False


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

        def fast_topk_v2(
            values: torch.Tensor,
            seq_lens: torch.Tensor,
            indices: torch.Tensor,
            topk: int,
            next_n: int = 1,
        ):
            seq_lens = seq_lens.to(torch.int32).reshape(-1).contiguous()
            if next_n == 1:
                torch.ops.trtllm.indexer_topk_decode(
                    values, seq_lens, indices, next_n, topk
                )
            else:
                row_ends = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
                row_starts = row_ends - seq_lens
                torch.ops.trtllm.indexer_topk_prefill(
                    values, row_starts, row_ends, indices, topk
                )

        def has_mhc_kernels() -> bool:
            """Return whether the installed TRT-LLM extension exposes all mHC ops."""

            return all(
                hasattr(torch.ops.trtllm, name)
                for name in ("mhc_big_fuse", "mhc_fused_hc", "mhc_post_mapping")
            )

        def mhc_big_fuse(
            y_acc: torch.Tensor,
            r_acc: torch.Tensor,
            residual: torch.Tensor,
            hc_scale: torch.Tensor,
            hc_base: torch.Tensor,
            post_mix: torch.Tensor,
            comb_mix: torch.Tensor,
            layer_input: torch.Tensor,
            num_tokens: int,
            hc_dim: int,
            hidden_size: int,
            rms_eps: float,
            hc_pre_eps: float,
            hc_sinkhorn_eps: float,
            hc_post_mult_value: float,
            sinkhorn_repeat: int,
            num_splits: int,
            block_size: int,
        ) -> None:
            """Launch TRT-LLM's fused mHC pre-mapping epilogue.

            Args:
                y_acc: FP32 GEMM accumulators.
                r_acc: FP32 residual square-sum accumulators.
                residual: BF16 hyper-connected residual input.
                hc_scale: FP32 pre/post/comb scale values.
                hc_base: FP32 pre/post/comb bias values.
                post_mix: FP32 post-mapping output buffer.
                comb_mix: FP32 combination-matrix output buffer.
                layer_input: BF16 reduced layer-input output buffer.
                num_tokens: Flattened token count.
                hc_dim: Flattened hyper-connected hidden dimension.
                hidden_size: Per-lane hidden dimension.
                rms_eps: RMS normalization epsilon.
                hc_pre_eps: Pre-mapping sigmoid epsilon.
                hc_sinkhorn_eps: Sinkhorn normalization epsilon.
                hc_post_mult_value: Post-mapping sigmoid multiplier.
                sinkhorn_repeat: Number of Sinkhorn iterations.
                num_splits: Number of split-K accumulator partitions.
                block_size: CUDA block size selected for the epilogue.

            Returns:
                None. ``post_mix``, ``comb_mix``, and ``layer_input`` are
                populated in place.
            """

            torch.ops.trtllm.mhc_big_fuse(
                y_acc,
                r_acc,
                residual,
                hc_scale,
                hc_base,
                post_mix,
                comb_mix,
                layer_input,
                num_tokens,
                hc_dim,
                hidden_size,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                num_splits,
                block_size,
            )

        def mhc_post_mapping(
            residual: torch.Tensor,
            hidden_states: torch.Tensor,
            post_mix: torch.Tensor,
            comb_mix: torch.Tensor,
            output: torch.Tensor,
            num_tokens: int,
            hidden_size: int,
        ) -> None:
            """Launch TRT-LLM's mHC post-mapping kernel.

            Args:
                residual: BF16 hyper-connected residual input.
                hidden_states: BF16 layer output to mix into the residual.
                post_mix: FP32 per-lane post-mapping weights.
                comb_mix: FP32 lane-combination matrices.
                output: BF16 hyper-connected residual output buffer.
                num_tokens: Flattened token count.
                hidden_size: Per-lane hidden dimension.

            Returns:
                None. ``output`` is populated in place.
            """

            torch.ops.trtllm.mhc_post_mapping(
                residual,
                hidden_states,
                post_mix,
                comb_mix,
                output,
                num_tokens,
                hidden_size,
            )

        def mhc_fused_hc(
            x_prev: torch.Tensor,
            residual_prev: torch.Tensor,
            post_mix_prev: torch.Tensor,
            comb_mix_prev: torch.Tensor,
            weight: torch.Tensor,
            hc_scale: torch.Tensor,
            hc_base: torch.Tensor,
            residual_cur: torch.Tensor,
            post_mix_cur: torch.Tensor,
            comb_mix_cur: torch.Tensor,
            layer_input_cur: torch.Tensor,
            y_acc_workspace: torch.Tensor,
            r_acc_workspace: torch.Tensor,
            done_counter_workspace: torch.Tensor,
            num_tokens: int,
            hidden_size: int,
            hc_mult: int,
            rms_eps: float,
            hc_pre_eps: float,
            hc_sinkhorn_eps: float,
            hc_post_mult_value: float,
            sinkhorn_repeat: int,
            backend: int,
            tile_n: int,
            num_k_splits: int,
            bigfuse_block_size: int,
            tile_m: int,
            norm_weight: torch.Tensor | None,
            norm_eps: float,
        ) -> None:
            """Launch TRT-LLM's fused previous-post/current-pre mHC kernel.

            Args:
                x_prev: BF16 previous layer output.
                residual_prev: BF16 previous hyper-connected residual.
                post_mix_prev: FP32 previous post-mapping weights.
                comb_mix_prev: FP32 previous lane-combination matrices.
                weight: FP32 current pre-mapping projection weight.
                hc_scale: FP32 current pre/post/comb scale values.
                hc_base: FP32 current pre/post/comb bias values.
                residual_cur: BF16 current residual output buffer.
                post_mix_cur: FP32 current post-mapping output buffer.
                comb_mix_cur: FP32 current combination output buffer.
                layer_input_cur: BF16 current layer-input output buffer.
                y_acc_workspace: FP32 projection accumulator workspace.
                r_acc_workspace: FP32 square-sum accumulator workspace.
                done_counter_workspace: INT32 kernel synchronization workspace.
                num_tokens: Flattened token count.
                hidden_size: Per-lane hidden dimension.
                hc_mult: Number of hyper-connection lanes.
                rms_eps: RMS normalization epsilon.
                hc_pre_eps: Pre-mapping sigmoid epsilon.
                hc_sinkhorn_eps: Sinkhorn normalization epsilon.
                hc_post_mult_value: Post-mapping sigmoid multiplier.
                sinkhorn_repeat: Number of Sinkhorn iterations.
                backend: TRT-LLM fused-mHC backend code.
                tile_n: FMA output tile size.
                num_k_splits: Hidden-axis split count.
                bigfuse_block_size: Block size for half-fused backends.
                tile_m: Tokens processed per CTA by all-in-one FMA.
                norm_weight: Optional fused RMSNorm weight.
                norm_eps: Optional fused RMSNorm epsilon.

            Returns:
                None. The four current-state outputs and workspaces are
                populated in place.
            """

            torch.ops.trtllm.mhc_fused_hc(
                x_prev,
                residual_prev,
                post_mix_prev,
                comb_mix_prev,
                weight,
                hc_scale,
                hc_base,
                residual_cur,
                post_mix_cur,
                comb_mix_cur,
                layer_input_cur,
                y_acc_workspace,
                r_acc_workspace,
                done_counter_workspace,
                num_tokens,
                hidden_size,
                hc_mult,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                backend,
                tile_n,
                num_k_splits,
                bigfuse_block_size,
                tile_m,
                norm_weight,
                norm_eps,
            )


__all__ = [
    "dsv3_fused_a_gemm",
    "fp8_blockwise_scaled_mm",
    "per_token_group_quant_8bit",
    "per_tensor_quant_fp8",
    "per_token_quant_fp8",
    "fast_topk_v2",
    "has_mhc_kernels",
    "mhc_big_fuse",
    "mhc_fused_hc",
    "mhc_post_mapping",
]
