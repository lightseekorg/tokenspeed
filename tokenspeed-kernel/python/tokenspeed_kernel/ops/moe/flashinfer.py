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

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import Priority, error_fn, register_kernel

platform = current_platform()

cutlass_fused_moe = error_fn
fp4_quantize = error_fn
mxfp8_quantize = error_fn
nvfp4_block_scale_interleave = error_fn
trtllm_bf16_moe = error_fn
trtllm_fp4_block_scale_moe = error_fn
autotune = None
scaled_fp4_grouped_quantize = error_fn
silu_and_mul_scaled_nvfp4_experts_quantize = error_fn
grouped_gemm_nt_masked = error_fn
ActivationType = error_fn
_maybe_get_cached_w3_w1_permute_indices = error_fn
get_w2_permute_indices_with_cache = error_fn
convert_to_block_layout = error_fn
moe_wna16_marlin_gemm = error_fn

if platform.is_nvidia:
    try:
        from flashinfer import (
            cutlass_fused_moe,
            fp4_quantize,
            mxfp8_quantize,
            nvfp4_block_scale_interleave,
            trtllm_bf16_moe,
            trtllm_fp4_block_scale_moe,
        )
    except ImportError:
        pass

    try:
        from flashinfer.autotuner import autotune
    except ImportError:
        pass

    try:
        from flashinfer.cute_dsl import scaled_fp4_grouped_quantize
    except ImportError:
        pass

    try:
        from flashinfer.cute_dsl import silu_and_mul_scaled_nvfp4_experts_quantize
    except ImportError:
        pass

    try:
        from flashinfer.cute_dsl.blockscaled_gemm import grouped_gemm_nt_masked
    except ImportError:
        pass

    try:
        from flashinfer.fused_moe.core import (
            ActivationType,
            _maybe_get_cached_w3_w1_permute_indices,
            get_w2_permute_indices_with_cache,
        )
    except ImportError:
        pass

    try:
        from flashinfer.fused_moe.core import convert_to_block_layout
    except ImportError:
        pass

    try:
        from flashinfer.moe import moe_wna16_marlin_gemm
    except ImportError:
        pass

if trtllm_bf16_moe is not error_fn:
    trtllm_bf16_moe = register_kernel(
        "moe",
        "fused",
        name="flashinfer_trtllm_bf16_fused_moe",
        features={"self_routing"},
        solution="trtllm",
        dtypes={torch.bfloat16},
        traits={"weight_dtype": frozenset({"bf16"})},
        priority=Priority.SPECIALIZED,
        tags={"throughput"},
    )(trtllm_bf16_moe)

if cutlass_fused_moe is not error_fn:
    flashinfer_cutlass_fused_moe = register_kernel(
        "moe",
        "fused",
        name="flashinfer_cutlass_fused_moe",
        features={"pre_routed"},
        solution="flashinfer",
        dtypes={torch.bfloat16, torch.float16, torch.uint8, torch.float8_e4m3fn},
        traits={
            "weight_dtype": frozenset({"bf16", "fp8", "nvfp4"}),
            "tp": frozenset({True, False}),
            "ep": frozenset({True, False}),
            "cuda_graph": frozenset({False}),
        },
        priority=Priority.SPECIALIZED,
        tags={"throughput"},
    )(cutlass_fused_moe)

if trtllm_fp4_block_scale_moe is not error_fn:
    trtllm_fp4_block_scale_moe = register_kernel(
        "moe",
        "fused",
        name="flashinfer_trtllm_fp4_fused_moe",
        features={"self_routing"},
        solution="trtllm",
        dtypes={torch.uint8, torch.bfloat16},
        traits={"weight_dtype": frozenset({"mxfp4", "nvfp4"})},
        priority=Priority.SPECIALIZED,
        tags={"throughput"},
    )(trtllm_fp4_block_scale_moe)

try:
    if not platform.is_nvidia:
        raise ImportError
    from flashinfer.fused_moe import CuteDslMoEWrapper

    _cutedsl_wrapper_cache: dict[tuple, CuteDslMoEWrapper] = {}

    def _get_or_create_cutedsl_wrapper(wrapper_kwargs: dict) -> CuteDslMoEWrapper:
        """Return a cached wrapper for cuda-graph runs, or a fresh one otherwise.

        When ``use_cuda_graph=False`` (prefill / autotune), a new wrapper is
        created every time so its workspace doesn't stay resident.  When
        ``use_cuda_graph=True``, the wrapper is cached by its full param set.
        """
        if not wrapper_kwargs.get("use_cuda_graph", False):
            return CuteDslMoEWrapper(**wrapper_kwargs)

        cache_key = tuple(sorted(wrapper_kwargs.items()))
        if cache_key not in _cutedsl_wrapper_cache:
            _cutedsl_wrapper_cache[cache_key] = CuteDslMoEWrapper(**wrapper_kwargs)
        return _cutedsl_wrapper_cache[cache_key]

    @register_kernel(
        "moe",
        "fused",
        name="flashinfer_cutedsl_nvfp4_fused_moe",
        features={"pre_routed"},
        solution="cutedsl",
        dtypes={torch.uint8},
        traits={
            "weight_dtype": frozenset({"nvfp4"}),
            "tp": frozenset({False}),
            "ep": frozenset({True, False}),
            "cuda_graph": frozenset({True, False}),
        },
        priority=Priority.SPECIALIZED,
        tags={"throughput"},
    )
    def cutedsl_nvfp4_fused_moe(
        x_fp4,
        x_scale,
        topk_ids,
        topk_weights,
        w1_weight,
        w1_weight_sf,
        w1_alpha,
        fc2_input_scale,
        w2_weight,
        w2_weight_sf,
        w2_alpha,
        *,
        num_experts,
        top_k,
        num_local_experts,
        local_expert_offset,
        output_dtype,
        use_cuda_graph=False,
        capacity=None,
    ):
        wrapper_kwargs = dict(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=w2_weight.shape[1],
            intermediate_size=w1_weight.shape[1] // 2,
            use_cuda_graph=use_cuda_graph,
            num_local_experts=num_local_experts,
            local_expert_offset=local_expert_offset,
            output_dtype=output_dtype,
            device=str(w2_weight.device),
        )
        if use_cuda_graph:
            assert capacity is not None
            wrapper_kwargs["max_num_tokens"] = capacity

        wrapper = _get_or_create_cutedsl_wrapper(wrapper_kwargs)
        return wrapper.run(
            x=x_fp4,
            x_sf=x_scale,
            token_selected_experts=topk_ids,
            token_final_scales=topk_weights,
            w1_weight=w1_weight,
            w1_weight_sf=w1_weight_sf,
            w1_alpha=w1_alpha,
            fc2_input_scale=fc2_input_scale,
            w2_weight=w2_weight,
            w2_weight_sf=w2_weight_sf,
            w2_alpha=w2_alpha,
            tactic=None,
        )

except ImportError:
    pass
