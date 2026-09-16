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

import importlib

from tokenspeed_kernel.profiling import bootstrap_profiling_from_env

bootstrap_profiling_from_env()

_exports = {
    "NoKernelFoundError": ("tokenspeed_kernel.selection", "NoKernelFoundError"),
    "add3": ("tokenspeed_kernel.ops.activation", "add3"),
    "attn_merge_state": ("tokenspeed_kernel.ops.attention", "attn_merge_state"),
    "attn_res_fwd": ("tokenspeed_kernel.ops.residual", "attn_res_fwd"),
    "attn_res_fwd_available": (
        "tokenspeed_kernel.ops.residual",
        "attn_res_fwd_available",
    ),
    "argmax": ("tokenspeed_kernel.ops.sampling", "argmax"),
    "bmm": ("tokenspeed_kernel.ops.gemm", "bmm"),
    "dsv4_grouped_output_projection": (
        "tokenspeed_kernel.ops.gemm",
        "dsv4_grouped_output_projection",
    ),
    "dsv4_grouped_output_projection_plan": (
        "tokenspeed_kernel.ops.gemm",
        "dsv4_grouped_output_projection_plan",
    ),
    "dsv4_grouped_output_projection_process_weights": (
        "tokenspeed_kernel.ops.gemm",
        "dsv4_grouped_output_projection_process_weights",
    ),
    "dsv4_grouped_output_projection_warmup": (
        "tokenspeed_kernel.ops.gemm",
        "dsv4_grouped_output_projection_warmup",
    ),
    "dsv4_grouped_output_projection_warmup_model": (
        "tokenspeed_kernel.ops.gemm",
        "dsv4_grouped_output_projection_warmup_model",
    ),
    "dsv4_linear_fp32": ("tokenspeed_kernel.ops.gemm", "dsv4_linear_fp32"),
    "dsv4_mega_moe_apply": ("tokenspeed_kernel.ops.moe", "dsv4_mega_moe_apply"),
    "dsv4_mega_moe_plan": ("tokenspeed_kernel.ops.moe", "dsv4_mega_moe_plan"),
    "dsv4_mega_moe_process_weights": (
        "tokenspeed_kernel.ops.moe",
        "dsv4_mega_moe_process_weights",
    ),
    "dsv4_mega_moe_warmup": (
        "tokenspeed_kernel.ops.moe",
        "dsv4_mega_moe_warmup",
    ),
    "dsv4_select_experts": ("tokenspeed_kernel.ops.moe", "dsv4_select_experts"),
    "fp8_linear": ("tokenspeed_kernel.ops.gemm", "fp8_linear"),
    "fp8_quantize_dequantize": (
        "tokenspeed_kernel.ops.quantization",
        "fp8_quantize_dequantize",
    ),
    "gated_residual_combine": (
        "tokenspeed_kernel.ops.residual",
        "gated_residual_combine",
    ),
    "gated_residual_mix": (
        "tokenspeed_kernel.ops.residual",
        "gated_residual_mix",
    ),
    "grouped_gemma_rmsnorm": (
        "tokenspeed_kernel.ops.layernorm",
        "grouped_gemma_rmsnorm",
    ),
    "hadamard_transform": (
        "tokenspeed_kernel.ops.transform",
        "hadamard_transform",
    ),
    "has_flashinfer_cute_dsl_nvfp4_a16": (
        "tokenspeed_kernel.ops.gemm",
        "has_flashinfer_cute_dsl_nvfp4_a16",
    ),
    "kimi3_latent_projection": (
        "tokenspeed_kernel.ops.gemm",
        "kimi3_latent_projection",
    ),
    "kimi3_latent_projection_add3": (
        "tokenspeed_kernel.ops.gemm",
        "kimi3_latent_projection_add3",
    ),
    "kimi3_mla_qkv_gate_projection": (
        "tokenspeed_kernel.ops.gemm",
        "kimi3_mla_qkv_gate_projection",
    ),
    "kimi3_qkvfab_projection": (
        "tokenspeed_kernel.ops.gemm",
        "kimi3_qkvfab_projection",
    ),
    "kimi3_router_projection": (
        "tokenspeed_kernel.ops.gemm",
        "kimi3_router_projection",
    ),
    "kimi3_shared_down_projection": (
        "tokenspeed_kernel.ops.gemm",
        "kimi3_shared_down_projection",
    ),
    "kimi3_shared_situ_projection": (
        "tokenspeed_kernel.ops.gemm",
        "kimi3_shared_situ_projection",
    ),
    "mhc_fused_hc": ("tokenspeed_kernel.ops.residual", "mhc_fused_hc"),
    "mhc_mixes": ("tokenspeed_kernel.ops.residual", "mhc_mixes"),
    "mhc_post": ("tokenspeed_kernel.ops.residual", "mhc_post"),
    "mhc_pre": ("tokenspeed_kernel.ops.residual", "mhc_pre"),
    "mm": ("tokenspeed_kernel.ops.gemm", "mm"),
    "moe_apply": ("tokenspeed_kernel.ops.moe", "moe_apply"),
    "moe_plan": ("tokenspeed_kernel.ops.moe", "moe_plan"),
    "moe_process_weights": ("tokenspeed_kernel.ops.moe", "moe_process_weights"),
    "moe_sigmoid_bias_topk": (
        "tokenspeed_kernel.ops.moe",
        "moe_sigmoid_bias_topk",
    ),
    "moe_softmax_topk": ("tokenspeed_kernel.ops.moe", "moe_softmax_topk"),
    "native_latent_moe_available": (
        "tokenspeed_kernel.ops.moe",
        "native_latent_moe_available",
    ),
    "nvfp4_gemm_swiglu_nvfp4_quant": (
        "tokenspeed_kernel.ops.gemm",
        "nvfp4_gemm_swiglu_nvfp4_quant",
    ),
    "pack_topk_router_logits": (
        "tokenspeed_kernel.ops.moe",
        "pack_topk_router_logits",
    ),
    "prepare_fp8_linear": ("tokenspeed_kernel.ops.gemm", "prepare_fp8_linear"),
    "prepare_fp8_linear_activation": (
        "tokenspeed_kernel.ops.activation",
        "prepare_fp8_linear_activation",
    ),
    "prepare_gated_residual_weight_cache": (
        "tokenspeed_kernel.ops.residual",
        "prepare_gated_residual_weight_cache",
    ),
    "prepare_nvfp4_a16_weights": (
        "tokenspeed_kernel.ops.gemm",
        "prepare_nvfp4_a16_weights",
    ),
    "quantize_fp8": ("tokenspeed_kernel.ops.quantization", "quantize_fp8"),
    "quantize_fp8_with_scale": (
        "tokenspeed_kernel.ops.quantization",
        "quantize_fp8_with_scale",
    ),
    "quantize_mxfp4": ("tokenspeed_kernel.ops.quantization", "quantize_mxfp4"),
    "quantize_mxfp8": ("tokenspeed_kernel.ops.quantization", "quantize_mxfp8"),
    "quantize_nvfp4": ("tokenspeed_kernel.ops.quantization", "quantize_nvfp4"),
    "silu_and_mul": ("tokenspeed_kernel.ops.activation", "silu_and_mul"),
    "situ_and_mul": ("tokenspeed_kernel.ops.activation", "situ_and_mul"),
    "warmup_prepared_fp8_linears": (
        "tokenspeed_kernel.ops.gemm",
        "warmup_prepared_fp8_linears",
    ),
}

__all__ = sorted(_exports)


def __getattr__(name: str) -> object:
    target = _exports.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_exports))
