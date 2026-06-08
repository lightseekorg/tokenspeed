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

import tokenspeed_kernel
import torch
from tokenspeed_kernel.ops.moe.triton_kernels import (
    FlexCtx,
    InFlexData,
    PrecisionConfig,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import error_fn
from torch import nn
from torch.nn.parameter import Parameter

from tokenspeed.runtime.layers.moe.backends.mxfp4.triton_kernel import (
    Mxfp4TritonKernelBackend,
    swizzle_mxfp4,
)
from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec
from tokenspeed.runtime.layers.quantization import Mxfp4Config
from tokenspeed.runtime.layers.quantization.utils import should_ignore_quant_layer

# Block_N used by the gluon combine kernel.  Pre-padding w2 along N to a
# multiple of this lets the W_VIA_VGPR path drop its n-mask.
_GLUON_COMBINE_BLOCK_N = 128


def _pad_w2_to_block_n(layer: nn.Module, block_n: int) -> None:
    original_n = int(layer.w2_weight.shape[-2])
    layer._w2_logical_n = original_n

    if original_n % block_n == 0:
        return

    n_padded = (original_n + block_n - 1) // block_n * block_n
    extra_n = n_padded - original_n

    w2_w = layer.w2_weight.data
    w2_s = layer.w2_weight_scale.data
    w2_w_padded = torch.cat(
        [
            w2_w,
            torch.zeros(
                *w2_w.shape[:-2],
                extra_n,
                w2_w.shape[-1],
                dtype=w2_w.dtype,
                device=w2_w.device,
            ),
        ],
        dim=-2,
    )
    w2_s_padded = torch.cat(
        [
            w2_s,
            torch.full(
                (*w2_s.shape[:-2], extra_n, w2_s.shape[-1]),
                127,
                dtype=w2_s.dtype,
                device=w2_s.device,
            ),
        ],
        dim=-2,
    )
    layer.w2_weight = Parameter(w2_w_padded, requires_grad=False)
    layer.w2_weight_scale = Parameter(w2_s_padded, requires_grad=False)


def _attach_gluon_bpreshuffle(layer: nn.Module) -> None:
    try:
        from tokenspeed_kernel.ops.moe.gluon import (
            _extract_gluon_raw_w,
            shuffle_weight_for_gluon_dot_layout,
        )
    except ImportError:
        return
    if (
        _extract_gluon_raw_w is error_fn
        or shuffle_weight_for_gluon_dot_layout is error_fn
    ):
        return

    targets = (
        ("w13_weight_triton_tensor", None),
        ("w2_weight_triton_tensor", getattr(layer, "_w2_logical_n", None)),
    )
    for attr, logical_n in targets:
        wrapped = getattr(layer, attr, None)
        if wrapped is None:
            continue
        w_raw = _extract_gluon_raw_w(wrapped)
        try:
            w_shuffled = shuffle_weight_for_gluon_dot_layout(w_raw)
        except (ValueError, AssertionError):
            continue
        if logical_n is not None and logical_n != w_shuffled.shape[-1]:
            # combine GEMM: backend pre-padded N to a multiple of BLOCK_N.
            # Stamp the logical N on BOTH the shuffled tensor (consumed by
            # ``gluon_mxfp_combine`` on the W_VIA_VGPR path) and the raw
            # tensor (consumed by the LDS fallback when BLOCK_M<=32).
            w_shuffled.original_n = int(logical_n)
            w_raw.original_n = int(logical_n)
        w_raw._gluon_shuffled = w_shuffled


class Mxfp4GluonKernelBackend(Mxfp4TritonKernelBackend):
    """MXFP4 + FP8-activation MoE backend that dispatches gluon kernels."""

    supported_arches = frozenset({"any"})

    @classmethod
    def supports(cls, spec: MoELayerSpec, quant_config: object) -> bool:
        import os

        if not isinstance(quant_config, Mxfp4Config):
            return False
        if should_ignore_quant_layer(
            prefix=spec.prefix,
            ignored_layers=getattr(quant_config, "ignored_layers", []) or [],
        ):
            return False
        # Gluon kernels are only validated for AMD w-mxfp4 / a-fp8.
        if not quant_config.is_w4a8_fp8:
            return False
        if not current_platform().is_amd:
            return False
        if os.environ.get("TOKENSPEED_MOE_GLUON", "").strip().lower() in {
            "0",
            "false",
            "no",
            "off",
            "disable",
            "disabled",
        }:
            return False
        return spec.ep_size <= 1 and spec.activation in {"silu", "swiglu"}

    def process_weights_after_loading(self, layer: nn.Module) -> None:

        MXFP_BLOCK_SIZE = 32

        w13_weight_bias = layer.w13_weight_bias.to(torch.float32)
        w2_weight_bias = layer.w2_weight_bias.to(torch.float32)
        layer.w13_weight_bias = Parameter(w13_weight_bias, requires_grad=False)
        layer.w2_weight_bias = Parameter(w2_weight_bias, requires_grad=False)

        num_warps = 8
        w13_weight, w13_flex, w13_scale = swizzle_mxfp4(
            layer.w13_weight, layer.w13_weight_scale, num_warps
        )
        w2_weight, w2_flex, w2_scale = swizzle_mxfp4(
            layer.w2_weight, layer.w2_weight_scale, num_warps
        )

        # Collapse per-expert input scales to a single per-tensor scale
        # per GEMM. Quark exports a constant value across experts for
        # static ``per_tensor`` quantisation; ``max`` is a safe reduction
        # in case individual experts reach slightly different values.
        w13_in_scale = (
            layer.w13_input_scale.data.to(torch.float32)
            .max()
            .reshape(1)
            .to(layer.w13_input_scale.device)
            .contiguous()
        )
        w2_in_scale = (
            layer.w2_input_scale.data.to(torch.float32)
            .max()
            .reshape(1)
            .to(layer.w2_input_scale.device)
            .contiguous()
        )
        layer.w13_act_scale = w13_in_scale
        layer.w2_act_scale = w2_in_scale

        fp8_dtype = current_platform().fp8e4m3fn.dtype
        w13_lhs = InFlexData(dtype=fp8_dtype, scale=w13_in_scale)
        w2_lhs = InFlexData(dtype=fp8_dtype, scale=w2_in_scale)
        # Force bf16 output so the swiglu / down-proj results stay in a
        # standard floating dtype; without this, ``triton_kernels.matmul``
        # defaults ``out_dtype`` to the input dtype (fp8) which would
        # make the subsequent reductions / re-quantisation blow up.
        out_dtype = torch.bfloat16

        layer.w13_precision_config = PrecisionConfig(
            flex_ctx=FlexCtx(lhs_data=w13_lhs, rhs_data=w13_flex),
            b_mx_scale=w13_scale,
            b_microblock_size=MXFP_BLOCK_SIZE,
            out_dtype=out_dtype,
        )
        layer.w2_precision_config = PrecisionConfig(
            flex_ctx=FlexCtx(lhs_data=w2_lhs, rhs_data=w2_flex),
            b_mx_scale=w2_scale,
            b_microblock_size=MXFP_BLOCK_SIZE,
            out_dtype=out_dtype,
        )

        layer.w13_weight_triton_tensor = w13_weight
        layer.w2_weight_triton_tensor = w2_weight
        del layer.w13_weight
        del layer.w2_weight

        torch.cuda.empty_cache()

    def forward(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        topk_output: object,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ) -> torch.Tensor:
        del num_global_tokens, max_num_tokens_per_gpu
        router_logits = topk_output.router_logits
        top_k = topk_output.topk_config.top_k

        swiglu_alpha = self._swiglu_arg.alpha if self._swiglu_arg else 1.702
        swiglu_limit = self._swiglu_arg.limit if self._swiglu_arg else 7.0

        # All of routing, dispatch GEMM (with fused SwiGLU + FP8 output
        # quant) and combine GEMM are encapsulated by the registered
        # ``gluon_mxfp4_fp8_fused_moe`` kernel.  The ``self_routing``
        # feature + ``activation_dtype=fp8`` trait + (bf16/fp16 dense or
        # FP8 per-tensor x, mxfp4 weight) signature uniquely select the
        # gluon implementation on gfx950.  When ``hidden_states`` is
        # already FP8 (e.g. fused into an upstream layer), advertise the
        # per-tensor FP8 input so the matching signature is built.
        is_fp8_input = hidden_states.dtype in (
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        )
        return tokenspeed_kernel.moe_fused(
            hidden_states,
            router_logits,
            layer.w13_weight_triton_tensor,
            layer.w2_weight_triton_tensor,
            w13_bias=getattr(layer, "w13_weight_bias", None),
            w2_bias=getattr(layer, "w2_weight_bias", None),
            w13_precision_config=getattr(layer, "w13_precision_config", None),
            w2_precision_config=getattr(layer, "w2_precision_config", None),
            w13_act_scale=layer.w13_act_scale,
            w2_act_scale=layer.w2_act_scale,
            top_k=top_k,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            dtype=hidden_states.dtype,
            weight_format="mxfp4",
            fp8_scale_granularity="tensor" if is_fp8_input else "block",
            features={"self_routing"},
            traits={"activation_dtype": "fp8"},
            expected_kernel_name="gluon_mxfp4_fp8_fused_moe",
        )


__all__ = ["Mxfp4GluonKernelBackend"]
