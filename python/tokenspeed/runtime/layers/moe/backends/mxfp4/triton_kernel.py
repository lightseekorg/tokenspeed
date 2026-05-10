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
from tokenspeed_kernel.platform import current_platform
from torch import nn
from torch.nn.parameter import Parameter

from tokenspeed.runtime.layers.moe.backends.base import MoEBackend
from tokenspeed.runtime.layers.moe.backends.mxfp4.weights import (
    MXFP4_BLOCK,
    create_mxfp4_weights,
)
from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec
from tokenspeed.runtime.layers.moe.topk import TopKOutputFormat
from tokenspeed.runtime.layers.quantization import Mxfp4Config

_is_nvidia = current_platform().is_nvidia
_is_blackwell = current_platform().is_blackwell
_is_hopper = current_platform().is_hopper
_is_amd = current_platform().is_amd


def swizzle_mxfp4(quant_tensor, scale, num_warps):
    """Weight swizzle for mxfp4 MoE, used for OAI mxfp4 kernel."""
    from tokenspeed_kernel.ops.moe.triton_kernels import (
        FP4,
        InFlexData,
        convert_layout,
        layout,
        opt_flags,
        wrap_torch_tensor,
    )

    if layout is None:
        raise RuntimeError("triton_kernels backend unavailable")

    value_layout = layout.make_default_matmul_mxfp4_w_layout(mx_axis=-2)
    scale_layout = layout.make_default_matmul_mxfp4_w_scale_layout(
        mx_axis=-2, num_warps=num_warps
    )
    if _is_blackwell:
        constraints = {
            "is_persistent": True,
            "epilogue_subtile": 1,
        }
        opt_flags.update_opt_flags_constraints(constraints)
    elif _is_hopper:
        constraints = {
            "split_k": 1,
        }
        opt_flags.update_opt_flags_constraints(constraints)
    elif _is_amd:
        # Fix block_k=256 to support scale swizzling.
        constraints = {
            "block_k": 256,
        }
        opt_flags.update_opt_flags_constraints(constraints)
    # transpose the tensor so that the quantization axis is on dim1
    quant_tensor = quant_tensor.transpose(-2, -1)
    scale = scale.transpose(-2, -1)
    quant_tensor = convert_layout(
        wrap_torch_tensor(quant_tensor, dtype=FP4), value_layout
    )
    scale = convert_layout(wrap_torch_tensor(scale), scale_layout)
    return quant_tensor, InFlexData(), scale


class Mxfp4TritonKernelBackend(MoEBackend):
    supported_arches = frozenset({"any"})

    def __init__(
        self,
        key,
        spec: MoELayerSpec,
        quant_config: object,
        routing_config: dict | None = None,
    ):
        del routing_config
        self.key = key
        self.spec = spec
        self.quant_config = quant_config
        self._activation: str | None = None
        self._swiglu_arg = None

    @classmethod
    def supports(cls, spec: MoELayerSpec, quant_config: object) -> bool:
        if not isinstance(quant_config, Mxfp4Config):
            return False
        return spec.ep_size <= 1 and spec.activation in {
            "silu",
            "swiglu",
        }

    @property
    def topk_output_format(self) -> TopKOutputFormat:
        return TopKOutputFormat.BYPASSED

    def create_layer_weights(
        self, layer: nn.Module, *, with_bias: bool = False
    ) -> None:
        from tokenspeed.runtime.utils import round_up

        hidden = self.spec.hidden_size
        ispp = self.spec.intermediate_size // self.spec.tp_size

        if current_platform().is_blackwell:
            ispp_padded = round_up(ispp, 64)
        else:
            ispp_padded = round_up(ispp, MXFP4_BLOCK)
        hidden_padded = hidden

        create_mxfp4_weights(
            self,
            layer,
            self.spec.num_local_experts,
            hidden_padded,
            ispp_padded,
            with_bias=with_bias,
        )

        self._activation = layer.activation
        self._swiglu_arg = getattr(layer, "swiglu_arg", None)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        from tokenspeed_kernel.ops.moe.triton_kernels import (
            FlexCtx,
            PrecisionConfig,
        )

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

        layer.w13_precision_config = PrecisionConfig(
            flex_ctx=FlexCtx(rhs_data=w13_flex),
            b_mx_scale=w13_scale,
            b_microblock_size=MXFP_BLOCK_SIZE,
        )
        layer.w2_precision_config = PrecisionConfig(
            flex_ctx=FlexCtx(rhs_data=w2_flex),
            b_mx_scale=w2_scale,
            b_microblock_size=MXFP_BLOCK_SIZE,
        )

        layer.w13_weight_triton_tensor = w13_weight
        layer.w2_weight_triton_tensor = w2_weight
        # Free original weights (replaced by shuffled versions)
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
        from tokenspeed_kernel.ops.moe.triton_kernels import (
            FnSpecs,
            FusedActivation,
            swiglu_fn,
        )

        router_logits = topk_output.router_logits
        top_k = topk_output.topk_config.top_k
        n_tokens = router_logits.shape[0]

        ragged_metadata, gather_indx, scatter_indx, gate_scal = (
            tokenspeed_kernel.moe_route(
                router_logits,
                top_k,
                sm_first=False,
                dtype=router_logits.dtype,
                traits={"output_type": "ragged_metadata"},
                expected_kernel_name="triton_kernels_routing",
            )
        )

        w13_weight = layer.w13_weight_triton_tensor
        w2_weight = layer.w2_weight_triton_tensor
        w13_bias = getattr(layer, "w13_weight_bias", None)
        w2_bias = getattr(layer, "w2_weight_bias", None)
        w13_pc = getattr(layer, "w13_precision_config", None)
        w2_pc = getattr(layer, "w2_precision_config", None)

        gemm1_alpha = self._swiglu_arg.alpha if self._swiglu_arg else 1.702
        gemm1_clamp = self._swiglu_arg.limit if self._swiglu_arg else 7.0

        act = FusedActivation(
            FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2),
            (gemm1_alpha, gemm1_clamp),
        )

        # First GEMM: gate_up projection with fused activation
        intermediate_cache = tokenspeed_kernel.moe_experts(
            hidden_states,
            w13_weight,
            w13_bias,
            a_ragged_metadata=ragged_metadata,
            gather_indx=gather_indx,
            precision_config=w13_pc,
            fused_activation=act,
            dtype=hidden_states.dtype,
            features={"ragged_metadata", "dispatch_gemm"},
            traits={"weight_dtype": "mxfp4"},
            expected_kernel_name="triton_kernels_dispatch_gemm",
        )

        # Second GEMM: down projection with scatter (combine)
        # gammas applies the routing weights (expert contribution weights)
        return tokenspeed_kernel.moe_experts(
            intermediate_cache,
            w2_weight,
            w2_bias,
            a_ragged_metadata=ragged_metadata,
            scatter_indx=scatter_indx,
            precision_config=w2_pc,
            gammas=gate_scal,
            n_tokens=n_tokens,
            n_expts_act=top_k,
            dtype=hidden_states.dtype,
            features={"ragged_metadata", "gemm_combine"},
            traits={"weight_dtype": "mxfp4"},
            expected_kernel_name="triton_kernels_gemm_combine",
        )


__all__ = ["Mxfp4TritonKernelBackend"]
