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

"""FP8 (per-tensor static activation) x MXFP4 weight MoE backend.

Targets AMD Quark-quantised checkpoints such as
``amd/gpt-oss-120b-w-mxfp4-a-fp8`` where the experts ship MXFP4 weights and a
per-tensor static FP8 activation scale (``input_scale``). The implementation
re-uses the regular MXFP4 ``triton_kernels`` GEMM path and only changes the
left-hand-side numerics to FP8 via ``FlexCtx.lhs_data``.
"""

from __future__ import annotations

import tokenspeed_kernel
import torch
from tokenspeed_kernel.platform import current_platform
from torch import nn
from torch.nn.parameter import Parameter

from tokenspeed.runtime.layers.moe.backends.mxfp4.triton_kernel import (
    Mxfp4TritonKernelBackend,
    swizzle_mxfp4,
)
from tokenspeed.runtime.layers.moe.backends.mxfp4.weights import (
    MXFP4_BLOCK,
    create_mxfp4_weights,
)
from tokenspeed.runtime.layers.moe.core.types import MoELayerSpec
from tokenspeed.runtime.layers.quantization import Mxfp4Config
from tokenspeed.runtime.utils import set_weight_attrs


def _amd_fp8_dtype() -> torch.dtype:
    """Pick the FP8 e4m3 dtype matching the current AMD architecture."""
    platform = current_platform()
    # CDNA3 (gfx942 / MI300) uses ``fnuz``; CDNA4 (gfx950 / MI355) and Nvidia
    # use ``fn``.
    if getattr(platform, "is_cdna3", False):
        return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn


def _quantize_to_fp8(
    x: torch.Tensor, scale: torch.Tensor, fp8_dtype: torch.dtype
) -> torch.Tensor:
    """Quantize a row-major tensor ``x`` to FP8 with a per-tensor ``scale``.

    ``scale`` is the *dequantization* scale; the produced FP8 values are the
    direct input to ``triton_kernels.matmul`` together with
    ``InFlexData(dtype=fp8, scale=scale)``. The math is performed in float32
    so that the routine works regardless of whether ``x`` arrives in bf16 or
    fp8 (some matmul outputs ride on the input dtype).
    """
    fp8_max = float(torch.finfo(fp8_dtype).max)
    inv_scale = (1.0 / scale.to(torch.float32)).reshape(())
    return (
        (x.to(torch.float32) * inv_scale)
        .clamp_(min=-fp8_max, max=fp8_max)
        .to(fp8_dtype)
    )


def _per_tensor_input_scale_loader(
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    shard_id: str,
    local_expert_id: int,
) -> None:
    """Per-expert input-scale loader.

    The HF checkpoint stores one scalar per expert per GEMM. ``shard_id`` is
    one of ``w1`` / ``w3`` / ``w2`` (matching ``MoECheckpointLoader``); for the
    fused gate/up GEMM we keep the maximum across the two halves so the
    backend can collapse it into a single per-tensor activation scale.
    """
    value = loaded_weight.detach().to(torch.float32).reshape(())
    if shard_id in ("w1", "w3"):
        prev = param.data[local_expert_id]
        param.data[local_expert_id] = torch.maximum(prev, value)
    elif shard_id == "w2":
        param.data[local_expert_id] = value
    else:
        raise ValueError(f"Unknown shard_id for input_scale: {shard_id!r}")


def create_mxfp4_fp8_input_scales(
    layer: nn.Module, num_local_experts: int
) -> None:
    """Allocate per-expert ``w13_input_scale`` and ``w2_input_scale`` params."""
    w13 = nn.Parameter(
        torch.zeros(num_local_experts, dtype=torch.float32),
        requires_grad=False,
    )
    w2 = nn.Parameter(
        torch.zeros(num_local_experts, dtype=torch.float32),
        requires_grad=False,
    )
    layer.register_parameter("w13_input_scale", w13)
    layer.register_parameter("w2_input_scale", w2)
    set_weight_attrs(w13, {"weight_loader": _per_tensor_input_scale_loader})
    set_weight_attrs(w2, {"weight_loader": _per_tensor_input_scale_loader})


class Mxfp4Fp8TritonKernelBackend(Mxfp4TritonKernelBackend):
    """FP8 x MXFP4 MoE backend, intended for AMD CDNA4 (MI355).

    Selected for ``Mxfp4Config(is_w4a8_fp8=True)`` checkpoints, e.g. the AMD
    Quark export ``amd/gpt-oss-120b-w-mxfp4-a-fp8``.
    """

    supported_arches = frozenset({"any"})

    @classmethod
    def supports(cls, spec: MoELayerSpec, quant_config: object) -> bool:
        if not isinstance(quant_config, Mxfp4Config):
            return False
        if not getattr(quant_config, "is_w4a8_fp8", False):
            return False
        if not current_platform().is_amd:
            return False
        return spec.ep_size <= 1 and spec.activation in {"silu", "swiglu"}

    def create_layer_weights(
        self, layer: nn.Module, *, with_bias: bool = False
    ) -> None:
        super().create_layer_weights(layer, with_bias=with_bias)
        create_mxfp4_fp8_input_scales(layer, self.spec.num_local_experts)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        from tokenspeed_kernel.ops.moe.triton_kernels import (
            FlexCtx,
            InFlexData,
            PrecisionConfig,
        )

        MXFP_BLOCK_SIZE = 32
        fp8_dtype = _amd_fp8_dtype()

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

        # Collapse per-expert input scales to a single per-tensor scale per
        # GEMM. Quark exports the same scalar across experts for static
        # ``per_tensor`` quantisation; ``max`` is a safe reduction in case
        # individual experts reach slightly different values.
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
        layer._fp8_dtype = fp8_dtype

        # Force bf16 output so the swiglu / down-proj results stay in a
        # standard floating dtype; without this, ``triton_kernels.matmul``
        # defaults ``out_dtype`` to the input dtype (fp8) which would make
        # the subsequent reductions (and our re-quantisation step) blow up.
        out_dtype = torch.bfloat16

        layer.w13_precision_config = PrecisionConfig(
            flex_ctx=FlexCtx(
                lhs_data=InFlexData(dtype=fp8_dtype, scale=w13_in_scale),
                rhs_data=w13_flex,
            ),
            b_mx_scale=w13_scale,
            b_microblock_size=MXFP_BLOCK_SIZE,
            out_dtype=out_dtype,
        )
        layer.w2_precision_config = PrecisionConfig(
            flex_ctx=FlexCtx(
                lhs_data=InFlexData(dtype=fp8_dtype, scale=w2_in_scale),
                rhs_data=w2_flex,
            ),
            b_mx_scale=w2_scale,
            b_microblock_size=MXFP_BLOCK_SIZE,
            out_dtype=out_dtype,
        )

        layer.w13_weight_triton_tensor = w13_weight
        layer.w2_weight_triton_tensor = w2_weight
        del layer.w13_weight
        del layer.w2_weight
        # ``w*_input_scale`` parameters are no longer needed (collapsed into
        # the precision configs above) but keep them attached so downstream
        # code can still introspect the loader-populated values.
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
        w13_pc = layer.w13_precision_config
        w2_pc = layer.w2_precision_config
        fp8_dtype = layer._fp8_dtype

        gemm1_alpha = self._swiglu_arg.alpha if self._swiglu_arg else 1.702
        gemm1_clamp = self._swiglu_arg.limit if self._swiglu_arg else 7.0

        act = FusedActivation(
            FnSpecs("swiglu", swiglu_fn, ("alpha", "limit"), reduction_n=2),
            (gemm1_alpha, gemm1_clamp),
        )

        # Quantize hidden states with the per-tensor static FP8 scale baked
        # into ``w13_precision_config.flex_ctx.lhs_data``.
        x_fp8 = _quantize_to_fp8(hidden_states, layer.w13_act_scale, fp8_dtype)

        intermediate_cache = tokenspeed_kernel.moe_experts(
            x_fp8,
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

        intermediate_fp8 = _quantize_to_fp8(
            intermediate_cache, layer.w2_act_scale, fp8_dtype
        )

        return tokenspeed_kernel.moe_experts(
            intermediate_fp8,
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


__all__ = ["Mxfp4Fp8TritonKernelBackend", "create_mxfp4_fp8_input_scales"]
