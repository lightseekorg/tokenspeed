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
from tokenspeed_kernel.registry import error_fn
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

_flashinfer_mxfp4_permute_indices_cache: dict[tuple[str, torch.Size], torch.Tensor] = {}
_flashinfer_mxfp4_permute_indices_device_cache: dict[
    tuple[str, tuple[int, ...], int, int, str, int], torch.Tensor
] = {}
_is_nvidia = current_platform().is_nvidia


def _get_flashinfer_mxfp4_device_permute_indices(
    x: torch.Tensor,
    epilogue_tile_m: int,
    num_elts_per_sf: int | None = None,
    *,
    kind: str = "w2",
) -> torch.Tensor:
    from tokenspeed_kernel.ops.moe.flashinfer import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    extra_args = {} if num_elts_per_sf is None else {"num_elts_per_sf": num_elts_per_sf}
    if kind == "w13":
        permute_indices = _maybe_get_cached_w3_w1_permute_indices(
            _flashinfer_mxfp4_permute_indices_cache,
            x,
            epilogue_tile_m,
            **extra_args,
        )
    elif kind == "w2":
        permute_indices = get_w2_permute_indices_with_cache(
            _flashinfer_mxfp4_permute_indices_cache,
            x,
            epilogue_tile_m,
            **extra_args,
        )
    else:
        raise ValueError(f"unknown FlashInfer MXFP4 permute kind: {kind}")

    device_index = -1 if x.device.index is None else x.device.index
    num_elts_per_sf_key = -1 if num_elts_per_sf is None else num_elts_per_sf
    cache_key = (
        kind,
        tuple(x.shape),
        epilogue_tile_m,
        num_elts_per_sf_key,
        x.device.type,
        device_index,
    )
    cached_device_indices = _flashinfer_mxfp4_permute_indices_device_cache.get(
        cache_key
    )
    if cached_device_indices is None:
        cached_device_indices = permute_indices.to(x.device)
        _flashinfer_mxfp4_permute_indices_device_cache[cache_key] = (
            cached_device_indices
        )

    return cached_device_indices


def _float8_view_dtype(x: torch.Tensor) -> torch.dtype | None:
    float8_dtypes = tuple(
        dtype
        for dtype in (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e5m2", None),
            getattr(torch, "float8_e8m0fnu", None),
        )
        if dtype is not None
    )
    return x.dtype if x.dtype in float8_dtypes else None


def _pair_swap_rows(x: torch.Tensor, dim: int = -2) -> torch.Tensor:
    """Pair-swap adjacent rows along ``dim``: ``[r0, r1, r2, r3, ...]`` ->
    ``[r1, r0, r3, r2, ...]``.

    HF gpt-oss stores ``gate_up_proj_blocks`` interleaved as
    ``[w1_0, w3_0, w1_1, w3_1, ...]``. The fused gated activation expects
    ``[w3, w1]`` order per pair, so we swap within pairs without disturbing
    the interleaving.
    """
    view_dtype = _float8_view_dtype(x)
    if view_dtype is not None:
        x = x.view(torch.uint8)
    if dim < 0:
        dim += x.dim()
    shape = list(x.shape)
    if shape[dim] % 2 != 0:
        raise ValueError(f"expected even size in dim {dim}, got {shape[dim]}")
    new_shape = shape[:dim] + [shape[dim] // 2, 2] + shape[dim + 1 :]
    out = x.reshape(new_shape).flip(dim + 1).reshape(shape).contiguous()
    return out.view(view_dtype) if view_dtype is not None else out


def _reorder_w1w3_to_w3w1(x: torch.Tensor, dim: int = -2) -> torch.Tensor:
    """Reorder concatenated [w1, w3] expert tensors to [w3, w1] block layout.

    Used by loaders (e.g. the shared MoE checkpoint loader) that store
    ``[w1_all | w3_all]`` block-concatenated rather than interleaved.
    """
    view_dtype = _float8_view_dtype(x)
    if view_dtype is not None:
        x = x.view(torch.uint8)
    if dim < 0:
        dim += x.dim()
    size = x.shape[dim]
    if size % 2 != 0:
        raise ValueError(f"expected even size in dim {dim}, got {size}")
    first, second = x.split(size // 2, dim=dim)
    out = torch.cat([second, first], dim=dim).contiguous()
    return out.view(view_dtype) if view_dtype is not None else out


class Mxfp4FlashinferMxfp4Backend(MoEBackend):
    supported_arches = frozenset({"sm100"})

    def __init__(
        self,
        key,
        spec: MoELayerSpec,
        quant_config: object,
        routing_config: dict | None = None,
    ):
        from tokenspeed.runtime.utils.env import global_server_args_dict

        del routing_config
        self.key = key
        self.spec = spec
        self.quant_config = quant_config
        self._autotuned = False
        self._ispp_padded = 0
        self._hidden_padded = 0
        self._hidden_original = spec.hidden_size
        self._mxfp4_precision = global_server_args_dict.get(
            "flashinfer_mxfp4_moe_precision", "default"
        )

    @classmethod
    def supports(cls, spec: MoELayerSpec, quant_config: object) -> bool:
        return (
            _is_nvidia
            and isinstance(quant_config, Mxfp4Config)
            and spec.activation in {"silu", "swiglu"}
        )

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
            ispp_padded = round_up(ispp, 256)
            hidden_padded = round_up(hidden, 256)
        else:
            ispp_padded = ispp
            hidden_padded = hidden

        self._ispp_padded = ispp_padded
        self._hidden_padded = hidden_padded

        create_mxfp4_weights(
            self,
            layer,
            self.spec.num_local_experts,
            hidden_padded,
            ispp_padded,
            with_bias=with_bias,
        )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        from tokenspeed_kernel.ops.quantization.flashinfer import (
            nvfp4_block_scale_interleave,
        )

        sf_block_size = MXFP4_BLOCK
        num_experts = self.spec.num_local_experts
        ispp_padded = self._ispp_padded
        hidden_padded = self._hidden_padded

        # SwiGLU constants for the fused kernel.
        #   - alpha   = α in silu(α·gate)
        #   - beta    = β in (up + β); gpt-oss uses 1.0, standard SwiGLU None
        #   - limit   = clamp limit on gate (and up, in some recipes)
        # Models override the gpt-oss defaults via ``MoELayer(swiglu_limit=...,
        # activation_alpha=..., swiglu_beta=...)``.
        swiglu_arg = getattr(layer, "swiglu_arg", None)
        if swiglu_arg is None:
            alpha = 1.702
            limit = 7.0
            beta = 1.0
        else:
            alpha = swiglu_arg.alpha
            limit = swiglu_arg.limit
            beta = getattr(layer, "swiglu_beta", None)

        def _maybe_param(value):
            if value is None:
                return None
            return Parameter(
                torch.full(
                    (num_experts,),
                    float(value),
                    dtype=torch.float32,
                    device="cuda",
                ),
                requires_grad=False,
            )

        layer.gemm1_alpha = _maybe_param(alpha)
        layer.gemm1_beta = _maybe_param(beta)
        layer.gemm1_clamp_limit = _maybe_param(limit)

        w13_weight_scale = layer.w13_weight_scale.data
        w2_weight_scale = layer.w2_weight_scale.data
        w13_weight = layer.w13_weight.data
        w2_weight = layer.w2_weight.data
        w13_bias = layer.w13_weight_bias.data.to(torch.float32)
        w2_bias = layer.w2_weight_bias.data.to(torch.float32)

        # Swap w1 and w3 because the fused gated activation expects
        # [w3, w1] order. The exact transform depends on the loader's row
        # layout for ``w13_weight``:
        #   - "interleaved": HF gpt-oss copies ``gate_up_proj_blocks`` whose
        #     rows are ``[w1_0, w3_0, w1_1, w3_1, ...]``. Pair-swap within
        #     pairs and the kernel's mma shuffle (``kind="w2"``) is enough.
        #   - "concatenated": shared MoE checkpoint loader writes
        #     ``[w1_all | w3_all]`` block-concatenated. Block-swap to
        #     ``[w3 | w1]`` and let the gated-activation reorder
        #     (``kind="w13"``) interleave before shuffling.
        w13_layout = getattr(layer, "w13_input_layout", "concatenated")
        if w13_layout == "interleaved":
            w13_weight_scale = _pair_swap_rows(w13_weight_scale, -2)
            w13_weight = _pair_swap_rows(w13_weight, -2)
            w13_bias = _pair_swap_rows(w13_bias, -1)
            w13_permute_kind = "w2"
        elif w13_layout == "concatenated":
            w13_weight_scale = _reorder_w1w3_to_w3w1(w13_weight_scale, -2)
            w13_weight = _reorder_w1w3_to_w3w1(w13_weight, -2)
            w13_bias = _reorder_w1w3_to_w3w1(w13_bias, -1)
            w13_permute_kind = "w13"
        else:
            raise ValueError(f"unknown w13_input_layout: {w13_layout!r}")

        # Shuffle weights and scaling factors for transposed mma output
        gemm1_weights_shuffled = []
        gemm1_scales_shuffled = []
        gemm2_weights_shuffled = []
        gemm2_scales_shuffled = []
        gemm1_bias_shuffled = []
        gemm2_bias_shuffled = []
        epilogue_tile_m = 128

        w13_weight_perm = _get_flashinfer_mxfp4_device_permute_indices(
            w13_weight[0].view(torch.uint8),
            epilogue_tile_m,
            kind=w13_permute_kind,
        )
        w13_scale_perm = _get_flashinfer_mxfp4_device_permute_indices(
            w13_weight_scale[0].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
            kind=w13_permute_kind,
        )
        w13_bias_perm = _get_flashinfer_mxfp4_device_permute_indices(
            w13_bias[0].reshape(-1, 1), epilogue_tile_m
        )
        w2_weight_perm = _get_flashinfer_mxfp4_device_permute_indices(
            w2_weight[0].view(torch.uint8), epilogue_tile_m, kind="w2"
        )
        w2_scale_perm = _get_flashinfer_mxfp4_device_permute_indices(
            w2_weight_scale[0].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
            kind="w2",
        )
        w2_bias_perm = _get_flashinfer_mxfp4_device_permute_indices(
            w2_bias[0].reshape(-1, 1), epilogue_tile_m
        )

        for idx in range(num_experts):
            gemm1_weights_shuffled.append(
                w13_weight[idx].view(torch.uint8)[w13_weight_perm].contiguous()
            )
            gemm1_scales_shuffled.append(
                nvfp4_block_scale_interleave(
                    w13_weight_scale[idx].view(torch.uint8)[w13_scale_perm].contiguous()
                )
            )
            gemm1_bias_shuffled.append(
                w13_bias[idx].reshape(-1, 1)[w13_bias_perm].contiguous()
            )
            gemm2_weights_shuffled.append(
                w2_weight[idx].view(torch.uint8)[w2_weight_perm].contiguous()
            )
            gemm2_scales_shuffled.append(
                nvfp4_block_scale_interleave(
                    w2_weight_scale[idx].view(torch.uint8)[w2_scale_perm].contiguous()
                )
            )
            gemm2_bias_shuffled.append(
                w2_bias[idx].reshape(-1, 1)[w2_bias_perm].contiguous()
            )

        layer.w13_weight = Parameter(
            torch.stack(gemm1_weights_shuffled), requires_grad=False
        )
        layer.w13_weight_scale = Parameter(
            torch.stack(gemm1_scales_shuffled)
            .reshape(num_experts, 2 * ispp_padded, hidden_padded // sf_block_size)
            .view(torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.w2_weight = Parameter(
            torch.stack(gemm2_weights_shuffled), requires_grad=False
        )
        layer.w2_weight_scale = Parameter(
            torch.stack(gemm2_scales_shuffled)
            .reshape(num_experts, hidden_padded, ispp_padded // sf_block_size)
            .view(torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.w13_weight_bias = Parameter(
            torch.stack(gemm1_bias_shuffled).reshape(num_experts, -1),
            requires_grad=False,
        )
        layer.w2_weight_bias = Parameter(
            torch.stack(gemm2_bias_shuffled).reshape(num_experts, -1),
            requires_grad=False,
        )
        torch.cuda.empty_cache()

    def _call_kernel(self, router_logits, x_quant, x_scale, layer, top_k, output):
        from tokenspeed.runtime.utils import next_power_of_2

        num_local = self.spec.num_local_experts
        local_offset = self.spec.ep_rank * num_local
        return tokenspeed_kernel.moe_fused(
            router_logits.to(torch.bfloat16),
            None,
            x_quant,
            x_scale,
            layer.w13_weight,
            layer.w13_weight_scale,
            layer.w13_weight_bias,
            layer.gemm1_alpha,
            layer.gemm1_beta,
            layer.gemm1_clamp_limit,
            layer.w2_weight,
            layer.w2_weight_scale,
            layer.w2_weight_bias,
            None,
            None,
            None,
            self.spec.num_experts,
            top_k,
            None,
            None,
            self._ispp_padded,
            local_offset,
            num_local,
            None,
            1,
            True,
            tune_max_num_tokens=next_power_of_2(x_quant.shape[0]),
            output=output,
            dtype=torch.bfloat16,
            features={"self_routing"},
            weight_format="mxfp4",
            expected_kernel_name="flashinfer_trtllm_fp4_fused_moe",
        )[0]

    def forward(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        topk_output: object,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ) -> torch.Tensor:
        del num_global_tokens, max_num_tokens_per_gpu
        x = hidden_states
        hidden_padded = self._hidden_padded
        hidden_original = self._hidden_original

        # After dispatch, some ranks may receive 0 tokens. The
        # The fused kernel cannot handle empty input, so return
        # an empty tensor directly.
        if x.shape[0] == 0:
            return x.new_empty(0, hidden_original)

        # Quantize or pass through based on precision mode
        if self._mxfp4_precision == "bf16":
            assert x.dtype == torch.bfloat16
            x_quant = x
            x_scale = None
            if hidden_padded != x_quant.shape[-1]:
                x_quant = torch.nn.functional.pad(
                    x_quant,
                    (0, hidden_padded - x_quant.shape[-1]),
                    mode="constant",
                    value=0.0,
                )
        elif self._mxfp4_precision == "default":
            from tokenspeed_kernel.ops.quantization.flashinfer import mxfp8_quantize

            x_quant, x_scale = mxfp8_quantize(x, False, alignment=hidden_padded)
            x_scale = x_scale.view(torch.float8_e4m3fn).reshape(*x.shape[:-1], -1)
        else:
            raise NotImplementedError(
                f"Unknown flashinfer_mxfp4_moe_precision: {self._mxfp4_precision}"
            )

        assert x_quant.shape[-1] == hidden_padded

        # BypassedTopKOutput provides router_logits and topk_config
        top_k = topk_output.topk_config.top_k
        router_logits = topk_output.router_logits

        num_tokens = x_quant.shape[0]
        h_dim = (
            x_quant.shape[-1] * 2 if x_quant.dtype == torch.uint8 else x_quant.shape[-1]
        )
        output = torch.empty(
            num_tokens, h_dim, dtype=torch.bfloat16, device=x_quant.device
        )

        try:
            from tokenspeed_kernel.ops.moe.flashinfer import (
                autotune as flashinfer_autotune,
            )
        except ImportError:
            flashinfer_autotune = None

        # Autotune on first call to pre-compile all kernel variants.
        # Equivalent to tokenspeed's _flashinfer_autotune() which runs a dummy
        # forward inside autotune() context. Without this, calls with new
        # token counts trigger JIT compilation that desyncs TP ranks.
        if not self._autotuned and flashinfer_autotune not in (None, error_fn):
            with flashinfer_autotune():
                self._call_kernel(router_logits, x_quant, x_scale, layer, top_k, output)
            self._autotuned = True

        result = self._call_kernel(
            router_logits, x_quant, x_scale, layer, top_k, output
        )
        # Trim output to original (unpadded) hidden size if needed.
        if hidden_original != hidden_padded:
            result = result[:, :hidden_original].contiguous()
        return result


__all__ = ["Mxfp4FlashinferMxfp4Backend"]
