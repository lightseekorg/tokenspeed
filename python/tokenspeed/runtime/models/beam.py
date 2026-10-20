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

"""Beam baseline assembled from the existing decoder, attention and MoE layers."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from copy import copy
from typing import Any

import torch
import torch.distributed as dist
from compressed_tensors.quantization import QuantizationStrategy
from tokenspeed_kernel.ops.activation.triton import (
    sigmoid_mul,
    tanh_softcap,
    tanh_softcap_rejection,
)
from tokenspeed_kernel.ops.gemm import mm as kernel_mm
from tokenspeed_kernel.ops.gemm import mv as kernel_mv
from tokenspeed_kernel.ops.gemm import (
    mv_weight_split,
)
from tokenspeed_kernel.ops.layernorm.qk_rmsnorm_ssmax import qk_rmsnorm_ssmax
from tokenspeed_kernel.ops.layernorm.triton import (
    UnscaledFP8QKVGate,
    fused_qk_rmsnorm_rope,
    rmsnorm,
)
from tokenspeed_kernel.ops.moe._triton.router_select import (
    router_route_rejection,
    triton_router_route_lanepar,
)
from tokenspeed_kernel.ops.moe.grouped_route import torch_grouped_route
from tokenspeed_kernel.ops.quantization import (
    quantize_fp8,
    rmsnorm_quantize_fp8,
    select_fp8_token_quantizer,
    silu_and_mul_quantize_fp8,
)
from tokenspeed_kernel.ops.quantization.flashinfer import fp4_quantize
from tokenspeed_kernel.platform import current_platform, pdl_enabled
from torch import nn

from tokenspeed.runtime.configs.model_config import beam_default_platform
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.execution.forward_step import (
    get_is_capture_mode,
    get_is_cuda_graph_phase,
)
from tokenspeed.runtime.layers.attention.gating import apply_attention_output_gate
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
)
from tokenspeed.runtime.layers.logits_processor import LogitsProcessor
from tokenspeed.runtime.layers.moe import (
    ExpertCheckpointSchema,
    build_moe_checkpoint_loader,
)
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.layers.moe.utils import get_moe_backend
from tokenspeed.runtime.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsLinearMethod,
)
from tokenspeed.runtime.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
    CompressedTensorsW8A8Fp8,
    fp8_channel_gemm_available,
    fp8_dynamic_linear,
    fp8_unscaled_gemm_available,
    fp8_unscaled_linear,
)
from tokenspeed.runtime.layers.rotary_embedding import RotaryEmbedding
from tokenspeed.runtime.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from tokenspeed.runtime.model_loader.utils import set_default_torch_dtype
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.base import (
    BaseCausalLM,
    BaseDecoderLayer,
    BaseTransformerModel,
)
from tokenspeed.runtime.models.beam_quant import (
    beam_checkpoint_format,
    beam_quant_policy,
)
from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3MoE
from tokenspeed.runtime.models.llama import LlamaMLP
from tokenspeed.runtime.models.qwen3 import Qwen3Attention
from tokenspeed.runtime.utils import add_prefix, make_layers, set_weight_attrs
from tokenspeed.runtime.utils.cuda_stream import StreamFork
from tokenspeed.runtime.utils.env import global_server_args_dict
from tokenspeed.runtime.utils.server_args import spec_rope_positions

logger = logging.getLogger(__name__)

_CHECKPOINT_DTYPE = torch.bfloat16
_ROUTER_DTYPE = torch.float32

BEAM_MOE_PRECISION_POLICY = "bf16_route_accum_fp32"
BEAM_MOE_VENDOR_POLICY = "bf16_vendor_fp32_route"
# The vendor policy on the NVFP4 experts of a ModelOpt NVFP4 checkpoint.
BEAM_MOE_NVFP4_POLICY = "nvfp4_vendor_fp32_route"
# The vendor policy on the FP8 per-channel experts of a compressed-tensors FP8
# dynamic W8A8 checkpoint.
BEAM_MOE_FP8_POLICY = "fp8ch_vendor_fp32_route"


def beam_kernels() -> bool:
    """Whether Beam's Triton kernels serve its calls: NVIDIA SM100 to SM103, where
    each was checked byte for byte against the Torch statements it replaces
    (``beam_default_platform``). Elsewhere Beam runs the statements."""
    return not beam_default_platform()


# The hidden size the residual boundary's one-launch kernel was checked at.
BEAM_HIDDEN = 5120


def scale_pair_rejection(x_scale: object, residual_scale: object) -> str:
    """Why a depth-scale pair cannot be taken; empty when it can: both ``None``
    (the unscaled call) or both finite Python floats."""
    if (x_scale is None) != (residual_scale is None):
        return (
            "x_scale and residual_scale must both be None or both be Python floats, "
            f"got {x_scale!r} and {residual_scale!r}"
        )
    for name, value in (("x_scale", x_scale), ("residual_scale", residual_scale)):
        if value is not None and not (
            isinstance(value, float) and math.isfinite(value)
        ):
            return f"{name} must be a finite Python float, got {value!r}"
    return ""


def _round_residual_admits(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None,
) -> bool:
    """Whether the shared kernel's residual arm with the BF16 sum rounding serves
    the residual boundary in one launch: BF16 contiguous ``[rows, 5120]`` inputs
    with at least one row on one CUDA device, Beam's platforms, no ``out``."""
    return (
        out is None
        and x.is_cuda
        and x.ndim == 2
        and x.shape[0] > 0
        and x.shape[1] == BEAM_HIDDEN
        and residual.shape == x.shape
        and x.dtype == residual.dtype == weight.dtype == torch.bfloat16
        and x.is_contiguous()
        and residual.is_contiguous()
        and weight.is_contiguous()
        and beam_kernels()
    )


def depth_scale_coefficients(
    layer_id: int,
    is_attention: bool,
) -> tuple[float, float]:
    """Return the sublayer and residual coefficients for one residual add."""

    if layer_id < 0:
        raise ValueError(f"layer_id must be non-negative, got {layer_id}")
    one_based_layer = layer_id + 1
    depth = 2 * one_based_layer + (0 if is_attention else 1)
    return (
        1.0 / math.sqrt(float(depth)),
        math.sqrt(float(depth - 1)) / math.sqrt(float(depth)),
    )


class BeamRMSNorm(RMSNorm):
    """Round the residual sum first; normalize and affine together in FP32.
    The weight is stored in the checkpoint dtype.

    An input norm whose attention runs its Q/K/V and gate projections as one
    FP8 GEMM (``BeamDecoderLayer`` sets ``_token_fp8_consumer``) normalizes
    that attention's decode rows with ``rmsnorm_quantize_fp8``
    (``quantization.rmsnorm_fp8``), whose launch also writes them quantized per
    token, and offers the codes to the attention, which then skips its own
    quantization of the rows.
    """

    # No depth-scale pair of its own: a scale reaches this norm only as an
    # explicit keyword of the call (the attention boundary) or not at all.
    residual_fold: tuple[float, float] | None = None

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype) -> None:
        super().__init__(hidden_size=hidden_size, eps=eps)
        self.weight.data = self.weight.data.to(dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        *,
        x_scale: float | None = None,
        residual_scale: float | None = None,
    ):
        if residual is None and (x_scale is not None or residual_scale is not None):
            raise ValueError("x_scale and residual_scale need a residual")
        consumer = self.__dict__.get("_token_fp8_consumer")
        if out is None and consumer is not None and consumer.takes_token_fp8(x):
            return self._forward_token_fp8(
                consumer, x, residual, x_scale, residual_scale
            )
        if residual is not None:
            rejection = scale_pair_rejection(x_scale, residual_scale)
            if rejection:
                raise ValueError(f"x_scale / residual_scale: {rejection}")
            weight = self.weight.data
            if _round_residual_admits(x, residual, weight, out):
                # The statements below in one launch, byte for byte.
                return rmsnorm(
                    x,
                    weight,
                    self.variance_epsilon,
                    residual=residual,
                    out=None,
                    enable_pdl=None,
                    round_residual_sum_bf16=True,
                    x_scale=x_scale,
                    residual_scale=residual_scale,
                )
            if x_scale is not None:
                x, residual = x * x_scale, residual * residual_scale
            combined = x + residual
            normed = rmsnorm(
                combined,
                weight,
                self.variance_epsilon,
                residual=None,
                out=out,
                enable_pdl=None,
            )
            return normed, combined
        return rmsnorm(
            x,
            self.weight.data,
            self.variance_epsilon,
            residual=None,
            out=out,
            enable_pdl=None,
        )

    def _forward_token_fp8(
        self,
        consumer: "BeamAttention",
        x: torch.Tensor,
        residual: torch.Tensor | None,
        x_scale: float | None,
        residual_scale: float | None,
    ):
        """``forward`` for the rows the attention's fused FP8 GEMM reads: one
        ``rmsnorm_quantize_fp8`` launch normalizes them as ``forward`` does
        (with a residual, the BF16-rounded sum of the scaled pair) and also
        writes them quantized per token for the attention."""
        rejection = scale_pair_rejection(x_scale, residual_scale)
        if rejection:
            raise ValueError(f"x_scale / residual_scale: {rejection}")
        out, residual_sum, codes, scales = rmsnorm_quantize_fp8(
            x,
            self.weight.data,
            self.variance_epsilon,
            residual=residual,
            round_residual_sum_bf16=residual is not None,
            x_scale=x_scale,
            residual_scale=residual_scale,
            enable_pdl=pdl_enabled(),
        )
        consumer.offer_token_fp8(out, codes, scales)
        return out if residual is None else (out, residual_sum)

    def forward_unfused(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize this boundary's BF16 products and sum before RMSNorm.

        A fold norm owns the coefficients on the incoming raw pair. The
        ordinary forward still requires that pair; this explicit unfused
        entry normalizes the materialized sum without applying a second fold.
        """
        if self.residual_fold is not None:
            x_scale, residual_scale = self.residual_fold
            x = x * x_scale
            residual = residual * residual_scale
        combined = x + residual
        return BeamRMSNorm.forward(self, combined), combined


class BeamResidualFoldRMSNorm(BeamRMSNorm):
    """The consumer of a layer's MLP depth-scale pair.

    Layer i's MLP boundary hands its raw ``(output, residual)`` pair to the
    next input norm (or the final norm); that norm holds layer i's
    ``depth_scale_coefficients(i, False)`` as ``residual_fold`` and passes the
    pair to the residual norm, which forms the two BF16 products then the add
    exactly as ``residual_pair`` used to (in one launch on Beam's platforms).
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        dtype: torch.dtype,
        *,
        residual_fold: tuple[float, float],
    ) -> None:
        super().__init__(hidden_size, eps, dtype)
        x_scale, residual_scale = residual_fold
        rejection = scale_pair_rejection(x_scale, residual_scale)
        if x_scale is None or rejection:
            raise ValueError(
                f"residual_fold: {rejection or 'both coefficients are needed'}"
            )
        self.residual_fold = (x_scale, residual_scale)

    def extra_repr(self) -> str:
        return f"residual_fold={self.residual_fold!r}"

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        *,
        x_scale: float | None = None,
        residual_scale: float | None = None,
    ):
        fold = self.residual_fold
        if fold is not None:
            if x_scale is not None or residual_scale is not None:
                raise ValueError(
                    "residual_fold and x_scale / residual_scale are two sources "
                    "of one depth-scale pair"
                )
            if residual is None:
                raise ValueError("a norm with a residual_fold consumes a residual pair")
            x_scale, residual_scale = fold
        return super().forward(
            x, residual, out, x_scale=x_scale, residual_scale=residual_scale
        )

    def forward_with_allreduce_fusion(
        self,
        rank: int,
        group: tuple[int, ...],
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        if residual is None:
            raise ValueError("a norm with a residual_fold consumes a residual pair")
        x_scale, residual_scale = self.residual_fold
        # The fused all-reduce + add + norm kernel has no scale arguments: the
        # two products are formed here, in Torch, exactly as residual_pair did.
        x, residual = x * x_scale, residual * residual_scale
        # The shared path falls back to self.forward when it does not fuse; the
        # pair is scaled already, so the fold is off while it runs.
        self.residual_fold = None
        try:
            return super().forward_with_allreduce_fusion(
                rank, group, x, residual, *args, **kwargs
            )
        finally:
            self.residual_fold = (x_scale, residual_scale)

    def forward_with_reducescatter_fusion(self, *args: Any, **kwargs: Any):
        # The shared reduce-scatter fusion would add the raw pair without the
        # fold; no consumer of a Beam pair reaches it, so it refuses loudly.
        raise NotImplementedError(
            "a norm with a residual_fold has no reduce-scatter fusion path"
        )


class BeamRotaryEmbedding(RotaryEmbedding):
    """One model-owned FP32 cache, reused by all sliding-attention layers."""

    def __init__(self, config: Any) -> None:
        params = config.rope_parameters["sliding_attention"]
        self.scaling_factor = params.get("factor", 1.0)
        # The kernels index the table by raw position. Like the drafter's
        # (draft_rope_positions), it covers the longer of the checkpoint's
        # context and --max-model-len, and what speculative decoding writes
        # past it (spec_rope_positions).
        context = max(
            int(config.max_position_embeddings),
            int(global_server_args_dict["max_model_len"] or 0),
        )
        verify_width = (
            int(global_server_args_dict["speculative_num_draft_tokens"])
            if global_server_args_dict.get("speculative_algorithm") is not None
            else 0
        )
        super().__init__(
            head_size=config.head_dim,
            rotary_dim=config.head_dim,
            max_position_embeddings=spec_rope_positions(context, verify_width),
            base=params["rope_theta"],
            is_neox_style=True,
            dtype=torch.float32,
        )

    def _compute_inv_freq(self, base: int | float) -> torch.Tensor:
        return super()._compute_inv_freq(base) / self.scaling_factor

    def _apply(self, fn, recurse=True):
        # A model dtype conversion must never round the FP32 trigonometry.
        # Inspect only a zero-size tensor to discover the requested device.
        destination = fn(self.cos_sin_cache.new_empty(0))
        self.cos_sin_cache = self.cos_sin_cache.to(
            device=destination.device, dtype=torch.float32
        )
        return self


# The per-token FP8 quantizer whose codes the input norm's per-token FP8 output
# reproduces bit for bit (``rmsnorm_quantize_fp8``).
_TOKEN_FP8_QUANTIZER = "triton_quantize_fp8_per_token"


def _token_fp8_rows(rows: int, dtype: torch.dtype) -> bool:
    """Whether ``rows`` rows of ``dtype`` take Beam's FP8 decode paths: those
    ``quantize_fp8(granularity="token")`` quantizes with the one-launch
    per-token quantizer (NVIDIA SM100 and newer, up to its row bound), so the
    input norm's codes and the quantizer's are the same bytes."""
    return (
        rows > 0
        and select_fp8_token_quantizer(rows, dtype, solution=None, override=None).name
        == _TOKEN_FP8_QUANTIZER
    )


def _fp8_swiglu_available(intermediate_size: int) -> bool:
    """Whether ``silu_and_mul_quantize_fp8`` serves a BF16 gate/up output of
    ``intermediate_size`` columns per half on this platform."""
    return intermediate_size <= 16_384


def _dynamic_fp8_channel_linear(layer: nn.Module) -> bool:
    """A compressed-tensors FP8 W8A8 linear with channel weight scales and
    per-token activations quantized at run time."""
    if not isinstance(layer.quant_method, CompressedTensorsLinearMethod):
        return False
    scheme = layer.scheme
    return (
        isinstance(scheme, CompressedTensorsW8A8Fp8)
        and scheme.strategy == QuantizationStrategy.CHANNEL
        and not scheme.is_static_input_scheme
    )


class BeamAttention(Qwen3Attention):
    """Qwen3's GQA projections and full/sliding attention setup, with Beam's
    output gate and its own Q/K norms, RoPE and SSMax."""

    def __init__(
        self,
        config: Any,
        mapping: Any,
        layer_id: int,
        prefix: str,
        rotary: BeamRotaryEmbedding | None,
        quant_config: Any,
    ) -> None:
        # ``quant_config``: the checkpoint's, for quantized projections (Beam's
        # FP8 linears), or None for BF16 ones.
        super().__init__(
            config=config,
            mapping=mapping,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=config.rope_theta,
            rope_scaling=None,
            head_dim=config.head_dim,
            # Qwen's constructor creates a rotary module that the model's table
            # replaces below: do not allocate its full 1M-position cache.
            max_position_embeddings=0,
            quant_config=quant_config,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=False,
            prefix=prefix,
        )
        tp = mapping.attn
        self.uses_rope = bool(config.no_rope_layers[layer_id])
        self.gate_proj = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.num_attention_heads * self.head_dim,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            params_dtype=_CHECKPOINT_DTYPE,
            quant_config=quant_config,
            output_sizes=None,
            prefix=add_prefix("gate_proj", prefix),
            tp_rank=tp.tp_rank,
            tp_size=tp.tp_size,
            tp_group=tp.tp_group,
            use_presharded_weights=False,
            override_kernel_name=None,
            interleave_linear_and_gate=False,
        )
        self.gate_bias = nn.Parameter(
            torch.empty(self.num_heads, dtype=_CHECKPOINT_DTYPE), requires_grad=False
        )
        set_weight_attrs(
            self.gate_bias,
            {"output_dim": 0, "weight_loader": self.gate_proj.weight_loader},
        )
        self.q_norm = BeamRMSNorm(self.head_dim, config.rms_norm_eps, _CHECKPOINT_DTYPE)
        self.k_norm = BeamRMSNorm(self.head_dim, config.rms_norm_eps, _CHECKPOINT_DTYPE)
        # The sliding layers share the model's one rotary table; NoPE has none.
        self.rotary_emb = rotary if self.uses_rope else None
        # The attention layer was built with Qwen3's norms and rotary, which
        # the lines above replace. Q/K norm, RoPE and SSMax are the model's
        # own (they round in a different order than the prologue's fused
        # kernel), so the prologue gets no steps and only writes K/V.
        self.attn.qk_norm = None
        self.attn.rotary_emb = None
        # Q, K, V and the output gate as one FP8 GEMM: the [K, N] FP8 weight
        # and the FP32 [N, 1] channel scales of all four, which
        # process_weights_after_loading makes for FP8 W8A8 projections with
        # dynamic activations; None otherwise (BF16 projections, or FP8 ones
        # dequantized at load where no FP8 GEMM is registered).
        self.register_buffer("qkv_gate_weight", None, persistent=False)
        self.register_buffer("qkv_gate_scale", None, persistent=False)
        # A one-element FP32 1 where decode rows run the fused GEMM unscaled on
        # the vendor per-tensor FP8 GEMM (fp8_unscaled_linear) and the Q/K norm
        # kernel applies the scales; None otherwise.
        self.register_buffer("fp8_unit_scale", None, persistent=False)
        # The input norm's per-token FP8 of the rows it returned last, offered
        # for this layer's next projection: (rows, codes, scales) or None.
        self._token_fp8: tuple[torch.Tensor, ...] | None = None

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Put the Q/K/V and output gate FP8 weights in one buffer.

        The loader visits a module before its children, so this runs before
        the two projections' own processing: their weights and channel scales
        become views of one ``[N, K]`` FP8 buffer and one ``[N, 1]`` FP32
        buffer (Q/K/V rows, then the gate rows), which that processing keeps
        (the weight becomes its ``[K, N]`` transpose, the scales stay). The
        fused GEMM and the separate projections read the same bytes.
        """
        projections = (self.qkv_proj, self.gate_proj)
        if self.qkv_gate_weight is not None:
            return
        if not all(_dynamic_fp8_channel_linear(p) for p in projections):
            return
        if not fp8_channel_gemm_available():
            return
        # The loaded, not yet processed layout: contiguous [N, hidden] FP8 rows.
        if not all(
            p.weight.dtype == torch.float8_e4m3fn
            and p.weight.is_contiguous()
            and p.weight.shape[1] == self.hidden_size
            for p in projections
        ):
            return
        rows = [p.weight.shape[0] for p in projections]
        weight = torch.cat([p.weight.data.view(torch.uint8) for p in projections])
        weight = weight.view(torch.float8_e4m3fn)
        scale = torch.cat([p.weight_scale.data for p in projections])
        for projection, w, s in zip(
            projections, weight.split(rows), scale.split(rows), strict=True
        ):
            projection.weight.data = w
            projection.weight_scale.data = s
        self.qkv_gate_weight = weight.t()
        self.qkv_gate_scale = scale
        if fp8_unscaled_gemm_available() and _token_fp8_rows(1, torch.bfloat16):
            self.fp8_unit_scale = torch.ones(
                1, dtype=torch.float32, device=scale.device
            )

    def takes_token_fp8(self, x: torch.Tensor) -> bool:
        """Whether the input norm writes ``x``'s rows quantized per token for
        this attention: decode rows of a layer whose Q/K/V and gate run as one
        FP8 GEMM."""
        return self.qkv_gate_weight is not None and (
            x.ndim == 2
            and x.shape[1] == self.hidden_size
            and x.dtype == torch.bfloat16
            and _token_fp8_rows(x.shape[0], x.dtype)
        )

    def offer_token_fp8(
        self, rows: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor
    ) -> None:
        self._token_fp8 = (rows, codes, scales)

    def _input_fp8(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """The input norm's codes and scales of exactly these rows, if it
        offered them; consumed once."""
        offered, self._token_fp8 = self._token_fp8, None
        if offered is not None and offered[0] is hidden_states:
            return offered[1], offered[2]
        return None

    def _decode_unscaled(self, hidden_states: torch.Tensor) -> bool:
        return (
            self.fp8_unit_scale is not None
            and hidden_states.ndim == 2
            and _token_fp8_rows(hidden_states.shape[0], hidden_states.dtype)
        )

    def project_qkv_gate(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The QKV projection's and the output gate's outputs (column views of
        one GEMM output where the FP8 weights were fused at load)."""
        if self.qkv_gate_weight is not None:
            offered = self._input_fp8(hidden_states)
            if offered is not None:
                out = kernel_mm(
                    offered[0],
                    self.qkv_gate_weight,
                    A_scales=offered[1],
                    B_scales=self.qkv_gate_scale,
                    out_dtype=hidden_states.dtype,
                    quant="fp8",
                )
            else:
                out = fp8_dynamic_linear(
                    hidden_states, self.qkv_gate_weight, self.qkv_gate_scale
                )
            qkv, gate = out.split([self.q_size + 2 * self.kv_size, self.q_size], -1)
            return qkv, gate
        qkv, _ = self.qkv_proj(hidden_states)
        gate, _ = self.gate_proj(hidden_states)
        return qkv, gate

    def prepare_qkv_unscaled(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Decode rows: the fused Q/K/V and gate FP8 GEMM with unit scales,
        whose scales the Q/K norm kernel applies (and writes the dequantized V
        and gate) in its one launch; ``prepare_qkv``'s outputs."""
        offered = self._input_fp8(hidden_states)
        if offered is None:
            offered = quantize_fp8(
                hidden_states, granularity="token", enable_pdl=pdl_enabled()
            )
        codes, token_scales = offered
        raw = fp8_unscaled_linear(codes, self.qkv_gate_weight, self.fp8_unit_scale)
        widths = [self.q_size, self.kv_size, self.kv_size, self.q_size]
        q, k, v, gate = raw.split(widths, dim=-1)
        q_scales, k_scales, v_scales, gate_scales = self.qkv_gate_scale.split(widths)
        rows = hidden_states.shape[0]
        dequant = UnscaledFP8QKVGate(
            token_scales=token_scales,
            q_scales=q_scales,
            k_scales=k_scales,
            v=v,
            v_scales=v_scales,
            gate=gate,
            gate_scales=gate_scales,
            v_out=torch.empty((rows, self.kv_size), dtype=raw.dtype, device=raw.device),
            gate_out=torch.empty(
                (rows, self.q_size), dtype=raw.dtype, device=raw.device
            ),
        )
        q, k = self.qk_norm(q, k, positions, dequant=dequant)
        return (
            q.reshape(-1, self.num_heads, self.head_dim),
            k.reshape(-1, self.num_kv_heads, self.head_dim),
            dequant.v_out.reshape(-1, self.num_kv_heads, self.head_dim),
            dequant.gate_out.reshape(-1, self.num_heads, self.head_dim),
        )

    def qk_norm(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
        dequant: UnscaledFP8QKVGate | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The Q/K norms (with RoPE on sliding layers, SSMax on the others)."""
        extra = {} if dequant is None else {"dequant": dequant}
        if self.rotary_emb is not None:
            return fused_qk_rmsnorm_rope(
                q,
                k,
                self.q_norm.weight.data,
                self.k_norm.weight.data,
                self.rotary_emb.cos_sin_cache,
                positions,
                self.q_norm.variance_epsilon,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                **extra,
            )
        return qk_rmsnorm_ssmax(
            q,
            k,
            self.q_norm.weight.data,
            self.k_norm.weight.data,
            self.q_norm.variance_epsilon,
            positions,
            solution=None,
            override=None,
            **extra,
        )

    def prepare_qkv(
        self, hidden_states: torch.Tensor, positions: torch.Tensor, ctx: Any | None
    ):
        if self.qkv_gate_weight is not None and self._decode_unscaled(hidden_states):
            return self.prepare_qkv_unscaled(hidden_states, positions)
        qkv, gate = self.project_qkv_gate(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.qk_norm(q, k, positions)
        return (
            q.reshape(-1, self.num_heads, self.head_dim),
            k.reshape(-1, self.num_kv_heads, self.head_dim),
            v.reshape(-1, self.num_kv_heads, self.head_dim),
            gate.reshape(-1, self.num_heads, self.head_dim),
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, ctx: Any):
        q, k, v, gate = self.prepare_qkv(hidden_states, positions, ctx)
        # Q and K are final here; the attention layer has no norm or rotary
        # step, so its prologue only writes K and V to the cache.
        attended = self.attn(q, k.flatten(1), v.flatten(1), positions, ctx).reshape(
            -1, self.q_size
        )
        # The gate after every backend: gate + bias, the sigmoid and the product
        # in FP32, rounded once (the checkpoint's reference rounds each of the
        # three to BF16). On NVIDIA the shared kernel, in place; its biased branch
        # calls libdevice.div_rn, which ROCm's libdevice lacks, so other platforms
        # run the same expression in Torch.
        if current_platform().is_nvidia:
            sigmoid_mul(attended, gate, head_bias=self.gate_bias.data)
        else:
            attended = apply_attention_output_gate(
                attended.view_as(gate), gate, self.gate_bias.data, 0.0, 1.0
            ).flatten(1)
        output, _ = self.o_proj(attended)
        return output


def beam_route_weights(
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_groups: int,
    topk_groups: int,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select sorted grouped experts and return FP32 coefficients.

    Unscaled, or with ``scale`` multiplied by it in the route epilogue: the bytes
    TopK's in-place ``topk_weights *= routed_scaling_factor`` gives, one kernel
    fewer. On Beam's platforms one launch runs the selection and the epilogue,
    byte for byte, where its guard admits the call.
    """
    args = (gating_output, correction_bias, num_groups, 2, topk_groups, topk)
    if beam_kernels() and not router_route_rejection(*args, renormalize, scale):
        return triton_router_route_lanepar(*args, renormalize, scale)
    return torch_grouped_route(*args, renormalize, scale)


class BeamRouter(ReplicatedLinear):
    """The FP32 expert router: BF16 rows times the FP32 weight, FP32 logits."""

    def __init__(self, config: Any, prefix: str, correction_bias: nn.Parameter) -> None:
        super().__init__(
            input_size=config.hidden_size,
            output_size=config.num_experts,
            bias=False,
            skip_bias_add=False,
            params_dtype=_ROUTER_DTYPE,
            quant_config=None,
            prefix=prefix,
        )
        # Keep the exact parameter already referenced by TopK and MoELayer.
        self.e_score_correction_bias = correction_bias
        # BF16 pieces of the FP32 weight, made once after loading where a
        # registered mv kernel takes them (SM100 to SM103).
        self.register_buffer("weight_split", None, persistent=False)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        self.weight_split = mv_weight_split(self.weight)

    def forward(self, hidden_states: torch.Tensor):
        # BF16 rows times the FP32 weight, FP32 logits (gemm.mv): the
        # FP32 row-CTA kernel up to 16 rows, the FP32 CUDA-core leaf from 17 to
        # 96 rows, the BF16x3 tensor-core kernel on the weight split from 97
        # rows, Torch's FP32 matmul where no kernel takes the call.
        return kernel_mv(hidden_states, self.weight, weight_split=self.weight_split)


class BeamMoE(DeepseekV3MoE):
    """DeepSeek-V3's grouped sigmoid routing and expert execution without
    shared experts, on Beam's FP32 router, with the explicit BF16 route policy."""

    # The routed scale the route epilogue applies; None leaves the coefficients
    # unscaled (this policy's experts apply the routed scale in the combine).
    route_scale: float | None = None

    def __init__(self, config: Any, mapping: Any, layer_id: int, prefix: str) -> None:
        adapted = copy(config)
        adapted.moe_precision_policy = BEAM_MOE_PRECISION_POLICY
        adapted.moe_output_scale = config.routed_scaling_factor
        adapted.routed_scaling_factor = 1.0
        self._init_moe(adapted, mapping, layer_id, prefix, None)
        if (
            self.experts.plan.get("precision_policy") != BEAM_MOE_PRECISION_POLICY
            or self.experts.plan.get("output_scale") != config.routed_scaling_factor
        ):
            raise NotImplementedError(
                "Beam requires the BF16-route FP32-combine MoE policy"
            )
        self.num_groups = config.n_group
        self.topk_groups = config.topk_group
        # The common TopK callback retains normal output containers/recording;
        # its grouped fast path cannot express this denominator contract yet.
        self.topk.topk_config.use_grouped_topk = False
        self.topk.topk_config.custom_routing_function = self._route
        # The callback owns selection-only bias. A non-None common bias would
        # select the CUDA routing shortcut before the custom callback.
        self.topk.topk_config.correction_bias = None
        self.topk.topk_config.routed_scaling_factor = None

    def _init_moe(
        self, config: Any, mapping: Any, layer_id: int, prefix: str, quant_config: Any
    ) -> None:
        """The shared MoE constructor under Beam's checkpoint names, then
        the FP32 router in place of the shared gate."""
        # Adapt constructor vocabulary without changing checkpoint metadata.
        moe_config = copy(config)
        moe_config.n_routed_experts = config.num_experts
        moe_config.n_shared_experts = None
        super().__init__(
            config=moe_config,
            mapping=mapping,
            quant_config=quant_config,
            layer_index=layer_id,
            prefix=prefix,
            alt_stream=None,
        )
        self.gate = BeamRouter(
            config, add_prefix("gate", prefix), self.gate.e_score_correction_bias
        )

    def _route(self, hidden_states, gating_output, topk, renormalize):
        return beam_route_weights(
            gating_output,
            self.gate.e_score_correction_bias,
            topk,
            renormalize,
            self.num_groups,
            self.topk_groups,
            self.route_scale,
        )


class BeamVendorMoE(BeamMoE):
    """Beam's MoE on FlashInfer's TRT-LLM BF16 experts (bf16_vendor_fp32_route).

    The plan routes in the kernel when the kernel's hook builds FlashInfer's
    DeepSeekV3 routing with the FP32 correction bias: it selects Beam's grouped
    top-k from the FP32 router logits and applies routing_config's routed scale
    to its BF16 route weights. Otherwise, and always under ``--ep-size``, the
    experts run on Beam's routes and the route epilogue folds the routed scale
    into the FP32 route weights (the bytes of TopK's multiply). Either way the
    finalized output takes no later scale. Under ``--ep-size`` each rank holds
    whole experts and its output sums only its local routes; the all-reduce
    after the MoE adds the ranks.

    Given a ModelOpt NVFP4 ``quant_config``, the checkpoint's NVFP4 experts
    run on FlashInfer's TRT-LLM NVFP4 routed kernel (``nvfp4_vendor_fp32_route``),
    always on Beam's routes, on the input :meth:`forward` quantizes: the call the
    kernel makes itself, moved off the route's critical path onto ``alt_stream``
    under CUDA graph capture. Given a compressed-tensors FP8 dynamic W8A8
    ``quant_config``, the FP8 per-channel experts run on FlashInfer's TRT-LLM
    FP8 per-channel routed kernel (``fp8ch_vendor_fp32_route``), always on
    Beam's routes with their FP32 weights; the kernel quantizes its input per
    token.

    On AMD gfx950, BF16, FP8 per-channel and Quark MXFP4 experts use the default
    precision policy and registry selection, with precomputed FP32 routes.
    Explicit backend selection is passed through to the registry. Routing
    applies the routed scale once; experts use their standard default arithmetic.
    """

    def __init__(
        self,
        config: Any,
        mapping: Any,
        layer_id: int,
        prefix: str,
        quant_config: Any = None,
        alt_stream: torch.cuda.Stream | None = None,
    ) -> None:
        if not config.norm_topk_prob:
            raise NotImplementedError("Beam's vendor MoE requires normalized routes")
        fmt = None if quant_config is None else beam_checkpoint_format(quant_config)
        platform = current_platform()
        amd_default = platform.is_cdna4
        if platform.is_amd and fmt == "nvfp4":
            raise ValueError("Beam NVFP4 experts require NVIDIA vendor MoE")
        if amd_default:
            policy = "default"
        elif fmt == "mxfp4":
            raise ValueError("Beam MXFP4 experts require AMD gfx950")
        else:
            policy = {
                None: BEAM_MOE_VENDOR_POLICY,
                "nvfp4": BEAM_MOE_NVFP4_POLICY,
                "fp8_channel": BEAM_MOE_FP8_POLICY,
            }[fmt]
        adapted = copy(config)
        adapted.moe_precision_policy = policy
        if amd_default:
            # Beam owns grouped selection and folds the routed scale into its
            # FP32 coefficients once, independent of the selected expert backend.
            adapted.moe_routing_mode = "precomputed_topk"
        adapted.moe_output_scale = 1.0
        self._init_moe(adapted, mapping, layer_id, prefix, quant_config)
        experts = self.experts
        if amd_default:
            # Beam consumes finalized output, even when the selected kernel
            # also supports separate routes for other models' combiners.
            experts.plan["supports_deferred_finalize"] = False
        self.kernel_routing = experts.support_routing
        if (
            (not amd_default and experts.plan.get("precision_policy") != policy)
            or experts.plan.get("output_scale", 1.0) != adapted.moe_output_scale
            or experts.supports_deferred_finalize
            or self.kernel_routing != self.topk.topk_config.output_format.is_bypassed()
            or (
                amd_default
                and (self.kernel_routing or not experts.supports_precomputed_topk)
            )
        ):
            raise NotImplementedError(
                f"Beam's vendor MoE requires the finalized {policy} policy"
            )
        # The NVFP4 input is quantized here, off the route's critical path;
        # the FP8 per-channel kernel quantizes its own input.
        self.quantizes_input = fmt == "nvfp4"
        if self.quantizes_input:
            self.stream_fork = StreamFork(alt_stream)
        if self.kernel_routing:
            # A non-finite bias makes the kernel's top-k degenerate instead of
            # failing.
            bias = self.gate.e_score_correction_bias
            bias.weight_loader = _finite_bias_loader(
                getattr(bias, "weight_loader", default_weight_loader)
            )
        else:
            self.num_groups = config.n_group
            self.topk_groups = config.topk_group
            self.topk.topk_config.use_grouped_topk = False
            self.topk.topk_config.custom_routing_function = self._route
            self.topk.topk_config.correction_bias = None
            # _route's epilogue applies the routed scale; TopK must not again.
            self.route_scale = config.routed_scaling_factor
            self.topk.topk_config.routed_scaling_factor = None
        if layer_id == config.mlp_layer_types.index("sparse"):
            logger.info(
                f"Beam finalized MoE rank {mapping.rank}/{mapping.world_size}: "
                f"routing={'kernel' if self.kernel_routing else 'beam'} "
                f"apply={experts.plan['apply_kernel_name']} "
                f"ep_size={experts.ep_size} ep_rank={experts.ep_rank} "
                f"local_experts={experts.num_local_experts}/{experts.num_experts}"
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ) -> torch.Tensor:
        if not self.quantizes_input or hidden_states.size(0) == 0:
            return super().forward(
                hidden_states, num_global_tokens, max_num_tokens_per_gpu
            )
        # The NVFP4 input needs neither the router nor the route: the call the
        # kernel would make (the experts' input scale, linear block scales)
        # runs first, on the auxiliary stream beside the router and TopK under
        # CUDA graph capture (warm-up runs it there serially, eager in line),
        # and the experts take the (packed FP4, block scales) pair. Launched
        # without PDL: on the auxiliary stream it follows the fork event.
        with self.stream_fork.scope(
            enable=get_is_cuda_graph_phase(), overlap=get_is_capture_mode()
        ) as fork:
            with fork.branch():
                quantized = fp4_quantize(
                    hidden_states,
                    self.experts.w13_input_scale_quant,
                    is_sf_swizzled_layout=False,
                    enable_pdl=False,
                )
            router_logits = self.gate(hidden_states)
            topk_output = self.topk(hidden_states, router_logits)
        # The scope's exit orders the experts after the quantization.
        return self.experts(
            hidden_states=quantized,
            topk_output=topk_output,
            num_global_tokens=num_global_tokens,
            max_num_tokens_per_gpu=max_num_tokens_per_gpu,
        )


def _finite_bias_loader(load):
    """Refuse a non-finite checkpoint correction bias before it is stored."""

    def finite_bias_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        if not bool(torch.isfinite(loaded_weight).all()):
            raise ValueError(
                "Beam's kernel-routed MoE requires a finite e_score_correction_bias"
            )
        load(param, loaded_weight)

    return finite_bias_loader


def beam_moe_class(quantized_experts: bool = False) -> type[BeamMoE]:
    """The Beam MoE class this process's server arguments select.

    ``--moe-backend flashinfer_trtllm`` (Beam's model default on NVIDIA SM100 to
    SM103 where the policy starts) selects :class:`BeamVendorMoE`; every other
    backend keeps BeamMoE. AMD gfx950 uses the finalized default-policy path
    for BF16, FP8 and MXFP4 experts, with backend selection left to the registry.
    Unsupported quantized backends fail before loading.
    """
    if get_moe_backend().value == "flashinfer_trtllm":
        return BeamVendorMoE
    platform = current_platform()
    if platform.is_cdna4:
        return BeamVendorMoE
    if quantized_experts:
        raise ValueError(
            "Beam runs FP8 experts on AMD gfx950 with --moe-backend auto, triton "
            "or gluon (BF16 projection operands); "
            "NVFP4 and NVIDIA FP8 experts use the vendor MoE, which needs "
            "--moe-backend flashinfer_trtllm (Beam's default on NVIDIA SM100 "
            f"to SM103 where it applies), not {get_moe_backend().value}"
        )
    return BeamMoE


def beam_moe_agreement(mapping: Any, decision: tuple) -> None:
    """Fail startup unless every rank runs Beam's MoE the same way.

    The MoE backend default and whether the vendor kernel routes are decided per
    process. Ranks that disagreed could pick different experts for a near-tied
    token and all-reduce partials of different experts.
    """
    if not dist.is_initialized() or mapping.world_size == 1:
        return
    pg_manager.init_process_group(mapping.world_group, backend="gloo")
    decisions = [None] * mapping.world_size
    group = pg_manager.get_process_group("gloo", mapping.world_group)
    dist.all_gather_object(decisions, decision, group=group)
    if len(set(decisions)) != 1:
        raise RuntimeError(f"Beam's MoE differs across ranks: {decisions}")


class BeamMLP(LlamaMLP):
    """A dense MLP layer; FP8 W8A8 decode rows run gate/up unscaled.

    Where gate/up and down are compressed-tensors FP8 W8A8 linears with
    dynamic activations on the FP8 GEMMs, and a vendor per-tensor FP8 GEMM is
    registered (``fp8_unscaled_gemm_available``), the rows the one-launch
    per-token quantizer takes (up to 64) run: the
    per-token quantization of the input, gate/up on that GEMM with unit
    scales (``fp8_unscaled_linear``), one launch that applies its scales, the
    SwiGLU and the down projection's per-token quantization
    (``silu_and_mul_quantize_fp8``), and the scaled FP8 down GEMM. Everything
    else (BF16 checkpoints, other rows) runs ``LlamaMLP.forward``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer("fp8_unit_scale", None, persistent=False)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        # The loader visits this module before its projections, whose own
        # processing then leaves the [K, N] FP8 weights and FP32 scales.
        projections = (self.gate_up_proj, self.down_proj)
        if self.fp8_unit_scale is not None or not all(
            _dynamic_fp8_channel_linear(p) for p in projections
        ):
            return
        if not (
            fp8_channel_gemm_available()
            and fp8_unscaled_gemm_available()
            and _token_fp8_rows(1, torch.bfloat16)
        ):
            return
        weight = self.gate_up_proj.weight
        # The unit-scale GEMM writes BF16 for the SwiGLU consumer to read.
        if weight.dtype != torch.float8_e4m3fn or not _fp8_swiglu_available(
            weight.shape[0] // 2
        ):
            return
        self.fp8_unit_scale = torch.ones(1, dtype=torch.float32, device=weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (
            self.fp8_unit_scale is not None
            and x.ndim == 2
            and _token_fp8_rows(x.shape[0], x.dtype)
        ):
            return super().forward(x)
        codes, token_scales = quantize_fp8(
            x, granularity="token", enable_pdl=pdl_enabled()
        )
        gate_up = fp8_unscaled_linear(
            codes, self.gate_up_proj.weight, self.fp8_unit_scale
        )
        act_codes, act_scales = silu_and_mul_quantize_fp8(
            gate_up,
            token_scales,
            self.gate_up_proj.weight_scale,
            enable_pdl=pdl_enabled(),
        )
        return kernel_mm(
            act_codes,
            self.down_proj.weight,
            A_scales=act_scales,
            B_scales=self.down_proj.weight_scale,
            out_dtype=x.dtype,
            quant="fp8",
        )


class BeamCommManager(CommManager):
    @staticmethod
    def _unfused_add_norm(
        norm: BeamRMSNorm,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Beam's consuming norm owns depth scaling on the raw residual pair;
        # the generic helper would add it before those coefficients apply.
        return norm.forward_unfused(hidden_states, residual)

    def use_all_reduce_norm_fusion(self) -> bool:
        # The output norm must run after the TP reduction and before the scaled
        # add; the standard fused reduce/add/norm changes that order.
        return False


class BeamDecoderLayer(BaseDecoderLayer):
    def __init__(
        self,
        config: Any,
        layer_id: int,
        mapping: Any,
        quant_config: Any,
        prefix: str,
        rotary: BeamRotaryEmbedding | None,
        alt_stream: torch.cuda.Stream | None = None,
    ) -> None:
        object.__setattr__(self, "_beam_rotary", rotary)
        # The MoE takes the checkpoint's quant config (NVFP4 or FP8 experts).
        # Attention and the dense MLP layers take it for an FP8 checkpoint
        # (FP8 linears) and keep BF16 parameters for an NVFP4 one.
        object.__setattr__(self, "_beam_moe_quant_config", quant_config)
        object.__setattr__(self, "_beam_moe_alt_stream", alt_stream)
        dense_quant_config = (
            quant_config
            if quant_config is not None
            and beam_checkpoint_format(quant_config) == "fp8_channel"
            else None
        )
        super().__init__(
            config=config,
            layer_id=layer_id,
            mapping=mapping,
            quant_config=dense_quant_config,
            prefix=prefix,
        )
        object.__delattr__(self, "_beam_rotary")
        object.__delattr__(self, "_beam_moe_quant_config")
        object.__delattr__(self, "_beam_moe_alt_stream")
        # The sandwich norms after the attention and after the MLP.
        self.attn_output_layernorm = self.resolve_norm()
        self.mlp_output_layernorm = self.resolve_norm()
        # The shared constructor resolves every norm alike; the input norm is
        # re-resolved with the producer it consumes.
        self.input_layernorm = self.resolve_input_norm()
        self.comm_manager = BeamCommManager(
            mapping=mapping,
            layer_id=layer_id,
            is_moe=self.is_moe_layer,
            prev_is_moe=(layer_id - 1 not in config.mlp_only_layers),
            dense_batch_invariant=False,
            input_layernorm=self.input_layernorm,
            post_attn_layernorm=self.post_attention_layernorm,
            query_sharded=False,
        )
        # The input norm writes its decode rows' per-token FP8 for the
        # attention's fused FP8 GEMM where that runs (a plain attribute, not a
        # submodule).
        if isinstance(self.self_attn, BeamAttention):
            object.__setattr__(
                self.input_layernorm, "_token_fp8_consumer", self.self_attn
            )

    @property
    def is_moe_layer(self) -> bool:
        return self.layer_id not in self.config.mlp_only_layers

    def resolve_norm(self):
        return BeamRMSNorm(
            self.config.hidden_size, self.config.rms_norm_eps, torch.bfloat16
        )

    def resolve_input_norm(self):
        # Layer 0 receives the embeddings alone; layer i >= 1 receives layer
        # i-1's raw MLP pair and its input norm folds that layer's pair.
        if self.layer_id == 0:
            return self.resolve_norm()
        return BeamResidualFoldRMSNorm(
            self.config.hidden_size,
            self.config.rms_norm_eps,
            torch.bfloat16,
            residual_fold=depth_scale_coefficients(self.layer_id - 1, False),
        )

    def resolve_attn(self, prefix: str):
        return BeamAttention(
            self.config,
            self.mapping,
            self.layer_id,
            add_prefix("self_attn", prefix),
            self._beam_rotary,
            self.quant_config,
        )

    def resolve_mlp(self, prefix: str):
        if not self.is_moe_layer:
            return BeamMLP(
                hidden_size=self.config.hidden_size,
                intermediate_size=self.config.intermediate_size,
                hidden_act=self.config.hidden_act,
                mapping=self.mapping,
                quant_config=self.quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        quant_config = self._beam_moe_quant_config
        args = (self.config, self.mapping, self.layer_id, add_prefix("mlp", prefix))
        if quant_config is None:
            return beam_moe_class()(*args)
        return beam_moe_class(quantized_experts=True)(
            *args, quant_config, alt_stream=self._beam_moe_alt_stream
        )

    def residual_add(self, output, residual, output_scale, residual_scale):
        return output * output_scale + residual * residual_scale

    def residual_pair(self, output, residual, output_scale, residual_scale):
        # The MLP boundary hands the raw pair on: the consuming norm (the next
        # layer's input norm or the final norm) holds this layer's pair as its
        # residual_fold and applies both products there, before the add.
        return output, residual

    def attn_residual_norm(
        self, hidden_states, residual, sublayer_scale, residual_scale
    ):
        # The raw sublayer output, the residual stream and the depth-scale pair go
        # to the residual norm: residual_pair's two BF16 products then the BF16
        # add, exactly as before (one launch on Beam's platforms).
        return self.post_attention_layernorm(
            self.attn_output_layernorm(hidden_states),
            residual=residual,
            x_scale=sublayer_scale,
            residual_scale=residual_scale,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: Any,
        residual: torch.Tensor | None,
        aux_hidden_states: list | None = None,
    ):
        comm = self.comm_manager
        if not ctx.forward_mode.is_idle():
            hidden_states, residual = comm.input_reduce_norm(hidden_states, residual)
            if aux_hidden_states is not None:
                aux_hidden_states.append(comm.gather_residual(residual, ctx).clone())
            hidden_states = comm.pre_attn_comm(hidden_states, ctx)
            hidden_states = self.self_attn(
                positions=positions, hidden_states=hidden_states, ctx=ctx
            )
            hidden_states, residual = comm.post_attn_comm(hidden_states, residual, ctx)
            sublayer_scale, residual_scale = depth_scale_coefficients(
                self.layer_id, True
            )
            hidden_states, residual = self.attn_residual_norm(
                hidden_states, residual, sublayer_scale, residual_scale
            )
        elif residual is None:
            residual = hidden_states
        hidden_states = comm.pre_mlp_comm(hidden_states, ctx)
        if self.is_moe_layer:
            total, maximum = comm.get_num_tokens(ctx)
            hidden_states = self.mlp(hidden_states, total, maximum)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states, residual = comm.post_mlp_comm(hidden_states, residual, ctx)
        hidden_states = self.mlp_output_layernorm(hidden_states)
        output_scale, residual_scale = depth_scale_coefficients(self.layer_id, False)
        # The base model's two-tensor residual contract: the consuming norm adds
        # the pair (see residual_pair).
        return self.residual_pair(hidden_states, residual, output_scale, residual_scale)


class BeamModel(BaseTransformerModel):
    layer_cls = BeamDecoderLayer

    def __init__(
        self, config: Any, mapping: Any, quant_config: Any, prefix: str
    ) -> None:
        super().__init__(
            config=config, mapping=mapping, quant_config=quant_config, prefix=prefix
        )
        # The final norm consumes the last layer's raw MLP pair; the shared
        # constructor built a plain norm and its final-norm op, both re-resolved.
        self.norm = self.resolve_final_norm(config)
        self._final_norm_op = self._build_final_norm_op()
        # Under --enable-allreduce-fusion every boundary runs through the
        # sandwich norm helper (fused where the MNNVL workspace serves it).
        from tokenspeed.runtime.models.beam_sandwich_fusion import (
            install_beam_sandwich_fusion,
        )

        install_beam_sandwich_fusion(self)

    def resolve_embed(self, config: Any, prefix: str):
        # Replicated on every rank (BeamDraftEmbedding hands it to a drafter).
        return VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            params_dtype=_CHECKPOINT_DTYPE,
            org_num_embeddings=config.vocab_size,
            padding_size=1,
            quant_config=None,
            prefix=add_prefix("embed_tokens", prefix),
            tp_rank=None,
            tp_size=None,
            tp_group=None,
            use_presharded_weights=False,
        )

    def resolve_final_norm(self, config: Any):
        return BeamResidualFoldRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            torch.bfloat16,
            residual_fold=depth_scale_coefficients(config.num_hidden_layers - 1, False),
        )

    def resolve_layers(
        self, config: Any, quant_config: Any, prefix: str
    ) -> nn.ModuleList:
        rotary = (
            BeamRotaryEmbedding(config)
            if "sliding_attention" in config.layer_types
            else None
        )
        # One auxiliary stream for every NVFP4 MoE layer's input quantization.
        alt_stream = (
            torch.cuda.Stream()
            if quant_config is not None
            and beam_checkpoint_format(quant_config) == "nvfp4"
            and torch.cuda.is_available()
            else None
        )
        return make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: self.layer_cls(
                config=config,
                layer_id=idx,
                mapping=self.mapping,
                quant_config=quant_config,
                prefix=prefix,
                rotary=rotary,
                alt_stream=alt_stream,
            ),
            prefix=add_prefix("layers", prefix),
        )


class BeamLogitsProcessor(LogitsProcessor):
    def __init__(self, config: Any, mapping: Any) -> None:
        projection_config = copy(config)
        projection_config.final_logit_softcapping = None
        tp = mapping.attn
        super().__init__(
            projection_config,
            skip_all_gather=tp.has_dp,
            do_argmax=False,
            logit_scale=None,
            tp_rank=tp.tp_rank,
            tp_size=tp.tp_size,
            tp_group=tp.tp_group,
            dp_lm_head_tp=False,
        )
        self.beam_softcap = config.final_logit_softcapping

    def fuse_argmax(self) -> None:
        # The shared processor fuses no argmax into a soft-capped head, and
        # this one caps after it runs: keep the gathered vocabulary.
        if self.beam_softcap is None:
            super().fuse_argmax()

    def _get_logits(self, *args, **kwargs):
        logits = super()._get_logits(*args, **kwargs)
        cap = self.beam_softcap
        if cap is None:
            return logits
        if beam_kernels() and not tanh_softcap_rejection(logits, cap):
            # The statement below in one pass, byte for byte.
            return tanh_softcap(logits, cap)
        return (logits.float() / cap).tanh() * cap


class BeamDraftEmbedding:
    """Beam's replicated embedding, looked up as a block drafter expects.

    A DFLASH drafter looks up its block with ``reduce_results=False`` and its
    first layer all-reduces the rows over the dense TP group: the contract of
    a vocabulary-sharded table, whose ranks each hold their shard's rows and
    zeros elsewhere. Beam's table is replicated, so every rank would hand on
    whole rows and the all-reduce would multiply them by the TP size. Here
    rank 0 hands on the rows and the other ranks zeros, so the all-reduce
    returns the rows exactly. With ``reduce_results=True`` every rank returns
    the rows, as the table itself does.
    """

    def __init__(self, embedding: nn.Module, tp_rank: int, tp_size: int) -> None:
        self.embedding = embedding
        self.tp_rank = tp_rank
        self.tp_size = tp_size

    def __call__(
        self, input_ids: torch.Tensor, reduce_results: bool = True
    ) -> torch.Tensor:
        rows = self.embedding(input_ids)
        if not reduce_results and self.tp_size > 1 and self.tp_rank != 0:
            rows.zero_()
        return rows


class BeamForCausalLM(BaseCausalLM):
    model_cls = BeamModel

    def __init__(self, config: Any, mapping: Any, quant_config: Any) -> None:
        config.validate()
        # beam_quant_policy admits or refuses a quantized checkpoint.
        quant = None if quant_config is None else beam_quant_policy(quant_config)
        if (
            mapping.moe.ep_size != 1
            and beam_moe_class(quant is not None) is not BeamVendorMoE
        ):
            raise ValueError(
                "Beam's expert parallelism runs on its vendor MoE: --ep-size needs "
                "--moe-backend flashinfer_trtllm (Beam's default on NVIDIA SM100 "
                "to SM103 where it applies)"
            )
        config.validate_parallelism(
            tensor_parallel_size=mapping.attn.tp_size, ep_size=mapping.moe.ep_size
        )
        if global_server_args_dict["ep_num_redundant_experts"]:
            raise ValueError("Beam does not support redundant experts")
        # MoE TP x EP spans the attention TP group: one all-reduce per layer.
        moe = mapping.moe
        if len({mapping.attn.tp_size, mapping.dense.tp_size, moe.tp_ep_size}) != 1:
            raise ValueError(
                "Beam requires the same TP size for attention, dense and MoE"
            )
        if len({mapping.attn.tp_rank, mapping.dense.tp_rank, moe.tp_ep_rank}) != 1:
            raise ValueError(
                "Beam requires the same TP rank for attention, dense and MoE"
            )
        if mapping.pp_size != 1:
            raise ValueError("Beam pipeline parallelism is not implemented")
        # The layers take the admitted checkpoint's quant config: NVFP4 experts,
        # or FP8 linears and FP8 experts.
        with set_default_torch_dtype(_CHECKPOINT_DTYPE):
            super().__init__(
                config=config,
                mapping=mapping,
                quant_config=quant_config,
                prefix="",
                encoder_only=False,
            )
        self.beam_quant = quant
        # The positions the sliding layers' shared RoPE table covers, which the
        # executor checks the physical context extent against (RopeTableModel).
        self.rope_positions: int | None = next(
            (
                int(module.cos_sin_cache.shape[0])
                for module in self.modules()
                if isinstance(module, BeamRotaryEmbedding)
            ),
            None,
        )
        # The plans the layers above made, not the server arguments.
        kernel_routing = any(
            isinstance(module, BeamVendorMoE) and module.kernel_routing
            for module in self.modules()
        )
        beam_moe_agreement(
            mapping, (beam_moe_class(quant is not None).__name__, kernel_routing)
        )

    def resolve_lm_head(self, config: Any, quant_config: Any, prefix: str):
        # Vocabulary-sharded BF16 head: each rank stores and multiplies its
        # slice; the shared logits processor gathers the full vocabulary.
        tp = self.mapping.attn
        return ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            bias=False,
            params_dtype=_CHECKPOINT_DTYPE,
            org_num_embeddings=config.vocab_size,
            padding_size=1,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            tp_rank=tp.tp_rank,
            tp_size=tp.tp_size,
            tp_group=tp.tp_group,
            use_presharded_weights=False,
        )

    def resolve_logits_processor(self, config: Any):
        return BeamLogitsProcessor(config, self.mapping)

    def prepare_model_kwargs(self, ctx: Any, input_ids: torch.Tensor, kwargs: dict):
        return {"input_embeds": kwargs.get("input_embeds", kwargs.get("inputs_embeds"))}

    def get_stacked_params_mapping(self) -> list[tuple[str, str, int | str]]:
        # Qualify the block so the attention gate is not treated as an MLP gate.
        return [
            (".self_attn.qkv_proj.", ".self_attn.q_proj.", "q"),
            (".self_attn.qkv_proj.", ".self_attn.k_proj.", "k"),
            (".self_attn.qkv_proj.", ".self_attn.v_proj.", "v"),
            (".mlp.gate_up_proj.", ".mlp.gate_proj.", 0),
            (".mlp.gate_up_proj.", ".mlp.up_proj.", 1),
        ]

    def get_input_embeddings(self) -> BeamDraftEmbedding:
        """The embedding a DFLASH drafter borrows (``DFlash.wire_target``)."""
        dense = self.mapping.dense
        return BeamDraftEmbedding(self.model.embed_tokens, dense.tp_rank, dense.tp_size)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]], **kwargs: Any):
        external_names = {}
        if self.beam_quant is not None:
            # FP8 weights with their channel scales and NVFP4 expert tensors
            # pass on unchanged, checked.
            weights = self.beam_quant.adapt(weights)

        def adapted_weights():
            for name, weight in weights:
                mapped = beam_checkpoint_name(name)
                if name == "lm_head.weight" and tuple(weight.shape) != (
                    self.config.vocab_size,
                    self.config.hidden_size,
                ):
                    raise ValueError(
                        "Beam lm_head.weight must have its unpadded logical shape"
                    )
                external_names[mapped] = name
                yield mapped, weight

        loaded = _load_beam_weights(self, adapted_weights())
        if self.beam_quant is not None:
            logger.info(self.beam_quant.load_witness())
        return {external_names[name] for name in loaded}


def _load_beam_weights(
    model: BeamForCausalLM, weights: Iterable[tuple[str, torch.Tensor]]
) -> set[str]:
    """Check names/dtypes and delegate shape handling and placement to loaders."""
    if any(
        isinstance(layer, MoELayer) and layer._weights_processed
        for layer in model.modules()
    ):
        raise RuntimeError("Beam weights cannot be reloaded after MoE backend packing")
    parameters = dict(model.named_parameters())
    expert_schema = ExpertCheckpointSchema(
        gate_proj_name="gate_proj",
        down_proj_name="down_proj",
        up_proj_name="up_proj",
    )
    moe_loader = build_moe_checkpoint_loader(
        params_dict=parameters,
        expert_schema=expert_schema,
        fused_schema=None,
        num_experts=model.config.num_experts,
        ep_rank=model.mapping.moe.ep_rank,
        ep_size=model.mapping.moe.ep_size,
        fused_gate_up_as_w13=False,
        include_bias=False,
        fused_load_style="per_expert",
        transpose_local_tensor_non_bias=False,
    )
    # Expand the actual parameters into checkpoint names. Each Q/K/V and expert
    # tensor must arrive, even when several fill the same runtime parameter.
    expected = set()
    # Checkpoint names of quantized tensors (expert weights and their scales,
    # FP8 linears' weights and channel scales) and their storage dtype, the
    # parameter's; any other tensor is BF16.
    quantized = {}
    stacked_mapping = model.get_stacked_params_mapping()
    for name, parameter in parameters.items():
        prefix, experts, packed_name = name.rpartition(".experts.")
        if experts and packed_name.startswith(("w13_", "w2_")):
            packed, suffix = packed_name.split("_", 1)
            projections = (
                ("gate_proj", "up_proj") if packed == "w13" else ("down_proj",)
            )
            names = {
                f"{prefix}.{expert_schema.make_expert_weight_name(expert, projection)}{suffix}"
                for expert in range(model.config.num_experts)
                for projection in projections
            }
            expected.update(names)
            if parameter.dtype != _CHECKPOINT_DTYPE:
                quantized.update(dict.fromkeys(names, parameter.dtype))
        else:
            names = [
                name.replace(parameter_name, checkpoint_name)
                for parameter_name, checkpoint_name, _ in stacked_mapping
                if parameter_name in name
            ] or [name]
            expected.update(names)
            if parameter.dtype == torch.float8_e4m3fn or name.endswith(".weight_scale"):
                # An FP8 linear's weight and channel scale.
                quantized.update(dict.fromkeys(names, parameter.dtype))
    loaded = set()

    def ordinary_weights():
        for name, weight in weights:
            if name not in expected:
                raise KeyError(f"Unexpected Beam checkpoint tensor: {name}")
            if name in loaded:
                raise ValueError(f"Duplicate Beam checkpoint tensor: {name}")
            if not isinstance(weight, torch.Tensor):
                raise TypeError(f"Checkpoint value for {name} is not a tensor")
            storage_dtype = (
                _ROUTER_DTYPE
                if name.endswith(
                    (".mlp.gate.weight", ".mlp.gate.e_score_correction_bias")
                )
                else quantized.get(name, _CHECKPOINT_DTYPE)
            )
            # FP32 channel scales load exactly from the scale dtypes a
            # compressed-tensors tool writes (the model's dtype or FP32).
            admitted = (
                (torch.float32, torch.bfloat16, torch.float16)
                if storage_dtype == torch.float32 and name.endswith(".weight_scale")
                else (storage_dtype,)
            )
            if weight.dtype not in admitted:
                raise TypeError(
                    f"Checkpoint dtype mismatch for {name}: expected {storage_dtype}, got {weight.dtype}"
                )
            if name == "lm_head.weight" and weight.ndim == 2:
                weight = weight[:, : model.lm_head.embedding_dim]
            if moe_loader.is_expert_checkpoint_weight(name):
                # Under expert parallelism another rank owns non-local experts;
                # they still count toward the global completeness check.
                if moe_loader.matches(name):
                    moe_loader.load(name, weight)
            else:
                yield name, weight
            loaded.add(name)

    BaseCausalLM.load_weights(model, ordinary_weights())
    missing = sorted(expected - loaded)
    if missing:
        raise ValueError(f"Missing Beam checkpoint tensors: {missing}")
    return loaded


def beam_checkpoint_name(name: str) -> str:
    """Map canonical Beam norm tensors to the reused decoder's module names."""
    if ".attn_output_layernorm." in name or ".mlp_output_layernorm." in name:
        raise KeyError(f"obsolete checkpoint norm name: {name}")
    for external, internal in (
        (".pre_feedforward_layernorm.", ".post_attention_layernorm."),
        (".post_feedforward_layernorm.", ".mlp_output_layernorm."),
        (".post_attention_layernorm.", ".attn_output_layernorm."),
    ):
        if external in name:
            return name.replace(external, internal)
    return name


EntryClass = BeamForCausalLM
