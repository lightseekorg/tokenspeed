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

"""V4.1 multimodal model with FlatKV attention and single-pass hyperconnections.

``load_weights`` consumes the generic CPU checkpoint iterator, rejects unknown,
duplicate and missing local text tensors, and reports explicit vision/draft skips.
Dense FP8 codes stay unchanged: 32x32 E8M0 scale rows expand losslessly to 1x32.
On Hopper the dense projections load as BF16 weights instead -- codes widened on
copy, scales folded in after loading, both exact -- and run the BF16 GEMM.
Only grouped wo_a is dequantized to BF16, after selecting this TP rank's rows.
Engram tables load local FP8/E8M0 rows in bounded chunks without table conversion.
The unquantized LM head follows the model loading dtype, including its logits.
Packed routed experts use the V4 MoE loader, with zero-padded intermediate lanes
for MegaMoE's TMA alignment (2304 -> 2560); shared experts are not padded.
The generic model loader still owns
dense/MoELayer postprocessing, while post_load_weights finalizes MegaMoE.

Call initialize_engram(tokenizer) once after construction, then pass caller-owned
``engram_previous_tokens`` [T,3] and bool ``engram_token_mask`` [T] as forward
kwargs. These are borrowed forward inputs, never request state or ForwardContext
fields. The runner owns refresh/overlap/rollback and can use stable input views.
No setter retains a mutable per-forward tensor; no hashes survive a forward.

Every scheduled token runs through the encoder layers; the CED decoder (from
the candidate source on) runs on the rows the backend's ``decoder_view()``
keeps -- each prompt's last window (one row for a chunk that leaves its prompt
open) plus every decode row -- after the candidate source has written the
decoder's global KV for all rows. DSpark captures the kept rows and the model
reports them as ``ctx.captured_rows``. PP, CP and narrowing under attention DP
are not implemented. Attention and MoE TPxEP widths must match to keep the HC
stream replicated on attention TP.

The forward is four stages on one path -- ``encoder_forward`` (embedding and
the layers below the candidate source, one row per token), ``narrowing_forward``
(the candidate source layer: all rows in, the decoder view's rows out),
``decoder_forward`` (the remaining layers and the final norm on the narrowed
rows) and ``finish_forward`` (the sampled-row gather and the DSpark row
report). ``forward`` composes them; the prefill graph replays the first and
third as captured graphs of fixed row counts around the eager narrowing
(``PrefillGraph``'s ``NarrowingPrefillModel`` contract). Each layer reads its
row plan from the live context, never as a loose argument a captured break
would freeze.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import Counter
from collections.abc import Iterable
from copy import copy
from dataclasses import dataclass
from weakref import WeakValueDictionary

import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.attention.dsv41 import (
    rope_inplace,
    rope_pad_query,
)
from tokenspeed_kernel.ops.gemm import dsv4_linear_fp32, grouped_bf16_projection
from tokenspeed_kernel.ops.quantization import quantize_fp8_with_scale
from tokenspeed_kernel.platform import current_platform
from torch import nn

from tokenspeed.runtime.configs.deepseek_v41_config import DeepseekV41Config
from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.comm_ops import all_reduce
from tokenspeed.runtime.distributed.pp_stage import PPStageState
from tokenspeed.runtime.execution.breakable_cuda_graph import (
    break_point,
    current_forward_ctx,
    slice_to_real_tokens,
)
from tokenspeed.runtime.execution.context import (
    CapturedRows,
    ForwardContext,
    report_collective_sizing,
)
from tokenspeed.runtime.layers.attention.backends.specific.deepseek_v41 import (
    V41RowPlan,
)
from tokenspeed.runtime.layers.dense.fp8 import Fp8LinearMethod
from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.moe.loader import build_moe_checkpoint_loader
from tokenspeed.runtime.layers.moe.schema import ExpertCheckpointSchema
from tokenspeed.runtime.layers.moe.utils import get_moe_backend
from tokenspeed.runtime.layers.parameter import ModelWeightParameter
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config, Mxfp8Config
from tokenspeed.runtime.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from tokenspeed.runtime.model_loader.utils import set_default_torch_dtype
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.base import BaseCausalLM
from tokenspeed.runtime.models.deepseek_v4 import (
    DeepseekV4ForCausalLM,
    DeepseekV4MegaMoEExperts,
    DeepseekV4MLP,
    DeepseekV4MoE,
)
from tokenspeed.runtime.models.deepseek_v41_engram import (
    DeepseekV41Engram,
    EngramHashState,
    is_engram_embed_checkpoint_name,
    resolve_engram_host_layout,
)
from tokenspeed.runtime.models.deepseek_v41_vision import DeepseekV41Vision
from tokenspeed.runtime.multimodal.embedder import (
    EncoderSpec,
    VisionEmbedder,
    pad_input_tokens,
)
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalInputs,
    is_mm_pad_value_for,
    substitute_mm_pad_,
)
from tokenspeed.runtime.utils import add_prefix
from tokenspeed.runtime.utils.cuda_stream import StreamFork
from tokenspeed.runtime.utils.env import global_server_args_dict

logger = logging.getLogger(__name__)
_ROPE_TABLES = WeakValueDictionary()


def v41_quantize_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP8 codes and uint8 E8M0 scales using reference 1x32 quantization.

    The reference rounds the FP32 product ``max(amax, 1e-4) * (1/448)`` by
    inspecting its exponent/mantissa. Generic MXFP8 quantizers differ at zero,
    tiny activations, and power-of-two boundaries, so do not substitute one.
    """
    if x.shape[-1] % 32:
        raise ValueError(
            "V4.1 FP8 activations require a last dimension divisible by 32"
        )
    if x.is_cuda and x.dtype in (torch.bfloat16, torch.float16):
        return quantize_fp8_with_scale(
            x.reshape(-1, x.shape[-1]),
            granularity="token_group",
            group_size=32,
            scale_encoding="ue8m0",
            enable_pdl=False,
            override="triton_quantize_fp8_group32_ue8m0",
            solution=None,
        )
    grouped = x.float().unflatten(-1, (-1, 32))
    unrounded = grouped.abs().amax(-1).clamp_min(1e-4) * (1.0 / 448.0)
    bits = unrounded.contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    scales = (exponent << 23).view(torch.float32)
    codes = (
        (grouped / scales.unsqueeze(-1))
        .clamp(-448, 448)
        .flatten(-2)
        .to(torch.float8_e4m3fn)
    )
    return codes, exponent.to(torch.uint8)


def v41_mxfp8_config(quant_config: QuantizationConfig | None) -> Mxfp8Config | None:
    """Return a lossless per-row scale configuration from checkpoint 32x32 FP8."""
    if quant_config is None:
        return None
    if (
        not isinstance(quant_config, Fp8Config)
        or not quant_config.is_checkpoint_fp8_serialized
        or quant_config.weight_block_size != [32, 32]
        or quant_config.scale_fmt != "ue8m0"
    ):
        raise ValueError(
            "V4.1 dense weights require checkpoint FP8 with 32x32 E8M0 scales"
        )
    return Mxfp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        ignored_layers=quant_config.ignored_layers,
        weight_block_size=[1, 32],
        scale_fmt="ue8m0",
    )


class _ReferenceFp8LinearMethod(Fp8LinearMethod):
    def __init__(self, quant_config: Mxfp8Config):
        super().__init__(quant_config)
        # Hopper has no tensor-core kernel for 32-wide FP8 blocks; its FP8 GEMMs
        # rescale on CUDA cores every 32 K and trail the plain BF16 GEMM at
        # every shape. So load the codes straight into a BF16 weight and fold
        # the power-of-two scales in once -- both exact in BF16 -- and run the
        # BF16 GEMM. Costs the FP8 weight size again.
        self.load_as_bf16 = current_platform().is_hopper

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if not self.load_as_bf16:
            super().process_weights_after_loading(layer)
            return
        scale = layer.weight_scale_inv.data.view(torch.float8_e8m0fnu).to(
            torch.bfloat16
        )
        layer.weight.data.unflatten(-1, (-1, 32)).mul_(scale.unsqueeze(-1))
        layer.register_parameter("weight_scale_inv", None)
        layer.quant_method = UnquantizedLinearMethod()

    def apply(
        self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        if x.shape[0] == 0:
            return x.new_empty((*x.shape[:-1], layer.weight.shape[0]))
        plan = getattr(layer, "_prepared_fp8_linear", None)
        if (
            x.is_cuda
            and x.dtype in (torch.bfloat16, torch.float16)
            and plan is not None
        ):
            from tokenspeed_kernel.ops.gemm import quantize_fp8_group32_for_linear

            codes, scales = quantize_fp8_group32_for_linear(
                plan, x.reshape(-1, x.shape[-1])
            )
        else:
            codes, scales = v41_quantize_fp8(x)
        return super().apply(layer, codes, bias, scales, x.dtype)

    def apply_with_activation(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        activation: nn.Module,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        # Fused activation quantizers do not implement the reference amax floor.
        return self.apply(layer, activation(x), bias)


def configure_v41_fp8_linear(layer: LinearBase, expand_checkpoint_scales: bool) -> None:
    """Bind exact activation quantization and an optional checkpoint scale loader.

    Scale expansion precedes the existing TP/merged loader, which still owns
    sharding. Pass False for Engram, whose loader already expands scale rows.
    This helper never changes FP8 weight values or routed-expert parameters.
    """
    if layer.weight.dtype != torch.float8_e4m3fn:
        return
    if not isinstance(
        layer.quant_config, Mxfp8Config
    ) or layer.quant_config.weight_block_size != [1, 32]:
        raise ValueError("Configure V4.1 linears with the lossless 1x32 MXFP8 config")
    layer.quant_method = _ReferenceFp8LinearMethod(layer.quant_config)
    if layer.quant_method.load_as_bf16:
        # Same shape and sharding loader as the FP8 parameter; the loader's
        # copy widens each E4M3 code to BF16 exactly.
        codes = layer.weight
        layer.weight = ModelWeightParameter(
            data=torch.empty_like(codes.data, dtype=torch.bfloat16),
            input_dim=codes.input_dim,
            output_dim=codes.output_dim,
            weight_loader=codes.weight_loader,
        )
        layer.weight.loads_fp8_codes = True
    if not expand_checkpoint_scales:
        return
    scale = layer.weight_scale_inv
    original_loader = scale.weight_loader

    def load_scale(
        param: nn.Parameter, loaded_weight: torch.Tensor, *shard_ids
    ) -> None:
        if (
            loaded_weight.dtype not in (torch.uint8, torch.float8_e8m0fnu)
            or loaded_weight.ndim != 2
        ):
            raise TypeError(
                "V4.1 projection scales must be a 2D E8M0 checkpoint tensor"
            )
        expanded = loaded_weight.view(torch.uint8).repeat_interleave(32, dim=0)
        original_loader(param, expanded, *shard_ids)

    scale._weight_loader = load_scale
    scale.v41_checkpoint_block_size = (32, 32)


def _replicated(input_size, output_size, dtype, quant_config, prefix):
    layer = ReplicatedLinear(
        input_size=input_size,
        output_size=output_size,
        bias=False,
        skip_bias_add=False,
        params_dtype=dtype,
        quant_config=quant_config,
        prefix=prefix,
    )
    configure_v41_fp8_linear(layer, True)
    return layer


def _column(input_size, output_size, dtype, quant_config, prefix, mapping):
    layer = ColumnParallelLinear(
        input_size=input_size,
        output_size=output_size,
        bias=False,
        gather_output=False,
        skip_bias_add=False,
        params_dtype=dtype,
        quant_config=quant_config,
        output_sizes=None,
        prefix=prefix,
        tp_rank=mapping.attn.tp_rank,
        tp_size=mapping.attn.tp_size,
        tp_group=mapping.attn.tp_group,
        use_presharded_weights=False,
        override_kernel_name=None,
        interleave_linear_and_gate=False,
    )
    configure_v41_fp8_linear(layer, True)
    return layer


def _norm(x: torch.Tensor, norm: RMSNorm) -> torch.Tensor:
    # ponytail: eager reference arithmetic; fuse only with cast-order parity tests.
    values = x.float()
    values = values * torch.rsqrt(
        values.square().mean(-1, keepdim=True) + norm.variance_epsilon
    )
    return (norm.weight.float() * values).to(x.dtype)


def _attention_norm(x: torch.Tensor, norm: RMSNorm) -> torch.Tensor:
    return norm(x, residual=None, out=None) if x.is_cuda else _norm(x, norm)


def _merged(input_size, output_sizes, dtype, quant_config, prefix):
    layer = MergedColumnParallelLinear(
        input_size,
        output_sizes,
        bias=False,
        gather_output=False,
        skip_bias_add=False,
        params_dtype=dtype,
        quant_config=quant_config,
        prefix=prefix,
        tp_rank=0,
        tp_size=1,
        tp_group=None,
        use_presharded_weights=False,
        override_kernel_name=None,
        interleave_linear_and_gate=False,
    )
    configure_v41_fp8_linear(layer, True)
    return layer


class DeepseekV41RotaryEmbedding(nn.Module):
    """Adjacent-pair RoPE, with YaRN for BOTH ratio-1 and ratio-2 global layers."""

    def __init__(self, config, compress_ratio: int):
        super().__init__()
        self.dim = config.qk_rope_head_dim
        if self.dim < 2 or self.dim % 2:
            raise ValueError("V4.1 RoPE dimension must be positive and even")
        base = config.compress_rope_theta if compress_ratio else config.rope_theta
        freqs = 1.0 / (
            base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        if compress_ratio:
            scaling = config.rope_scaling
            original = scaling["original_max_position_embeddings"]
            factor = scaling["factor"]
            fast, slow = scaling["beta_fast"], scaling["beta_slow"]
            low = max(
                math.floor(
                    self.dim
                    * math.log(original / (fast * 2 * math.pi))
                    / (2 * math.log(base))
                ),
                0,
            )
            high = min(
                math.ceil(
                    self.dim
                    * math.log(original / (slow * 2 * math.pi))
                    / (2 * math.log(base))
                ),
                self.dim - 1,
            )
            ramp = (
                (torch.arange(self.dim // 2, dtype=torch.float32) - low)
                / max(high - low, 1e-3)
            ).clamp(0, 1)
            smooth = 1 - ramp
            freqs = freqs / factor * (1 - smooth) + freqs * smooth
        self.register_buffer("inv_freq", freqs, persistent=False)
        self.max_positions = config.max_position_embeddings
        self._table_key = (
            self.dim,
            base,
            self.max_positions,
            tuple(sorted(config.rope_scaling.items())) if compress_ratio else None,
        )
        self.register_buffer("cos_sin_cache", None, persistent=False)
        self.register_buffer("inverse_cos_sin_cache", None, persistent=False)

    def _apply(self, fn, *args, **kwargs):
        frequencies = self.inv_freq
        self.inv_freq = frequencies.new_empty(0)
        self.cos_sin_cache = self.inverse_cos_sin_cache = None
        self._shared_table = None
        super()._apply(fn, *args, **kwargs)
        self.inv_freq = frequencies.to(device=self.inv_freq.device, dtype=torch.float32)
        return self

    def _prepare_cache(self, device: torch.device) -> None:
        if self.cos_sin_cache is None or self.cos_sin_cache.device != device:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm up V4.1 RoPE before graph capture")
            key = (self._table_key, device)
            table = _ROPE_TABLES.get(key)
            if table is None:
                angles = torch.arange(
                    self.max_positions, device=device, dtype=torch.float32
                )[:, None] * self.inv_freq.to(device)
                cosine, sine = angles.cos(), angles.sin()
                table = torch.stack(
                    (
                        torch.cat((cosine, sine), dim=-1),
                        torch.cat((cosine, -sine), dim=-1),
                    )
                )
                _ROPE_TABLES[key] = table
            self._shared_table = table
            self.cos_sin_cache, self.inverse_cos_sin_cache = table.unbind(0)

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor, inverse: bool
    ) -> torch.Tensor:
        """Rotate only x's trailing RoPE dimensions at absolute token positions."""
        if not x.is_cuda:
            angles = positions.clamp_min(0).float().unsqueeze(-1) * self.inv_freq
            freqs = torch.polar(torch.ones_like(angles), angles)
            if inverse:
                freqs = freqs.conj()
            freqs = freqs.reshape(
                positions.numel(), *([1] * (x.ndim - 2)), self.dim // 2
            )
            tail = torch.view_as_complex(
                x[..., -self.dim :].float().contiguous().unflatten(-1, (-1, 2))
            )
            rotated = torch.view_as_real(tail * freqs).flatten(-2).to(x.dtype)
            return torch.cat((x[..., : -self.dim], rotated), dim=-1)
        self._prepare_cache(x.device)
        cache = self.inverse_cos_sin_cache if inverse else self.cos_sin_cache
        return rope_inplace(x.clone(), positions, cache, None)

    def apply_owned(
        self, x: torch.Tensor, positions: torch.Tensor, inverse: bool
    ) -> torch.Tensor:
        """Rotate a fresh projection/output in place; never pass shared latents before index K reads them."""
        if not x.is_cuda:
            return self(x, positions, inverse)
        self._prepare_cache(x.device)
        cache = self.inverse_cos_sin_cache if inverse else self.cos_sin_cache
        return rope_inplace(x, positions, cache, None)


def v41_hc_mixes(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive pre/post/comb from the full HC stream; comb is [input HC, output HC]."""
    if x.is_cuda:
        from tokenspeed_kernel.ops.residual import mhc_mixes

        return mhc_mixes(x, weight, scale, base, norm_eps, hc_eps, sinkhorn_iters)
    hc = x.shape[-2]
    flat = x.flatten(-2).float()
    mixes = F.linear(flat, weight.float()) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + norm_eps
    )
    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + hc_eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * scale[2] + base[2 * hc :]).unflatten(
        -1, (hc, hc)
    ).softmax(-1) + hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    return pre, post, comb


def v41_hc_pre(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """Collapse copies with the PREVIOUS sublayer's pre-mix, accumulating FP32."""
    if not x.is_cuda:
        return (pre_mix.unsqueeze(-1) * x.float()).sum(-2).to(x.dtype)
    from tokenspeed_kernel.ops.residual.triton import mhc_apply_pre

    return mhc_apply_pre(x, pre_mix)


def v41_hc_post(
    x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
) -> torch.Tensor:
    """Apply the current sublayer's post/comb, summing input—not output—HC copies."""
    if not x.is_cuda:
        mixed = (comb.unsqueeze(-1) * residual.float().unsqueeze(-2)).sum(-3)
        return (post.unsqueeze(-1) * x.float().unsqueeze(-2) + mixed).to(x.dtype)
    from tokenspeed_kernel import mhc_post

    return mhc_post(x, residual, post.unsqueeze(-1), comb, override=None, solution=None)


def _v41_hc_input(x: torch.Tensor, pre: torch.Tensor, norm: RMSNorm) -> torch.Tensor:
    if not x.is_cuda:
        return _norm(v41_hc_pre(x, pre), norm)
    from tokenspeed_kernel.ops.residual.triton import mhc_pre_layer_norm_hc4

    out = x.new_empty((x.shape[0], x.shape[-1]))
    mhc_pre_layer_norm_hc4(pre, x, norm.weight, out, eps=norm.variance_epsilon)
    return out


class DeepseekV41Compressor(nn.Module):
    def __init__(self, config, layer_id: int, prefix: str):
        super().__init__()
        self.ratio = config.compress_ratios[layer_id]
        if self.ratio not in (1, 2):
            raise ValueError("V4.1 KV owners require ratio 1 or 2")
        if self.ratio == 1:
            self.wkv = _replicated(
                config.hidden_size,
                config.head_dim,
                torch.bfloat16,
                None,
                add_prefix("wkv", prefix),
            )
        else:
            # The checkpoint loader requires BF16 compressor operands. Keep a
            # single merged parameter and accumulate directly into FP32 tails.
            self.wkv_wgate = _merged(
                config.hidden_size,
                [config.head_dim, config.head_dim],
                torch.bfloat16,
                None,
                add_prefix("wkv_wgate", prefix),
            )
        self.norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps).to(
            dtype=torch.bfloat16
        )

    def forward(self, x, owner, positions, requests, backend, mode):
        if self.ratio == 1:
            content, _ = self.wkv(x, block_scale=None, output_dtype=None)
        else:
            projected = (
                dsv4_linear_fp32(
                    x.contiguous(), self.wkv_wgate.weight, override=None, solution=None
                )
                if x.is_cuda
                else F.linear(x.float(), self.wkv_wgate.weight.float())
            )
            content, scores = projected.chunk(2, dim=-1)
            content, positions, requests = backend.compress(
                owner,
                content,
                scores,
                mode,
                self.norm.weight,
                self.norm.variance_epsilon,
            )
            return content, positions, requests
        return _attention_norm(content, self.norm), positions, requests


class DeepseekV41Indexer(nn.Module):
    def __init__(
        self, config, mapping: Mapping, owns_k: bool, quant_config, prefix: str
    ):
        super().__init__()
        self.n_local_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.scale = self.head_dim**-0.5 * config.index_n_heads**-0.5
        self.wq_b = _replicated(
            config.q_lora_rank,
            config.index_n_heads * self.head_dim,
            torch.bfloat16,
            quant_config,
            add_prefix("wq_b", prefix),
        )
        self.weights_proj = _replicated(
            config.hidden_size,
            config.index_n_heads,
            torch.bfloat16,
            None,
            add_prefix("weights_proj", prefix),
        )
        self.wk = (
            _replicated(
                config.head_dim,
                self.head_dim,
                torch.bfloat16,
                None,
                add_prefix("wk", prefix),
            )
            if owns_k
            else None
        )
        self.k_norm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps).to(dtype=torch.bfloat16)
            if owns_k
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        rotary: DeepseekV41RotaryEmbedding,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, _ = self.wq_b(qr, block_scale=None, output_dtype=None)
        q = rotary.apply_owned(
            q.unflatten(-1, (self.n_local_heads, self.head_dim)), positions, False
        )
        weights, _ = self.weights_proj(x, block_scale=None, output_dtype=None)
        return q, weights * self.scale

    def key(
        self,
        latent: torch.Tensor,
        positions: torch.Tensor,
        rotary: DeepseekV41RotaryEmbedding,
    ) -> torch.Tensor:
        if self.wk is None:
            raise RuntimeError("Only a KV owner can project index keys")
        key, _ = self.wk(latent, block_scale=None, output_dtype=None)
        return rotary.apply_owned(_attention_norm(key, self.k_norm), positions, False)


def _row_plan(layer_id: int, ced_decoder_start: int, ctx: ForwardContext) -> V41RowPlan:
    """Rows layer ``layer_id`` receives and attends in this forward.

    Read from the live context on every call. A captured prefill break rebinds
    ``ctx`` at replay but freezes its other non-tensor arguments, so the plan
    must be derived here rather than passed in.
    """
    backend = ctx.attn_backend
    full = backend.query_metadata(ctx.forward_mode)
    if layer_id < ced_decoder_start:
        return V41RowPlan(full, full, None)
    view = backend.decoder_view()
    if layer_id == ced_decoder_start:
        return V41RowPlan(full, view.metadata, view.keep_rows)
    return V41RowPlan(view.metadata, view.metadata, None)


class DeepseekV41Attention(nn.Module):
    def __init__(
        self,
        config,
        mapping: Mapping,
        layer_id: int,
        ced_decoder_start: int,
        quant_config,
        prefix: str,
        *,
        aux_stream: torch.cuda.Stream | None,
    ):
        super().__init__()
        self.stream_fork = StreamFork(aux_stream)
        self.layer_id = layer_id
        self.ced_decoder_start = ced_decoder_start
        self.mapping = mapping
        self.head_dim = config.head_dim
        self.n_local_heads = config.num_attention_heads // mapping.attn.tp_size
        self.n_local_groups = config.o_groups // mapping.attn.tp_size
        self.o_lora_rank = config.o_lora_rank
        self.compress_ratio = config.compress_ratios[layer_id]
        self.is_kv_source = layer_id in config.kv_source_layer_ids
        self.is_index_source = layer_id in config.index_source_layer_ids
        if self.is_kv_source and not self.is_index_source:
            raise ValueError("V4.1 KV owners must also own an indexer")
        self.attn_sink = nn.Parameter(
            torch.empty(self.n_local_heads, dtype=torch.float32), requires_grad=False
        )
        self.attn_sink.weight_loader = self.load_sink
        self.register_buffer("_padded_attn_sink", None, persistent=False)
        self.wq_a_wkv = _merged(
            config.hidden_size,
            [config.q_lora_rank, config.head_dim],
            torch.bfloat16,
            quant_config,
            add_prefix("wq_a_wkv", prefix),
        )
        self.q_norm = RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps).to(
            dtype=torch.bfloat16
        )
        self.wq_b = _column(
            config.q_lora_rank,
            config.num_attention_heads * config.head_dim,
            torch.bfloat16,
            quant_config,
            add_prefix("wq_b", prefix),
            mapping,
        )
        self.kv_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps).to(
            dtype=torch.bfloat16
        )
        self.wo_a = _column(
            config.num_attention_heads * config.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            torch.bfloat16,
            None,
            add_prefix("wo_a", prefix),
            mapping,
        )
        self.wo_b = RowParallelLinear(
            input_size=config.o_groups * config.o_lora_rank,
            output_size=config.hidden_size,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            params_dtype=torch.bfloat16,
            reduce_results=False,
            quant_config=quant_config,
            prefix=add_prefix("wo_b", prefix),
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
            use_presharded_weights=False,
            override_kernel_name=None,
            interleave_linear_and_gate=False,
        )
        configure_v41_fp8_linear(self.wo_b, True)
        self.compressor = (
            DeepseekV41Compressor(config, layer_id, add_prefix("compressor", prefix))
            if self.is_kv_source
            else None
        )
        self.indexer = (
            DeepseekV41Indexer(
                config,
                mapping,
                self.is_kv_source,
                quant_config,
                add_prefix("indexer", prefix),
            )
            if self.is_index_source
            else None
        )
        self.rotary_emb = DeepseekV41RotaryEmbedding(config, self.compress_ratio)

    def load_sink(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        start = self.mapping.attn.tp_rank * self.n_local_heads
        default_weight_loader(param, loaded_weight[start : start + self.n_local_heads])
        if self._padded_attn_sink is not None:
            self._padded_attn_sink[: self.n_local_heads].copy_(param)

    def _kernel_attn_sink(self):
        padded = 64 if self.n_local_heads <= 64 else 128
        if not self.attn_sink.is_cuda or padded == self.n_local_heads:
            return self.attn_sink
        if self._padded_attn_sink is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm up attention sink before graph capture")
            self._padded_attn_sink = F.pad(
                self.attn_sink, (0, padded - self.n_local_heads), value=-float("inf")
            )
        return self._padded_attn_sink

    @break_point
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
    ) -> torch.Tensor:
        backend, mode = ctx.attn_backend, ctx.forward_mode
        if mode is None:
            raise ValueError("V4.1 attention requires an explicit forward mode")
        rows = _row_plan(self.layer_id, self.ced_decoder_start, ctx)
        meta = rows.source
        # Prefill buckets pad token-local compute; cache writes and selection
        # must use only the live rows, including the decode suffix of a mixed batch.
        if current_forward_ctx() is not None:
            positions, hidden_states = slice_to_real_tokens(
                meta.positions.numel(), positions, hidden_states
            )
        if (
            positions.shape != meta.positions.shape
            or positions.numel() != hidden_states.shape[0]
        ):
            raise ValueError("V4.1 hidden rows and backend query metadata disagree")
        # Backend positions carry -1 for padding; request indices are batch rows.
        positions, requests = meta.positions, meta.request_indices
        qkv, _ = self.wq_a_wkv(hidden_states, block_scale=None, output_dtype=None)
        qr, swa = qkv.split((self.q_norm.weight.numel(), self.head_dim), dim=-1)
        qr = _attention_norm(qr, self.q_norm)
        if hidden_states.is_cuda:
            # Lazy shared constants are produced before the fork event, so both
            # branches observe them even on the first eager decode after loading.
            self.rotary_emb._prepare_cache(hidden_states.device)
        overlap = (
            mode.is_decode()
            and self.is_index_source
            and hidden_states.is_cuda
            and hidden_states.numel() > 0
            and self.stream_fork.aux_stream is not None
            and self.stream_fork.aux_stream.device == hidden_states.device
        )
        consumer = torch.cuda.current_stream() if overlap else None
        if overlap:
            for tensor in (hidden_states, qr, positions, requests):
                tensor.record_stream(self.stream_fork.aux_stream)
        index_q, index_weights, index_group, prepared = None, None, None, None
        with self.stream_fork.scope(enable=overlap, overlap=True) as fork:
            # Submit the longer compressor/index branch first. Serial eager and
            # captured decode execute these identical operations and dependency joins.
            with fork.branch():
                if self.compressor is not None:
                    latent, row_positions, row_requests = self.compressor(
                        hidden_states, self.layer_id, positions, requests, backend, mode
                    )
                    index_k = self.indexer.key(latent, row_positions, self.rotary_emb)
                    main_k = self.rotary_emb.apply_owned(latent, row_positions, False)
                    backend.write_global(
                        self.layer_id,
                        main_k,
                        index_k,
                        row_positions,
                        row_requests,
                        mode,
                    )
                if rows.keep_rows is not None:
                    # CED: the decoder's global KV above came from every
                    # encoder row; from here on only the per-request tail
                    # is attended and carried forward.
                    hidden_states, qr, swa = (
                        t.index_select(0, rows.keep_rows)
                        for t in (hidden_states, qr, swa)
                    )
                    positions, requests = (
                        rows.query.positions,
                        rows.query.request_indices,
                    )
                if self.indexer is not None:
                    index_q, index_weights = self.indexer(
                        hidden_states, qr, positions, self.rotary_emb
                    )
                    if mode.is_decode():
                        prepared = backend.prepare_global_selection(
                            self.layer_id,
                            index_q,
                            index_weights,
                            positions,
                            requests,
                            mode,
                            index_group,
                        )
                        index_q = index_weights = None
            q, _ = self.wq_b(qr, block_scale=None, output_dtype=None)
            q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
            if q.is_cuda and mode.is_decode() and self.head_dim == 512:
                q = rope_pad_query(q, positions, self.rotary_emb.cos_sin_cache, None)
            else:
                q = self.rotary_emb.apply_owned(q, positions, False)
            swa = _attention_norm(swa, self.kv_norm)
            swa_rope_cache = None
            if swa.is_cuda and self.head_dim == 512 and self.rotary_emb.dim == 64:
                self.rotary_emb._prepare_cache(swa.device)
                swa_rope_cache = self.rotary_emb.cos_sin_cache
            else:
                swa = self.rotary_emb.apply_owned(swa, positions, False)
        if overlap:
            # Aux-owned scratch can outlive this layer through source/reuse
            # records. Main consumers and allocator reuse both follow the join.
            for tensor in prepared:
                if tensor is not None:
                    tensor.record_stream(consumer)
            record = backend.sparse_topk.decode
            for tensor in (record.logical_rows, record.lengths):
                tensor.record_stream(consumer)
            if record.candidates is not None:
                for tensor in (record.candidates.block_ids, record.candidates.lengths):
                    tensor.record_stream(consumer)
        out = backend.forward_v41(
            q,
            swa,
            layer_id=self.layer_id,
            positions=positions,
            request_indices=requests,
            forward_mode=mode,
            index_q=index_q,
            index_weights=index_weights,
            attn_sink=self._kernel_attn_sink(),
            softmax_scale=self.head_dim**-0.5,
            index_process_group=index_group,
            swa_rope_cache=swa_rope_cache,
        )
        out = self.rotary_emb.apply_owned(out[:, : self.n_local_heads], positions, True)
        grouped = out.reshape(out.shape[0], self.n_local_groups, -1)
        weight = self.wo_a.weight.reshape(self.n_local_groups, self.o_lora_rank, -1)
        out = grouped_bf16_projection(grouped, weight, None, None).flatten(1)
        out, _ = self.wo_b(out, scale=None)
        if self.mapping.attn.has_tp:
            out = all_reduce(
                out,
                group=self.mapping.attn.tp_group,
                backend=None,
                op=torch.distributed.ReduceOp.SUM,
            )
        return out


class DeepseekV41MoE(DeepseekV4MoE):
    """V4 expert execution with V4.1 routing and lossless MegaMoE alignment."""

    def __init__(self, config, mapping, quant_config, layer_index, prefix, aux_stream):
        expert_config = config
        padded = (
            get_moe_backend().is_mega_moe() and config.moe_intermediate_size % 512 != 0
        )
        if padded:
            # MegaMoE TMA loads require a 16-byte-aligned 1x32 scale row.
            # V4.1's 2304-wide routed FFN therefore needs 2560-wide storage.
            expert_config = copy(config)
            expert_config.moe_intermediate_size = (
                (config.moe_intermediate_size + 511) // 512 * 512
            )
            expert_config.n_shared_experts = None
        super().__init__(
            expert_config, mapping, quant_config, layer_index, prefix, aux_stream
        )
        self.gate.register_parameter("bias_vl", None)
        if padded:
            self.config = config
            self.n_shared_experts = config.n_shared_experts
            if config.n_shared_experts is not None:
                self.shared_experts = DeepseekV4MLP(
                    config.hidden_size,
                    config.moe_intermediate_size * config.n_shared_experts,
                    config.hidden_act,
                    mapping,
                    quant_config,
                    add_prefix("shared_experts", prefix),
                    swiglu_limit=config.swiglu_limit,
                    reduce_results=False,
                    is_shared_expert=False,
                )

    def _select_experts(self, hidden_states, image_mask):
        bias_vl = self.gate.bias_vl
        if hidden_states.is_cuda and (bias_vl is None or image_mask is None):
            return super()._select_experts(hidden_states, None)
        bias = self.gate.e_score_correction_bias
        if bias_vl is not None and image_mask is not None:
            bias = torch.where(image_mask.unsqueeze(-1), bias_vl, bias)
        logits = F.linear(hidden_states.float(), self.gate.weight.float())
        scores = F.softplus(logits).sqrt()
        ids = (scores + bias).topk(self.config.num_experts_per_tok, dim=-1).indices
        weights = scores.gather(1, ids)
        if self.config.norm_topk_prob and self.config.num_experts_per_tok > 1:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        # V4 applies routed_scaling_factor at its expert execution boundary.
        return weights, ids.to(self.hash_indices_dtype), scores


class DeepseekV41DecoderLayer(nn.Module):
    """Overlap HC coefficients with decode sublayer work on CUDA.

    Sublayer inputs use the previous pre-mix, so the current coefficients need
    only join before HC post. Eager and captured forwards use the same forks.
    Prefill stays on the main stream: extra stream submissions can outweigh
    overlap in small eager batches. Decode has no token-count threshold.
    """

    def __init__(
        self,
        config,
        mapping: Mapping,
        layer_id: int,
        ced_decoder_start: int,
        quant_config,
        prefix: str,
        aux_stream,
        hc_stream: torch.cuda.Stream | None,
        host_table: bool,
        host_layout: str,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.ced_decoder_start = ced_decoder_start
        self.norm_eps, self.hc_eps = config.rms_norm_eps, config.hc_eps
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_stream_fork = StreamFork(hc_stream)
        dense_quant = v41_mxfp8_config(quant_config)
        self.attn = DeepseekV41Attention(
            config,
            mapping,
            layer_id,
            ced_decoder_start,
            dense_quant,
            add_prefix("attn", prefix),
            aux_stream=aux_stream,
        )
        self.ffn = DeepseekV41MoE(
            config,
            mapping,
            dense_quant,
            layer_id,
            add_prefix("ffn", prefix),
            aux_stream=aux_stream,
        )
        if self.ffn.shared_experts is not None:
            for module in self.ffn.shared_experts.modules():
                if isinstance(module, LinearBase):
                    configure_v41_fp8_linear(module, True)
        self.comm_manager = CommManager(
            mapping=mapping,
            layer_id=layer_id,
            is_moe=True,
            prev_is_moe=True,
            input_layernorm=None,
            post_attn_layernorm=None,
        )
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.engram = (
            DeepseekV41Engram(
                config,
                layer_id,
                mapping,
                quant_config,
                add_prefix("engram", prefix),
                self.attn.wq_a_wkv.weight.device,
                host_table,
                host_layout,
            )
            if layer_id in config.engram_layer_ids
            else None
        )
        if self.engram is not None:
            configure_v41_fp8_linear(self.engram.wkv, False)
        mix_hc = (2 + config.hc_mult) * config.hc_mult
        for name in ("attn", "ffn"):
            for suffix, shape in (
                ("fn", (mix_hc, config.hc_mult * config.hidden_size)),
                ("base", (mix_hc,)),
                ("scale", (3,)),
            ):
                param = nn.Parameter(
                    torch.empty(shape, dtype=torch.float32), requires_grad=False
                )
                param.weight_loader = default_weight_loader
                self.register_parameter(f"hc_{name}_{suffix}", param)

    def forward(self, hidden_states, pre_mix, positions, image_mask, ctx):
        rows = _row_plan(self.layer_id, self.ced_decoder_start, ctx)
        residual = hidden_states
        overlap = (
            residual.is_cuda
            and ctx.forward_mode is not None
            and ctx.forward_mode.is_decode()
            and self.hc_stream_fork.aux_stream is not None
            and self.hc_stream_fork.aux_stream.device == residual.device
        )
        consumer = torch.cuda.current_stream() if overlap else None
        if overlap:
            residual.record_stream(self.hc_stream_fork.aux_stream)
        with self.hc_stream_fork.scope(enable=overlap, overlap=True) as fork:
            with fork.branch():
                attn_pre, post, comb = v41_hc_mixes(
                    residual,
                    self.hc_attn_fn,
                    self.hc_attn_scale,
                    self.hc_attn_base,
                    self.norm_eps,
                    self.hc_eps,
                    self.hc_sinkhorn_iters,
                )
            if overlap:
                for tensor in (attn_pre, post, comb):
                    tensor.record_stream(consumer)
            x = _v41_hc_input(residual, pre_mix, self.attn_norm)
            x = self.attn(positions, x, ctx)
        if rows.keep_rows is not None:
            residual, post, comb, attn_pre = (
                t.index_select(0, rows.keep_rows)
                for t in (residual, post, comb, attn_pre)
            )
            if image_mask is not None:
                image_mask = image_mask.index_select(0, rows.keep_rows)
        hidden_states = v41_hc_post(x, residual, post, comb)
        residual = hidden_states
        if overlap:
            residual.record_stream(self.hc_stream_fork.aux_stream)
        with self.hc_stream_fork.scope(enable=overlap, overlap=True) as fork:
            with fork.branch():
                ffn_pre, post, comb = v41_hc_mixes(
                    residual,
                    self.hc_ffn_fn,
                    self.hc_ffn_scale,
                    self.hc_ffn_base,
                    self.norm_eps,
                    self.hc_eps,
                    self.hc_sinkhorn_iters,
                )
            if overlap:
                for tensor in (ffn_pre, post, comb):
                    tensor.record_stream(consumer)
            x = _v41_hc_input(residual, attn_pre, self.ffn_norm)
            if self.ffn.use_mega_moe:
                counts = self.comm_manager.moe_tp_ep_group_scattered_num_tokens(ctx)
                x = self.ffn(
                    x,
                    image_mask,
                    sum(counts),
                    max(counts),
                    ctx=ctx,
                    comm_manager=self.comm_manager,
                )
            else:
                x = self.comm_manager.pre_mlp_comm(x, ctx)
                total, maximum = self.comm_manager.get_num_tokens(ctx)
                x = self.ffn(x, image_mask, total, maximum, ctx=None, comm_manager=None)
                x, _ = self.comm_manager.post_mlp_comm(x, None, ctx)
        return v41_hc_post(x, residual, post, comb), ffn_pre


def _ced_decoder_start(config) -> int:
    """First decoder layer of the causal encoder-decoder split.

    The decoder's global KV is projected by its first layer from the encoder
    output and reused by every later layer, so that layer is the candidate
    source and the last KV owner: layers after it never run a compressor, so
    they can run on the narrowed decoder rows. Anything else is not the CED
    layout this model shortens.
    """
    layers = int(config.num_hidden_layers)
    owners = [int(layer) for layer in config.kv_source_layer_ids]
    if not owners:
        # A window-only stack (the DSpark draft) has no global KV and hence no
        # decoder to shorten; every layer runs on the full rows.
        return layers
    start = int(config.candidate_source_layer_id)
    if not 0 < start < layers or owners[-1] != start:
        raise ValueError(
            "V4.1 CED requires the candidate source to be the last KV owner"
        )
    return start


@dataclass
class V41RowState:
    """The row-shaped activations carried between forward stages.

    ``hidden`` is the HC residual stream ``[rows, hc_mult, hidden]`` and
    ``pre_mix`` its ``[rows, hc_mult]`` fp32 input mix; ``positions`` and the
    optional ``image_mask`` describe the same rows. ``hashes`` and
    ``engram_token_mask`` are present only while a later layer runs Engram.
    ``captured`` holds the DSpark taps taken so far, one ``[rows, hidden]``
    tensor per tap layer. The prefill graph allocates a zero state of a fixed
    row count as the decoder graph's input (``allocate_decoder_state``) and
    lands each forward's narrowed state into it (``land_into``).
    """

    hidden: torch.Tensor
    pre_mix: torch.Tensor
    positions: torch.Tensor
    image_mask: torch.Tensor | None
    hashes: torch.Tensor | None
    engram_token_mask: torch.Tensor | None
    captured: list[torch.Tensor]

    @property
    def rows(self) -> int:
        return int(self.hidden.shape[0])

    def _tensors(self) -> list[torch.Tensor | None]:
        return [
            self.hidden,
            self.pre_mix,
            self.positions,
            self.image_mask,
            self.hashes,
            self.engram_token_mask,
            *self.captured,
        ]

    def _rebuilt(self, tensors: list[torch.Tensor | None]) -> V41RowState:
        hidden, pre_mix, positions, image_mask, hashes, mask, *captured = tensors
        return V41RowState(
            hidden, pre_mix, positions, image_mask, hashes, mask, list(captured)
        )

    def leading(self, rows: int) -> V41RowState:
        """The first ``rows`` rows of every tensor, as views."""
        return self._rebuilt([t if t is None else t[:rows] for t in self._tensors()])

    def land_into(self, dst: V41RowState) -> None:
        """Copy this state into the leading rows of ``dst`` and zero its tail.

        ``dst`` is a fixed-row static state (see ``allocate_decoder_state``);
        the two must agree on which optional tensors are present.
        """
        rows = self.rows
        if rows > dst.rows:
            raise ValueError(
                f"V4.1 row state of {rows} rows exceeds the {dst.rows}-row target"
            )
        for src, target in zip(self._tensors(), dst._tensors(), strict=True):
            if (src is None) != (target is None):
                raise ValueError("V4.1 row states disagree on their optional tensors")
            if src is None:
                continue
            target[:rows].copy_(src)
            if rows < target.shape[0]:
                target[rows:].zero_()


class DeepseekV41Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        prefix: str,
        host_table: bool,
        host_layout: str,
    ):
        super().__init__()
        if mapping.pp_size != 1 or mapping.attn.cp_size != 1:
            raise NotImplementedError("V4.1 full-prompt baseline requires PP=CP=1")
        if mapping.attn.tp_size != mapping.moe.tp_ep_size:
            raise NotImplementedError(
                "V4.1 HC residuals require attention TP == MoE TPxEP"
            )
        if config.hc_mult != 4 or config.hc_sinkhorn_iters < 1:
            raise ValueError(
                "V4.1 requires four HC copies and at least one Sinkhorn iteration"
            )
        if config.num_hash_layers != 0:
            raise ValueError("V4.1 uses learned MoE routing, not V4 hash routing")
        if (
            any(
                n % mapping.attn.tp_size
                for n in (
                    config.num_attention_heads,
                    config.o_groups,
                    config.index_n_heads,
                )
            )
            or config.num_attention_heads % config.o_groups
        ):
            raise ValueError(
                "Attention/index heads and output groups must divide across attention TP"
            )
        if (
            config.num_hidden_layers < 1
            or len(config.compress_ratios) < config.num_hidden_layers
        ):
            raise ValueError("V4.1 layer count and compression ratios disagree")
        self.config, self.mapping = config, mapping
        self.pp_start_layer, self.pp_end_layer = 0, config.num_hidden_layers
        self.engram_hash = None
        self.dspark_capture_layers = ()
        self.ced_decoder_start = _ced_decoder_start(config)
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            params_dtype=torch.bfloat16,
            org_num_embeddings=None,
            padding_size=64,
            quant_config=None,
            prefix=add_prefix("embed_tokens", prefix),
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
            use_presharded_weights=False,
        )
        device = self.embed_tokens.weight.device
        self.aux_stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )
        # HC must also overlap the attention/index and shared-expert branches.
        self.hc_stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV41DecoderLayer(
                    config,
                    mapping,
                    layer_id,
                    self.ced_decoder_start,
                    quant_config,
                    add_prefix(f"layers.{layer_id}", prefix),
                    self.aux_stream,
                    self.hc_stream,
                    host_table,
                    host_layout,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # The decoder view keeps at most a prompt's last window per extend
        # request (and one row per decode request), which bounds the rows the
        # decoder stage can see for a given request count.
        self.max_decoder_rows_per_request = int(config.sliding_window)
        # Engram history only travels past the narrowing while a decoder
        # layer still reads it.
        self.decoder_uses_engram = any(
            layer_id > self.ced_decoder_start for layer_id in config.engram_layer_ids
        )

    def initialize_engram(self, tokenizer) -> None:
        """Initialize immutable tokenizer/hash constants once; never request history."""
        if self.engram_hash is not None:
            raise RuntimeError("Engram tokenizer is already initialized")
        if self.config.engram_layer_ids:
            self.engram_hash = EngramHashState(
                self.config, tokenizer, self.embed_tokens.weight.device
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        input_embeds: torch.Tensor | None,
        pp_inbound: PPStageState | None,
        *,
        engram_previous_tokens: torch.Tensor | None,
        engram_token_mask: torch.Tensor | None,
        image_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """Run all layers and return normalized [T,hidden] plus optional hidden capture.

        Engram history/mask must describe every input row in the same TP-replicated
        order as backend metadata. image_mask marks image-span rows; None uses
        text routing. pp_inbound must be None; draft and memory-only/CED
        invocations are unsupported. The four stages below are the whole
        forward; the prefill graph calls them individually.
        """
        if input_ids.numel() == 0:
            return (
                self.embed_tokens.weight.new_empty((0, self.config.hidden_size)),
                None,
            )
        state = self.encoder_forward(
            input_ids,
            positions,
            ctx,
            input_embeds,
            pp_inbound,
            engram_previous_tokens=engram_previous_tokens,
            engram_token_mask=engram_token_mask,
            image_mask=image_mask,
        )
        state = self.narrowing_forward(state, ctx)
        hidden, captured = self.decoder_forward(state, ctx)
        return self.finish_forward(hidden, captured, ctx)

    def _run_layer(
        self,
        layer: DeepseekV41DecoderLayer,
        hidden: torch.Tensor,
        pre_mix: torch.Tensor,
        state: V41RowState,
        ctx: ForwardContext,
        captured: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One layer on ``hidden``/``pre_mix`` over the rows ``state`` describes.

        Engram and the DSpark taps sit at the layer input; a tap is the
        unweighted HC mean the draft was trained on, appended to ``captured``.
        """
        if layer.engram is not None:
            hidden = layer.engram(
                hidden,
                state.hashes[:, layer.engram.layer_hash_index],
                state.engram_token_mask,
            )
        if layer.layer_id in self.dspark_capture_layers:
            captured.append(hidden.mean(dim=1))
        return layer(hidden, pre_mix, state.positions, state.image_mask, ctx)

    def encoder_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        input_embeds: torch.Tensor | None,
        pp_inbound: PPStageState | None,
        *,
        engram_previous_tokens: torch.Tensor | None,
        engram_token_mask: torch.Tensor | None,
        image_mask: torch.Tensor | None,
    ) -> V41RowState:
        """Embed and run the layers below the candidate source, one row per token."""
        if pp_inbound is not None:
            raise NotImplementedError("V4.1 baseline does not accept pipeline state")
        if input_ids.ndim != 1 or positions.shape != input_ids.shape:
            raise ValueError(
                "V4.1 expects packed one-dimensional token IDs and positions"
            )
        hashes = None
        if self.config.engram_layer_ids:
            if (
                self.engram_hash is None
                or engram_previous_tokens is None
                or engram_token_mask is None
            ):
                raise RuntimeError(
                    "Initialize Engram and provide previous-three tokens and current token mask"
                )
            hashes = self.engram_hash(
                input_ids, engram_previous_tokens, engram_token_mask
            )
        h = self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        if h.shape != (input_ids.numel(), self.config.hidden_size):
            raise ValueError("V4.1 input embeddings have the wrong shape")
        h = h.unsqueeze(1).repeat(1, self.config.hc_mult, 1)
        pre_mix = torch.zeros(h.shape[:2], dtype=torch.float32, device=h.device)
        pre_mix[:, 0] = 1
        state = V41RowState(
            h, pre_mix, positions, image_mask, hashes, engram_token_mask, []
        )
        for layer in self.layers[: self.ced_decoder_start]:
            h, pre_mix = self._run_layer(layer, h, pre_mix, state, ctx, state.captured)
        state.hidden, state.pre_mix = h, pre_mix
        return state

    def narrowing_forward(self, state: V41RowState, ctx: ForwardContext) -> V41RowState:
        """Run the candidate source layer and narrow to the decoder view.

        The layer projects the decoder's global KV from every row of ``state``
        and returns the view's rows; the row descriptors and the taps taken so
        far follow ``keep_rows``. ``state`` may carry a padded tail past the
        forward's real rows (an encoder graph output): only the real rows are
        used. A window-only stack has no candidate source and passes through.
        """
        start = self.ced_decoder_start
        if start >= len(self.layers):
            return state
        backend = ctx.attn_backend
        rows = backend.query_metadata(ctx.forward_mode).positions.numel()
        if state.rows < rows:
            raise ValueError("V4.1 encoder rows fall short of the forward's tokens")
        if state.rows > rows:
            state = state.leading(rows)
        view = backend.decoder_view()
        # Checked here, in the stage that always runs eagerly: a replayed
        # encoder graph would skip a check placed before it.
        if view.keep_rows is not None and ctx.global_num_tokens is not None:
            raise NotImplementedError(
                "V4.1 CED narrowing under attention data parallelism needs the "
                "narrowed row counts exchanged across ranks"
            )
        captured = list(state.captured)
        with report_collective_sizing(ctx, view.metadata.positions.numel(), None):
            hidden, pre_mix = self._run_layer(
                self.layers[start], state.hidden, state.pre_mix, state, ctx, captured
            )
        hashes, mask = (
            (state.hashes, state.engram_token_mask)
            if self.decoder_uses_engram
            else (None, None)
        )
        keep = view.keep_rows

        def kept(t: torch.Tensor | None) -> torch.Tensor | None:
            return t if (t is None or keep is None) else t.index_select(0, keep)

        # Taps before this layer captured every row; the drafter reads all
        # taps in the layout ctx.captured_rows reports.
        return V41RowState(
            hidden,
            pre_mix,
            kept(state.positions),
            kept(state.image_mask),
            kept(hashes),
            kept(mask),
            [kept(tap) for tap in captured],
        )

    def decoder_forward(
        self, state: V41RowState, ctx: ForwardContext
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the layers above the candidate source and the final norm.

        Every row of ``state`` is computed and sized into the collectives: the
        prefill graph hands a fixed-row state whose tail is padding and slices
        the result. Returns the normalized rows and the DSpark taps.
        """
        start = self.ced_decoder_start
        captured = list(state.captured)
        h, pre_mix = state.hidden, state.pre_mix
        layers = self.layers[start + 1 :]
        if layers:
            with report_collective_sizing(ctx, state.rows, None):
                for layer in layers:
                    h, pre_mix = self._run_layer(
                        layer, h, pre_mix, state, ctx, captured
                    )
        return _norm(v41_hc_pre(h, pre_mix), self.norm), captured

    def finish_forward(
        self,
        hidden: torch.Tensor,
        captured: list[torch.Tensor],
        ctx: ForwardContext,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """Gather the sampled rows and report the narrowed tap layout."""
        view = ctx.attn_backend.decoder_view()
        if view.logits_rows is not None:
            hidden = hidden.index_select(0, view.logits_rows)
        capture = (
            ctx.capture_hidden_mode is not None
            and ctx.capture_hidden_mode.need_capture()
        )
        if capture and view.keep_rows is not None:
            ctx.captured_rows = CapturedRows(
                view.metadata.positions,
                tuple((span.offset, span.count) for span in view.spans),
            )
        return hidden, (captured or [hidden]) if capture else None

    def decoder_rows(self, ctx: ForwardContext) -> int:
        """Rows ``decoder_forward`` receives for the forward ``ctx`` describes."""
        return int(ctx.attn_backend.decoder_view().metadata.positions.numel())

    def allocate_decoder_state(self, rows: int) -> V41RowState:
        """A zero ``rows``-row state laid out like ``narrowing_forward``'s output.

        Text routing only (no image mask): a multimodal forward never lands
        here. Taps taken at or below the candidate source get one static each.
        """
        weight = self.embed_tokens.weight
        device, dtype = weight.device, weight.dtype
        hidden_size = int(self.config.hidden_size)
        hashes = mask = None
        if self.decoder_uses_engram:
            # Hashes are [rows, layers, hash columns], the offsets' trailing shape.
            hashes = torch.zeros(
                (rows, *self.engram_hash.offsets.shape),
                dtype=torch.int64,
                device=device,
            )
            mask = torch.zeros(rows, dtype=torch.bool, device=device)
        taps = sum(
            1
            for layer_id in self.dspark_capture_layers
            if layer_id <= self.ced_decoder_start
        )
        return V41RowState(
            torch.zeros(
                (rows, self.config.hc_mult, hidden_size), dtype=dtype, device=device
            ),
            torch.zeros(
                (rows, self.config.hc_mult), dtype=torch.float32, device=device
            ),
            torch.zeros(rows, dtype=torch.int64, device=device),
            None,
            hashes,
            mask,
            [
                torch.zeros((rows, hidden_size), dtype=dtype, device=device)
                for _ in range(taps)
            ],
        )


class DeepseekV41ForCausalLM(BaseCausalLM):
    """V4.1 language and vision model with strict checkpoint coverage."""

    model_cls = DeepseekV41Model

    def __init__(
        self,
        config: DeepseekV41Config,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        is_multimodal_active: bool,
        mm_attention_backend: str | None,
    ) -> None:
        super().__init__(
            config=config,
            mapping=mapping,
            quant_config=quant_config,
            prefix="",
            encoder_only=getattr(config, "encoder_only", False),
        )
        self.is_multimodal_active = is_multimodal_active
        self.vision = None
        self.vision_embedder = None
        self.image_encoder = None
        if is_multimodal_active:
            dtype = (
                self.get_input_embeddings().weight.dtype
                if self.model is not None
                else torch.get_default_dtype()
            )
            with set_default_torch_dtype(dtype):
                self.vision = DeepseekV41Vision(config, mapping, mm_attention_backend)
            if self.model is not None:
                for layer in self.model.layers:
                    layer.ffn.gate.bias_vl = nn.Parameter(
                        torch.empty(
                            config.text_config.n_routed_experts, dtype=torch.float32
                        ),
                        requires_grad=False,
                    )
            self.vision_embedder = VisionEmbedder(encoder_mapping=mapping.vision)
            self.image_encoder = self.vision.embed_media

    def initialize_engram(self, tokenizer) -> None:
        if self.model is not None:
            self.model.initialize_engram(tokenizer)

    def set_dspark_layers_to_capture(self, layer_ids: list[int]) -> None:
        """Capture ordered, unique target layer inputs for the checkpoint draft."""
        layers = tuple(layer_ids)
        if (
            not layers
            or tuple(sorted(set(layers))) != layers
            or layers[0] < 0
            or layers[-1] >= self.model.config.num_hidden_layers
        ):
            raise ValueError("DSpark capture layers must be ordered target layer IDs")
        self.model.dspark_capture_layers = layers
        self.capture_aux_hidden_states = True

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return DeepseekV4ForCausalLM.get_model_config_for_expert_location(
            config.text_config
        )

    def resolve_model(self, config, mapping, quant_config, prefix):
        host_table = global_server_args_dict["engram_host_table"]
        return self.model_cls(
            config.text_config,
            mapping,
            quant_config,
            add_prefix("model", prefix),
            host_table,
            resolve_engram_host_layout(host_table, mapping.attn.tp_size),
        )

    def bind_checkpoint_dir(self, checkpoint_dir: str) -> None:
        self._checkpoint_dir = checkpoint_dir

    def checkpoint_weight_name_filter(self, name: str) -> bool:
        if self.encoder_only:
            raw = self._checkpoint_name(name)
            return raw.startswith(("vision.", "aligner.")) or raw in (
                "image_start",
                "image_end",
                "image_newline",
            )
        return not is_engram_embed_checkpoint_name(self._checkpoint_name(name))

    def resolve_lm_head(self, config, quant_config, prefix):
        """Keep the checkpoint head unquantized in the model loading dtype."""
        config = config.text_config
        params_dtype = torch.get_default_dtype()
        if self.mapping.attn.has_dp:
            return ReplicatedLinear(
                input_size=config.hidden_size,
                output_size=config.vocab_size,
                bias=False,
                skip_bias_add=False,
                params_dtype=params_dtype,
                quant_config=None,
                prefix=add_prefix("lm_head", prefix),
            )
        return ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            bias=False,
            params_dtype=params_dtype,
            org_num_embeddings=None,
            padding_size=64,
            quant_config=None,
            prefix=add_prefix("lm_head", prefix),
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
            use_presharded_weights=False,
        )

    def resolve_logits_processor(self, config):
        return super().resolve_logits_processor(config.text_config)

    def prepare_model_kwargs(
        self, ctx: ForwardContext, input_ids: torch.Tensor, kwargs: dict
    ) -> dict:
        result = {
            "input_embeds": kwargs.get("input_embeds", kwargs.get("inputs_embeds")),
            "pp_inbound": kwargs.get("pp_inbound"),
            "engram_previous_tokens": kwargs.get("engram_previous_tokens"),
            "engram_token_mask": kwargs.get("engram_token_mask"),
            "image_mask": kwargs.get("image_mask"),
        }
        mm_context = kwargs.get("multimodal_context")
        if (
            mm_context is not None
            and mm_context.has_extend_inputs()
            and not ctx.forward_mode.is_decode_or_idle()
        ):
            # Capture the image mask before restoring hashed placeholder IDs.
            result["image_mask"] = is_mm_pad_value_for(input_ids, Modality.IMAGE)
            substitute_mm_pad_(input_ids, {Modality.IMAGE: self.config.image_token_id})
            embeds, _ = self.vision_embedder.apply(
                input_ids=input_ids,
                text_embedding=self.get_input_embeddings(),
                ctx=mm_context,
                encoders=self.get_multimodal_encoder_specs(),
                multimodal_model=self,
            )
            if embeds is not None:
                result["input_embeds"] = embeds
        return result

    @staticmethod
    def _checkpoint_name(name: str) -> str:
        name = name.removeprefix("model.")
        if name.startswith("embed_tokens."):
            name = "embed." + name.removeprefix("embed_tokens.")
        elif name.startswith("lm_head."):
            name = "head." + name.removeprefix("lm_head.")
        return name.replace(".weight_scale_inv", ".scale").replace(
            ".ffn.gate.e_score_correction_bias", ".ffn.gate.bias"
        )

    def _checkpoint_targets(self) -> dict[str, tuple[str, int | None]]:
        """Map each required local checkpoint constituent, not just fused parameters.

        Separate w1/w3 and every local expert's weight/scale are required. Remote
        EP experts are validated and skipped by load_weights, never mistaken for
        missing local shards. Engram's own alias map takes precedence over scale
        normalization. wo_a.scale is required dynamically for FP8 wo_a weights.
        """
        targets = {}
        for name, _ in self.named_parameters():
            if name.startswith("vision."):
                continue
            raw = self._checkpoint_name(name)
            if ".ffn.experts." in raw:
                prefix, field = raw.split(".experts.")
                projection, suffix = field.split("_", 1)
                if projection not in ("w13", "w2") or suffix not in (
                    "weight",
                    "weight_scale",
                ):
                    raise ValueError(f"Unsupported V4.1 expert parameter: {name}")
                count = self.model.config.n_routed_experts // self.mapping.moe.ep_size
                for expert in range(
                    self.mapping.moe.ep_rank * count,
                    (self.mapping.moe.ep_rank + 1) * count,
                ):
                    for shard in (("w1", "w3") if projection == "w13" else ("w2",)):
                        field = "scale" if suffix == "weight_scale" else "weight"
                        targets[f"{prefix}.experts.{expert}.{shard}.{field}"] = (
                            name,
                            None,
                        )
            elif ".shared_experts.gate_up_proj." in raw:
                for shard_id, shard in enumerate(("w1", "w3")):
                    targets[raw.replace(".gate_up_proj.", f".{shard}.")] = (
                        name,
                        shard_id,
                    )
            elif ".wq_a_wkv." in raw or ".wkv_wgate." in raw:
                merged, shards = (
                    ("wq_a_wkv", ("wq_a", "wkv"))
                    if ".wq_a_wkv." in raw
                    else ("wkv_wgate", ("wkv", "wgate"))
                )
                for shard_id, shard in enumerate(shards):
                    targets[raw.replace(f".{merged}.", f".{shard}.")] = (name, shard_id)
            elif ".engram." not in raw:
                targets[
                    raw.replace(".shared_experts.down_proj.", ".shared_experts.w2.")
                ] = (name, None)
        for module_name, module in self.named_modules():
            if isinstance(module, DeepseekV41Engram):
                for raw, target in module.checkpoint_weight_aliases().items():
                    targets[raw] = (
                        module_name + target.removeprefix(module.prefix),
                        None,
                    )
        return targets

    def _skip_checkpoint_weight(self, name: str) -> str | None:
        if name.startswith(("vision.", "aligner.")) or name in (
            "image_start",
            "image_end",
            "image_newline",
        ):
            return "vision"
        if name.startswith("mtp."):
            return "draft"
        match = re.fullmatch(r"layers\.(\d+)\.ffn\.gate\.bias_vl", name)
        if (
            match
            and int(match[1]) < self.model.config.num_hidden_layers
            and self.model.layers[int(match[1])].ffn.gate.bias_vl is None
        ):
            return "bias_vl"
        return None

    def _load_wo_a(self, name, tensor, pending) -> None:
        layer_id = int(name.split(".")[1])
        linear = self.model.layers[layer_id].attn.wo_a
        field = name.rsplit(".", 1)[1]
        expected = (linear.output_size, linear.input_size)
        if field == "scale":
            expected = tuple((dim + 31) // 32 for dim in expected)
            if tensor.dtype not in (torch.uint8, torch.float8_e8m0fnu):
                raise TypeError(f"{name} must contain E8M0 exponent bytes")
        elif tensor.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            raise TypeError(f"{name} must be BF16 or checkpoint FP8 E4M3")
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{name}: expected {expected}, got {tuple(tensor.shape)}")
        key = name.rsplit(".", 1)[0]
        rows = linear.weight.shape[0]
        start = self.mapping.attn.tp_rank * rows
        # Retain only local codes/scales even when the two arrive in different files.
        if field == "scale":
            local = (
                tensor[start // 32 : (start + rows + 31) // 32]
                .view(torch.uint8)
                .clone()
            )
        else:
            local = tensor[start : start + rows].clone()
        pair = pending.setdefault(key, {})
        pair[field] = local
        weight = pair.get("weight")
        scale = pair.get("scale")
        if weight is None:
            return
        if weight.dtype == torch.bfloat16:
            if scale is not None:
                raise ValueError(f"{key}: BF16 wo_a must not have a quantization scale")
            default_weight_loader(linear.weight, weight)
            # Keep the dtype marker to reject a scale arriving after BF16 weights.
            pair["weight"] = weight.new_empty(0)
        elif scale is not None:
            scale = scale.to(linear.weight.device).view(torch.float8_e8m0fnu).float()
            scale = scale.repeat_interleave(32, dim=0)[start % 32 : start % 32 + rows]
            weight = weight.to(linear.weight.device).float().unflatten(-1, (-1, 32))
            default_weight_loader(
                linear.weight,
                (weight * scale.unsqueeze(-1)).flatten(-2).to(torch.bfloat16),
            )
            del pending[key]

    @torch.no_grad()
    def _load_text_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]], **kwargs
    ) -> None:
        """Load one complete text checkpoint; raise on missing/duplicate/unknown data.

        Args:
            weights: One-pass generic iterator of raw or model-prefixed names and
                CPU tensors. Engram embed.weight/scale must not appear here;
                bind_checkpoint_dir plus safetensors get_slice loads those.
                Other Engram parameters use this iterator. No whole-table
                dtype/device conversion is performed.

        Returns:
            None. checkpoint_load_report records loaded/local and explicit skip
            counts after successful validation. The generic runtime loader must
            subsequently run its usual quantization postprocessing hooks.
        """
        if kwargs:
            raise TypeError(
                f"Unsupported V4.1 load_weights arguments: {sorted(kwargs)}"
            )
        params = dict(self.named_parameters())
        targets = self._checkpoint_targets()
        config = self.model.config
        moe_loader = build_moe_checkpoint_loader(
            params_dict=params,
            expert_schema=ExpertCheckpointSchema(
                gate_proj_name="w1",
                up_proj_name="w3",
                down_proj_name="w2",
                gate_up_fused_name=None,
                extra_names={},
            ),
            fused_schema=None,
            num_experts=config.n_routed_experts,
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
            fused_gate_up_as_w13=False,
            include_bias=False,
            fused_load_style="per_expert",
            transpose_local_tensor_non_bias=False,
        )
        seen, loaded = set(), set()
        skipped = Counter()
        pending = {}
        for raw_name, tensor in weights:
            name = self._checkpoint_name(raw_name)
            if name in seen:
                raise ValueError(f"Duplicate V4.1 checkpoint tensor: {raw_name}")
            seen.add(name)
            reason = self._skip_checkpoint_weight(name)
            if reason is not None:
                skipped[reason] += 1
                continue
            expert = re.fullmatch(
                r"layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(weight|scale)", name
            )
            if expert:
                layer_id, expert_id = int(expert[1]), int(expert[2])
                if (
                    layer_id >= config.num_hidden_layers
                    or expert_id >= config.n_routed_experts
                ):
                    raise ValueError(f"Unexpected V4.1 expert tensor: {raw_name}")
                n, k = (
                    (config.hidden_size, config.moe_intermediate_size)
                    if expert[3] == "w2"
                    else (config.moe_intermediate_size, config.hidden_size)
                )
                expected = (n, k // (32 if expert[4] == "scale" else 2))
                if tuple(tensor.shape) != expected:
                    raise ValueError(
                        f"{raw_name}: expected {expected}, got {tuple(tensor.shape)}"
                    )
                allowed = (
                    (torch.uint8, torch.float8_e8m0fnu)
                    if expert[4] == "scale"
                    else (torch.int8, torch.uint8, torch.float4_e2m1fn_x2)
                )
                if tensor.dtype not in allowed:
                    raise TypeError(
                        f"{raw_name}: expected packed FP4/E8M0, got {tensor.dtype}"
                    )
                local_count = config.n_routed_experts // self.mapping.moe.ep_size
                if expert_id // local_count != self.mapping.moe.ep_rank:
                    skipped["remote_expert"] += 1
                    continue
                if name not in targets:
                    raise ValueError(f"Unexpected V4.1 expert tensor: {raw_name}")
                tensor = tensor.view(torch.uint8)
                ffn = self.model.layers[layer_id].ffn
                if ffn.use_mega_moe:
                    extra = ffn.experts.intermediate_size - config.moe_intermediate_size
                    if extra:
                        padding = (
                            (0, extra // (32 if expert[4] == "scale" else 2))
                            if expert[3] == "w2"
                            else (0, 0, 0, extra)
                        )
                        tensor = F.pad(
                            tensor,
                            padding,
                            mode="constant",
                            value=127 if expert[4] == "scale" else 0,
                        )
                mapped = "model." + name.replace(".scale", ".weight_scale")
                moe_loader.load(mapped, tensor)
            elif name.endswith((".attn.wo_a.weight", ".attn.wo_a.scale")):
                weight_name = name.rsplit(".", 1)[0] + ".weight"
                if weight_name not in targets:
                    raise ValueError(f"Unexpected V4.1 checkpoint tensor: {raw_name}")
                self._load_wo_a(name, tensor, pending)
            elif is_engram_embed_checkpoint_name(name):
                raise ValueError(
                    f"{raw_name}: Engram embed tables load only via safetensors "
                    "get_slice; omit them from the generic iterator and call "
                    "bind_checkpoint_dir"
                )
            elif name in targets:
                param_name, shard_id = targets[name]
                param = params[param_name]
                module = self.get_submodule(param_name.rsplit(".", 1)[0])
                expected = tuple(param.shape)
                if isinstance(module, LinearBase):
                    n = (
                        module.output_size
                        if shard_id is None
                        else module.output_sizes[shard_id]
                    )
                    expected = (n, module.input_size)
                    if name.endswith(".scale"):
                        expected = tuple((dim + 31) // 32 for dim in expected)
                elif isinstance(module, VocabParallelEmbedding):
                    expected = (module.num_embeddings, module.embedding_dim)
                elif name.endswith(".attn_sink"):
                    expected = (config.num_attention_heads,)
                if tuple(tensor.shape) != expected:
                    raise ValueError(
                        f"{raw_name}: expected {expected}, got {tuple(tensor.shape)}"
                    )
                expects_fp8 = param.dtype == torch.float8_e4m3fn or getattr(
                    param, "loads_fp8_codes", False
                )
                if expects_fp8 and tensor.dtype != torch.float8_e4m3fn:
                    raise TypeError(
                        f"{raw_name}: expected checkpoint FP8 E4M3, got {tensor.dtype}"
                    )
                if not expects_fp8 and not name.endswith(".scale"):
                    fp32 = ".hc_" in name or name.endswith(
                        (".attn_sink", ".ffn.gate.bias", ".ffn.gate.bias_vl")
                    )
                    dtype = torch.float32 if fp32 else torch.bfloat16
                    if tensor.dtype != dtype:
                        raise TypeError(
                            f"{raw_name}: expected {dtype}, got {tensor.dtype}"
                        )
                loader = getattr(param, "weight_loader", default_weight_loader)
                if shard_id is not None:
                    loader(param, tensor, shard_id)
                else:
                    loader(param, tensor)
            else:
                raise ValueError(f"Unexpected V4.1 checkpoint tensor: {raw_name}")
            loaded.add(name)
        self._load_missing_engram_tables(loaded)
        missing = set(targets) - loaded
        for key, pair in pending.items():
            if "weight" not in pair:
                missing.add(key + ".weight")
            elif pair["weight"].dtype == torch.float8_e4m3fn:
                missing.add(key + ".scale")
        if missing:
            raise ValueError(
                f"Missing {len(missing)} V4.1 checkpoint tensors: {', '.join(sorted(missing)[:16])}"
            )
        self.checkpoint_load_report = {"loaded": len(loaded), "skipped": dict(skipped)}
        logger.info(f"V4.1 checkpoint coverage: {self.checkpoint_load_report!s}")
        self.post_load_weights()

    def _load_missing_engram_tables(self, loaded: set[str]) -> None:
        """Load Engram embed tables from safetensors get_slice only."""
        pending = [
            raw
            for raw in self._checkpoint_targets()
            if is_engram_embed_checkpoint_name(raw) and raw not in loaded
        ]
        if not pending:
            return
        checkpoint_dir = getattr(self, "_checkpoint_dir", None)
        if checkpoint_dir is None:
            return
        from safetensors import safe_open
        from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

        index_path = os.path.join(checkpoint_dir, SAFE_WEIGHTS_INDEX_NAME)
        if not os.path.isfile(index_path):
            raise FileNotFoundError(
                f"Engram slice load requires {SAFE_WEIGHTS_INDEX_NAME} in "
                f"{checkpoint_dir}"
            )
        with open(index_path) as handle:
            weight_map = json.load(handle)["weight_map"]
        targets = self._checkpoint_targets()
        for raw in pending:
            mapped = None
            for key in (raw, "model." + raw, raw.removeprefix("model.")):
                if key in weight_map:
                    mapped = key
                    break
            if mapped is None:
                raise ValueError(
                    f"Missing {raw} in {index_path} weight_map; "
                    "Engram embed tables load only via get_slice"
                )
            param_name, _ = targets[raw]
            embed = self.get_submodule(param_name.rsplit(".", 1)[0])
            shard = os.path.join(checkpoint_dir, weight_map[mapped])
            field = param_name.rsplit(".", 1)[1]
            with safe_open(shard, framework="pt", device="cpu") as handle:
                embed.load_sharded(field, handle.get_slice(mapped), 65536)
            loaded.add(raw)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def post_load_weights(self) -> None:
        """Finalize packed MegaMoE weights; generic hooks own all other modules."""
        for module in self.modules():
            if isinstance(module, DeepseekV4MegaMoEExperts):
                module.finalize_weights()

    def post_quant_warmup(self) -> None:
        for module in self.modules():
            if isinstance(module, DeepseekV4MegaMoEExperts):
                module.warmup()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def get_multimodal_encoder_specs(self) -> dict[Modality, EncoderSpec]:
        if self.vision is None:
            return {}
        return {
            Modality.IMAGE: EncoderSpec(
                fn=self.image_encoder,
                deepstack=False,
                make_warmup_items=self.vision.make_image_warmup_items,
            )
        }

    def pad_input_ids(
        self, input_ids: list[int], mm_inputs: MultimodalInputs
    ) -> list[int]:
        return pad_input_tokens(input_ids, mm_inputs)

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]], **kwargs
    ) -> None:
        params = dict(self.vision.named_parameters()) if self.vision is not None else {}
        loaded = set()

        def text_weights():
            for name, tensor in weights:
                raw = self._checkpoint_name(name)
                if self.vision is not None and (
                    raw.startswith(("vision.", "aligner."))
                    or raw
                    in (
                        "image_start",
                        "image_end",
                        "image_newline",
                    )
                ):
                    target = raw.replace(".attn.wqkv.", ".attn.qkv_proj.").replace(
                        ".attn.wo.", ".attn.proj."
                    )
                    if target in loaded:
                        raise ValueError(f"Duplicate V4.1 vision tensor: {name}")
                    param = params[target]
                    loader = getattr(param, "weight_loader", default_weight_loader)
                    loader(param, tensor)
                    loaded.add(target)
                else:
                    yield name, tensor

        if self.encoder_only:
            for _ in text_weights():
                pass
        else:
            self._load_text_weights(text_weights(), **kwargs)
        missing = params.keys() - loaded
        if missing:
            raise ValueError(f"Missing V4.1 vision tensors: {sorted(missing)}")


EntryClass = DeepseekV41ForCausalLM
