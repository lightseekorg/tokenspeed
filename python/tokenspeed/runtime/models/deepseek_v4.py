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

"""Inference-only DeepSeek V4 model skeleton.

This module intentionally registers only architecture pieces that map to the
DeepSeek V4 Flash checkpoint. The sparse MLA forward path still fails loudly
until the HCA/CSA cache kernels are wired into TokenSpeed.
"""

from __future__ import annotations

import glob
import importlib
import os
import re
import site
import sys
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

try:
    # Optional dependency; the module-level wrapper imports the external
    # `deep_gemm` package unguarded, which is not installed in baseline V4
    # builds. Callsites guard usage with `deep_gemm is not None`.
    from tokenspeed_kernel.thirdparty import deep_gemm
except ImportError:
    deep_gemm = None  # type: ignore[assignment]

from tokenspeed_kernel.ops.routing.cuda import dsv3_router_gemm
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.thirdparty.cuda import (
    hash_softplus_sqrt_topk_flash,
    softplus_sqrt_topk_flash,
)
from tokenspeed_kernel.thirdparty.trtllm import (
    per_token_group_quant_8bit as trtllm_fp8_quantize_1x128,
)
from torch import nn
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.comm_ops import (
    all_reduce,
    token_all_gather,
    token_reduce_scatter,
)
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.attention.deepseek_v4_ops import (
    DEEPSEEK_V4_INDEXER_DIM,
    DeepseekV4AttentionOpUnavailable,
    deepseek_v4_csa_compress_kv_cache_insert,
    deepseek_v4_csa_indexer_cache_insert,
    deepseek_v4_hca_compress_kv_cache_insert,
    deepseek_v4_inv_rope_reference,
    deepseek_v4_prepare_indexer_q_mxfp4,
    deepseek_v4_prepare_indexer_q_reference,
    deepseek_v4_profile_scope,
    dequantize_deepseek_v4_fp8_ds_mla_cache,
    fused_qnorm_rope_kv_insert,
    read_deepseek_v4_indexer_fp8_cache,
    read_deepseek_v4_indexer_mxfp4_cache,
    save_deepseek_v4_compressor_state,
)
from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4 import (
    _group_slot_mapping_from_raw,
)
from tokenspeed.runtime.layers.layernorm import FusedRMSNorm, RMSNorm
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.moe.checkpoint import (
    ExpertCheckpointSchema,
    build_moe_checkpoint_loader,
)
from tokenspeed.runtime.layers.moe.layer import MoELayer
from tokenspeed.runtime.layers.moe.topk import (
    BypassedTopKOutput,
    StandardTopKOutput,
    TopK,
)
from tokenspeed.runtime.layers.moe.utils import RoutingMethodType
from tokenspeed.runtime.layers.quantization import Mxfp4Config
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.rotary_embedding import get_rope
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.base import BaseCausalLM
from tokenspeed.runtime.utils import (
    add_prefix,
    get_colorful_logger,
    set_weight_attrs,
)
from tokenspeed.runtime.utils.env import global_server_args_dict, pdl_enabled

_platform = current_platform()


def is_blackwell() -> bool:
    return _platform.is_blackwell


def is_sm90_supported(device: object | None = None) -> bool:
    del device
    return _platform.is_hopper or _platform.is_blackwell


logger = get_colorful_logger(__name__)


def _dequant_fp8_weight(layer: nn.Module, shape: tuple[int, ...]) -> torch.Tensor:
    weight = layer.weight.view(*shape)
    scale = getattr(layer, "weight_scale_inv", None)
    if scale is None or weight.dtype != torch.float8_e4m3fn:
        return weight.float()

    cache = getattr(layer, "_deepseek_v4_dequant_cache", None)
    if cache is not None:
        cached_shape, cached_weight = cache
        if cached_shape == tuple(shape):
            return cached_weight

    block_n, block_k = getattr(layer.quant_config, "weight_block_size", (128, 128))
    if len(shape) == 2:
        out_dim, in_dim = shape
        scale = scale.view(
            (out_dim + block_n - 1) // block_n,
            (in_dim + block_k - 1) // block_k,
        )
        expanded_scale = (
            scale.float()
            .repeat_interleave(block_n, dim=0)
            .repeat_interleave(block_k, dim=1)
        )
        out = weight.float() * expanded_scale[:out_dim, :in_dim]
        layer._deepseek_v4_dequant_cache = (tuple(shape), out)
        return out

    groups, out_dim, in_dim = shape
    scale = scale.view(
        groups,
        (out_dim + block_n - 1) // block_n,
        (in_dim + block_k - 1) // block_k,
    )
    expanded_scale = (
        scale.float()
        .repeat_interleave(block_n, dim=1)
        .repeat_interleave(block_k, dim=2)
    )
    out = weight.float() * expanded_scale[:, :out_dim, :in_dim]
    layer._deepseek_v4_dequant_cache = (tuple(shape), out)
    return out


def _fp8_act_quant_dequant(x: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """Simulate DeepSeek V4's block FP8 activation quantization."""

    if x.shape[-1] % block_size != 0:
        raise ValueError(
            f"DeepSeek V4 FP8 activation quantization expects K divisible by "
            f"{block_size}, got {x.shape[-1]}"
        )
    orig_shape = x.shape

    if x.is_cuda and block_size == DEEPSEEK_V4_FP8_BLOCK_SIZE:
        x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
        try:
            quantized, scale = trtllm_fp8_quantize_1x128(
                x_2d,
                block_size,
                use_ue8m0=True,
            )
            scale = scale.float().transpose(0, 1).contiguous()
            return (
                (
                    quantized.float().unflatten(-1, (-1, block_size))
                    * scale.unsqueeze(-1)
                )
                .flatten(-2)
                .reshape(orig_shape)
            )
        except RuntimeError:
            pass

    x_blocks = x.float().reshape(-1, orig_shape[-1]).unflatten(-1, (-1, block_size))
    amax = x_blocks.abs().amax(dim=-1).clamp_min(1.0e-4)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    scale = scale.to(torch.float8_e8m0fnu).float()
    quantized = (
        (x_blocks / scale.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    )
    return (quantized.float() * scale.unsqueeze(-1)).flatten(-2).reshape(orig_shape)


def _fp8_linear(
    layer: nn.Module,
    x: torch.Tensor,
    shape: tuple[int, ...],
    *,
    quantize_act: bool = True,
) -> torch.Tensor:
    weight = _dequant_fp8_weight(layer, shape)
    x_eff = (
        _fp8_act_quant_dequant(x, DEEPSEEK_V4_FP8_BLOCK_SIZE)
        if quantize_act and layer.weight.dtype == torch.float8_e4m3fn
        else x.float()
    )
    return torch.matmul(x_eff, weight.transpose(-2, -1)).to(x.dtype)


def _deepseek_v4_router_gemm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    if (
        hidden_states.dim() == 2
        and hidden_states.shape[0] > 0
        and hidden_states.is_cuda
        and hidden_states.dtype == torch.bfloat16
        and weight.dtype in (torch.bfloat16, torch.float32)
        and (is_sm90_supported(hidden_states.device) or is_blackwell())
    ):
        return dsv3_router_gemm(
            hidden_states,
            weight,
            out_dtype=torch.float32,
            enable_pdl=pdl_enabled(),
        )

    x = (
        hidden_states
        if hidden_states.dtype == weight.dtype
        else hidden_states.to(weight.dtype)
    )
    return F.linear(x, weight, None).to(torch.float32)


def _deepseek_v4_bf16_linear_fp32(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor | None:
    if (
        hidden_states.dim() == 2
        and hidden_states.shape[0] > 0
        and hidden_states.is_cuda
        and hidden_states.dtype == torch.bfloat16
        and weight.is_cuda
        and weight.dtype == torch.bfloat16
        and weight.dim() == 2
        and hidden_states.shape[1] == weight.shape[1]
        and (is_sm90_supported(hidden_states.device) or is_blackwell())
    ):
        return dsv3_router_gemm(
            hidden_states,
            weight,
            out_dtype=torch.float32,
            enable_pdl=False,
        )
    return None


def _deepseek_v4_fused_select_experts(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    *,
    correction_bias: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    global _DEEPSEEK_V4_FUSED_ROUTER_AVAILABLE

    if (
        not _DEEPSEEK_V4_FUSED_ROUTER_AVAILABLE
        or not router_logits.is_cuda
        or router_logits.dim() != 2
        or router_logits.shape[1] != 256
        or top_k <= 0
        or top_k > 32
        or router_logits.dtype not in (torch.float32, torch.float16, torch.bfloat16)
    ):
        return None

    topk_weights = torch.empty(
        router_logits.shape[0],
        top_k,
        dtype=torch.float32,
        device=router_logits.device,
    )
    topk_ids = torch.empty(
        router_logits.shape[0],
        top_k,
        dtype=torch.int32,
        device=router_logits.device,
    )

    try:
        if hash_indices_table is not None:
            if input_ids is None:
                raise ValueError("hash-routed DeepSeek V4 MoE requires input_ids")
            hash_softplus_sqrt_topk_flash(
                router_logits.contiguous(),
                input_ids.reshape(-1).to(device=router_logits.device).contiguous(),
                hash_indices_table.to(
                    device=router_logits.device, dtype=torch.int32
                ).contiguous(),
                topk_ids,
                topk_weights,
                1.0,
                renormalize,
            )
        elif correction_bias is not None:
            softplus_sqrt_topk_flash(
                router_logits.contiguous(),
                correction_bias.to(
                    device=router_logits.device, dtype=torch.float32
                ).contiguous(),
                topk_ids,
                topk_weights,
                1.0,
                renormalize,
            )
        else:
            return None
    except (AttributeError, RuntimeError):
        _DEEPSEEK_V4_FUSED_ROUTER_AVAILABLE = False
        return None

    return topk_weights, topk_ids


def _deepseek_v4_reorder_c4_ape_2604(ape: torch.Tensor) -> torch.Tensor:
    """Convert C4 overlap APE from checkpoint layout to runtime window layout."""

    if ape.dim() != 2 or ape.shape[0] != 4 or ape.shape[1] % 2 != 0:
        raise ValueError(f"expected C4 APE [4, even], got {tuple(ape.shape)}")
    older, newer = ape.chunk(2, dim=-1)
    return torch.cat([older, newer], dim=0).reshape_as(ape)


def _sinkhorn(mixes: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
    if iters < 1:
        raise ValueError(f"sinkhorn iterations must be >= 1, got {iters}")
    mixes = torch.softmax(mixes, dim=-1) + eps
    mixes = mixes / (mixes.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        mixes = mixes / (mixes.sum(dim=-1, keepdim=True) + eps)
        mixes = mixes / (mixes.sum(dim=-2, keepdim=True) + eps)
    return mixes


_DEEPSEEK_V4_FAST_MHC_UNAVAILABLE = False


def _mhc_pre_reference(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure PyTorch hidden-compression pre step.

    Shapes follow the DeepSeek V4 hidden-compression contract:
    residual [T, M, H], layer input [T, H], post [T, M, 1], comb [T, M, M].
    """

    if residual.dim() != 3:
        raise ValueError(f"expected residual [T, M, H], got {tuple(residual.shape)}")
    num_tokens, hc_mult, hidden_size = residual.shape
    x = residual.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_eps)
    mixes = F.linear(x, fn.float()) * rsqrt
    expected = (2 + hc_mult) * hc_mult
    if mixes.shape[-1] != expected:
        raise ValueError(f"expected {expected} HC mixes, got {mixes.shape[-1]}")

    pre_raw, post_raw, comb_raw = torch.split(
        mixes, [hc_mult, hc_mult, hc_mult * hc_mult], dim=-1
    )
    pre_base, post_base, comb_base = torch.split(
        hc_base.float(), [hc_mult, hc_mult, hc_mult * hc_mult], dim=-1
    )
    pre = torch.sigmoid(pre_raw * hc_scale[0].float() + pre_base) + hc_eps
    post = (torch.sigmoid(post_raw * hc_scale[1].float() + post_base) * 2.0).unsqueeze(
        -1
    )
    comb_logits = comb_raw.reshape(num_tokens, hc_mult, hc_mult)
    comb_base = comb_base.reshape(1, hc_mult, hc_mult)
    comb = _sinkhorn(
        comb_logits * hc_scale[2].float() + comb_base,
        iters=sinkhorn_iters,
        eps=hc_eps,
    )
    layer_input = torch.sum(pre.unsqueeze(-1) * residual.float(), dim=1)
    return layer_input.to(residual.dtype), post, comb


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global _DEEPSEEK_V4_FAST_MHC_UNAVAILABLE

    if (
        not _DEEPSEEK_V4_FAST_MHC_UNAVAILABLE
        and not global_server_args_dict.get("disable_deepseek_v4_fast_mhc", False)
        and residual.is_cuda
    ):
        try:
            from tokenspeed.runtime.layers.deepseek_v4_mhc import (
                mhc_pre as fast_mhc_pre,
            )

            return fast_mhc_pre(
                residual,
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_eps,
                sinkhorn_iters,
            )
        except Exception as exc:
            _DEEPSEEK_V4_FAST_MHC_UNAVAILABLE = True
            logger.warning(
                "DeepSeek V4 fast mHC pre is unavailable; falling back to "
                f"PyTorch reference. reason={type(exc).__name__}: {exc}"
            )

    return _mhc_pre_reference(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_eps,
        sinkhorn_iters,
    )


def _mhc_post_reference(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    if post.dim() == 2:
        post = post.unsqueeze(-1)
    mixed_residual = torch.einsum("tnm,tnh->tmh", comb.float(), residual.float())
    block_update = post.float() * hidden_states.float().unsqueeze(1)
    return (mixed_residual + block_update).to(hidden_states.dtype)


def mhc_post(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    global _DEEPSEEK_V4_FAST_MHC_UNAVAILABLE

    if (
        not _DEEPSEEK_V4_FAST_MHC_UNAVAILABLE
        and not global_server_args_dict.get("disable_deepseek_v4_fast_mhc", False)
        and hidden_states.is_cuda
    ):
        try:
            from tokenspeed.runtime.layers.deepseek_v4_mhc import (
                mhc_post as fast_mhc_post,
            )

            return fast_mhc_post(hidden_states, residual, post, comb)
        except Exception as exc:
            _DEEPSEEK_V4_FAST_MHC_UNAVAILABLE = True
            logger.warning(
                "DeepSeek V4 fast mHC post is unavailable; falling back to "
                f"PyTorch reference. reason={type(exc).__name__}: {exc}"
            )

    return _mhc_post_reference(hidden_states, residual, post, comb)


def hc_head(
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    shape, dtype = hidden_states.size(), hidden_states.dtype
    x = hidden_states.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_norm_eps)
    mixes = F.linear(x, hc_fn.float()) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale.float() + hc_base.float()) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
    return y.to(dtype)


def deepseek_v4_select_experts(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    *,
    correction_bias: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DeepSeek V4 MoE routing.

    DeepSeek V4 uses sqrt(softplus(logits)) as expert scores. Correction bias
    only affects expert selection; the gathered expert weights come from the
    unbiased scores. Hash-routed layers use checkpoint-provided expert ids but
    still gather weights from the gate scores.
    """

    fused_topk = _deepseek_v4_fused_select_experts(
        router_logits,
        top_k,
        renormalize,
        correction_bias=correction_bias,
        hash_indices_table=hash_indices_table,
        input_ids=input_ids,
    )
    if fused_topk is not None:
        topk_weights, topk_ids = fused_topk
        scores = torch.sqrt(F.softplus(router_logits.float()))
        return topk_weights, topk_ids, scores

    scores = torch.sqrt(F.softplus(router_logits.float()))
    if hash_indices_table is not None:
        if input_ids is None:
            raise ValueError("hash-routed DeepSeek V4 MoE requires input_ids")
        topk_ids = hash_indices_table[input_ids.reshape(-1)].to(
            device=scores.device,
            dtype=torch.long,
        )
    else:
        scores_for_choice = scores
        if correction_bias is not None:
            scores_for_choice = scores_for_choice + correction_bias.to(
                device=scores.device,
                dtype=scores.dtype,
            ).unsqueeze(0)
        topk_ids = torch.topk(scores_for_choice, k=top_k, dim=-1, sorted=True)[1]

    topk_weights = scores.gather(1, topk_ids)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(topk_weights.dtype).tiny
        )
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32), scores


def pack_topk_as_router_logits(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Encode preselected top-k weights for BYPASSED TokenSpeed MoE backends.

    MXFP4 backends currently build routing data from logits internally. Packing
    the normalized top-k weights as log-probabilities with very negative values
    elsewhere makes their TopK -> Softmax/Renormalize route recover the same
    selected ids and weights without changing the shared backend.
    """

    router_logits = torch.full(
        (topk_ids.shape[0], num_experts),
        -1e20,
        dtype=torch.float32,
        device=topk_weights.device,
    )
    safe_weights = topk_weights.clamp_min(torch.finfo(torch.float32).tiny)
    router_logits.scatter_(1, topk_ids.long(), safe_weights.log())
    return router_logits


def _deepseek_v4_indexer_topk_from_cache_batched(
    *,
    cache_reader,
    cache_2d: torch.Tensor,
    positions: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    cache_block_size: int,
    index_q: torch.Tensor,
    weights: torch.Tensor,
    compress_ratio: int,
    topk_tokens: int,
    preserve_topk_order: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batch the decode indexer cache read while preserving per-token top-k."""

    num_tokens = positions.numel()
    if out is None:
        topk = torch.empty(
            (num_tokens, topk_tokens),
            device=positions.device,
            dtype=torch.int32,
        )
    else:
        topk = out[:num_tokens]
    topk.fill_(-1)
    if num_tokens == 0:
        return topk

    compressed_lens = torch.div(
        positions.to(torch.int64) + 1,
        compress_ratio,
        rounding_mode="floor",
    ).clamp_min(0)
    max_len = int(compressed_lens.max().item())
    if max_len <= 0:
        return topk

    offsets = torch.arange(max_len, device=positions.device, dtype=torch.int64)
    local = offsets[None, :].expand(num_tokens, -1)
    valid = local < compressed_lens[:, None]
    req_idx = token_to_req_indices[:num_tokens].to(torch.int64)
    pages = torch.div(local, cache_block_size, rounding_mode="floor")
    page_offsets = local % cache_block_size
    page_ids = block_table[req_idx[:, None], pages.long()].to(torch.int64)
    slots = page_ids * cache_block_size + page_offsets

    k_flat = cache_reader(
        cache_2d,
        slots[valid],
        block_size=cache_block_size,
    )
    k_padded = torch.zeros(
        (num_tokens, max_len, index_q.shape[-1]),
        device=positions.device,
        dtype=k_flat.dtype,
    )
    k_padded[valid] = k_flat

    scores = torch.bmm(index_q.float(), k_padded.float().transpose(1, 2)).relu_()
    logits = torch.bmm(weights.float().unsqueeze(1), scores).squeeze(1)
    logits = logits.masked_fill(~valid, -float("inf"))

    if preserve_topk_order:
        for raw_len in torch.unique(compressed_lens).tolist():
            num_compressed = int(raw_len)
            selected = min(num_compressed, topk_tokens)
            if selected <= 0:
                continue
            row_mask = compressed_lens == num_compressed
            token_topk = torch.topk(
                logits[row_mask, :num_compressed],
                k=selected,
                dim=-1,
                sorted=False,
            ).indices
            topk[row_mask, :selected] = token_topk.to(torch.int32)
        return topk

    if logits.is_cuda and topk_tokens in (512, 1024, 2048):
        from tokenspeed_kernel.thirdparty.trtllm import fast_topk_v2

        fast_topk_v2(
            logits.contiguous(),
            compressed_lens.to(torch.int32).contiguous(),
            topk,
            topk_tokens,
        )
        return topk

    selected = min(max_len, topk_tokens)
    values, indices = torch.topk(logits, k=selected, dim=-1, sorted=False)
    indices = torch.where(
        torch.isfinite(values),
        indices,
        torch.full_like(indices, -1),
    ).to(torch.int32)
    topk[:, :selected] = indices
    return topk


def _deepseek_v4_indexer_prefill_topk_chunks(
    positions: torch.Tensor,
    compress_ratio: int,
    max_logits_bytes: int | None = None,
) -> list[tuple[int, int]]:
    num_tokens = positions.numel()
    if num_tokens == 0:
        return []
    if max_logits_bytes is None:
        max_logits_mb = global_server_args_dict.get(
            "deepseek_v4_indexer_prefill_max_logits_mb",
            _DEEPSEEK_V4_INDEXER_PREFILL_MAX_LOGITS_MB,
        )
        max_logits_bytes = max_logits_mb * 1024 * 1024
    max_logits_elems = max(1, int(max_logits_bytes) // 4)
    compressed_lens = torch.div(
        positions.to(torch.int64) + 1,
        compress_ratio,
        rounding_mode="floor",
    ).clamp_min(0)
    lengths = compressed_lens.detach().cpu().tolist()

    chunks: list[tuple[int, int]] = []
    end = 0
    while end < num_tokens:
        start = end
        chunk_tokens = 0
        max_len = 0
        while end < num_tokens:
            candidate_tokens = chunk_tokens + 1
            candidate_max_len = max(max_len, max(0, int(lengths[end])))
            if (
                chunk_tokens > 0
                and candidate_tokens * candidate_max_len > max_logits_elems
            ):
                break
            chunk_tokens = candidate_tokens
            max_len = candidate_max_len
            end += 1
        if end == start:
            end += 1
        chunks.append((start, end))
    return chunks


def _deepseek_v4_deepgemm_fp4_indexer_available(index_q: torch.Tensor) -> bool:
    return (
        deep_gemm is not None
        and index_q.is_cuda
        and index_q.dim() >= 3
        and index_q.shape[-2] in (32, 64)
        and index_q.shape[-1] == DEEPSEEK_V4_INDEXER_DIM // 2
        and getattr(deep_gemm, "fp8_fp4_mqa_logits", None) is not None
        and getattr(deep_gemm, "fp8_fp4_paged_mqa_logits", None) is not None
        and getattr(deep_gemm, "get_paged_mqa_logits_metadata", None) is not None
    )


def _deepseek_v4_indexer_mxfp4_cache_view(
    cache_2d: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    row_bytes = (
        DEEPSEEK_V4_INDEXER_DIM // 2
        + DEEPSEEK_V4_INDEXER_DIM // DEEPSEEK_V4_MXFP4_BLOCK_SIZE
    )
    return torch.as_strided(
        cache_2d,
        (cache_2d.shape[0], block_size, 1, row_bytes),
        (cache_2d.stride(0), row_bytes, row_bytes, 1),
    )


def _deepseek_v4_indexer_decode_max_len(
    block_table: torch.Tensor,
    cache_block_size: int,
    compress_ratio: int,
) -> int:
    context_len = global_server_args_dict.get("max_model_len")
    if isinstance(context_len, int) and context_len > 0:
        return max(1, (context_len + compress_ratio - 1) // compress_ratio)
    return max(
        1,
        (block_table.shape[1] * cache_block_size + compress_ratio - 1)
        // compress_ratio,
    )


def _deepseek_v4_gather_indexer_mxfp4_cache(
    cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    value_bytes = DEEPSEEK_V4_INDEXER_DIM // 2
    scale_bytes = DEEPSEEK_V4_INDEXER_DIM // DEEPSEEK_V4_MXFP4_BLOCK_SIZE
    if slot_mapping.numel() == 0:
        return (
            torch.empty(
                (0, value_bytes),
                dtype=torch.int8,
                device=cache_2d.device,
            ),
            torch.empty(0, dtype=torch.int32, device=cache_2d.device),
        )

    flat_cache = cache_2d.reshape(-1)
    slots = slot_mapping.to(torch.int64)
    pages = torch.div(slots, block_size, rounding_mode="floor")
    pos = slots % block_size
    page_base = pages * cache_2d.stride(0)
    value_base = page_base + pos * value_bytes
    scale_base = page_base + block_size * value_bytes + pos * scale_bytes
    value_offsets = (
        value_base[:, None]
        + torch.arange(
            value_bytes,
            device=cache_2d.device,
            dtype=torch.int64,
        )[None, :]
    )
    scale_offsets = (
        scale_base[:, None]
        + torch.arange(
            scale_bytes,
            device=cache_2d.device,
            dtype=torch.int64,
        )[None, :]
    )
    values = flat_cache[value_offsets].contiguous().view(torch.int8)
    scales = flat_cache[scale_offsets].contiguous().view(torch.int32).squeeze(-1)
    return values, scales


def _deepseek_v4_indexer_topk_from_logits(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    topk_tokens: int,
    *,
    preserve_topk_order: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    num_tokens = lengths.numel()
    if out is None:
        topk = torch.empty(
            (num_tokens, topk_tokens),
            device=logits.device,
            dtype=torch.int32,
        )
    else:
        topk = out[:num_tokens]
    topk.fill_(-1)
    if num_tokens == 0:
        return topk
    max_len = logits.shape[1] if logits.dim() == 2 else 0
    if max_len <= 0:
        return topk

    if not preserve_topk_order and logits.is_cuda and topk_tokens in (512, 1024, 2048):
        from tokenspeed_kernel.thirdparty.trtllm import fast_topk_v2

        fast_topk_v2(
            logits.contiguous(),
            lengths.to(torch.int32).contiguous(),
            topk,
            topk_tokens,
        )
        return topk

    offsets = torch.arange(max_len, device=logits.device, dtype=torch.int64)
    masked_logits = logits.masked_fill(
        offsets[None, :] >= lengths[:, None], -float("inf")
    )

    if preserve_topk_order:
        for raw_len in torch.unique(lengths).tolist():
            num_compressed = int(raw_len)
            selected = min(num_compressed, topk_tokens)
            if selected <= 0:
                continue
            row_mask = lengths == num_compressed
            token_topk = torch.topk(
                masked_logits[row_mask, :num_compressed],
                k=selected,
                dim=-1,
                sorted=False,
            ).indices
            topk[row_mask, :selected] = token_topk.to(torch.int32)
        return topk

    selected = min(max_len, topk_tokens)
    values, indices = torch.topk(masked_logits, k=selected, dim=-1, sorted=False)
    indices = torch.where(
        torch.isfinite(values),
        indices,
        torch.full_like(indices, -1),
    ).to(torch.int32)
    topk[:, :selected] = indices
    return topk


def _deepseek_v4_indexer_ascending_prefill_topk(
    positions: torch.Tensor,
    compress_ratio: int,
    topk_tokens: int,
) -> torch.Tensor:
    num_tokens = positions.numel()
    offsets = torch.arange(topk_tokens, device=positions.device, dtype=torch.int32)
    lengths = torch.div(
        positions.to(torch.int64) + 1,
        compress_ratio,
        rounding_mode="floor",
    ).clamp(min=0, max=topk_tokens)
    return torch.where(
        offsets[None, :] < lengths[:, None],
        offsets[None, :],
        torch.full(
            (num_tokens, topk_tokens),
            -1,
            device=positions.device,
            dtype=torch.int32,
        ),
    )


def _deepseek_v4_indexer_topk_from_cache_deepgemm_prefill(
    *,
    cache_2d: torch.Tensor,
    positions: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    cache_block_size: int,
    index_q: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    compress_ratio: int,
    topk_tokens: int,
    preserve_topk_order: bool,
) -> torch.Tensor | None:
    q_values, q_scales = index_q
    if not _deepseek_v4_deepgemm_fp4_indexer_available(q_values):
        return None

    num_tokens = positions.numel()
    if num_tokens == 0:
        return torch.empty(
            (0, topk_tokens),
            device=positions.device,
            dtype=torch.int32,
        )
    compressed_lens = torch.div(
        positions.to(torch.int64) + 1,
        compress_ratio,
        rounding_mode="floor",
    ).clamp_min(0)
    max_len = int(compressed_lens.max().item())
    if max_len <= 0:
        return torch.full(
            (num_tokens, topk_tokens),
            -1,
            device=positions.device,
            dtype=torch.int32,
        )

    offsets = torch.arange(max_len, device=positions.device, dtype=torch.int64)
    local = offsets[None, :].expand(num_tokens, -1)
    valid = local < compressed_lens[:, None]
    req_idx = token_to_req_indices[:num_tokens].to(torch.int64)
    pages = torch.div(local, cache_block_size, rounding_mode="floor")
    page_offsets = local % cache_block_size
    page_ids = block_table[req_idx[:, None], pages.long()].to(torch.int64)
    slots = page_ids * cache_block_size + page_offsets

    with deepseek_v4_profile_scope("indexer_topk_prefill_gather_mxfp4"):
        k_values, k_scales = _deepseek_v4_gather_indexer_mxfp4_cache(
            cache_2d,
            slots[valid],
            cache_block_size,
        )
    row_lens = valid.sum(dim=1, dtype=torch.int32)
    cu_end = torch.cumsum(row_lens, dim=0, dtype=torch.int32)
    cu_start = torch.empty_like(cu_end)
    cu_start[0] = 0
    cu_start[1:] = cu_end[:-1]

    try:
        with deepseek_v4_profile_scope("indexer_topk_prefill_deepgemm_logits"):
            logits = deep_gemm.fp8_fp4_mqa_logits(
                q=(q_values.contiguous().view(torch.int8), q_scales.contiguous()),
                kv=(k_values.contiguous(), k_scales.contiguous()),
                weights=weights.contiguous(),
                cu_seq_len_k_start=cu_start,
                cu_seq_len_k_end=cu_end,
                clean_logits=False,
                max_seqlen_k=max_len,
                logits_dtype=torch.float32,
            )
    except RuntimeError:
        return None

    with deepseek_v4_profile_scope("indexer_topk_prefill_select"):
        return _deepseek_v4_indexer_topk_from_logits(
            logits,
            row_lens,
            topk_tokens,
            preserve_topk_order=preserve_topk_order,
        )


def _deepseek_v4_indexer_topk_from_cache_deepgemm_decode(
    *,
    cache_2d: torch.Tensor,
    positions: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    cache_block_size: int,
    index_q: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    compress_ratio: int,
    topk_tokens: int,
    metadata: Any | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | None:
    q_values, q_scales = index_q
    if not _deepseek_v4_deepgemm_fp4_indexer_available(q_values):
        return None

    num_tokens = positions.numel()
    if num_tokens == 0:
        if out is not None:
            return out[:0]
        return torch.empty((0, topk_tokens), device=positions.device, dtype=torch.int32)
    compressed_lens = torch.div(
        positions.to(torch.int64) + 1,
        compress_ratio,
        rounding_mode="floor",
    ).clamp_min(0)
    if positions.is_cuda and torch.cuda.is_current_stream_capturing():
        max_len = _deepseek_v4_indexer_decode_max_len(
            block_table,
            cache_block_size,
            compress_ratio,
        )
    else:
        max_len = int(compressed_lens.max().item())
        if max_len <= 0:
            topk = (
                torch.empty(
                    (num_tokens, topk_tokens),
                    device=positions.device,
                    dtype=torch.int32,
                )
                if out is None
                else out[:num_tokens]
            )
            topk.fill_(-1)
            return topk

    max_blocks = max(1, (max_len + cache_block_size - 1) // cache_block_size)
    req_idx = token_to_req_indices[:num_tokens].to(torch.int64)
    block_tables = block_table[req_idx, :max_blocks].contiguous()
    context_lens = compressed_lens.to(torch.int32).view(num_tokens, 1).contiguous()
    kv_cache = _deepseek_v4_indexer_mxfp4_cache_view(cache_2d, cache_block_size)
    schedule_key = (compress_ratio, cache_block_size, num_tokens)
    schedule_cache = getattr(metadata, "decode_indexer_schedule_metadata", None)
    schedule_metadata = (
        schedule_cache.get(schedule_key) if schedule_cache is not None else None
    )
    if schedule_metadata is None:
        schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
            context_lens,
            cache_block_size,
            deep_gemm.get_num_sms(),
        )
        if schedule_cache is not None:
            schedule_cache[schedule_key] = schedule_metadata

    try:
        logits = deep_gemm.fp8_fp4_paged_mqa_logits(
            q=(
                q_values.contiguous().view(torch.int8).unsqueeze(1),
                q_scales.contiguous().unsqueeze(1),
            ),
            kv_cache=kv_cache,
            weights=weights.contiguous(),
            context_lens=context_lens,
            block_table=block_tables,
            schedule_meta=schedule_metadata,
            max_context_len=max_len,
            clean_logits=False,
            logits_dtype=torch.float32,
        )
    except RuntimeError:
        return None

    return _deepseek_v4_indexer_topk_from_logits(
        logits,
        compressed_lens.to(torch.int32),
        topk_tokens,
        out=out,
    )


DEEPSEEK_V4_COMPRESSED_CACHE_ALIGNMENT = 576
DEEPSEEK_V4_FP8_BLOCK_SIZE = 128
_DEEPSEEK_V4_INDEXER_PREFILL_MAX_LOGITS_MB = 512
_DEEPSEEK_V4_FUSED_ROUTER_AVAILABLE = True


def _deepseek_v4_maybe_execute_in_parallel(
    fn0: Callable[[], Any],
    fn1: Callable[[], Any],
    event0: torch.cuda.Event | None,
    event1: torch.cuda.Event | None,
    aux_stream: torch.cuda.Stream | None,
) -> tuple[Any, Any]:
    if aux_stream is None or event0 is None or event1 is None:
        return fn0(), fn1()

    event0.record()
    result0 = fn0()
    with torch.cuda.stream(aux_stream):
        event0.wait()
        result1 = fn1()
        event1.record()
    event1.wait()
    return result0, result1


_DEEPSEEK_V4_MEGA_DEEP_GEMM = None
_DEEPSEEK_V4_MEGA_DEEP_GEMM_CHECKED = False
_DEEPSEEK_V4_MEGA_DEEP_GEMM_REQUIRED = (
    "fp8_fp4_mega_moe",
    "get_symm_buffer_for_mega_moe",
    "transform_weights_for_mega_moe",
    "transform_sf_into_required_layout",
)


def _deepseek_v4_prepare_deep_gemm_jit_env() -> None:
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not cuda_home and os.path.exists("/usr/local/cuda/include/cuda_runtime.h"):
        cuda_home = "/usr/local/cuda"
        os.environ["CUDA_HOME"] = cuda_home
    if not cuda_home:
        return

    include_dir = os.path.join(cuda_home, "include")
    if os.path.exists(os.path.join(include_dir, "cuda_runtime.h")):
        cpath = os.environ.get("CPATH", "")
        paths = [path for path in cpath.split(os.pathsep) if path]
        if include_dir not in paths:
            os.environ["CPATH"] = os.pathsep.join([include_dir] + paths)

    path = os.environ.get("PATH", "")
    path_entries = [entry for entry in path.split(os.pathsep) if entry]
    ptxas_dirs = []
    try:
        from torch.utils.cpp_extension import CUDA_HOME as torch_cuda_home
    except Exception:
        torch_cuda_home = None

    candidates = []
    site_paths = []
    try:
        site_paths.extend(site.getsitepackages())
    except Exception:
        pass
    site_paths.extend(sys.path)
    for base in site_paths:
        candidates.extend(
            sorted(glob.glob(os.path.join(base, "nvidia", "cu*", "bin")), reverse=True)
        )
    if torch_cuda_home:
        candidates.append(os.path.join(torch_cuda_home, "bin"))
    candidates.extend(
        [
            os.path.join(os.path.dirname(torch.__file__), "bin"),
            os.path.join(
                os.path.dirname(torch.__file__),
                "..",
                "triton",
                "backends",
                "nvidia",
                "bin",
            ),
        ]
    )
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if (
            os.path.exists(os.path.join(candidate, "ptxas"))
            and candidate not in ptxas_dirs
        ):
            ptxas_dirs.append(candidate)
    for candidate in reversed(ptxas_dirs):
        if candidate not in path_entries:
            path_entries.insert(0, candidate)
    if path_entries:
        os.environ["PATH"] = os.pathsep.join(path_entries)


def _deepseek_v4_get_mega_deep_gemm():
    global _DEEPSEEK_V4_MEGA_DEEP_GEMM
    global _DEEPSEEK_V4_MEGA_DEEP_GEMM_CHECKED

    if _DEEPSEEK_V4_MEGA_DEEP_GEMM_CHECKED:
        return _DEEPSEEK_V4_MEGA_DEEP_GEMM

    _DEEPSEEK_V4_MEGA_DEEP_GEMM_CHECKED = True
    _deepseek_v4_prepare_deep_gemm_jit_env()
    candidates = []
    try:
        candidates.append(importlib.import_module("deep_gemm"))
    except Exception:
        pass
    candidates.append(deep_gemm)

    for module in candidates:
        if all(hasattr(module, name) for name in _DEEPSEEK_V4_MEGA_DEEP_GEMM_REQUIRED):
            _DEEPSEEK_V4_MEGA_DEEP_GEMM = module
            return module

    return None


def _deepseek_v4_mega_moe_max_num_tokens() -> int:
    override = int(
        global_server_args_dict.get("deepseek_v4_mega_moe_max_num_tokens", 0) or 0
    )
    if override > 0:
        return override

    candidates = [
        global_server_args_dict.get("chunked_prefill_size", 0),
        global_server_args_dict.get("prefill_graph_max_tokens", 0),
        global_server_args_dict.get("cuda_graph_max_bs", 0),
        global_server_args_dict.get("cuda_graph_max_tokens", 0),
        global_server_args_dict.get("max_running_requests", 0),
    ]
    return max([int(value or 0) for value in candidates] + [1])


class _DeepseekV4TopKBuffer:
    def __init__(self, topk_tokens: int) -> None:
        self.topk_tokens = topk_tokens
        self.buffer: torch.Tensor | None = None

    def get(self, num_tokens: int, device: torch.device) -> torch.Tensor:
        rows = max(num_tokens, _deepseek_v4_mega_moe_max_num_tokens())
        needs_alloc = (
            self.buffer is None
            or self.buffer.device != device
            or self.buffer.shape[0] < rows
            or self.buffer.shape[1] != self.topk_tokens
        )
        if needs_alloc:
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "DeepSeek V4 top-k buffer must be allocated before CUDA graph "
                    "capture"
                )
            self.buffer = torch.empty(
                (rows, self.topk_tokens),
                dtype=torch.int32,
                device=device,
            )
        return self.buffer[:num_tokens]


@triton.jit
def _deepseek_v4_stage_mega_moe_inputs_kernel(
    hidden_states,
    x_fp8,
    x_sf,
    topk_ids,
    topk_weights,
    topk_idx_out,
    topk_weights_out,
    hidden_stride_m: tl.constexpr,
    hidden_stride_k: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_stride_k: tl.constexpr,
    x_sf_stride_m: tl.constexpr,
    x_sf_stride_k: tl.constexpr,
    topk_ids_stride_m: tl.constexpr,
    topk_ids_stride_k: tl.constexpr,
    topk_weights_stride_m: tl.constexpr,
    topk_weights_stride_k: tl.constexpr,
    topk_idx_stride_m: tl.constexpr,
    topk_idx_stride_k: tl.constexpr,
    topk_weights_out_stride_m: tl.constexpr,
    topk_weights_out_stride_k: tl.constexpr,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
) -> None:
    token_id = tl.program_id(0)
    k_block_id = tl.program_id(1)

    k_offsets = k_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < hidden_size
    hidden = tl.load(
        hidden_states + token_id * hidden_stride_m + k_offsets * hidden_stride_k,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)

    num_groups: tl.constexpr = BLOCK_K // GROUP_K
    hidden_groups = tl.reshape(tl.abs(hidden), [num_groups, GROUP_K])
    amax = tl.max(hidden_groups, axis=1)
    amax = tl.maximum(amax, 1.0e-4)

    scale = amax / 448.0
    scale_bits = scale.to(tl.uint32, bitcast=True)
    scale_exp = ((scale_bits >> 23) & 0xFF) + ((scale_bits & 0x7FFFFF) != 0).to(
        tl.uint32
    )
    scale_exp = tl.minimum(tl.maximum(scale_exp, 1), 254)
    rounded_scale = (scale_exp << 23).to(tl.float32, bitcast=True)

    hidden_groups = tl.reshape(hidden, [num_groups, GROUP_K])
    scaled = hidden_groups * (1.0 / rounded_scale)[:, None]
    scaled = tl.reshape(scaled, [BLOCK_K])
    fp8 = scaled.to(tl.float8e4nv)
    tl.store(
        x_fp8 + token_id * x_stride_m + k_offsets * x_stride_k,
        fp8,
        mask=k_mask,
    )

    scale_offsets = tl.arange(0, num_groups)
    packed_scale = tl.sum(scale_exp << (scale_offsets * 8), axis=0).to(tl.int32)
    tl.store(
        x_sf + token_id * x_sf_stride_m + k_block_id * x_sf_stride_k,
        packed_scale,
    )

    if k_block_id == 0:
        topk_offsets = tl.arange(0, BLOCK_TOPK)
        topk_mask = topk_offsets < top_k

        ids = tl.load(
            topk_ids + token_id * topk_ids_stride_m + topk_offsets * topk_ids_stride_k,
            mask=topk_mask,
            other=0,
        ).to(tl.int64)
        tl.store(
            topk_idx_out
            + token_id * topk_idx_stride_m
            + topk_offsets * topk_idx_stride_k,
            ids,
            mask=topk_mask,
        )

        weights = tl.load(
            topk_weights
            + token_id * topk_weights_stride_m
            + topk_offsets * topk_weights_stride_k,
            mask=topk_mask,
            other=0.0,
        )
        tl.store(
            topk_weights_out
            + token_id * topk_weights_out_stride_m
            + topk_offsets * topk_weights_out_stride_k,
            weights,
            mask=topk_mask,
        )


def _stage_deepseek_v4_mega_moe_inputs(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
) -> None:
    num_tokens, hidden_size = hidden_states.shape
    if num_tokens == 0:
        return
    if hidden_size % DEEPSEEK_V4_FP8_BLOCK_SIZE != 0:
        raise ValueError(
            "DeepSeek V4 MegaMoE input staging requires hidden_size to be "
            f"a multiple of {DEEPSEEK_V4_FP8_BLOCK_SIZE}."
        )
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "DeepSeek V4 MegaMoE input staging requires topk_weights and "
            "topk_ids to have the same shape."
        )

    block_k = DEEPSEEK_V4_FP8_BLOCK_SIZE
    grid = (num_tokens, triton.cdiv(hidden_size, block_k))
    block_topk = triton.next_power_of_2(topk_ids.shape[1])
    _deepseek_v4_stage_mega_moe_inputs_kernel[grid](
        hidden_states,
        x_fp8,
        x_sf,
        topk_ids,
        topk_weights,
        topk_idx_out,
        topk_weights_out,
        hidden_states.stride(0),
        hidden_states.stride(1),
        x_fp8.stride(0),
        x_fp8.stride(1),
        x_sf.stride(0),
        x_sf.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        topk_idx_out.stride(0),
        topk_idx_out.stride(1),
        topk_weights_out.stride(0),
        topk_weights_out.stride(1),
        hidden_size,
        topk_ids.shape[1],
        BLOCK_K=block_k,
        GROUP_K=32,
        BLOCK_TOPK=block_topk,
        num_warps=4,
    )


DEEPSEEK_V4_MXFP4_BLOCK_SIZE = 32


@dataclass(frozen=True)
class DeepseekV4AttentionLayout:
    """Static DeepSeek V4 sparse MLA contract.

    `swa_head_bytes` is the uint8 SWA cache row width used by DeepSeek V4:
    FP8 NoPE bytes, BF16 RoPE bytes, UE8M0 scale bytes, then one pad byte.
    """

    kind: str
    compress_ratio: int
    num_heads: int
    num_local_heads: int
    padded_heads: int
    head_dim: int
    nope_head_dim: int
    rope_head_dim: int
    swa_window: int
    swa_head_bytes: int
    compressed_cache_alignment: int
    needs_compressed_cache: bool
    needs_indexer: bool
    indexer_cache_head_bytes: int | None = None


def _deepseek_v4_padded_heads(num_local_heads: int) -> int:
    if num_local_heads <= 64:
        return 64
    if num_local_heads <= 128:
        return 128
    raise ValueError(
        f"DeepSeek V4 attention supports at most 128 local heads, "
        f"got {num_local_heads}"
    )


def _attention_use_fp4_indexer_cache(config: PretrainedConfig) -> bool:
    override = global_server_args_dict.get("attention_use_fp4_indexer_cache", None)
    if override is not None:
        return bool(override)
    attention_config = getattr(config, "attention_config", None)
    if isinstance(attention_config, dict):
        return bool(attention_config.get("use_fp4_indexer_cache", False))
    return bool(getattr(attention_config, "use_fp4_indexer_cache", False))


def deepseek_v4_rope_config(
    config: PretrainedConfig, compress_ratio: int
) -> tuple[float, dict | None]:
    """Return the per-layer DeepSeek V4 RoPE base and scaling config.

    DeepSeek V4 uses ordinary RoPE for SWA-only layers. Compressed layers
    use the checkpoint's separate `compress_rope_theta` together with YaRN.
    """

    if compress_ratio <= 1:
        return float(getattr(config, "rope_theta", 10000.0)), None

    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is not None:
        rope_scaling = dict(rope_scaling)
        rope_scaling["rope_type"] = "deepseek_yarn"
        rope_scaling["mscale"] = 0
        rope_scaling["mscale_all_dim"] = 0
    return (
        float(
            getattr(
                config,
                "compress_rope_theta",
                getattr(config, "rope_theta", 10000.0),
            )
        ),
        rope_scaling,
    )


def deepseek_v4_attention_layout(
    config: PretrainedConfig,
    layer_index: int,
    attn_tp_size: int = 1,
    use_fp4_indexer_cache: bool = False,
) -> DeepseekV4AttentionLayout:
    """Return the per-layer V4 attention/cache layout before kernel wiring.

    This keeps TokenSpeed's model code aligned with the three DeepSeek V4
    attention cases: SWA-only (`compress_ratio <= 1`), HCA (`128`), and CSA
    (`4` with indexer).
    """

    compress_ratio = max(1, int(config.compress_ratios[layer_index]))
    if compress_ratio <= 1:
        kind = "swa"
    elif compress_ratio == 4:
        kind = "csa"
    elif compress_ratio == 128:
        kind = "hca"
    else:
        raise ValueError(
            f"Unsupported DeepSeek V4 compress_ratio={compress_ratio}; "
            "expected 1, 4, or 128."
        )

    if config.num_attention_heads % attn_tp_size != 0:
        raise ValueError(
            f"num_attention_heads={config.num_attention_heads} must be divisible "
            f"by attn_tp_size={attn_tp_size}"
        )
    num_local_heads = config.num_attention_heads // attn_tp_size
    head_dim = int(config.head_dim)
    rope_head_dim = int(config.qk_rope_head_dim)
    nope_head_dim = head_dim - rope_head_dim
    if nope_head_dim <= 0:
        raise ValueError(
            f"head_dim={head_dim} must be larger than rope_head_dim={rope_head_dim}"
        )
    if nope_head_dim % 64 != 0:
        raise ValueError(
            f"DeepSeek V4 FP8 NoPE dim must be divisible by 64, got {nope_head_dim}"
        )

    swa_head_bytes = nope_head_dim + rope_head_dim * 2 + nope_head_dim // 64 + 1
    indexer_cache_head_bytes = None
    if kind == "csa":
        index_head_dim = int(config.index_head_dim)
        if use_fp4_indexer_cache:
            indexer_cache_head_bytes = (
                index_head_dim // 2 + index_head_dim // DEEPSEEK_V4_MXFP4_BLOCK_SIZE
            )
        else:
            indexer_cache_head_bytes = (
                index_head_dim + (index_head_dim // DEEPSEEK_V4_FP8_BLOCK_SIZE) * 4
            )

    return DeepseekV4AttentionLayout(
        kind=kind,
        compress_ratio=compress_ratio,
        num_heads=int(config.num_attention_heads),
        num_local_heads=num_local_heads,
        padded_heads=_deepseek_v4_padded_heads(num_local_heads),
        head_dim=head_dim,
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
        swa_window=int(getattr(config, "sliding_window", 128)),
        swa_head_bytes=swa_head_bytes,
        compressed_cache_alignment=DEEPSEEK_V4_COMPRESSED_CACHE_ALIGNMENT,
        needs_compressed_cache=compress_ratio > 1,
        needs_indexer=kind == "csa",
        indexer_cache_head_bytes=indexer_cache_head_bytes,
    )


class DeepseekV4MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        prefix: str,
        is_shared_expert: bool = False,
        swiglu_limit: float | None = None,
        reduce_results: bool = False,
    ) -> None:
        super().__init__()
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}")
        tp = mapping.moe if is_shared_expert else mapping.dense
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            tp_rank=tp.tp_ep_rank if is_shared_expert else tp.tp_rank,
            tp_size=tp.tp_ep_size if is_shared_expert else tp.tp_size,
            tp_group=tp.tp_ep_group if is_shared_expert else tp.tp_group,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            reduce_results=reduce_results,
            tp_rank=tp.tp_ep_rank if is_shared_expert else tp.tp_rank,
            tp_size=tp.tp_ep_size if is_shared_expert else tp.tp_size,
            tp_group=tp.tp_ep_group if is_shared_expert else tp.tp_group,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.swiglu_limit = swiglu_limit
        self.reduce_results = reduce_results
        self.tp_rank = tp.tp_ep_rank if is_shared_expert else tp.tp_rank
        self.tp_size = tp.tp_ep_size if is_shared_expert else tp.tp_size
        self.tp_group = tp.tp_ep_group if is_shared_expert else tp.tp_group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return x.new_empty((0, self.down_proj.output_size))
        gate_up = _fp8_linear(
            self.gate_up_proj,
            x,
            (
                self.gate_up_proj.output_size_per_partition,
                self.gate_up_proj.input_size,
            ),
        )
        gate, up = gate_up.float().chunk(2, dim=-1)
        if self.swiglu_limit is not None and self.swiglu_limit > 0:
            gate = torch.clamp(gate, max=self.swiglu_limit)
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
        x = (F.silu(gate) * up).to(x.dtype)
        out = _fp8_linear(
            self.down_proj,
            x,
            (self.down_proj.output_size, self.down_proj.input_size_per_partition),
        )
        if self.reduce_results and self.tp_size > 1:
            out = all_reduce(out, self.tp_rank, self.tp_group)
        return out


class DeepseekV4MoEGate(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_index: int,
        hash_indices_dtype: torch.dtype = torch.int32,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.hidden_size)
        )
        self.is_hash_moe = layer_index < config.num_hash_layers
        if self.is_hash_moe:
            self.tid2eid = nn.Parameter(
                torch.empty(
                    config.vocab_size,
                    config.num_experts_per_tok,
                    dtype=hash_indices_dtype,
                ),
                requires_grad=False,
            )
            self.e_score_correction_bias = None
        elif getattr(config, "topk_method", None) == "noaux_tc":
            self.register_parameter("tid2eid", None)
            self.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )
        else:
            self.register_parameter("tid2eid", None)
            self.e_score_correction_bias = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return _deepseek_v4_router_gemm(hidden_states, self.weight)


class DeepseekV4MegaMoEExperts(nn.Module):
    _symm_buffer_cache: dict[tuple[int, int, int, int, int, int, int], object] = {}

    def __init__(
        self,
        *,
        num_experts: int,
        num_local_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        mapping: Mapping,
        prefix: str,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.mapping = mapping
        self.max_num_tokens = _deepseek_v4_mega_moe_max_num_tokens()

        weight_attrs = {"weight_loader": self.weight_loader}
        self.w13_weight = nn.Parameter(
            torch.zeros(
                num_local_experts,
                2 * intermediate_size,
                hidden_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w13_weight, weight_attrs)

        self.w13_weight_scale = nn.Parameter(
            torch.zeros(
                num_local_experts,
                2 * intermediate_size,
                hidden_size // DEEPSEEK_V4_MXFP4_BLOCK_SIZE,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w13_weight_scale, weight_attrs)

        self.w2_weight = nn.Parameter(
            torch.zeros(
                num_local_experts,
                hidden_size,
                intermediate_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w2_weight, weight_attrs)

        self.w2_weight_scale = nn.Parameter(
            torch.zeros(
                num_local_experts,
                hidden_size,
                intermediate_size // DEEPSEEK_V4_MXFP4_BLOCK_SIZE,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w2_weight_scale, weight_attrs)

        self._transformed_l1_weights: tuple[torch.Tensor, torch.Tensor] | None = None
        self._transformed_l2_weights: tuple[torch.Tensor, torch.Tensor] | None = None

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_id: str,
        local_expert_id: int,
    ) -> None:
        expert_data = param.data[local_expert_id]
        if shard_id in ("w1", "w3"):
            if param is not self.w13_weight and param is not self.w13_weight_scale:
                raise ValueError(f"Unexpected MegaMoE w13 shard target: {shard_id}")
            shard_offset = 0 if shard_id == "w1" else self.intermediate_size
            expert_data = expert_data.narrow(0, shard_offset, self.intermediate_size)
        elif shard_id == "w2":
            if param is not self.w2_weight and param is not self.w2_weight_scale:
                raise ValueError(f"Unexpected MegaMoE w2 shard target: {shard_id}")
        else:
            raise ValueError(f"Unsupported DeepSeek V4 MegaMoE shard id: {shard_id}")

        if expert_data.dtype == torch.uint8 and loaded_weight.dtype == getattr(
            torch, "float8_e8m0fnu", None
        ):
            loaded_weight = loaded_weight.view(torch.uint8)
        if expert_data.shape != loaded_weight.shape:
            raise ValueError(
                f"DeepSeek V4 MegaMoE expert weight shape mismatch for "
                f"{self.prefix}: parameter shard {tuple(expert_data.shape)} "
                f"vs checkpoint {tuple(loaded_weight.shape)}"
            )
        expert_data.copy_(loaded_weight)

    @staticmethod
    def _ue8m0_to_float(sf: torch.Tensor) -> torch.Tensor:
        if sf.dtype == torch.uint8:
            return (sf.to(torch.int32) << 23).view(torch.float32)
        return sf.float()

    def _check_runtime_supported(self) -> None:
        if not torch.cuda.is_available():
            raise NotImplementedError("DeepSeek V4 MegaMoE requires CUDA.")
        device = self.w13_weight.device
        if device.type != "cuda":
            raise NotImplementedError(
                "DeepSeek V4 MegaMoE expert weights must be loaded on CUDA."
            )
        if torch.cuda.get_device_capability(device)[0] != 10:
            raise NotImplementedError("DeepGEMM MegaMoE requires SM100 GPUs.")
        if self.hidden_size % 128 != 0 or self.intermediate_size % 128 != 0:
            raise ValueError(
                "DeepGEMM MegaMoE requires hidden and intermediate sizes "
                "to be multiples of 128."
            )

    def finalize_weights(self) -> None:
        if self._transformed_l1_weights is not None:
            return

        self._check_runtime_supported()
        mega_deep_gemm = _deepseek_v4_get_mega_deep_gemm()
        if mega_deep_gemm is None:
            raise RuntimeError(
                "DeepSeek V4 MegaMoE backend requires a DeepGEMM package with "
                "fp8_fp4_mega_moe support; pass --moe-backend mega_moe only "
                "when DeepGEMM is installed."
            )

        w13_scale = mega_deep_gemm.transform_sf_into_required_layout(
            self._ue8m0_to_float(self.w13_weight_scale.data).contiguous(),
            2 * self.intermediate_size,
            self.hidden_size,
            (1, DEEPSEEK_V4_MXFP4_BLOCK_SIZE),
            self.num_local_experts,
        )
        w2_scale = mega_deep_gemm.transform_sf_into_required_layout(
            self._ue8m0_to_float(self.w2_weight_scale.data).contiguous(),
            self.hidden_size,
            self.intermediate_size,
            (1, DEEPSEEK_V4_MXFP4_BLOCK_SIZE),
            self.num_local_experts,
        )
        self._transformed_l1_weights, self._transformed_l2_weights = (
            mega_deep_gemm.transform_weights_for_mega_moe(
                (self.w13_weight.data.view(torch.int8).contiguous(), w13_scale),
                (self.w2_weight.data.view(torch.int8).contiguous(), w2_scale),
            )
        )

        self.w13_weight = None
        self.w13_weight_scale = None
        self.w2_weight = None
        self.w2_weight_scale = None

    def get_symm_buffer(self):
        mega_deep_gemm = _deepseek_v4_get_mega_deep_gemm()
        if mega_deep_gemm is None:
            raise RuntimeError("DeepGEMM MegaMoE symbols are unavailable.")
        group = pg_manager.get_process_group("nccl", self.mapping.moe.tp_ep_group)
        device = torch.cuda.current_device()
        key = (
            id(group),
            device,
            self.num_experts,
            self.max_num_tokens,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
        )
        symm_buffer = self._symm_buffer_cache.get(key)
        if symm_buffer is None:
            symm_buffer = mega_deep_gemm.get_symm_buffer_for_mega_moe(
                group,
                self.num_experts,
                self.max_num_tokens,
                self.top_k,
                self.hidden_size,
                self.intermediate_size,
            )
            self._symm_buffer_cache[key] = symm_buffer
        return symm_buffer

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        activation_clamp: float | None,
        fast_math: bool = True,
    ) -> torch.Tensor:
        if hidden_states.shape[0] > self.max_num_tokens:
            raise ValueError(
                f"DeepSeek V4 MegaMoE got {hidden_states.shape[0]} tokens, "
                f"but the symmetric buffer was sized for {self.max_num_tokens}."
            )

        y = torch.empty_like(hidden_states, dtype=torch.bfloat16)
        symm_buffer = self.get_symm_buffer()
        num_tokens = hidden_states.shape[0]
        _stage_deepseek_v4_mega_moe_inputs(
            hidden_states,
            topk_weights,
            topk_ids,
            symm_buffer.x[:num_tokens],
            symm_buffer.x_sf[:num_tokens],
            symm_buffer.topk_idx[:num_tokens],
            symm_buffer.topk_weights[:num_tokens],
        )

        assert (
            self._transformed_l1_weights is not None
            and self._transformed_l2_weights is not None
        ), (
            "DeepseekV4MegaMoEExperts.finalize_weights() must run via "
            "post_load_weights() before forward()"
        )
        mega_deep_gemm = _deepseek_v4_get_mega_deep_gemm()
        mega_deep_gemm.fp8_fp4_mega_moe(
            y,
            self._transformed_l1_weights,
            self._transformed_l2_weights,
            symm_buffer,
            activation_clamp=activation_clamp,
            fast_math=fast_math,
        )
        return y


class DeepseekV4MoE(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        layer_index: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.layer_index = layer_index
        self.n_shared_experts = config.n_shared_experts
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.scoring_func = getattr(config, "scoring_func", "sqrtsoftplus")
        if self.scoring_func != "sqrtsoftplus":
            raise ValueError(
                f"Unsupported DeepSeek V4 MoE scoring: {self.scoring_func}"
            )
        from tokenspeed.runtime.layers.moe.utils import get_moe_backend

        self.use_mega_moe = get_moe_backend().is_mega_moe()
        if self.use_mega_moe:
            if _deepseek_v4_get_mega_deep_gemm() is None:
                raise RuntimeError(
                    "DeepSeek V4 MegaMoE backend requires an external DeepGEMM "
                    "package with fp8_fp4_mega_moe support."
                )
            if mapping.moe.ep_size <= 1:
                raise ValueError("DeepSeek V4 MegaMoE requires expert parallelism.")
            if mapping.moe.tp_size != 1:
                raise ValueError("DeepSeek V4 MegaMoE does not support mixed TP/EP.")
            if global_server_args_dict.get("ep_num_redundant_experts", 0):
                raise ValueError(
                    "DeepSeek V4 MegaMoE does not support redundant EP experts."
                )
            if config.n_routed_experts % mapping.moe.ep_size != 0:
                raise ValueError(
                    "DeepSeek V4 MegaMoE requires n_routed_experts divisible by "
                    "EP size."
                )
        self.hash_indices_dtype = torch.int64 if self.use_mega_moe else torch.int32
        self.gate = DeepseekV4MoEGate(
            config,
            layer_index,
            hash_indices_dtype=self.hash_indices_dtype,
        )

        if config.n_shared_experts is not None:
            self.shared_experts = DeepseekV4MLP(
                config.hidden_size,
                config.moe_intermediate_size * config.n_shared_experts,
                config.hidden_act,
                mapping,
                quant_config,
                add_prefix("shared_experts", prefix),
                is_shared_expert=True,
                swiglu_limit=getattr(config, "swiglu_limit", None),
                reduce_results=False,
            )
        else:
            self.shared_experts = None

        if self.use_mega_moe:
            self.experts = DeepseekV4MegaMoEExperts(
                num_experts=config.n_routed_experts,
                num_local_experts=config.n_routed_experts // mapping.moe.ep_size,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                mapping=mapping,
                prefix=add_prefix("experts", prefix),
            )
            self.topk = None
        else:
            routed_quant_config = Mxfp4Config(
                ignored_layers=getattr(quant_config, "ignored_layers", None),
                is_checkpoint_mxfp4_serialized=True,
            )
            self.experts = MoELayer(
                top_k=config.num_experts_per_tok,
                num_experts=config.n_routed_experts
                + global_server_args_dict["ep_num_redundant_experts"],
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                quant_config=routed_quant_config,
                layer_index=layer_index,
                prefix=prefix,
                tp_rank=mapping.moe.tp_rank,
                tp_size=mapping.moe.tp_size,
                ep_rank=mapping.moe.ep_rank,
                ep_size=mapping.moe.ep_size,
                activation="swiglu",
                swiglu_limit=getattr(config, "swiglu_limit", None),
                with_bias=True,
                routing_config={
                    "routed_scaling_factor": self.routed_scaling_factor,
                    "correction_bias": self.gate.e_score_correction_bias,
                    "routing_method_type": RoutingMethodType.Renormalize,
                },
            )
            self.topk = TopK(
                top_k=config.num_experts_per_tok,
                renormalize=config.norm_topk_prob,
                correction_bias=self.gate.e_score_correction_bias,
                routed_scaling_factor=self.routed_scaling_factor,
                apply_routed_scaling_factor_on_output=(
                    self.experts.apply_routed_scaling_factor_on_output
                ),
                output_format=self.experts.topk_output_format,
            )

    def _select_experts(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = self.gate(hidden_states)
        return deepseek_v4_select_experts(
            router_logits,
            self.config.num_experts_per_tok,
            self.config.norm_topk_prob,
            correction_bias=self.gate.e_score_correction_bias,
            hash_indices_table=self.gate.tid2eid,
            input_ids=input_ids,
        )

    def _make_topk_output(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        router_scores: torch.Tensor,
    ):
        if self.experts.topk_output_format.is_bypassed():
            router_logits = pack_topk_as_router_logits(
                topk_weights, topk_ids, self.config.n_routed_experts
            )
            return BypassedTopKOutput(
                hidden_states, router_logits, self.topk.topk_config
            )
        return StandardTopKOutput(topk_weights, topk_ids, router_scores)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
        global_num_tokens_per_rank: list[int] | None = None,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0 and not self.use_mega_moe:
            return hidden_states
        if self.use_mega_moe:
            if hidden_states.shape[0] == 0:
                topk_weights = hidden_states.new_empty(
                    (0, self.config.num_experts_per_tok), dtype=torch.float32
                )
                topk_ids = torch.empty(
                    (0, self.config.num_experts_per_tok),
                    device=hidden_states.device,
                    dtype=torch.int64,
                )
            else:
                with deepseek_v4_profile_scope("moe_select_experts"):
                    topk_weights, topk_ids, _ = self._select_experts(
                        hidden_states, input_ids
                    )
            with deepseek_v4_profile_scope("moe_mega_experts"):
                if topk_ids.dtype != torch.int64:
                    topk_ids = topk_ids.to(torch.int64)
                if self.routed_scaling_factor != 1.0:
                    topk_weights = topk_weights * self.routed_scaling_factor
                routed = self.experts(
                    hidden_states,
                    topk_weights,
                    topk_ids,
                    activation_clamp=getattr(self.config, "swiglu_limit", None),
                )
            if self.shared_experts is not None:
                with deepseek_v4_profile_scope("moe_shared_experts"):
                    shared = self._forward_shared_experts_for_mega_moe(
                        hidden_states, global_num_tokens_per_rank
                    )
                routed = routed + shared
            return routed
        with deepseek_v4_profile_scope("moe_select_experts"):
            topk_weights, topk_ids, router_scores = self._select_experts(
                hidden_states, input_ids
            )
        with deepseek_v4_profile_scope("moe_make_topk_output"):
            topk_output = self._make_topk_output(
                hidden_states, topk_weights, topk_ids, router_scores
            )
        with deepseek_v4_profile_scope("moe_experts"):
            routed = self.experts(
                hidden_states=hidden_states,
                topk_output=topk_output,
                num_global_tokens=num_global_tokens,
                max_num_tokens_per_gpu=max_num_tokens_per_gpu,
            )
        if self.routed_scaling_factor != 1.0:
            routed *= self.routed_scaling_factor
        if self.shared_experts is not None:
            with deepseek_v4_profile_scope("moe_shared_experts"):
                shared = self.shared_experts(hidden_states)
            routed = routed + shared
        return routed

    def _forward_shared_experts_for_mega_moe(
        self,
        hidden_states: torch.Tensor,
        global_num_tokens_per_rank: list[int] | None,
    ) -> torch.Tensor:
        if self.shared_experts is None:
            return torch.empty_like(hidden_states)
        if not global_num_tokens_per_rank or self.mapping.moe.tp_ep_size <= 1:
            return self.shared_experts(hidden_states)

        token_counts = [int(count) for count in global_num_tokens_per_rank]
        total_tokens = sum(token_counts)
        if total_tokens == 0:
            return hidden_states.new_empty((0, hidden_states.shape[-1]))

        with deepseek_v4_profile_scope("moe_shared_all_gather"):
            gathered = token_all_gather(
                hidden_states,
                rank=self.mapping.moe.tp_ep_rank,
                group=self.mapping.moe.tp_ep_group,
                scattered_num_tokens=token_counts,
            )
        with deepseek_v4_profile_scope("moe_shared_mlp"):
            shared = self.shared_experts(gathered)
        with deepseek_v4_profile_scope("moe_shared_reduce_scatter"):
            return token_reduce_scatter(
                shared,
                rank=self.mapping.moe.tp_ep_rank,
                group=self.mapping.moe.tp_ep_group,
                scattered_num_tokens=token_counts,
            )


class DeepseekV4Compressor(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        head_dim: int,
        compress_ratio: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.overlap = compress_ratio == 4
        self.coff = 2 if self.overlap else 1
        state_dtype = torch.float32
        self.ape = nn.Parameter(
            torch.empty(compress_ratio, self.coff * head_dim, dtype=state_dtype),
            requires_grad=False,
        )
        self._ape_reordered = False
        self.fused_wkv_wgate = MergedColumnParallelLinear(
            hidden_size,
            [self.coff * head_dim, self.coff * head_dim],
            bias=False,
            quant_config=None,
            prefix=add_prefix("fused_wkv_wgate", prefix),
        )
        self.norm = RMSNorm(head_dim, eps=config.rms_norm_eps)

    def process_weights_after_loading(self, module=None) -> None:
        del module
        if not self.overlap or self._ape_reordered:
            return
        with torch.no_grad():
            self.ape.data.copy_(_deepseek_v4_reorder_c4_ape_2604(self.ape.data))
        self._ape_reordered = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        layer_index: int,
        cos_sin_cache: torch.Tensor,
        *,
        state_cache: torch.Tensor | None = None,
        state_block_table: torch.Tensor | None = None,
        state_block_size: int | None = None,
        write_compressed_cache: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pool = ctx.token_to_kv_pool
        metadata = ctx.attn_backend.forward_metadata
        if metadata is None:
            raise RuntimeError("DeepSeek V4 compressor requires forward metadata")
        profile_prefix = (
            f"indexer_compressor_c{self.compress_ratio}"
            if not write_compressed_cache
            else f"compressor_c{self.compress_ratio}"
        )
        with deepseek_v4_profile_scope(f"{profile_prefix}_dequant_weight"):
            weight_shape = (
                self.fused_wkv_wgate.output_size_per_partition,
                self.fused_wkv_wgate.input_size,
            )
            weight = self.fused_wkv_wgate.weight.view(*weight_shape)
            if weight.dtype == torch.float8_e4m3fn:
                weight = _dequant_fp8_weight(self.fused_wkv_wgate, weight_shape)
        with deepseek_v4_profile_scope(f"{profile_prefix}_matmul"):
            kv_score = _deepseek_v4_bf16_linear_fp32(hidden_states, weight)
            if kv_score is None:
                kv_score = torch.matmul(hidden_states.float(), weight.float().T)
            kv, score = kv_score.split([self.coff * self.head_dim] * 2, dim=-1)
        if state_cache is None:
            state_cache = pool.get_compressor_state_buffer(layer_index)
        if state_block_table is None:
            state_block_table = metadata.compressor_state_block_tables.get(
                self.compress_ratio
            )
        if state_block_size is None:
            state_block_size = (
                pool.get_compressor_state_block_size(layer_index)
                if state_block_table is not None
                else pool.state_block_size
            )
        if state_block_table is not None:
            state_slot_mapping = _group_slot_mapping_from_raw(
                positions,
                metadata.token_to_req_indices[: positions.numel()],
                state_block_table,
                state_block_size,
            )
        else:
            state_block_table = metadata.block_table
            state_slot_mapping = out_cache_loc
        with deepseek_v4_profile_scope(f"{profile_prefix}_save_state"):
            save_deepseek_v4_compressor_state(
                kv=kv,
                score=score,
                ape=self.ape,
                state_cache=state_cache,
                slot_mapping=state_slot_mapping,
                positions=positions,
                block_size=state_block_size,
                compress_ratio=self.compress_ratio,
            )
        if not write_compressed_cache:
            return kv, score

        kv_cache_block_size = pool.get_compressed_block_size(layer_index)
        with deepseek_v4_profile_scope(f"{profile_prefix}_compressed_slot_mapping"):
            compressed_slots = metadata.compressed_slot_mapping(
                positions,
                self.compress_ratio,
                kv_cache_block_size=kv_cache_block_size,
            )
        with deepseek_v4_profile_scope(f"{profile_prefix}_cache_insert"):
            insert = (
                deepseek_v4_csa_compress_kv_cache_insert
                if self.compress_ratio == 4
                else deepseek_v4_hca_compress_kv_cache_insert
            )
            insert(
                state_cache=state_cache,
                token_to_req_indices=metadata.token_to_req_indices[: positions.numel()],
                positions=positions,
                compressor_slot_mapping=state_slot_mapping,
                block_table=state_block_table,
                compressor_block_size=state_block_size,
                rms_norm_weight=self.norm.weight,
                rms_norm_eps=self.norm.variance_epsilon,
                cos_sin_cache=cos_sin_cache,
                kv_cache_2d=pool.get_compressed_kv_buffer_2d(layer_index),
                kv_slot_mapping=compressed_slots,
                kv_cache_block_size=kv_cache_block_size,
                compress_ratio=self.compress_ratio,
            )
        return kv, score


class DeepseekV4Indexer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        prefix: str,
        compress_ratio: int,
        topk_buffer: _DeepseekV4TopKBuffer | None = None,
    ) -> None:
        super().__init__()
        self.wq_b = ReplicatedLinear(
            config.q_lora_rank,
            config.index_n_heads * config.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
        )
        self.weights_proj = ReplicatedLinear(
            config.hidden_size,
            config.index_n_heads,
            bias=False,
            quant_config=None,
            prefix=add_prefix("weights_proj", prefix),
        )
        self.compressor = DeepseekV4Compressor(
            config,
            config.hidden_size,
            config.index_head_dim,
            compress_ratio,
            add_prefix("compressor", prefix),
        )
        self.use_fp4_cache = _attention_use_fp4_indexer_cache(config)
        self.compress_ratio = compress_ratio
        self.n_head = int(config.index_n_heads)
        self.head_dim = int(config.index_head_dim)
        self.topk_tokens = int(config.index_topk)
        self.topk_buffer = topk_buffer
        self.softmax_scale = self.head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        layer_index: int,
        cos_sin_cache: torch.Tensor,
    ) -> torch.Tensor:
        pool = ctx.token_to_kv_pool
        metadata = ctx.attn_backend.forward_metadata
        if metadata is None:
            raise RuntimeError("DeepSeek V4 indexer requires forward metadata")
        indexer_state = pool.get_indexer_state_buffer(layer_index)
        indexer_state_block_table = metadata.indexer_state_block_table
        if indexer_state_block_table is not None:
            indexer_state_block_size = pool.get_indexer_state_block_size(layer_index)
            indexer_state_slot_mapping = _group_slot_mapping_from_raw(
                positions,
                metadata.token_to_req_indices[: positions.numel()],
                indexer_state_block_table,
                indexer_state_block_size,
            )
        else:
            indexer_state_block_table = metadata.block_table
            indexer_state_block_size = pool.state_block_size
            indexer_state_slot_mapping = out_cache_loc
        with deepseek_v4_profile_scope("indexer_compressor_total"):
            self.compressor(
                hidden_states=hidden_states,
                positions=positions,
                ctx=ctx,
                out_cache_loc=out_cache_loc,
                layer_index=layer_index,
                cos_sin_cache=cos_sin_cache,
                state_cache=indexer_state,
                state_block_table=indexer_state_block_table,
                state_block_size=indexer_state_block_size,
                write_compressed_cache=False,
            )
        with deepseek_v4_profile_scope("indexer_compressed_slot_mapping"):
            indexer_block_size = pool.get_indexer_block_size(layer_index)
            indexer_block_table = metadata.compressed_block_table(
                self.compress_ratio,
                indexer_block_size,
            )
            compressed_slots = metadata.compressed_slot_mapping(
                positions,
                self.compress_ratio,
                kv_cache_block_size=indexer_block_size,
            )
        with deepseek_v4_profile_scope("indexer_cache_insert"):
            deepseek_v4_csa_indexer_cache_insert(
                state_cache=indexer_state,
                token_to_req_indices=metadata.token_to_req_indices[: positions.numel()],
                positions=positions,
                compressor_slot_mapping=indexer_state_slot_mapping,
                block_table=indexer_state_block_table,
                compressor_block_size=indexer_state_block_size,
                rms_norm_weight=self.compressor.norm.weight,
                rms_norm_eps=self.compressor.norm.variance_epsilon,
                cos_sin_cache=cos_sin_cache,
                kv_cache_2d=pool.get_indexer_kv_buffer_2d(layer_index),
                kv_slot_mapping=compressed_slots,
                kv_cache_block_size=indexer_block_size,
                use_fp4_cache=self.use_fp4_cache,
                compress_ratio=self.compress_ratio,
            )
        with deepseek_v4_profile_scope("indexer_wq_b"):
            index_q, _ = self.wq_b(qr)
            index_q = index_q.view(-1, self.n_head, self.head_dim)
        with deepseek_v4_profile_scope("indexer_weights_proj"):
            weights, _ = self.weights_proj(hidden_states)
        packed_index_q = None
        packed_weights = None
        if self.use_fp4_cache:
            with deepseek_v4_profile_scope("indexer_prepare_mxfp4"):
                packed_index_q, packed_weights = deepseek_v4_prepare_indexer_q_mxfp4(
                    index_q=index_q,
                    positions=positions,
                    cos_sin_cache=cos_sin_cache,
                    weights=weights,
                    softmax_scale=self.softmax_scale,
                    head_scale=self.n_head**-0.5,
                )
            if ctx.forward_mode is not None and ctx.forward_mode.is_decode():
                topk_out = (
                    self.topk_buffer.get(positions.numel(), positions.device)
                    if self.topk_buffer is not None
                    else None
                )
                with deepseek_v4_profile_scope("indexer_topk_deepgemm_decode"):
                    topk = _deepseek_v4_indexer_topk_from_cache_deepgemm_decode(
                        cache_2d=pool.get_indexer_kv_buffer_2d(layer_index),
                        positions=positions,
                        token_to_req_indices=metadata.token_to_req_indices,
                        block_table=indexer_block_table,
                        cache_block_size=indexer_block_size,
                        index_q=packed_index_q,
                        weights=packed_weights,
                        compress_ratio=self.compress_ratio,
                        topk_tokens=self.topk_tokens,
                        metadata=metadata,
                        out=topk_out,
                    )
                if topk is not None:
                    return topk

        with deepseek_v4_profile_scope("indexer_prepare_reference"):
            index_q_fallback, weights_fallback = (
                deepseek_v4_prepare_indexer_q_reference(
                    index_q=index_q,
                    positions=positions,
                    cos_sin_cache=cos_sin_cache,
                    weights=weights,
                    softmax_scale=self.softmax_scale,
                    head_scale=self.n_head**-0.5,
                    use_fp4=self.use_fp4_cache,
                )
            )
        cache_reader = (
            read_deepseek_v4_indexer_mxfp4_cache
            if self.use_fp4_cache
            else read_deepseek_v4_indexer_fp8_cache
        )
        if ctx.forward_mode is not None and ctx.forward_mode.is_decode():
            topk_out = (
                self.topk_buffer.get(positions.numel(), positions.device)
                if self.topk_buffer is not None
                else None
            )
            return _deepseek_v4_indexer_topk_from_cache_batched(
                cache_reader=cache_reader,
                cache_2d=pool.get_indexer_kv_buffer_2d(layer_index),
                positions=positions,
                token_to_req_indices=metadata.token_to_req_indices,
                block_table=indexer_block_table,
                cache_block_size=indexer_block_size,
                index_q=index_q_fallback,
                weights=weights_fallback,
                compress_ratio=self.compress_ratio,
                topk_tokens=self.topk_tokens,
                out=topk_out,
            )

        indexer_cache = pool.get_indexer_kv_buffer_2d(layer_index)
        topk_chunks = []
        for start, end in _deepseek_v4_indexer_prefill_topk_chunks(
            positions,
            self.compress_ratio,
        ):
            if packed_index_q is not None and packed_weights is not None:
                with deepseek_v4_profile_scope("indexer_topk_deepgemm_prefill"):
                    topk = _deepseek_v4_indexer_topk_from_cache_deepgemm_prefill(
                        cache_2d=indexer_cache,
                        positions=positions[start:end],
                        token_to_req_indices=metadata.token_to_req_indices[start:end],
                        block_table=indexer_block_table,
                        cache_block_size=indexer_block_size,
                        index_q=(
                            packed_index_q[0][start:end],
                            packed_index_q[1][start:end],
                        ),
                        weights=packed_weights[start:end],
                        compress_ratio=self.compress_ratio,
                        topk_tokens=self.topk_tokens,
                        preserve_topk_order=True,
                    )
                if topk is not None:
                    topk_chunks.append(topk)
                    continue
            with deepseek_v4_profile_scope("indexer_topk_fallback_prefill"):
                topk_chunks.append(
                    _deepseek_v4_indexer_topk_from_cache_batched(
                        cache_reader=cache_reader,
                        cache_2d=indexer_cache,
                        positions=positions[start:end],
                        token_to_req_indices=metadata.token_to_req_indices[start:end],
                        block_table=indexer_block_table,
                        cache_block_size=indexer_block_size,
                        index_q=index_q_fallback[start:end],
                        weights=weights_fallback[start:end],
                        compress_ratio=self.compress_ratio,
                        topk_tokens=self.topk_tokens,
                        preserve_topk_order=True,
                    )
                )
        if not topk_chunks:
            return torch.empty(
                (0, self.topk_tokens),
                device=positions.device,
                dtype=torch.int32,
            )
        with deepseek_v4_profile_scope("indexer_topk_cat"):
            return torch.cat(topk_chunks, dim=0)


class DeepseekV4Attention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        layer_index: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
        aux_stream: torch.cuda.Stream | None = None,
        topk_buffer: _DeepseekV4TopKBuffer | None = None,
    ) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.aux_stream = aux_stream
        if self.aux_stream is not None:
            self.ln_events: list[torch.cuda.Event | None] = [
                torch.cuda.Event(),
                torch.cuda.Event(),
            ]
        else:
            self.ln_events = [None, None]
        use_fp4_indexer_cache = _attention_use_fp4_indexer_cache(config)
        self.layout = deepseek_v4_attention_layout(
            config,
            layer_index,
            attn_tp_size=mapping.attn.tp_size,
            use_fp4_indexer_cache=use_fp4_indexer_cache,
        )
        self.compress_ratio = self.layout.compress_ratio
        self.attention_kind = self.layout.kind
        self.num_heads = self.layout.num_heads
        self.num_local_heads = self.layout.num_local_heads
        self.padded_heads = self.layout.padded_heads
        self.head_dim = self.layout.head_dim
        self.qk_rope_head_dim = self.layout.rope_head_dim
        self.nope_head_dim = self.layout.nope_head_dim
        self.scale = self.head_dim**-0.5
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.o_groups = config.o_groups
        self.num_local_groups = self.o_groups // mapping.attn.tp_size
        self.attn_sink = nn.Parameter(
            torch.full((self.padded_heads,), -float("inf"), dtype=torch.float32),
            requires_grad=False,
        )
        rope_base, rope_scaling = deepseek_v4_rope_config(config, self.compress_ratio)
        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position=getattr(config, "max_position_embeddings", 8192),
            base=rope_base,
            rope_scaling=rope_scaling,
            is_neox_style=False,
        )
        if not rope_scaling and hasattr(self.rotary_emb, "forward_cuda"):
            self.rotary_emb.forward = self.rotary_emb.forward_cuda
        self.indexer_rotary_emb = self.rotary_emb
        self.fused_wqa_wkv = MergedColumnParallelLinear(
            config.hidden_size,
            [self.q_lora_rank, self.head_dim],
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("fused_wqa_wkv", prefix),
        )
        self.q_norm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        self.kv_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.fused_qkv_norm = FusedRMSNorm(self.q_norm, self.kv_norm)
        # Fused QKV-RMSNorm is opt-in; reference path is the default until
        # a wider correctness sweep confirms the fused kernel.
        self.use_fused_qkv_rmsnorm = False
        self.wo_a = ColumnParallelLinear(
            self.num_heads * self.head_dim // self.o_groups,
            self.o_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wo_a", prefix),
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        self.wo_a.is_bmm = True
        self.wo_a.bmm_batch_size = self.num_local_groups
        self.wo_b = RowParallelLinear(
            self.o_groups * self.o_lora_rank,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wo_b", prefix),
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        if self.compress_ratio > 1:
            self.compressor = DeepseekV4Compressor(
                config,
                config.hidden_size,
                self.head_dim,
                self.compress_ratio,
                add_prefix("compressor", prefix),
            )
        else:
            self.compressor = None
        if self.compress_ratio == 4:
            self.indexer = DeepseekV4Indexer(
                config,
                mapping,
                quant_config,
                add_prefix("indexer", prefix),
                self.compress_ratio,
                topk_buffer=topk_buffer,
            )
        else:
            self.indexer = None

    def _split_qr_kv(self, qr_kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)

    def _project_q_kv(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qr_kv = _fp8_linear(
            self.fused_wqa_wkv,
            hidden_states,
            (
                self.fused_wqa_wkv.output_size_per_partition,
                self.fused_wqa_wkv.input_size,
            ),
        )
        qr, kv = self._split_qr_kv(qr_kv)
        if self.use_fused_qkv_rmsnorm and qr.is_cuda and qr.shape[0] > 0:
            qr_norm = torch.empty(
                qr.shape,
                dtype=qr.dtype,
                device=qr.device,
            )
            kv_norm = torch.empty(
                kv.shape,
                dtype=kv.dtype,
                device=kv.device,
            )
            self.fused_qkv_norm(
                input_q_a=qr,
                input_kv_a=kv,
                output_q_a=qr_norm,
                output_kv_a=kv_norm,
            )
            qr = qr_norm
            kv = kv_norm
        else:
            qr = self.q_norm(qr)
            kv = self.kv_norm(kv.contiguous())
        q = _fp8_linear(
            self.wq_b,
            qr,
            (self.wq_b.output_size_per_partition, self.wq_b.input_size),
        )
        q = q.view(-1, self.num_local_heads, self.head_dim)
        return q, kv, qr

    def _make_padded_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.empty(
            (hidden_states.shape[0], self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

    def _cos_sin_cache(self) -> torch.Tensor:
        cache = self.rotary_emb.cos_sin_cache
        return cache if cache.dtype == torch.float32 else cache.float()

    def _insert_swa_cache(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        swa_kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        positions: torch.Tensor,
        block_size: int,
    ) -> None:
        if q.shape[0] == 0:
            return
        fused_qnorm_rope_kv_insert(
            q=q,
            kv=kv,
            swa_kv_cache_2d=swa_kv_cache.view(swa_kv_cache.shape[0], -1),
            slot_mapping=slot_mapping,
            positions=positions,
            cos_sin_cache=self._cos_sin_cache(),
            rms_norm_eps=self.q_norm.variance_epsilon,
            block_size=block_size,
        )

    def _slots_from_local_indices(
        self,
        metadata,
        req_idx: int,
        local_indices: torch.Tensor,
        block_size: int,
        block_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if local_indices.numel() == 0:
            return torch.empty(0, device=local_indices.device, dtype=torch.int64)
        if block_table is None:
            block_table = metadata.block_table
        pages = torch.div(local_indices, block_size, rounding_mode="floor")
        offsets = local_indices % block_size
        req = torch.full_like(pages, req_idx, dtype=torch.int64)
        page_ids = metadata.safe_page_ids(block_table, req, pages.long())
        slots = page_ids * block_size + offsets
        return torch.where(page_ids >= 0, slots, torch.full_like(slots, -1))

    def _swa_slots_for_token(
        self,
        metadata,
        token_idx: int,
        position: int,
        block_size: int,
    ) -> torch.Tensor:
        start = max(0, position - self.layout.swa_window + 1)
        local = torch.arange(
            start,
            position + 1,
            device=(
                metadata.swa_block_table.device
                if metadata.swa_block_table is not None
                else metadata.block_table.device
            ),
            dtype=torch.int64,
        )
        req_idx = int(metadata.token_to_req_indices[token_idx].item())
        block_table = (
            metadata.swa_block_table
            if metadata.swa_block_table is not None
            else metadata.block_table
        )
        return self._slots_from_local_indices(
            metadata,
            req_idx,
            local,
            block_size,
            block_table=block_table,
        )

    def _compressed_slots_for_token(
        self,
        metadata,
        token_idx: int,
        position: int,
        block_size: int,
        topk_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.compress_ratio <= 1:
            return torch.empty(0, device=metadata.block_table.device, dtype=torch.int64)
        block_table = metadata.compressed_block_table(self.compress_ratio, block_size)
        if self.compress_ratio == 4:
            if topk_indices is None:
                raise RuntimeError("CSA attention requires indexer top-k indices")
            local = topk_indices[token_idx].to(torch.int64)
            local = local[local >= 0]
        else:
            num_compressed = (position + 1) // self.compress_ratio
            local = torch.arange(
                num_compressed,
                device=block_table.device,
                dtype=torch.int64,
            )
        req_idx = int(metadata.token_to_req_indices[token_idx].item())
        return self._slots_from_local_indices(
            metadata,
            req_idx,
            local,
            block_size,
            block_table=block_table,
        )

    def _forward_flashmla_sparse(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        topk_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        try:
            from flash_mla import flash_mla_sparse_fwd
        except Exception as exc:
            raise DeepseekV4AttentionOpUnavailable(
                "DeepSeek V4 requires FlashMLA sparse attention. Build/install "
                "`tokenspeed-kernel/python` with FlashMLA before serving V4."
            ) from exc

        metadata = ctx.attn_backend.forward_metadata
        if metadata is None:
            raise RuntimeError("DeepSeek V4 attention requires forward metadata")
        pool = ctx.token_to_kv_pool
        kernel_heads = self.padded_heads
        q_padded = torch.zeros(
            (q.shape[0], kernel_heads, self.head_dim),
            device=q.device,
            dtype=q.dtype,
        )
        q_padded[:, : self.num_local_heads].copy_(q)

        per_token_slots: list[tuple[torch.Tensor, torch.Tensor]] = []
        max_candidates = 0
        compressed_block_size = pool.get_compressed_block_size(self.layer_index)
        for token_idx in range(positions.numel()):
            position = int(positions[token_idx].item())
            compressed = self._compressed_slots_for_token(
                metadata,
                token_idx,
                position,
                compressed_block_size,
                topk_indices,
            )
            swa = self._swa_slots_for_token(
                metadata, token_idx, position, pool.swa_block_size
            )
            per_token_slots.append((compressed, swa))
            max_candidates = max(max_candidates, compressed.numel() + swa.numel())
        padded_topk = max(128, ((max_candidates + 127) // 128) * 128)
        indices = torch.full(
            (positions.numel(), padded_topk),
            -1,
            device=q.device,
            dtype=torch.int32,
        )
        lengths = torch.zeros(positions.numel(), device=q.device, dtype=torch.int32)
        rows = []
        cursor = 0
        compressed_cache = (
            pool.get_compressed_kv_buffer_2d(self.layer_index)
            if self.compress_ratio > 1
            else None
        )
        swa_cache = pool.get_swa_kv_buffer(self.layer_index)
        for token_idx, (compressed, swa) in enumerate(per_token_slots):
            token_rows = []
            if compressed.numel() > 0:
                assert compressed_cache is not None
                token_rows.append(
                    dequantize_deepseek_v4_fp8_ds_mla_cache(
                        compressed_cache,
                        compressed,
                        block_size=compressed_block_size,
                    )
                )
            if swa.numel() > 0:
                token_rows.append(
                    dequantize_deepseek_v4_fp8_ds_mla_cache(
                        swa_cache,
                        swa,
                        block_size=pool.swa_block_size,
                    )
                )
            if token_rows:
                joined = torch.cat(token_rows, dim=0)
                rows.append(joined)
                count = joined.shape[0]
                indices[token_idx, :count] = torch.arange(
                    cursor, cursor + count, device=q.device, dtype=torch.int32
                )
                lengths[token_idx] = count
                cursor += count
        if rows:
            kv = torch.cat(rows, dim=0)
        else:
            kv = torch.zeros(1, self.head_dim, device=q.device, dtype=torch.bfloat16)

        out, _, _ = flash_mla_sparse_fwd(
            q=q_padded,
            kv=kv.view(-1, 1, self.head_dim),
            indices=indices.unsqueeze(1),
            sm_scale=self.scale,
            attn_sink=self.attn_sink,
            topk_length=lengths,
        )
        return out[:, : self.num_local_heads]

    def _dequant_fp8_weight(
        self, layer: nn.Module, shape: tuple[int, ...]
    ) -> torch.Tensor:
        return _dequant_fp8_weight(layer, shape)

    def _fp8_linear(self, layer: nn.Module, x: torch.Tensor, shape: tuple[int, ...]):
        return _fp8_linear(layer, x, shape)

    def _dequant_wo_a_weight(self) -> torch.Tensor:
        in_dim = self.num_heads * self.head_dim // self.o_groups
        return self._dequant_fp8_weight(
            self.wo_a, (self.num_local_groups, self.o_lora_rank, in_dim)
        )

    def _project_attention_output(
        self,
        attn_output: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        heads_per_group = self.num_local_heads // self.num_local_groups
        grouped = deepseek_v4_inv_rope_reference(
            attn_output,
            positions,
            self._cos_sin_cache(),
            n_groups=self.num_local_groups,
            heads_per_group=heads_per_group,
            nope_dim=self.nope_head_dim,
            rope_dim=self.qk_rope_head_dim,
        )
        weight = self._dequant_wo_a_weight()
        z = torch.bmm(
            grouped.float().transpose(0, 1),
            weight.transpose(1, 2),
        ).transpose(0, 1)
        z = z.to(attn_output.dtype).contiguous()
        out = self._fp8_linear(
            self.wo_b,
            z.flatten(1),
            (self.wo_b.output_size, self.wo_b.input_size_per_partition),
        )
        return out

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        if ctx.forward_mode is not None and ctx.forward_mode.is_mixed():
            raise NotImplementedError(
                "DeepSeek V4 mixed prefill/decode attention is not wired yet; "
                "leave TOKENSPEED_ENABLE_MIXED_PREFILL_DECODE unset until the "
                "attention backend can split prefill and decode slices."
            )
        profile_prefix = f"attn_{self.attention_kind}"
        with deepseek_v4_profile_scope(f"{profile_prefix}_project_q_kv"):
            q, kv, qr = self._project_q_kv(hidden_states)
        pool = ctx.token_to_kv_pool
        metadata = ctx.attn_backend.forward_metadata
        if metadata is None:
            raise RuntimeError("DeepSeek V4 attention requires forward metadata")
        if metadata.swa_block_table is not None:
            swa_slot_mapping = _group_slot_mapping_from_raw(
                positions,
                metadata.token_to_req_indices[: positions.numel()],
                metadata.swa_block_table,
                pool.swa_block_size,
            )
        else:
            swa_slot_mapping = out_cache_loc

        def insert_swa_cache() -> None:
            with deepseek_v4_profile_scope(f"{profile_prefix}_insert_swa_cache"):
                self._insert_swa_cache(
                    q=q,
                    kv=kv,
                    swa_kv_cache=pool.get_swa_kv_buffer(self.layer_index),
                    slot_mapping=swa_slot_mapping,
                    positions=positions,
                    block_size=pool.swa_block_size,
                )

        def run_compressor() -> None:
            if self.compressor is None:
                return
            with deepseek_v4_profile_scope(f"{profile_prefix}_compressor"):
                self.compressor(
                    hidden_states=hidden_states,
                    positions=positions,
                    ctx=ctx,
                    out_cache_loc=out_cache_loc,
                    layer_index=self.layer_index,
                    cos_sin_cache=self._cos_sin_cache(),
                )

        topk_indices = None
        if self.indexer is not None:
            assert self.compressor is not None

            def run_indexer() -> torch.Tensor:
                with deepseek_v4_profile_scope(f"{profile_prefix}_indexer"):
                    return self.indexer(
                        hidden_states=hidden_states,
                        qr=qr,
                        positions=positions,
                        ctx=ctx,
                        out_cache_loc=out_cache_loc,
                        layer_index=self.layer_index,
                        cos_sin_cache=self._cos_sin_cache(),
                    )

            def insert_and_compress() -> None:
                insert_swa_cache()
                run_compressor()

            topk_indices, _ = _deepseek_v4_maybe_execute_in_parallel(
                run_indexer,
                insert_and_compress,
                self.ln_events[0],
                self.ln_events[1],
                self.aux_stream,
            )
        elif self.compressor is not None:
            _deepseek_v4_maybe_execute_in_parallel(
                run_compressor,
                insert_swa_cache,
                self.ln_events[0],
                self.ln_events[1],
                self.aux_stream,
            )
        else:
            insert_swa_cache()
        backend_decode = getattr(
            ctx.attn_backend,
            "forward_deepseek_v4_decode",
            None,
        )
        backend_prefill = getattr(
            ctx.attn_backend,
            "forward_deepseek_v4_prefill",
            None,
        )
        if (
            backend_decode is not None
            and ctx.forward_mode is not None
            and ctx.forward_mode.is_decode()
        ):
            with deepseek_v4_profile_scope(f"{profile_prefix}_decode_backend"):
                attn_output = backend_decode(
                    q=q,
                    positions=positions,
                    token_to_kv_pool=pool,
                    layer_id=self.layer_index,
                    kind=self.attention_kind,
                    compress_ratio=self.compress_ratio,
                    num_local_heads=self.num_local_heads,
                    padded_heads=self.padded_heads,
                    head_dim=self.head_dim,
                    window_size=self.layout.swa_window,
                    softmax_scale=self.scale,
                    attn_sink=self.attn_sink,
                    topk_indices=topk_indices,
                )
        elif (
            backend_prefill is not None
            and ctx.forward_mode is not None
            and ctx.forward_mode.is_extend()
        ):
            with deepseek_v4_profile_scope(f"{profile_prefix}_prefill_backend"):
                attn_output = backend_prefill(
                    q=q,
                    positions=positions,
                    token_to_kv_pool=pool,
                    layer_id=self.layer_index,
                    kind=self.attention_kind,
                    compress_ratio=self.compress_ratio,
                    num_local_heads=self.num_local_heads,
                    padded_heads=self.padded_heads,
                    head_dim=self.head_dim,
                    window_size=self.layout.swa_window,
                    softmax_scale=self.scale,
                    attn_sink=self.attn_sink,
                    topk_indices=topk_indices,
                )
        else:
            with deepseek_v4_profile_scope(f"{profile_prefix}_fallback_sparse"):
                attn_output = self._forward_flashmla_sparse(
                    q, positions, ctx, topk_indices
                )
        with deepseek_v4_profile_scope(f"{profile_prefix}_output_proj"):
            return self._project_attention_output(attn_output, positions)


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        prefix: str,
        aux_stream: torch.cuda.Stream | None = None,
        topk_buffer: _DeepseekV4TopKBuffer | None = None,
    ) -> None:
        super().__init__()
        self.mapping = mapping
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * config.hidden_size
        self.attn = DeepseekV4Attention(
            config,
            mapping,
            layer_id,
            quant_config,
            add_prefix("attn", prefix),
            aux_stream=aux_stream,
            topk_buffer=topk_buffer,
        )
        self.ffn = DeepseekV4MoE(
            config, mapping, quant_config, layer_id, add_prefix("ffn", prefix)
        )
        self.comm_manager = CommManager(
            mapping=mapping,
            layer_id=layer_id,
            is_moe=True,
            prev_is_moe=True,
        )
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_attn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32), requires_grad=False
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32), requires_grad=False
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32), requires_grad=False
        )

    def _hc_pre(
        self, x: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor
    ):
        return mhc_pre(
            x,
            fn,
            scale,
            base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_sinkhorn_iters,
        )

    def _pre_mlp_input_ids_comm(
        self, input_ids: torch.Tensor, ctx: ForwardContext
    ) -> torch.Tensor:
        if not self.mapping.moe.has_tp_ep:
            return input_ids
        if self.comm_manager.use_all_reduce(is_moe=True):
            return input_ids

        token_counts = self.comm_manager.moe_tp_ep_group_scattered_num_tokens(ctx)
        max_tokens = max(token_counts)
        padded = torch.empty(
            (max_tokens,), device=input_ids.device, dtype=input_ids.dtype
        )
        padded[: input_ids.shape[0]].copy_(input_ids)
        if input_ids.shape[0] < max_tokens:
            padded[input_ids.shape[0] :].zero_()

        gathered = [torch.empty_like(padded) for _ in token_counts]
        group = pg_manager.get_process_group("nccl", self.mapping.moe.tp_ep_group)
        torch.distributed.all_gather(gathered, padded, group=group)
        return torch.cat(
            [tokens[:count] for tokens, count in zip(gathered, token_counts)], dim=0
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        with deepseek_v4_profile_scope("hc_attn_pre"):
            hidden_states, post, comb = self._hc_pre(
                hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
            )
        with deepseek_v4_profile_scope("attn_norm"):
            hidden_states = self.attn_norm(hidden_states)
        with deepseek_v4_profile_scope("attn_total"):
            hidden_states = self.attn(positions, hidden_states, ctx, out_cache_loc)
        with deepseek_v4_profile_scope("hc_attn_post"):
            hidden_states = mhc_post(hidden_states, residual, post, comb)

        residual = hidden_states
        with deepseek_v4_profile_scope("hc_ffn_pre"):
            hidden_states, post, comb = self._hc_pre(
                hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
            )
        with deepseek_v4_profile_scope("ffn_norm"):
            hidden_states = self.ffn_norm(hidden_states)
        ffn_input_ids = input_ids
        use_mega_moe = getattr(self.ffn, "use_mega_moe", False)
        if use_mega_moe:
            token_counts = (
                [int(count) for count in ctx.global_num_tokens]
                if ctx.global_num_tokens is not None
                else [int(hidden_states.shape[0])]
            )
            num_global_tokens = sum(token_counts)
            max_num_tokens_per_gpu = max(token_counts) if token_counts else 0
        else:
            token_counts = None
            with deepseek_v4_profile_scope("pre_mlp_comm"):
                hidden_states = self.comm_manager.pre_mlp_comm(hidden_states, ctx)
            if self.ffn.gate.is_hash_moe:
                with deepseek_v4_profile_scope("pre_mlp_input_ids_comm"):
                    ffn_input_ids = self._pre_mlp_input_ids_comm(input_ids, ctx)
            with deepseek_v4_profile_scope("moe_get_num_tokens"):
                num_global_tokens, max_num_tokens_per_gpu = (
                    self.comm_manager.get_num_tokens(ctx)
                )
        with deepseek_v4_profile_scope("ffn_total"):
            hidden_states = self.ffn(
                hidden_states,
                ffn_input_ids,
                num_global_tokens,
                max_num_tokens_per_gpu,
                global_num_tokens_per_rank=token_counts,
            )
        if not use_mega_moe:
            with deepseek_v4_profile_scope("post_mlp_comm"):
                hidden_states, _ = self.comm_manager.post_mlp_comm(
                    hidden_states, None, ctx
                )
        with deepseek_v4_profile_scope("hc_ffn_post"):
            hidden_states = mhc_post(hidden_states, residual, post, comb)
        return hidden_states


class DeepseekV4Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        # Attention overlap (compressor/CSA on aux stream) is opt-in; the
        # synchronous reference path is the default until a wider correctness
        # sweep confirms the overlap path.
        self.attention_aux_stream = None
        self.topk_indices_buffer = _DeepseekV4TopKBuffer(int(config.index_topk))
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV4DecoderLayer(
                    config,
                    layer_id,
                    mapping,
                    quant_config,
                    add_prefix(f"layers.{layer_id}", prefix),
                    aux_stream=self.attention_aux_stream,
                    topk_buffer=self.topk_indices_buffer,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        hc_dim = config.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(config.hc_mult, hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(config.hc_mult, dtype=torch.float32), requires_grad=False
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32), requires_grad=False
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        hidden_states = input_embeds
        if hidden_states is None:
            with deepseek_v4_profile_scope("embed_tokens"):
                hidden_states = self.embed_tokens(input_ids)
        with deepseek_v4_profile_scope("hc_repeat"):
            hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
        for layer in self.layers:
            hidden_states = layer(
                positions, hidden_states, ctx, out_cache_loc, input_ids
            )
        with deepseek_v4_profile_scope("hc_head"):
            hidden_states = hc_head(
                hidden_states,
                self.hc_head_fn,
                self.hc_head_scale,
                self.hc_head_base,
                self.rms_norm_eps,
                self.hc_eps,
            )
        with deepseek_v4_profile_scope("final_norm"):
            hidden_states = self.norm(hidden_states)
        return hidden_states, None


class DeepseekV4ForCausalLM(BaseCausalLM):
    model_cls = DeepseekV4Model

    def get_stacked_params_mapping(self):
        return [
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
            ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
            ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
        ]

    @staticmethod
    def _map_weight_name(name: str) -> str:
        if name.startswith("layers."):
            name = "model." + name
        elif name.startswith("embed."):
            name = name.replace("embed.", "model.embed_tokens.", 1)
        elif name.startswith("norm."):
            name = "model." + name
        elif name.startswith("hc_head"):
            name = "model." + name
        elif name == "head.weight":
            name = "lm_head.weight"
        if ".shared_experts.w2" in name:
            name = name.replace(".shared_experts.w2", ".shared_experts.down_proj")
        if ".ffn.gate.bias" in name:
            name = name.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
        if re.search(r"\.experts\.\d+\.w[123]\.scale$", name):
            name = name.replace(".scale", ".weight_scale")
        elif name.endswith(".scale"):
            name = name[:-6] + ".weight_scale_inv"
        return name

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = self.get_stacked_params_mapping()
        params_dict = dict(self.named_parameters())
        moe_loader = build_moe_checkpoint_loader(
            params_dict=params_dict,
            expert_schema=ExpertCheckpointSchema(
                gate_proj_name="w1",
                down_proj_name="w2",
                up_proj_name="w3",
            ),
            num_experts=self.config.n_routed_experts,
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
        )
        for raw_name, loaded_weight in weights:
            name = self._map_weight_name(raw_name)
            if name.startswith("mtp."):
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name or ".experts." in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict.get(name)
                if param is None:
                    break
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if moe_loader.matches(name):
                    moe_loader.load(name, loaded_weight)
                    continue
                param = params_dict.get(name)
                if param is None:
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
        self.post_load_weights()

    def post_load_weights(self):
        for module in self.modules():
            if isinstance(module, DeepseekV4Compressor):
                module.process_weights_after_loading()
            elif isinstance(module, DeepseekV4MegaMoEExperts):
                module.finalize_weights()
            elif isinstance(module, MoELayer):
                module.process_weights_after_loading(module)

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        from tokenspeed.runtime.moe.expert_location import ModelConfigForExpertLocation

        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=getattr(config, "n_group", 0),
        )


EntryClass = [
    DeepseekV4ForCausalLM,
]
