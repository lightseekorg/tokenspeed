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

import math
from collections.abc import Mapping

import torch

from tokenspeed.runtime.configs.cache_runtime import require_positive_int
from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
from tokenspeed.runtime.configs.lcm_layouts import (
    kimi_k3_lcm_fields as build_lcm_fields,
)
from tokenspeed.runtime.configs.lcm_memory_plan import (
    LcmMemoryPlan,
    plan_lcm_fields,
)
from tokenspeed.runtime.configs.paged_cache_spec import FULL_ATTENTION, LINEAR_ATTENTION

_KIMI_K3_LAYERS = 93
_KIMI_K3_KDA_LAYERS = 69
_KIMI_K3_MLA_LAYERS = 24
_KIMI_K3_LOGICAL_BLOCK_TOKENS = 128
_KIMI_K3_STATE_GROUPS = 3
_KIMI_K3_MLA_PACKING = 12


def _require_non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def _one_based_layers(value: object, name: str, num_layers: int) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple of 1-based layer numbers")
    layers = tuple(value)
    if any(isinstance(layer, bool) or not isinstance(layer, int) for layer in layers):
        raise ValueError(f"{name} must contain integer layer numbers")
    if len(layers) != len(set(layers)):
        raise ValueError(f"{name} contains duplicate layer numbers")
    if any(layer < 1 or layer > num_layers for layer in layers):
        raise ValueError(f"{name} contains a layer outside 1..{num_layers}")
    return tuple(sorted(layers))


def kimi_k3_layer_group_ids(text_config: KimiLinearConfig) -> tuple[str, ...]:
    """Map every Kimi-K3 layer to one Full or KDA cache group.

    Counts derive from the config (the released checkpoint is 93 layers =
    69 KDA + 24 MLA) rather than being hardcoded, so structurally-identical
    reduced-layer variants plan the same way. The KDA layer count must split
    evenly into the fixed number of state groups.
    """
    num_layers = require_positive_int(
        "num_hidden_layers", text_config.num_hidden_layers
    )
    linear = text_config.linear_attn_config
    if not isinstance(linear, Mapping):
        raise ValueError("linear_attn_config must be a mapping")
    kda_layers = _one_based_layers(
        linear.get("kda_layers"), "linear_attn_config.kda_layers", num_layers
    )
    kda_layer_ids = tuple(layer - 1 for layer in kda_layers)
    full_layer_ids = tuple(
        layer_id for layer_id in range(num_layers) if layer_id not in set(kda_layer_ids)
    )
    if not kda_layer_ids or not full_layer_ids:
        raise ValueError(
            f"Kimi-K3 FlatKV requires both KDA and full-attention layers, got "
            f"{len(kda_layer_ids)} and {len(full_layer_ids)}"
        )
    if len(kda_layer_ids) % _KIMI_K3_STATE_GROUPS:
        raise ValueError(
            f"Kimi-K3 FlatKV requires the KDA layer count to divide into "
            f"{_KIMI_K3_STATE_GROUPS} state groups, got {len(kda_layer_ids)}"
        )
    if "full_attn_layers" in linear:
        declared = [
            layer
            for layer in linear["full_attn_layers"]
            if not (isinstance(layer, int) and layer > num_layers)
        ]
        declared_full = _one_based_layers(
            declared,
            "linear_attn_config.full_attn_layers",
            num_layers,
        )
        if declared_full != tuple(layer_id + 1 for layer_id in full_layer_ids):
            raise ValueError(
                "linear_attn_config.full_attn_layers must equal the kda_layers complement"
            )

    group_ids = [FULL_ATTENTION] * num_layers
    per_group = len(kda_layer_ids) // _KIMI_K3_STATE_GROUPS
    for index, layer_id in enumerate(kda_layer_ids):
        group_ids[layer_id] = f"{LINEAR_ATTENTION}_{index // per_group}"
    return tuple(group_ids)


def build_kimi_k3_lcm_fields(
    text_config: KimiLinearConfig,
    *,
    tp_size: int,
    mla_cache_dtype: torch.dtype,
    mla_quant_method: str | None,
) -> tuple:
    """Build the Kimi-K3 fields consumed by the common LCM planner."""
    tp_size = require_positive_int("tp_size", tp_size)
    if mla_cache_dtype != torch.float8_e4m3fn:
        raise ValueError(
            "Kimi-K3 Paged cache initially requires mla_cache_dtype=torch.float8_e4m3fn"
        )
    if mla_quant_method == "per_token_head":
        raise ValueError(
            "Kimi-K3 Paged cache does not support per_token_head MLA cache"
        )
    if getattr(text_config, "mla_use_nope", None) is not True:
        raise ValueError("Kimi-K3 Paged cache requires mla_use_nope=True")

    linear = text_config.linear_attn_config
    num_heads = require_positive_int(
        "linear_attn_config.num_heads", linear.get("num_heads")
    )
    head_dim = require_positive_int(
        "linear_attn_config.head_dim", linear.get("head_dim")
    )
    kernel_size = require_positive_int(
        "linear_attn_config.short_conv_kernel_size",
        linear.get("short_conv_kernel_size"),
    )
    if num_heads % tp_size:
        raise ValueError(
            f"KDA num_heads={num_heads} must be divisible by tp_size={tp_size}"
        )
    kv_lora_rank = require_positive_int("kv_lora_rank", text_config.kv_lora_rank)
    rope_dim = require_positive_int("qk_rope_head_dim", text_config.qk_rope_head_dim)
    return build_lcm_fields(
        layer_group_ids=kimi_k3_layer_group_ids(text_config),
        logical_block_tokens=_KIMI_K3_LOGICAL_BLOCK_TOKENS,
        latent_width=kv_lora_rank + rope_dim,
        mla_element_size=mla_cache_dtype.itemsize,
        conv_shape=(3 * num_heads * head_dim // tp_size, kernel_size - 1),
        conv_element_size=torch.bfloat16.itemsize,
        recurrent_shape=(num_heads // tp_size, head_dim, head_dim),
        recurrent_element_size=torch.float32.itemsize,
    )


def plan_kimi_k3_lcm_cache(
    text_config: KimiLinearConfig,
    *,
    tp_size: int,
    mla_cache_dtype: torch.dtype,
    mla_quant_method: str | None,
    cache_budget_bytes: int | None = None,
    num_lcm_blocks: int | None = None,
) -> LcmMemoryPlan:
    """Plan Kimi-K3 on the common P=128 two-level LCM geometry."""
    fields = build_kimi_k3_lcm_fields(
        text_config,
        tp_size=tp_size,
        mla_cache_dtype=mla_cache_dtype,
        mla_quant_method=mla_quant_method,
    )
    # The MLA latent history is TP-invariant while the KDA state shards by
    # TP, so pack as many KDA pages per MLA-sized plane as fit to keep the
    # planner's padding fraction bounded at any TP.
    mla_plane_bytes = _KIMI_K3_MLA_PACKING * next(
        field.payload_bytes for field in fields if field.group_id == FULL_ATTENTION
    )
    linear_plane_bytes = sum(
        field.payload_bytes
        for field in fields
        if field.group_id == f"{LINEAR_ATTENTION}_0" and field.plane_id == "slot.0"
    )
    linear_packing = max(1, mla_plane_bytes // linear_plane_bytes)
    plan = plan_lcm_fields(
        fields,
        logical_block_tokens=_KIMI_K3_LOGICAL_BLOCK_TOKENS,
        budget_bytes=cache_budget_bytes,
        num_lcm_blocks=num_lcm_blocks,
        cache_blocks_per_lcm_block={
            FULL_ATTENTION: _KIMI_K3_MLA_PACKING,
            f"{LINEAR_ATTENTION}_0": linear_packing,
            f"{LINEAR_ATTENTION}_1": linear_packing,
            f"{LINEAR_ATTENTION}_2": linear_packing,
        },
        alignment=256,
    )
    packing = {
        group.group_id: group.cache_blocks_per_lcm_block for group in plan.groups
    }
    expected_packing = {
        FULL_ATTENTION: _KIMI_K3_MLA_PACKING,
        f"{LINEAR_ATTENTION}_0": linear_packing,
        f"{LINEAR_ATTENTION}_1": linear_packing,
        f"{LINEAR_ATTENTION}_2": linear_packing,
    }
    if packing != expected_packing:
        raise ValueError(
            f"Kimi-K3 LCM packing must be {expected_packing}, got {packing}"
        )
    num_mla_layers = sum(
        1 for gid in kimi_k3_layer_group_ids(text_config) if gid == FULL_ATTENTION
    )
    if len(plan.planes) != num_mla_layers:
        raise ValueError(
            f"Kimi-K3 LCM requires {num_mla_layers} planes (one per MLA "
            f"layer), got {len(plan.planes)}"
        )
    return plan


def kimi_k3_lcm_blocks_needed(
    plan: LcmMemoryPlan,
    *,
    token_capacity: int,
    max_scheduled_tokens: int,
    max_live_requests: int,
    decode_input_tokens: int = 1,
    overlap_schedule_depth: int = 0,
) -> int:
    """Return physical LCM parents needed at the configured concurrency."""
    require_positive_int("plan.logical_block_tokens", plan.logical_block_tokens)
    require_positive_int("token_capacity", token_capacity)
    _require_non_negative_int("max_scheduled_tokens", max_scheduled_tokens)
    require_positive_int("max_live_requests", max_live_requests)
    _require_non_negative_int("decode_input_tokens", decode_input_tokens)
    if overlap_schedule_depth not in (0, 1):
        raise ValueError(
            f"overlap_schedule_depth must be 0 or 1, got {overlap_schedule_depth}"
        )
    if overlap_schedule_depth and decode_input_tokens == 0:
        raise ValueError(
            "overlapped Paged cache sizing requires decode_input_tokens > 0"
        )

    page_tokens = plan.logical_block_tokens
    protected_pages = max_live_requests * math.ceil(
        overlap_schedule_depth * decode_input_tokens / page_tokens
    )
    scheduled_pages = math.ceil(min(max_scheduled_tokens, token_capacity) / page_tokens)
    total_lcm_blocks = 0
    for group in plan.groups:
        if group.group_id == FULL_ATTENTION:
            child_pages = (
                math.ceil(token_capacity / page_tokens)
                + max_live_requests
                - 1
                + protected_pages
            )
        else:
            # MambaStateManager keeps the current and next writable state page
            # per live request in addition to the pages materialized by the
            # scheduled chunk.
            child_pages = 2 * max_live_requests + scheduled_pages + protected_pages
        total_lcm_blocks += math.ceil(child_pages / group.cache_blocks_per_lcm_block)
    return total_lcm_blocks


def kimi_k3_token_capacity_for_lcm_pool(
    plan: LcmMemoryPlan,
    *,
    num_lcm_blocks: int,
    max_scheduled_tokens: int,
    max_live_requests: int,
    decode_input_tokens: int = 1,
    overlap_schedule_depth: int = 0,
    upper_bound_tokens: int | None = None,
) -> int:
    """Invert :func:`kimi_k3_lcm_blocks_needed` by monotonic binary search."""
    require_positive_int("num_lcm_blocks", num_lcm_blocks)
    if upper_bound_tokens is None:
        full_group = plan.group(FULL_ATTENTION)
        upper_bound_tokens = (
            num_lcm_blocks
            * full_group.cache_blocks_per_lcm_block
            * plan.logical_block_tokens
        )
    require_positive_int("upper_bound_tokens", upper_bound_tokens)

    low, high = 0, upper_bound_tokens
    while low < high:
        candidate = (low + high + 1) // 2
        required = kimi_k3_lcm_blocks_needed(
            plan,
            token_capacity=candidate,
            max_scheduled_tokens=max_scheduled_tokens,
            max_live_requests=max_live_requests,
            decode_input_tokens=decode_input_tokens,
            overlap_schedule_depth=overlap_schedule_depth,
        )
        if required <= num_lcm_blocks:
            low = candidate
        else:
            high = candidate - 1
    if low == 0:
        raise ValueError(
            f"num_lcm_blocks={num_lcm_blocks} cannot admit one token with "
            "the configured Kimi-K3 Paged cache scheduler limits"
        )
    return low
