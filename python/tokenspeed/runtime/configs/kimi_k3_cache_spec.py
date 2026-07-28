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

from tokenspeed.runtime.configs.flat_cache_runtime import require_positive_int
from tokenspeed.runtime.configs.hybrid_cache_plan import (
    CacheComponentSpec,
    FlatHybridCachePlan,
    LayerCacheSpec,
    plan_flat_hybrid_cache,
)
from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
from tokenspeed.runtime.configs.paged_cache_spec import FULL_ATTENTION, LINEAR_ATTENTION

_KIMI_K3_LAYERS = 93
_KIMI_K3_KDA_LAYERS = 69
_KIMI_K3_MLA_LAYERS = 24


def _require_non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def kimi_k3_shared_pool_pages(
    plan: FlatHybridCachePlan,
    *,
    token_capacity: int,
    max_scheduled_tokens: int,
    max_live_requests: int,
    decode_input_tokens: int = 1,
    overlap_schedule_depth: int = 0,
) -> int:
    """Return usable shared-pool pages needed for a Kimi-K3 token capacity.

    The scheduler allocates globally unique page ids from one physical pool,
    so every logical group's peak demand must be summed. Full-history groups
    retain all admitted tokens and include per-request tail fragmentation.
    State groups retain one recurrent window per live request plus the largest
    scheduled chunk. Overlap protects the next decode allocation in every
    group.
    """
    require_positive_int("plan.block_size", plan.block_size)
    require_positive_int("token_capacity", token_capacity)
    _require_non_negative_int("max_scheduled_tokens", max_scheduled_tokens)
    require_positive_int("max_live_requests", max_live_requests)
    _require_non_negative_int("decode_input_tokens", decode_input_tokens)
    if overlap_schedule_depth not in (0, 1):
        raise ValueError(
            f"overlap_schedule_depth must be 0 or 1, got {overlap_schedule_depth}"
        )
    if overlap_schedule_depth and decode_input_tokens == 0:
        raise ValueError("overlapped FlatKV sizing requires decode_input_tokens > 0")

    page_size = plan.block_size
    protected_pages = max_live_requests * math.ceil(
        overlap_schedule_depth * decode_input_tokens / page_size
    )
    scheduled_pages = math.ceil(min(max_scheduled_tokens, token_capacity) / page_size)
    total_pages = 0
    for group in plan.groups:
        if group.family == "history":
            total_pages += (
                math.ceil(token_capacity / page_size)
                + max_live_requests
                - 1
                + protected_pages
            )
        elif group.family == "state":
            # State+FullHistory is the scheduler's recurrent-state kind. It
            # keeps one live state page/request; explicit sliding groups use
            # their configured resident window instead.
            resident_tokens = (
                1
                if group.sliding_window_tokens is None
                else max(group.sliding_window_tokens - 1, 0)
            )
            resident_pages = max_live_requests * math.ceil(resident_tokens / page_size)
            total_pages += resident_pages + scheduled_pages + protected_pages
        else:
            raise ValueError(
                f"Kimi-K3 group {group.group_id!r} has unsupported family "
                f"{group.family!r}"
            )
    return total_pages


def kimi_k3_token_capacity_for_shared_pool(
    plan: FlatHybridCachePlan,
    *,
    usable_pages: int,
    max_scheduled_tokens: int,
    max_live_requests: int,
    decode_input_tokens: int = 1,
    overlap_schedule_depth: int = 0,
    upper_bound_tokens: int | None = None,
) -> int:
    """Invert :func:`kimi_k3_shared_pool_pages` by monotonic binary search."""
    require_positive_int("usable_pages", usable_pages)
    if upper_bound_tokens is None:
        upper_bound_tokens = usable_pages * plan.block_size
    require_positive_int("upper_bound_tokens", upper_bound_tokens)

    low, high = 0, upper_bound_tokens
    while low < high:
        candidate = (low + high + 1) // 2
        required_pages = kimi_k3_shared_pool_pages(
            plan,
            token_capacity=candidate,
            max_scheduled_tokens=max_scheduled_tokens,
            max_live_requests=max_live_requests,
            decode_input_tokens=decode_input_tokens,
            overlap_schedule_depth=overlap_schedule_depth,
        )
        if required_pages <= usable_pages:
            low = candidate
        else:
            high = candidate - 1
    if low == 0:
        raise ValueError(
            f"usable_pages={usable_pages} cannot admit one token with the "
            "configured Kimi-K3 FlatKV scheduler limits"
        )
    return low


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


def build_kimi_k3_cache_specs(
    text_config: KimiLinearConfig,
    *,
    tp_size: int,
    mla_cache_dtype: torch.dtype,
    mla_quant_method: str | None,
    preferred_block_size: int,
    kernel_alignment: int,
) -> tuple[LayerCacheSpec, ...]:
    if text_config.num_hidden_layers != _KIMI_K3_LAYERS:
        raise ValueError(
            f"Kimi-K3 FlatKV requires 93 layers, got {text_config.num_hidden_layers}"
        )
    tp_size = require_positive_int("tp_size", tp_size)
    preferred_block_size = require_positive_int(
        "preferred_block_size", preferred_block_size
    )
    kernel_alignment = require_positive_int("kernel_alignment", kernel_alignment)
    if mla_cache_dtype != torch.float8_e4m3fn:
        raise ValueError(
            "Kimi-K3 FlatKV initially requires mla_cache_dtype=torch.float8_e4m3fn"
        )
    if mla_quant_method == "per_token_head":
        raise ValueError("Kimi-K3 FlatKV does not support per_token_head MLA cache")
    if getattr(text_config, "mla_use_nope", None) is not True:
        raise ValueError("Kimi-K3 FlatKV requires mla_use_nope=True")

    linear = text_config.linear_attn_config
    if not isinstance(linear, Mapping):
        raise ValueError("linear_attn_config must be a mapping")
    kda_layers = _one_based_layers(
        linear.get("kda_layers"), "linear_attn_config.kda_layers", _KIMI_K3_LAYERS
    )
    kda_set = set(kda_layers)
    complement = tuple(
        layer for layer in range(1, _KIMI_K3_LAYERS + 1) if layer not in kda_set
    )
    if "full_attn_layers" in linear:
        # MTP checkpoints also list the NextN draft layer(s) past the base
        # depth; they belong to the draft worker, not the base cache plan.
        declared = [
            layer
            for layer in linear["full_attn_layers"]
            if not (isinstance(layer, int) and layer > _KIMI_K3_LAYERS)
        ]
        declared_full = _one_based_layers(
            declared,
            "linear_attn_config.full_attn_layers",
            _KIMI_K3_LAYERS,
        )
        if declared_full != complement:
            raise ValueError(
                "linear_attn_config.full_attn_layers must equal the kda_layers complement"
            )
    if len(kda_layers) != _KIMI_K3_KDA_LAYERS or len(complement) != _KIMI_K3_MLA_LAYERS:
        raise ValueError(
            f"Kimi-K3 FlatKV requires 69 KDA and 24 MLA layers, got "
            f"{len(kda_layers)} and {len(complement)}"
        )

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

    latent_shape = (1, kv_lora_rank + rope_dim)
    history_components = (
        CacheComponentSpec(
            "latent_kv",
            latent_shape,
            mla_cache_dtype,
            math.prod(latent_shape) * mla_cache_dtype.itemsize,
            0,
            mla_cache_dtype.itemsize,
        ),
    )
    conv_shape = (3 * num_heads * head_dim // tp_size, kernel_size - 1)
    recurrent_shape = (num_heads // tp_size, head_dim, head_dim)
    state_components = (
        CacheComponentSpec(
            "conv_state",
            conv_shape,
            torch.bfloat16,
            0,
            math.prod(conv_shape) * torch.bfloat16.itemsize,
            torch.bfloat16.itemsize,
        ),
        CacheComponentSpec(
            "recurrent_state",
            recurrent_shape,
            torch.float32,
            0,
            math.prod(recurrent_shape) * torch.float32.itemsize,
            torch.float32.itemsize,
        ),
    )
    specs: list[LayerCacheSpec] = []
    for layer_id in range(_KIMI_K3_LAYERS):
        is_kda = layer_id + 1 in kda_set
        specs.append(
            LayerCacheSpec(
                layer_id=layer_id,
                family="state" if is_kda else "history",
                retention="full_history",
                transfer_policy="latest_snapshot" if is_kda else "full_suffix",
                group_id_prefix=LINEAR_ATTENTION if is_kda else FULL_ATTENTION,
                group_order=1 if is_kda else 0,
                compatibility_key=(
                    (
                        "kda_state",
                        conv_shape,
                        recurrent_shape,
                        torch.bfloat16,
                        torch.float32,
                    )
                    if is_kda
                    else ("mla_history", latent_shape, mla_cache_dtype)
                ),
                preferred_block_size=preferred_block_size,
                kernel_alignment=kernel_alignment,
                components=state_components if is_kda else history_components,
            )
        )
    return tuple(specs)


def plan_kimi_k3_flat_cache(
    text_config: KimiLinearConfig,
    *,
    flat_kvcache_enabled: bool,
    tp_size: int,
    mla_cache_dtype: torch.dtype,
    mla_quant_method: str | None,
    preferred_block_size: int,
    kernel_alignment: int,
    cache_budget_bytes: int,
    minimum_usable_pages: int = 1,
) -> FlatHybridCachePlan:
    if not flat_kvcache_enabled:
        raise RuntimeError(
            "Kimi-K3 is FlatKV-only and requires tokenspeed_scheduler.FLAT_KVCACHE=True"
        )
    specs = build_kimi_k3_cache_specs(
        text_config,
        tp_size=tp_size,
        mla_cache_dtype=mla_cache_dtype,
        mla_quant_method=mla_quant_method,
        preferred_block_size=preferred_block_size,
        kernel_alignment=kernel_alignment,
    )
    plan = plan_flat_hybrid_cache(
        specs,
        cache_budget_bytes=cache_budget_bytes,
        minimum_usable_pages=minimum_usable_pages,
    )
    expected_group_ids = (
        FULL_ATTENTION,
        f"{LINEAR_ATTENTION}_0",
        f"{LINEAR_ATTENTION}_1",
        f"{LINEAR_ATTENTION}_2",
    )
    actual_group_ids = tuple(group.group_id for group in plan.groups)
    if actual_group_ids != expected_group_ids:
        raise ValueError(
            f"Kimi-K3 FlatKV expected groups {expected_group_ids}, got {actual_group_ids}"
        )
    if len(plan.physical_slots) != 24:
        raise ValueError(
            f"Kimi-K3 FlatKV expected 24 physical slots, got {len(plan.physical_slots)}"
        )
    if plan.diagnostics.padding_binding_count != 3:
        raise ValueError(
            "Kimi-K3 FlatKV expected three padding bindings, got "
            f"{plan.diagnostics.padding_binding_count}"
        )
    if len(plan.layer_bindings) != _KIMI_K3_LAYERS:
        raise ValueError(
            f"Kimi-K3 FlatKV expected 93 layer bindings, got {len(plan.layer_bindings)}"
        )
    if any(group.block_size != plan.block_size for group in plan.groups):
        raise ValueError("Kimi-K3 FlatKV requires one common block size for all groups")
    return plan
