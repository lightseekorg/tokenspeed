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

"""Model recipes and pool construction for LCM cache arenas."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch

from tokenspeed.runtime.configs.lcm_layouts import (
    draft_history_lcm_fields,
    inkling_lcm_fields,
    mla_history_lcm_fields,
    qwen_gdn_lcm_fields,
)
from tokenspeed.runtime.configs.lcm_memory_plan import (
    LcmMemoryPlan,
    plan_lcm_fields,
)
from tokenspeed.runtime.configs.paged_cache_spec import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    PagedCacheGroupSpec,
    split_recurrent_state_groups,
)
from tokenspeed.runtime.layers.attention.configs.base import BaseAttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.kv_cache.base import BaseTokenToKVPool

LcmModelFamily = Literal["qwen_gdn", "inkling", "kimi_k3"]

_LOGICAL_BLOCK_TOKENS = 128
_MAX_PADDING_FRACTION = 1.0


@dataclass(frozen=True)
class LcmPoolSpec:
    """Everything needed to bind one model's compute views to an LCM arena."""

    memory_plan: LcmMemoryPlan
    layer_types: tuple[str, ...]
    layer_group_ids: tuple[str, ...]
    state_field_dtypes: Mapping[str, torch.dtype]
    token_capacity: int
    layer_kv_head_counts: tuple[int, ...] | None = None
    extra_paged_groups: tuple[PagedCacheGroupSpec, ...] = ()

    @property
    def pool_size(self) -> int:
        max_packing = max(
            group.cache_blocks_per_lcm_block for group in self.memory_plan.groups
        )
        return (
            self.memory_plan.num_lcm_blocks
            * max_packing
            * self.memory_plan.logical_block_tokens
        )


@dataclass(frozen=True)
class LcmSetup:
    target: LcmPoolSpec
    draft: LcmPoolSpec | None
    cache_budget_bytes: int
    fixed_workspace_bytes: int


def _packing(plan: LcmMemoryPlan) -> dict[str, int]:
    return {
        group.group_id: int(group.cache_blocks_per_lcm_block) for group in plan.groups
    }


def _token_limit(server_args) -> int | None:
    from tokenspeed.runtime.utils.env import envs

    limit = server_args.max_total_tokens
    ci_size = envs.TOKENSPEED_CI_SMALL_KV_SIZE.get_set_value_or(None)
    if ci_size is not None and int(ci_size) > 0:
        ci_limit = int(ci_size)
        limit = ci_limit if limit is None else min(limit, ci_limit)
    return limit


def inkling_layer_kv_head_counts(model_config) -> tuple[int, ...]:
    from tokenspeed.runtime.configs.inkling_config import inkling_kv_heads_for_layer

    text_config = model_config.hf_config.get_text_config()
    return tuple(
        inkling_kv_heads_for_layer(text_config, layer_id, True)
        for layer_id in range(text_config.num_hidden_layers)
    )


def _inkling_fields(attn_config, model_config):
    text_config = model_config.hf_config.get_text_config()
    layer_kv_head_counts = inkling_layer_kv_head_counts(model_config)
    per_rank_heads = tuple(
        max(1, heads // attn_config.attn_tp_size) for heads in layer_kv_head_counts
    )
    hiddenconv_element_size = (
        2 if os.environ.get("INKLING_FP8_SCONV", "0") == "0" else 1
    )
    fields = inkling_lcm_fields(
        layer_group_ids=attn_config.layer_types,
        logical_block_tokens=_LOGICAL_BLOCK_TOKENS,
        layer_kv_heads=per_rank_heads,
        head_dim=attn_config.head_dim,
        kv_element_size=attn_config.kv_cache_dtype.itemsize,
        hidden_size=text_config.hidden_size,
        checkpoint_rows=text_config.sconv_kernel_size - 1,
        kvconv_element_size=2,
        hiddenconv_element_size=hiddenconv_element_size,
        kv_scale_block_size=32 if attn_config.kv_cache_mxfp8 else 0,
        kv_scale_element_size=1 if attn_config.kv_cache_mxfp8 else 0,
    )
    return fields, layer_kv_head_counts


def _inkling_checkpoint_groups(
    plan: LcmMemoryPlan,
) -> tuple[PagedCacheGroupSpec, ...]:
    packing = _packing(plan)
    missing = {"kvconv", "hiddenconv"} - packing.keys()
    if missing:
        raise ValueError(f"Inkling LCM plan is missing groups: {sorted(missing)}")
    return tuple(
        PagedCacheGroupSpec(
            group_id=group_id,
            retention="full_history",
            rows_per_page=plan.logical_block_tokens,
            entry_stride_tokens=1,
            sliding_window_tokens=None,
            family="state",
            cache_blocks_per_lcm_block=packing[group_id],
        )
        for group_id in ("kvconv", "hiddenconv")
    )


def _inkling_workspace_bytes(
    *,
    text_config,
    attn_config,
    num_layers: int,
    verify_tokens: int = 0,
    lagged_window: bool = False,
) -> int:
    from tokenspeed.runtime.configs.inkling_config import inkling_conv_total_dim

    rows = int(attn_config.max_bs) + 2
    state_rows = int(text_config.sconv_kernel_size) - 1
    conv_dim = inkling_conv_total_dim(text_config, attn_config.attn_tp_size)
    element_size = torch.bfloat16.itemsize
    rolling = num_layers * rows * state_rows * conv_dim * element_size
    verify = (
        num_layers
        * int(attn_config.max_bs)
        * int(verify_tokens)
        * conv_dim
        * element_size
    )
    return rolling + verify + (rolling if lagged_window else 0)


def _draft_history_fields(
    *,
    family: LcmModelFamily,
    server_args,
    target_plan: LcmMemoryPlan,
    draft_attn_config,
    draft_model_config,
):
    num_layers = draft_model_config.num_attention_layers
    if family == "qwen_gdn":
        layer_types = (FULL_ATTENTION,) * num_layers
        group_ids = layer_types
        layer_kv_head_counts = None
        per_rank_heads = (
            max(
                draft_attn_config.num_kv_heads // draft_attn_config.attn_tp_size,
                1,
            ),
        ) * num_layers
    elif family == "inkling":
        num_steps = server_args.speculative_num_steps
        if num_steps > num_layers:
            raise ValueError(
                f"Inkling MTP has {num_layers} depth layers; "
                f"--speculative-num-steps {num_steps} would wrap depths "
                "with no trained meaning."
            )
        layer_types = tuple(draft_attn_config.layer_types)
        group_ids = layer_types
        layer_kv_head_counts = inkling_layer_kv_head_counts(draft_model_config)
        per_rank_heads = tuple(
            max(1, heads // draft_attn_config.attn_tp_size)
            for heads in layer_kv_head_counts
        )
    else:
        raise ValueError("draft history helper only supports MHA LCM recipes")

    unknown = set(group_ids) - set(_packing(target_plan))
    if unknown:
        raise ValueError(
            f"draft cache groups are not present in the target plan: {sorted(unknown)}"
        )
    fields = draft_history_lcm_fields(
        layer_group_ids=group_ids,
        enabled_layer_ids=range(num_layers),
        logical_block_tokens=target_plan.logical_block_tokens,
        layer_kv_heads=per_rank_heads,
        head_dim=draft_attn_config.head_dim,
        kv_element_size=draft_attn_config.kv_cache_dtype.itemsize,
        kv_scale_block_size=32 if draft_attn_config.kv_cache_mxfp8 else 0,
        kv_scale_element_size=1 if draft_attn_config.kv_cache_mxfp8 else 0,
    )
    return fields, layer_types, group_ids, layer_kv_head_counts


def _prepare_kimi_k3(
    *,
    server_args,
    model_config,
    attn_config,
    draft_model_config,
    draft_attn_config,
    cache_budget_bytes: int,
    decode_input_tokens: int,
    overlap_schedule_depth: int,
) -> LcmSetup:
    from tokenspeed.runtime.configs.kimi_k3_cache_spec import (
        kimi_k3_layer_group_ids,
        kimi_k3_lcm_blocks_needed,
        kimi_k3_token_capacity_for_lcm_pool,
        plan_kimi_k3_lcm_cache,
    )

    text_config = getattr(model_config.hf_config, "text_config", model_config.hf_config)
    group_ids = kimi_k3_layer_group_ids(text_config)
    layer_types = tuple(
        FULL_ATTENTION if group_id == FULL_ATTENTION else LINEAR_ATTENTION
        for group_id in group_ids
    )
    (
        _,
        _,
        conv_dtype,
        recurrent_dtype,
        _,
    ) = text_config.mamba2_cache_params
    state_dtypes = {
        f"layer.{layer_id}.conv_state": conv_dtype
        for layer_id, layer_type in enumerate(layer_types)
        if layer_type == LINEAR_ATTENTION
    } | {
        f"layer.{layer_id}.recurrent_state": recurrent_dtype
        for layer_id, layer_type in enumerate(layer_types)
        if layer_type == LINEAR_ATTENTION
    }
    reference_plan = plan_kimi_k3_lcm_cache(
        text_config,
        tp_size=attn_config.attn_tp_size,
        mla_cache_dtype=attn_config.kv_cache_dtype,
        mla_quant_method=attn_config.kv_cache_quant_method or None,
        num_lcm_blocks=1,
    )

    draft_fields = None
    draft_parent_bytes = 0
    draft_layer_types = ()
    if draft_attn_config is not None:
        draft_layers = draft_model_config.num_attention_layers
        draft_layer_types = (FULL_ATTENTION,) * draft_layers
        draft_fields = mla_history_lcm_fields(
            layer_group_ids=draft_layer_types,
            logical_block_tokens=reference_plan.logical_block_tokens,
            latent_width=(
                draft_attn_config.kv_lora_rank + draft_attn_config.qk_rope_head_dim
            ),
            element_size=draft_attn_config.kv_cache_dtype.itemsize,
        )
        draft_parent_bytes = plan_lcm_fields(
            draft_fields,
            logical_block_tokens=reference_plan.logical_block_tokens,
            num_lcm_blocks=1,
            cache_blocks_per_lcm_block={FULL_ATTENTION: 12},
            alignment=256,
        ).lcm_block_bytes

    bytes_per_parent = reference_plan.lcm_block_bytes + draft_parent_bytes
    num_lcm_blocks = cache_budget_bytes // bytes_per_parent - 1
    if num_lcm_blocks < 1:
        raise ValueError(
            "Kimi-K3 cache budget must hold a null parent and one usable LCM parent"
        )

    token_limit = _token_limit(server_args)
    sizing = dict(
        max_scheduled_tokens=server_args.chunked_prefill_size,
        max_live_requests=attn_config.max_bs,
        decode_input_tokens=decode_input_tokens,
        overlap_schedule_depth=overlap_schedule_depth,
    )
    if token_limit is not None:
        num_lcm_blocks = min(
            num_lcm_blocks,
            kimi_k3_lcm_blocks_needed(
                reference_plan,
                token_capacity=token_limit,
                **sizing,
            ),
        )
    admitted_tokens = kimi_k3_token_capacity_for_lcm_pool(
        reference_plan,
        num_lcm_blocks=num_lcm_blocks,
        upper_bound_tokens=token_limit,
        **sizing,
    )
    target_plan = plan_kimi_k3_lcm_cache(
        text_config,
        tp_size=attn_config.attn_tp_size,
        mla_cache_dtype=attn_config.kv_cache_dtype,
        mla_quant_method=attn_config.kv_cache_quant_method or None,
        num_lcm_blocks=num_lcm_blocks,
    )
    draft_spec = None
    if draft_fields is not None:
        draft_spec = LcmPoolSpec(
            memory_plan=plan_lcm_fields(
                draft_fields,
                logical_block_tokens=target_plan.logical_block_tokens,
                num_lcm_blocks=num_lcm_blocks,
                cache_blocks_per_lcm_block={FULL_ATTENTION: 12},
                alignment=256,
            ),
            layer_types=draft_layer_types,
            layer_group_ids=draft_layer_types,
            state_field_dtypes={},
            token_capacity=admitted_tokens,
        )
    return LcmSetup(
        target=LcmPoolSpec(
            memory_plan=target_plan,
            layer_types=layer_types,
            layer_group_ids=group_ids,
            state_field_dtypes=state_dtypes,
            token_capacity=admitted_tokens,
        ),
        draft=draft_spec,
        cache_budget_bytes=cache_budget_bytes,
        fixed_workspace_bytes=0,
    )


def _prepare_mha(
    *,
    family: LcmModelFamily,
    server_args,
    model_config,
    attn_config,
    draft_model_config,
    draft_attn_config,
    cache_budget_bytes: int,
) -> LcmSetup:
    if family == "qwen_gdn":
        if attn_config.kv_cache_mxfp8:
            raise RuntimeError(
                "Qwen LCM backing does not yet support the MXFP8 interleaved "
                "scale layout"
            )
        text_config = getattr(
            model_config.hf_config, "text_config", model_config.hf_config
        )
        conv_shape, ssm_shape, conv_dtype, ssm_dtype, _ = (
            text_config.mamba2_cache_params
        )
        layer_types = tuple(attn_config.layer_types)
        group_ids = tuple(split_recurrent_state_groups(layer_types))
        fields = qwen_gdn_lcm_fields(
            layer_types=layer_types,
            layer_group_ids=group_ids,
            logical_block_tokens=_LOGICAL_BLOCK_TOKENS,
            kv_shape=(
                _LOGICAL_BLOCK_TOKENS,
                max(attn_config.num_kv_heads // attn_config.attn_tp_size, 1),
                attn_config.head_dim,
            ),
            kv_element_size=attn_config.kv_cache_dtype.itemsize,
            conv_shape=conv_shape,
            conv_element_size=conv_dtype.itemsize,
            ssm_shape=ssm_shape,
            ssm_element_size=ssm_dtype.itemsize,
        )
        state_dtypes = {
            f"layer.{layer_id}.conv": conv_dtype
            for layer_id, layer_type in enumerate(layer_types)
            if layer_type == LINEAR_ATTENTION
        } | {
            f"layer.{layer_id}.ssm": ssm_dtype
            for layer_id, layer_type in enumerate(layer_types)
            if layer_type == LINEAR_ATTENTION
        }
        layer_kv_head_counts = None
        max_padding_fraction = _MAX_PADDING_FRACTION
        fixed_workspace_bytes = 0
        if draft_attn_config is not None:
            verify_rows = attn_config.max_bs * (
                int(server_args.speculative_num_draft_tokens) + 1
            )
            fixed_workspace_bytes = verify_rows * sum(
                field.payload_bytes
                for field in fields
                if field.field_id.endswith((".conv", ".ssm"))
            )
    else:
        fields, layer_kv_head_counts = _inkling_fields(attn_config, model_config)
        layer_types = tuple(attn_config.layer_types)
        group_ids = layer_types
        state_dtypes = {}
        max_padding_fraction = (
            float("inf")
            if os.environ.get("INKLING_FP8_SCONV", "0") == "0"
            else _MAX_PADDING_FRACTION
        )
        text_config = model_config.hf_config.get_text_config()
        draft_tokens = (
            int(server_args.speculative_num_draft_tokens)
            if draft_attn_config is not None
            else 0
        )
        fixed_workspace_bytes = _inkling_workspace_bytes(
            text_config=text_config,
            attn_config=attn_config,
            num_layers=text_config.num_hidden_layers,
            verify_tokens=draft_tokens,
        )
        if draft_attn_config is not None:
            fixed_workspace_bytes += _inkling_workspace_bytes(
                text_config=draft_model_config.hf_config.get_text_config(),
                attn_config=draft_attn_config,
                num_layers=draft_model_config.num_attention_layers,
                lagged_window=(
                    int(server_args.speculative_num_steps) > 1
                    and os.environ.get("INKLING_MTP_DECODE_LOOKBACK", "1") != "0"
                ),
            )

    reference_plan = plan_lcm_fields(
        fields,
        logical_block_tokens=_LOGICAL_BLOCK_TOKENS,
        budget_bytes=cache_budget_bytes,
        alignment=256,
        max_padding_fraction=max_padding_fraction,
    )
    target_packing = _packing(reference_plan)
    draft_fields = None
    draft_layer_types = ()
    draft_group_ids = ()
    draft_layer_kv_head_counts = None
    draft_group_packing = {}
    draft_parent_bytes = 0
    if draft_attn_config is not None:
        (
            draft_fields,
            draft_layer_types,
            draft_group_ids,
            draft_layer_kv_head_counts,
        ) = _draft_history_fields(
            family=family,
            server_args=server_args,
            target_plan=reference_plan,
            draft_attn_config=draft_attn_config,
            draft_model_config=draft_model_config,
        )
        draft_group_packing = {
            group_id: target_packing[group_id]
            for group_id in {field.group_id for field in draft_fields}
        }
        draft_parent_bytes = plan_lcm_fields(
            draft_fields,
            logical_block_tokens=_LOGICAL_BLOCK_TOKENS,
            num_lcm_blocks=1,
            cache_blocks_per_lcm_block=draft_group_packing,
            alignment=256,
            max_padding_fraction=max_padding_fraction,
        ).lcm_block_bytes

    usable_cache_bytes = cache_budget_bytes - fixed_workspace_bytes
    parent_bytes = reference_plan.lcm_block_bytes + draft_parent_bytes
    num_lcm_blocks = usable_cache_bytes // parent_bytes - 1
    if num_lcm_blocks < 1:
        raise ValueError(
            "cache budget must hold a null parent and one usable LCM parent"
        )

    max_packing = max(target_packing.values())
    token_limit = _token_limit(server_args)
    if token_limit is not None:
        requested = token_limit // _LOGICAL_BLOCK_TOKENS // max_packing
        if requested < 1:
            raise ValueError(
                "the configured token limit must hold at least one LCM parent "
                f"({_LOGICAL_BLOCK_TOKENS * max_packing} child tokens)"
            )
        num_lcm_blocks = min(num_lcm_blocks, requested)

    target_plan = plan_lcm_fields(
        fields,
        logical_block_tokens=_LOGICAL_BLOCK_TOKENS,
        num_lcm_blocks=num_lcm_blocks,
        cache_blocks_per_lcm_block=target_packing,
        alignment=256,
        max_padding_fraction=max_padding_fraction,
    )
    target_spec = LcmPoolSpec(
        memory_plan=target_plan,
        layer_types=layer_types,
        layer_group_ids=group_ids,
        state_field_dtypes=state_dtypes,
        token_capacity=(num_lcm_blocks * max_packing * _LOGICAL_BLOCK_TOKENS),
        layer_kv_head_counts=layer_kv_head_counts,
        extra_paged_groups=(
            _inkling_checkpoint_groups(target_plan) if family == "inkling" else ()
        ),
    )
    draft_spec = None
    if draft_fields is not None:
        draft_plan = plan_lcm_fields(
            draft_fields,
            logical_block_tokens=_LOGICAL_BLOCK_TOKENS,
            num_lcm_blocks=num_lcm_blocks,
            cache_blocks_per_lcm_block=draft_group_packing,
            alignment=256,
            max_padding_fraction=max_padding_fraction,
        )
        draft_spec = LcmPoolSpec(
            memory_plan=draft_plan,
            layer_types=draft_layer_types,
            layer_group_ids=draft_group_ids,
            state_field_dtypes={},
            token_capacity=target_spec.token_capacity,
            layer_kv_head_counts=draft_layer_kv_head_counts,
        )
    return LcmSetup(
        target=target_spec,
        draft=draft_spec,
        cache_budget_bytes=cache_budget_bytes,
        fixed_workspace_bytes=fixed_workspace_bytes,
    )


def prepare_lcm_setup(
    *,
    family: LcmModelFamily,
    server_args,
    model_config,
    attn_config: BaseAttnConfig,
    draft_model_config,
    draft_attn_config: BaseAttnConfig | None,
    cache_budget_bytes: int,
    decode_input_tokens: int,
    overlap_schedule_depth: int,
) -> LcmSetup:
    """Apply one model recipe and size target/draft arenas from one budget."""
    if family == "kimi_k3":
        return _prepare_kimi_k3(
            server_args=server_args,
            model_config=model_config,
            attn_config=attn_config,
            draft_model_config=draft_model_config,
            draft_attn_config=draft_attn_config,
            cache_budget_bytes=cache_budget_bytes,
            decode_input_tokens=decode_input_tokens,
            overlap_schedule_depth=overlap_schedule_depth,
        )
    return _prepare_mha(
        family=family,
        server_args=server_args,
        model_config=model_config,
        attn_config=attn_config,
        draft_model_config=draft_model_config,
        draft_attn_config=draft_attn_config,
        cache_budget_bytes=cache_budget_bytes,
    )


def create_lcm_pool(
    spec: LcmPoolSpec,
    config: BaseAttnConfig,
    *,
    num_layers: int,
    rank: int,
    enable_memory_saver: bool,
) -> BaseTokenToKVPool:
    """Create the concrete compute interface for a prepared LCM spec."""
    plan = spec.memory_plan
    if isinstance(config, MHAConfig):
        from tokenspeed.runtime.layers.attention.kv_cache.lcm_mha import (
            LcmMHATokenToKVPool,
            LcmMHATokenToKVPoolMXFP8,
        )

        pool_cls = (
            LcmMHATokenToKVPoolMXFP8 if config.kv_cache_mxfp8 else LcmMHATokenToKVPool
        )
        return pool_cls(
            size=spec.pool_size,
            dtype=config.kv_cache_dtype,
            head_num=max(config.num_kv_heads // config.attn_tp_size, 1),
            head_dim=config.head_dim,
            layer_num=num_layers,
            device=config.device,
            enable_memory_saver=enable_memory_saver,
            max_batch_size=config.max_bs,
            max_context_len=config.context_len,
            page_size=plan.logical_block_tokens,
            rank=rank,
            layer_types=spec.layer_types,
            sliding_window_tokens=config.sliding_window_tokens,
            max_scheduled_tokens=config.max_scheduled_tokens,
            pd_disaggregation_enabled=config.pd_disaggregation_enabled,
            extra_paged_groups=spec.extra_paged_groups,
            layer_kv_head_counts=spec.layer_kv_head_counts,
            kv_alloc_head_count=config.num_kv_heads,
            memory_plan=plan,
            layer_group_ids=spec.layer_group_ids,
            state_field_dtypes=spec.state_field_dtypes,
        )
    if isinstance(config, MLAConfig):
        from tokenspeed.runtime.layers.attention.kv_cache.lcm_mla import (
            LcmMLATokenToKVPool,
        )

        return LcmMLATokenToKVPool(
            size=spec.pool_size,
            dtype=config.kv_cache_dtype,
            model_dtype=config.dtype,
            quant_method=config.kv_cache_quant_method,
            kv_lora_rank=config.kv_lora_rank,
            qk_rope_head_dim=config.qk_rope_head_dim,
            layer_num=num_layers,
            device=config.device,
            enable_memory_saver=enable_memory_saver,
            max_batch_size=config.max_bs,
            max_context_len=config.context_len,
            page_size=plan.logical_block_tokens,
            rank=rank,
            layer_types=spec.layer_types,
            layer_group_ids=spec.layer_group_ids,
            max_scheduled_tokens=config.max_scheduled_tokens,
            pd_disaggregation_enabled=config.pd_disaggregation_enabled,
            state_field_dtypes=spec.state_field_dtypes,
            memory_plan=plan,
            token_capacity=spec.token_capacity,
        )
    raise TypeError(f"LCM cache does not support config type {type(config).__name__}")
