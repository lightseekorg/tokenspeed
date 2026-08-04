"""Concrete cache-pool construction from a prepared cache spec."""

from tokenspeed.runtime.layers.attention.configs.base import BaseAttnConfig
from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.configs.msa import MSAConfig
from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
from tokenspeed.runtime.layers.attention.kv_cache.setup import CachePoolSpec


def create_cache_pool(
    spec: CachePoolSpec,
    config: BaseAttnConfig,
    *,
    num_layers: int,
    rank: int,
    enable_memory_saver: bool,
) -> CachePool:
    """Create the concrete compute interface for a prepared cache spec."""
    plan = spec.memory_plan
    if spec.family == "deepseek_v4":
        from tokenspeed.runtime.layers.attention.kv_cache.hybrid_deepseek_v4 import (
            HybridDeepseekV4TokenToKVPool,
        )
        from tokenspeed.runtime.layers.attention.kv_cache.recipes.deepseek_v4 import (
            DeepseekV4PoolOptions,
        )

        options = spec.pool_options
        if not isinstance(options, DeepseekV4PoolOptions):
            raise TypeError("DeepSeek V4 cache spec is missing pool options")
        return HybridDeepseekV4TokenToKVPool(
            size=spec.pool_size,
            model_dtype=config.dtype,
            layout=options.layout,
            layer_num=num_layers,
            device=config.device,
            enable_memory_saver=enable_memory_saver,
            max_batch_size=config.max_bs,
            max_context_len=config.context_len,
            page_size=config.page_size,
            rank=rank,
            hf_config=options.hf_config,
            memory_plan=plan,
            token_capacity=spec.token_capacity,
        )
    if isinstance(config, DSAConfig):
        from tokenspeed.runtime.layers.attention.kv_cache.dsa import (
            DSATokenToKVPool,
        )

        return DSATokenToKVPool(
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
            index_head_dim=config.index_head_dim,
            memory_plan=plan,
        )
    if isinstance(config, MSAConfig):
        from tokenspeed.runtime.layers.attention.kv_cache.msa import (
            MSATokenToKVPool,
        )

        return MSATokenToKVPool(
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
            index_head_dim=config.index_head_dim,
            index_dtype=config.dtype,
            indexed_layer_ids=config.sparse_layer_ids,
            layer_types=spec.layer_types,
            sliding_window_tokens=config.sliding_window_tokens,
            max_scheduled_tokens=config.max_scheduled_tokens,
            pd_disaggregation_enabled=config.pd_disaggregation_enabled,
            memory_plan=plan,
        )
    if isinstance(config, MHAConfig):
        if spec.family == "mha":
            from tokenspeed.runtime.layers.attention.kv_cache.mha import (
                MHATokenToKVPool,
                MHATokenToKVPoolMXFP8,
            )

            pool_cls = (
                MHATokenToKVPoolMXFP8 if config.kv_cache_mxfp8 else MHATokenToKVPool
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
                memory_plan=plan,
            )
        if spec.family == "inkling":
            from tokenspeed.runtime.layers.attention.kv_cache.hybrid_inkling import (
                HybridInklingTokenToKVPool,
                HybridInklingTokenToKVPoolMXFP8,
            )

            pool_cls = (
                HybridInklingTokenToKVPoolMXFP8
                if config.kv_cache_mxfp8
                else HybridInklingTokenToKVPool
            )
        elif spec.family == "qwen_gdn":
            from tokenspeed.runtime.layers.attention.kv_cache.hybrid_mha import (
                HybridMHATokenToKVPool,
                HybridMHATokenToKVPoolMXFP8,
            )

            pool_cls = (
                HybridMHATokenToKVPoolMXFP8
                if config.kv_cache_mxfp8
                else HybridMHATokenToKVPool
            )
        else:
            raise TypeError(
                f"cache family {spec.family!r} is incompatible with MHAConfig"
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
            token_capacity=spec.token_capacity,
        )
    if isinstance(config, MLAConfig):
        if spec.family == "mla":
            from tokenspeed.runtime.layers.attention.kv_cache.mla import (
                MLATokenToKVPool,
            )

            return MLATokenToKVPool(
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
                max_scheduled_tokens=config.max_scheduled_tokens,
                memory_plan=plan,
            )

        if spec.family != "kimi_k3":
            raise TypeError(
                f"cache family {spec.family!r} is incompatible with MLAConfig"
            )

        from tokenspeed.runtime.layers.attention.kv_cache.hybrid_kda import (
            HybridKDATokenToKVPool,
        )

        return HybridKDATokenToKVPool(
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
    raise TypeError(f"cache setup does not support config type {type(config).__name__}")
