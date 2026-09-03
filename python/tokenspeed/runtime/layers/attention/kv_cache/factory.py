"""Concrete cache-pool construction from a prepared cache spec."""

from tokenspeed.runtime.layers.attention.configs.base import (
    AttnConfig,
    SoftmaxAttnConfig,
)
from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.configs.msa import MSAConfig
from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import CachePoolSpec


def create_cache_arena(
    spec: CachePoolSpec,
    *,
    device: str,
    enable_memory_saver: bool,
) -> CacheArena:
    """Allocate the one arena every compute view of this spec shares."""
    return CacheArena(
        spec.memory_plan,
        device,
        cache_group_specs=spec.cache_group_specs,
        token_capacity=spec.token_capacity,
        enable_memory_saver=enable_memory_saver,
    )


def _mha_pool_class(family: str, *, mxfp8: bool) -> type[CachePool]:
    """The MHA-shaped pool for one family, in its plain or mxfp8 variant.

    All supported families take the same arguments; only the recurrent-state
    aliasing (and the scale planes) differ, which is the class's business, not
    the caller's.
    """
    from tokenspeed.runtime.layers.attention.kv_cache.hybrid_inkling import (
        HybridInklingTokenToKVPool,
        HybridInklingTokenToKVPoolMXFP8,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.hybrid_mha import (
        HybridMHATokenToKVPool,
        HybridMHATokenToKVPoolMXFP8,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.mha import (
        MHATokenToKVPool,
        MHATokenToKVPoolMXFP8,
    )

    by_family = {
        "mha": (MHATokenToKVPool, MHATokenToKVPoolMXFP8),
        "inkling": (HybridInklingTokenToKVPool, HybridInklingTokenToKVPoolMXFP8),
        "qwen_gdn": (HybridMHATokenToKVPool, HybridMHATokenToKVPoolMXFP8),
        "qwen4_exp": (HybridMHATokenToKVPool, HybridMHATokenToKVPoolMXFP8),
    }
    try:
        plain, scaled = by_family[family]
    except KeyError:
        raise TypeError(
            f"cache family {family!r} is incompatible with MHAConfig"
        ) from None
    return scaled if mxfp8 else plain


def create_cache_pool(
    spec: CachePoolSpec,
    config: AttnConfig,
    arena: CacheArena,
    *,
    num_layers: int,
    rank: int,
    field_layer_offset: int = 0,
) -> CachePool:
    """Bind one model's compute views to an already-allocated arena.

    ``field_layer_offset`` places this view's local layer ids onto the
    merged plan's global layer window, so a draft view names the
    continuation fields the target's plan already reserved.
    """
    softmax_attn = config.component(SoftmaxAttnConfig)
    if spec.family == "glm53_flash":
        from tokenspeed.runtime.layers.attention.kv_cache.hybrid_glm53_flash import (
            HybridGlm53FlashTokenToKVPool,
        )
        from tokenspeed.runtime.layers.attention.kv_cache.recipes.glm53_flash import (
            Glm53FlashPoolOptions,
        )

        options = spec.pool_options
        if not isinstance(options, Glm53FlashPoolOptions):
            raise TypeError("GLM-5.3-Flash cache spec is missing pool options")
        if not isinstance(softmax_attn, DSAConfig):
            raise TypeError("GLM-5.3-Flash cache requires DSA attention")
        if options.index_head_dim != softmax_attn.index_head_dim:
            raise ValueError("GLM-5.3-Flash cache spec and attention config disagree")

        return HybridGlm53FlashTokenToKVPool(
            arena=arena,
            dtype=config.kv_cache_dtype,
            model_dtype=config.dtype,
            quant_method=config.kv_cache_quant_method,
            kv_lora_rank=softmax_attn.kv_lora_rank,
            qk_rope_head_dim=softmax_attn.qk_rope_head_dim,
            layer_num=num_layers,
            rank=rank,
            pool_options=options,
            layer_types=spec.layer_types,
            field_layer_offset=field_layer_offset,
        )
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
            arena,
            model_dtype=config.dtype,
            layout=options.layout,
            layer_num=num_layers,
            rank=rank,
            field_layer_offset=field_layer_offset,
        )
    if isinstance(softmax_attn, DSAConfig):
        from tokenspeed.runtime.layers.attention.kv_cache.dsa import (
            DSATokenToKVPool,
        )

        return DSATokenToKVPool(
            arena,
            dtype=config.kv_cache_dtype,
            model_dtype=config.dtype,
            quant_method=config.kv_cache_quant_method,
            kv_lora_rank=softmax_attn.kv_lora_rank,
            qk_rope_head_dim=softmax_attn.qk_rope_head_dim,
            layer_num=num_layers,
            rank=rank,
            index_head_dim=softmax_attn.index_head_dim,
            field_layer_offset=field_layer_offset,
        )
    if isinstance(softmax_attn, MSAConfig):
        from tokenspeed.runtime.layers.attention.kv_cache.msa import (
            MSATokenToKVPool,
        )

        return MSATokenToKVPool(
            arena=arena,
            dtype=config.kv_cache_dtype,
            head_num=max(softmax_attn.num_kv_heads // softmax_attn.attn_tp_size, 1),
            head_dim=softmax_attn.head_dim,
            layer_num=num_layers,
            rank=rank,
            field_layer_offset=field_layer_offset,
        )
    if isinstance(softmax_attn, MHAConfig):
        pool_cls = _mha_pool_class(spec.family, mxfp8=bool(config.kv_cache_mxfp8))
        return pool_cls(
            arena=arena,
            dtype=config.kv_cache_dtype,
            head_num=max(softmax_attn.num_kv_heads // softmax_attn.attn_tp_size, 1),
            head_dim=softmax_attn.head_dim,
            layer_num=num_layers,
            rank=rank,
            layer_types=spec.layer_types,
            layer_kv_head_counts=spec.layer_kv_head_counts,
            kv_alloc_head_count=softmax_attn.num_kv_heads,
            field_layer_offset=field_layer_offset,
        )
    if isinstance(softmax_attn, MLAConfig):
        if spec.family == "mla":
            from tokenspeed.runtime.layers.attention.kv_cache.mla import (
                MLATokenToKVPool,
            )

            return MLATokenToKVPool(
                arena,
                dtype=config.kv_cache_dtype,
                model_dtype=config.dtype,
                quant_method=config.kv_cache_quant_method,
                kv_lora_rank=softmax_attn.kv_lora_rank,
                qk_rope_head_dim=softmax_attn.qk_rope_head_dim,
                layer_num=num_layers,
                rank=rank,
                field_layer_offset=field_layer_offset,
            )

        if spec.family != "kimi_k3":
            raise TypeError(
                f"cache family {spec.family!r} is incompatible with MLAConfig"
            )

        from tokenspeed.runtime.layers.attention.kv_cache.hybrid_kda import (
            HybridKDATokenToKVPool,
        )

        return HybridKDATokenToKVPool(
            arena=arena,
            dtype=config.kv_cache_dtype,
            model_dtype=config.dtype,
            quant_method=config.kv_cache_quant_method,
            kv_lora_rank=softmax_attn.kv_lora_rank,
            qk_rope_head_dim=softmax_attn.qk_rope_head_dim,
            layer_num=num_layers,
            rank=rank,
            layer_types=spec.layer_types,
            field_layer_offset=field_layer_offset,
        )
    raise TypeError(
        f"cache setup does not support config type {type(softmax_attn).__name__}"
    )
