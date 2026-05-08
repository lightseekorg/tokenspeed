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

import logging
import math
import os
from typing import TYPE_CHECKING

from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.configs.model_config import AttentionArch, is_deepseek_v4
from tokenspeed.runtime.layers.attention.configs.base import BaseAttnConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.kv_cache.base import BaseTokenToKVPool
from tokenspeed.runtime.layers.attention.utils import profile_max_num_pages
from tokenspeed.runtime.utils.env import envs

logger = logging.getLogger(__name__)

_CI_SMALL_KV_SIZE = envs.TOKENSPEED_CI_SMALL_KV_SIZE.get_set_value_or(None)

if TYPE_CHECKING:
    from tokenspeed.runtime.configs.model_config import ModelConfig
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.utils.server_args import ServerArgs


def _resolve_max_num_tokens(
    profiled_num_pages: int,
    page_size: int,
    max_total_tokens: int | None,
) -> int:
    profiled_tokens = profiled_num_pages * page_size
    if max_total_tokens is None:
        return profiled_tokens
    requested_pages = max_total_tokens // page_size
    if requested_pages < 1:
        raise ValueError(
            f"max_total_tokens={max_total_tokens} must contain at least one full page "
            f"(page_size={page_size})"
        )
    return min(profiled_tokens, requested_pages * page_size)


# ---------- backend registry ----------

# Maps backend_name -> (supported archs, backend class)
_BACKEND_REGISTRY: dict[str, tuple[set[AttentionArch], type[AttentionBackend]]] = {}


def register_backend(
    name: str,
    archs: set[AttentionArch],
    cls: type[AttentionBackend],
) -> None:
    _BACKEND_REGISTRY[name] = (archs, cls)


_HYBRID_GDN_ARCHITECTURES = {
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5MoeForConditionalGenerationNextN",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5ForConditionalGenerationNextN",
}


# Aliases for backward compatibility with server_args choices
_BACKEND_ALIASES = {
    "trtllm_mha": "trtllm",
}


def _get_default_backend_name(arch: AttentionArch) -> str:
    platform = current_platform()
    if arch == AttentionArch.MLA:
        if platform.is_blackwell:
            return "trtllm_mla"
        if platform.is_hopper:
            return "flashmla"
        return "trtllm_mla"
    else:
        return "mha"


def _get_backend_cls(name: str, arch: AttentionArch) -> type[AttentionBackend]:
    if name is None:
        candidates = [_get_default_backend_name(arch)]
        for candidate in candidates:
            entry = _BACKEND_REGISTRY.get(candidate)
            if entry is not None and arch in entry[0]:
                return entry[1]
        raise ValueError(
            f"No backend supports arch {arch}. Available: {list(_BACKEND_REGISTRY)}"
        )
    name = _BACKEND_ALIASES.get(name, name)
    entry = _BACKEND_REGISTRY.get(name)
    if entry is None:
        raise ValueError(
            f"Unknown attention backend: {name!r}. Available: {list(_BACKEND_REGISTRY)}"
        )
    supported_archs, cls = entry
    if arch not in supported_archs:
        raise ValueError(
            f"Backend {name!r} does not support arch {arch}. "
            f"Supported archs: {supported_archs}"
        )
    return cls


# ---------- arch -> config class ----------

_CONFIG_CLS: dict[AttentionArch, type[BaseAttnConfig]] = {
    AttentionArch.MHA: MHAConfig,
    AttentionArch.MLA: MLAConfig,
}


def _create_attn_config(
    server_args: ServerArgs, model_config: ModelConfig, is_draft: bool = False
) -> BaseAttnConfig:
    arch = model_config.attention_arch
    if arch not in _CONFIG_CLS:
        raise NotImplementedError(f"Not supported Attention Arch: {arch!r}")
    return _CONFIG_CLS[arch].generate(server_args, model_config, is_draft)


def _create_attn_backend(
    arch: AttentionArch,
    config: BaseAttnConfig,
) -> AttentionBackend:
    return _get_backend_cls(config.backend_name, arch)(config)


def _create_attn_pool(
    config: BaseAttnConfig,
    num_layers: int,
    max_total_num_tokens: int,
    rank: int,
    enable_memory_saver: bool = False,
) -> BaseTokenToKVPool:
    return config.create_pool(
        num_layers, max_total_num_tokens, rank, enable_memory_saver
    )


def _attention_use_fp4_indexer_cache(server_args: "ServerArgs", hf_config) -> bool:
    if getattr(server_args, "attention_use_fp4_indexer_cache", None) is not None:
        return bool(server_args.attention_use_fp4_indexer_cache)
    attention_config = getattr(hf_config, "attention_config", None)
    if isinstance(attention_config, dict):
        return bool(attention_config.get("use_fp4_indexer_cache", False))
    return bool(getattr(attention_config, "use_fp4_indexer_cache", False))


def _create_hybrid_linear_attn(
    server_args: ServerArgs,
    model_config: ModelConfig,
    config: BaseAttnConfig,
    arch: AttentionArch,
    max_num_tokens: int,
    rank: int,
    enable_memory_saver: bool = False,
    full_attn_backend_name: str = None,
    mamba_pool_total_chunks: int = 0,
) -> tuple[AttentionBackend, BaseTokenToKVPool, object]:
    """Create a hybrid backend + pool for GDN hybrid models (Qwen3.5, Qwen3Next)."""
    from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (
        HybridLinearAttnBackend,
        LayerMappedKVPool,
        MambaAttnBackend,
        SimpleMambaPool,
    )

    hf_config = model_config.hf_config
    text_config = getattr(hf_config, "text_config", hf_config)
    full_attn_layers = text_config.full_attention_layer_ids

    # Create the full attention backend for standard MHA layers.
    # Use user's original choice if provided, otherwise auto-select.
    full_attn_backend = _get_backend_cls(full_attn_backend_name, arch)(config)

    # Create mamba/linear attention backend
    config.speculative_num_draft_tokens = getattr(
        server_args, "speculative_num_draft_tokens", 0
    )
    linear_attn_backend = MambaAttnBackend(config)

    # Create KV cache pool (only for full attention layers)
    num_full_attn_layers = len(full_attn_layers)
    inner_pool = config.create_pool(
        num_full_attn_layers, max_num_tokens, rank, enable_memory_saver
    )
    # Wrap with layer ID mapping (global layer IDs -> pool indices)
    pool = LayerMappedKVPool(inner_pool, full_attn_layers)

    # Create mamba state pool using mamba2_cache_params from the model config
    (
        conv_state_shape,
        temporal_state_shape,
        conv_dtype,
        ssm_dtype,
        mamba_layer_ids,
    ) = text_config.mamba2_cache_params
    max_bs = server_args.max_num_seqs // max(
        server_args.data_parallel_size or server_args.mapping.attn.dp_size, 1
    )
    mamba_pool_size = (
        (mamba_pool_total_chunks + 1) if mamba_pool_total_chunks > 0 else max_bs
    )
    mamba_pool = SimpleMambaPool(
        size=mamba_pool_size,
        num_mamba_layers=len(mamba_layer_ids),
        conv_state_shape=conv_state_shape,
        temporal_state_shape=temporal_state_shape,
        conv_dtype=conv_dtype,
        ssm_dtype=ssm_dtype,
        mamba_layer_ids=mamba_layer_ids,
        device=config.device,
        speculative_num_draft_tokens=(
            server_args.speculative_num_draft_tokens
            if server_args.speculative_algorithm is not None
            else 0
        ),
    )
    linear_attn_backend.set_pool(mamba_pool)

    backend = HybridLinearAttnBackend(
        full_attn_backend, linear_attn_backend, full_attn_layers
    )
    logger.info(
        "Created hybrid_linear_attn backend: %d full attn layers, %d linear attn layers, mamba pool size %d",
        len(full_attn_layers),
        len(mamba_layer_ids),
        mamba_pool_size,
    )
    return backend, pool, mamba_pool


# ---------- public API ----------
def create_attn_components(
    server_args: ServerArgs,
    model_config: ModelConfig,
    gpu_id: int,
    rank: int,
    gpu_memory: int,
    enable_memory_saver: bool = False,
    draft_model_config: ModelConfig | None = None,
) -> tuple[
    AttentionBackend,
    BaseTokenToKVPool,
    AttentionBackend | None,
    BaseTokenToKVPool | None,
    int,
    int,
    object | None,
]:
    arch = model_config.attention_arch

    architectures = getattr(model_config.hf_config, "architectures", None) or []
    is_deepseek_v4_model = is_deepseek_v4(model_config.hf_config)
    original_attn_backend = server_args.attention_backend
    if is_deepseek_v4_model:
        server_args.attention_backend = "deepseek_v4"
    if any(a in _HYBRID_GDN_ARCHITECTURES for a in architectures):
        server_args.attention_backend = "hybrid_linear_attn"

    config = _create_attn_config(server_args, model_config)
    draft_attn_config = None
    if draft_model_config:
        draft_attn_config = _create_attn_config(
            server_args, draft_model_config, is_draft=True
        )
    num_layers = model_config.num_attention_layers
    deepseek_v4_layout = None
    profile_cache_cell_size = None
    if is_deepseek_v4_model:
        from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4 import (
            deepseek_v4_cache_layout_from_config,
        )

        deepseek_v4_layout = deepseek_v4_cache_layout_from_config(
            model_config.hf_config,
            page_size=server_args.block_size,
            use_fp4_indexer_cache=_attention_use_fp4_indexer_cache(
                server_args, model_config.hf_config
            ),
        )
        profile_cache_cell_size = deepseek_v4_layout.cache_cell_size(num_layers)

    max_total_num_pages = profile_max_num_pages(
        config,
        gpu_id,
        server_args.mapping.world_size,
        server_args.gpu_memory_utilization,
        server_args.block_size,
        num_layers,
        gpu_memory,
        world_group=server_args.mapping.world_group,
        draft_attn_config=draft_attn_config if draft_attn_config else None,
        draft_num_attention_layers=(
            draft_model_config.num_attention_layers if draft_attn_config else None
        ),
        cache_cell_size=profile_cache_cell_size,
    )
    max_num_tokens = _resolve_max_num_tokens(
        max_total_num_pages,
        server_args.block_size,
        server_args.max_total_tokens,
    )

    if _CI_SMALL_KV_SIZE is not None and int(_CI_SMALL_KV_SIZE) > 0:
        max_num_tokens = int(_CI_SMALL_KV_SIZE)
    if max_num_tokens <= 0:
        raise ValueError(
            f"KV cache token pool size must be positive, got {max_num_tokens}"
        )

    mamba_pool_total_chunks = 0
    mamba_pool = None

    if is_deepseek_v4_model:
        from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4 import (
            DeepseekV4TokenToKVPool,
        )

        backend = _create_attn_backend(arch, config)
        pool = DeepseekV4TokenToKVPool(
            size=max_num_tokens,
            model_dtype=model_config.dtype,
            layout=deepseek_v4_layout,
            layer_num=num_layers,
            device=config.device,
            enable_memory_saver=enable_memory_saver,
            max_batch_size=config.max_bs,
            max_context_len=config.context_len,
            page_size=server_args.block_size,
            rank=rank,
        )
    elif server_args.attention_backend == "hybrid_linear_attn":
        # Budget mamba state pool alongside KV cache from the same GPU memory.
        # Each mamba slot stores conv_state + ssm_state across all mamba layers;
        # we convert that to an equivalent number of KV tokens so both pools
        # are carved from the profiled rest_memory in one pass.
        hf_config = model_config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        (
            conv_shape,
            ssm_shape,
            conv_dtype,
            ssm_dtype,
            mamba_layer_ids,
        ) = text_config.mamba2_cache_params
        mamba_bytes_per_slot = (
            math.prod(conv_shape) * conv_dtype.itemsize
            + math.prod(ssm_shape) * ssm_dtype.itemsize
        ) * len(mamba_layer_ids)

        kv_cell_size = config.cache_cell_size() * model_config.num_attention_layers
        kv_tokens_per_mamba_slot = (
            mamba_bytes_per_slot + kv_cell_size - 1
        ) // kv_cell_size

        # Allocate a fraction of the profiled KV budget to mamba slots.
        # Default 10%: enough for prefix-cache checkpoints beyond working set.
        mamba_mem_fraction = float(
            os.environ.get("TOKENSPEED_MAMBA_MEM_FRACTION", "0.1")
        )
        mamba_budget_tokens = int(max_num_tokens * mamba_mem_fraction)
        mamba_pool_size = max(
            mamba_budget_tokens // max(kv_tokens_per_mamba_slot, 1), 1
        )
        max_num_tokens -= mamba_pool_size * kv_tokens_per_mamba_slot
        page_size = server_args.block_size
        max_num_tokens = (max_num_tokens // page_size) * page_size
        logger.info(
            "Hybrid model: mamba_bytes_per_slot=%d, kv_tokens_per_slot=%d, "
            "mamba_pool_size=%d (%.1f MiB), kv_budget_tokens=%d",
            mamba_bytes_per_slot,
            kv_tokens_per_mamba_slot,
            mamba_pool_size,
            mamba_bytes_per_slot * (mamba_pool_size + 1) / (1 << 20),
            max_num_tokens,
        )

        resolved_original_backend = _BACKEND_ALIASES.get(
            original_attn_backend, original_attn_backend
        )
        backend, pool, mamba_pool = _create_hybrid_linear_attn(
            server_args,
            model_config,
            config,
            arch,
            max_num_tokens,
            rank,
            enable_memory_saver,
            full_attn_backend_name=(
                resolved_original_backend
                if resolved_original_backend != "hybrid_linear_attn"
                else None
            ),
            mamba_pool_total_chunks=mamba_pool_size,
        )
        mamba_pool_total_chunks = mamba_pool.size if mamba_pool is not None else 0
    else:
        backend = _create_attn_backend(arch, config)
        pool = _create_attn_pool(
            config, num_layers, max_num_tokens, rank, enable_memory_saver
        )
    draft_attn_backend = None
    draft_pool = None
    if draft_attn_config:
        draft_archs = getattr(draft_model_config.hf_config, "architectures", None) or []
        if any(a in _HYBRID_GDN_ARCHITECTURES for a in draft_archs):
            resolved_draft_backend = _BACKEND_ALIASES.get(
                original_attn_backend, original_attn_backend
            )
            draft_attn_backend, draft_pool, _ = _create_hybrid_linear_attn(
                server_args,
                draft_model_config,
                draft_attn_config,
                draft_model_config.attention_arch,
                max_num_tokens,
                rank,
                enable_memory_saver,
                full_attn_backend_name=(
                    resolved_draft_backend
                    if resolved_draft_backend != "hybrid_linear_attn"
                    else None
                ),
            )
        else:
            draft_attn_backend = _create_attn_backend(
                draft_model_config.attention_arch, draft_attn_config
            )
            draft_layers = draft_model_config.num_attention_layers
            draft_pool = _create_attn_pool(
                draft_attn_config,
                draft_layers,
                max_num_tokens,
                rank,
                enable_memory_saver,
            )

    return (
        backend,
        pool,
        draft_attn_backend,
        draft_pool,
        max_num_tokens,
        mamba_pool_total_chunks,
        mamba_pool,
    )
