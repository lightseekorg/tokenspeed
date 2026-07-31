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
import os
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.configs.model_config import AttentionArch, is_deepseek_v4
from tokenspeed.runtime.configs.paged_cache_spec import (
    STATE_LAYER_TYPES,
    hybrid_slab_group_size,
)
from tokenspeed.runtime.layers.attention.configs.base import BaseAttnConfig
from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.configs.msa import (
    MSAConfig,
)
from tokenspeed.runtime.layers.attention.kv_cache.base import BaseTokenToKVPool
from tokenspeed.runtime.layers.attention.lcm_setup import (
    LcmPoolSpec,
    create_lcm_pool,
    inkling_layer_kv_head_counts,
    prepare_lcm_setup,
)
from tokenspeed.runtime.layers.attention.utils import (
    profile_available_cache_memory_bytes,
    profile_max_num_pages,
)
from tokenspeed.runtime.utils.env import envs

logger = logging.getLogger(__name__)

_CI_SMALL_KV_SIZE = envs.TOKENSPEED_CI_SMALL_KV_SIZE.get_set_value_or(None)
if TYPE_CHECKING:
    from tokenspeed.runtime.configs.model_config import ModelConfig
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.utils.server_args import ServerArgs


def _kv_profile_layer_divisor(num_layers, layer_types, *, sliding_window_tokens=None):
    """Attention layers to charge per token in the KV memory profile:
    layers-per-group under the slab layout, else all layers (single
    source: hybrid_slab_group_size)."""
    gs = hybrid_slab_group_size(
        layer_types,
        sliding_window_tokens=sliding_window_tokens,
    )
    return gs if gs is not None else num_layers


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


def _resolve_draft_cache_cell_size_for_profile(
    draft_attn_config: BaseAttnConfig | None,
    draft_model_config: ModelConfig | None,
    draft_profile_cache_cell_size: int | None,
) -> int:
    if draft_profile_cache_cell_size is not None:
        return draft_profile_cache_cell_size
    if draft_attn_config is None or draft_model_config is None:
        return 0
    storage_layers = _kv_profile_layer_divisor(
        draft_model_config.num_attention_layers,
        getattr(draft_attn_config, "layer_types", None),
        sliding_window_tokens=getattr(draft_attn_config, "sliding_window_tokens", None),
    )
    return draft_attn_config.cache_cell_size() * storage_layers


def _pool_allocated_bytes(pool) -> int:
    if pool is None:
        return 0
    lcm_pool = getattr(pool, "lcm_pool", None)
    if lcm_pool is not None:
        return int(lcm_pool.backing.nbytes)
    sizes = pool.get_kv_size_bytes()
    if isinstance(sizes, tuple):
        return sum(int(size) for size in sizes)
    return int(sizes)


def _validate_shared_lcm_geometry(pool, draft_pool) -> None:
    """Validate the draft's independent arena against target-owned tables."""
    if draft_pool is None:
        return
    target_plan = getattr(pool, "_lcm_memory_plan", None)
    draft_plan = getattr(draft_pool, "_lcm_memory_plan", None)
    if draft_plan is None:
        return
    if target_plan is None:
        raise RuntimeError("an LCM draft pool requires an LCM target pool")
    if (
        draft_plan.logical_block_tokens != target_plan.logical_block_tokens
        or draft_plan.num_lcm_blocks != target_plan.num_lcm_blocks
    ):
        raise RuntimeError("target and draft LCM parent geometry does not match")

    target_groups = {group.group_id: group for group in target_plan.groups}
    for draft_group in draft_plan.groups:
        target_group = target_groups.get(draft_group.group_id)
        if target_group is None:
            raise RuntimeError(
                f"draft LCM group {draft_group.group_id!r} is absent from target"
            )
        if (
            draft_group.cache_blocks_per_lcm_block
            != target_group.cache_blocks_per_lcm_block
            or draft_group.page_count != target_group.page_count
        ):
            raise RuntimeError(
                f"target and draft LCM group {draft_group.group_id!r} "
                "do not share page-id geometry"
            )

    target_specs = {
        spec.group_id: spec for spec in getattr(pool, "paged_cache_group_specs", ())
    }
    for draft_spec in getattr(draft_pool, "paged_cache_group_specs", ()):
        target_spec = target_specs.get(draft_spec.group_id)
        if target_spec is None:
            raise RuntimeError(
                f"draft cache group {draft_spec.group_id!r} is absent from target"
            )
        shared_policy = (
            "retention",
            "rows_per_page",
            "entry_stride_tokens",
            "sliding_window_tokens",
            "family",
            "block_size",
            "cache_blocks_per_lcm_block",
        )
        if any(
            getattr(draft_spec, field) != getattr(target_spec, field)
            for field in shared_policy
        ):
            raise RuntimeError(
                f"target and draft cache group {draft_spec.group_id!r} "
                "do not share scheduler semantics"
            )

    target_lcm_pool = getattr(pool, "lcm_pool", None)
    draft_lcm_pool = getattr(draft_pool, "lcm_pool", None)
    if target_lcm_pool is None or draft_lcm_pool is None:
        raise RuntimeError("LCM target and draft pools must allocate their arenas")
    if (
        target_lcm_pool.backing.untyped_storage().data_ptr()
        == draft_lcm_pool.backing.untyped_storage().data_ptr()
    ):
        raise RuntimeError("target and draft LCM arenas must not share backing")


def _cache_storage_report(
    *,
    configured_cache_bytes: int,
    pool,
    draft_pool,
    fixed_workspace_bytes: int = 0,
) -> dict:
    """Describe cache storage from allocated tensors, not scheduler counts."""
    plan = getattr(pool, "_lcm_memory_plan", None)
    if plan is not None:
        packing = {
            group.group_id: int(group.cache_blocks_per_lcm_block)
            for group in plan.groups
        }
        physical_token_capacity = (
            int(plan.num_lcm_blocks)
            * max(packing.values())
            * int(plan.logical_block_tokens)
        )
        if physical_token_capacity != int(pool.size):
            raise RuntimeError(
                "LCM geometry capacity does not match the allocated pool size"
            )
        capacity_source = "lcm_geometry"
        geometry = {
            "logical_block_tokens": int(plan.logical_block_tokens),
            "num_lcm_blocks": int(plan.num_lcm_blocks),
            "cache_blocks_per_lcm_block": packing,
        }
    else:
        rows = {
            int(tensor.shape[0])
            for tensor in getattr(pool, "k_buffer", ())
            if tensor is not None
        }
        slot_tokens = int(getattr(pool, "_slot_tokens", pool.page_size))
        capacities = {row - slot_tokens for row in rows}
        if capacities != {int(pool.size)}:
            raise RuntimeError(
                "allocated KV rows do not match the profiled token capacity"
            )
        physical_token_capacity = capacities.pop()
        capacity_source = "allocated_token_rows"
        geometry = {
            "logical_block_tokens": int(pool.page_size),
            "allocated_rows": physical_token_capacity + slot_tokens,
            "reserved_rows": slot_tokens,
        }

    target_bytes = _pool_allocated_bytes(pool)
    draft_bytes = _pool_allocated_bytes(draft_pool)
    allocated_cache_bytes = target_bytes + draft_bytes + fixed_workspace_bytes
    if allocated_cache_bytes > configured_cache_bytes:
        raise RuntimeError(
            "allocated cache storage exceeds its profiled budget: "
            f"{allocated_cache_bytes} > {configured_cache_bytes}"
        )
    return {
        "configured_cache_bytes": int(configured_cache_bytes),
        "allocated_cache_bytes": allocated_cache_bytes,
        "physical_token_capacity": physical_token_capacity,
        "capacity_source": capacity_source,
        "geometry": geometry
        | {
            "target_bytes": target_bytes,
            "draft_bytes": draft_bytes,
            "fixed_workspace_bytes": fixed_workspace_bytes,
        },
    }


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
# Hybrid linear-attention models whose full-attention layers are MLA (not MHA)
# and whose linear layers are KDA (per-channel gated delta rule), not GDN.
# They share the same HybridLinearAttnBackend wrapper and cache-group pool;
# the base sub-backend auto-resolves to MLA from the arch, and the linear
# sub-backend runs the KDA kernels (MambaAttnBackend.is_kda branch).
_HYBRID_MLA_KDA_ARCHITECTURES = {
    "KimiK3ForConditionalGeneration",
}

# Inkling stays on the MHA path plus its thin sconv wrapper; it is not hybrid-GDN.
_INKLING_ARCHITECTURES = {
    "InklingForConditionalGeneration",
    "InklingForConditionalGenerationNextN",
}


# Aliases for backward compatibility with server_args choices
_BACKEND_ALIASES = {
    "trtllm_mha": "trtllm",
}


def _get_default_backend_name(arch: AttentionArch) -> str:
    if arch == AttentionArch.MLA:
        return "mla"
    if arch == AttentionArch.DSA:
        return "dsa"
    if arch == AttentionArch.MSA:
        return "msa"
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


def _validate_lcm_page_size(
    config: BaseAttnConfig,
    *,
    logical_page_size: int,
) -> None:
    """Require the scheduler page to contain whole configured kernel pages."""
    kernel_page_size = int(config.page_size)
    if (
        logical_page_size <= 0
        or kernel_page_size <= 0
        or logical_page_size % kernel_page_size
    ):
        raise ValueError(
            "logical page size must be a positive multiple of kernel page "
            f"size, got {logical_page_size} and {kernel_page_size}"
        )


# ---------- arch -> config class ----------

_CONFIG_CLS: dict[AttentionArch, type[BaseAttnConfig]] = {
    AttentionArch.MHA: MHAConfig,
    AttentionArch.MLA: MLAConfig,
    AttentionArch.DSA: DSAConfig,
    AttentionArch.MSA: MSAConfig,
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


def _create_attn_backend_with_name(
    name: str | None,
    arch: AttentionArch,
    config: BaseAttnConfig,
) -> AttentionBackend:
    original_name = config.backend_name
    config.backend_name = name
    try:
        return _get_backend_cls(name, arch)(config)
    finally:
        config.backend_name = original_name


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


def _resolve_kda_backend(kda_backend: str) -> str:
    """Resolve the KDA prefill backend policy to a concrete choice.

    ``auto`` picks the fastest available kernel — ``cutedsl_kda`` (the tokenspeed-cutedsl-kda AOT build
    matching this device), then ``flashkda`` (optional source-built package),
    falling back to the portable FLA scan. ``fla`` forces the portable scan.
    Explicit choices are validated against availability and fail fast with an
    install hint instead of silently mis-routing. Decode is unaffected either
    way.
    """
    from tokenspeed_kernel.ops.attention.cutedsl_kda import is_cutedsl_kda_installed
    from tokenspeed_kernel.ops.attention.flash_kda import is_flash_kda_installed

    if kda_backend == "auto":
        if is_cutedsl_kda_installed():
            resolved = "cutedsl_kda"
        elif is_flash_kda_installed():
            resolved = "flashkda"
        else:
            resolved = "fla"
        logger.info("KDA prefill backend auto-resolved to %s", resolved)
        return resolved
    if kda_backend == "flashkda" and not is_flash_kda_installed():
        raise ValueError(
            "--kda-backend flashkda requires the tokenspeed-flashkda "
            "package (SM90+, CUDA 12.9+): pip install tokenspeed-flashkda"
        )
    if kda_backend == "cutedsl_kda" and not is_cutedsl_kda_installed():
        raise ValueError(
            "--kda-backend cutedsl_kda requires the tokenspeed-cutedsl-kda package with a "
            "build matching this device (sm_100a / sm_103a) and the public "
            "nvidia-cutlass-dsl, apache-tvm-ffi, cuda-python wheels"
        )
    return kda_backend


def _resolve_hybrid_full_backend_name(
    requested_name: str | None,
    *,
    is_kda: bool,
    has_lcm_plan: bool,
) -> str | None:
    """Resolve the compute backend that consumes the hybrid history cache."""
    name = _BACKEND_ALIASES.get(requested_name, requested_name)
    if name == "hybrid_linear_attn":
        name = None
    # NVIDIA K3 defaults to its CuteDSL history consumer. AMD keeps the
    # generic MLA backend; explicit user choices remain authoritative.
    if has_lcm_plan and is_kda and name is None and not current_platform().is_amd:
        return "tokenspeed_mla"
    return name


def _create_hybrid_linear_attn(
    server_args: ServerArgs,
    model_config: ModelConfig,
    config: BaseAttnConfig,
    arch: AttentionArch,
    max_num_tokens: int,
    rank: int,
    enable_memory_saver: bool = False,
    full_attn_backend_name: str = None,
    is_kda: bool = False,
    lcm_spec: LcmPoolSpec | None = None,
) -> tuple[AttentionBackend, BaseTokenToKVPool]:
    """Create a hybrid backend + pool for a linear-attention model.

    GDN (Qwen3.5, MHA base) or, when ``is_kda`` is set, KDA (Kimi-K3,
    MLA base). The ordinary MHA/MLA pool owns one ``LcmCachePool`` and
    publishes the per-group tables consumed by both sub-backends.
    """
    from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (
        HybridLinearAttnBackend,
        LayerMappedKVPool,
        MambaAttnBackend,
    )

    hf_config = model_config.hf_config
    text_config = getattr(hf_config, "text_config", hf_config)
    full_attn_layers = text_config.full_attention_layer_ids
    has_lcm_plan = lcm_spec is not None

    # Create the full attention backend for standard MHA layers.
    # Use user's original choice if provided, otherwise auto-select.
    full_attn_backend = _create_attn_backend_with_name(
        full_attn_backend_name,
        arch,
        config,
    )

    if has_lcm_plan and is_kda:
        # Paged cache contract: see CuteDSLMLABackend.mark_cache_contract.
        mark_cache_contract = getattr(full_attn_backend, "mark_cache_contract", None)
        if mark_cache_contract is not None:
            mark_cache_contract()

    # Create mamba/linear attention backend. Only propagate the configured
    # verify width when spec-dec is actually enabled — matches MLAConfig /
    # MHAConfig.generate. Otherwise the BaseAttnConfig sentinel (1) wins so
    # non-spec hybrid decode doesn't get misclassified as target verify /
    # draft extend by `self.spec_num_tokens > 1`.
    if server_args.speculative_algorithm is not None:
        config.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens

    # The pool covers all layers, so pool indices equal global layer ids and
    # its layer_types line up with the planned state fields. State layers
    # carry no K/V tensors, matching the LCM plan's separate state fields.
    num_total_layers = len(text_config.layers_block_type)
    inner_pool = (
        create_lcm_pool(
            lcm_spec,
            config,
            num_layers=num_total_layers,
            rank=rank,
            enable_memory_saver=enable_memory_saver,
        )
        if lcm_spec is not None
        else config.create_pool(
            num_total_layers, max_num_tokens, rank, enable_memory_saver
        )
    )
    pool = LayerMappedKVPool(inner_pool, list(range(num_total_layers)))

    # Read mamba2_cache_params to decide whether this model actually has
    # any linear / mamba layers. A draft model on a hybrid-GDN target
    # (e.g. MTP on Qwen3.5) shares the same architecture class as the
    # target but commonly ships with *zero* mamba layers — in that case
    # we skip the mamba backend / pool entirely so that its
    # ``init_forward_metadata_*`` hooks do not run (they would otherwise
    # touch a zero-sized pool on the same persistent state_indices_list
    # as the target, which breaks the captured CUDA graph).
    mamba_layer_ids = text_config.mamba2_cache_params[-1]

    if len(mamba_layer_ids) == 0:
        logger.info(
            "Created hybrid_linear_attn backend: %d full attn layers, 0 linear "
            "attn layers (skipping mamba backend / pool)",
            len(full_attn_layers),
        )
        return full_attn_backend, pool

    kda_backend = (getattr(server_args, "kda_backend", None) or "auto").strip().lower()
    if is_kda:
        kda_backend = _resolve_kda_backend(kda_backend)
    linear_attn_backend = MambaAttnBackend(
        config, is_kda=is_kda, kda_backend=kda_backend
    )

    # Recurrent state lives in the LCM arena and is addressed by the
    # per-group block tables, so no separate request-indexed Mamba pool exists.
    linear_attn_backend.set_kv_pool(pool)

    backend = HybridLinearAttnBackend(
        full_attn_backend, linear_attn_backend, full_attn_layers
    )
    logger.info(
        "Created hybrid_linear_attn backend: %d full attn layers, %d linear attn layers, %s",
        len(full_attn_layers),
        len(mamba_layer_ids),
        "LCM state fields",
    )
    return backend, pool


def _floor_tokens_to_layout_grid(max_num_tokens: int, config) -> int:
    """Floor the memory-profiled token capacity onto the KV layout's grid.

    The profile is memory-dependent (``--gpu-memory-utilization`` over the
    free bytes at boot), so the raw value can land ANYWHERE on the 128-token
    page grid — but the hetero slot layout is only viewable on a coarser
    grid. With ``slot_tokens`` S and a largest group page P (both derived
    from the config's KV head counts), the pool row counts that must divide
    are:

    - target hybrid slab: rows = size + S viewed at base-page granularity
      after the per-layer head reinterpretation -> needs ``128 | size``
      (any 128-grid value satisfies it);
    - MTP draft pool: SAME page-id space and the SAME byte-uniform slot
      layout (the draft is a depth-count miniature of the target pool, see
      the Inkling draft branch below), so its full-attention view pages at
      P likewise divide on the 128 grid. Flooring to ``size ≡ 0 (mod P)``
      keeps the id count itself on the coarse grid.
      (Historical: before the drafter consumed the table at P, the draft
      pool held one 128-row slot per id and its P-row page view required
      ``size ≡ 128 (mod 256)`` — an ODD id count. The 2026-07-14 cont. 9
      boot lottery was literally the parity of the profiled id count.)

    Flooring costs at most 255 tokens (~0.02%). No-op for layouts without
    hetero head counts.
    """
    if not getattr(config, "layer_kv_head_counts", None):
        return max_num_tokens
    slot = int(getattr(config, "slot_tokens", 0) or config.page_size)
    page_sizes = getattr(config, "group_page_sizes", None) or {}
    largest_page = max([config.page_size, *page_sizes.values()])
    # size ≡ 0 (mod largest_page). Degenerates to plain page flooring when
    # every group shares the base page.
    residue = 0
    floored = max_num_tokens - ((max_num_tokens - residue) % largest_page)
    if floored != max_num_tokens:
        logger.info(
            "KV capacity floored to the hetero layout grid: %d -> %d tokens "
            "(slot %d, largest group page %d)",
            max_num_tokens,
            floored,
            slot,
            largest_page,
        )
    return floored


def _apply_inkling_hetero_kv(config, model_config) -> None:
    """Publish the byte-uniform slot layout: per-group page sizes and per-layer KV head counts.

    A layer serving ``heads_l`` of the slab's ``num_kv_heads`` allocation
    width reinterprets one slot's bytes as ``page_size * alloc // heads_l``
    tokens -- the same ratio ``MHATokenToKVPool._layer_page_tokens`` applies
    when it views the slab. Both sides derive from the config's KV head
    counts rather than a fixed pairing, so any
    ``swa_num_key_value_heads`` / ``ckpt_num_key_value_heads`` combination
    stays consistent. A published page wider than the slot's bytes afford
    would over-subscribe that group's ids.

    ``alloc`` comes from ``config.num_kv_heads`` (what the pool receives as
    ``kv_alloc_head_count``), not ``max(counts)``: an all-narrow pool whose
    own layers never include an alloc-width layer is still allocated at the
    config max and must keep its widened page.

    Args:
        config: The attention config to populate in place.
        model_config: Model config supplying the Inkling text config.
    """
    tc = model_config.hf_config.get_text_config()
    counts = inkling_layer_kv_head_counts(model_config)
    alloc = config.num_kv_heads
    page_tokens = {
        label: config.page_size * alloc // heads
        for label, heads in zip(tc.paged_cache_layer_types, counts, strict=True)
    }
    config.slot_tokens = max(page_tokens.values(), default=config.page_size)
    config.group_page_sizes = {
        label: pages
        for label, pages in page_tokens.items()
        if pages != config.page_size
    }
    config.layer_kv_head_counts = counts


def _publish_inkling_conv_groups(config, model_config, server_args) -> tuple[int, int]:
    """Publish per-label kvconv (+ optional hiddenconv) paged groups.

    Conv blocks scale with the KV element size (an fp8 slot holds half the bf16 column tokens).

    Returns ``(conv_block_tokens, hidden_block_tokens)``.
    """
    from tokenspeed.runtime.configs.paged_cache_spec import PagedCacheGroupSpec

    tc = model_config.hf_config.get_text_config()
    bt = server_args.block_size
    kv_elem = config.kv_cache_dtype.itemsize
    conv_bt = bt * kv_elem // 2
    slot_bytes = (
        bt
        * max(config.num_kv_heads // config.attn_tp_size, 1)
        * config.head_dim
        * kv_elem
    )
    # fp8 doubles the hiddenconv block (halves ids/token); must agree with the pool's conv_col_dtype
    hcol_elem = 2 if os.environ.get("INKLING_FP8_SCONV", "1") == "0" else 1
    hbt = 1
    while hbt * 2 * tc.hidden_size * hcol_elem <= slot_bytes:
        hbt *= 2
    assert conv_bt > 0 and hbt * tc.hidden_size * hcol_elem <= slot_bytes
    config.extra_paged_groups = tuple(
        PagedCacheGroupSpec(
            group_id=f"kvconv_{label}",
            retention="sliding_window",
            rows_per_page=conv_bt,
            entry_stride_tokens=1,
            sliding_window_tokens=conv_bt + tc.sconv_kernel_size,
            family="history",
            block_size=conv_bt,
        )
        for label in dict.fromkeys(tc.paged_cache_layer_types)
    )
    if (
        True
    ):  # paged hiddenconv is unconditional (INKLING_PAGED_HIDDENCONV gate retired 2026-07-15)
        # ATTN columns ride the K slot and MLP columns ride the V slot; validate
        # each new base geometry against the Inkling shape suite.
        config.extra_paged_groups = config.extra_paged_groups + tuple(
            PagedCacheGroupSpec(
                group_id=f"hiddenconv_{label}",
                retention="sliding_window",
                rows_per_page=hbt,
                entry_stride_tokens=1,
                sliding_window_tokens=hbt + tc.sconv_kernel_size,
                family="history",
                block_size=hbt,
            )
            for label in dict.fromkeys(tc.paged_cache_layer_types)
        )
    return conv_bt, hbt


def _wrap_inkling_backend(inner, text_config, attn_config, *, num_layers, is_draft):
    """Wrap a dense backend with the engine-side Inkling sconv state pool.

    The wrapper only adds conv metadata; all attention delegates to ``inner``.
    Returns ``(backend, conv_pool)``.
    """
    from tokenspeed.runtime.configs.inkling_config import inkling_conv_total_dim
    from tokenspeed.runtime.layers.attention.backends.inkling import (
        InklingAttnBackend,
        InklingConvStatePool,
    )

    conv_pool = InklingConvStatePool(
        num_layers=num_layers,
        # Row 0 is reserved (1-based indices); +2 covers it plus a padding slot
        num_slots=attn_config.max_bs + 2,
        conv_dim=inkling_conv_total_dim(text_config, attn_config.attn_tp_size),
        kernel_size=text_config.sconv_kernel_size,
        dtype=torch.bfloat16,
        device=attn_config.device,
    )
    logger.info(
        "Inkling %sconv state pool: %d layers x %d slots, %.1f MiB",
        "draft " if is_draft else "",
        num_layers,
        attn_config.max_bs + 2,
        conv_pool.mem_usage_bytes() / (1 << 20),
    )
    backend = InklingAttnBackend(
        inner,
        conv_pool,
        spec_num_tokens=getattr(attn_config, "speculative_num_draft_tokens", 1),
        is_draft=is_draft,
    )
    return backend, conv_pool


def _inkling_conv_columns(pool, text_config, conv_bt=None, hbt=None):
    """kvconv-as-swa geometry for the backend (None if no kvconv pages).

    Columns live in the layers' own K/V slots; backend gets only geometry + layer->group map.
    """
    layer_labels = text_config.paged_cache_layer_types
    memory_plan = getattr(pool, "_lcm_memory_plan", None)
    if memory_plan is not None:
        block_tokens = memory_plan.logical_block_tokens
        conv_columns = {
            "mode": "checkpoint",
            "block_tokens": block_tokens,
            "conv_group_of_layer": ("kvconv",) * len(layer_labels),
            "hidden_group_of_layer": ("hiddenconv",) * len(layer_labels),
            "group_block_tokens": {
                "kvconv": block_tokens,
                "hiddenconv": block_tokens,
            },
        }
        logger.info(
            "Inkling ShortConv boundary checkpoints: P=%d, groups=%s",
            block_tokens,
            tuple(conv_columns["group_block_tokens"]),
        )
        return conv_columns

    labels = list(dict.fromkeys(layer_labels))
    num_conv_blocks = sum(
        pool.paged_cache_group_page_counts.get(f"kvconv_{label}", 0) for label in labels
    )
    if not num_conv_blocks:
        logger.warning(
            "paged sconv expected but no kvconv pages published;"
            " rolling conv state only."
        )
        return None
    paged_hidden = True  # unconditional (gate retired 2026-07-15)
    conv_columns = {
        "block_tokens": conv_bt,
        "lcm_align": 256,
        "conv_group_of_layer": tuple(f"kvconv_{label}" for label in layer_labels),
        "hidden_block_tokens": hbt,
        "hidden_group_of_layer": (
            tuple(f"hiddenconv_{label}" for label in layer_labels)
            if paged_hidden
            else None
        ),
        # Per-group block sizes for the backend's table buffers.
        "group_block_tokens": {
            **{f"kvconv_{label}": conv_bt for label in labels},
            **(
                {f"hiddenconv_{label}": hbt for label in labels} if paged_hidden else {}
            ),
        },
    }
    logger.info(
        "Inkling kvconv-as-swa: %d groups, block %d%s",
        len(labels),
        conv_bt,
        f" + hiddenconv block {hbt}" if paged_hidden else "",
    )
    return conv_columns


def _start_inkling_pool_probe(kv_pool, conv_pool, rank, probe_dir) -> None:
    """Diagnostic (INKLING_POOL_PROBE_DIR=<dir>): background thread
    checksumming pool regions that must stay INVARIANT under traffic — the
    dummy page's tail rows and the top quarter of each layer buffer — to catch
    wrong-location KV writes in the act. One JSONL line per interval per rank;
    ~seconds of bandwidth per sweep, diagnosis only."""
    import json
    import threading
    import time

    path = f"{probe_dir}/pool_probe_rank{rank}.jsonl"

    def _pool_probe():
        while True:
            try:
                rec = {"t": time.time()}
                d0 = top = tot = 0.0
                d0_layers = []
                for lid, bufs in enumerate(zip(kv_pool.k_buffer, kv_pool.v_buffer)):
                    for tb in bufs:
                        if tb is None:
                            continue
                        n = tb.shape[0]
                        v = float(tb[4:128].float().abs().sum())
                        d0 += v
                        if v > 0.0:
                            d0_layers.append((lid, round(v, 3)))
                        top += float(tb[(3 * n) // 4 :].float().abs().sum())
                        tot += float(tb.float().abs().sum())
                rec["dummy_tail_abs"] = round(d0, 4)
                rec["dummy_tail_layers"] = d0_layers[:8]
                rec["top_quarter_abs"] = round(top, 4)
                rec["total_abs"] = round(tot, 2)
                rec["conv_abs"] = round(
                    float(conv_pool.conv_state.float().abs().sum()), 2
                )
                with open(path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception as exc:  # diagnostics must not kill serving
                with open(path, "a") as f:
                    f.write(json.dumps({"error": repr(exc)}) + "\n")
            time.sleep(10)

    threading.Thread(target=_pool_probe, daemon=True).start()
    logger.info("Inkling pool probe enabled -> %s", probe_dir)


# ---------- public API ----------
def create_attn_components(
    server_args: ServerArgs,
    model_config: ModelConfig,
    gpu_id: int,
    rank: int,
    gpu_memory: int,
    enable_memory_saver: bool = False,
    draft_model_config: ModelConfig | None = None,
    decode_input_tokens: int = 1,
    overlap_schedule_depth: int = 0,
) -> tuple[
    AttentionBackend,
    BaseTokenToKVPool,
    AttentionBackend | None,
    BaseTokenToKVPool | None,
    int,
    dict | None,
]:
    arch = model_config.attention_arch

    architectures = getattr(model_config.hf_config, "architectures", None) or []
    is_hybrid_gdn = any(a in _HYBRID_GDN_ARCHITECTURES for a in architectures)
    is_inkling = any(a in _INKLING_ARCHITECTURES for a in architectures)
    is_hybrid_mla_kda = any(a in _HYBRID_MLA_KDA_ARCHITECTURES for a in architectures)
    # Both take the hybrid-linear path; they differ only in the linear kernel
    # (GDN scalar decay vs KDA per-channel) and the base attn arch (MHA vs MLA).
    is_hybrid_linear = is_hybrid_gdn or is_hybrid_mla_kda
    is_deepseek_v4_model = is_deepseek_v4(model_config.hf_config)
    is_deepseek_v4_draft_model = draft_model_config is not None and is_deepseek_v4(
        draft_model_config.hf_config
    )
    original_attn_backend = server_args.attention_backend
    if is_deepseek_v4_model:
        server_args.attention_backend = "deepseek_v4"
    if is_deepseek_v4_draft_model:
        server_args.drafter_attention_backend = "deepseek_v4"
    if is_hybrid_linear:
        # GDN (Qwen3.5) / KDA (Kimi-K3) hybrid models always need
        # hybrid_linear_attn. Save the user's original choice for the
        # full-attention sub-backend (MHA for GDN, MLA for KDA).
        server_args.attention_backend = "hybrid_linear_attn"
    elif server_args.attention_backend == "hybrid_linear_attn":
        logger.warning(
            "Ignoring hybrid_linear_attn backend for non-hybrid model architectures=%s",
            architectures,
        )
        server_args.attention_backend = None
        if server_args.drafter_attention_backend == "hybrid_linear_attn":
            server_args.drafter_attention_backend = None

    config = _create_attn_config(server_args, model_config)
    if is_inkling and server_args.disaggregation_mode in ("prefill", "decode"):
        raise NotImplementedError("Inkling PD is not supported")
    target_text_config = getattr(
        model_config.hf_config, "text_config", model_config.hf_config
    )
    target_mamba_params = getattr(target_text_config, "mamba2_cache_params", None)
    has_state = bool(
        target_mamba_params
        and target_mamba_params[-1]
        and any(
            layer_type in STATE_LAYER_TYPES
            for layer_type in getattr(config, "layer_types", ())
        )
    )
    use_lcm_gdn = is_hybrid_gdn and has_state
    use_lcm_k3 = is_hybrid_mla_kda
    use_lcm_inkling = is_inkling
    lcm_family = (
        "qwen_gdn"
        if use_lcm_gdn
        else "kimi_k3" if use_lcm_k3 else "inkling" if use_lcm_inkling else None
    )
    if has_state and lcm_family is None:
        raise RuntimeError(
            "State cache requires an LCM layout recipe; none is "
            f"registered for architectures={architectures!r}"
        )
    target_full_attn_backend_name = (
        _resolve_hybrid_full_backend_name(
            original_attn_backend,
            is_kda=is_hybrid_mla_kda,
            has_lcm_plan=lcm_family is not None,
        )
        if is_hybrid_linear
        else config.backend_name
    )
    draft_attn_config = (
        _create_attn_config(server_args, draft_model_config, is_draft=True)
        if draft_model_config
        else None
    )
    draft_architectures = (
        getattr(draft_model_config.hf_config, "architectures", None) or []
        if draft_model_config is not None
        else []
    )
    draft_is_hybrid_gdn = any(
        architecture in _HYBRID_GDN_ARCHITECTURES
        for architecture in draft_architectures
    )
    draft_is_hybrid_mla_kda = any(
        architecture in _HYBRID_MLA_KDA_ARCHITECTURES
        for architecture in draft_architectures
    )
    draft_full_attn_backend_name = None
    if draft_attn_config is not None:
        if draft_is_hybrid_gdn or draft_is_hybrid_mla_kda:
            requested_draft_backend = (
                draft_attn_config.backend_name
                if lcm_family is not None
                else original_attn_backend
            )
            draft_full_attn_backend_name = _resolve_hybrid_full_backend_name(
                requested_draft_backend,
                is_kda=draft_is_hybrid_mla_kda,
                has_lcm_plan=lcm_family is not None,
            )
        else:
            draft_full_attn_backend_name = draft_attn_config.backend_name
    num_layers = model_config.num_attention_layers
    deepseek_v4_layout = None
    draft_deepseek_v4_layout = None
    profile_cache_cell_size = None
    draft_profile_cache_cell_size = None
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
            layer_indices=range(num_layers),
        )
        profile_cache_cell_size = deepseek_v4_layout.cache_cell_size(num_layers)
    if is_deepseek_v4_draft_model:
        from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4 import (
            deepseek_v4_cache_layout_from_config,
        )

        draft_layer_start = draft_model_config.num_hidden_layers
        draft_num_layers = draft_model_config.num_attention_layers
        draft_deepseek_v4_layout = deepseek_v4_cache_layout_from_config(
            draft_model_config.hf_config,
            page_size=server_args.block_size,
            use_fp4_indexer_cache=_attention_use_fp4_indexer_cache(
                server_args, draft_model_config.hf_config
            ),
            layer_indices=range(
                draft_layer_start,
                draft_layer_start + draft_num_layers,
            ),
        )
        draft_profile_cache_cell_size = draft_deepseek_v4_layout.cache_cell_size(
            draft_model_config.num_attention_layers
        )

    cache_budget_bytes = None
    fixed_workspace_bytes = 0
    draft_max_num_tokens = None
    lcm_setup = None

    _profile_kwargs = dict(
        attn_config=config,
        gpu_id=gpu_id,
        tp_size=server_args.mapping.world_size,
        page_size=server_args.block_size,
        num_attention_layers=num_layers,
        total_gpu_memory=gpu_memory,
        world_group=server_args.mapping.world_group,
        draft_attn_config=draft_attn_config if draft_attn_config else None,
        draft_num_attention_layers=(
            draft_model_config.num_attention_layers if draft_attn_config else None
        ),
    )

    if is_deepseek_v4_model:
        from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4 import (
            profile_deepseek_v4_max_num_pages,
        )

        draft_cache_cell_size = _resolve_draft_cache_cell_size_for_profile(
            draft_attn_config,
            draft_model_config,
            draft_profile_cache_cell_size,
        )
        max_total_num_pages = profile_deepseek_v4_max_num_pages(
            layout=deepseek_v4_layout,
            hf_config=model_config.hf_config,
            layer_num=num_layers,
            max_live_requests=config.max_bs,
            max_scheduled_tokens=server_args.chunked_prefill_size,
            max_context_len=config.context_len,
            available_cache_memory_bytes=profile_available_cache_memory_bytes(
                attn_config=config,
                gpu_id=gpu_id,
                tp_size=server_args.mapping.world_size,
                gpu_memory_utilization=server_args.gpu_memory_utilization,
                total_gpu_memory=gpu_memory,
                world_group=server_args.mapping.world_group,
            ),
            draft_cache_cell_size=draft_cache_cell_size,
            decode_input_tokens=decode_input_tokens,
            overlap_schedule_depth=overlap_schedule_depth,
        )
        logger.info(
            "DeepSeek V4 grouped KV profile: max_live_requests=%s "
            "(attn config max_bs=%s, attn_dp_size=%s), max_total_num_pages=%s",
            config.max_bs,
            config.max_bs,
            server_args.mapping.attn.dp_size,
            max_total_num_pages,
        )
        max_num_tokens = _resolve_max_num_tokens(
            max_total_num_pages,
            server_args.block_size,
            server_args.max_total_tokens,
        )
    elif lcm_family is not None:
        cache_memory = profile_available_cache_memory_bytes(
            attn_config=config,
            gpu_id=gpu_id,
            tp_size=server_args.mapping.world_size,
            gpu_memory_utilization=server_args.gpu_memory_utilization,
            total_gpu_memory=gpu_memory,
            world_group=server_args.mapping.world_group,
        )
        lcm_setup = prepare_lcm_setup(
            family=lcm_family,
            server_args=server_args,
            model_config=model_config,
            attn_config=config,
            draft_model_config=draft_model_config,
            draft_attn_config=draft_attn_config,
            cache_budget_bytes=cache_memory,
            decode_input_tokens=decode_input_tokens,
            overlap_schedule_depth=overlap_schedule_depth,
        )
        logical_page_size = lcm_setup.target.memory_plan.logical_block_tokens
        _validate_lcm_page_size(
            config,
            logical_page_size=logical_page_size,
        )
        if draft_attn_config is not None:
            _validate_lcm_page_size(
                draft_attn_config,
                logical_page_size=logical_page_size,
            )
        cache_budget_bytes = lcm_setup.cache_budget_bytes
        fixed_workspace_bytes = lcm_setup.fixed_workspace_bytes
        max_num_tokens = lcm_setup.target.pool_size
        draft_max_num_tokens = (
            lcm_setup.draft.pool_size if lcm_setup.draft is not None else max_num_tokens
        )
        max_total_num_pages = lcm_setup.target.memory_plan.num_lcm_blocks
        logger.info(
            "LCM profile: target_parent_bytes=%d, draft_parent_bytes=%d, "
            "P=%d, parents=%d, token_capacity=%d, groups=%s",
            lcm_setup.target.memory_plan.lcm_block_bytes,
            (
                lcm_setup.draft.memory_plan.lcm_block_bytes
                if lcm_setup.draft is not None
                else 0
            ),
            lcm_setup.target.memory_plan.logical_block_tokens,
            lcm_setup.target.memory_plan.num_lcm_blocks,
            lcm_setup.target.token_capacity,
            {
                group.group_id: group.cache_blocks_per_lcm_block
                for group in lcm_setup.target.memory_plan.groups
            },
        )
    else:
        # config.layer_types / config.sliding_window_tokens are the exact
        # values forwarded to the KV pool, so sizing and layout consume
        # identical inputs (MLA configs carry neither -> legacy divisor).
        slab_divisor = _kv_profile_layer_divisor(
            num_layers,
            getattr(config, "layer_types", None),
            sliding_window_tokens=getattr(config, "sliding_window_tokens", None),
        )
        if profile_cache_cell_size is not None and slab_divisor != num_layers:
            # A cell-size override can't compose with the slab divisor.
            logger.warning(
                "hybrid slab sizing disabled: profile cache_cell_size "
                "override is set; charging all %d layers instead of %d",
                num_layers,
                slab_divisor,
            )
            slab_divisor = num_layers
        max_total_num_pages = profile_max_num_pages(
            **{**_profile_kwargs, "num_attention_layers": slab_divisor},
            gpu_memory_utilization=server_args.gpu_memory_utilization,
            cache_cell_size=profile_cache_cell_size,
            draft_cache_cell_size=draft_profile_cache_cell_size,
        )
        max_num_tokens = _resolve_max_num_tokens(
            max_total_num_pages,
            server_args.block_size,
            server_args.max_total_tokens,
        )

    if (
        lcm_setup is None
        and _CI_SMALL_KV_SIZE is not None
        and int(_CI_SMALL_KV_SIZE) > 0
    ):
        # LCM plans already folded the CI cap into their physical geometry;
        # clobbering here would desynchronize tokens from that plan.
        max_num_tokens = int(_CI_SMALL_KV_SIZE)
    if draft_max_num_tokens is None:
        draft_max_num_tokens = max_num_tokens
    if max_num_tokens <= 0:
        raise ValueError(
            f"KV cache token pool size must be positive, got {max_num_tokens}"
        )

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
            hf_config=model_config.hf_config,
            max_scheduled_tokens=server_args.chunked_prefill_size,
            decode_input_tokens=decode_input_tokens,
            overlap_schedule_depth=overlap_schedule_depth,
        )
    elif is_hybrid_linear:
        backend, pool = _create_hybrid_linear_attn(
            server_args,
            model_config,
            config,
            arch,
            max_num_tokens,
            rank,
            enable_memory_saver,
            full_attn_backend_name=target_full_attn_backend_name,
            is_kda=is_hybrid_mla_kda,
            lcm_spec=lcm_setup.target if lcm_setup is not None else None,
        )
    else:
        # Hetero KV + paged sconv are unconditional for Inkling (the
        # INKLING_HETERO_KV / INKLING_PAGED_SCONV bisection gates were
        # retired 2026-07-15 after the experiments settled).
        paged_sconv = is_inkling
        if is_inkling and not use_lcm_inkling:
            _apply_inkling_hetero_kv(config, model_config)
            # Must run AFTER the hetero fields exist on the config.
            max_num_tokens = _floor_tokens_to_layout_grid(max_num_tokens, config)
        if paged_sconv and not use_lcm_inkling:
            _conv_bt, _hbt = _publish_inkling_conv_groups(
                config, model_config, server_args
            )
        backend = _create_attn_backend(arch, config)
        pool = (
            create_lcm_pool(
                lcm_setup.target,
                config,
                num_layers=num_layers,
                rank=rank,
                enable_memory_saver=enable_memory_saver,
            )
            if lcm_setup is not None
            else _create_attn_pool(
                config, num_layers, max_num_tokens, rank, enable_memory_saver
            )
        )
        if is_inkling:
            # Wrapper only adds sconv state keyed on req_pool_indices; attention stays on the dense backend
            text_config = model_config.hf_config.get_text_config()
            backend, conv_pool = _wrap_inkling_backend(
                backend,
                text_config,
                config,
                num_layers=text_config.num_hidden_layers,
                is_draft=False,
            )
            _probe_dir = os.environ.get("INKLING_POOL_PROBE_DIR")
            if _probe_dir:
                _start_inkling_pool_probe(pool, conv_pool, rank, _probe_dir)
            conv_columns = None
            if paged_sconv:
                conv_columns = _inkling_conv_columns(
                    pool,
                    text_config,
                    _conv_bt if not use_lcm_inkling else None,
                    _hbt if not use_lcm_inkling else None,
                )
            backend.conv_columns = conv_columns
            if conv_columns is not None:
                # Wrapper-owned conv groups: the attention mixin skips their write-loc math and capture fills
                backend.inner.engine_owned_group_ids = frozenset(
                    conv_columns["group_block_tokens"]
                )
    draft_attn_backend = None
    draft_pool = None
    if draft_attn_config:
        # Check if draft model is also a hybrid GDN model.
        if is_deepseek_v4_draft_model:
            from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4 import (
                DeepseekV4TokenToKVPool,
            )

            draft_attn_backend = _create_attn_backend(
                draft_model_config.attention_arch, draft_attn_config
            )
            draft_pool = DeepseekV4TokenToKVPool(
                size=draft_max_num_tokens,
                model_dtype=draft_model_config.dtype,
                layout=draft_deepseek_v4_layout,
                layer_num=draft_model_config.num_attention_layers,
                device=draft_attn_config.device,
                enable_memory_saver=enable_memory_saver,
                max_batch_size=draft_attn_config.max_bs,
                max_context_len=draft_attn_config.context_len,
                page_size=server_args.block_size,
                rank=rank,
                hf_config=draft_model_config.hf_config,
                max_scheduled_tokens=server_args.chunked_prefill_size,
                decode_input_tokens=decode_input_tokens,
                overlap_schedule_depth=overlap_schedule_depth,
            )
        elif draft_is_hybrid_gdn:
            draft_attn_backend, draft_pool = _create_hybrid_linear_attn(
                server_args,
                draft_model_config,
                draft_attn_config,
                draft_model_config.attention_arch,
                draft_max_num_tokens,
                rank,
                enable_memory_saver,
                full_attn_backend_name=draft_full_attn_backend_name,
                lcm_spec=lcm_setup.draft if lcm_setup is not None else None,
            )
        elif any(a in _INKLING_ARCHITECTURES for a in draft_architectures):
            draft_text_config = draft_model_config.hf_config.get_text_config()
            num_depths = draft_model_config.num_attention_layers
            # Hetero KV, symmetric with the target pool: per-depth head
            # counts come from the MTP text config's own local ids
            # (ModelConfig swapped it in for the draft worker), giving the
            # draft the same byte-uniform slot layout as the target. The pool
            # shares the TARGET's page-id space (the drafter consumes the
            # target's per-group group tables), so it is sized at the same
            # max_num_tokens; per-layer views turn the max-head allocation
            # into the group geometry, and the profile's max-head draft charge
            # prices exactly that allocation.
            if not use_lcm_inkling:
                _apply_inkling_hetero_kv(draft_attn_config, draft_model_config)
            logger.info(
                "Inkling MTP draft pool: hetero KV layer head counts=%s, "
                "layer types=%s (%d depths, %d ids)",
                (
                    lcm_setup.draft.layer_kv_head_counts
                    if lcm_setup is not None and lcm_setup.draft is not None
                    else draft_attn_config.layer_kv_head_counts
                ),
                (
                    lcm_setup.draft.layer_types
                    if lcm_setup is not None and lcm_setup.draft is not None
                    else draft_attn_config.layer_types
                ),
                num_depths,
                draft_max_num_tokens // config.page_size,
            )
            draft_attn_backend = _create_attn_backend(
                draft_model_config.attention_arch, draft_attn_config
            )
            draft_pool = (
                create_lcm_pool(
                    lcm_setup.draft,
                    draft_attn_config,
                    num_layers=num_depths,
                    rank=rank,
                    enable_memory_saver=enable_memory_saver,
                )
                if lcm_setup is not None and lcm_setup.draft is not None
                else _create_attn_pool(
                    draft_attn_config,
                    num_depths,
                    draft_max_num_tokens,
                    rank,
                    enable_memory_saver,
                )
            )
            draft_attn_backend, _ = _wrap_inkling_backend(
                draft_attn_backend,
                draft_text_config,
                draft_attn_config,
                num_layers=num_depths,
                is_draft=True,
            )
        else:
            draft_attn_backend = _create_attn_backend(
                draft_model_config.attention_arch, draft_attn_config
            )
            draft_layers = draft_model_config.num_attention_layers
            draft_pool = (
                create_lcm_pool(
                    lcm_setup.draft,
                    draft_attn_config,
                    num_layers=draft_layers,
                    rank=rank,
                    enable_memory_saver=enable_memory_saver,
                )
                if lcm_setup is not None and lcm_setup.draft is not None
                else _create_attn_pool(
                    draft_attn_config,
                    draft_layers,
                    draft_max_num_tokens,
                    rank,
                    enable_memory_saver,
                )
            )

    _validate_shared_lcm_geometry(pool, draft_pool)
    if use_lcm_gdn and fixed_workspace_bytes:
        actual_workspace_bytes = (
            backend.linear_attn_backend.preallocate_verify_workspace(
                config.max_bs,
                int(server_args.speculative_num_draft_tokens),
            )
        )
        if actual_workspace_bytes != fixed_workspace_bytes:
            raise RuntimeError(
                "planned GDN verify workspace does not match allocated tensors: "
                f"{fixed_workspace_bytes} planned, {actual_workspace_bytes} allocated"
            )
    elif use_lcm_inkling:
        backend.preallocate_verify_workspace(config.max_bs)
        if draft_attn_backend is not None:
            lookback = (
                int(server_args.speculative_num_steps) - 1
                if int(server_args.speculative_num_steps) > 1
                and os.environ.get("INKLING_MTP_DECODE_LOOKBACK", "1") != "0"
                else 0
            )
            if lookback and not draft_attn_backend.configure_draft_lookback(lookback):
                raise RuntimeError("Inkling MTP draft rejected its planned lookback")
        actual_workspace_bytes = backend.fixed_workspace_bytes()
        if draft_attn_backend is not None:
            actual_workspace_bytes += draft_attn_backend.fixed_workspace_bytes()
        if actual_workspace_bytes != fixed_workspace_bytes:
            raise RuntimeError(
                "planned Inkling workspace does not match allocated tensors: "
                f"{fixed_workspace_bytes} planned, {actual_workspace_bytes} allocated"
            )

    cache_storage = None
    if lcm_setup is not None:
        if cache_budget_bytes is None:
            raise RuntimeError("LCM cache profile did not record its byte budget")
        cache_storage = _cache_storage_report(
            configured_cache_bytes=cache_budget_bytes,
            pool=pool,
            draft_pool=draft_pool,
            fixed_workspace_bytes=fixed_workspace_bytes,
        )

    return (
        backend,
        pool,
        draft_attn_backend,
        draft_pool,
        max_num_tokens,
        cache_storage,
    )
