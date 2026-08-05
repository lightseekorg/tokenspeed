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

import inspect
import logging
import os
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.configs.model_config import AttentionArch, is_deepseek_v4
from tokenspeed.runtime.configs.paged_cache_spec import STATE_LAYER_TYPES
from tokenspeed.runtime.layers.attention.configs.base import BaseAttnConfig
from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.configs.msa import (
    MSAConfig,
)
from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
from tokenspeed.runtime.layers.attention.kv_cache.factory import create_cache_pool
from tokenspeed.runtime.layers.attention.kv_cache.setup import (
    CachePoolSpec,
    prepare_cache_setup,
)
from tokenspeed.runtime.layers.attention.utils import (
    profile_available_cache_memory_bytes,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tokenspeed.runtime.configs.model_config import ModelConfig
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.utils.server_args import ServerArgs


def _pool_allocated_bytes(pool) -> int:
    if pool is None:
        return 0
    buffer = getattr(pool, "buffer", None)
    if buffer is not None:
        return int(buffer.nbytes)
    sizes = pool.get_kv_size_bytes()
    if isinstance(sizes, tuple):
        return sum(int(size) for size in sizes)
    return int(sizes)


def _validate_shared_cache_geometry(pool, draft_pool) -> None:
    """Validate the draft's independent arena against target-owned tables."""
    if draft_pool is None:
        return
    draft_contract = getattr(draft_pool, "runtime_contract", None)
    if draft_contract is None:
        return
    if getattr(pool, "runtime_contract", None) is None:
        raise RuntimeError(
            "a draft pool with shared scheduler geometry requires a matching "
            "target pool"
        )
    target_plan = getattr(pool, "plan", None)
    draft_plan = getattr(draft_pool, "plan", None)
    if target_plan is None or draft_plan is None:
        raise RuntimeError("shared scheduler geometry requires memory plans")
    if (
        draft_plan.logical_block_tokens != target_plan.logical_block_tokens
        or draft_plan.num_lcm_blocks != target_plan.num_lcm_blocks
    ):
        raise RuntimeError("target and draft parent geometry does not match")

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

    target_buffer = getattr(pool, "buffer", None)
    draft_buffer = getattr(draft_pool, "buffer", None)
    if target_buffer is None or draft_buffer is None:
        raise RuntimeError("target and draft pools must allocate their buffers")
    if (
        target_buffer.untyped_storage().data_ptr()
        == draft_buffer.untyped_storage().data_ptr()
    ):
        raise RuntimeError("target and draft pools must not share a buffer")


def _cache_storage_report(
    *,
    configured_cache_bytes: int,
    pool,
    draft_pool,
    fixed_workspace_bytes: int = 0,
) -> dict:
    """Describe cache storage from allocated tensors, not scheduler counts."""
    plan = getattr(pool, "plan", None)
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


def _set_cache_group_page_sizes(config: BaseAttnConfig, spec: CachePoolSpec) -> None:
    """Publish scheduler page sizes to group-aware MHA backends."""
    if not hasattr(config, "group_page_sizes"):
        return
    page_size = spec.memory_plan.logical_block_tokens
    config.group_page_sizes = {group_id: page_size for group_id in spec.layer_group_ids}


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
    has_cache_plan: bool,
) -> str | None:
    """Resolve the compute backend that consumes the hybrid history cache."""
    name = _BACKEND_ALIASES.get(requested_name, requested_name)
    if name == "hybrid_linear_attn":
        name = None
    # NVIDIA K3 defaults to its CuteDSL history consumer. AMD keeps the
    # generic MLA backend; explicit user choices remain authoritative.
    if has_cache_plan and is_kda and name is None and not current_platform().is_amd:
        return "tokenspeed_mla"
    return name


def _create_hybrid_linear_attn(
    server_args: ServerArgs,
    model_config: ModelConfig,
    config: BaseAttnConfig,
    arch: AttentionArch,
    rank: int,
    *,
    cache_spec: CachePoolSpec,
    enable_memory_saver: bool = False,
    full_attn_backend_name: str | None = None,
    is_kda: bool = False,
) -> tuple[AttentionBackend, CachePool]:
    """Create a hybrid backend + pool for a linear-attention model.

    GDN (Qwen3.5, MHA base) or, when ``is_kda`` is set, KDA (Kimi-K3,
    MLA base). The concrete pool owns one byte buffer and publishes the
    per-group tables consumed by both sub-backends.
    """
    from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (
        HybridLinearAttnBackend,
        LayerMappedKVPool,
        MambaAttnBackend,
    )

    hf_config = model_config.hf_config
    text_config = getattr(hf_config, "text_config", hf_config)
    full_attn_layers = text_config.full_attention_layer_ids
    # Create the full attention backend for standard MHA layers.
    # Use user's original choice if provided, otherwise auto-select.
    full_attn_backend = _create_attn_backend_with_name(
        full_attn_backend_name,
        arch,
        config,
    )

    if is_kda:
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
    inner_pool = create_cache_pool(
        cache_spec,
        config,
        num_layers=num_total_layers,
        rank=rank,
        enable_memory_saver=enable_memory_saver,
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


def _inkling_conv_columns(pool, text_config):
    """Return the ShortConv checkpoint groups backed by the cache plan."""
    layer_labels = text_config.paged_cache_layer_types
    block_tokens = pool.plan.logical_block_tokens
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
            except Exception as exc:  # noqa: BLE001 - diagnostics must not kill serving
                with open(path, "a") as f:
                    f.write(json.dumps({"error": repr(exc)}) + "\n")
            time.sleep(10)

    threading.Thread(target=_pool_probe, daemon=True).start()
    logger.info("Inkling pool probe enabled -> %s", probe_dir)


def _create_target_components(
    *,
    server_args,
    model_config,
    config,
    cache_spec: CachePoolSpec,
    rank: int,
    enable_memory_saver: bool,
    full_attn_backend_name: str | None,
    is_hybrid_linear: bool,
    is_kda: bool,
    is_inkling: bool,
):
    if is_hybrid_linear:
        return _create_hybrid_linear_attn(
            server_args,
            model_config,
            config,
            model_config.attention_arch,
            rank,
            cache_spec=cache_spec,
            enable_memory_saver=enable_memory_saver,
            full_attn_backend_name=full_attn_backend_name,
            is_kda=is_kda,
        )

    backend = _create_attn_backend(model_config.attention_arch, config)
    pool = create_cache_pool(
        cache_spec,
        config,
        num_layers=model_config.num_attention_layers,
        rank=rank,
        enable_memory_saver=enable_memory_saver,
    )
    if not is_inkling:
        return backend, pool

    text_config = model_config.hf_config.get_text_config()
    backend, conv_pool = _wrap_inkling_backend(
        backend,
        text_config,
        config,
        num_layers=text_config.num_hidden_layers,
        is_draft=False,
    )
    probe_dir = os.environ.get("INKLING_POOL_PROBE_DIR")
    if probe_dir:
        _start_inkling_pool_probe(pool, conv_pool, rank, probe_dir)
    conv_columns = _inkling_conv_columns(pool, text_config)
    backend.conv_columns = conv_columns
    backend.inner.engine_owned_group_ids = frozenset(conv_columns["group_block_tokens"])
    return backend, pool


def _create_draft_components(
    *,
    server_args,
    model_config,
    config,
    cache_spec: CachePoolSpec | None,
    rank: int,
    enable_memory_saver: bool,
    full_attn_backend_name: str | None,
    is_hybrid_linear: bool,
    is_kda: bool,
    is_inkling: bool,
):
    if config is None:
        return None, None
    if cache_spec is None:
        raise RuntimeError("draft attention requires a draft cache spec")
    if is_hybrid_linear:
        return _create_hybrid_linear_attn(
            server_args,
            model_config,
            config,
            model_config.attention_arch,
            rank,
            cache_spec=cache_spec,
            enable_memory_saver=enable_memory_saver,
            full_attn_backend_name=full_attn_backend_name,
            is_kda=is_kda,
        )

    backend = _create_attn_backend(model_config.attention_arch, config)
    num_layers = model_config.num_attention_layers
    pool = create_cache_pool(
        cache_spec,
        config,
        num_layers=num_layers,
        rank=rank,
        enable_memory_saver=enable_memory_saver,
    )
    if not is_inkling:
        return backend, pool

    logger.info(
        "Inkling MTP draft pool: hetero KV layer head counts=%s, "
        "layer types=%s (%d depths, %d ids)",
        cache_spec.layer_kv_head_counts,
        cache_spec.layer_types,
        num_layers,
        cache_spec.pool_size // cache_spec.memory_plan.logical_block_tokens,
    )
    text_config = model_config.hf_config.get_text_config()
    backend, _ = _wrap_inkling_backend(
        backend,
        text_config,
        config,
        num_layers=num_layers,
        is_draft=True,
    )
    return backend, pool


def _prepare_verify_workspace(
    *,
    server_args,
    config,
    backend,
    draft_backend,
    is_qwen_gdn: bool,
    is_inkling: bool,
    expected_bytes: int,
) -> None:
    if is_qwen_gdn and expected_bytes:
        model_name = "GDN"
        actual_bytes = backend.linear_attn_backend.preallocate_verify_workspace(
            config.max_bs,
            int(server_args.speculative_num_draft_tokens),
        )
    elif is_inkling:
        model_name = "Inkling"
        backend.preallocate_verify_workspace(config.max_bs)
        if draft_backend is not None:
            lookback = (
                int(server_args.speculative_num_steps) - 1
                if int(server_args.speculative_num_steps) > 1
                and os.environ.get("INKLING_MTP_DECODE_LOOKBACK", "1") != "0"
                else 0
            )
            if lookback and not draft_backend.configure_draft_lookback(lookback):
                raise RuntimeError("Inkling MTP draft rejected its planned lookback")
        actual_bytes = backend.fixed_workspace_bytes()
        if draft_backend is not None:
            actual_bytes += draft_backend.fixed_workspace_bytes()
    else:
        return
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"planned {model_name} verify workspace does not match allocated tensors: "
            f"{expected_bytes} planned, {actual_bytes} allocated"
        )


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
    CachePool,
    AttentionBackend | None,
    CachePool | None,
    int,
    dict | None,
]:
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
    if is_deepseek_v4_model:
        config.sliding_window_tokens = int(model_config.hf_config.sliding_window)
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
    use_cache_gdn = is_hybrid_gdn and has_state
    use_cache_k3 = is_hybrid_mla_kda
    use_cache_inkling = is_inkling
    if is_deepseek_v4_model:
        cache_family = "deepseek_v4"
    elif use_cache_gdn:
        cache_family = "qwen_gdn"
    elif use_cache_k3:
        cache_family = "kimi_k3"
    elif use_cache_inkling:
        cache_family = "inkling"
    elif type(config) is MHAConfig:
        cache_family = "mha"
    elif type(config) is MLAConfig:
        cache_family = "mla"
    elif isinstance(config, DSAConfig):
        cache_family = "dsa"
    elif isinstance(config, MSAConfig):
        cache_family = "msa"
    else:
        cache_family = None
    if cache_family is None:
        raise RuntimeError(
            "No cache recipe is registered for "
            f"attention config {type(config).__name__}"
        )
    target_full_attn_backend_name = (
        _resolve_hybrid_full_backend_name(
            original_attn_backend,
            is_kda=is_hybrid_mla_kda,
            has_cache_plan=True,
        )
        if is_hybrid_linear
        else config.backend_name
    )
    draft_attn_config = (
        _create_attn_config(server_args, draft_model_config, is_draft=True)
        if draft_model_config
        else None
    )
    if is_deepseek_v4_draft_model:
        draft_attn_config.sliding_window_tokens = int(
            draft_model_config.hf_config.sliding_window
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
            draft_full_attn_backend_name = _resolve_hybrid_full_backend_name(
                draft_attn_config.backend_name,
                is_kda=draft_is_hybrid_mla_kda,
                has_cache_plan=True,
            )
        else:
            draft_full_attn_backend_name = draft_attn_config.backend_name
    cache_memory = profile_available_cache_memory_bytes(
        attn_config=config,
        gpu_id=gpu_id,
        tp_size=server_args.mapping.world_size,
        gpu_memory_utilization=server_args.gpu_memory_utilization,
        total_gpu_memory=gpu_memory,
        world_group=server_args.mapping.world_group,
    )
    cache_setup = prepare_cache_setup(
        family=cache_family,
        server_args=server_args,
        model_config=model_config,
        attn_config=config,
        draft_model_config=draft_model_config,
        draft_attn_config=draft_attn_config,
        cache_budget_bytes=cache_memory,
        decode_input_tokens=decode_input_tokens,
        overlap_schedule_depth=overlap_schedule_depth,
    )
    logical_page_size = cache_setup.target.memory_plan.logical_block_tokens
    _validate_lcm_page_size(
        config,
        logical_page_size=logical_page_size,
    )
    _set_cache_group_page_sizes(config, cache_setup.target)
    if draft_attn_config is not None:
        _validate_lcm_page_size(
            draft_attn_config,
            logical_page_size=logical_page_size,
        )
        if cache_setup.draft is not None:
            _set_cache_group_page_sizes(draft_attn_config, cache_setup.draft)
    cache_budget_bytes = cache_setup.cache_budget_bytes
    fixed_workspace_bytes = cache_setup.fixed_workspace_bytes
    max_num_tokens = cache_setup.target.pool_size
    logger.info(
        "Cache profile: target_parent_bytes=%d, draft_parent_bytes=%d, "
        "P=%d, parents=%d, token_capacity=%d, groups=%s",
        cache_setup.target.memory_plan.lcm_block_bytes,
        (
            cache_setup.draft.memory_plan.lcm_block_bytes
            if cache_setup.draft is not None
            else 0
        ),
        cache_setup.target.memory_plan.logical_block_tokens,
        cache_setup.target.memory_plan.num_lcm_blocks,
        cache_setup.target.token_capacity,
        {
            group.group_id: group.cache_blocks_per_lcm_block
            for group in cache_setup.target.memory_plan.groups
        },
    )

    if max_num_tokens <= 0:
        raise ValueError(
            f"KV cache token pool size must be positive, got {max_num_tokens}"
        )

    backend, pool = _create_target_components(
        server_args=server_args,
        model_config=model_config,
        config=config,
        cache_spec=cache_setup.target,
        rank=rank,
        enable_memory_saver=enable_memory_saver,
        full_attn_backend_name=target_full_attn_backend_name,
        is_hybrid_linear=is_hybrid_linear,
        is_kda=is_hybrid_mla_kda,
        is_inkling=is_inkling,
    )
    draft_attn_backend, draft_pool = _create_draft_components(
        server_args=server_args,
        model_config=draft_model_config,
        config=draft_attn_config,
        cache_spec=cache_setup.draft,
        rank=rank,
        enable_memory_saver=enable_memory_saver,
        full_attn_backend_name=draft_full_attn_backend_name,
        is_hybrid_linear=draft_is_hybrid_gdn or draft_is_hybrid_mla_kda,
        is_kda=draft_is_hybrid_mla_kda,
        is_inkling=any(a in _INKLING_ARCHITECTURES for a in draft_architectures),
    )

    backend.set_cache_pool(pool)
    # A cache-group contract backend needs the contract marked before CUDA-graph
    # state allocation (mark_cache_contract sizes the per-group write-location
    # buffer). Composite/wrapper backends without the hook are a no-op.
    if getattr(pool, "runtime_contract", None) is not None:
        target_mark_contract = getattr(backend, "mark_cache_contract", None)
        if target_mark_contract is not None:
            params = inspect.signature(target_mark_contract).parameters
            if "logical_page_size" in params:
                target_mark_contract(logical_page_size=pool.plan.logical_block_tokens)
            else:
                target_mark_contract()
    if draft_attn_backend is not None and draft_pool is not None:
        draft_attn_backend.set_cache_pool(draft_pool)
        if getattr(draft_pool, "runtime_contract", None) is not None:
            mark_cache_contract = getattr(
                draft_attn_backend, "mark_cache_contract", None
            )
            if mark_cache_contract is not None:
                params = inspect.signature(mark_cache_contract).parameters
                if "logical_page_size" in params:
                    mark_cache_contract(
                        logical_page_size=draft_pool.plan.logical_block_tokens
                    )
                else:
                    mark_cache_contract()

    _validate_shared_cache_geometry(pool, draft_pool)
    _prepare_verify_workspace(
        server_args=server_args,
        config=config,
        backend=backend,
        draft_backend=draft_attn_backend,
        is_qwen_gdn=use_cache_gdn,
        is_inkling=use_cache_inkling,
        expected_bytes=fixed_workspace_bytes,
    )

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
