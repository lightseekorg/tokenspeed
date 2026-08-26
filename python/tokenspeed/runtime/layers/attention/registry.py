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
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.configs.model_config import (
    AttentionArch,
    is_deepseek_v4,
    is_qwen4_exp,
)
from tokenspeed.runtime.layers.attention.configs.base import BaseAttnConfig
from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.configs.msa import (
    MSAConfig,
)
from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.base import (
    CachePool,
)
from tokenspeed.runtime.layers.attention.kv_cache.factory import (
    create_cache_arena,
    create_cache_pool,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import (
    CacheModelFamily,
    CachePoolSpec,
    prepare_cache_setup,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    STATE_LAYER_TYPES,
)
from tokenspeed.runtime.layers.attention.utils import (
    profile_available_cache_memory_bytes,
)

logger = logging.getLogger(__name__)

_ORDINARY_CACHE_FAMILIES = frozenset({"mha", "mla", "dsa", "msa"})

if TYPE_CHECKING:
    from tokenspeed.runtime.configs.model_config import ModelConfig
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.utils.server_args import ServerArgs


def _ordinary_cache_family(config: BaseAttnConfig | None) -> CacheModelFamily | None:
    if type(config) is MHAConfig:
        return "mha"
    if type(config) is MLAConfig:
        return "mla"
    if isinstance(config, DSAConfig):
        return "dsa"
    if isinstance(config, MSAConfig):
        return "msa"
    return None


def _resolve_heterogeneous_draft_family(
    target_family: CacheModelFamily,
    draft_family: CacheModelFamily | None,
) -> CacheModelFamily | None:
    """Validate and return the supported heterogeneous draft family."""
    if draft_family is None:
        return None
    if target_family == "kimi_k3":
        if draft_family != "mla":
            raise RuntimeError(
                "Kimi-K3 unified cache currently requires an ordinary MLA draft view"
            )
        return draft_family
    if target_family not in _ORDINARY_CACHE_FAMILIES or draft_family == target_family:
        return None
    if draft_family != "mha":
        raise RuntimeError(
            "heterogeneous ordinary cache views currently require an MHA draft"
        )
    return draft_family


def _arena_allocated_bytes(arena) -> int:
    """Bytes this model's cache actually occupies: the one arena allocation.

    Summing a pool's per-layer view sizes would answer a different question
    (and double-count aliased views), so read the owner directly.
    """
    return int(arena.buffer.nbytes)


def _cache_storage_report(
    *,
    configured_cache_bytes: int,
    pool,
    fixed_workspace_bytes: int = 0,
) -> dict:
    """Describe cache storage from allocated tensors, not scheduler counts."""
    arena = getattr(pool, "arena", None)
    plan = getattr(arena, "plan", None)
    if plan is None:
        raise RuntimeError("cache pool has no memory plan; every pool is LCM-planned")
    packing = {
        group.group_id: int(group.cache_blocks_per_lcm_block) for group in plan.groups
    }
    # Admission capacity is a scheduler-contract fact. ``arena.size`` is the
    # most finely packed group's addressable row span and can exceed it for
    # heterogeneous/Kimi layouts (especially under DCP).
    token_capacity = int(arena.runtime_contract.token_capacity)
    geometry = {
        "prefix_granularity": int(plan.prefix_granularity),
        "num_lcm_blocks": int(plan.num_lcm_blocks),
        "max_addressable_row_span": int(arena.size),
        "cache_blocks_per_lcm_block": packing,
        # Fraction of a parent each group's binding actually uses;
        # aliased slabs are sized by their widest tenant, so a narrow
        # binding strands the rest.
        "binding_utilization": {
            group_id: round(entry["binding_utilization"], 4)
            for group_id, entry in plan.capacity_report().items()
        },
    }

    # One arena: the draft view shares this allocation, so it already covers
    # both models' layers.
    arena_bytes = _arena_allocated_bytes(arena)
    allocated_cache_bytes = arena_bytes + fixed_workspace_bytes
    if allocated_cache_bytes > configured_cache_bytes:
        raise RuntimeError(
            "allocated cache storage exceeds its profiled budget: "
            f"{allocated_cache_bytes} > {configured_cache_bytes}"
        )
    return {
        "configured_cache_bytes": int(configured_cache_bytes),
        "allocated_cache_bytes": allocated_cache_bytes,
        "physical_token_capacity": token_capacity,
        "capacity_source": "scheduler_contract",
        "geometry": geometry
        | {
            "arena_bytes": arena_bytes,
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


def _validate_dcp_target(
    config: BaseAttnConfig,
    *,
    is_deepseek_v4_model: bool,
    full_attn_backend_name: str,
) -> None:
    if type(config) is not MLAConfig:
        raise ValueError(
            "decode context parallelism is implemented only for dense MLA "
            f"targets, got {type(config).__name__}"
        )
    if is_deepseek_v4_model:
        raise ValueError(
            "DeepSeek V4 decode context parallelism is not supported: its "
            "cyclic indexer cache requires a distributed global sparse-indexer "
            "top-k before attention outputs can match TP"
        )
    if full_attn_backend_name != "tokenspeed_mla":
        raise ValueError(
            "decode context parallelism requires the resolved full-attention "
            "backend to be 'tokenspeed_mla', got "
            f"{full_attn_backend_name!r}"
        )


_HYBRID_GDN_ARCHITECTURES = {
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5MoeForConditionalGenerationNextN",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5ForConditionalGenerationNextN",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForCausalLMNextN",
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForCausalLMNextN",
}
# Hybrid linear-attention models whose full-attention layers are MLA (not MHA)
# and whose linear layers are KDA (per-channel gated delta rule), not GDN.
# They share the same HybridLinearAttnBackend wrapper and cache-group pool;
# the base sub-backend auto-resolves to MLA from the arch, and the linear
# sub-backend runs the KDA kernels (KdaAttnBackend).
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
    prefix_granularity: int,
) -> None:
    """Require the scheduler page to contain whole configured kernel pages.

    An unset kernel_page_size means the backend resolves its registry
    default itself and owns the divisibility check for it.
    """
    if config.kernel_page_size is None:
        return
    kernel_page_size = int(config.kernel_page_size)
    if (
        prefix_granularity <= 0
        or kernel_page_size <= 0
        or prefix_granularity % kernel_page_size
    ):
        raise ValueError(
            "prefix granularity must be a positive multiple of kernel page "
            f"size, got {prefix_granularity} and {kernel_page_size}"
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


def _resolve_kda_backend(kda_backend: str) -> str:
    """Resolve the KDA prefill backend policy.

    On AMD, the backend policy is ignored and compatible kernels are selected
    using registry priority. On NVIDIA, ``auto`` picks the fastest available
    kernel — ``cutedsl_kda``, then ``flashkda``, falling back to the portable
    FLA scan. Explicit NVIDIA choices are validated against availability and
    fail fast with an install hint. Decode is unaffected.
    """
    if current_platform().is_amd:
        # Named backend policies are NVIDIA-specific; let the registry decide.
        return "auto"

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


def _create_hybrid_linear_attn_backend(
    server_args: ServerArgs,
    model_config: ModelConfig,
    config: BaseAttnConfig,
    *,
    pool,
    full_attn_backend_name: str | None = None,
    is_kda: bool = False,
) -> AttentionBackend:
    """Create a hybrid backend for a linear-attention model over one pool.

    GDN (Qwen3.5, MHA base) or, when ``is_kda`` is set, KDA (Kimi-K3,
    MLA base). ``pool`` is the model's layer-mapped view over the one
    shared cache pool; both sub-backends consume its per-group tables.
    """
    from tokenspeed.runtime.layers.attention.backends.hybrid_kda import (
        HybridKDABackend,
        KdaAttnBackend,
    )
    from tokenspeed.runtime.layers.attention.backends.hybrid_linear_attn import (
        HybridLinearAttnBackend,
        MambaAttnBackend,
    )

    hf_config = model_config.hf_config
    text_config = getattr(hf_config, "text_config", hf_config)
    full_attn_layers = text_config.full_attention_layer_ids
    # Create the full attention backend for standard MHA layers.
    # Use user's original choice if provided, otherwise auto-select.
    full_attn_backend = _create_attn_backend_with_name(
        full_attn_backend_name,
        model_config.attention_arch,
        config,
    )

    if is_kda:
        # Cache contract: see CuteDSLMLABackend.mark_cache_contract.
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

    # Read mamba2_cache_params to decide whether this model actually has
    # any linear / mamba layers. A draft model on a hybrid-GDN target
    # (e.g. MTP on Qwen3.5) shares the same architecture class as the
    # target but commonly ships with *zero* mamba layers — in that case
    # we skip the mamba backend entirely so that its
    # ``init_forward_metadata_*`` hooks do not run (they would otherwise
    # touch a zero-sized pool on the same persistent state_indices_list
    # as the target, which breaks the captured CUDA graph).
    mamba_layer_ids = text_config.mamba2_cache_params[-1]

    if len(mamba_layer_ids) == 0:
        logger.info(
            "Created hybrid_linear_attn backend: %d full attn layers, 0 linear "
            "attn layers (skipping mamba backend)",
            len(full_attn_layers),
        )
        return full_attn_backend

    kda_backend = (getattr(server_args, "kda_backend", None) or "auto").strip().lower()
    if is_kda:
        kda_backend = _resolve_kda_backend(kda_backend)
        linear_attn_backend = KdaAttnBackend(config, kda_backend=kda_backend)
    elif is_qwen4_exp(hf_config):
        from tokenspeed.runtime.layers.attention.backends.qwen4_exp import (
            Qwen4ExpMambaAttnBackend,
        )

        linear_attn_backend = Qwen4ExpMambaAttnBackend(config)
    else:
        linear_attn_backend = MambaAttnBackend(config)

    # Recurrent state lives in the LCM arena and is addressed by the
    # per-group block tables, so no separate request-indexed Mamba pool exists.
    linear_attn_backend.set_kv_pool(pool)

    hybrid_cls = HybridKDABackend if is_kda else HybridLinearAttnBackend
    backend = hybrid_cls(full_attn_backend, linear_attn_backend, full_attn_layers)
    logger.info(
        "Created hybrid_linear_attn backend: %d full attn layers, %d linear attn layers, %s",
        len(full_attn_layers),
        len(mamba_layer_ids),
        "LCM state fields",
    )
    return backend


def _wrap_inkling_backend(
    inner,
    text_config,
    attn_config,
    *,
    num_layers,
    is_draft,
    conv_columns,
    enable_layerwise_cache_ready=False,
):
    """Wrap a dense backend with the engine-side Inkling sconv state pool.

    The wrapper only adds conv metadata; all attention delegates to ``inner``.
    Returns ``(backend, conv_pool)``.
    """
    from tokenspeed.runtime.configs.inkling_config import inkling_conv_total_dim
    from tokenspeed.runtime.layers.attention.backends.inkling import (
        InklingAttnBackend,
        InklingConvStatePool,
    )

    kernel_size = text_config.sconv_kernel_size
    spec_tokens = attn_config.speculative_num_draft_tokens
    # Ring row of absolute position p is p % R. R must keep a round's
    # pre-chunk tap reads and chunk-row writes disjoint mod R: (W-1) history
    # taps + K chunk rows. Uniform across target and draft.
    ring_size = (kernel_size - 1) + spec_tokens
    conv_pool = InklingConvStatePool(
        num_layers=num_layers,
        # Row 0 is reserved (1-based indices); +2 covers it plus a padding slot
        num_slots=attn_config.max_bs + 2,
        conv_dim=inkling_conv_total_dim(text_config, attn_config.attn_tp_size),
        kernel_size=kernel_size,
        ring_size=ring_size,
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
        conv_columns=conv_columns,
        spec_num_tokens=spec_tokens,
        is_draft=is_draft,
        enable_layerwise_cache_ready=enable_layerwise_cache_ready,
    )
    return backend, conv_pool


def _inkling_conv_columns(pool, text_config):
    """Return the ShortConv checkpoint groups backed by the cache plan."""
    layer_labels = text_config.cache_layer_types
    prefix_granularity = pool.arena.plan.prefix_granularity
    # The checkpoint grain belongs to the conv groups' own specs; P is only
    # the fallback when a group is absent from the plan.
    specs_by_id = {spec.group_id: spec for spec in pool.arena.cache_group_specs}

    def conv_grain(group_id):
        spec = specs_by_id.get(group_id)
        return spec.block_granularity if spec is not None else prefix_granularity

    conv_columns = {
        "block_tokens": conv_grain("kvconv"),
        "conv_group_of_layer": ("kvconv",) * len(layer_labels),
        "hidden_group_of_layer": ("hiddenconv",) * len(layer_labels),
        "group_block_tokens": {
            "kvconv": conv_grain("kvconv"),
            "hiddenconv": conv_grain("hiddenconv"),
        },
        "pd_endpoint_snapshots": all(
            spec.transfer_policy == "latest_snapshot"
            for spec in pool.arena.cache_group_specs
            if spec.group_id in ("kvconv", "hiddenconv")
        )
        and any(
            spec.group_id in ("kvconv", "hiddenconv")
            for spec in pool.arena.cache_group_specs
        ),
    }
    logger.info(
        "Inkling ShortConv boundary checkpoints: P=%d, groups=%s",
        prefix_granularity,
        tuple(conv_columns["group_block_tokens"]),
    )
    return conv_columns


def _create_target_components(
    *,
    server_args,
    model_config,
    config,
    cache_spec: CachePoolSpec,
    arena: CacheArena,
    rank: int,
    full_attn_backend_name: str | None,
    is_hybrid_linear: bool,
    is_kda: bool,
    is_inkling: bool,
):
    """The target's compute view onto the shared arena + target backend."""
    # The arena owns every planned field; this view binds only the target
    # model's layer window.
    pool = create_cache_pool(
        cache_spec,
        config,
        arena,
        num_layers=len(cache_spec.layer_types),
        rank=rank,
    )
    if is_hybrid_linear:
        backend = _create_hybrid_linear_attn_backend(
            server_args,
            model_config,
            config,
            pool=pool,
            full_attn_backend_name=full_attn_backend_name,
            is_kda=is_kda,
        )
        return backend, pool

    backend = _create_attn_backend(model_config.attention_arch, config)
    if not is_inkling:
        return backend, pool

    text_config = model_config.hf_config.get_text_config()
    backend, _ = _wrap_inkling_backend(
        backend,
        text_config,
        config,
        num_layers=text_config.num_hidden_layers,
        is_draft=False,
        conv_columns=_inkling_conv_columns(pool, text_config),
        enable_layerwise_cache_ready=(
            server_args.disaggregation_mode == "prefill"
            and getattr(server_args, "disaggregation_layerwise_interval", 0) > 0
        ),
    )
    return backend, pool


def _create_draft_components(
    *,
    server_args,
    model_config,
    config,
    pool,
    cache_spec: CachePoolSpec,
    num_target_layers: int,
    full_attn_backend_name: str | None,
    is_heterogeneous: bool,
    is_hybrid_linear: bool,
    is_kda: bool,
    is_inkling: bool,
):
    """Draft backend + the ONE arena viewed through the draft's layer window.

    One big model, one arena: draft layers are continuation layers of the
    merged plan, so the draft pool is a second compute view whose
    ``field_layer_offset`` places its LOCAL layer ids (a NextN draft's one
    layer is layer 0) onto the continuation range. Nothing remaps ids on the
    way through -- the view *is* the mapping, and an id outside the window is
    rejected rather than silently offset onto another model's planes.
    """
    if config is None:
        return None, None
    if is_heterogeneous and (is_hybrid_linear or is_inkling):
        raise RuntimeError(
            "heterogeneous cache views currently support ordinary drafts only"
        )
    num_layers = model_config.num_attention_layers
    # The draft view's transfer counter stays local/None; heterogeneous PD is
    # rejected before construction.
    draft_pool = create_cache_pool(
        cache_spec,
        config,
        pool.arena,
        num_layers=num_layers,
        rank=pool.rank,
        field_layer_offset=num_target_layers,
    )
    if is_hybrid_linear:
        backend = _create_hybrid_linear_attn_backend(
            server_args,
            model_config,
            config,
            pool=draft_pool,
            full_attn_backend_name=full_attn_backend_name,
            is_kda=is_kda,
        )
        return backend, draft_pool

    backend = _create_attn_backend(model_config.attention_arch, config)
    if is_inkling:
        # Depth layers carry conv checkpoint fields as continuation tenants
        # of the target's kvconv/hiddenconv groups; the draft gets the same
        # paged bridges (publish/restore) the target wrapper gets.
        text_config = model_config.hf_config.get_text_config()
        backend, _ = _wrap_inkling_backend(
            backend,
            text_config,
            config,
            num_layers=num_layers,
            is_draft=True,
            conv_columns=_inkling_conv_columns(draft_pool, text_config),
        )
    return backend, draft_pool


def _prepare_verify_workspace(
    *,
    server_args,
    config,
    backend,
    draft_backend,
    uses_paged_state_verify: bool,
    is_inkling: bool,
    expected_bytes: int,
) -> None:
    if uses_paged_state_verify and expected_bytes:
        model_name = "paged-state"
        actual_bytes = backend.linear_attn_backend.preallocate_verify_workspace(
            config.max_bs,
            int(server_args.speculative_num_draft_tokens),
        )
    elif is_inkling:
        model_name = "Inkling"
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
    draft_architectures = (
        getattr(draft_model_config.hf_config, "architectures", None) or []
        if draft_model_config is not None
        else []
    )
    is_inkling_draft_model = any(
        architecture in _INKLING_ARCHITECTURES for architecture in draft_architectures
    )
    is_dspark_draft_model = any(
        architecture == "DeepseekV4ForCausalLMDSpark"
        for architecture in draft_architectures
    )
    is_deepseek_v4_draft_model = (
        draft_model_config is not None
        and not is_dspark_draft_model
        and is_deepseek_v4(draft_model_config.hf_config)
    )
    original_attn_backend = server_args.attention_backend
    if is_deepseek_v4_model:
        server_args.attention_backend = "deepseek_v4"
    if is_deepseek_v4_draft_model:
        server_args.drafter_attention_backend = "deepseek_v4"
        if server_args.disaggregation_mode in ("prefill", "decode"):
            raise NotImplementedError(
                "DeepSeek V4 PD supports target-only decoding; a DeepSeek V4 "
                "draft cache is not transferable"
            )
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
    if (
        (is_inkling or is_inkling_draft_model)
        and server_args.disaggregation_mode in ("prefill", "decode")
        and (
            draft_model_config is not None
            or getattr(server_args, "speculative_algorithm", None) is not None
        )
    ):
        raise NotImplementedError(
            "Inkling PD supports target-only decoding; speculative/draft "
            "ShortConv checkpoint transfer is not implemented"
        )
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
    # The qwen4_exp family check needs the top-level config: the nested
    # text_config has no ``architectures`` so resolve_architecture would
    # return its class name and the check would always be False.
    use_qwen4_exp_cache = use_cache_gdn and is_qwen4_exp(model_config.hf_config)
    use_cache_k3 = is_hybrid_mla_kda
    use_cache_inkling = is_inkling
    if is_deepseek_v4_model:
        cache_family = "deepseek_v4"
    elif use_qwen4_exp_cache:
        cache_family = "qwen4_exp"
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
    if server_args.mapping.attn.dcp_size > 1:
        _validate_dcp_target(
            config,
            is_deepseek_v4_model=is_deepseek_v4_model,
            full_attn_backend_name=target_full_attn_backend_name,
        )
    draft_attn_config = (
        _create_attn_config(server_args, draft_model_config, is_draft=True)
        if draft_model_config and not is_dspark_draft_model
        else None
    )
    if is_deepseek_v4_draft_model:
        draft_attn_config.sliding_window_tokens = int(
            draft_model_config.hf_config.sliding_window
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
    draft_cache_family = _ordinary_cache_family(draft_attn_config)
    heterogeneous_draft_family = _resolve_heterogeneous_draft_family(
        cache_family,
        draft_cache_family,
    )
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
    spec = cache_setup.spec
    target_spec = spec
    draft_view_spec = None
    if cache_setup.num_draft_layers:
        # Transfer fields need one owner even when target and draft share a
        # cache family, so both compute views use disjoint layer windows.
        target_spec = spec.layer_view(
            first_layer=0,
            num_layers=cache_setup.num_target_layers,
        )
        draft_view_spec = spec.layer_view(
            first_layer=cache_setup.num_target_layers,
            num_layers=cache_setup.num_draft_layers,
            family=heterogeneous_draft_family,
        )
    prefix_granularity = spec.memory_plan.prefix_granularity
    _validate_lcm_page_size(
        config,
        prefix_granularity=prefix_granularity,
    )
    if draft_attn_config is not None:
        _validate_lcm_page_size(
            draft_attn_config,
            prefix_granularity=prefix_granularity,
        )
    cache_budget_bytes = cache_setup.cache_budget_bytes
    fixed_workspace_bytes = cache_setup.fixed_workspace_bytes
    logger.info(
        "Cache profile: parent_bytes=%d, P=%d, parents=%d, token_capacity=%d, "
        "layers=%d (draft %d), groups=%s",
        spec.memory_plan.lcm_block_bytes,
        spec.memory_plan.prefix_granularity,
        spec.memory_plan.num_lcm_blocks,
        spec.token_capacity,
        len(spec.layer_types),
        cache_setup.num_draft_layers,
        {
            group.group_id: group.cache_blocks_per_lcm_block
            for group in spec.memory_plan.groups
        },
    )

    # One model, one arena: the merged plan's single allocation, which every
    # compute view below (target, draft) is a layer window onto.
    pp_logical_plan = None
    if server_args.mapping.has_pp:
        # Chunk-pipeline stage: physically allocate only this stage's layers'
        # planes. The logical geometry (parents, packing, page math) stays
        # the full model's so every rank's scheduler plans identically; keep
        # the full plan for the PD wire contract (every stage registers the
        # same logical layout, Decode plans stage windows against it).
        from dataclasses import replace as _dc_replace

        from tokenspeed.runtime.distributed.pp_stage import pp_layer_window

        stage_start, stage_end = pp_layer_window(
            len(spec.layer_types), server_args.mapping
        )
        pp_logical_plan = spec.memory_plan
        spec = _dc_replace(
            spec,
            memory_plan=spec.memory_plan.narrow_to_layers(stage_start, stage_end),
        )
        target_spec = spec
    arena = create_cache_arena(
        spec,
        device=config.device,
        enable_memory_saver=enable_memory_saver,
    )
    if pp_logical_plan is not None:
        arena.pp_logical_plan = pp_logical_plan
    backend, pool = _create_target_components(
        server_args=server_args,
        model_config=model_config,
        config=config,
        cache_spec=target_spec,
        arena=arena,
        rank=rank,
        full_attn_backend_name=target_full_attn_backend_name,
        is_hybrid_linear=is_hybrid_linear,
        is_kda=is_hybrid_mla_kda,
        is_inkling=is_inkling,
    )
    draft_attn_backend, draft_pool = _create_draft_components(
        server_args=server_args,
        model_config=draft_model_config,
        config=draft_attn_config,
        pool=pool,
        cache_spec=draft_view_spec,
        num_target_layers=cache_setup.num_target_layers,
        full_attn_backend_name=draft_full_attn_backend_name,
        is_heterogeneous=heterogeneous_draft_family is not None,
        is_hybrid_linear=draft_is_hybrid_gdn or draft_is_hybrid_mla_kda,
        is_kda=draft_is_hybrid_mla_kda,
        is_inkling=any(a in _INKLING_ARCHITECTURES for a in draft_architectures),
    )

    # A cache-group contract backend needs the contract marked before CUDA-graph
    # state allocation (mark_cache_contract sizes the per-group write-location
    # buffer). Composite/wrapper backends without the hook are a no-op.
    for side_backend, side_pool in ((backend, pool), (draft_attn_backend, draft_pool)):
        if side_backend is None or side_pool is None:
            continue
        side_backend.set_cache_pool(side_pool)
        side_arena = getattr(side_pool, "arena", None)
        if getattr(side_arena, "runtime_contract", None) is None:
            continue
        mark_cache_contract = getattr(side_backend, "mark_cache_contract", None)
        if mark_cache_contract is None:
            continue
        mark_cache_contract()

    _prepare_verify_workspace(
        server_args=server_args,
        config=config,
        backend=backend,
        draft_backend=draft_attn_backend,
        uses_paged_state_verify=use_cache_gdn or use_cache_k3,
        is_inkling=use_cache_inkling,
        expected_bytes=fixed_workspace_bytes,
    )

    cache_storage = _cache_storage_report(
        configured_cache_bytes=cache_budget_bytes,
        pool=pool,
        fixed_workspace_bytes=fixed_workspace_bytes,
    )

    return (
        backend,
        pool,
        draft_attn_backend,
        draft_pool,
        cache_storage,
    )
