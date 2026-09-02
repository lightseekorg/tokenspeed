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

import dataclasses
import logging
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.configs.model_config import (
    AttentionArch,
    is_deepseek_v4,
    is_qwen4_exp,
)
from tokenspeed.runtime.layers.attention.configs.base import (
    AttnConfig,
    SoftmaxAttnConfig,
)
from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
from tokenspeed.runtime.layers.attention.configs.linear_attn import LinearAttnConfig
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


def _ordinary_cache_family(config: AttnConfig | None) -> CacheModelFamily | None:
    if config is None:
        return None
    softmax_attn = config.component(SoftmaxAttnConfig)
    if type(softmax_attn) is MHAConfig:
        return "mha"
    if type(softmax_attn) is MLAConfig:
        return "mla"
    if isinstance(softmax_attn, DSAConfig):
        return "dsa"
    if isinstance(softmax_attn, MSAConfig):
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
    # The arena is the one definition of child-token capacity.
    physical_token_capacity = int(arena.size)
    geometry = {
        "prefix_granularity": int(plan.prefix_granularity),
        "num_lcm_blocks": int(plan.num_lcm_blocks),
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
        "physical_token_capacity": physical_token_capacity,
        "capacity_source": "lcm_geometry",
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
_HYBRID_DSA_KDA_TARGET_ARCHITECTURES = {
    "Glm53FlashForConditionalGeneration",
}
_HYBRID_DSA_KDA_ARCHITECTURES = {
    *_HYBRID_DSA_KDA_TARGET_ARCHITECTURES,
    "Glm53FlashForConditionalGenerationNextN",
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

_DSPARK_DRAFT_ARCHITECTURE = "DeepseekV4ForCausalLMDSpark"


@dataclasses.dataclass(frozen=True)
class _AttnSideProfile:
    """Architecture-derived family facts for one side (target or draft).

    Everything here resolves from the model's ``hf_config`` before any
    attention config exists, so the target and draft sides share one
    derivation instead of two interleaved copies.
    """

    architectures: tuple[str, ...]
    requested_backend: str | None
    is_hybrid_gdn: bool
    is_kda: bool
    # KDA hybrid whose full-attention layers are DSA (GLM-5.3-Flash); a
    # subset of ``is_kda`` that selects the DSA history consumer and the
    # glm53_flash cache family.
    is_dsa_kda: bool
    is_inkling: bool
    is_deepseek_v4: bool
    is_dspark: bool

    @property
    def is_hybrid_linear(self) -> bool:
        # GDN and KDA both take the hybrid-linear path; they differ only in
        # the linear kernel (GDN scalar decay vs KDA per-channel) and the
        # base attn arch (MHA vs MLA vs DSA).
        return self.is_hybrid_gdn or self.is_kda


def _resolve_attn_side(
    model_config: ModelConfig, requested_backend: str | None
) -> _AttnSideProfile:
    hf_config = model_config.hf_config
    architectures = getattr(hf_config, "architectures", None) or []
    is_dspark = _DSPARK_DRAFT_ARCHITECTURE in architectures
    is_dsa_kda = any(a in _HYBRID_DSA_KDA_ARCHITECTURES for a in architectures)
    return _AttnSideProfile(
        architectures=tuple(architectures),
        requested_backend=requested_backend,
        is_hybrid_gdn=any(a in _HYBRID_GDN_ARCHITECTURES for a in architectures),
        is_kda=is_dsa_kda
        or any(a in _HYBRID_MLA_KDA_ARCHITECTURES for a in architectures),
        is_dsa_kda=is_dsa_kda,
        is_inkling=any(a in _INKLING_ARCHITECTURES for a in architectures),
        # The DSpark draft resolves as a V4 architecture but has no paged
        # attention config of its own; it must not take the V4 branches.
        is_deepseek_v4=not is_dspark and is_deepseek_v4(hf_config),
        is_dspark=is_dspark,
    )


def _check_pd_support(
    server_args: ServerArgs,
    target: _AttnSideProfile,
    draft: _AttnSideProfile | None,
    *,
    has_draft_model: bool,
) -> None:
    """Every disaggregated-serving support gate, raised up front."""
    if server_args.disaggregation_mode not in ("prefill", "decode"):
        return
    if draft is not None and draft.is_deepseek_v4:
        raise NotImplementedError(
            "DeepSeek V4 PD supports target-only decoding; a DeepSeek V4 "
            "draft cache is not transferable"
        )
    if (target.is_inkling or (draft is not None and draft.is_inkling)) and (
        has_draft_model
        or getattr(server_args, "speculative_algorithm", None) is not None
    ):
        raise NotImplementedError(
            "Inkling PD supports target-only decoding; speculative/draft "
            "ShortConv checkpoint transfer is not implemented"
        )


def _apply_backend_overrides(
    server_args: ServerArgs,
    target: _AttnSideProfile,
    draft: _AttnSideProfile | None,
) -> None:
    """The one place family resolution writes back into ``server_args``.

    The mutation is deliberate, not a shortcut: ``_create_attn_config`` reads
    the backend choice through the generate() protocol, and the
    ``global_server_args_dict`` snapshot serves models that pick kernel paths
    at build time (e.g. ``deepseek_v3.attention_backend``). Must run before
    any ``_create_attn_config`` call. The user's pre-override choice survives
    as ``profile.requested_backend``.
    """
    if target.is_deepseek_v4:
        server_args.attention_backend = "deepseek_v4"
    if draft is not None and draft.is_deepseek_v4:
        server_args.drafter_attention_backend = "deepseek_v4"

    if target.is_hybrid_linear:
        # GDN (Qwen3.5) / KDA (Kimi-K3) hybrid models always need
        # hybrid_linear_attn. The user's original choice stays in the profile
        # for the full-attention sub-backend (MHA for GDN, MLA for KDA).
        server_args.attention_backend = "hybrid_linear_attn"
    elif server_args.attention_backend == "hybrid_linear_attn":
        logger.warning(
            "Ignoring hybrid_linear_attn backend for non-hybrid model architectures=%s",
            target.architectures,
        )
        server_args.attention_backend = None
        if server_args.drafter_attention_backend == "hybrid_linear_attn":
            logger.warning(
                "Ignoring hybrid_linear_attn backend for non-hybrid model architectures=%s",
                draft.architectures if draft is not None else (),
            )
            server_args.drafter_attention_backend = None


def _resolve_full_attn_backend_name(
    profile: _AttnSideProfile, softmax_attn, hybrid_request: str | None
) -> str:
    """The name the full-attention layers run on (the hybrid sub-backend,
    or the config's own resolution)."""
    if profile.is_hybrid_linear:
        return _resolve_hybrid_full_backend_name(
            hybrid_request,
            is_kda=profile.is_kda,
            is_dsa=profile.is_dsa_kda,
            has_cache_plan=True,
        )
    return softmax_attn.backend_name


def _has_state_layers(config: AttnConfig) -> bool:
    """The plan actually carries recurrent state (hybrid arch + state labels)."""
    if config.component(LinearAttnConfig) is None:
        return False
    layer_types = getattr(config.component(SoftmaxAttnConfig), "layer_types", ())
    return any(layer_type in STATE_LAYER_TYPES for layer_type in layer_types)


def _resolve_cache_family(
    profile: _AttnSideProfile,
    model_config: ModelConfig,
    config: AttnConfig,
) -> CacheModelFamily:
    """The one dispatch from family facts (plus built config) to the recipe."""
    if profile.is_deepseek_v4:
        return "deepseek_v4"
    if profile.is_hybrid_gdn and _has_state_layers(config):
        # The qwen4_exp check needs the top-level config: the nested
        # text_config has no ``architectures`` so resolve_architecture would
        # return its class name and the check would always be False.
        if is_qwen4_exp(model_config.hf_config):
            return "qwen4_exp"
        return "qwen_gdn"
    if profile.is_dsa_kda:
        return "glm53_flash"
    if profile.is_kda:
        return "kimi_k3"
    if profile.is_inkling:
        return "inkling"
    family = _ordinary_cache_family(config)
    if family is None:
        raise RuntimeError(
            "No cache recipe is registered for "
            f"attention config {type(config.component(SoftmaxAttnConfig)).__name__}"
        )
    return family


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


def create_paged_router(
    config: AttnConfig,
    arch: AttentionArch,
    *,
    backend_name: str | None = None,
) -> AttentionBackend:
    """Build the CacheGroupRouter for one side's paged attention.

    The router builds one ``PagedAttentionBackend`` leaf per paged
    (history-family) cache group of the pool view bound later via
    ``set_cache_pool``; each leaf's kernel page size resolves from the
    config override, the leaf class default, or the group's own block
    granularity (``PagedAttentionBackend.resolve_kernel_page_size``).
    """
    from tokenspeed.runtime.layers.attention.backends.router import (
        CacheGroupRouter,
    )

    spec = config.component(SoftmaxAttnConfig)
    name = backend_name if backend_name is not None else spec.backend_name
    if name == "hybrid_linear_attn":
        # The composite sentinel _apply_backend_overrides writes into
        # server_args (and MHAConfig.generate copies into the spec). It
        # names the WRAPPER; the leaf under it auto-resolves from the arch.
        name = None
    leaf_cls = _get_backend_cls(name, arch)

    def leaf_factory(group_id: str, block_granularity: int):
        del group_id
        kernel_page_size = leaf_cls.resolve_kernel_page_size(config, block_granularity)
        # A fresh spec, never a mutate-restore of the shared component: leaf
        # construction happens lazily at set_cache_pool, and several leaves
        # interpret backend_name themselves (MHA/MLA kernel-solution maps),
        # so a wrapper-selecting name like 'dsa' must not reach them.
        leaf_spec = dataclasses.replace(spec, backend_name=name)
        return leaf_cls(config, leaf_spec, kernel_page_size=kernel_page_size)

    return CacheGroupRouter(
        leaf_factory,
        is_draft=bool(config.is_draft),
        spec_num_tokens=config.speculative_num_draft_tokens or 1,
        device=config.device,
    )


def _validate_lcm_page_size(
    config: AttnConfig,
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

_CONFIG_CLS: dict[AttentionArch, type[SoftmaxAttnConfig]] = {
    AttentionArch.MHA: MHAConfig,
    AttentionArch.MLA: MLAConfig,
    AttentionArch.DSA: DSAConfig,
    AttentionArch.MSA: MSAConfig,
}

# Architectures declaring a linear-attention component, registered like
# _CONFIG_CLS. Whether a given checkpoint actually has linear layers is
# decided by generate() (NextN drafts may carry none).
_LINEAR_ATTN_CLS: dict[str, type[LinearAttnConfig]] = {
    arch: LinearAttnConfig
    for arch in (
        *_HYBRID_GDN_ARCHITECTURES,
        *_HYBRID_MLA_KDA_ARCHITECTURES,
        # GLM NextN is one DSA layer. It reuses the target's mixed-layer
        # metadata but must not acquire a linear-attention component.
        *_HYBRID_DSA_KDA_TARGET_ARCHITECTURES,
    )
}


def _create_attn_config(
    server_args: ServerArgs, model_config: ModelConfig, is_draft: bool = False
) -> AttnConfig:
    arch = model_config.attention_arch
    if arch not in _CONFIG_CLS:
        raise NotImplementedError(f"Not supported Attention Arch: {arch!r}")
    config = _CONFIG_CLS[arch].generate(server_args, model_config, is_draft)
    # Extra components are built through the same generate() protocol and
    # composed into config.components (consumers look them up by class via
    # ``component()``).
    architectures = getattr(model_config.hf_config, "architectures", None) or ()
    linear_cls = next(
        (_LINEAR_ATTN_CLS[a] for a in architectures if a in _LINEAR_ATTN_CLS), None
    )
    if linear_cls is not None:
        linear_attn = linear_cls.generate(server_args, model_config, is_draft)
        if linear_attn is not None:
            config = dataclasses.replace(
                config, components=config.components + (linear_attn,)
            )
    return config


def _create_attn_backend(
    arch: AttentionArch,
    config: AttnConfig,
) -> AttentionBackend:
    return _create_attn_backend_with_name(
        config.component(SoftmaxAttnConfig).backend_name, arch, config
    )


def _create_attn_backend_with_name(
    name: str | None,
    arch: AttentionArch,
    config: AttnConfig,
) -> AttentionBackend:
    from tokenspeed.runtime.layers.attention.backends.paged import (
        PagedAttentionBackend,
    )

    cls = _get_backend_cls(name, arch)
    if issubclass(cls, PagedAttentionBackend):
        # Paged leaves are served through the cache-group router: one leaf
        # per history group, blocks -> kernel pages mapped in one place.
        return create_paged_router(config, arch, backend_name=name)
    spec = dataclasses.replace(
        config.component(SoftmaxAttnConfig),
        backend_name=name,
    )
    return cls(config, spec)


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
    is_dsa: bool,
    has_cache_plan: bool,
) -> str | None:
    """Resolve the compute backend that consumes the hybrid history cache."""
    name = _BACKEND_ALIASES.get(requested_name, requested_name)
    if name == "hybrid_linear_attn":
        name = None
    if has_cache_plan and is_dsa and name is None:
        return "dsa"
    # NVIDIA K3 defaults to its CuteDSL history consumer. AMD keeps the
    # generic MLA backend; explicit user choices remain authoritative.
    if has_cache_plan and is_kda and name is None and not current_platform().is_amd:
        return "tokenspeed_mla"
    return name


def _create_hybrid_linear_attn_backend(
    server_args: ServerArgs,
    model_config: ModelConfig,
    config: AttnConfig,
    *,
    pool,
    full_attn_backend_name: str | None = None,
    is_kda: bool = False,
) -> AttentionBackend:
    """Create a hybrid backend for a linear-attention model over one pool.

    GDN (Qwen3.5, MHA base) or, when ``is_kda`` is set, KDA (Kimi-K3,
    MLA base; GLM-5.3-Flash, DSA base). ``pool`` is the model's layer-mapped
    view over the one shared cache pool; both sub-backends consume its
    per-group tables.
    """
    from tokenspeed.runtime.layers.attention.backends.hybrid import (
        HybridLinearAttnBackend,
    )
    from tokenspeed.runtime.layers.attention.backends.kda import (
        KdaAttnBackend,
    )
    from tokenspeed.runtime.layers.attention.backends.mamba import (
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

    # Create mamba/linear attention backend. Only propagate the configured
    # verify width when spec-dec is actually enabled — matches MLAConfig /
    # MHAConfig.generate. Otherwise the AttnConfig sentinel (1) wins so
    # non-spec hybrid decode doesn't get misclassified as target verify /
    # draft extend by `self.spec_num_tokens > 1`.
    if server_args.speculative_algorithm is not None:
        config.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens

    # The linear component's presence decides whether this model actually
    # has any linear / mamba layers. A draft model on a hybrid-GDN target
    # (e.g. MTP on Qwen3.5) shares the same architecture class as the
    # target but commonly ships with *zero* mamba layers — in that case
    # we skip the mamba backend entirely so that its
    # ``init_forward_metadata_*`` hooks do not run (they would otherwise
    # touch a zero-sized pool on the same persistent state_indices_list
    # as the target, which breaks the captured CUDA graph).
    linear_attn = config.component(LinearAttnConfig)

    if linear_attn is None:
        logger.info(
            "Created hybrid_linear_attn backend: %d full attn layers, 0 linear "
            "attn layers in this cache view (skipping linear backend)",
            len(full_attn_layers),
        )
        return full_attn_backend

    kda_backend = (getattr(server_args, "kda_backend", None) or "auto").strip().lower()
    if is_kda:
        kda_backend = _resolve_kda_backend(kda_backend)
        linear_attn_backend = KdaAttnBackend(
            config, config.component(SoftmaxAttnConfig), kda_backend=kda_backend
        )
    elif is_qwen4_exp(hf_config):
        from tokenspeed.runtime.layers.attention.backends.qwen4_exp import (
            Qwen4ExpMambaAttnBackend,
        )

        linear_attn_backend = Qwen4ExpMambaAttnBackend(
            config, config.component(SoftmaxAttnConfig)
        )
    else:
        linear_attn_backend = MambaAttnBackend(
            config, config.component(SoftmaxAttnConfig)
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
        len(linear_attn.layer_ids),
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
        conv_dim=inkling_conv_total_dim(
            text_config, attn_config.component(SoftmaxAttnConfig).attn_tp_size
        ),
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
def _narrow_spec_for_pp(spec: CachePoolSpec, mapping) -> tuple[CachePoolSpec, object]:
    """Chunk-pipeline stage: physically allocate only this stage's layers'
    planes. The logical geometry (parents, packing, page math) stays the
    full model's so every rank's scheduler plans identically; the returned
    full plan serves the PD wire contract (every stage registers the same
    logical layout, Decode plans stage windows against it).
    """
    from tokenspeed.runtime.distributed.pp_stage import pp_layer_window

    stage_start, stage_end = pp_layer_window(len(spec.layer_types), mapping)
    pp_logical_plan = spec.memory_plan
    spec = dataclasses.replace(
        spec,
        memory_plan=spec.memory_plan.narrow_to_layers(stage_start, stage_end),
    )
    return spec, pp_logical_plan


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
    target = _resolve_attn_side(model_config, server_args.attention_backend)
    draft = (
        _resolve_attn_side(draft_model_config, server_args.drafter_attention_backend)
        if draft_model_config is not None
        else None
    )
    _check_pd_support(
        server_args, target, draft, has_draft_model=draft_model_config is not None
    )
    _apply_backend_overrides(server_args, target, draft)

    config = _create_attn_config(server_args, model_config)
    softmax_attn = config.component(SoftmaxAttnConfig)
    if target.is_deepseek_v4:
        softmax_attn.sliding_window_tokens = int(model_config.hf_config.sliding_window)
    cache_family = _resolve_cache_family(target, model_config, config)
    target_full_attn_backend_name = _resolve_full_attn_backend_name(
        target, softmax_attn, hybrid_request=target.requested_backend
    )
    draft_attn_config = (
        _create_attn_config(server_args, draft_model_config, is_draft=True)
        if draft is not None and not draft.is_dspark
        else None
    )
    draft_softmax_attn = (
        draft_attn_config.component(SoftmaxAttnConfig)
        if draft_attn_config is not None
        else None
    )
    if draft is not None and draft.is_deepseek_v4:
        draft_softmax_attn.sliding_window_tokens = int(
            draft_model_config.hf_config.sliding_window
        )
    draft_full_attn_backend_name = (
        # The draft's hybrid sub-backend request is its config's own
        # resolution, not the user's target choice.
        _resolve_full_attn_backend_name(
            draft, draft_softmax_attn, hybrid_request=draft_softmax_attn.backend_name
        )
        if draft_attn_config is not None
        else None
    )
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
        spec, pp_logical_plan = _narrow_spec_for_pp(spec, server_args.mapping)
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
        is_hybrid_linear=target.is_hybrid_linear,
        is_kda=target.is_kda,
        is_inkling=target.is_inkling,
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
        is_hybrid_linear=draft is not None and draft.is_hybrid_linear,
        is_kda=draft is not None and draft.is_kda,
        is_inkling=draft is not None and draft.is_inkling,
    )

    # Bind the pools before CUDA-graph state allocation: backends learn
    # their group geometry (and buffer sizing) from the pool's published
    # specs. Every LCM pool publishes a cache contract, so there is no
    # separate contract-marking step.
    for side_backend, side_pool in ((backend, pool), (draft_attn_backend, draft_pool)):
        if side_backend is None or side_pool is None:
            continue
        side_backend.set_cache_pool(side_pool)

    _prepare_verify_workspace(
        server_args=server_args,
        config=config,
        backend=backend,
        draft_backend=draft_attn_backend,
        uses_paged_state_verify=cache_family in ("qwen4_exp", "qwen_gdn", "kimi_k3"),
        is_inkling=cache_family == "inkling",
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
