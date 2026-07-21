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

from tokenspeed.runtime.configs.flat_memory_plan import (
    components_from_layers,
    equalized_block_size,
    plan_component_tensors,
    state_const_bytes,
)
from tokenspeed.runtime.configs.model_config import AttentionArch, is_deepseek_v4
from tokenspeed.runtime.configs.paged_cache_spec import (
    hybrid_slab_group_size,
    scheduler_ext_flat_kvcache,
)
from tokenspeed.runtime.layers.attention.configs.base import BaseAttnConfig
from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.configs.msa import (
    MSAConfig,
)
from tokenspeed.runtime.layers.attention.kv_cache.base import BaseTokenToKVPool
from tokenspeed.runtime.layers.attention.utils import (
    profile_available_cache_memory_bytes,
    profile_cache_budget,
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
    return draft_attn_config.cache_cell_size() * draft_model_config.num_attention_layers


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

# Inkling stays on the plain dense path (thin sconv wrapper); deliberately NOT hybrid-GDN
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
    full_attn_backend = _create_attn_backend_with_name(
        full_attn_backend_name,
        arch,
        config,
    )

    # Create mamba/linear attention backend. Only propagate the configured
    # verify width when spec-dec is actually enabled — matches MLAConfig /
    # MHAConfig.generate. Otherwise the BaseAttnConfig sentinel (1) wins so
    # non-spec hybrid decode doesn't get misclassified as target verify /
    # draft extend by `self.spec_num_tokens > 1`.
    if server_args.speculative_algorithm is not None:
        config.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens

    flat_kvcache = scheduler_ext_flat_kvcache()
    if flat_kvcache:
        # Flat path: the pool covers ALL layers, so pool indices == global
        # layer ids, its layer_types line up with the state slabs, and the
        # group specs publish both the "full_attention" and
        # "linear_attention" groups. State layers carry NO k/v tensors
        # (None slots, M18a T4) -- matching the plan sizing, which charges
        # only full-layer KV + state rows. The identity mapping keeps
        # the wrapper type identical to the radix path.
        num_total_layers = len(text_config.layers_block_type)
        inner_pool = config.create_pool(
            num_total_layers, max_num_tokens, rank, enable_memory_saver
        )
        pool = LayerMappedKVPool(inner_pool, list(range(num_total_layers)))
    else:
        # Create KV cache pool (only for full attention layers)
        num_full_attn_layers = len(full_attn_layers)
        inner_pool = config.create_pool(
            num_full_attn_layers, max_num_tokens, rank, enable_memory_saver
        )
        # Wrap with layer ID mapping (global layer IDs -> pool indices)
        pool = LayerMappedKVPool(inner_pool, full_attn_layers)

    # Read mamba2_cache_params to decide whether this model actually has
    # any linear / mamba layers. A draft model on a hybrid-GDN target
    # (e.g. MTP on Qwen3.5) shares the same architecture class as the
    # target but commonly ships with *zero* mamba layers — in that case
    # we skip the mamba backend / pool entirely so that its
    # ``init_forward_metadata_*`` hooks do not run (they would otherwise
    # touch a zero-sized pool on the same persistent state_indices_list
    # as the target, which breaks the captured CUDA graph).
    (
        conv_state_shape,
        temporal_state_shape,
        conv_dtype,
        ssm_dtype,
        mamba_layer_ids,
    ) = text_config.mamba2_cache_params

    if len(mamba_layer_ids) == 0:
        logger.info(
            "Created hybrid_linear_attn backend: %d full attn layers, 0 linear "
            "attn layers (skipping mamba backend / pool)",
            len(full_attn_layers),
        )
        return full_attn_backend, pool, None

    linear_attn_backend = MambaAttnBackend(config)

    if flat_kvcache:
        # Flat mode never touches a SimpleMambaPool: the recurrent state
        # lives in the KV pool's state slabs, addressed by the flat block
        # tables (set_kv_pool below activates the dual-index state paging),
        # so skip the pool and its set_pool binding entirely.
        mamba_pool = None
    else:
        # Mamba radix cache uses C++ chunk indices. Without radix cache, the
        # backend uses 1-based req_pool_indices directly, so keep slot 0 as
        # padding.
        per_rank_max_batch = server_args.max_num_seqs // max(
            server_args.data_parallel_size or server_args.mapping.attn.dp_size, 1
        )
        req_pool_padding_index = per_rank_max_batch + 1
        mamba_pool_size = (
            mamba_pool_total_chunks + 1
            if mamba_pool_total_chunks > 0
            else per_rank_max_batch + 1
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
            page_size=server_args.block_size,
            speculative_num_draft_tokens=(
                server_args.speculative_num_draft_tokens
                if server_args.speculative_algorithm is not None
                else 0
            ),
            # ``current_input_indices`` is keyed by the scheduler's rank-local,
            # 1-based req_pool_idx; the row after that range is the CUDA graph
            # padding sentinel.
            max_req_pool_size=req_pool_padding_index,
        )
        linear_attn_backend.set_pool(mamba_pool)
    # Flat state paging (dual-index) keys off the KV pool's state slabs +
    # published "linear_attention" group; no-op on the radix path.
    linear_attn_backend.set_kv_pool(pool)

    backend = HybridLinearAttnBackend(
        full_attn_backend, linear_attn_backend, full_attn_layers
    )
    logger.info(
        "Created hybrid_linear_attn backend: %d full attn layers, %d linear attn layers, %s",
        len(full_attn_layers),
        len(mamba_layer_ids),
        (
            "flat state slabs (no mamba slot pool)"
            if mamba_pool is None
            else f"mamba pool size {mamba_pool.size}"
        ),
    )
    return backend, pool, mamba_pool


def _floor_tokens_to_layout_grid(max_num_tokens: int, config) -> int:
    """Floor the memory-profiled token capacity onto the KV layout's grid.

    The profile is memory-dependent (``--gpu-memory-utilization`` over the
    free bytes at boot), so the raw value can land ANYWHERE on the 128-token
    page grid — but the hetero slot layout is only viewable on a coarser
    grid. With ``slot_tokens`` S and a largest group page P (Inkling: S=256,
    P=256, base page 128), the pool row counts that must divide are:

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
    """Zero-padding slots: full=256 fills the slot, swa=128 leads (tail padded); ids sized by slot."""
    from tokenspeed.runtime.configs.inkling_config import inkling_kv_heads_for_layer

    config.slot_tokens = 256
    config.group_page_sizes = {"full_attention": 256}
    tc = model_config.hf_config.get_text_config()
    config.layer_kv_head_counts = tuple(
        inkling_kv_heads_for_layer(tc, i, True) for i in range(tc.num_hidden_layers)
    )


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
        # ATTN cols ride the K slot, MLP the V slot; each new base needs a FlatInklingShapeSuite pass
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


def _inkling_conv_columns(pool, text_config, conv_bt, hbt):
    """kvconv-as-swa geometry for the backend (None if no kvconv pages).

    Columns live in the layers' own K/V slots; backend gets only geometry + layer->group map.
    """
    layer_labels = text_config.paged_cache_layer_types
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
    int,
    object | None,
]:
    arch = model_config.attention_arch

    architectures = getattr(model_config.hf_config, "architectures", None) or []
    is_hybrid_gdn = any(a in _HYBRID_GDN_ARCHITECTURES for a in architectures)
    is_deepseek_v4_model = is_deepseek_v4(model_config.hf_config)
    is_deepseek_v4_draft_model = draft_model_config is not None and is_deepseek_v4(
        draft_model_config.hf_config
    )
    original_attn_backend = server_args.attention_backend
    if is_deepseek_v4_model:
        server_args.attention_backend = "deepseek_v4"
    if is_deepseek_v4_draft_model:
        server_args.drafter_attention_backend = "deepseek_v4"
    if is_hybrid_gdn:
        # Qwen3.5 GDN hybrid models always need hybrid_linear_attn.
        # Save the user's original choice for the full-attention sub-backend.
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
    is_flat_gdn = getattr(config, "conv_state_shape", None) is not None
    gdn_state_bytes = (
        state_const_bytes(
            config.conv_state_shape,
            config.conv_dtype,
            config.temporal_state_shape,
            config.ssm_dtype,
        )
        if is_flat_gdn
        else None
    )
    if is_flat_gdn:
        equalized_block_size_value = equalized_block_size(
            layer_types=list(config.layer_types),
            kv_bytes_per_slot=config.cache_cell_size(),
            state_const_bytes=gdn_state_bytes,
            block_size=server_args.block_size,
        )
        if equalized_block_size_value != server_args.block_size:
            logger.info(
                "Setting attention block size to %d tokens to cover the GDN "
                "state row (configured block size %d)",
                equalized_block_size_value,
                server_args.block_size,
            )
            server_args.block_size = equalized_block_size_value
            config.page_size = equalized_block_size_value
    draft_attn_config = None
    if draft_model_config:
        draft_attn_config = _create_attn_config(
            server_args, draft_model_config, is_draft=True
        )
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

    hf_config = getattr(model_config, "hf_config", None)
    text_config = getattr(hf_config, "text_config", hf_config) if hf_config else None
    mamba_cache_params = (
        getattr(text_config, "mamba2_cache_params", None) if text_config else None
    )
    # Unpack once with names; every consumer below reads these instead of
    # indexing into the raw tuple.
    if mamba_cache_params:
        (
            mamba_conv_state_shape,
            mamba_temporal_state_shape,
            mamba_conv_dtype,
            mamba_ssm_dtype,
            mamba_layer_ids,
        ) = mamba_cache_params
    else:
        mamba_conv_state_shape = mamba_temporal_state_shape = None
        mamba_conv_dtype = mamba_ssm_dtype = None
        mamba_layer_ids = ()
    has_mamba_layers = len(mamba_layer_ids) > 0
    has_mamba = getattr(model_config, "mambaish_config", None) is not None or (
        has_mamba_layers
    )
    mamba_pool_total_chunks = 0
    mamba_pool = None

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
    elif has_mamba and is_flat_gdn:
        draft_row_bytes = 0
        if draft_attn_config is not None:
            draft_row_bytes = (
                _resolve_draft_cache_cell_size_for_profile(
                    draft_attn_config, draft_model_config, draft_profile_cache_cell_size
                )
                * server_args.block_size
            )
        cache_memory = profile_available_cache_memory_bytes(
            attn_config=config,
            gpu_id=gpu_id,
            tp_size=server_args.mapping.world_size,
            gpu_memory_utilization=server_args.gpu_memory_utilization,
            total_gpu_memory=gpu_memory,
            world_group=server_args.mapping.world_group,
        )
        flat_plan = plan_component_tensors(
            components_from_layers(
                layer_types=list(config.layer_types),
                kv_bytes_per_slot=config.cache_cell_size(),
                state_const_bytes=gdn_state_bytes,
            ),
            block_size=server_args.block_size,
            budget_bytes=cache_memory,
            reserved_bytes_per_block=draft_row_bytes,
        )
        max_total_num_pages = flat_plan.geometry.num_blocks
        logger.info(
            "Flat GDN KV profile: block_bytes=%d (%d component tensors, "
            "block_size=%d), max_total_num_pages=%d",
            flat_plan.geometry.block_bytes,
            len(flat_plan.tensors),
            server_args.block_size,
            max_total_num_pages,
        )
        max_num_tokens = _resolve_max_num_tokens(
            max_total_num_pages, server_args.block_size, server_args.max_total_tokens
        )
    elif has_mamba and server_args.max_mamba_cache_size is not None:
        mamba_pool_total_chunks = server_args.max_mamba_cache_size
        full_attn_layer_ids = getattr(text_config, "full_attention_layer_ids", None)
        num_kv_layers = (
            len(full_attn_layer_ids)
            if full_attn_layer_ids is not None
            else num_layers - len(mamba_layer_ids)
        )
        max_total_num_pages = profile_max_num_pages(
            **{**_profile_kwargs, "num_attention_layers": num_kv_layers},
            gpu_memory_utilization=server_args.gpu_memory_utilization,
            cache_cell_size=profile_cache_cell_size,
            draft_cache_cell_size=draft_profile_cache_cell_size,
        )
        max_num_tokens = _resolve_max_num_tokens(
            max_total_num_pages,
            server_args.block_size,
            server_args.max_total_tokens,
        )
    elif has_mamba and server_args.max_mamba_cache_size is None:
        num_mamba_layers = len(mamba_layer_ids)
        speculative_num_draft_tokens = (
            server_args.speculative_num_draft_tokens
            if server_args.speculative_algorithm is not None
            else 0
        )
        per_layer_mamba_chunk_memory = sum(
            state_const_bytes(
                mamba_conv_state_shape,
                mamba_conv_dtype,
                mamba_temporal_state_shape,
                mamba_ssm_dtype,
            ).values()
        ) * (1 + speculative_num_draft_tokens)
        memory_per_mamba_chunk = num_mamba_layers * per_layer_mamba_chunk_memory
        full_attn_layer_ids = getattr(text_config, "full_attention_layer_ids", None)
        num_kv_layers = (
            len(full_attn_layer_ids)
            if full_attn_layer_ids is not None
            else num_layers - num_mamba_layers
        )
        kv_max_num_pages, mamba_pool_total_chunks = profile_cache_budget(
            **{**_profile_kwargs, "num_attention_layers": num_kv_layers},
            mem_fraction_static=server_args.gpu_memory_utilization,
            mamba_memory_per_chunk=memory_per_mamba_chunk,
            mamba_ratio=server_args.mamba_full_memory_ratio,
        )
        max_num_tokens = _resolve_max_num_tokens(
            kv_max_num_pages,
            server_args.block_size,
            server_args.max_total_tokens,
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

    if _CI_SMALL_KV_SIZE is not None and int(_CI_SMALL_KV_SIZE) > 0:
        max_num_tokens = int(_CI_SMALL_KV_SIZE)
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
    elif is_hybrid_gdn:
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
            mamba_pool_total_chunks=mamba_pool_total_chunks,
        )
    else:
        is_inkling = any(a in _INKLING_ARCHITECTURES for a in architectures)
        # Hetero KV + paged sconv are unconditional for Inkling (the
        # INKLING_HETERO_KV / INKLING_PAGED_SCONV bisection gates were
        # retired 2026-07-15 after the experiments settled).
        paged_sconv = is_inkling
        if is_inkling:
            _apply_inkling_hetero_kv(config, model_config)
            # Must run AFTER the hetero fields exist on the config.
            max_num_tokens = _floor_tokens_to_layout_grid(max_num_tokens, config)
        if paged_sconv:
            _conv_bt, _hbt = _publish_inkling_conv_groups(
                config, model_config, server_args
            )
        backend = _create_attn_backend(arch, config)
        pool = _create_attn_pool(
            config, num_layers, max_num_tokens, rank, enable_memory_saver
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
                conv_columns = _inkling_conv_columns(pool, text_config, _conv_bt, _hbt)
            backend.conv_columns = conv_columns
            if conv_columns is not None:
                # Wrapper-owned conv groups: the attention mixin skips their write-loc math and capture fills
                backend.inner.flat_engine_owned_group_ids = frozenset(
                    conv_columns["group_block_tokens"]
                )
    draft_attn_backend = None
    draft_pool = None
    if draft_attn_config:
        # Check if draft model is also a hybrid GDN model.
        draft_archs = getattr(draft_model_config.hf_config, "architectures", None) or []
        if is_deepseek_v4_draft_model:
            from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4 import (
                DeepseekV4TokenToKVPool,
            )

            draft_attn_backend = _create_attn_backend(
                draft_model_config.attention_arch, draft_attn_config
            )
            draft_pool = DeepseekV4TokenToKVPool(
                size=max_num_tokens,
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
        elif any(a in _HYBRID_GDN_ARCHITECTURES for a in draft_archs):
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
                mamba_pool_total_chunks=mamba_pool_total_chunks,
            )
        elif any(a in _INKLING_ARCHITECTURES for a in draft_archs):
            draft_text_config = draft_model_config.hf_config.get_text_config()
            num_depths = draft_model_config.num_attention_layers
            if server_args.speculative_num_steps > num_depths:
                raise ValueError(
                    f"Inkling MTP has {num_depths} depth layers; "
                    f"--speculative-num-steps {server_args.speculative_num_steps} "
                    "would wrap depths with no trained meaning."
                )
            # Hetero KV, symmetric with the target pool: per-depth head
            # counts come from the MTP text config's own local ids
            # (ModelConfig swapped it in for the draft worker), giving the
            # draft the same byte-uniform slot layout as the target — full
            # depths at 256 tokens/id x ckpt heads, SWA depths at the base
            # 128 x swa heads. The pool shares the TARGET's page-id space
            # (the drafter consumes the target's per-group flat tables), so
            # it is sized at the same max_num_tokens; per-layer views turn
            # the max-head allocation into the group geometry, and the
            # profile's max-head draft charge prices exactly that
            # allocation. (The former num_kv_heads pin + explicit 2x row
            # sizing described the all-full-attention special case of this
            # same byte math.)
            _apply_inkling_hetero_kv(draft_attn_config, draft_model_config)
            logger.info(
                "Inkling MTP draft pool: hetero KV layer head counts=%s, "
                "layer types=%s, group page sizes=%s (%d depths, %d ids)",
                draft_attn_config.layer_kv_head_counts,
                draft_attn_config.layer_types,
                draft_attn_config.group_page_sizes,
                num_depths,
                max_num_tokens // config.page_size,
            )
            draft_attn_backend = _create_attn_backend(
                draft_model_config.attention_arch, draft_attn_config
            )
            draft_pool = _create_attn_pool(
                draft_attn_config,
                num_depths,
                max_num_tokens,
                rank,
                enable_memory_saver,
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
