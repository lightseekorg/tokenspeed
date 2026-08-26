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

"""DeepSeek V4 cache recipe: SWA window, compressed chains, compressor state.

V4 is the one family whose groups are not per-layer: a single layer costs bytes
in three or four groups at once, and their retention comes from the compression
ratio rather than an attention label. So each group is declared whole -- its id
written once, in its spec, next to the fields the layers deposit in it.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import cached_property
from typing import TYPE_CHECKING

from tokenspeed_kernel import dsv4_indexer_cache_format
from typing_extensions import override

from tokenspeed.runtime.layers.attention.deepseek_v4_geometry import (
    V4_COMPRESSOR_STATE_ROWS_PER_PAGE,
    V4_COMPRESSOR_STATE_WINDOW_TOKENS,
    V4_INDEXER_COMPRESSOR_STATE_GROUP_ID,
    V4_KERNEL_BLOCK_ROWS,
    V4_SWA_KV_GROUP_ID,
    DeepseekV4CacheLayout,
    deepseek_v4_cache_layout_from_config,
    v4_compressed_kv_group_id,
    v4_compressed_rows_per_page,
    v4_compressor_state_group_id,
)
from tokenspeed.runtime.layers.attention.kernel_page_sizes import (
    DEEPSEEK_V4_PAGE_SIZE,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.base import (
    CacheGroupDeclaration,
    CacheRecipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    CacheLayout,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    CacheGroupSpec,
    apply_pd_transfer_policies,
    cyclic_history_spec,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import (
        CachePlacement,
    )

_MAX_PADDING_FRACTION = 2.0


def v4_c4_state_window(decode_input_tokens: int) -> int:
    """Tokens the ratio-4 compressor state must retain.

    c4 compression consumes the prior four-token state plus every token in the
    target verify block. Preserve the historical eight-token window for verify
    widths <= 4 and grow it for wider block-speculative decoders.
    """
    if (
        isinstance(decode_input_tokens, bool)
        or not isinstance(decode_input_tokens, int)
        or decode_input_tokens <= 0
    ):
        raise ValueError("decode_input_tokens must be a positive integer")
    return max(V4_COMPRESSOR_STATE_WINDOW_TOKENS[4], 4 + decode_input_tokens)


def v4_swa_kv_spec(hf_config) -> CacheGroupSpec:
    """SWA kv: trailing window only, so State family."""
    return CacheGroupSpec(
        group_id=V4_SWA_KV_GROUP_ID,
        retention="sliding_window",
        rows_per_page=V4_KERNEL_BLOCK_ROWS,
        entry_stride_tokens=1,
        sliding_window_tokens=_resolve_sliding_window(hf_config),
        family="state",
    )


def v4_compressor_state_spec(ratio: int, *, c4_state_window: int) -> CacheGroupSpec:
    """Compressor state for one ratio: tail buffer, so State family."""
    _check_ratio(ratio)
    return CacheGroupSpec(
        group_id=v4_compressor_state_group_id(ratio),
        retention="sliding_window",
        rows_per_page=V4_COMPRESSOR_STATE_ROWS_PER_PAGE[ratio],
        entry_stride_tokens=1,
        sliding_window_tokens=(
            c4_state_window if ratio == 4 else V4_COMPRESSOR_STATE_WINDOW_TOKENS[ratio]
        ),
        family="state",
    )


def v4_compressed_kv_spec(ratio: int) -> CacheGroupSpec:
    """Compressed kv for one ratio: full-history chain (indexer K shares it)."""
    _check_ratio(ratio)
    return CacheGroupSpec(
        group_id=v4_compressed_kv_group_id(ratio),
        retention="full_history",
        rows_per_page=v4_compressed_rows_per_page(ratio),
        entry_stride_tokens=ratio,
        sliding_window_tokens=None,
        family="history",
    )


def v4_indexer_state_spec(*, c4_state_window: int) -> CacheGroupSpec:
    """Indexer compressor state: tail buffer, so State family."""
    return CacheGroupSpec(
        group_id=V4_INDEXER_COMPRESSOR_STATE_GROUP_ID,
        retention="sliding_window",
        rows_per_page=V4_COMPRESSOR_STATE_ROWS_PER_PAGE[4],
        entry_stride_tokens=1,
        sliding_window_tokens=c4_state_window,
        family="state",
    )


def _check_ratio(ratio: int) -> None:
    if ratio not in V4_COMPRESSOR_STATE_WINDOW_TOKENS:
        raise ValueError(f"unsupported DeepSeek V4 compress_ratio={ratio}")


def _resolve_sliding_window(hf_config) -> int:
    for source in (hf_config, getattr(hf_config, "text_config", None)):
        if source is None:
            continue
        if hasattr(source, "sliding_window"):
            value = source.sliding_window
            if value is None:
                raise ValueError("DeepSeek V4 sliding_window is None")
            window = int(value)
            if window <= 0:
                raise ValueError(f"sliding_window must be positive, got {value!r}")
            return window
    raise ValueError("DeepSeek V4 hf_config is missing sliding_window")


@dataclass(frozen=True)
class DeepseekV4PoolOptions:
    """What the V4 pool needs beyond the plan: its kernel cache layout."""

    layout: DeepseekV4CacheLayout

    def layer_view(self, *, first_layer: int, num_layers: int):
        """Narrow the layout to one compute view's layer window.

        The layout's ``layer_ratio`` is per-layer and the pool indexes it by
        its own local layer id, so a draft view carries only its own ratios.
        """
        ratios = self.layout.layer_ratio[first_layer : first_layer + num_layers]
        if len(ratios) != num_layers:
            raise ValueError(
                f"DeepSeek V4 cache layout has no ratios for layers "
                f"[{first_layer}, {first_layer + num_layers})"
            )
        return DeepseekV4PoolOptions(layout=replace(self.layout, layer_ratio=ratios))


class DeepseekV4Recipe(CacheRecipe):
    """DeepSeek V4: one arena for SWA, compressed chains and indexer state.

    The MTP layer already is a continuation layer of the target config
    (``compress_ratios`` carries ``num_hidden_layers + num_nextn`` entries), so
    one pass over the full layer range builds the merged plan.
    """

    family = "deepseek_v4"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.pd_disaggregation_enabled and (
            getattr(self.server_args, "speculative_algorithm", None) is not None
            or self.draft_model_config is not None
            or self.draft_attn_config is not None
            or self.decode_input_tokens != 1
        ):
            raise NotImplementedError(
                "DeepSeek V4 PD supports target-only decoding; speculative/MTP "
                "cache transfer is not implemented"
            )

    # ---- layer vocabulary ----

    @cached_property
    def _cache_layout(self) -> DeepseekV4CacheLayout:
        """Kernel cache geometry over target and draft layers together."""
        source = (
            self.draft_model_config.hf_config
            if self.draft_attn_config is not None
            else self.model_config.hf_config
        )
        return deepseek_v4_cache_layout_from_config(
            source,
            page_size=DEEPSEEK_V4_PAGE_SIZE,
            use_fp4_indexer_cache=self._use_fp4_indexer(source),
            layer_indices=range(self.num_target_layers + self.num_draft_layers),
        )

    @cached_property
    def layer_types(self) -> tuple[str, ...]:
        """The compression ratio per layer -- V4's own layer vocabulary."""
        return tuple(str(ratio) for ratio in self._cache_layout.layer_ratio)

    @cached_property
    @override
    def layer_placements(self) -> tuple[CachePlacement, ...]:
        """Describe the sharded history without inventing per-layer groups."""
        dcp_size = int(getattr(self.attn_config, "dcp_size", 1))
        return tuple(
            (
                "cyclic_history"
                if layer_id < self.num_target_layers and dcp_size > 1 and ratio > 1
                else "replicated"
            )
            for layer_id, ratio in enumerate(self._cache_layout.layer_ratio)
        )

    # ---- geometry ----

    @property
    @override
    def prefix_granularity(self) -> int:
        configured = super().prefix_granularity
        dcp_size = int(getattr(self.attn_config, "dcp_size", 1))
        if dcp_size == 1:
            return configured
        spans = [configured]
        for ratio in set(self._cache_layout.layer_ratio):
            if ratio > 1:
                spans.append(
                    cyclic_history_spec(
                        v4_compressed_kv_spec(int(ratio)), dcp_size=dcp_size
                    ).block_granularity
                )
        return math.lcm(*spans)

    @property
    @override
    def max_padding_fraction(self) -> float:
        return _MAX_PADDING_FRACTION

    # ---- groups: declared whole, one walk over the layers ----

    @override
    def groups(self) -> tuple[CacheGroupDeclaration, ...]:
        layout = self._cache_layout
        dcp_size = int(getattr(self.attn_config, "dcp_size", 1))
        sliding_window = int(
            (
                self.draft_model_config.hf_config
                if self.draft_attn_config is not None
                else self.model_config.hf_config
            ).sliding_window
        )
        if sliding_window <= 0 or self.prefix_granularity % sliding_window:
            raise ValueError(
                "DeepSeek V4 sliding_window must divide the prefix granularity"
            )
        if sliding_window % V4_KERNEL_BLOCK_ROWS:
            raise ValueError(
                "DeepSeek V4 sliding_window must be divisible by the SWA kernel page"
            )
        ratios = tuple(int(ratio) for ratio in layout.layer_ratio)
        if any(ratio not in (1, 4, 128) for ratio in ratios):
            raise ValueError("DeepSeek V4 layer ratios must be 1, 4, or 128")

        c4_window = v4_c4_state_window(self.decode_input_tokens)
        swa_bytes = layout.swa_block_bytes(V4_KERNEL_BLOCK_ROWS)
        stride_alignment = layout.swa_token_stride
        ratio_counts = Counter(ratios)
        declared: dict[str, tuple[CacheGroupSpec, tuple[CacheFieldSpec, ...]]] = {}
        # A group's plane numbering: one plane per layer that stores in it.
        occurrences: Counter[str] = Counter()

        def declare(spec: CacheGroupSpec, *fields: CacheFieldSpec) -> None:
            """Add fields to a group, creating it the first time it is named.

            The group id is written once -- in ``spec`` -- right where its
            fields are added, so the two halves cannot drift apart.
            """
            existing = declared.get(spec.group_id)
            declared[spec.group_id] = (
                spec if existing is None else existing[0],
                (() if existing is None else existing[1]) + fields,
            )

        swa_spec = v4_swa_kv_spec(self.model_config.hf_config)
        for layer_id, ratio in enumerate(ratios):
            swa_slot = occurrences[swa_spec.group_id]
            occurrences[swa_spec.group_id] += 1
            declare(
                swa_spec,
                CacheFieldSpec(
                    f"layer.{layer_id}.swa",
                    f"unit.{swa_slot}",
                    (swa_bytes,),
                    "uint8",
                    exact_page_stride=False,
                    page_stride_alignment_bytes=stride_alignment,
                ),
            )
            if ratio == 1:
                continue

            compressed_spec = cyclic_history_spec(
                v4_compressed_kv_spec(ratio), dcp_size=dcp_size
            )
            state_spec = v4_compressor_state_spec(ratio, c4_state_window=c4_window)
            compressed_slot = occurrences[compressed_spec.group_id]
            occurrences[compressed_spec.group_id] += 1
            state_slot = occurrences[state_spec.group_id]
            occurrences[state_spec.group_id] += 1
            declare(
                compressed_spec,
                CacheFieldSpec(
                    f"layer.{layer_id}.compressed_kv",
                    f"unit.{compressed_slot}",
                    (layout.swa_block_bytes(int(compressed_spec.rows_per_page)),),
                    "uint8",
                    exact_page_stride=False,
                    page_stride_alignment_bytes=stride_alignment,
                ),
            )
            declare(
                state_spec,
                CacheFieldSpec(
                    f"layer.{layer_id}.compressor_state",
                    f"unit.{state_slot}",
                    (
                        layout.compressor_state_block_size(ratio),
                        layout.head_dim * (2 if ratio == 4 else 1) * 2,
                    ),
                    "float32",
                    exact_page_stride=False,
                ),
            )
            if ratio != 4:
                continue

            # The indexer's K shares the compressed chain's group but sits on
            # planes after every compressed tenant; its state is its own group.
            indexer_state_spec = v4_indexer_state_spec(c4_state_window=c4_window)
            indexer_state_slot = occurrences[indexer_state_spec.group_id]
            occurrences[indexer_state_spec.group_id] += 1
            declare(
                compressed_spec,
                CacheFieldSpec(
                    f"layer.{layer_id}.indexer_kv",
                    f"unit.{ratio_counts[4] + compressed_slot}",
                    (V4_KERNEL_BLOCK_ROWS * layout.indexer_row_bytes,),
                    "uint8",
                    exact_page_stride=False,
                ),
            )
            declare(
                indexer_state_spec,
                CacheFieldSpec(
                    f"layer.{layer_id}.indexer_state",
                    f"unit.{indexer_state_slot}",
                    (
                        layout.compressor_state_block_size(4),
                        layout.index_head_dim * 2 * 2,
                    ),
                    "float32",
                    exact_page_stride=False,
                ),
            )

        groups = tuple(declared.values())
        if not self.pd_disaggregation_enabled:
            return groups
        policies = apply_pd_transfer_policies(tuple(spec for spec, _ in groups))
        return tuple(
            (spec, fields) for spec, (_, fields) in zip(policies, groups, strict=True)
        )

    # ---- packing: powers of two so every field stride stays aligned ----

    @override
    def packing(self, groups: tuple[CacheGroupDeclaration, ...]) -> Mapping[str, int]:
        raw_bytes = {
            spec.group_id: sum(field.payload_bytes for field in fields)
            for spec, fields in groups
        }
        largest = max(raw_bytes.values())
        # Exact byte ratios would inflate a parent through their large common
        # LCM; powers of two keep every field stride naturally aligned.
        return {
            group_id: 1 << max(0, (largest // group_bytes).bit_length() - 1)
            for group_id, group_bytes in raw_bytes.items()
        }

    @override
    def check_layout(self, layout: CacheLayout) -> None:
        """Every group's CacheBlock span must divide the identity grain.

        A page id is derived by dividing a prefix-granularity token offset by
        the group's block span; a remainder would put two different token
        ranges on one page id.
        """
        for spec in self._group_specs:
            if layout.prefix_granularity % spec.block_granularity:
                raise ValueError(
                    f"group {spec.group_id!r} cache block tokens must divide "
                    f"prefix_granularity={layout.prefix_granularity}"
                )

    # ---- capacity: per-group demand at the configured concurrency ----

    @override
    def num_lcm_blocks(self, layout: CacheLayout) -> int:
        """Parents the per-group demand needs, not what the budget affords.

        The budget only bounds the search: V4 sizes from what each group
        demands at the configured concurrency, and the token limit enters
        through :meth:`token_capacity`'s upper bound rather than as a cap on
        parents (a V4 parent spans the kernel page, not the prefix grain).
        """
        budgeted = self.cache_budget_bytes // layout.lcm_block_bytes - 1
        if budgeted < 1:
            raise ValueError(
                "DeepSeek V4 cache budget must hold a null parent and one usable parent"
            )
        return self.parents_needed(layout, self.token_capacity(layout, budgeted))

    @override
    def token_capacity(self, layout: CacheLayout, num_lcm_blocks: int) -> int:
        return self._capacity_from_parents(
            layout,
            num_lcm_blocks,
            upper_bound=(
                self.token_limit
                if self.token_limit is not None
                # Tokens per parent = packing x the group block span (kernel
                # page), independent of the scheduler prefix granularity.
                else num_lcm_blocks * self._max_packing(layout) * DEEPSEEK_V4_PAGE_SIZE
            ),
        )

    # ---- extras ----

    @override
    def pool_options(self) -> DeepseekV4PoolOptions:
        return DeepseekV4PoolOptions(layout=self._cache_layout)

    def _use_fp4_indexer(self, hf_config) -> bool:
        forced = getattr(self.server_args, "attention_use_fp4_indexer_cache", None)
        attention_config = getattr(hf_config, "attention_config", None)
        if isinstance(attention_config, dict):
            configured = attention_config.get("use_fp4_indexer_cache", None)
        else:
            configured = getattr(attention_config, "use_fp4_indexer_cache", None)
        requested = forced if forced is not None else configured
        return dsv4_indexer_cache_format(requested) == "mxfp4"
