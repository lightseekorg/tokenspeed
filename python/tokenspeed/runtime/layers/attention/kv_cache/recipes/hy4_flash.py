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

"""Hy4-flash cache recipe: hybrid compressed-DSA and SWA-MLA paged groups."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property

import torch
from typing_extensions import override

from tokenspeed.runtime.configs.hy4_flash_config import SLIDING_ATTENTION
from tokenspeed.runtime.layers.attention.configs.dsa import dsa_index_k_row_bytes
from tokenspeed.runtime.layers.attention.configs.hy4_flash import (
    Hy4FlashAttentionConfig,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.base import (
    CacheGroupDeclaration,
    CacheRecipe,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    require_positive_int,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    CacheLayout,
    scatter_stored_dtype_name,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    CacheGroupSpec,
    apply_pd_transfer_policies,
)

HY4_FLASH_LOGICAL_BLOCK_TOKENS = 64
_FLAT_KV_ALIGNMENT_BYTES = 256


@dataclass(frozen=True)
class Hy4FlashPoolOptions:
    """Fixed request-local compressed-indexer tail geometry for Hy4-flash."""

    index_kpool: int
    tail_extra_slots: int
    index_head_dim: int
    num_request_slots: int
    dsa_layer_ids: tuple[int, ...]

    @property
    def tail_width(self) -> int:
        return self.index_kpool + self.tail_extra_slots

    @property
    def workspace_bytes(self) -> int:
        token_key_bytes = (
            len(self.dsa_layer_ids)
            * self.num_request_slots
            * self.tail_width
            * self.index_head_dim
            * torch.bfloat16.itemsize
        )
        pos0_bytes = self.num_request_slots * 3 * torch.int64.itemsize
        return token_key_bytes + pos0_bytes


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _choose_group_packing(group_bytes: Mapping[str, int]) -> dict[str, int]:
    """Choose compact integer packing ratios minimizing LCM padding."""
    if len(group_bytes) == 1:
        gid = next(iter(group_bytes))
        return {gid: 1}
    if set(group_bytes) != {"dsa", "swa"}:
        return dict.fromkeys(group_bytes, 1)

    dsa_bytes = group_bytes["dsa"]
    swa_bytes = group_bytes["swa"]
    best_pair = (1, 1)
    best_pad = float("inf")
    for p_swa in range(1, 5):
        for p_dsa in range(1, 9):
            if math.gcd(p_dsa, p_swa) != 1:
                continue
            dsa_span = p_dsa * dsa_bytes
            swa_span = p_swa * swa_bytes
            parent = max(dsa_span, swa_span)
            pad = max(parent - dsa_span, parent - swa_span) / parent
            if pad < best_pad - 1e-12 or (
                abs(pad - best_pad) <= 1e-12 and (p_dsa + p_swa) < sum(best_pair)
            ):
                best_pad = pad
                best_pair = (p_dsa, p_swa)
    return {"dsa": best_pair[0], "swa": best_pair[1]}


class Hy4FlashRecipe(CacheRecipe):
    """Hybrid DSA (latent KV + 4:1 pooled FP8 index-K) and SWA-MLA cache recipe."""

    family = "hy4_flash"

    @cached_property
    def _hy4_spec(self) -> Hy4FlashAttentionConfig:
        spec = self.attn_config.component(Hy4FlashAttentionConfig)
        if spec is None:
            raise TypeError(
                "Hy4FlashRecipe requires Hy4FlashAttentionConfig on attn_config"
            )
        return spec

    @property
    @override
    def prefix_granularity(self) -> int:
        return HY4_FLASH_LOGICAL_BLOCK_TOKENS

    @property
    @override
    def layer_types(self) -> tuple[str, ...]:
        return tuple(self._hy4_spec.cache_layer_types)

    @property
    @override
    def max_padding_fraction(self) -> float:
        return 0.25

    @cached_property
    def tail_extra_slots(self) -> int:
        if self.server_args.speculative_algorithm is None:
            return 0
        extra = int(self.server_args.speculative_num_draft_tokens or 0)
        if extra < 0:
            raise ValueError(
                f"speculative_num_draft_tokens must be non-negative, got {extra}"
            )
        return extra

    @override
    def groups(self) -> tuple[CacheGroupDeclaration, ...]:
        config = self.attn_config
        spec = self._hy4_spec
        if config.kvp_size != 1 or config.qcp_size != 1:
            raise NotImplementedError(
                "Hy4-flash cache currently requires KVP=1 and QCP=1"
            )
        if config.kv_cache_quant_method not in (None, "none"):
            raise ValueError(
                "Hy4-flash cache does not support kv_cache_quant_method="
                f"{config.kv_cache_quant_method!r}"
            )
        if config.kv_cache_dtype not in (
            torch.bfloat16,
            torch.float16,
            torch.float8_e4m3fn,
        ):
            raise ValueError(
                "Hy4-flash cache supports bf16, fp16, or fp8_e4m3fn KV cache, "
                f"got {config.kv_cache_dtype}"
            )
        if spec.index_k_format != "fp8_scaled":
            raise ValueError(
                f"Hy4-flash requires fp8_scaled index-K, got {spec.index_k_format!r}"
            )
        if self.num_draft_layers != 0:
            raise ValueError(
                "Hy4-flash DFly2 shares target KV cache layers and expects "
                f"num_draft_layers == 0, got {self.num_draft_layers}"
            )
        if len(spec.cache_layer_types) != self.num_target_layers:
            raise ValueError(
                f"Hy4-flash cache_layer_types length ({len(spec.cache_layer_types)}) "
                f"!= num_target_layers ({self.num_target_layers})"
            )

        kpool = require_positive_int("index_kpool", spec.index_kpool)
        if HY4_FLASH_LOGICAL_BLOCK_TOKENS % kpool != 0:
            raise ValueError(
                f"HY4_FLASH_LOGICAL_BLOCK_TOKENS ({HY4_FLASH_LOGICAL_BLOCK_TOKENS}) "
                f"must be divisible by index_kpool ({kpool})"
            )
        index_rows_per_page = HY4_FLASH_LOGICAL_BLOCK_TOKENS // kpool
        index_head_dim = require_positive_int("index_head_dim", spec.index_head_dim)
        index_row_bytes = dsa_index_k_row_bytes(index_head_dim)
        kv_dtype_name = scatter_stored_dtype_name(config.kv_cache_dtype)

        fields_by_group: dict[str, list[CacheFieldSpec]] = {}
        for layer_id, layer_type in enumerate(spec.cache_layer_types):
            if layer_type == FULL_ATTENTION:
                gid = "dsa"
                fields = fields_by_group.setdefault(gid, [])
                fields.append(
                    CacheFieldSpec(
                        field_id=f"layer.{layer_id}.latent_kv",
                        plane_id="flatkv",
                        shape=(HY4_FLASH_LOGICAL_BLOCK_TOKENS, 1, spec.kv_cache_dim),
                        dtype=kv_dtype_name,
                        exact_page_stride=False,
                        page_stride_alignment_bytes=_FLAT_KV_ALIGNMENT_BYTES,
                    )
                )
                fields.append(
                    CacheFieldSpec(
                        field_id=f"layer.{layer_id}.index_k",
                        plane_id="flatkv",
                        shape=(index_rows_per_page, index_row_bytes),
                        dtype="uint8",
                        exact_page_stride=False,
                        page_stride_alignment_bytes=_FLAT_KV_ALIGNMENT_BYTES,
                    )
                )
            elif layer_type == SLIDING_ATTENTION:
                gid = "swa"
                fields = fields_by_group.setdefault(gid, [])
                fields.append(
                    CacheFieldSpec(
                        field_id=f"layer.{layer_id}.latent_kv",
                        plane_id="flatkv",
                        shape=(
                            HY4_FLASH_LOGICAL_BLOCK_TOKENS,
                            1,
                            spec.swa.kv_cache_dim,
                        ),
                        dtype=kv_dtype_name,
                        exact_page_stride=False,
                        page_stride_alignment_bytes=_FLAT_KV_ALIGNMENT_BYTES,
                    )
                )
            else:
                raise ValueError(
                    f"Unsupported Hy4-flash cache layer type {layer_type!r} "
                    f"at layer {layer_id}"
                )

        specs: list[CacheGroupSpec] = []
        if "dsa" in fields_by_group:
            specs.append(
                CacheGroupSpec(
                    group_id="dsa",
                    attn_kind="full_history",
                    rows_per_page=HY4_FLASH_LOGICAL_BLOCK_TOKENS,
                    entry_stride_tokens=1,
                    sliding_window_tokens=None,
                    replayable=False,
                )
            )
        if "swa" in fields_by_group:
            window = require_positive_int(
                "sliding_window_tokens", spec.swa.sliding_window_tokens
            )
            specs.append(
                CacheGroupSpec(
                    group_id="swa",
                    attn_kind="sliding_window",
                    rows_per_page=HY4_FLASH_LOGICAL_BLOCK_TOKENS,
                    entry_stride_tokens=1,
                    sliding_window_tokens=window,
                    replayable=False,
                )
            )
        if self.pd_disaggregation_enabled:
            specs = apply_pd_transfer_policies(specs)
        return tuple(
            (group_spec, tuple(fields_by_group[group_spec.group_id]))
            for group_spec in specs
        )

    @override
    def packing(self, groups: Sequence[CacheGroupDeclaration]) -> Mapping[str, int]:
        group_bytes: dict[str, int] = {}
        for group_spec, fields in groups:
            total = 0
            for field in fields:
                total += _align_up(field.payload_bytes, _FLAT_KV_ALIGNMENT_BYTES)
            group_bytes[group_spec.group_id] = total
        return _choose_group_packing(group_bytes)

    @override
    def workspace_bytes(self) -> int:
        return self.pool_options().workspace_bytes

    @override
    def pool_options(self) -> Hy4FlashPoolOptions:
        spec = self._hy4_spec
        return Hy4FlashPoolOptions(
            index_kpool=require_positive_int("index_kpool", spec.index_kpool),
            tail_extra_slots=self.tail_extra_slots,
            index_head_dim=require_positive_int("index_head_dim", spec.index_head_dim),
            num_request_slots=self.attn_config.max_resident_requests + 2,
            dsa_layer_ids=spec.dsa_layer_ids,
        )

    @override
    def select_num_lcm_blocks(self, layout: CacheLayout, *, max_lcm_blocks: int) -> int:
        if self.token_limit is None:
            return max_lcm_blocks
        return min(max_lcm_blocks, self.parents_needed(layout, self.token_limit))

    @override
    def token_capacity(self, layout: CacheLayout, num_lcm_blocks: int) -> int:
        packing = dict(layout.group_packing)
        primary_packing = packing.get("dsa") or next(iter(packing.values()))
        upper = (
            self.token_limit
            if self.token_limit is not None
            else num_lcm_blocks * primary_packing * layout.prefix_granularity
        )
        return self._capacity_from_parents(layout, num_lcm_blocks, upper_bound=upper)


__all__ = [
    "HY4_FLASH_LOGICAL_BLOCK_TOKENS",
    "Hy4FlashPoolOptions",
    "Hy4FlashRecipe",
]
