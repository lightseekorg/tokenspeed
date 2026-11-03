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

"""FP8 DSA and SWA history packed into one shared LCM plane."""

from collections.abc import Mapping, Sequence

import torch
from typing_extensions import override

from tokenspeed.runtime.layers.attention.configs.dots3_note import (
    Dots3NoteAttentionConfig,
)
from tokenspeed.runtime.layers.attention.configs.dsa import dsa_index_k_row_bytes
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig
from tokenspeed.runtime.layers.attention.kv_cache.recipes.base import CacheRecipe
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    CacheLayout,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    CacheGroupDeclaration,
    CacheGroupSpec,
    apply_pd_transfer_policies,
)

# A multiple of both groups' spans (64-row DSA, 32-row SWA) that exceeds each:
# a grain equal to a field's page size would request a flat, unstrided view.
_DOTS3_NOTE_PREFIX_GRANULARITY = 128


class Dots3NoteRecipe(CacheRecipe):
    family = "dots3_note"

    @property
    @override
    def prefix_granularity(self) -> int:
        return _DOTS3_NOTE_PREFIX_GRANULARITY

    @property
    @override
    def layer_types(self) -> tuple[str, ...]:
        spec = self.attn_config.component(Dots3NoteAttentionConfig)
        return spec.cache_layer_types + ("sliding_attention",) * self.num_draft_layers

    @property
    @override
    def max_padding_fraction(self) -> float:
        return 0.03

    @override
    def groups(self) -> tuple[CacheGroupDeclaration, ...]:
        config = self.attn_config
        spec = config.component(Dots3NoteAttentionConfig)
        if config.kvp_size != 1 or config.qcp_size != 1:
            raise NotImplementedError("dots3-note cache currently requires KVP=QCP=1")
        if (
            config.kv_cache_dtype != torch.float8_e4m3fn
            or config.kv_cache_quant_method not in (None, "none")
            or spec.index_k_format != "fp8_scaled"
        ):
            raise ValueError(
                "dots3-note requires unit-scale E4M3 KV and scaled FP8 index-K"
            )
        if self.server_args.pipeline_parallel_size != 1:
            raise NotImplementedError("dots3-note cache currently requires PP=1")
        if (
            self.num_target_layers != 46
            or len(spec.cache_layer_types) != 46
            or spec.cache_layer_types.count("full_attention") != 13
            or spec.cache_layer_types.count("sliding_attention") != 33
            or (spec.kv_cache_dim, spec.swa.kv_cache_dim) != (576, 1088)
            or (spec.kv_lora_rank, spec.qk_rope_head_dim) != (512, 64)
            or (spec.swa.kv_lora_rank, spec.swa.qk_rope_head_dim) != (1024, 64)
            or spec.index_head_dim != 128
            or spec.swa.sliding_window_tokens != 513
        ):
            raise ValueError("dots3-note cache requires the note checkpoint geometry")
        if self.num_draft_layers not in (0, 1):
            raise ValueError("dots3-note supports one MTP cache layer")
        if self.num_draft_layers:
            draft = self.draft_attn_config
            draft_spec = draft.component(MLAConfig)
            if (
                draft.kv_cache_dtype != config.kv_cache_dtype
                or draft.kv_cache_quant_method not in (None, "none")
                or draft.kernel_page_size != 32
                or draft_spec.kv_cache_dim != spec.swa.kv_cache_dim
                or (draft_spec.kv_lora_rank, draft_spec.qk_rope_head_dim)
                != (spec.swa.kv_lora_rank, spec.swa.qk_rope_head_dim)
                or draft_spec.sliding_window_tokens != spec.swa.sliding_window_tokens
            ):
                raise ValueError(
                    "dots3-note MTP must use the target SWA cache geometry"
                )

        fields: dict[str, list[CacheFieldSpec]] = {"dsa": [], "swa": []}
        for layer, layer_type in enumerate(self.layer_types):
            is_dsa = layer_type == "full_attention"
            gid, rows = ("dsa", 64) if is_dsa else ("swa", 32)
            width = spec.kv_cache_dim if is_dsa else spec.swa.kv_cache_dim
            layer_fields = [("latent_kv", (rows, 1, width))]
            if is_dsa:
                layer_fields.append(
                    ("index_k", (rows, dsa_index_k_row_bytes(spec.index_head_dim)))
                )
            for name, shape in layer_fields:
                fields[gid].append(
                    CacheFieldSpec(
                        field_id=f"layer.{layer}.{name}",
                        plane_id="flatkv",
                        shape=shape,
                        dtype="uint8",
                        exact_page_stride=False,
                        page_stride_alignment_bytes=256,
                    )
                )
        specs = [
            CacheGroupSpec(
                group_id=gid,
                attn_kind="full_history" if gid == "dsa" else "sliding_window",
                rows_per_page=rows,
                entry_stride_tokens=1,
                sliding_window_tokens=None if gid == "dsa" else 513,
                replayable=False,
            )
            for gid, rows in (("dsa", 64), ("swa", 32))
        ]
        if self.pd_disaggregation_enabled:
            specs = apply_pd_transfer_policies(specs)
        return tuple((group, tuple(fields[group.group_id])) for group in specs)

    @override
    def packing(self, groups: Sequence[CacheGroupDeclaration]) -> Mapping[str, int]:
        return {"dsa": 2, "swa": 1}

    @override
    def select_num_lcm_blocks(self, layout: CacheLayout, *, max_lcm_blocks: int) -> int:
        return self.parents_needed(layout, self.token_capacity(layout, max_lcm_blocks))

    @override
    def token_capacity(self, layout: CacheLayout, num_lcm_blocks: int) -> int:
        return self._capacity_from_parents(
            layout,
            num_lcm_blocks,
            upper_bound=(
                self.token_limit
                if self.token_limit is not None
                else num_lcm_blocks * 2 * 64
            ),
        )
