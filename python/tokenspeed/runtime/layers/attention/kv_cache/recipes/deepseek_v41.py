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

"""V4.1 Flash: four history groups sharing one 1,382,400-byte LCM plane.

Only source layers own global fields; Reuse/Reindex layers read those views.
The FP32 compressor input is position-addressed history, not a mutable
per-request ring. Prefix matching, allocation, protection, and reclamation
therefore stay in LCM, including odd-token chunk boundaries.
"""

from collections.abc import Mapping, Sequence

from typing_extensions import override

from tokenspeed.runtime.layers.attention.deepseek_v41_geometry import (
    V41_COMPRESSOR_TAIL_GROUP_ID,
    V41_GLOBAL_R1_GROUP_ID,
    V41_GLOBAL_R2_GROUP_ID,
    V41_GLOBAL_ROW_BYTES,
    V41_GROUP_GEOMETRY,
    V41_GROUP_PACKING,
    V41_HEAD_DIM,
    V41_INDEX_HEAD_DIM,
    V41_INDEX_ROW_BYTES,
    V41_SWA_GROUP_ID,
    V41_SWA_ROW_BYTES,
    V41_WINDOW_SIZE,
    v41_layer_mapping,
    v41_table_widths,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.base import CacheRecipe
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    CacheLayout,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    CacheGroupDeclaration,
    CacheGroupSpec,
)


class DeepseekV41Recipe(CacheRecipe):
    """Target-only Flash FlatKV; packing is independent of prefix granularity."""

    family = "deepseek_v41"

    @property
    @override
    def layer_types(self) -> tuple[str, ...]:
        return ("sliding_attention",) * self.num_target_layers

    @property
    @override
    def max_padding_fraction(self) -> float:
        return 0.05

    @override
    def groups(self) -> tuple[CacheGroupDeclaration, ...]:
        if self.num_draft_layers or self.decode_input_tokens != 1:
            raise NotImplementedError(
                "DeepSeek V4.1 FlatKV does not support speculation yet"
            )
        if self.pd_disaggregation_enabled:
            raise NotImplementedError(
                "DeepSeek V4.1 PD cache transfer is not validated"
            )
        if int(self.server_args.pipeline_parallel_size) != 1:
            raise NotImplementedError("DeepSeek V4.1 shared-owner FlatKV requires PP=1")
        hf = self.model_config.hf_config
        hf = getattr(hf, "text_config", hf)
        ratios = tuple(int(r) for r in hf.compress_ratios[: self.num_target_layers])
        owners = tuple(int(layer) for layer in hf.kv_source_layers)
        if (
            self.num_target_layers != 40
            or int(hf.head_dim) != V41_HEAD_DIM
            or int(hf.index_head_dim) != V41_INDEX_HEAD_DIM
            or int(hf.sliding_window) != V41_WINDOW_SIZE
            or len(ratios) != self.num_target_layers
            or any(r not in (0, 1, 2) for r in ratios)
            or owners != (2, 8, 14, 20)
            or tuple(ratios[layer] for layer in owners) != (2, 2, 2, 1)
        ):
            raise ValueError(
                "DeepSeek V4.1 FlatKV packing requires the Flash cache geometry"
            )
        v41_layer_mapping(
            ratios,
            owners,
            tuple(hf.index_source_layers),
            int(hf.candidate_source_layer),
        )
        if self.prefix_granularity <= 0 or any(
            self.prefix_granularity % (rows * stride)
            for rows, stride in V41_GROUP_GEOMETRY.values()
        ):
            raise ValueError(
                "DeepSeek V4.1 prefix granularity must be a positive multiple of 128"
            )

        fields: dict[str, list[CacheFieldSpec]] = {
            gid: [] for gid in V41_GROUP_GEOMETRY
        }

        def add(
            gid: str, layer: int, name: str, shape: tuple[int, ...], dtype: str
        ) -> None:
            fields[gid].append(
                CacheFieldSpec(
                    field_id=f"layer.{layer}.{name}",
                    plane_id="flatkv",
                    shape=shape,
                    dtype=dtype,
                    exact_page_stride=False,
                    page_stride_alignment_bytes=256,
                )
            )

        for layer in range(self.num_target_layers):
            add(V41_SWA_GROUP_ID, layer, "swa", (64, V41_SWA_ROW_BYTES), "uint8")
        for owner in owners:
            gid = (
                V41_GLOBAL_R2_GROUP_ID if ratios[owner] == 2 else V41_GLOBAL_R1_GROUP_ID
            )
            add(gid, owner, "global_kv", (64, V41_GLOBAL_ROW_BYTES), "uint8")
            add(gid, owner, "index_k", (64, V41_INDEX_ROW_BYTES), "uint8")
            if ratios[owner] == 2:
                add(
                    V41_COMPRESSOR_TAIL_GROUP_ID,
                    owner,
                    "compressor_tail",
                    (2, 2, V41_HEAD_DIM),
                    "float32",
                )

        # Admission can be ahead of the completed forward. Retain its input
        # window as well as the unfinished pair; the allocator also reserves
        # in-flight pages through the shared scheduler_limits demand formula.
        protection = (1 + self.overlap_schedule_depth) * self.decode_input_tokens
        windows = {
            V41_SWA_GROUP_ID: V41_WINDOW_SIZE + protection,
            V41_GLOBAL_R2_GROUP_ID: None,
            V41_GLOBAL_R1_GROUP_ID: None,
            V41_COMPRESSOR_TAIL_GROUP_ID: 2 + protection,
        }
        return tuple(
            (
                CacheGroupSpec(
                    group_id=gid,
                    retention=(
                        "full_history" if windows[gid] is None else "sliding_window"
                    ),
                    rows_per_page=rows,
                    entry_stride_tokens=stride,
                    sliding_window_tokens=windows[gid],
                    family="history",
                    transfer_policy=None,
                    checkpoint_granularity=None,
                ),
                tuple(fields[gid]),
            )
            for gid, (rows, stride) in V41_GROUP_GEOMETRY.items()
        )

    @override
    def packing(self, groups: Sequence[CacheGroupDeclaration]) -> Mapping[str, int]:
        return {spec.group_id: V41_GROUP_PACKING[spec.group_id] for spec, _ in groups}

    @override
    def check_layout(self, layout: CacheLayout) -> None:
        if layout.lcm_block_bytes != 1_382_400 or len(layout.plane_bytes) != 1:
            raise ValueError("DeepSeek V4.1 Flash requires one 1,382,400-byte plane")

    @override
    def num_lcm_blocks(self, layout: CacheLayout) -> int:
        budgeted = self._budgeted_parents(
            self.cache_budget_bytes - self.workspace_bytes(), layout.lcm_block_bytes
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
                else num_lcm_blocks * V41_GROUP_PACKING[V41_GLOBAL_R1_GROUP_ID] * 64
            ),
        )

    @override
    def workspace_bytes(self) -> int:
        """Reserve decode tables and bounded per-forward index/attention scratch.

        The kernel baseline tiles queries; it must never gather all selected
        BF16 KV rows for an entire prefill at once. Candidate/top-k ids still
        live for every scheduled query across Reindex/Reuse layers.
        """
        max_bs = self.attn_config.max_bs
        horizon = (1 + self.overlap_schedule_depth) * self.decode_input_tokens
        tables = (
            4
            * max_bs
            * sum(v41_table_widths(self.attn_config.context_len, horizon).values())
        )
        queries = max(0, int(self.server_args.chunked_prefill_size)) + int(
            self.server_args.max_num_seqs
        )
        # Request ids, positions, starts, lengths, four write-location vectors;
        # two generations allow an extend followed by a decode metadata view.
        metadata = 2 * (tables + max_bs * 96 + 4)
        # Both MIXED windows together are bounded by queries. Include the
        # packed current SWA rows and two generations of selection records.
        selections = queries * ((2048 + 2 * 512 + 128) * 8 + 528)
        # ponytail: reserve a 64 MiB tiled baseline workspace; replace this
        # bound with the native kernel's declared workspace when optimized.
        # Ratio-2 pooling returns T FP32 rows (inactive positions are -1), not
        # a compact T/2 result. History/softmax scratch is tiled to eight rows.
        compressor_output = queries * V41_HEAD_DIM * 4
        return metadata + selections + compressor_output + (64 << 20)
