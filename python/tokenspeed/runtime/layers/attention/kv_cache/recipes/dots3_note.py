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

"""Dots3-note Plan A: four history groups sharing one BF16 FlatKV plane.

Full layers retain latent KV and page-planar FP8 index keys; sliding layers
are split in layer order into groups of 15/15/3. Native MTP adds one independent
SWA layer, packed 15 pages per parent. All groups participate in normal prefix
caching (no replay). See docs/design/dots3-note.md, including section 10.
"""

from collections.abc import Mapping, Sequence
from dataclasses import replace

import torch
from typing_extensions import override

from tokenspeed.runtime.layers.attention.configs.dots3_note import (
    Dots3NoteAttnConfig,
    is_dots3_note_config,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.base import CacheRecipe
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    CacheLayout,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    FULL_ATTENTION,
    CacheGroupDeclaration,
    CacheGroupSpec,
    apply_pd_transfer_policies,
)


class Dots3NoteRecipe(CacheRecipe):
    family = "dots3_note"

    @property
    @override
    def layer_types(self) -> tuple[str, ...]:
        target = self.attn_config.component(Dots3NoteAttnConfig).cache_layer_types
        return target + ("sliding_attention",) * self.num_draft_layers

    @property
    @override
    def max_padding_fraction(self) -> float:
        return 0.026 if self.draft_attn_config is not None else 0.025

    @override
    def groups(self) -> tuple[CacheGroupDeclaration, ...]:
        config = self.attn_config
        spec = config.component(Dots3NoteAttnConfig)
        if spec is None:
            raise ValueError("dots3 Plan A requires Dots3NoteAttnConfig")
        algorithm = self.server_args.speculative_algorithm
        if (
            algorithm not in (None, "MTP")
            or config.is_draft
            or config.draft_block_decode
        ):
            raise NotImplementedError(
                "dots3 Plan A supports only native MTP speculation"
            )
        if algorithm is None:
            if (
                self.draft_model_config is not None
                or self.draft_attn_config is not None
                or self.server_args.speculative_draft_model_path is not None
                or config.speculative_num_steps != 0
                or config.speculative_num_draft_tokens != 1
                or self.decode_input_tokens != 1
            ):
                raise NotImplementedError(
                    "dots3 drafts require speculative_algorithm=MTP"
                )
        else:
            if (
                self.draft_model_config is None
                or self.draft_attn_config is None
                or self.num_draft_layers != 1
                or not is_dots3_note_config(self.draft_model_config.hf_text_config)
            ):
                raise ValueError(
                    "dots3 MTP requires an actual one-layer dots3 draft config"
                )
            expected_draft = replace(
                config,
                components=(
                    replace(
                        spec,
                        num_attention_heads=spec.swa.num_attention_heads,
                        num_kv_heads=spec.swa.num_kv_heads,
                        head_dim=spec.swa.head_dim,
                        cache_layer_types=("sliding_attention",),
                    ),
                ),
                kernel_page_size=32,
                is_draft=True,
            )
            if self.draft_attn_config != expected_draft:
                raise ValueError(
                    "dots3 MTP requires a matching BF16 page32 SWA MLA draft"
                )
            if (
                config.speculative_num_steps < 1
                or config.speculative_num_steps
                != self.server_args.speculative_num_steps
                or config.speculative_num_draft_tokens
                != config.speculative_num_steps + 1
                or config.speculative_num_draft_tokens
                != self.server_args.speculative_num_draft_tokens
                or self.decode_input_tokens != config.speculative_num_draft_tokens
                or self.server_args.speculative_eagle_topk != 1
            ):
                raise ValueError("dots3 MTP requires matching chain speculation widths")
            draft_path = self.server_args.speculative_draft_model_path
            if draft_path is not None and draft_path != self.server_args.model:
                raise ValueError("dots3 MTP requires the target checkpoint as draft")
        if self.server_args.pipeline_parallel_size != 1:
            raise NotImplementedError("dots3 Plan A requires PP=1")
        if self.server_args.mapping.attn.qcp_size != 1 or config.dcp_size != 1:
            raise NotImplementedError("dots3 Plan A requires QCP=1 and DCP=1")
        if (
            config.dtype != torch.bfloat16
            or config.kv_cache_dtype != torch.bfloat16
            or config.kv_cache_mxfp8
            or config.kv_cache_quant_method != "none"
        ):
            raise ValueError("dots3 Plan A requires unquantized BF16 latent KV")
        if config.kernel_page_size is not None:
            raise ValueError(
                "dots3 Plan A requires per-group page sizes, not a global override"
            )
        if self.prefix_granularity <= 0 or self.prefix_granularity % 64:
            raise ValueError(
                "dots3 prefix granularity must be a positive multiple of 64"
            )
        if (
            self.num_target_layers != 46
            or len(spec.cache_layer_types) != 46
            or spec.cache_layer_types.count(FULL_ATTENTION) != 13
            or spec.cache_layer_types.count("sliding_attention") != 33
        ):
            raise ValueError("dots3 Plan A requires 46 layers: 13 Full and 33 SWA")

        full, swa = spec.full, spec.swa
        # Validate the resolved attention geometry, not a second HF layer map.
        for leaf, expected in (
            (full, (128, 128, 192, 512, 128, 64, 128, 576)),
            (swa, (64, 64, 256, 1024, 192, 64, 128, 1088)),
        ):
            if (
                leaf.num_attention_heads,
                leaf.num_kv_heads,
                leaf.head_dim,
                leaf.kv_lora_rank,
                leaf.qk_nope_head_dim,
                leaf.qk_rope_head_dim,
                leaf.v_head_dim,
                leaf.kv_cache_dim,
            ) != expected:
                raise ValueError(
                    "dots3 Plan A requires the Full/SWA attention geometry"
                )
        if (
            (full.index_n_heads, full.index_head_dim, full.index_topk)
            != (64, 128, 2048)
            or full.index_k_format != "fp8_scaled"
            or full.index_kpool is not None
            or spec.sliding_window_tokens != 513
            or swa.sliding_window_tokens != 513
            or full.sliding_window_tokens is not None
        ):
            raise ValueError(
                "dots3 Plan A requires fp8_scaled index64x128/topk2048 and SWA513"
            )
        if (
            spec.attn_tp_size <= 0
            or 64 % spec.attn_tp_size
            or full.attn_tp_size != spec.attn_tp_size
            or swa.attn_tp_size != spec.attn_tp_size
        ):
            raise ValueError(
                "dots3 attention TP size must agree across leaves and divide 64"
            )
        if spec.backend_name != "dots3_note" or any(
            leaf.backend_name != "triton" for leaf in (full, swa)
        ):
            raise ValueError(
                "dots3 Plan A workspace requires the Triton attention leaves"
            )

        full_layers = [
            i for i, kind in enumerate(spec.cache_layer_types) if kind == FULL_ATTENTION
        ]
        swa_layers = [
            i
            for i, kind in enumerate(spec.cache_layer_types)
            if kind == "sliding_attention"
        ]
        group_layers = [
            ("full", full_layers, 64, full.kv_cache_dim),
            ("swa.0", swa_layers[:15], 32, swa.kv_cache_dim),
            ("swa.1", swa_layers[15:30], 32, swa.kv_cache_dim),
            ("swa.2", swa_layers[30:], 32, swa.kv_cache_dim),
        ]
        if self.num_draft_layers:
            group_layers.append(
                ("draft.swa", [self.num_target_layers], 32, swa.kv_cache_dim)
            )
        declarations = []
        for gid, layers, rows, width in group_layers:
            fields = []
            for layer in layers:
                fields.append(
                    CacheFieldSpec(
                        field_id=f"layer.{layer}.latent_kv",
                        plane_id="flatkv",
                        shape=(rows, 1, width),
                        dtype="bfloat16",
                        exact_page_stride=False,
                        page_stride_alignment_bytes=256,
                    )
                )
                if gid == "full":
                    # Byte envelope only: all 64 FP8 rows precede all 64 FP32
                    # scales inside a page; this is NOT 64 interleaved records.
                    fields.append(
                        CacheFieldSpec(
                            field_id=f"layer.{layer}.index_k",
                            plane_id="flatkv",
                            shape=(64, 132),
                            dtype="uint8",
                            exact_page_stride=False,
                            page_stride_alignment_bytes=256,
                        )
                    )
            group_spec = CacheGroupSpec(
                group_id=gid,
                retention="full_history" if gid == "full" else "sliding_window",
                rows_per_page=rows,
                entry_stride_tokens=1,
                sliding_window_tokens=None if gid == "full" else 513,
                family="history",
                replayable=False,
            )
            if self.pd_disaggregation_enabled:
                group_spec = apply_pd_transfer_policies((group_spec,))[0]
            declarations.append((group_spec, tuple(fields)))
        return tuple(declarations)

    @override
    def packing(self, groups: Sequence[CacheGroupDeclaration]) -> Mapping[str, int]:
        return {
            spec.group_id: {"swa.2": 5, "draft.swa": 15}.get(spec.group_id, 1)
            for spec, _ in groups
        }

    @override
    def check_layout(self, layout: CacheLayout) -> None:
        # packing15 adds a 3,840-byte parent alignment; target-only keeps Plan A.
        parent_bytes = 1_071_360 if self.num_draft_layers else 1_068_800
        if (
            layout.plane_bytes != (("flatkv", parent_bytes),)
            or layout.lcm_block_bytes != parent_bytes
        ):
            raise ValueError(
                f"dots3 Plan A requires one {parent_bytes:,}-byte FlatKV plane"
            )

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
                else num_lcm_blocks * dict(layout.group_packing)["full"] * 64
            ),
        )

    @override
    def workspace_bytes(self) -> int:
        """Conservative live-buffer bound for the current model and Triton ops.

        This covers routed metadata, index quantization/selection, absorbed
        attention and the model's 256-query SWA prefill tile. DSA reads paged
        latent/index fields directly: no allocation scales with arena capacity.
        Prefill scores target a 64-MiB tile (at least one complete score row);
        decode scores are batch * verify width by page-rounded context, not
        capped at 64 MiB. MTP owns separate metadata and fusion/dense scratch.
        """
        spec = self.attn_config.component(Dots3NoteAttnConfig)
        full, swa = spec.full, spec.swa
        limits = self.scheduler_limits
        batch, context = limits.max_live_requests, limits.max_context_len
        decode_queries = batch * self.attn_config.speculative_num_draft_tokens
        queries = max(limits.max_scheduled_tokens, decode_queries)
        full_pages, swa_pages = (context + 63) // 64, (context + 31) // 32
        # Router stack uses the widest group; leaves retain their own tables.
        # Include extend/raw tables, lengths, write locations and row maps.
        tables = 4 * batch * (4 * swa_pages + full_pages + 3 * swa_pages)
        metadata = 2 * (tables + 4 * (batch * 32 + queries * 8 + 4))
        # Packed verify leaves repeat their page tables per query at forward time.
        metadata += 4 * (decode_queries - batch) * (full_pages + 3 * swa_pages)
        history_rows = batch * context
        chunk_len = max(
            1,
            limits.max_scheduled_tokens
            * self.server_args.mla_chunk_multiplier
            // batch,
        )
        chunks = (context + chunk_len - 1) // chunk_len
        # Four leaves' int32 prefix slots/chunk descriptors coexist with the
        # int64 request/offset/page/slot temporaries in the Full workspace builder.
        metadata += history_rows * (4 * 4 + 6 * 8) + 4 * chunks * (6 * batch + 1) * 4

        # Replicated index heads: BF16 Hadamard Q, FP32 quantization temporaries,
        # FP8 Q and BF16 converted Q, plus scales/weights and current K rows.
        index_queries = queries * (
            full.index_n_heads * (full.index_head_dim * 13 + 24)
            + full.index_head_dim * 16
        )
        prefill_logits = min(
            queries * history_rows * 4, max(64 << 20, history_rows * 4)
        )
        decode_logits = decode_queries * full_pages * 64 * 4
        logits = max(prefill_logits, decode_logits)
        # Two score tiles may coexist across assignment. Radix TopK adds a
        # 16-bin int32 histogram per 4096 columns and per-row prefix/counters.
        scores = 2 * logits + logits // 256 + queries * (64 + 8)
        # Radix values/indices/output, local/global selections and validity masks.
        selections = queries * (full.index_topk * (6 * 4 + 3) + 64)

        full_heads = full.num_attention_heads // full.attn_tp_size
        swa_heads = swa.num_attention_heads // swa.attn_tp_size
        # Projected Q, absorption input/output and attention output coexist;
        # expanded values include the model's final and per-path output buffers.
        full_attention = (
            queries
            * 2
            * (
                full.kv_cache_dim
                + full_heads
                * (
                    full.head_dim
                    + full.kv_lora_rank * 2
                    + full.kv_cache_dim
                    + 3 * full.v_head_dim
                )
            )
        )
        swa_attention = (
            queries
            * 2
            * (swa.kv_cache_dim + swa_heads * (swa.head_dim + 3 * swa.v_head_dim))
            + decode_queries * swa_heads * (2 * swa.kv_lora_rank + swa.kv_cache_dim) * 2
        )
        tile = min(queries, 256)  # Dots3NoteAttention._swa_prefill query tile.
        visible_rows = min(context, 512 + tile)
        # Gathered latent + contiguous projection input, expanded KV, concatenated
        # K and contiguous V. Previous tile views can keep one generation alive.
        swa_tile = (
            2
            * visible_rows
            * (
                (swa.kv_cache_dim + swa.kv_lora_rank) * 2
                + swa_heads
                * (swa.qk_nope_head_dim + 2 * swa.v_head_dim + swa.head_dim)
                * 2
                + 4 * 8
            )
            + tile * swa_heads * (swa.head_dim + swa.v_head_dim) * 2
            + 16
        )
        draft_workspace = 0
        if self.num_draft_layers:
            # The draft's one router/leaf keeps independent tables and metadata.
            # Step 0 writes every verify row before any accepted-row narrowing.
            draft_metadata = 2 * (
                4 * batch * 2 * swa_pages + 4 * (batch * 32 + queries * 8 + 4)
            )
            draft_metadata += history_rows * 4 + 4 * chunks * (6 * batch + 1)
            hf = self.draft_model_config.hf_text_config
            draft_temporaries = (
                queries * 2 * (8 * hf.hidden_size + 3 * hf.intermediate_size)
            )
            draft_workspace = (
                draft_metadata + draft_temporaries + swa_attention + swa_tile
            )
        return (
            metadata
            + draft_workspace
            + index_queries
            + scores
            + selections
            + full_attention
            + swa_attention
            + swa_tile
        )
