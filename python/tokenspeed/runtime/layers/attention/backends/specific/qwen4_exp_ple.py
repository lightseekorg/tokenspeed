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

"""PLE checkpoint metadata and speculative state, independent of GDN."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.attention.kda.triton import (
    verify_state_blocks,
)
from tokenspeed_kernel.ops.kvcache.triton import (
    copy_state_rows,
    state_verify_commit_rows,
)

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.base import (
    AttentionBackend,
    CudaGraphSupport,
)
from tokenspeed.runtime.layers.attention.backends.state.checkpoint import (
    compute_state_block_indices,
    gather_verified_state_blocks,
    verified_state_block_slots,
)
from tokenspeed.runtime.layers.attention.backends.state.utils import row_stride_i32
from tokenspeed.runtime.layers.attention.kv_cache.qwen4_exp import (
    QWEN4_EXP_PLE_CACHE_GROUP,
    qwen4_exp_ple_conv_field,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    cache_debug_enabled,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    cache_field_layer_id,
    cache_field_plane,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.configs.base import (
        AttnConfig,
        SoftmaxAttnConfig,
    )
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool


@dataclass
class PLEForwardMetadata:
    """PLE's block ids and query shape; no recurrent-attention metadata."""

    input_blocks: torch.Tensor
    output_blocks: torch.Tensor
    query_lengths: list[int]
    verify_width: int | None


class Qwen4ExpPLEBackend(AttentionBackend):
    """Bind PLE cache fields and own their metadata and verify workspace."""

    cache_consumer_families = frozenset({"state"})
    # Prefill graph buckets pad tokens; PLE's request-shaped updates must stay
    # eager so padding cannot advance n-gram or short-convolution state.
    cuda_graph_support = CudaGraphSupport(prefill_graph=False)

    def __init__(self, config: AttnConfig, spec: SoftmaxAttnConfig) -> None:
        super().__init__(config, spec)
        self.forward_metadata: PLEForwardMetadata | None = None
        self._context_field_id: str | None = None
        self._conv_field_ids: tuple[str, ...] = ()
        self._checkpoint_granularity = 1
        self._ple_verify_scratch: dict[str, torch.Tensor] = {}
        self._ple_verify_tables: dict | None = None
        self._ple_commit_rows: torch.Tensor | None = None
        self._verify_commit_ctx: tuple[torch.Tensor, int, int] | None = None
        self._decode_input_blocks: torch.Tensor | None = None
        self._decode_output_blocks: torch.Tensor | None = None
        self._decode_committed: torch.Tensor | None = None
        self._decode_views_by_bs: dict[int, PLEForwardMetadata] = {}

    def _cache_fields(
        self, cache_pool: CachePool
    ) -> tuple[str | None, tuple[str, ...], int]:
        """Validate the local PLE fields before the backend tree publishes a pool."""
        if self.is_draft:
            return None, (), 1
        fields = tuple(
            field
            for field in cache_pool.arena.plan.fields
            if field.group_id == QWEN4_EXP_PLE_CACHE_GROUP
            and cache_field_layer_id(field.field_id) in cache_pool.field_layer_range
        )
        conv_fields = tuple(
            field.field_id
            for field in fields
            if cache_field_plane(field.field_id) == "qwen4_exp.ple.conv"
        )
        if not conv_fields:
            return None, (), 1
        context_fields = tuple(
            field.field_id
            for field in fields
            if cache_field_plane(field.field_id) == "qwen4_exp.ple.context"
        )
        if len(context_fields) != 1:
            raise RuntimeError(
                "PLE cache view requires exactly one shared context field"
            )
        contract = cache_pool.arena.runtime_contract
        group = next(
            (
                spec
                for spec in contract.group_specs
                if spec.group_id == QWEN4_EXP_PLE_CACHE_GROUP
            ),
            None,
        )
        if (
            group is None
            or group.family != "state"
            or group.checkpoint_granularity is None
        ):
            raise RuntimeError("PLE requires a checkpoint-shaped state cache group")
        return (
            context_fields[0],
            tuple(sorted(conv_fields, key=cache_field_layer_id)),
            int(group.checkpoint_granularity),
        )

    def validate_cache_pool(self, cache_pool: CachePool) -> None:
        super().validate_cache_pool(cache_pool)
        if self.cache_pool is not None and self.cache_pool is not cache_pool:
            raise RuntimeError("PLE backend cannot be rebound to another cache pool")
        self._cache_fields(cache_pool)

    def _publish_cache_pool(self, cache_pool: CachePool) -> None:
        already_bound = self.cache_pool is cache_pool
        super()._publish_cache_pool(cache_pool)
        if already_bound:
            return
        (
            self._context_field_id,
            self._conv_field_ids,
            self._checkpoint_granularity,
        ) = self._cache_fields(cache_pool)

    def _block_rows(self, block_tables: Mapping[str, torch.Tensor]) -> torch.Tensor:
        rows = block_tables.get(QWEN4_EXP_PLE_CACHE_GROUP)
        if rows is None:
            raise RuntimeError("block_tables is missing the PLE cache group")
        return rows

    def init_cuda_graph_state(self, max_bs: int, **kwargs) -> None:
        del kwargs
        if self._decode_input_blocks is not None:
            if max_bs > self._decode_input_blocks.numel():
                raise RuntimeError("PLE decode batch exceeds the preallocated capacity")
            return
        self._decode_input_blocks = torch.full(
            (max_bs,), -1, dtype=torch.int32, device=self.device
        )
        self._decode_output_blocks = torch.full_like(self._decode_input_blocks, -1)
        self._decode_committed = torch.empty(
            max_bs, dtype=torch.int64, device=self.device
        )

    def _decode_view(self, bs: int) -> PLEForwardMetadata:
        if (
            self._decode_input_blocks is None
            or not 0 < bs <= self._decode_input_blocks.numel()
        ):
            raise RuntimeError("PLE decode batch exceeds the preallocated capacity")
        metadata = self._decode_views_by_bs.get(bs)
        if metadata is None:
            metadata = PLEForwardMetadata(
                input_blocks=self._decode_input_blocks[:bs],
                output_blocks=self._decode_output_blocks[:bs],
                # Capture records a full bucket. Invalid blocks isolate the
                # padded rows; model eager breaks slice by the live ctx.bs.
                query_lengths=[self.spec_num_tokens] * bs,
                verify_width=self.spec_num_tokens if self.spec_num_tokens > 1 else None,
            )
            self._decode_views_by_bs[bs] = metadata
        return metadata

    def init_forward_metadata(
        self,
        bs: int,
        num_extends: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        *,
        block_tables: Mapping[str, torch.Tensor],
        extend_seq_lens: torch.Tensor,
        extend_seq_lens_cpu: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_prefix_lens_cpu: torch.Tensor,
        extend_with_prefix: bool,
        **kwargs,
    ) -> None:
        del (
            req_pool_indices,
            extend_seq_lens,
            extend_prefix_lens_cpu,
            extend_with_prefix,
            kwargs,
        )
        self._verify_commit_ctx = None
        self.forward_metadata = None
        if not (forward_mode.is_extend_or_mixed() or forward_mode.is_idle()):
            raise RuntimeError(
                "PLE decode metadata goes through refresh_decode_metadata"
            )
        if not 0 <= num_extends <= bs:
            raise ValueError("num_extends must be between 0 and bs")
        if not self._conv_field_ids or forward_mode.is_idle():
            return
        after = seq_lens[:bs]
        before = torch.cat(
            (
                extend_prefix_lens[:num_extends].to(
                    device=after.device, dtype=after.dtype
                ),
                after[num_extends:] - self.spec_num_tokens,
            )
        )
        input_blocks, output_blocks = compute_state_block_indices(
            self._block_rows(block_tables),
            self._checkpoint_granularity,
            before,
            after,
            validate=cache_debug_enabled(),
            group_id=QWEN4_EXP_PLE_CACHE_GROUP,
        )
        self.forward_metadata = PLEForwardMetadata(
            input_blocks=input_blocks,
            output_blocks=output_blocks,
            query_lengths=[
                int(value) for value in extend_seq_lens_cpu[:num_extends].tolist()
            ]
            + [self.spec_num_tokens] * (bs - num_extends),
            # Mixed rounds keep their existing direct final-state writes.
            verify_width=None,
        )

    def refresh_decode_metadata(
        self,
        bs: int,
        actual_bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        forward_mode: ForwardMode,
        block_tables: Mapping[str, torch.Tensor],
        **kwargs,
    ) -> None:
        del req_pool_indices, kwargs
        self._verify_commit_ctx = None
        self.forward_metadata = None
        if not forward_mode.is_decode_or_idle():
            raise ValueError("PLE decode refresh requires decode or idle mode")
        if not 0 <= actual_bs <= bs:
            raise ValueError("actual_bs must be between 0 and bs")
        if not self._conv_field_ids:
            return
        metadata = self._decode_view(bs)
        metadata.input_blocks.fill_(-1)
        metadata.output_blocks.fill_(-1)
        if actual_bs:
            rows = self._block_rows(block_tables)
            if metadata.verify_width is not None:
                if self._ple_commit_rows is None:
                    raise RuntimeError("PLE verify workspace was not preallocated")
                verify_state_blocks(
                    seq_lens,
                    rows,
                    batch_size=actual_bs,
                    draft_tokens=metadata.verify_width,
                    granularity=self._checkpoint_granularity,
                    pages_out=metadata.input_blocks,
                    committed_out=self._decode_committed,
                )
                if forward_mode.is_decode():
                    self._verify_commit_ctx = (rows, actual_bs, metadata.verify_width)
            else:
                after = seq_lens[:actual_bs]
                input_blocks, output_blocks = compute_state_block_indices(
                    rows,
                    self._checkpoint_granularity,
                    after - 1,
                    after,
                    validate=cache_debug_enabled(),
                    group_id=QWEN4_EXP_PLE_CACHE_GROUP,
                )
                metadata.input_blocks[:actual_bs].copy_(input_blocks)
                metadata.output_blocks[:actual_bs].copy_(output_blocks)
        self.forward_metadata = metadata

    def preallocate_verify_workspace(self, max_bs: int, draft_token_num: int) -> int:
        """Allocate rollback rows once and return the recipe-budgeted bytes."""
        if not self._conv_field_ids or self.is_draft or self.spec_num_tokens <= 1:
            return 0
        if draft_token_num != self.spec_num_tokens:
            raise ValueError("PLE verify workspace width differs from the target width")
        if self._ple_commit_rows is not None:
            if max_bs * len(self._conv_field_ids) > self._ple_commit_rows.shape[1]:
                raise RuntimeError("PLE verify batch exceeds the preallocated capacity")
        else:
            rows = max_bs * (draft_token_num + 1)
            for field_id in (self._context_field_id, *self._conv_field_ids):
                field = self.cache_pool.arena.field(field_id)
                self._ple_verify_scratch[field_id] = field.new_zeros(
                    (rows, *field.shape[1:])
                )
            self._ple_commit_rows = torch.empty(
                (2, max_bs * len(self._conv_field_ids)),
                dtype=torch.int64,
                device=self.device,
            )
            self._ple_verify_tables = self._build_verify_tables()
        return (
            sum(tensor.nbytes for tensor in self._ple_verify_scratch.values())
            + self._ple_commit_rows.nbytes
        )

    def ple_verify_scratch(
        self, context_field_id: str, layer_id: int, bs: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return shared-context and local-layer rollback views for ``bs`` requests."""
        if context_field_id != self._context_field_id:
            raise RuntimeError("PLE layer names a context outside its cache view")
        global_layer_id = self.cache_pool._field_layer_id(layer_id)
        try:
            context = self._ple_verify_scratch[context_field_id]
            conv = self._ple_verify_scratch[qwen4_exp_ple_conv_field(global_layer_id)]
        except KeyError as exc:
            raise RuntimeError(
                "PLE verify workspace was not preallocated for this layer"
            ) from exc
        rows = bs * (self.spec_num_tokens + 1)
        if context.shape[0] < rows or conv.shape[0] < rows:
            raise RuntimeError(
                f"PLE verify workspace needs {rows} rows, exceeding preallocated capacity"
            )
        return context[:rows], conv[:rows]

    @staticmethod
    def _u64(values: list[int], device: torch.device) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.uint64, device=device)

    @staticmethod
    def _i64(values: list[int], device: torch.device) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.int64, device=device)

    def _build_verify_tables(self) -> dict:
        """Build pointer and stride tables for the batched PLE commit."""

        arena = self.cache_pool.arena
        context_field = arena.field(self._context_field_id)
        conv_fields = [arena.field(field_id) for field_id in self._conv_field_ids]
        context_scratch = self._ple_verify_scratch[self._context_field_id]
        conv_scratches = [
            self._ple_verify_scratch[field_id] for field_id in self._conv_field_ids
        ]
        conv_shape = tuple(conv_fields[0].shape[1:])
        conv_dtype = conv_fields[0].dtype
        conv_dst_stride = row_stride_i32(conv_fields[0])
        for layer, field in zip(self._conv_field_ids, conv_fields, strict=True):
            if tuple(field.shape[1:]) != conv_shape or field.dtype != conv_dtype:
                raise RuntimeError(
                    f"PLE layer {layer} convolution cache geometry "
                    f"{tuple(field.shape[1:])}/{field.dtype} differs from layer "
                    f"{self._conv_field_ids[0]} {conv_shape}/{conv_dtype}"
                )
            if row_stride_i32(field) != conv_dst_stride:
                raise RuntimeError(
                    f"PLE layer {layer} convolution page stride "
                    f"{row_stride_i32(field)} must match layer "
                    f"{self._conv_field_ids[0]} {conv_dst_stride}"
                )
        device = context_field.device
        tables = {
            "context_src": self._u64([context_scratch.data_ptr()], device),
            "context_dst": self._u64([context_field.data_ptr()], device),
            "context_src_stride": self._i64([row_stride_i32(context_scratch)], device),
            "context_dst_stride": self._i64([row_stride_i32(context_field)], device),
            "context_row_bytes": context_field[0].numel()
            * context_field.element_size(),
            "conv_src": self._u64(
                [scratch.data_ptr() for scratch in conv_scratches], device
            ),
            "conv_dst": self._u64([field.data_ptr() for field in conv_fields], device),
            "conv_src_stride": self._i64(
                [row_stride_i32(scratch) for scratch in conv_scratches], device
            ),
            "conv_dst_stride": self._i64([conv_dst_stride] * len(conv_fields), device),
            "conv_row_bytes": conv_fields[0][0].numel() * conv_fields[0].element_size(),
        }
        return tables

    def commit_verified_state(self, accepted_lengths: torch.Tensor) -> None:
        """Commit one completed decode verification's accepted PLE checkpoint."""
        context = self._verify_commit_ctx
        if context is None:
            return
        rows_table, actual_bs, width = context
        bs = accepted_lengths.shape[0]
        if bs != actual_bs:
            raise ValueError(
                "PLE accepted lengths must cover exactly the live verify batch"
            )
        rows = self._ple_commit_rows
        num_layers = len(self._conv_field_ids)
        row_count = bs * num_layers
        if rows is None or rows.shape[1] < row_count:
            raise RuntimeError("PLE commit rows exceed the preallocated capacity")
        steps = accepted_lengths.to(torch.int64).clamp(min=1, max=width)
        slots = verified_state_block_slots(
            self._decode_committed, steps, self._checkpoint_granularity
        )
        pages = gather_verified_state_blocks(rows_table, slots)
        src_rows, dst_rows = rows[0, :row_count], rows[1, :row_count]
        state_verify_commit_rows(
            accepted_lengths,
            pages,
            src_rows,
            dst_rows,
            verify_width=width,
            num_layers=num_layers,
        )
        tables = self._ple_verify_tables
        if tables["context_row_bytes"]:
            copy_state_rows(
                tables["context_src"],
                tables["context_dst"],
                src_rows[:bs],
                dst_rows[:bs],
                row_bytes=tables["context_row_bytes"],
                src_row_strides=tables["context_src_stride"],
                dst_row_strides=tables["context_dst_stride"],
            )
        copy_state_rows(
            tables["conv_src"],
            tables["conv_dst"],
            src_rows,
            dst_rows,
            row_bytes=tables["conv_row_bytes"],
            src_row_strides=tables["conv_src_stride"],
            dst_row_strides=tables["conv_dst_stride"],
        )
        self._verify_commit_ctx = None
