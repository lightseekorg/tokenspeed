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

"""FlashInfer PrimTS QSA adapter over TokenSpeed's existing KV cache."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import cache
from weakref import WeakValueDictionary

import torch
from flashinfer.attention.prims_ts.decode import (
    suggest_q_token_kv_block_sparse_group_size,
)
from flashinfer.attention.prims_ts.q_token_kv_block_sparse_metadata import (
    QTokenKvBlockSparsePagedTSWrapper,
    get_q_token_kv_block_sparse_workspace_size,
)
from tokenspeed_kernel.ops.attention.qsa.metadata import QSASparseMetadata, QSAWorkspace
from tokenspeed_kernel.ops.tuning import get_autotune_max_num_tokens, is_autotuning
from tokenspeed_kernel.workspace import retain_for_capture


def _tensor_layout(tensor: torch.Tensor) -> tuple:
    return (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)


def _group_offsets(offsets: tuple[int, ...], group_size: int) -> tuple[int, ...]:
    """Partition each request independently, preserving short final groups."""
    return tuple(
        offset
        for begin, end in zip(offsets, offsets[1:])
        for offset in range(begin, end, group_size)
    ) + (offsets[-1],)


@cache
def _warmup_group_sizes(q_heads: int, kv_heads: int, max_rows: int) -> tuple[int, ...]:
    """Cache native-legal groups for one head geometry and startup row budget."""
    # Keep the private policy dependency inside startup group enumeration.
    from flashinfer.attention.prims_ts._q_token_kv_block_sparse_policy import (
        SUPPORTED_GROUP_SIZES,
    )
    from flashinfer.attention.prims_ts.decode import (
        validate_q_token_kv_block_sparse_group_size,
    )

    offsets = torch.tensor([0, max_rows], dtype=torch.int32, device="cpu")
    # Validate the geometry before filtering unsupported groups, so invalid
    # model heads or row budgets still raise instead of returning no groups.
    validate_q_token_kv_block_sparse_group_size(
        offsets, max_rows, q_heads, kv_heads, group_size=1
    )
    groups = []
    # The public validator owns the membership and head-tile limits; do not
    # copy them into the adapter.
    for group in SUPPORTED_GROUP_SIZES:
        if group > max_rows:
            continue
        try:
            validate_q_token_kv_block_sparse_group_size(
                offsets, max_rows, q_heads, kv_heads, group_size=group
            )
        except ValueError:
            continue
        groups.append(group)
    return tuple(groups)


@dataclass
class _PrimTSQSAPlan:
    wrapper: QTokenKvBlockSparsePagedTSWrapper
    qo_indptr: torch.Tensor


class _PrimTSQSARunner:
    """Per-pool plans sharing a growable workspace on one device.

    Plans retain cache views and route offsets, but no request-owned history.
    Packed queries supply their offsets separately. With split_kv=False,
    every live route is rewritten before attention reads it; split scratch
    and attention sinks are unused. Runs do not depend on previous workspace
    contents.
    """

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        self._workspace = torch.empty(0, dtype=torch.uint8, device=device)
        self._plans: dict[tuple[int, int], OrderedDict[tuple, _PrimTSQSAPlan]] = {}
        self._captured_plans: WeakValueDictionary[tuple, _PrimTSQSAPlan] = (
            WeakValueDictionary()
        )
        self._warmed: set[tuple] = set()

    def warmup(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        metadata: QSASparseMetadata,
        *,
        max_rows: int,
    ) -> None:
        """Prepare native group and scheduler variants without binding model KV."""
        q_heads, head_dim = q.shape[1:]
        kv_heads = k_cache.shape[1]
        topk = metadata.selected_blocks.shape[1]
        key = (
            q_heads,
            kv_heads,
            head_dim,
            q.dtype,
            k_cache.dtype,
            v_cache.dtype,
            metadata.page_size,
            metadata.block_size,
            topk,
            metadata.block_table.shape[1],
            metadata.query_positions.dtype,
            max_rows,
        )
        if key in self._warmed:
            return

        # Packed batch sizes are dynamic. One route and more than one SM wave
        # cover the native static/persistent scheduler choices. Every request
        # has one query so short groups also cover ragged batches at every G.
        route_counts = sorted({1, min(max_rows, self._sm_count + 1)})
        rows = route_counts[-1]
        scratch_q = torch.zeros(
            (rows, q_heads, head_dim), dtype=q.dtype, device=self._device
        )
        scratch_k = torch.zeros(
            (metadata.page_size, kv_heads, head_dim),
            dtype=k_cache.dtype,
            device=self._device,
        )
        scratch_v = torch.zeros_like(scratch_k, dtype=v_cache.dtype)
        table = torch.zeros(
            (rows, metadata.block_table.shape[1]),
            dtype=torch.int32,
            device=self._device,
        )
        blocks = torch.full((rows, topk), -1, dtype=torch.int32, device=self._device)
        requests = torch.arange(rows, dtype=torch.int32, device=self._device)
        positions = torch.zeros(
            rows, dtype=metadata.query_positions.dtype, device=self._device
        )
        # Keep scratch plans and their workspace out of serving/graph state.
        scratch_runner = _PrimTSQSARunner(self._device)
        scratch_workspace = QSAWorkspace()
        for group_size in _warmup_group_sizes(q_heads, kv_heads, max_rows):
            for routes in route_counts:
                scratch_metadata = QSASparseMetadata(
                    blocks[:routes],
                    table[:routes],
                    requests[:routes],
                    positions[:routes],
                    tuple(range(routes + 1)),
                    metadata.page_size,
                    metadata.block_size,
                    scratch_workspace,
                )
                scratch_runner._run(
                    scratch_q[:routes],
                    scratch_k,
                    scratch_v,
                    scratch_metadata,
                    group_size=group_size,
                    scale=1.0,
                    v_scale=1.0,
                )
        self._warmed.add(key)

    def run(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        metadata: QSASparseMetadata,
        *,
        scale: float,
        v_scale: float,
    ) -> torch.Tensor:
        lengths = tuple(
            end - begin
            for begin, end in zip(
                metadata.query_start_loc, metadata.query_start_loc[1:]
            )
        )
        topk = metadata.selected_blocks.shape[1]
        group_size = suggest_q_token_kv_block_sparse_group_size(
            len(lengths),
            max(lengths),
            topk * metadata.block_size + metadata.block_size - 1,
            q.shape[1],
            k_cache.shape[1],
            self._sm_count,
            head_dim=q.shape[2],
            q_data_type=q.dtype,
            split_kv=False,
            kv_block_size=metadata.block_size,
        )
        return self._run(
            q,
            k_cache,
            v_cache,
            metadata,
            group_size=group_size,
            scale=scale,
            v_scale=v_scale,
        )

    def _run(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        metadata: QSASparseMetadata,
        *,
        group_size: int,
        scale: float,
        v_scale: float,
    ) -> torch.Tensor:
        q = q.contiguous()
        page_size = metadata.page_size
        # HND is the logical ABI; PrimTS accepts the underlying NHD strides.
        cache_shape = (-1, page_size, k_cache.shape[1], k_cache.shape[2])
        k = k_cache.view(cache_shape).permute(0, 2, 1, 3)
        v = v_cache.view(cache_shape).permute(0, 2, 1, 3)
        topk = metadata.selected_blocks.shape[1]
        key = (
            _tensor_layout(q),
            _tensor_layout(k),
            _tensor_layout(v),
            # PrimTS binds K/V storage in its prepared attention plan. Separate
            # warmed wrappers let each layer capture without rebinding inputs.
            k.data_ptr(),
            v.data_ptr(),
            _tensor_layout(metadata.block_table),
            _tensor_layout(metadata.selected_blocks),
            _tensor_layout(metadata.query_positions),
            _tensor_layout(metadata.token_to_request),
            metadata.query_start_loc,
            metadata.block_size,
            group_size,
        )
        binding = (k.data_ptr(), v.data_ptr())
        plans = self._plans.setdefault(binding, OrderedDict())
        plan = self._captured_plans.get(key)
        if plan is None:
            plan = plans.get(key)
        if plan is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("warm up the PrimTS QSA geometry before capture")
            offsets = _group_offsets(metadata.query_start_loc, group_size)
            qo_indptr = torch.tensor(offsets, dtype=torch.int32, device=self._device)
            max_kv = metadata.block_table.shape[1] * page_size
            size = get_q_token_kv_block_sparse_workspace_size(
                q,
                k,
                metadata.block_table,
                block_topk=topk,
                max_seq_len_kv=max_kv,
                o_data_type=q.dtype,
                qo_indptr=qo_indptr,
                seq_len_q=group_size,
                kv_block_size=metadata.block_size,
                split_kv=False,
                share_pattern_across_kv_heads=True,
            )
            if size > self._workspace.numel():
                # Older plans retain their allocation, including plans owned
                # by graphs. Growing eager scratch never changes their addresses.
                self._workspace = torch.zeros(
                    1 << (size - 1).bit_length(), dtype=torch.uint8, device=self._device
                )
            else:
                # Initialize the new geometry's sections. Cached nonsplit
                # plans remain safe: run rewrites every workspace value read
                # by attention, including on replay after another geometry.
                self._workspace.zero_()
            wrapper = QTokenKvBlockSparsePagedTSWrapper()
            wrapper.plan(
                len(offsets) - 1,
                group_size,
                q.shape[1],
                k.shape[1],
                q.shape[2],
                metadata.block_size,
                page_size,
                topk,
                max_kv,
                device=self._device,
                workspace_buffer=self._workspace,
                use_packed_q=True,
                split_kv=False,
                share_pattern_across_kv_heads=True,
                mask_type="causal",
                q_data_type=q.dtype,
                kv_data_type=k.dtype,
                o_data_type=q.dtype,
            )
            # Bound eager geometries per layer, not across every layer and
            # graph batch. Captures own their plans outside this LRU.
            if len(plans) >= 64:
                plans.popitem(last=False)
            plan = _PrimTSQSAPlan(wrapper, qo_indptr)
            plans[key] = plan
        elif key in plans:
            plans.move_to_end(key)
        if torch.cuda.is_current_stream_capturing():
            retain_for_capture(plan)
            self._captured_plans[key] = plan
            plans.pop(key, None)
            if not plans:
                del self._plans[binding]
        return plan.wrapper.run(
            q,
            (k, v),
            metadata.block_table,
            metadata.selected_blocks,
            metadata.token_to_request,
            metadata.query_positions,
            qo_indptr=plan.qo_indptr,
            sm_scale=scale,
            v_scale=v_scale,
            out=torch.empty_like(q),
        )


def _get_runner(workspace: QSAWorkspace, device: torch.device) -> _PrimTSQSARunner:
    # Calls, including graph replays, must be ordered across streams because
    # the runner shares mutable scratch on this device.
    runner = workspace.runners.get(device)
    if runner is None:
        runner = _PrimTSQSARunner(device)
        workspace.runners[device] = runner
    return runner


def qsa_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    metadata: QSASparseMetadata,
    *,
    scale: float,
    v_scale: float,
) -> torch.Tensor:
    """Run grouped causal QSA attention with paged K/V and logical IDs.

    Args:
        q: Packed BF16 query tensor [tokens, heads, head_dim].
        k_cache: Flattened BF16 key cache [slots, kv_heads, head_dim].
        v_cache: Flattened BF16 value cache matching the keys.
        metadata: Logical block selection and request geometry.
        scale: Softmax scale, including any key descale.
        v_scale: Scalar value descale applied to the output.

    Returns:
        A new BF16 output tensor with the query shape.
    """
    runner = _get_runner(metadata.workspace, q.device)
    if is_autotuning():
        runner.warmup(
            q, k_cache, v_cache, metadata, max_rows=get_autotune_max_num_tokens()
        )
    return runner.run(q, k_cache, v_cache, metadata, scale=scale, v_scale=v_scale)
