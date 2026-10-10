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

import torch
from flashinfer.attention.prims_ts.decode import (
    suggest_q_token_kv_block_sparse_group_size,
)
from flashinfer.attention.prims_ts.q_token_kv_block_sparse_metadata import (
    QTokenKvBlockSparsePagedTSWrapper,
    get_q_token_kv_block_sparse_workspace_size,
)
from tokenspeed_kernel.ops.attention.qsa.metadata import QSAPrefillMetadata


def _tensor_layout(tensor: torch.Tensor) -> tuple:
    return (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)


def _group_offsets(offsets: tuple[int, ...], group_size: int) -> tuple[int, ...]:
    """Partition each request independently, preserving short final groups."""
    return tuple(
        offset
        for begin, end in zip(offsets, offsets[1:])
        for offset in range(begin, end, group_size)
    ) + (offsets[-1],)


@dataclass
class _PrimTSQSAPlan:
    wrapper: QTokenKvBlockSparsePagedTSWrapper
    qo_indptr: torch.Tensor
    captured: bool = False


class _PrimTSQSARunner:
    """Bounded plans sharing one high-water workspace on one CUDA stream.

    Plans retain cache views and route offsets, but no request-owned history.
    The public wrapper refreshes candidate unions and membership on every run.
    """

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        self._workspace = torch.empty(0, dtype=torch.uint8, device=device)
        self._plans: OrderedDict[tuple, _PrimTSQSAPlan] = OrderedDict()

    def run(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        metadata: QSAPrefillMetadata,
        *,
        scale: float,
        v_scale: float,
    ) -> torch.Tensor:
        q = q.contiguous()
        page_size = metadata.page_size
        # HND is the logical ABI; PrimTS accepts the underlying NHD strides.
        cache_shape = (-1, page_size, k_cache.shape[1], k_cache.shape[2])
        k = k_cache.view(cache_shape).permute(0, 2, 1, 3)
        v = v_cache.view(cache_shape).permute(0, 2, 1, 3)
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
            k.shape[1],
            self._sm_count,
            head_dim=q.shape[2],
            q_data_type=q.dtype,
            split_kv=False,
            kv_block_size=metadata.block_size,
        )
        key = (
            _tensor_layout(q),
            _tensor_layout(k),
            _tensor_layout(v),
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
        plan = self._plans.get(key)
        if plan is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "warm up the PrimTS QSA prefill geometry before capture"
                )
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
                if any(cached.captured for cached in self._plans.values()):
                    raise RuntimeError(
                        "warm the largest PrimTS QSA workspace before capture"
                    )
                self._plans.clear()
                self._workspace = torch.zeros(
                    size, dtype=torch.uint8, device=self._device
                )
            else:
                # Workspace section offsets can change with the launch geometry.
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
            if len(self._plans) >= 64:
                evictable = next(
                    (key for key, cached in self._plans.items() if not cached.captured),
                    None,
                )
                if evictable is None:
                    raise RuntimeError("all 64 PrimTS QSA plans are CUDA Graph pinned")
                del self._plans[evictable]
            plan = _PrimTSQSAPlan(wrapper, qo_indptr)
            self._plans[key] = plan
        else:
            self._plans.move_to_end(key)
        if torch.cuda.is_current_stream_capturing():
            plan.captured = True
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


@cache
def _get_runner(device: torch.device, stream: int) -> _PrimTSQSARunner:
    # Stream identity separates mutable workspaces used concurrently.
    return _PrimTSQSARunner(device)


def qsa_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    metadata: QSAPrefillMetadata,
    *,
    scale: float,
    v_scale: float,
) -> torch.Tensor:
    """Run grouped causal QSA prefill with existing paged K/V and logical IDs.

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
    return _get_runner(q.device, torch.cuda.current_stream(q.device).cuda_stream).run(
        q, k_cache, v_cache, metadata, scale=scale, v_scale=v_scale
    )
