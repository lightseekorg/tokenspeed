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

"""QCP Prefill KV transport shared by DSA and MLA, with KVP owner writes."""

import torch
import torch.distributed as dist
import torch_npu
from tokenspeed_kernel.ops.attention.ascend_qcp_prefill_workspace import (
    AscendQCPPrefillWorkspaceLayout,
)
from tokenspeed_kernel_npu.ops._qcp_prefill_transport import (
    gather_history,
    pack_current,
    store_current,
)


class AscendQCPPrefillTransport:
    """One bounded Torch workspace, reused by successive attention layers.

    History gather, current-row commit and attention run in order on the model
    stream. The backend retains the allocation; layer-local views and cache
    references are passed between calls rather than stored in this workspace.
    """

    def __init__(
        self,
        kv_cache,
        index_cache,
        *,
        ranks,
        history_group,
        max_current_rows,
        max_history_rows,
        history_null_rows,
    ):
        self.degree = len(ranks)
        self.rank = ranks.index(dist.get_rank())
        self.device = kv_cache.device
        self.local_rows = kv_cache.numel() // 576
        self.layout = AscendQCPPrefillWorkspaceLayout(
            history_capacity=max_history_rows,
            current_capacity=max_current_rows,
            degree=self.degree,
            null_rows=history_null_rows,
            has_index=index_cache is not None,
        )
        for cache, width in ((kv_cache, 576), (index_cache, 128)):
            if cache is not None and (
                cache.device != self.device
                or cache.dtype != torch.bfloat16
                or not cache.is_contiguous()
                or cache.numel() != self.local_rows * width
            ):
                raise ValueError("Prefill needs matching contiguous BF16 cache planes")
        if self.degree > 1 and (
            history_group is None
            or tuple(dist.get_process_group_ranks(history_group)) != tuple(ranks)
        ):
            raise ValueError("Prefill collective group must match cache owners")
        self.history_group = history_group
        self._storage = torch.empty(
            self.layout.nbytes // 2, device=self.device, dtype=torch.bfloat16
        )
        storage = self._storage
        offset = 0

        def plane(rows, width):
            nonlocal offset
            result = storage[offset : offset + rows * width].view(rows, width)
            offset += rows * width
            return result

        # Final attention inputs also receive history directly. DSA uses both
        # planes; MLA needs only latent KV.
        self._attention_kv = plane(self.layout.attention_plane_rows, 576)
        self._attention_index = (
            plane(self.layout.attention_plane_rows, 128)
            if self.layout.has_index
            else None
        )
        # History/current share send storage. Only current needs a separate
        # receive packet before store_current unpacks into the attention planes.
        self._gather_send = plane(self.layout.gather_send_rows, self.layout.width)
        self._current_receive = plane(
            self.layout.current_receive_rows, self.layout.width
        )
        # History sends use separate contiguous planes so the collectives can
        # write directly to attention's planes. Current packets reuse this space.
        send = self._gather_send.view(-1)
        split = self.layout.gather_send_rows * 576
        self._history_send_kv = send[:split].view(-1, 576)
        self._history_send_index = (
            send[split:].view(-1, 128) if self.layout.has_index else None
        )
        # No later operation writes the null page; initialize it once.
        self._attention_kv[:history_null_rows].zero_()
        if self._attention_index is not None:
            self._attention_index[:history_null_rows].zero_()
        self.peak_buffer_bytes = 0

    @property
    def workspace_bytes(self):
        return self.layout.nbytes

    def gather_history(
        self,
        kv_cache,
        index_cache,
        *,
        total_rows,
        has_history,
        history_slots,
        history_counts,
    ):
        """Gather this layer's batch pages and return its KV/Index-K views."""
        if (
            len(history_counts) != self.degree
            or min(history_counts) < 0
            or sum(history_counts) > self.layout.history_capacity
            or history_slots.numel() != history_counts[self.rank]
            or not 0 <= total_rows <= self.layout.current_capacity
        ):
            raise ValueError("Prefill batch exceeds its planned workspace capacity")
        current_rank_stride = (
            (total_rows + self.degree * 128 - 1) // (self.degree * 128) * 128
        )
        history_rows = self.layout.null_rows + sum(history_counts)
        rows = history_rows + self.degree * current_rank_stride
        kv = self._attention_kv[:rows]
        index = (
            self._attention_index[:rows] if self._attention_index is not None else None
        )
        self.peak_buffer_bytes = max(
            self.peak_buffer_bytes,
            (rows + history_counts[self.rank]) * self.layout.width * 2,
        )
        if has_history:
            first = self.layout.null_rows
            receive_kv = kv[first:history_rows]
            receive_index = index[first:history_rows] if index is not None else None
            count = history_counts[self.rank]
            send_kv = receive_kv if self.degree == 1 else self._history_send_kv[:count]
            send_index = (
                (
                    receive_index
                    if self.degree == 1
                    else self._history_send_index[:count]
                )
                if self.layout.has_index
                else None
            )
            gather_history(kv_cache, index_cache, history_slots, send_kv, send_index)
            if self.degree > 1 and sum(history_counts):
                torch_npu.distributed.all_gather_into_tensor_uneven(
                    output=receive_kv,
                    input=send_kv,
                    output_split_sizes=list(history_counts),
                    group=self.history_group,
                )
                if send_index is not None:
                    torch_npu.distributed.all_gather_into_tensor_uneven(
                        output=receive_index,
                        input=send_index,
                        output_split_sizes=list(history_counts),
                        group=self.history_group,
                    )
        return kv, index

    def gather_current(self, kv, index, *, row_counts):
        """Pack both projections once and gather their rows in rank order."""
        local_rows = row_counts[self.rank]
        total_rows = sum(row_counts)
        self.peak_buffer_bytes = max(
            self.peak_buffer_bytes,
            (total_rows + local_rows) * self.layout.width * 2,
        )
        packet = self._gather_send[:local_rows]
        if local_rows:
            pack_current(kv, index, packet)
        if self.degree == 1:
            return packet
        receive = self._current_receive[:total_rows]
        if total_rows:
            torch_npu.distributed.all_gather_into_tensor_uneven(
                output=receive,
                input=packet,
                output_split_sizes=list(row_counts),
                group=self.history_group,
            )
        return receive

    def store_current(
        self,
        packet,
        kv,
        index,
        kv_cache,
        index_cache,
        slots,
        direct,
        *,
        mapped_slots,
        history_rows,
        current_rank_stride,
        block_rows,
        virtual_blocks,
    ):
        """Unpack current rows and write both cache planes on their owner rank.

        Complete pages alias the padded current region. Partial pages also
        update their compact history address, as described by the batch plan.
        """
        if (
            slots.numel() != packet.shape[0]
            or slots.dtype not in (torch.int32, torch.int64)
            or not slots.is_contiguous()
        ):
            raise ValueError("Prefill write slots must cover the current rows")
        store_current(
            packet,
            kv,
            index,
            kv_cache,
            index_cache,
            slots,
            mapped_slots,
            direct,
            history_rows=history_rows,
            current_rank_stride=current_rank_stride,
            block_rows=block_rows,
            virtual_blocks=virtual_blocks,
            degree=self.degree,
            rank=self.rank,
        )
