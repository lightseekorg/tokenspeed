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

"""Shared prefill history transport and owner-cache commit."""

import torch
import torch.distributed as dist
import torch_npu
from tokenspeed_kernel.ops.attention.ascend_prefill_workspace import (
    AscendPrefillWorkspaceLayout,
)
from tokenspeed_kernel_npu.ops._memory import NativeBuffer
from tokenspeed_kernel_npu.ops._prefill_transport import (
    gather_history,
    pack_current,
    store_current,
)


class AscendDSAPrefill:
    """One bounded native workspace, reused by successive attention layers.

    The executor orders prefetch, attention, and the next layer on its execution
    stream. HCCL's synchronous interface orders its internal communication stream
    before subsequent consumers. A synchronized warmup/serving stream handoff is
    valid; stream identity is not part of the workspace's contract.
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
        self.layout = AscendPrefillWorkspaceLayout(
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
        self._storage = NativeBuffer(self.layout.nbytes, self.device)
        storage = self._storage.bytes.view(torch.bfloat16)
        offset = 0

        def plane(rows, width):
            nonlocal offset
            result = storage[offset : offset + rows * width].view(rows, width)
            offset += rows * width
            return result

        self._kv = plane(self.layout.plane_rows, 576)
        self._index = (
            plane(self.layout.plane_rows, 128) if self.layout.has_index else None
        )
        self._send = plane(self.layout.send_rows, self.layout.width)
        self._receive = plane(self.layout.receive_rows, self.layout.width)
        # History sends use separate contiguous planes so the collectives can
        # write directly to attention's planes. Current packets reuse this space.
        send = self._send.view(-1)
        split = self.layout.send_rows * 576
        self._history_send_kv = send[:split].view(-1, 576)
        self._history_send_index = (
            send[split:].view(-1, 128) if self.layout.has_index else None
        )
        # No later operation writes the null page; initialize it once.
        self._kv[:history_null_rows].zero_()
        if self._index is not None:
            self._index[:history_null_rows].zero_()
        self.layer_id = None
        self.batch_key = None
        self._sources = None
        self._current = {}
        self._current_receive = None
        self._current_rows = 0
        self._send_rows = 0
        self.history_rows = 0
        self.kv = None
        self.index = None
        self.peak_buffer_bytes = 0

    @property
    def workspace_bytes(self):
        return self.layout.nbytes if self._storage is not None else 0

    def _gather(self, packet, counts, receive):
        if sum(counts):
            if self.degree == 1:
                return packet
            torch_npu.distributed.all_gather_into_tensor_uneven(
                output=receive,
                input=packet,
                output_split_sizes=list(counts),
                group=self.history_group,
            )
        return receive

    def prefetch(
        self,
        layer_id,
        kv_cache,
        index_cache,
        *,
        batch_key,
        total_rows,
        has_history,
        history_slots,
        history_counts,
    ):
        """Gather this layer's historical pages once, ahead of its projections."""
        if self._storage is None:
            raise RuntimeError("Prefill workspace is closed")
        if self.layer_id is not None:
            if self.layer_id != layer_id or self.batch_key is not batch_key:
                raise RuntimeError(
                    "Prefill history from a different batch/layer is active"
                )
            return
        if (
            len(history_counts) != self.degree
            or min(history_counts) < 0
            or sum(history_counts) > self.layout.history_capacity
            or history_slots.numel() != history_counts[self.rank]
            or not 0 <= total_rows <= self.layout.current_capacity
        ):
            raise ValueError("Prefill batch exceeds its planned workspace capacity")
        self.layer_id, self.batch_key = layer_id, batch_key
        self._sources = (kv_cache, index_cache)
        self._current_rows = total_rows
        self._send_rows = (
            (total_rows + self.degree * 128 - 1) // (self.degree * 128) * 128
        )
        self.history_rows = self.layout.null_rows + sum(history_counts)
        rows = self.history_rows + self.degree * self._send_rows
        self.kv = self._kv[:rows]
        self.index = self._index[:rows] if self._index is not None else None
        self.peak_buffer_bytes = max(
            self.peak_buffer_bytes,
            (rows + history_counts[self.rank]) * self.layout.width * 2,
        )
        if has_history:
            first = self.layout.null_rows
            receive_kv = self.kv[first : self.history_rows]
            receive_index = (
                self.index[first : self.history_rows]
                if self.index is not None
                else None
            )
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
            if self.degree > 1:
                self._gather(send_kv, history_counts, receive_kv)
                if send_index is not None:
                    self._gather(send_index, history_counts, receive_index)

    def gather_current(self, layer_id, value, *, width, total_rows):
        """Gather current KV and index keys together once both projections exist."""
        expected_widths = (128, 576) if self.layout.has_index else (576,)
        local_rows = total_rows // self.degree + (self.rank < total_rows % self.degree)
        if (
            layer_id != self.layer_id
            or width not in expected_widths
            or width in self._current
            or total_rows != self._current_rows
            or value.numel() != local_rows * width
            or value.dtype != torch.bfloat16
            or not value.is_contiguous()
        ):
            raise ValueError(
                "Current prefill rows must match the active layer's projections"
            )
        self._current[width] = value
        if len(self._current) != len(expected_widths):
            return
        packet = self._send[:local_rows]
        pack_current(self._current[576], self._current.get(128), packet)
        counts = tuple(
            total_rows // self.degree + (rank < total_rows % self.degree)
            for rank in range(self.degree)
        )
        self._current_receive = self._gather(packet, counts, self._receive[:total_rows])
        self.peak_buffer_bytes = max(
            self.peak_buffer_bytes,
            (self.kv.shape[0] + local_rows + total_rows) * self.layout.width * 2,
        )
        self._current.clear()

    def store_current(
        self, layer_id, slots, direct, *, mapped_slots, block_rows, virtual_blocks
    ):
        """Unpack current rows and write both cache planes on their owner rank.

        Complete pages alias the padded current region. Partial pages also
        update their compact history address, as described by the batch plan.
        """
        if layer_id != self.layer_id or self._current_receive is None:
            raise RuntimeError("Gather current prefill projections before storing them")
        if (
            slots.numel() != self._current_rows
            or slots.dtype not in (torch.int32, torch.int64)
            or not slots.is_contiguous()
        ):
            raise ValueError("Prefill write slots must cover the current rows")
        store_current(
            self._current_receive,
            self.kv,
            self.index,
            *self._sources,
            slots,
            mapped_slots,
            direct,
            history_rows=self.history_rows,
            send_rows=self._send_rows,
            block_rows=block_rows,
            virtual_blocks=virtual_blocks,
            degree=self.degree,
            rank=self.rank,
        )
        self._current_receive = None

    def finish(self, layer_id):
        """Release the active layer's references; retain the workspace allocation."""
        if layer_id != self.layer_id:
            raise RuntimeError("Prefill completion differs from its active layer")
        self.layer_id = self.batch_key = self._sources = None
        self.kv = self.index = self._current_receive = None
        self._current.clear()

    def close(self):
        """Finish queued consumers and free the native workspace on cache rebind."""
        self.layer_id = self.batch_key = self._sources = None
        self.kv = self.index = self._current_receive = None
        self._kv = self._index = self._send = self._receive = None
        self._history_send_kv = self._history_send_index = None
        self._current.clear()
        if self._storage is not None:
            self._storage.close()
            self._storage = None
