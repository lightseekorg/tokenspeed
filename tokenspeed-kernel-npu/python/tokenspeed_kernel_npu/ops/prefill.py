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

"""Shared Prefill KV transport and owner-cache commit."""

import torch
import torch.distributed as dist
import torch_npu


class AscendDSAPrefill:
    """Temporary KV/Indexer-K gathers on the caller's execution stream.

    History and current rows each use one collective. DSA packs both cache
    planes into the same packet; MLA supplies only KV. The synchronous HCCL
    API orders its internal stream before consumers on the execution stream.
    Gather, patch and attention rely on that execution stream's queue order;
    the transport does not insert dependencies between caller streams.
    Temporary buffers and TS forward inputs use that same allocation stream;
    cache and router owners retain persistent inputs.
    No transport workspace is kept between layers.
    """

    def __init__(
        self,
        kv_cache,
        index_cache,
        *,
        ranks,
        history_group,
        max_current_rows,
        history_null_rows,
    ):
        self.degree = len(ranks)
        self.rank = ranks.index(dist.get_rank())
        self.device = kv_cache.device
        self.local_rows = kv_cache.numel() // 576
        self.max_current_rows = max_current_rows
        self.history_null_rows = history_null_rows
        self._has_index = index_cache is not None
        self._width = 704 if self._has_index else 576
        if (
            self.degree < 1
            or self.local_rows < 1
            or max_current_rows < 1
            or history_null_rows < 1
        ):
            raise ValueError("Prefill needs positive cache and token geometry")
        self._validate_cache(kv_cache, index_cache)
        import flash_ops  # noqa: F401

        if not hasattr(torch.ops.custom, "dsa_cp_patch"):
            raise RuntimeError("flash_ops is missing the ACLNN DSA CP patch adapter")
        if self.degree > 1 and (
            history_group is None
            or tuple(dist.get_process_group_ranks(history_group)) != tuple(ranks)
        ):
            raise ValueError("Prefill collective group must match cache owners")
        self.history_group = history_group
        self.layer_id = None
        self.batch_key = None
        self._sources = None
        self._meta = {}
        self._current = {}
        self._current_rows = 0
        self._send_rows = 0
        self.history_rows = 0
        self.index = None
        self.kv = None
        self._current_gathered = False
        self._closed = False
        self.peak_buffer_bytes = 0

    def _validate_cache(self, kv_cache, index_cache):
        if (index_cache is not None) != self._has_index:
            raise ValueError("Prefill index-cache presence changed")
        for value, width in ((kv_cache, 576), (index_cache, 128)):
            if value is None:
                continue
            if (
                value.device != self.device
                or value.dtype != torch.bfloat16
                or not value.is_contiguous()
                or value.numel() != self.local_rows * width
            ):
                raise ValueError("Prefill needs matching contiguous BF16 cache planes")

    @property
    def workspace_bytes(self):
        return sum(
            x.numel() * x.element_size() for x in (self.kv, self.index) if x is not None
        )

    def _gather(self, packet, counts):
        receive = torch.empty(
            (sum(counts), self._width), dtype=torch.bfloat16, device=self.device
        )
        self.peak_buffer_bytes = max(
            self.peak_buffer_bytes,
            self.workspace_bytes
            + (packet.numel() + receive.numel()) * packet.element_size(),
        )
        if sum(counts):
            if self.degree == 1:
                receive.copy_(packet)
            else:
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
        """Submit history once; repeating the same batch/layer is an ensure."""
        if self._closed:
            raise RuntimeError("Prefill transport is closed")
        self._validate_cache(kv_cache, index_cache)
        if self.layer_id is not None:
            if self.layer_id != layer_id or self.batch_key is not batch_key:
                raise RuntimeError(
                    "Prefill history from a different batch/layer is active"
                )
            if any(
                old is not None and not old.view(-1).is_set_to(new.view(-1))
                for old, new in zip(self._sources, (kv_cache, index_cache), strict=True)
            ):
                raise ValueError("Prefill cache storage changed during consumption")
            return
        if (
            len(history_counts) != self.degree
            or min(history_counts) < 0
            or history_slots.device != self.device
            or history_slots.dtype != torch.int64
            or not history_slots.is_contiguous()
            or history_slots.numel() != history_counts[self.rank]
            or not 0 <= total_rows <= self.max_current_rows
        ):
            raise ValueError("Prefill requires matching batch page-gather geometry")
        self.layer_id, self.batch_key = layer_id, batch_key
        self._sources = (kv_cache, index_cache)
        self._meta = {"history_slots": history_slots}
        self._current = {}
        self._current_gathered = False
        self._current_rows = total_rows
        self._send_rows = -(-total_rows // (self.degree * 128)) * 128
        self.history_rows = self.history_null_rows + sum(history_counts)
        rows = self.history_rows + self.degree * self._send_rows
        self.kv = torch.empty((rows, 576), dtype=torch.bfloat16, device=self.device)
        self.index = (
            torch.empty((rows, 128), dtype=torch.bfloat16, device=self.device)
            if self._has_index
            else None
        )
        for plane in (self.kv, self.index):
            if plane is not None:
                plane[: self.history_null_rows].zero_()
        self.peak_buffer_bytes = max(self.peak_buffer_bytes, self.workspace_bytes)
        if has_history:
            packet = torch.empty(
                (history_counts[self.rank], self._width),
                dtype=torch.bfloat16,
                device=self.device,
            )
            offset = 0
            for cache, width in ((index_cache, 128), (kv_cache, 576)):
                if cache is not None:
                    selected = cache.view(-1, width).index_select(0, history_slots)
                    self.peak_buffer_bytes = max(
                        self.peak_buffer_bytes,
                        self.workspace_bytes + (packet.numel() + selected.numel()) * 2,
                    )
                    packet[:, offset : offset + width].copy_(selected)
                    del selected
                    offset += width
            receive = self._gather(packet, history_counts)
            begin, end = self.history_null_rows, self.history_rows
            if self._has_index:
                self.index[begin:end].copy_(receive[:, :128])
            self.kv[begin:end].copy_(receive[:, self._width - 576 :])
            # Packet and receive have no consumers beyond the queued copies.
            del packet, receive

    def gather_current(self, layer_id, value, *, width, total_rows):
        """Retain producers, then gather KV and Indexer-K together when ready."""
        if (
            layer_id != self.layer_id
            or self._sources is None
            or width not in (128, 576)
            or (width == 128 and not self._has_index)
        ):
            raise RuntimeError("Prefill current gather has no matching layer")
        local_count = total_rows // self.degree + (self.rank < total_rows % self.degree)
        if (
            width in self._current
            or total_rows != self._current_rows
            or value.device != self.device
            or value.dtype != torch.bfloat16
            or not value.is_contiguous()
            or value.numel() != local_count * width
        ):
            raise ValueError(
                "Current Prefill planes must match local token rows exactly once"
            )
        self._current[width] = value
        if len(self._current) != (2 if self._has_index else 1):
            return
        packet = torch.empty(
            (local_count, self._width), dtype=torch.bfloat16, device=self.device
        )
        offset = 0
        for channel in ((128, 576) if self._has_index else (576,)):
            packet[:, offset : offset + channel].copy_(
                self._current[channel].view(-1, channel)
            )
            offset += channel
        counts = tuple(
            total_rows // self.degree + (r < total_rows % self.degree)
            for r in range(self.degree)
        )
        receive = self._gather(packet, counts)
        offset = 0
        for owner, count in enumerate(counts):
            begin = self.history_rows + owner * self._send_rows
            if self._has_index:
                self.index[begin : begin + count].copy_(
                    receive[offset : offset + count, :128]
                )
            self.kv[begin : begin + count].copy_(
                receive[offset : offset + count, self._width - 576 :]
            )
            offset += count
        self._current_gathered = True
        self._current.clear()
        del packet, receive

    def patch(
        self,
        layer_id,
        slots,
        direct,
        *,
        slots_workspace,
        mapped_slots,
        width,
        block_rows,
        virtual_blocks,
    ):
        """Commit gathered current rows to the owner cache and compact pages."""
        if (
            layer_id != self.layer_id
            or not self._current_gathered
            or width not in (128, 576)
            or (width == 128 and not self._has_index)
        ):
            raise RuntimeError("Prefill update requires both current planes")
        if (
            slots.dtype not in (torch.int32, torch.int64)
            or not slots.is_contiguous()
            or slots.device != self.device
            or slots.numel() != self._current_rows
            or direct.dtype != torch.uint8
            or not direct.is_contiguous()
            or direct.device != self.device
            or direct.numel() != slots.numel()
            or slots_workspace.dtype != torch.int64
            or slots_workspace.device != self.device
            or not slots_workspace.is_contiguous()
            or slots_workspace.numel() < slots.numel()
            or mapped_slots is None
            or mapped_slots.dtype != torch.int64
            or mapped_slots.device != self.device
            or not mapped_slots.is_contiguous()
            or mapped_slots.numel() != slots.numel()
            or block_rows <= 0
            or self.local_rows % block_rows
        ):
            raise ValueError("Prefill update has incompatible slots or layout")
        committed = slots_workspace[: slots.numel()]
        committed.copy_(slots)
        self._meta[("patch", width)] = (slots, committed, direct, mapped_slots)
        target, cache = (
            (self.index, self._sources[1])
            if width == 128
            else (self.kv, self._sources[0])
        )
        torch.ops.custom.dsa_cp_patch(
            target[self.history_rows :],
            committed,
            cache,
            target,
            direct,
            slots.numel(),
            self.local_rows,
            virtual_blocks,
            self._send_rows,
            block_rows,
            self.degree,
            self.rank,
            width,
            mapped_slots=mapped_slots,
            gathered_rows=self.history_rows,
        )
        return target

    def finish(self, layer_id):
        """Release temporary buffers after queueing their last consumers."""
        if self.layer_id != layer_id:
            raise RuntimeError("Prefill completion differs from its active layer")
        self.layer_id = self.batch_key = self._sources = None
        self.kv = self.index = None
        self._current.clear()
        self._meta.clear()

    def close(self):
        """Release references; cache and collective owners manage completion."""
        if self._closed:
            return
        self.history_group = None
        self.kv = self.index = self._sources = self.batch_key = self.layer_id = None
        self._current.clear()
        self._meta.clear()
        self._closed = True
