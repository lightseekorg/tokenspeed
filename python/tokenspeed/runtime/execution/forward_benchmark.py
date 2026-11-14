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

"""Fixed decode-state snapshots for exclusive, direct-forward benchmarks.

Cache groups and slot exporters remain the owners. This module retains only
copies needed to undo one decode, not a second cache allocator or execution path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tokenspeed.runtime.execution.slot_state import SlotStateLayout


@dataclass
class _IndexedCopy:
    source: torch.Tensor
    indices: torch.Tensor
    saved: torch.Tensor

    def restore(self) -> None:
        self.source[self.indices] = self.saved


class FixedDecodeState:
    """Save the writable cache tail and per-slot state at a decode boundary.

    The caller settles all streams before capture/restore and keeps the same
    scheduler plan alive. Full-history pages before the writable tail cannot
    change in decode and need no copy. Save the allocation from one verify
    window before the committed frontier onward, including reserved future
    pages: a drafter can lag the target and write ahead after verification.
    All live sliding/state blocks are saved.
    """

    def __init__(self, arena, runtime_states, exporters, batch, *, stream):
        if batch.num_extends():
            raise ValueError("Fixed-state timing requires a decode-only batch")
        if arena.offload is not None or not arena.storage_plan.monolithic:
            raise NotImplementedError("Fixed-state timing requires device-resident KV")
        if any(spec.shard_count != 1 for spec in arena.cache_group_specs):
            raise NotImplementedError("Fixed-state timing requires replicated KV")
        self.layout = SlotStateLayout(exporters)
        self.stream = stream
        self.runtime_states = runtime_states
        self.slots = tuple(batch.request_pool_indices)
        self.request_ids = tuple(batch.request_ids)
        self.images = torch.empty(
            (len(self.slots), self.layout.nbytes),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=stream is not None,
        )
        self.copies: list[_IndexedCopy] = []
        self.ngram_owners = tuple(runtime_states.ngram_request_ids)
        prefix_lengths = runtime_states.valid_cache_lengths[list(self.slots)].tolist()
        tables = batch.block_tables
        groups = {spec.group_id: spec for spec in arena.cache_group_specs}
        for field in arena.plan.fields:
            spec = groups[field.group_id]
            pages = set()
            for row, width, prefix in zip(
                tables[field.group_id], batch.input_lengths, prefix_lengths, strict=True
            ):
                if spec.attn_kind == "full_history":
                    # Tables can include the entire output reservation. Their
                    # end is unrelated to the positions this forward writes.
                    first = max(0, prefix - width) // spec.block_granularity
                    row = row[first:]
                pages.update(page for page in row if page > 0)
            if pages:
                view = arena.field_pages(field.field_id).view(torch.uint8)
                ids = torch.tensor(sorted(pages), dtype=torch.int64, device=view.device)
                self.copies.append(_IndexedCopy(view, ids, view[ids].clone()))
        for tensor in (
            runtime_states.request_token_history_ids,
            runtime_states.draft_request_token_history_ids,
            runtime_states.ngram_accepted_tokens,
            runtime_states.ngram_needs_seed,
        ):
            if tensor is not None:
                ids = torch.tensor(self.slots, dtype=torch.int64, device=tensor.device)
                self.copies.append(_IndexedCopy(tensor, ids, tensor[ids].clone()))
        for i, (slot, rid) in enumerate(zip(self.slots, self.request_ids, strict=True)):
            self.layout.export(slot, self.images[i], stream, request_id=rid)
        if stream is not None:
            stream.synchronize()

    @property
    def nbytes(self) -> int:
        return self.images.nbytes + sum(copy.saved.nbytes for copy in self.copies)

    def restore(self) -> None:
        """Restore outside timing; the caller must enter the construction stream."""
        for copy in self.copies:
            copy.restore()
        for i, (slot, rid) in enumerate(zip(self.slots, self.request_ids, strict=True)):
            self.layout.import_(slot, self.images[i], self.stream, request_id=rid)
        self.runtime_states.ngram_request_ids[:] = self.ngram_owners
