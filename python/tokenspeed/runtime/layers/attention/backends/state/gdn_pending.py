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

"""Batch-banked GDN replay workspace and slot-generation ownership."""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention.gdn import (
    GdnPendingReplay,
    gdn_pending_metadata,
    gdn_replay_commit,
)

from tokenspeed.runtime.execution.slot_state import (
    pack_slot_rows,
    slot_state_image_bytes,
    stream_scope,
    unpack_slot_rows,
)


def deferred_gdn_metadata_bytes(*, layers: int, capacity: int, slot_count: int) -> int:
    """Bytes allocated for slot records and two batch metadata banks."""
    return 32 * slot_count + (81 + 16 * layers) * capacity + 12


class GdnPendingState:
    """Keep payloads by batch; only ownership metadata is indexed by slot.

    A slot record holds (generation, bank/row, frontier, current batch row).
    The payload row also keeps
    its generation, so reordering and slot reuse cannot revive an old prefix.
    All consumers run on the execution stream; a bank is overwritten only
    after its dropout materialization and the previous forward have landed.
    """

    token_derived_slot_state: tuple[str, ...] = ()
    constant_slot_state: tuple[str, ...] = ()

    def __init__(
        self,
        payload: torch.Tensor,
        parameters: torch.Tensor,
        *,
        slot_count: int,
        geometry: tuple[int, int, int, int],
        state_addresses: torch.Tensor,
        state_row_strides: torch.Tensor,
        rows_aligned: bool,
        group_indices: torch.Tensor,
    ):
        layers, tokens, width = payload.shape
        self.capacity = tokens // 4
        self.geometry = geometry
        self.parameters = parameters
        self.state_addresses = state_addresses
        self.state_row_strides = state_row_strides
        self.rows_aligned = rows_aligned
        self.payload = torch.empty(
            (2, layers, tokens, width), dtype=payload.dtype, device=payload.device
        )
        self.records = torch.zeros(
            (slot_count, 4), dtype=torch.int64, device=payload.device
        )
        self.records[:, 1].fill_(-1)
        self.records[:, 3].fill_(-1)
        self.row_slots = torch.zeros(
            (2, self.capacity), dtype=torch.int64, device=payload.device
        )
        self.generations = torch.zeros_like(self.row_slots)
        self.steps = torch.zeros(
            (2, self.capacity), dtype=torch.int32, device=payload.device
        )
        self.pages = torch.full(
            (2, layers, self.capacity), -1, dtype=torch.int32, device=payload.device
        )
        self.valid = torch.zeros(
            (2, self.capacity), dtype=torch.bool, device=payload.device
        )
        self.bank = torch.zeros(1, dtype=torch.int32, device=payload.device)
        self.rows = torch.full(
            (self.capacity,), -1, dtype=torch.int32, device=payload.device
        )
        self.slots = torch.zeros(
            self.capacity, dtype=torch.int64, device=payload.device
        )
        self.reads = torch.full(
            (layers, self.capacity), -1, dtype=torch.int32, device=payload.device
        )
        self.writes = torch.full_like(self.reads, -1)
        self.immediate = torch.zeros_like(self.valid)
        self.current_stopped = torch.zeros(
            self.capacity, dtype=torch.bool, device=payload.device
        )
        self.current_frontier = torch.zeros(
            self.capacity, dtype=torch.int32, device=payload.device
        )
        self.current_steps = torch.zeros(
            self.capacity, dtype=torch.int32, device=payload.device
        )
        self.consumed = torch.zeros_like(self.valid)
        self.flush = torch.zeros_like(self.valid)
        self.request_workspaces = torch.empty(
            (2, self.capacity + 1), dtype=torch.int32, device=payload.device
        )
        self.status = torch.empty(
            self.capacity, dtype=torch.int32, device=payload.device
        )
        self.group_indices = group_indices
        self.buffers = {
            name: value
            for name, value in vars(self).items()
            if isinstance(value, torch.Tensor)
        }
        self.enabled = False

    @property
    def nbytes(self) -> int:
        borrowed = {
            "parameters",
            "state_addresses",
            "state_row_strides",
            "group_indices",
        }
        return sum(t.nbytes for name, t in self.buffers.items() if name not in borrowed)

    def descriptor(self, layer: int, batch: int) -> GdnPendingReplay:
        return GdnPendingReplay(
            self.payload, layer, self.rows[:batch], self.steps, self.pages, self.bank
        )

    def warmup(self) -> None:
        """Prepare masked flush and ordinary commit before serving or capture.

        All destinations start at -1, so the ordinary launch touches no
        recurrent state. Mixed/unsupported forwards still need this variant.
        """
        self.materialize(None)
        # Target capture does not include the runner's acceptance. Prepare
        # those metadata stages too, while every slot and length is zero.
        self.prepare_acceptance(self.current_stopped, self.current_frontier)
        self.accept(self.current_steps)
        self.pending_status(self.slots)
        gdn_replay_commit(
            self.payload[0],
            self.parameters,
            state_addresses=self.state_addresses,
            state_row_strides=self.state_row_strides,
            read_indices=self.pages[0],
            write_indices=self.pages[0],
            accepted_length=self.steps[0],
            request_mask=None,
            draft_token_num=4,
            geometry=self.geometry,
            state_dtype=torch.bfloat16,
            rows_aligned=self.rows_aligned,
        )

    def _metadata(
        self,
        operation: str,
        *,
        input_slots=None,
        input_frontier=None,
        input_stopped=None,
        input_accepted=None,
        input_reads=None,
        input_writes=None,
        batch: int,
        requested: int,
    ) -> None:
        gdn_pending_metadata(
            operation=operation,
            buffers=self.buffers,
            input_slots=self.slots if input_slots is None else input_slots,
            input_frontier=(
                self.current_frontier if input_frontier is None else input_frontier
            ),
            input_stopped=(
                self.current_stopped if input_stopped is None else input_stopped
            ),
            input_accepted=(
                self.current_steps if input_accepted is None else input_accepted
            ),
            input_reads=self.reads if input_reads is None else input_reads,
            input_writes=self.writes if input_writes is None else input_writes,
            batch=batch,
            requested=requested,
        )

    def _commit(self, selection: torch.Tensor, *, current: bool) -> None:
        for bank in range(2):
            pages = self.pages[bank]
            gdn_replay_commit(
                self.payload[bank],
                self.parameters,
                state_addresses=self.state_addresses,
                state_row_strides=self.state_row_strides,
                read_indices=self.reads if current else pages,
                write_indices=self.writes if current else pages,
                accepted_length=self.current_steps if current else self.steps[bank],
                request_mask=selection[bank],
                request_workspace=self.request_workspaces[bank],
                draft_token_num=4,
                geometry=self.geometry,
                state_dtype=torch.bfloat16,
                rows_aligned=self.rows_aligned,
            )

    def materialize(self, slots: torch.Tensor | None) -> None:
        if slots is not None:
            slots = slots.to(dtype=torch.int64)
        self._metadata(
            "materialize",
            input_slots=slots,
            batch=0,
            requested=-1 if slots is None else slots.numel(),
        )
        self._commit(self.flush, current=False)
        self._metadata("retire", batch=0, requested=0)

    def prepare(
        self,
        slots: torch.Tensor,
        frontier: torch.Tensor,
        *,
        enabled: bool,
        reads: torch.Tensor | None,
        writes: torch.Tensor | None,
    ) -> None:
        self.enabled = enabled
        if not enabled:
            self.materialize(None)
            return
        count = slots.numel()
        self._metadata("begin", batch=count, requested=0)
        self._metadata(
            "snapshot",
            input_slots=slots,
            input_frontier=frontier,
            input_reads=reads,
            input_writes=writes,
            batch=count,
            requested=0,
        )
        self._metadata("dropouts", batch=count, requested=0)
        self._commit(self.flush, current=False)
        self._metadata("retire", batch=count, requested=0)

    def prepare_acceptance(self, stopped: torch.Tensor, frontier: torch.Tensor) -> None:
        self._metadata(
            "accept_facts",
            input_stopped=stopped,
            input_frontier=frontier,
            batch=0,
            requested=0,
        )

    def accept(self, accepted: torch.Tensor) -> None:
        self._metadata(
            "accept", input_accepted=accepted, batch=accepted.numel(), requested=0
        )
        self._commit(self.immediate, current=True)

    def pending_status(self, slots: torch.Tensor) -> torch.Tensor:
        """Capture readiness before the normal completion's asynchronous D2H."""
        self._metadata(
            "status",
            input_slots=slots.to(dtype=torch.int64),
            batch=slots.numel(),
            requested=0,
        )
        return self.status[: slots.numel()]

    def slot_state_rows(self, slot: int) -> list[torch.Tensor]:
        return [self.records[slot]]

    def slot_state_bytes(self) -> int:
        return slot_state_image_bytes(self.slot_state_rows(0))

    def export_slot_state(
        self, slot: int, out: torch.Tensor, stream, *, request_id: str
    ) -> None:
        del request_id
        with stream_scope(stream):
            torch._assert_async(
                self.records[slot, 1] < 0,
                "pending GDN state must be materialized before export",
            )
            pack_slot_rows(self.slot_state_rows(slot), out, None)

    def import_slot_state(
        self, slot: int, src: torch.Tensor, stream, *, request_id: str
    ) -> None:
        del request_id
        with stream_scope(stream):
            unpack_slot_rows(self.slot_state_rows(slot), src, None)
            torch._assert_async(
                self.records[slot, 1] < 0,
                "a GDN snapshot cannot contain pending payloads",
            )
