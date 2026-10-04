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

"""Arena-owned KV offloading: authoritative Host rows and GPU compute buffers."""

from dataclasses import dataclass

import torch
from tokenspeed_kernel.ops.kvcache.offload import (
    accepted_ids,
    copy_rows,
    current_slots,
    materialize,
    reset_lru,
    seed_rows,
)

from tokenspeed.runtime.layers.attention.kv_cache.offload_config import KVOffloadConfig


@dataclass
class OffloadField:
    """Per-field KV storage, residency scratch and prefetch/writeback state."""

    host: torch.Tensor  # Authoritative full-history KV in pinned CPU memory.
    device: torch.Tensor  # GPU hot/reserved KV; also reused for recovery staging.
    keys: torch.Tensor  # History row ID per request-owned GPU slot; -1 if empty.
    seeded: torch.Tensor  # Per-request flag preventing repeated history preload.
    miss_ids: torch.Tensor  # Miss history IDs at first occurrences; -1 otherwise.
    miss_dst: torch.Tensor  # GPU destination rows paired with miss_ids.
    indices: torch.Tensor  # Top-K GPU read rows, preserving order and duplicates.
    entry_dest: torch.Tensor  # GPU row per first selection occurrence (scratch).
    hash_keys: torch.Tensor  # Global hash IDs; empty when using a shared table.
    hash_owners: torch.Tensor  # Global hash owner indices; empty for shared tables.
    lru_slots: torch.Tensor  # Ordinary slot IDs per request, oldest first.
    slot_order: torch.Tensor  # Protected slots, then reversed unprotected slots.
    free_counts: torch.Tensor  # Eviction-candidate count per batch row; write-only.
    miss_counts: torch.Tensor  # Unique misses per batch row, used by validation.
    current_full: torch.Tensor  # History destinations for decode/extend inputs.
    current_hot: torch.Tensor  # GPU write rows paired with current_full.
    accepted_full: torch.Tensor  # Writeback history IDs; rejected inputs are -1.
    ready: torch.cuda.Event  # Signals prefetch metadata and swap-in completion.
    prefetched: bool = False  # resolve() must wait for ready instead of reloading.
    active_tokens: int = 0  # Prepared write rows awaiting commit or extend flush.

    @property
    def device_nbytes(self) -> int:
        return sum(
            t.nbytes
            for t in (
                self.device,
                self.keys,
                self.seeded,
                self.miss_ids,
                self.miss_dst,
                self.indices,
                self.entry_dest,
                self.hash_keys,
                self.hash_owners,
                self.lru_slots,
                self.slot_order,
                self.free_counts,
                self.miss_counts,
                self.current_full,
                self.current_hot,
                self.accepted_full,
            )
        )


class SparseKVOffload:
    """Load selected Host KV and write accepted GPU rows back to history."""

    def __init__(self, arena, config: KVOffloadConfig, *, workspaces):
        self.config = config
        self.fields: dict[str, OffloadField] = {}
        self.prefetch_stream = torch.cuda.Stream(device=arena.device)
        self.write_stream = torch.cuda.Stream(device=arena.device)
        self.write_done = torch.cuda.Event()
        self.requests: torch.Tensor | None = None
        self.valid_requests: torch.Tensor | None = None
        self.batch_size = 0
        self.execution_stream: torch.cuda.Stream | None = None
        self.is_extend = False
        for workspace in workspaces:
            name = workspace.field_id
            host = arena.field(name)
            if (
                host.device.type != "cpu"
                or not host.is_pinned()
                or not host.is_contiguous()
            ):
                raise ValueError("offloaded fields require contiguous pinned host rows")

            def ints(n, fill=0):
                return torch.full((n,), fill, dtype=torch.int32, device=arena.device)

            metadata = {
                key: ints(
                    count,
                    (
                        -1
                        if key
                        in {
                            "keys",
                            "miss_ids",
                            "miss_dst",
                            "indices",
                            "current_full",
                            "accepted_full",
                        }
                        else 0
                    ),
                )
                for key, count in workspace.metadata
            }
            self.fields[name] = OffloadField(
                host=host,
                device=torch.zeros(
                    (workspace.device_rows, *workspace.row_shape),
                    dtype=host.dtype,
                    device=arena.device,
                ),
                **metadata,
                ready=torch.cuda.Event(),
            )
            reset_lru(self.fields[name].lru_slots, None, hot=config.hot_tokens)
            # Admission uses scheduler int64 IDs and may first occur after graph
            # warmup. Compile this reset before accepting any live request.
            reset_lru(
                self.fields[name].lru_slots,
                torch.zeros(1, dtype=torch.int64, device=arena.device),
                hot=config.hot_tokens,
            )

    def begin(
        self, requests: torch.Tensor, *, num_extends: int, stream: torch.cuda.Stream
    ):
        """Bind an offloading batch and mask null/padded scheduler slots.

        Reset per-field prefetch/current-write bookkeeping. Recovery is a
        separate all-extend batch and also invalidates hot tags, seeds, LRU.
        """
        if num_extends and num_extends != requests.numel():
            raise ValueError("D recovery must run in its own extend batch")
        self.execution_stream = stream
        self.is_extend = num_extends > 0
        if requests.numel() > self.config.request_slots:
            raise ValueError("batch exceeds offload admission capacity")
        self.batch_size = requests.numel()
        with torch.cuda.stream(stream):
            self.valid_requests = (requests > 0) & (
                requests < self.config.request_slots - 1
            )
            # Null and graph padding cannot install tags or copy KV.
            self.requests = torch.where(self.valid_requests, requests, 0)
            for state in self.fields.values():
                state.prefetched = False
                state.active_tokens = 0
                if self.is_extend:
                    state.keys.fill_(-1)
                    state.seeded.zero_()
                    reset_lru(state.lru_slots, None, hot=self.config.hot_tokens)

    def reset_requests(self, requests: torch.Tensor, *, stream: torch.cuda.Stream):
        """Invalidate KV offloading slots when their history view is reset.

        Used by admission/PD landing and local extend/recovery length resets.
        After PD, the scheduler waits for transfer completion before decode;
        only then may the hot buffer be seeded from the new history.
        """
        # GPU fences preserve slot ownership without blocking the forward
        # thread on unrelated requests' side-stream work.
        stream.wait_stream(self.prefetch_stream)
        stream.wait_stream(self.write_stream)
        with torch.cuda.stream(stream):
            for state in self.fields.values():
                state.keys.view(self.config.request_slots, -1)[requests.long()] = -1
                state.seeded[requests.long()] = 0
                reset_lru(state.lru_slots, requests, hot=self.config.hot_tokens)

    def seed(self, name, history_rows, hot_rows):
        """Copy initial offloading history rows and install their hot tags.

        The GPU seed flags avoid reinitializing established request slots.
        """
        if name not in self.fields:
            return
        state = self.fields[name]
        with torch.cuda.stream(self.execution_stream):
            history_rows = torch.where(self.valid_requests[:, None], history_rows, -1)
            seed_rows(
                state.host,
                state.device,
                state.keys,
                state.seeded,
                self.requests,
                history_rows,
                hot_rows,
            )

    def prepare_extend(self, name, write_history):
        """Reserve offloading staging writes for this recovery chunk.

        Record history destinations and assign GPU rows 1..n. No selected
        history is gathered here; row zero remains the null row.
        """
        state = self.fields[name]
        n = write_history.numel()
        if n > self.config.max_extend_tokens:
            raise ValueError("recovery chunk exceeds planned forward capacity")
        with torch.cuda.stream(self.execution_stream):
            state.current_full[:n].copy_(write_history)
            state.current_hot[:n].copy_(
                torch.arange(1, n + 1, dtype=torch.int32, device=state.device.device)
            )
            state.active_tokens = n
        return state.current_hot[:n]

    def prefill_tiles(self, name, selected):
        """Flush recovery KV to Host, then gather offloading attention tiles.

        These operations run when the iterator advances, not when created.
        Clear active_tokens after flushing: later writeback must not read the
        overwritten projection rows. Consume each tile on the execution stream
        before next() reuses the same GPU storage for another tile.
        """
        state = self.fields[name]
        n = state.active_tokens
        if not self.is_extend or n != selected.shape[0]:
            raise ValueError("recovery selection must match the prepared chunk")
        if selected.shape[1] != self.config.topk:
            raise ValueError("recovery selection width differs from configured top-k")
        copy_rows(
            state.host,
            state.device,
            state.current_full[:n],
            state.current_hot[:n],
            writeback=True,
        )
        # The payload will now be overwritten. End-of-forward commit cannot
        # read these projection addresses again; the whole chunk is on Host.
        state.active_tokens = 0
        width = self.config.recovery_query_tokens
        for start in range(0, n, width):
            end = min(n, start + width)
            rows = selected[start:end].contiguous()
            destinations = torch.arange(
                1, rows.numel() + 1, device=rows.device, dtype=torch.int32
            )
            copy_rows(
                state.host, state.device, rows.view(-1), destinations, writeback=False
            )
            mapped = torch.where(rows > 0, destinations.view_as(rows), -1)
            yield slice(start, end), mapped

    def _prepare(self, state, positions, full):
        """Install current-token history tags and reserve GPU write rows.

        This prepares offloading metadata, not the consumer's KV projection.
        Current rows resolve to reserved/ring storage rather than stale Host KV.
        """
        if self.requests is None:
            raise RuntimeError("offload request metadata has not been prepared")
        n = full.numel()
        if n != self.batch_size * self.config.queries:
            raise ValueError(
                "offload query width differs from the configured verify width"
            )
        with torch.cuda.stream(self.execution_stream):
            if not self.config.cyclic_tokens:
                state.keys.view(self.config.request_slots, -1)[
                    :, self.config.hot_tokens :
                ].index_fill_(0, self.requests.long(), -1)
            full = torch.where(
                self.valid_requests.repeat_interleave(self.config.queries), full, -1
            )
            state.current_full[:n].copy_(full)
            state.active_tokens = n
            current_slots(
                self.requests,
                positions,
                full,
                state.keys,
                state.current_hot[:n],
                hot=self.config.hot_tokens,
                stride=self.config.buffer_tokens,
                queries=self.config.queries,
                cyclic=self.config.cyclic_tokens,
            )

    def _load(self, name, state, topk):
        """Resolve offloading hits/misses, update LRU, and load Host misses.

        materialize writes one hot slot per selection entry, retaining masks,
        order, and duplicates while coalescing physical loads internally.
        """
        n = topk.numel()
        topk = torch.where(
            self.valid_requests.repeat_interleave(self.config.queries)[:, None],
            topk,
            -1,
        )
        with torch.profiler.record_function(f"kv_offload.swap_in.{name}"):
            materialize(
                topk.contiguous(),
                self.requests,
                state.keys,
                state.current_hot,
                state.miss_ids,
                state.miss_dst,
                state.indices[:n],
                state.host,
                state.device,
                state.lru_slots,
                state.slot_order,
                state.free_counts,
                state.miss_counts,
                state.entry_dest,
                state.hash_keys,
                state.hash_owners,
                hot=self.config.hot_tokens,
                stride=self.config.buffer_tokens,
                queries=self.config.queries,
            )

    def prefetch(self, name: str, topk, positions, full):
        """Prepare offloading writes and load this consumer on a side stream.

        ready covers lookup, mapping, and Host-to-hot copies. It does not
        cover the consumer's later projection of its current-token KV.
        """
        if not self.config.overlap or name not in self.fields:
            return
        state = self.fields[name]
        self._prepare(state, positions, full)
        self.prefetch_stream.wait_stream(self.execution_stream)
        with torch.cuda.stream(self.prefetch_stream):
            self._load(name, state, topk)
            state.ready.record()
        topk.record_stream(self.prefetch_stream)
        state.prefetched = True

    def resolve(self, name: str, topk, positions, full):
        """Join offloading prefetch or load inline, then return GPU reads/writes."""
        state = self.fields[name]
        if state.prefetched:
            self.execution_stream.wait_event(state.ready)
            state.prefetched = False
        else:
            self._prepare(state, positions, full)
            with torch.cuda.stream(self.execution_stream):
                self._load(name, state, topk)
        return (
            state.indices[: topk.numel()].view_as(topk),
            state.current_hot[: full.numel()],
        )

    def commit(self, accepted: torch.Tensor):
        """Write accepted offloading compute rows back to authoritative Host KV.

        Decode masks rejected verify inputs; extend retains all chunk rows.
        Recovery tiles may have flushed them already (active_tokens == 0).
        The execution stream joins write_done before scheduler completion,
        including EOS rounds, cancellation cleanup, and request-slot reuse.
        """
        self.write_stream.wait_stream(self.execution_stream)
        with torch.cuda.stream(self.write_stream):
            for name, state in self.fields.items():
                n = state.active_tokens
                if not n:
                    continue
                with torch.profiler.record_function(f"kv_offload.write_through.{name}"):
                    if self.is_extend:
                        state.accepted_full[:n].copy_(state.current_full[:n])
                    else:
                        accepted_ids(
                            state.current_full[:n],
                            accepted,
                            state.accepted_full[:n],
                            queries=self.config.queries,
                        )
                    copy_rows(
                        state.host,
                        state.device,
                        state.accepted_full[:n],
                        state.current_hot[:n],
                        writeback=True,
                    )
            self.write_done.record()
        self.execution_stream.wait_event(self.write_done)

    def synchronize(self):
        self.prefetch_stream.synchronize()
        self.write_stream.synchronize()

    def clear(self):
        for state in self.fields.values():
            state.keys.fill_(-1)
            state.seeded.zero_()
            reset_lru(state.lru_slots, None, hot=self.config.hot_tokens)
            state.device.zero_()
