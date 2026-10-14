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

"""Sparse storage policy and its runtime workspace geometry."""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class KVOffloadPolicy:
    """Model declarations; no scheduler slots or physical page arithmetic."""

    field_ids: tuple[str, ...]
    hot_tokens: int
    reserved_tokens: int
    topk: int
    queries: int
    host_budget_bytes: int
    overlap: bool
    cyclic_tokens: int
    # Producer field -> fields consuming that producer's selection.
    selection_consumers: tuple[tuple[str, str], ...]

    def bind(self, *, request_slots: int, device_rows: int):
        return KVOffloadConfig(
            **{f.name: getattr(self, f.name) for f in fields(KVOffloadPolicy)},
            request_slots=request_slots,
            device_rows=device_rows,
        )


@dataclass(frozen=True)
class KVOffloadConfig(KVOffloadPolicy):
    """Bound once by the common recipe, from the executor's slot geometry."""

    request_slots: int
    device_rows: int

    def __post_init__(self):
        for name in (
            "hot_tokens",
            "reserved_tokens",
            "request_slots",
            "topk",
            "queries",
            "host_budget_bytes",
            "device_rows",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not self.field_ids or len(set(self.field_ids)) != len(self.field_ids):
            raise ValueError("offloaded fields must be nonempty and unique")
        if (
            type(self.cyclic_tokens) is not int
            or not 0 <= self.cyclic_tokens <= self.reserved_tokens
        ):
            raise ValueError("cyclic tokens must fit the reserved region")
        if self.cyclic_tokens and self.cyclic_tokens <= self.queries:
            raise ValueError("cyclic region must include history and current tokens")
        if self.reserved_tokens < self.queries:
            raise ValueError("reserved slots must hold the whole verify window")
        if self.hot_tokens < self.topk or self.hot_tokens & (self.hot_tokens - 1):
            raise ValueError("hot capacity must be a power of two covering top-k")
        if self.device_rows >= 2**31:
            raise ValueError("KV offload row IDs must fit int32")
        if self.request_slots < 3 or self.device_rows < self.minimum_device_rows:
            raise ValueError("KV offload pool must hold all request slots")
        if len(set(self.selection_consumers)) != len(self.selection_consumers):
            raise ValueError("duplicate selection dependency")

    @property
    def buffer_tokens(self):
        return self.hot_tokens + self.reserved_tokens

    @property
    def hot_rows(self):
        return self.request_slots * self.buffer_tokens

    @property
    def minimum_device_rows(self):
        return self.hot_rows

    def metadata_counts(self) -> dict[str, int]:
        """Named int32 allocations, shared by the allocator and byte planner."""
        currents = self.request_slots * self.queries
        from tokenspeed_kernel.ops.kvcache.offload import hash_geometry

        table_slots, shared = hash_geometry(self.queries, self.topk)
        hash_entries = 0 if shared else self.request_slots * table_slots
        selected = self.request_slots * self.queries * self.topk
        return {
            "keys": self.hot_rows,
            "seeded": self.request_slots,
            "miss_ids": selected,
            "miss_dst": selected,
            "indices": selected,
            "entry_dest": selected,
            "hash_keys": hash_entries,
            "hash_owners": hash_entries,
            "lru_slots": self.request_slots * self.hot_tokens,
            "slot_order": self.request_slots * self.hot_tokens,
            "free_counts": self.request_slots,
            "miss_counts": self.request_slots,
            "current_full": currents,
            "current_hot": currents,
            "accepted_full": currents,
        }

    def temporary_bytes(self) -> int:
        """Decode hash scratch is accounted for in persistent metadata."""
        return 0
