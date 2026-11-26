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

"""Physical state fields shared by an ordered group of Decode cache owners."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class StateShardField:
    """One rank's packed row slices for a logical latest-snapshot field.

    Each request owner has a separate interval of pages in this allocation.
    The full field splits equally across members within each of its rows.
    """

    base: int
    page_stride: int
    pages_per_owner: int
    rows_per_page: int


class StateShardRegistration:
    """Assemble and validate one device's contribution before publishing any grant."""

    def __init__(self, rank: int, attention_tp: int) -> None:
        self.rank = rank
        self.attention_tp = attention_tp
        self.group = None
        self.fields: dict[str, StateShardField] = {}

    def add(
        self,
        group,
        fields: dict[str, StateShardField],
        *,
        regions: tuple[tuple[int, int], ...],
        registered_regions: tuple[tuple[int, int], ...],
    ) -> tuple[tuple[int, int], ...]:
        """Register the local allocations before the native transfer engine starts."""
        import torch.distributed as dist

        if self.attention_tp != 1 or dist.get_backend(group) != "gloo":
            raise ValueError("distributed state requires attention TP1 and a CPU group")
        ranks = dist.get_process_group_ranks(group)
        if len(ranks) < 2 or self.rank not in ranks or not fields:
            raise ValueError(
                "distributed state needs local fields and a participating rank"
            )
        if self.group is not None and self.group is not group:
            raise ValueError("state fields must share one ordered process group")
        if self.fields.keys() & fields.keys():
            raise ValueError("duplicate distributed cache field")
        registered = list(registered_regions)
        for start, length in regions:
            if start <= 0 or length <= 0 or start + length > 2**64:
                raise ValueError("invalid state backing allocation")
            if not any(
                base <= start and start + length <= base + size
                for base, size in registered
            ):
                if any(
                    start < base + size and base < start + length
                    for base, size in registered
                ):
                    raise ValueError(
                        "state backing allocation partially overlaps a registered region"
                    )
                registered.append((start, length))
        for field in fields.values():
            if (
                min(field.base, field.page_stride, field.rows_per_page) <= 0
                or field.pages_per_owner < 2
            ):
                raise ValueError("invalid distributed cache field geometry")
            size = len(ranks) * field.pages_per_owner * field.page_stride
            if not any(
                start <= field.base and field.base + size <= start + length
                for start, length in registered
            ):
                raise ValueError("state field lies outside its backing allocation")
        self.fields.update(fields)
        self.group = group
        return tuple(registered)

    def confirm(self, local_error: str | None) -> None:
        """Publish the group only after every engine validated its local allocations."""
        import torch.distributed as dist

        errors = [None] * dist.get_world_size(self.group)
        dist.all_gather_object(errors, local_error, group=self.group)
        for error in errors:
            if error is not None:
                raise ValueError(error)

    def exchange(self, session: str, cache: str) -> bytes | None:
        """Share live transport identities once, after every rank registered its memory."""
        if self.group is None:
            return None
        import torch.distributed as dist

        group = self.group
        ranks = dist.get_process_group_ranks(group)
        local = dict(
            rank=self.rank,
            session=session,
            cache=cache,
            fields={key: asdict(field) for key, field in self.fields.items()},
        )
        members = [None] * len(ranks)
        dist.all_gather_object(members, local, group=group)
        if [member["rank"] for member in members] != ranks:
            raise ValueError(
                "distributed cache membership disagrees with process-group order"
            )
        return json.dumps(members, separators=(",", ":")).encode()
