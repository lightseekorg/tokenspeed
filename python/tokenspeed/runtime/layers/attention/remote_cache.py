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

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


@dataclass(frozen=True, slots=True)
class RemoteCacheAdmission:
    """The prompt end and destination blocks a Decode slot expects to receive."""

    request_id: str
    position: int
    block_tables: Mapping[str, tuple[int, ...]]


def copy_remote_cache_admissions(
    *,
    slots: Sequence[int],
    request_ids: Sequence[str],
    prefill_lengths: Sequence[int],
    block_tables: Mapping[str, np.ndarray],
) -> Mapping[int, RemoteCacheAdmission]:
    """Snapshot the admitted slots in batch order, keeping absolute block indices."""
    admissions = {}
    for row, slot in enumerate(slots):
        if slot < 0 or slot in admissions:
            raise ValueError(f"Invalid or repeated remote cache slot {slot}")
        # Copy only the admitted rows. A backend may retain them until the
        # transfer lands, after the scheduler has released this batch.
        tables = {
            group: tuple(rows[row].tolist()) for group, rows in block_tables.items()
        }
        admissions[slot] = RemoteCacheAdmission(
            request_ids[row], prefill_lengths[row], MappingProxyType(tables)
        )
    return MappingProxyType(admissions)
