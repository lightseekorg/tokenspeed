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

"""Integer-only planning for page-boundary ShortConv checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ShortConvCheckpointWrite:
    """One completed boundary and the rows that form its saved window.

    Exactly one of ``packed_rows[i]`` and ``prior_state_rows[i]`` is set. The
    latter is needed when a chunk starts fewer than ``state_rows`` tokens
    before the boundary.
    """

    request_index: int
    page_slot: int
    packed_rows: tuple[int | None, ...]
    prior_state_rows: tuple[int | None, ...]


def plan_shortconv_checkpoint_writes(
    *,
    page_size: int,
    state_rows: int,
    seq_lens_before: Sequence[int],
    seq_lens_after: Sequence[int],
    query_start_loc: Sequence[int],
) -> tuple[ShortConvCheckpointWrite, ...]:
    """Plan checkpoint writes for every completed ``page_size`` boundary.

    The function only resolves row provenance. It neither owns tensors nor
    mutates cache state, so the same plan can be materialized on any device.

    Args:
        page_size: Shared logical CacheBlock size ``P``.
        state_rows: Number of pre-token rows required by ShortConv (``W - 1``).
        seq_lens_before: Per-request sequence lengths before this packed chunk.
        seq_lens_after: Per-request sequence lengths after this packed chunk.
        query_start_loc: Packed chunk offsets, with one trailing sentinel.

    Returns:
        Checkpoint writes ordered first by request and then by boundary.

    Raises:
        ValueError: If geometry or varlen metadata is inconsistent.
    """
    if page_size <= 0:
        raise ValueError("page_size must be > 0")
    if state_rows <= 0:
        raise ValueError("state_rows must be > 0")
    if len(seq_lens_before) != len(seq_lens_after):
        raise ValueError("length arrays must have the same size")
    if len(query_start_loc) != len(seq_lens_before) + 1:
        raise ValueError("query_start_loc must have one trailing sentinel")
    if not query_start_loc or query_start_loc[0] != 0:
        raise ValueError("query_start_loc must start at zero")

    writes = []
    for request_index, (before, after) in enumerate(
        zip(seq_lens_before, seq_lens_after)
    ):
        before = int(before)
        after = int(after)
        chunk_begin = int(query_start_loc[request_index])
        chunk_end = int(query_start_loc[request_index + 1])
        if before < 0 or after < before:
            raise ValueError("length values must satisfy 0 <= before <= after")
        if chunk_begin < 0 or chunk_end < chunk_begin:
            raise ValueError("query_start_loc must be nondecreasing")
        if chunk_end - chunk_begin != after - before:
            raise ValueError("query_start_loc does not match request lengths")

        boundary = (before // page_size + 1) * page_size
        while boundary <= after:
            packed_rows = []
            prior_state_rows = []
            for absolute_row in range(boundary - state_rows, boundary):
                relative_row = absolute_row - before
                if relative_row >= 0:
                    packed_rows.append(chunk_begin + relative_row)
                    prior_state_rows.append(None)
                else:
                    packed_rows.append(None)
                    prior_state_rows.append(state_rows + relative_row)
            writes.append(
                ShortConvCheckpointWrite(
                    request_index=request_index,
                    page_slot=boundary // page_size - 1,
                    packed_rows=tuple(packed_rows),
                    prior_state_rows=tuple(prior_state_rows),
                )
            )
            boundary += page_size

    return tuple(writes)
