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

"""Import-light CUDA graph routing helpers.

These helpers are intentionally separate from ``cuda_graph_wrapper`` so code
that only needs routing decisions does not import the full CUDA graph runtime.
"""

from __future__ import annotations

import bisect
from collections.abc import Collection, Sequence

__all__ = ["global_graph_bs", "select_cuda_graph_route"]


def global_graph_bs(
    *,
    dp_size: int,
    global_num_tokens: list[int] | None,
    max_tokens_per_req: int,
) -> int | None:
    if dp_size <= 1 or global_num_tokens is None:
        return None
    max_num_tokens = max(global_num_tokens)
    return (max_num_tokens + max_tokens_per_req - 1) // max_tokens_per_req


def select_cuda_graph_route(
    *,
    bs: int,
    forward_mode,
    disable: bool,
    dp_size: int,
    all_decode_or_idle: bool,
    global_num_tokens: list[int] | None,
    max_tokens_per_req: int,
    disable_padding: bool,
    capture_bs: Sequence[int],
    max_bs: int,
    available_graph_bs: Collection[int],
) -> tuple[bool, int]:
    """Return whether to use a captured graph and the effective batch size."""

    use_graph = _can_use_graph(
        bs=bs,
        forward_mode=forward_mode,
        disable=disable,
        dp_size=dp_size,
        all_decode_or_idle=all_decode_or_idle,
        global_num_tokens=global_num_tokens,
        max_tokens_per_req=max_tokens_per_req,
        disable_padding=disable_padding,
        max_bs=max_bs,
        available_graph_bs=available_graph_bs,
    )
    if not use_graph:
        return False, bs
    return True, _padded_bs(
        bs=bs,
        dp_size=dp_size,
        global_num_tokens=global_num_tokens,
        max_tokens_per_req=max_tokens_per_req,
        capture_bs=capture_bs,
    )


def _can_use_graph(
    *,
    bs: int,
    forward_mode,
    disable: bool,
    dp_size: int,
    all_decode_or_idle: bool,
    global_num_tokens: list[int] | None,
    max_tokens_per_req: int,
    disable_padding: bool,
    max_bs: int,
    available_graph_bs: Collection[int],
) -> bool:
    if disable:
        return False
    if not (forward_mode.is_decode() or forward_mode.is_target_verify()):
        return False
    if dp_size > 1:
        if not all_decode_or_idle:
            return False
        graph_bs = global_graph_bs(
            dp_size=dp_size,
            global_num_tokens=global_num_tokens,
            max_tokens_per_req=max_tokens_per_req,
        )
        if graph_bs is None or graph_bs == 0:
            return False
        if disable_padding:
            return graph_bs in available_graph_bs
        return graph_bs <= max_bs
    if disable_padding:
        return bs in available_graph_bs
    return bs <= max_bs


def _padded_bs(
    *,
    bs: int,
    dp_size: int,
    global_num_tokens: list[int] | None,
    max_tokens_per_req: int,
    capture_bs: Sequence[int],
) -> int:
    graph_bs = global_graph_bs(
        dp_size=dp_size,
        global_num_tokens=global_num_tokens,
        max_tokens_per_req=max_tokens_per_req,
    )
    target_bs = graph_bs if graph_bs is not None else bs
    index = bisect.bisect_left(capture_bs, target_bs)
    return capture_bs[index]
