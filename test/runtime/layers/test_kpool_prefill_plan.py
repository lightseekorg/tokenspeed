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

"""Behavior of the KPool prefill plan, seen through what it drives.

The plan is built at the first sparse layer, behind the forward's queued GPU
work, so a synchronizing upload would stall the forward thread there. Triton
keys pointer arguments on 16-byte alignment, so plan arrays whose alignment
moved with the batch shape would recompile the kernels reading them.
"""

from __future__ import annotations

import os
import sys

import pytest
import tokenspeed_kernel.compile_monitor as compile_monitor
import torch
from tokenspeed_kernel.ops.attention.kpool import kpool_prefill_tail_write

from tokenspeed.runtime.layers.attention.kpool import build_kpool_prefill_plan

# CI registration is parsed via AST; standalone execution needs test/ on sys.path.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="runtime-1gpu")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

KPOOL = 4
TAIL_ROWS = 6
HEAD_DIM = 128
# (prefix lengths, extend lengths) per batch: unaligned prefixes, chunks that
# complete no pool, and batch sizes that move every array's upload offset.
BATCHES = (
    ([0, 0], [7, 9]),
    ([1, 2, 3], [5, 6, 7]),
    ([0], [3]),
    ([2, 5, 1, 0, 3], [9, 2, 7, 1, 6]),
    ([3, 1], [13, 22]),
    ([0, 1, 2, 3, 0, 1, 2], [1, 2, 3, 5, 6, 7, 9]),
)


def _plan(starts, lengths, token_capacity=None):
    table = torch.arange(64, dtype=torch.int32, device="cuda").view(8, 8)
    return build_kpool_prefill_plan(
        prefix_lens_cpu=torch.tensor(starts),
        extend_lens_cpu=torch.tensor(lengths),
        index_block_table=table[: len(starts)],
        request_slots=torch.arange(len(starts), device="cuda"),
        kpool=KPOOL,
        index_rows_per_page=table.shape[1],
        tail_rows_per_request=TAIL_ROWS,
        token_capacity=token_capacity,
    )


def _tails() -> tuple[torch.Tensor, torch.Tensor]:
    tail_k = torch.zeros(8, TAIL_ROWS, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    return tail_k, torch.zeros_like(tail_k)


def _write_tails(tail_k, tail_gate, starts, lengths) -> torch.Tensor:
    """Write one prefill chunk per request into the tails; return its keys."""
    write = _plan(starts, lengths).write
    key = torch.randn(sum(lengths), HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    kpool_prefill_tail_write(
        key,
        key,
        tail_k,
        tail_gate,
        write.tail_chunk_src,
        write.tail_slots,
        write.tail_dst_positions,
        write.tail_write_counts,
        pool_size=KPOOL,
    )
    return key


def test_prefill_leaves_the_incomplete_pool_in_the_tail() -> None:
    """Tokens past a request's last complete pool land in its tail ring."""
    tail_k, tail_gate = _tails()
    for starts, lengths in BATCHES:
        key = _write_tails(tail_k, tail_gate, starts, lengths)
        offset = 0
        for slot, (start, length) in enumerate(zip(starts, lengths)):
            end = start + length
            for position in range(max(start, end // KPOOL * KPOOL), end):
                assert torch.equal(
                    tail_k[slot, position % TAIL_ROWS],
                    key[offset + position - start],
                ), (starts, lengths, slot, position)
            offset += length


def test_plan_rejects_capacity_below_the_tokens() -> None:
    with pytest.raises(ValueError, match="capacity is smaller"):
        _plan([0], [9], token_capacity=8)


def test_plan_builds_without_synchronizing() -> None:
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        _plan([6, 3, 0], [21, 9, 5], token_capacity=48)
    finally:
        torch.cuda.set_sync_debug_mode("default")


@pytest.fixture
def serving(monkeypatch):
    # Marking serving closes the process-wide compile switch; reopen it afterwards.
    monkeypatch.setattr(compile_monitor, "_serving", False)
    compile_monitor.install_compile_monitor("warn")
    yield
    compile_monitor.uninstall_compile_monitor()


def test_batch_shapes_do_not_recompile_the_tail_write(serving) -> None:
    """Plans of different batch shapes reuse one tail-write binary."""
    tail_k, tail_gate = _tails()
    first, *rest = BATCHES
    _write_tails(tail_k, tail_gate, *first)
    compile_monitor.mark_serving()
    for starts, lengths in rest:
        _write_tails(tail_k, tail_gate, starts, lengths)
    torch.cuda.synchronize()
    assert compile_monitor.compile_stats().serving_compiles == 0
