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

"""CuTe tiled materialization at retained source-copy and capacity boundaries."""

import pytest
import torch
from utils import assert_no_cute_compile, is_nvidia

if not is_nvidia():
    pytest.skip("NVIDIA required", allow_module_level=True)
pytest.importorskip("cutlass")
pytest.importorskip("quack")
from tokenspeed_kernel.ops.attention.kda import KdaRetainedHistory
from tokenspeed_kernel.ops.attention.kda._cute_dsl.commit_state import (
    consolidate_history,
)
from tokenspeed_kernel.platform import ArchVersion, current_platform
from tokenspeed_kernel.thirdparty.vllm_recoverssm.kda import reset_retained_history

if not ArchVersion(10, 0) <= current_platform().arch_version <= ArchVersion(10, 3):
    pytest.skip("SM100/103 required", allow_module_level=True)


def _case(batch, capacity, count, *, dtype, distinct, strided):
    torch.manual_seed(50 + count)
    pages, heads, dim = 2 * batch + 3, 2, 128
    state = (torch.randn(pages, heads, dim, dim, device="cuda") * 0.1).to(dtype)
    fields = [
        torch.randn(batch, heads, capacity, dim, device="cuda") * 0.03 for _ in range(3)
    ]
    fields[1] = torch.nn.functional.normalize(fields[1], dim=-1)
    fields[2] = fields[2].sigmoid()
    if strided:
        extended = []
        for field in [state, *fields]:
            view = torch.empty(
                *field.shape[:-1], field.shape[-1] * 2, device="cuda", dtype=field.dtype
            )[..., ::2]
            view.copy_(field)
            extended.append(view)
        state, *fields = extended
    # One strided accepted count per request slot.
    length = torch.zeros(batch, 3, device="cuda", dtype=torch.int32)[:, 1:2]
    length[:] = count
    reads = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32)
    live = reads + batch if distinct else reads.clone()
    starts = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * 4
    slots = torch.arange(batch, device="cuda", dtype=torch.int32)
    history = KdaRetainedHistory(*fields, length, live, slots)
    return state, history, reads, starts


def _materialize(state, history, reads, starts):
    consolidate_history(state, history, reads, starts)
    reset_retained_history(history, reads, starts, page_count=state.shape[0])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("capacity,count", [(8, 0), (8, 4), (8, 5), (32, 28), (32, 29)])
@pytest.mark.parametrize("distinct", [False, True])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("captured", [False, True])
def test_tiled_materialization_writes_only_copy_or_overflow_boundaries(
    dtype, capacity, count, distinct, strided, captured
):
    state, history, reads, starts = _case(
        3, capacity, count, dtype=dtype, distinct=distinct, strided=strided
    )
    saved_state = state.clone()
    saved_fields = [x.clone() for x in (history.correction, history.key, history.decay)]
    saved_length = history.length.clone()
    expected = saved_state.clone()
    expected_length = saved_length.clone()
    if distinct or count + 4 > capacity:
        for slot, (source, target) in enumerate(
            zip(reads.tolist(), history.live_state_indices.tolist())
        ):
            current = saved_state[source].float()
            for token in range(count):
                current = (
                    current * history.decay[slot, :, token, None, :]
                    + history.correction[slot, :, token, :, None]
                    * history.key[slot, :, token, None, :]
                )
            expected[target] = current.to(dtype)
            expected_length[slot] = 0
    _materialize(state, history, reads, starts)
    if captured:
        state.copy_(saved_state)
        history.length.copy_(saved_length)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _materialize(state, history, reads, starts)
        state.copy_(saved_state)
        history.length.copy_(saved_length)
        graph.replay()
    torch.testing.assert_close(
        state,
        expected,
        atol=2e-6 if dtype == torch.float32 else 2e-3,
        rtol=2e-5 if dtype == torch.float32 else 4e-3,
    )
    torch.testing.assert_close(history.length, expected_length, atol=0, rtol=0)
    for actual, saved in zip(
        (history.correction, history.key, history.decay), saved_fields
    ):
        torch.testing.assert_close(actual, saved, atol=0, rtol=0)


def test_rows_outside_the_page_pool_are_skipped_by_both_launches():
    """A source or live page past the pool names no page: consolidation and
    the length reset both leave the row alone, so its records stay counted
    instead of being dropped unfolded."""
    state, history, reads, starts = _case(
        3, 8, 5, dtype=torch.float32, distinct=True, strided=False
    )
    pages = state.shape[0]
    reads[0] = pages
    history.live_state_indices[1] = pages
    saved_state = state.clone()
    saved_length = history.length.clone()
    _materialize(state, history, reads, starts)
    torch.testing.assert_close(history.length[:2], saved_length[:2], atol=0, rtol=0)
    assert history.length[2].item() == 0
    folded = history.live_state_indices[2].item()
    untouched = [page for page in range(pages) if page != folded]
    torch.testing.assert_close(state[untouched], saved_state[untouched], atol=0, rtol=0)


def test_materialization_does_not_trace_new_batch_page_or_slot_counts():
    _materialize(*_case(4, 8, 5, dtype=torch.float32, distinct=True, strided=False))
    with assert_no_cute_compile():
        for batch, strided in ((1, False), (3, True), (7, False)):
            _materialize(
                *_case(batch, 8, 5, dtype=torch.float32, distinct=True, strided=strided)
            )
