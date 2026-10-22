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

"""Zeroed kernel state across eager launches and graph capture."""

from __future__ import annotations

import contextlib
import gc
import weakref

import pytest
import tokenspeed_kernel.workspace as workspace
import torch
from tokenspeed_kernel.workspace import (
    capture_session,
    retain_for_capture,
    zeroed_state,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="graph capture needs a CUDA/HIP device"
)

_DEVICE = torch.device("cuda")


def _bump_and_restore(out: torch.Tensor) -> torch.Tensor:
    """Stand-in for a split-K launch: count an arrival, then reset the state."""
    state = zeroed_state(out.numel(), dtype=torch.int32, device=_DEVICE)
    state.add_(1)
    out.copy_(state)
    state.sub_(1)
    return state


def test_eager_launches_share_one_buffer_per_stream() -> None:
    first = zeroed_state(8, dtype=torch.int32, device=_DEVICE)
    assert zeroed_state(8, dtype=torch.int32, device=_DEVICE) is first
    with torch.cuda.stream(torch.cuda.Stream()):
        other = zeroed_state(8, dtype=torch.int32, device=_DEVICE)
    assert other is not first
    assert torch.all(other == 0)


@pytest.mark.parametrize("inference", [False, True], ids=["grad", "inference"])
@pytest.mark.parametrize("session", [True, False], ids=["session", "no-session"])
def test_capture_zeroes_state_once_or_per_replay(
    session: bool, inference: bool
) -> None:
    # Serving runs the captured forward under inference mode.
    out = torch.empty(4, dtype=torch.int32, device=_DEVICE)
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    mode = torch.inference_mode() if inference else contextlib.nullcontext()
    with torch.cuda.stream(stream):
        scope = capture_session(graph) if session else contextlib.nullcontext()
        with scope, torch.cuda.graph(graph, stream=stream):
            with mode:
                state = _bump_and_restore(out)
    stream.synchronize()

    graph.replay()
    torch.cuda.synchronize()
    assert torch.all(out == 1) and torch.all(state == 0)

    # With a session the graph itself never zeroes the state, so a launch
    # sees whatever the previous one left; without one every replay zeroes.
    with torch.inference_mode():
        state.fill_(5)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.all(out == (6 if session else 1))


def test_session_keeps_captured_state_alive_with_its_owner() -> None:
    out = torch.empty(64, dtype=torch.int32, device=_DEVICE)
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    later = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with capture_session(graph), torch.cuda.graph(graph, stream=stream):
            # The launch drops its state on return; later allocations in the
            # same capture must not reuse that memory.
            _bump_and_restore(out)
            inner = torch.full((64,), 7, dtype=torch.int32, device=_DEVICE)
        # Nor may a later capture into the same pool.
        with torch.cuda.graph(later, pool=graph.pool(), stream=stream):
            outer = torch.full((64,), 9, dtype=torch.int32, device=_DEVICE)
    stream.synchronize()

    for _ in range(3):
        graph.replay()
        later.replay()
    torch.cuda.synchronize()
    assert torch.all(out == 1)
    assert torch.all(inner == 7) and torch.all(outer == 9)


def test_session_keeps_prepared_resources_until_owner_release() -> None:
    state = torch.full((64,), 7, device=_DEVICE)
    state_ref = weakref.ref(state)
    output = torch.empty_like(state)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with capture_session(graph), torch.cuda.graph(graph, stream=stream):
        retain_for_capture(state)
        output.copy_(state)
    del state
    gc.collect()
    assert state_ref() is not None
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, torch.full_like(output, 7))
    del graph
    gc.collect()
    assert state_ref() is None


def test_session_state_never_reuses_memory_freed_earlier_in_the_capture() -> None:
    # The graph pool would hand the state memory a tensor freed earlier in the
    # capture used; every replay rewrites that tensor before the state's
    # launch reads it.
    out = torch.empty(64, dtype=torch.int32, device=_DEVICE)
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with capture_session(graph), torch.cuda.graph(graph, stream=stream):
            scratch = torch.full((64,), 3, dtype=torch.int32, device=_DEVICE)
            del scratch
            _bump_and_restore(out)
    stream.synchronize()

    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.all(out == 1)


def test_session_overflow_zeroes_inside_the_graph(monkeypatch) -> None:
    # A 4 KiB arena; restoring the peak afterwards keeps later sessions'
    # arenas independent of this test.
    monkeypatch.setattr(workspace, "_MIN_ARENA_BYTES", 4096)
    monkeypatch.setattr(workspace._sessions, "peak_bytes", 0)
    fits = torch.empty(64, dtype=torch.int32, device=_DEVICE)
    overflows = torch.empty(2048, dtype=torch.int32, device=_DEVICE)
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with pytest.warns(UserWarning, match="ran out"):
            with capture_session(graph), torch.cuda.graph(graph, stream=stream):
                arena_state = _bump_and_restore(fits)
                fallback_state = _bump_and_restore(overflows)
    stream.synchronize()

    graph.replay()
    torch.cuda.synchronize()
    assert torch.all(fits == 1) and torch.all(overflows == 1)

    # Only the overflow buffer is zeroed by the graph itself.
    arena_state.fill_(5)
    fallback_state.fill_(5)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.all(fits == 6) and torch.all(overflows == 1)
