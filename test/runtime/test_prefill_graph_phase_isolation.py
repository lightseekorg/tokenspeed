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

"""Breakable-prefill graph state must not impersonate a full CUDA graph.

A breakable graph ends its current capture segment before running an eager
attention break.  Full-graph stream forks may remain open across that call and
therefore cannot be enabled by the breakable-prefill phase.  These CPU tests
pin both the standalone context-manager lifecycle and ``PrefillGraph.capture``'s
success/failure cleanup without exercising CUDA.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.execution.forward_step import (
    get_is_capture_mode,
    get_is_cuda_graph_phase,
    get_is_prefill_graph_phase,
    prefill_graph_phase,
)
from tokenspeed.runtime.execution.memory_delta import NULL_MEMORY_DELTA_OBSERVER
from tokenspeed.runtime.execution.prefill_graph import PrefillGraph

register_cuda_ci(est_time=5, suite="runtime-1gpu")


def _bare_prefill_graph(callback, narrowing: bool) -> SimpleNamespace:
    return SimpleNamespace(
        disable=False,
        capture_buckets=[4],
        attn_backend=SimpleNamespace(init_prefill_graph_state=lambda **_: None),
        config=SimpleNamespace(max_num_seqs=4, data_parallel_size=1),
        _embed_tokens=SimpleNamespace(weight=torch.zeros(2, 8, dtype=torch.float32)),
        _input_embeds_buf=None,
        _captures={},
        _encoders={},
        _decoders={},
        _narrowing=object() if narrowing else None,
        _capture_all_buckets=callback,
        _capture_decoders=callback,
    )


def _assert_prefill_only_state() -> None:
    assert get_is_prefill_graph_phase()
    assert not get_is_cuda_graph_phase()
    assert not get_is_capture_mode()


def test_prefill_graph_phase_is_nested_and_exception_safe() -> None:
    assert not get_is_prefill_graph_phase()
    _assert_generic_graph_state_is_clear()

    with pytest.raises(RuntimeError, match="sentinel"):
        with prefill_graph_phase():
            _assert_prefill_only_state()
            with prefill_graph_phase():
                _assert_prefill_only_state()
            _assert_prefill_only_state()
            raise RuntimeError("sentinel")

    assert not get_is_prefill_graph_phase()
    _assert_generic_graph_state_is_clear()


def _assert_generic_graph_state_is_clear() -> None:
    assert not get_is_cuda_graph_phase()
    assert not get_is_capture_mode()


@pytest.mark.parametrize("narrowing", [False, True])
def test_prefill_capture_publishes_only_prefill_phase_and_restores_it(narrowing) -> None:
    observations: list[tuple[bool, bool, bool]] = []

    def observe(_decode_wrapper, entries, observer) -> None:
        assert entries == 2
        assert observer is NULL_MEMORY_DELTA_OBSERVER
        observations.append(
            (
                get_is_prefill_graph_phase(),
                get_is_cuda_graph_phase(),
                get_is_capture_mode(),
            )
        )

    PrefillGraph.capture(
        _bare_prefill_graph(observe, narrowing),
        None,
        entries=2,
        observer=NULL_MEMORY_DELTA_OBSERVER,
    )

    assert observations == [(True, False, False)] * (2 if narrowing else 1)
    assert not get_is_prefill_graph_phase()
    _assert_generic_graph_state_is_clear()


@pytest.mark.parametrize("fail_on_call", [1, 2])
def test_prefill_capture_failure_restores_prefill_phase(fail_on_call) -> None:
    cause = RuntimeError("capture failed")
    calls = 0

    def fail(_decode_wrapper, entries, observer) -> None:
        nonlocal calls
        _assert_prefill_only_state()
        calls += 1
        if calls == fail_on_call:
            raise cause

    with pytest.raises(RuntimeError) as caught:
        PrefillGraph.capture(
            _bare_prefill_graph(fail, True),
            None,
            entries=None,
            observer=NULL_MEMORY_DELTA_OBSERVER,
        )

    assert caught.value is cause
    assert not get_is_prefill_graph_phase()
    _assert_generic_graph_state_is_clear()
