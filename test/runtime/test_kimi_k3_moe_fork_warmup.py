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

"""``KimiLinearMoE`` must exercise its auxiliary stream during graph warmup.

``_capture_one`` runs several warmup forwards on the capture stream before
recording, and documents that "capture-only auxiliary branches use this graph
phase to warm their own streams serially". Honouring that contract requires
gating the stream fork on the graph *phase* and gating only the overlap on
capture mode.

Gating ``enable`` on capture mode instead left the auxiliary stream untouched
until capture itself. The first hipBLASLt call on that stream then performed its
lazy handle/workspace setup inside the capturing stream, which HIP rejects with
"operation not permitted when stream is capturing" (900) on every rank, and
startup deadlocked with the GPUs idle. It reproduced on gfx950 at TP8/EP1 --
where the tensor-parallel MoE places a projection on the forked branch -- while
TP8/EP8 captured cleanly, so the config coverage matters as much as the flag.

The TP8/EP1 CI jobs that exercise the real failure are manual-trigger only, so
this is the only per-commit signal guarding the contract. It is a deliberately
narrow one: it pins the enable/overlap decision at the call site, not the effect
on the auxiliary stream, so it would not catch a change to StreamFork's own
semantics. Run the EP1 perf job for that.

CPU-only: no real streams or capture, just the enable/overlap decision.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest import mock

import torch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.models.kimi_k3 import KimiLinearMoE

register_cuda_ci(est_time=5, suite="runtime-1gpu")


class _SpyFork:
    """Records the ``scope()`` arguments and behaves like an inactive fork."""

    def __init__(self) -> None:
        self.calls: list[dict[str, bool]] = []
        self._active = False

    def scope(self, *, enable: bool, overlap: bool = True):
        self.calls.append({"enable": enable, "overlap": overlap})
        # Mirror StreamFork: scope() yields the fork itself, and without an aux
        # stream it stays inactive, so branch() is a passthrough and the fake
        # collaborators below run inline.
        return _NullCtx(self)

    def branch(self):
        return _NullCtx(None)


class _NullCtx:
    def __init__(self, value):
        self._value = value

    def __enter__(self):
        return self._value

    def __exit__(self, *exc):
        return False


def _make_moe(fork: _SpyFork) -> SimpleNamespace:
    """Minimal stand-in exposing only what the fork path of forward() touches."""
    hidden = torch.zeros(2, 4)

    plan = SimpleNamespace(
        lane=None,
        split_shared_rs=False,
        routed_in_fork=False,
        defer_finalize=False,
    )
    comm = SimpleNamespace(
        # Absorb keyword axes so the stub does not pin plan's signature.
        plan=lambda num_tokens, hs, **_: plan,
        run=lambda *a, **k: hidden,
        reduce_scatter_shared=lambda x: x,
        reduce_project_routed=lambda x: x,
    )
    return SimpleNamespace(
        _gather_dp_tokens_for_moe=False,
        native_latent_moe=None,
        stream_fork=fork,
        _topk_ready=None,
        routed_hidden=4,
        comm=comm,
        # Stand in for TopKOutputFormat so the fake does not have to track the
        # enum; only is_standard() is consulted on this path.
        _routing_output_format=lambda ctx: SimpleNamespace(is_standard=lambda: True),
        gate=lambda hs: torch.zeros(2, 2),
        topk=lambda hs, logits, output_format=None: (
            torch.zeros(2, 1),
            torch.zeros(2, 1),
        ),
        shared_experts=lambda hs, down_out=None: hs,
        routed_expert_down_proj=lambda hs: (hs, None),
        experts=SimpleNamespace(_situ_output_buffer=None),
        _routed_experts=lambda *a, **k: hidden,
    )


def _run(*, graph_phase: bool, capture_mode: bool) -> dict[str, bool]:
    fork = _SpyFork()
    moe = _make_moe(fork)
    with (
        mock.patch(
            "tokenspeed.runtime.models.kimi_k3.get_is_cuda_graph_phase",
            return_value=graph_phase,
        ),
        mock.patch(
            "tokenspeed.runtime.models.kimi_k3.get_is_capture_mode",
            return_value=capture_mode,
        ),
    ):
        KimiLinearMoE.forward(
            moe,
            torch.zeros(2, 4),
            torch.zeros(2, 4),
            num_global_tokens=2,
            max_num_tokens_per_gpu=2,
        )
    assert len(fork.calls) == 1
    return fork.calls[0]


def test_warmup_forward_activates_the_auxiliary_stream():
    """Warmup (graph phase, not yet capturing) must still use the aux stream.

    This is the regression: with ``enable`` gated on capture mode the warmup
    forwards never touched the aux stream, so hipBLASLt initialized inside the
    capturing stream and capture died with HIP error 900.
    """
    call = _run(graph_phase=True, capture_mode=False)
    assert call["enable"] is True
    # Serial during warmup: the point is to initialize the stream, not to
    # overlap, and overlapping outside capture would race the main stream.
    assert call["overlap"] is False


def test_capture_forward_overlaps():
    call = _run(graph_phase=True, capture_mode=True)
    assert call["enable"] is True
    assert call["overlap"] is True


def test_eager_serving_leaves_the_fork_disabled():
    """Outside the graph phase behaviour is unchanged: no fork, no aux stream."""
    call = _run(graph_phase=False, capture_mode=False)
    assert call["enable"] is False
