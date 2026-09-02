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

"""The CUPTI graph-capture warm-up is opt-in, on every vendor.

``_init_for_cuda_graphs()`` opens an empty profiler session so that CUPTI is
loaded before any CUDA graph is captured. On ROCm there is no CUPTI: torch
profiles through roctracer, and that empty session leaves activity collection
permanently dead for the life of the process. Every later ``/start_profile``
then returns a trace containing ``cpu_op`` entries and zero ``"cat": "kernel"``
entries, on every rank, in eager and graph mode alike -- silently, since the
request still reports success.

Measured on 8x gfx950 serving Kimi-K3: 0 GPU events on all 8 ranks with the
warm-up, 62k kernel events in the same decode trace without it. CUPTI behaves
the same way -- measured on CUDA 13.0 / torch 2.13, 2xGB300 TP8: 0 kernel
events with the warm-up, 933k across 340 graph replays without it, and no
launch failure from attaching after capture. So the warm-up is off unless
``TOKENSPEED_CUPTI_GRAPH_WARMUP`` asks for it.

CPU-only: no CUDA context, no profiler session, just the gate decision.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest import mock

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

from tokenspeed.runtime.engine import event_loop  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")


def _run(*, is_amd: bool, cuda_available: bool = True, enabled: bool = True) -> bool:
    """Return whether the CUPTI warm-up was invoked."""
    called = False

    def _fake_init():
        nonlocal called
        called = True

    fake_utils = SimpleNamespace(_init_for_cuda_graphs=_fake_init)
    with (
        mock.patch.object(
            event_loop.torch.cuda, "is_available", return_value=cuda_available
        ),
        mock.patch(
            "tokenspeed_kernel.platform.current_platform",
            return_value=SimpleNamespace(is_amd=is_amd),
        ),
        mock.patch.dict(sys.modules, {"torch.profiler._utils": fake_utils}),
        mock.patch.object(
            event_loop.envs.TOKENSPEED_CUPTI_GRAPH_WARMUP, "get", return_value=enabled
        ),
    ):
        event_loop.maybe_warm_cupti_for_graph_capture()
    return called


def test_the_warmup_is_off_by_default():
    # It kills activity collection for the life of the process, which is a
    # worse deal than the capture-order hazard it guards.
    assert _run(is_amd=False, enabled=False) is False
    assert _run(is_amd=True, enabled=False) is False


def test_amd_skips_the_cupti_warmup():
    """The regression: running this on ROCm kills roctracer capture."""
    assert _run(is_amd=True) is False


def test_nvidia_warms_cupti_when_asked():
    """The warm-up must be preserved where it is needed."""
    assert _run(is_amd=False) is True


def test_no_cuda_is_a_noop():
    assert _run(is_amd=False, cuda_available=False) is False
