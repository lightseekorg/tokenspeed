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

"""The event loop's shared L2 hooks without a pool: every call is a no-op and a
pool command fails loudly."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.engine.shared_l2_hooks import SharedL2Hooks  # noqa: E402


class _NeverReady:
    """A forward result that never lands."""

    def ready(self) -> bool:
        return False


def test_without_a_pool_the_hooks_do_nothing():
    hooks = SharedL2Hooks(None, None)
    assert not hooks.active and hooks.scheduler_config is None
    assert hooks.poll_ready_events() == []
    hooks.lookup()
    hooks.submit([])
    hooks.wait_for_forward(_NeverReady())
    hooks.close()


def test_a_pool_command_without_a_pool_fails_loudly():
    with pytest.raises(RuntimeError, match="shared L2 client"):
        SharedL2Hooks(None, None).submit([object()])
