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

import os

from tokenspeed.runtime.utils.env import envs


def test_nvshmem_ib_traffic_class_unset_by_default():
    with envs.NVSHMEM_IB_TRAFFIC_CLASS.override(None):
        assert envs.NVSHMEM_IB_TRAFFIC_CLASS.get() is None
        assert "NVSHMEM_IB_TRAFFIC_CLASS" not in os.environ


def test_nvshmem_ib_traffic_class_reaches_process_environ():
    with envs.NVSHMEM_IB_TRAFFIC_CLASS.override(100):
        assert envs.NVSHMEM_IB_TRAFFIC_CLASS.is_set()
        assert envs.NVSHMEM_IB_TRAFFIC_CLASS.get() == 100
        # NVSHMEM reads the raw environment at bootstrap; the registry set()
        # must materialize the value there, not only in Python state.
        assert os.environ["NVSHMEM_IB_TRAFFIC_CLASS"] == "100"


def test_nvshmem_ib_traffic_class_reassert_is_idempotent():
    # run_event_loop re-asserts the value in each inference process; doing so
    # must not corrupt or drop the setting.
    with envs.NVSHMEM_IB_TRAFFIC_CLASS.override(100):
        envs.NVSHMEM_IB_TRAFFIC_CLASS.set(envs.NVSHMEM_IB_TRAFFIC_CLASS.get())
        assert envs.NVSHMEM_IB_TRAFFIC_CLASS.get() == 100
        assert os.environ["NVSHMEM_IB_TRAFFIC_CLASS"] == "100"
