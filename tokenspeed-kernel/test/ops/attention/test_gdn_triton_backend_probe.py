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

"""A failed Triton backend probe must report why it failed.

The GDN utils module probes the backend at import time. The probe can fail for
reasons unrelated to the platform, such as an unwritable ``TRITON_CACHE_DIR``
while the driver compiles its helper module; the error must carry that cause.
"""

from __future__ import annotations

import importlib
import sys

import pytest
import tokenspeed_kernel.ops.attention.gdn._triton  # noqa: F401  (parent package)
from tokenspeed_kernel._triton import triton

_MODULE = "tokenspeed_kernel.ops.attention.gdn._triton.utils"


@pytest.fixture
def fresh_probe(monkeypatch):
    """Re-import the module with ``driver`` active, restoring state after."""
    config = triton.runtime.driver
    previous = config.active
    monkeypatch.delitem(sys.modules, _MODULE, raising=False)

    def _import_with(driver):
        config.set_active(driver)
        return importlib.import_module(_MODULE)

    yield _import_with
    config.set_active(previous)


class _UnwritableCacheDriver:
    def get_current_target(self):
        raise PermissionError(13, "Permission denied", "/jit-cache")


class _InterruptedDriver:
    def get_current_target(self):
        raise KeyboardInterrupt


def test_import_time_probe_failure_chains_the_underlying_error(fresh_probe):
    with pytest.raises(RuntimeError) as excinfo:
        fresh_probe(_UnwritableCacheDriver())

    assert isinstance(excinfo.value.__cause__, PermissionError)
    assert "PermissionError" in str(excinfo.value)
    assert "/jit-cache" in str(excinfo.value)


def test_probe_does_not_swallow_keyboard_interrupt(fresh_probe):
    with pytest.raises(KeyboardInterrupt):
        fresh_probe(_InterruptedDriver())
