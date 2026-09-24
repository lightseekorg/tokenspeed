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

"""Optional vendor imports must not pin the importing caller's frame.

A module that catches a failed optional import and keeps the exception object
keeps its traceback, and a traceback references every frame that was on the
stack at raise time -- including whichever application function happened to
trigger the first import. CPython then keeps that frame's locals alive for the
rest of the process. Anything owned by such a local (a CUDA graph holding an
NCCL collective, for one) is never released, and a later
``destroy_process_group`` waits on it forever.

The probe runs in a fresh interpreter and preloads PyTorch, whose own first
import retains its caller's frame independently of TokenSpeed. The TokenSpeed
modules are still imported for the first time from inside the probing function.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

_OPTIONAL_VENDOR_MODULES = (
    "tokenspeed_kernel.ops.residual.gluon",
    "tokenspeed_kernel.ops.gemm.gluon",
    "tokenspeed_kernel.ops.sampling.gluon",
)

_PROBE = textwrap.dedent("""
    import importlib
    import sys
    import weakref

    import torch


    class Sentinel:
        pass


    def import_from_here(name):
        local = Sentinel()
        alive = weakref.ref(local)
        importlib.import_module(name)
        return alive


    pinned = [name for name in sys.argv[1:] if import_from_here(name)() is not None]
    print(",".join(pinned))
    """)


@pytest.mark.parametrize("module_name", _OPTIONAL_VENDOR_MODULES)
def test_first_import_releases_the_callers_frame(module_name: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, module_name],
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    pinned = result.stdout.strip()
    assert pinned == "", (
        f"importing {pinned} kept the calling frame alive: a caught import "
        "error is being stored with its traceback"
    )
