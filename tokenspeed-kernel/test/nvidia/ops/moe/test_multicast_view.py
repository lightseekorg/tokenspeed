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

"""The borrowed-address view, which nothing else in the suite constructs.

Every test of the wide producer mocks this module out, so its one real
consumer -- a view held in a class-level pool for the life of the process --
is exercised nowhere. That is the shape that broke: the view is deallocated
inside interpreter shutdown, and what runs there is the only part of DLPack
this module supplies.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)

# Held to exit, exactly as `_MulticastVaGemm` holds it through the slot pool.
_HELD = """
import torch
from tokenspeed_kernel.ops.moe.multicast_view import bf16_tensor_on_pointer

base = torch.zeros(4, 16, dtype=torch.bfloat16, device="cuda")
HELD = bf16_tensor_on_pointer(base.data_ptr(), (4, 3), (16, 1), 0)
torch.mm(
    torch.ones(4, 2, dtype=torch.bfloat16, device="cuda"),
    torch.full((2, 3), 2.0, dtype=torch.bfloat16, device="cuda"),
    out=HELD,
)
torch.cuda.synchronize()
assert bool((base[:, :3] == 4).all()), "view did not write through"
assert bool((base[:, 3:] == 0).all()), "view wrote past its columns"
print("ok")
"""


def _run(program: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=300
    )


@pytest.mark.skipif(
    not current_platform().is_nvidia, reason="the multicast view is NVIDIA-only"
)
def test_a_view_held_to_interpreter_exit_does_not_crash_the_process() -> None:
    """The deleter must be null, not a no-op callback.

    A ctypes callback stops being callable once ``Py_FinalizeEx`` has begun
    clearing module dicts, which is precisely when a process-lifetime view is
    deallocated. Every rank then exits 139 with a core dump on a clean
    shutdown, which orchestrators read as a crash and which hides real ones.

    The program also asserts that the view still aliases the address it
    borrowed, so a null deleter that cost the view its write-through would
    fail here rather than in a separate test that could only repeat this one.
    """
    done = _run(_HELD)
    assert done.returncode == 0, f"exit {done.returncode}\n{done.stderr[-2000:]}"
    assert "ok" in done.stdout
