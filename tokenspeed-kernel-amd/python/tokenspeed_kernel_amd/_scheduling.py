# MIT License
#
# Copyright (c) 2026 LightSeek Foundation <contact@lightseek.org>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""AMD instruction-scheduling barrier and its external-library dependency."""

import hashlib
from functools import lru_cache
from pathlib import Path

from tokenspeed_kernel_amd._triton import tl

_SCHED_LIBRARY_NAME = "tokenspeed_sched"
_SCHED_SYMBOL = "__tokenspeed_sched_barrier0"
_SCHED_LIBRARY_PATH = str(Path(__file__).with_name("sched_barrier.ll"))


@lru_cache(maxsize=1)
def _scheduler_library_hash() -> str:
    # File contents are immutable within a running process, like JIT source.
    # A constexpr carries their digest into Triton's compiled-kernel cache key.
    return hashlib.sha256(Path(_SCHED_LIBRARY_PATH).read_bytes()).hexdigest()


def sched_barrier_compile_options() -> dict:
    """Return launch options for kernels using :func:`sched_barrier`.

    The kernel must accept an otherwise unused ``SCHED_LIBRARY_HASH`` constexpr
    so edits to the library invalidate its compiled binary. Merge ``extern_libs``
    with any other device libraries required by the caller.
    """
    return {
        "SCHED_LIBRARY_HASH": _scheduler_library_hash(),
        "extern_libs": {_SCHED_LIBRARY_NAME: _SCHED_LIBRARY_PATH},
    }


@tl.core.extern
def sched_barrier(_semantic):
    """Prevent instruction scheduling across this point; no workgroup sync.

    Emits ``llvm.amdgcn.sched.barrier(0)``. Returns an unused int32 value required
    by the elementwise extern interface. Launch with
    :func:`sched_barrier_compile_options` to link the library and key its content.
    """
    return tl.core.extern_elementwise(
        _SCHED_LIBRARY_NAME,
        _SCHED_LIBRARY_PATH,
        [],
        {(): (_SCHED_SYMBOL, tl.int32)},
        is_pure=False,
        _semantic=_semantic,
    )
