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

"""Reporting Triton and CuTe DSL JIT compilations on the serving path.

A kernel compiles on the forward thread the first time it sees a new
compile-cache key, and the round waits for it (100+ ms per kernel). Startup
compiles on purpose -- weight processing, tuning, graph capture -- so the
scheduler process installs ``tokenspeed_kernel.compile_monitor`` before it
builds anything and marks the end of startup as its last step before entering
the round loop, next to the data-plane sync debug. From then on every
compilation is logged. A compile-time kernel parameter that keeps taking new
values (the signature of a per-batch value passed as ``tl.constexpr`` or as a
static CuTe DSL dimension), a call site that keeps recompiling specializations
a kernel already has, and a direct ``@cute.jit`` call, which traces on every
call, are named, or raise under ``TOKENSPEED_JIT_COMPILE_CHECK=error``.
"""

from __future__ import annotations

from tokenspeed_kernel.compile_monitor import (
    compile_stats,
    install_compile_monitor,
    mark_serving,
)
from tokenspeed_kernel.cute_compile_cache import (
    cute_compile_cache_stats,
    install_cute_compile_cache,
)

from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.env import envs

logger = get_colorful_logger(__name__)


def install_jit_compile_check() -> None:
    """Cache CuTe DSL binaries on disk and record JIT compilations.

    The CuTe DSL compile cache is installed whatever the check mode; its
    switches are cutlass-dsl's own (``CUTE_DSL_DISABLE_FILE_CACHING``).
    """
    install_cute_compile_cache()
    mode = envs.TOKENSPEED_JIT_COMPILE_CHECK.get()
    if mode == "off":
        return
    if mode not in ("warn", "error"):
        raise ValueError(
            f"TOKENSPEED_JIT_COMPILE_CHECK must be warn, error or off, got {mode!r}"
        )
    install_compile_monitor(on_violation=mode)


def mark_jit_compile_serving() -> None:
    """End startup: from here on a compilation is a serving stall."""
    mark_serving()
    cached = cute_compile_cache_stats()
    if cached is not None:
        logger.info(
            f"CuTe DSL compile cache: {cached.hits:d} of {cached.keyed:d} "
            "startup compilations loaded from disk"
        )
    stats = compile_stats()
    if stats is None:
        return
    logger.info(
        f"Startup JIT-compiled {stats.startup_compiles:d} kernels in "
        f"{stats.startup_seconds:.1f} s; compilations while serving are reported "
        f"(TOKENSPEED_JIT_COMPILE_CHECK={envs.TOKENSPEED_JIT_COMPILE_CHECK.get()!s})"
    )
