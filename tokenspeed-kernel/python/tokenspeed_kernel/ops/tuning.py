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

"""Kernel autotune lifecycle.

Ops that autotune lazily (e.g. the flashinfer trtllm MoE) consult
:func:`autotune_frozen` before starting a new tuning run. The runtime calls
:func:`freeze_autotuning` once engine startup (model load, CUDA-graph capture,
warmup) completes, because an autotune during serving is a hazard twice over:
it blocks the launch thread for the whole profiling run, and under tensor
parallelism ranks that tune at different times miss their next collective and
deadlock. Shapes first seen after the freeze run on each library's fallback
tactics instead -- slower, never stalled.
"""

__all__ = ["autotune_frozen", "freeze_autotuning"]

_frozen = False


def autotune_frozen() -> bool:
    """Whether lazy autotuning is frozen (engine is serving)."""
    return _frozen


def freeze_autotuning() -> None:
    """Disallow any further lazy autotune runs (call once startup completes)."""
    global _frozen
    _frozen = True
