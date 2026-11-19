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

"""Offline-tuned BLAS solutions (TunableOp) for dense GEMMs on ROCm.

The runtime entry point. The tuned files (one per GPU, in-repo, in
``_hipblaslt/configs/``), loading, the stack check and the tuning tool live in
``tokenspeed_kernel.ops.gemm._hipblaslt.tuning``; this module makes the calls
no-ops on non-ROCm builds.

With ``TOKENSPEED_HIPBLASLT_AUTOTUNE=1`` (off by default), serving also tunes
the GEMMs of its warmup decode steps that the loaded file does not list,
before graph capture, and keeps the picks in the process
(``hipblaslt_online_tuning``).
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
from typing import ContextManager

import torch
from tokenspeed_kernel.ops.gemm._hipblaslt import tuning

__all__ = [
    "AUTOTUNE_ENV",
    "hipblaslt_autotune_enabled",
    "hipblaslt_online_tuning",
    "load_hipblaslt_tuned_solutions",
]

logger = logging.getLogger(__name__)

AUTOTUNE_ENV = "TOKENSPEED_HIPBLASLT_AUTOTUNE"


def load_hipblaslt_tuned_solutions(path: str | None = None) -> bool:
    """Replay the BLAS solutions (hipBLASLt or rocBLAS) in a TunableOp results file.

    See ``tokenspeed_kernel.ops.gemm._hipblaslt.tuning.load_hipblaslt_tuned_solutions``.

    Args:
        path: TunableOp results CSV made on the same software stack and GPU
            architecture. When omitted, the in-repo file of the current GPU
            is loaded, at most once per process; on a non-ROCm build or a
            GPU without a file this is a no-op.

    Returns:
        Whether the file was loaded. False on a non-ROCm build or when the
        implementation rejects the file.
    """
    if path is None:
        return _load_default()
    if torch.version.hip is None:
        logger.warning(f"Ignoring tuned BLAS solutions {path}: not a ROCm build")
        return False
    return tuning.load_hipblaslt_tuned_solutions(path)


@functools.cache
def _load_default() -> bool:
    # Cached: every executor of a process calls this, and the outcome
    # (including a skipped file's log line) should appear once.
    if torch.version.hip is None:
        return False
    return tuning.load_hipblaslt_tuned_solutions()


def hipblaslt_autotune_enabled() -> bool:
    """Whether ``TOKENSPEED_HIPBLASLT_AUTOTUNE=1`` on a ROCm build."""
    return torch.version.hip is not None and os.environ.get(AUTOTUNE_ENV, "0") == "1"


def hipblaslt_online_tuning() -> ContextManager[None]:
    """Tune, inside the block, the GEMMs no loaded file lists (ROCm only).

    See ``tokenspeed_kernel.ops.gemm._hipblaslt.tuning.hipblaslt_online_tuning``.
    Run it outside graph capture. A no-op context off ROCm.
    """
    if torch.version.hip is None:
        return contextlib.nullcontext()
    return tuning.hipblaslt_online_tuning()
