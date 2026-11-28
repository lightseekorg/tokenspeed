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

The runtime entry point. The committed tuned files (in ``_hipblaslt/configs/``:
one architecture file per GPU architecture, and device files that add rows
for one GPU model), loading, the stack checks and the tuning tool live in
``tokenspeed_kernel.ops.gemm._hipblaslt.tuning``; this module makes the calls
no-ops on non-ROCm builds.

With ``TOKENSPEED_HIPBLASLT_AUTOTUNE=1``, or ``TOKENSPEED_KERNEL_AUTOTUNE=1``
while ``TOKENSPEED_HIPBLASLT_AUTOTUNE`` is unset (off by default), serving
also tunes the GEMMs of its warmup decode steps that the loaded files do not
list, before graph capture (``hipblaslt_online_tuning``), and saves the picks
in ``hipblaslt_online_cache_path()`` for later starts on the same stack
(``hipblaslt_online_tuning_cache``).
"""

from __future__ import annotations

import contextlib
import functools
import logging
from typing import ContextManager

import torch
from tokenspeed_kernel.ops.gemm._hipblaslt import tuning
from tokenspeed_kernel.selection import autotune_cache_file, autotune_switch

__all__ = [
    "AUTOTUNE_ENV",
    "hipblaslt_autotune_enabled",
    "hipblaslt_online_cache_path",
    "hipblaslt_online_tuning",
    "hipblaslt_online_tuning_cache",
    "load_hipblaslt_tuned_solutions",
]

logger = logging.getLogger(__name__)

AUTOTUNE_ENV = "TOKENSPEED_HIPBLASLT_AUTOTUNE"


def load_hipblaslt_tuned_solutions(path: str | None = None) -> bool:
    """Replay the BLAS solutions (hipBLASLt or rocBLAS) in TunableOp results files.

    See ``tokenspeed_kernel.ops.gemm._hipblaslt.tuning.load_hipblaslt_tuned_solutions``.

    Args:
        path: TunableOp results CSV made on the same software stack and GPU
            architecture. When omitted, the committed files of the current
            GPU (its architecture file, then its device file) are loaded, at
            most once per process; on a non-ROCm build or a GPU without a
            file this is a no-op.

    Returns:
        Whether any file was loaded. False on a non-ROCm build or when the
        implementation rejects every file.
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
    """Whether online BLAS tuning is on: ROCm and ``autotune_switch(AUTOTUNE_ENV)``.

    ``TOKENSPEED_HIPBLASLT_AUTOTUNE`` decides when set; unset, it follows
    ``TOKENSPEED_KERNEL_AUTOTUNE``.
    """
    return torch.version.hip is not None and autotune_switch(AUTOTUNE_ENV)


def hipblaslt_online_cache_path() -> str:
    """Where online tuning keeps its picks between starts.

    ``<hash>.hipblaslt.csv`` next to the kernel autotune cache (under
    ``$TOKENSPEED_KERNEL_AUTOTUNE_CACHE_DIR``), named after the device and
    toolchain; TunableOp's stamp in the file also covers the BLAS builds.
    """
    return str(autotune_cache_file(".hipblaslt.csv"))


def hipblaslt_online_tuning() -> ContextManager[None]:
    """Tune, inside the block, the GEMMs no loaded file lists (ROCm only).

    See ``tokenspeed_kernel.ops.gemm._hipblaslt.tuning.hipblaslt_online_tuning``.
    Run it outside graph capture. A no-op context off ROCm.
    """
    if torch.version.hip is None:
        return contextlib.nullcontext()
    return tuning.hipblaslt_online_tuning()


def hipblaslt_online_tuning_cache(path: str, *, write: bool) -> ContextManager[None]:
    """Replay and save the picks of the online tuning blocks inside (ROCm only).

    See
    ``tokenspeed_kernel.ops.gemm._hipblaslt.tuning.hipblaslt_online_tuning_cache``.
    A no-op context off ROCm.

    Args:
        path: The results file, such as ``hipblaslt_online_cache_path()``.
        write: Write ``path`` on a normal exit (one process per shared file).
    """
    if torch.version.hip is None:
        return contextlib.nullcontext()
    return tuning.hipblaslt_online_tuning_cache(path, write=write)
