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

The runtime entry point. Loading, the stack check and the tuning tool live in
``tokenspeed_kernel_amd.ops.gemm.hipblaslt``; this module only keeps that
AMD package out of non-ROCm processes.
"""

from __future__ import annotations

import logging

import torch

__all__ = ["load_hipblaslt_tuned_solutions"]

logger = logging.getLogger(__name__)


def load_hipblaslt_tuned_solutions(path: str) -> bool:
    """Replay the BLAS solutions (hipBLASLt or rocBLAS) in a TunableOp results file.

    See ``tokenspeed_kernel_amd.ops.gemm.hipblaslt.load_hipblaslt_tuned_solutions``.

    Args:
        path: TunableOp results CSV made on the same software stack and GPU
            architecture.

    Returns:
        Whether the file was loaded. False (with a warning) on a non-ROCm
        build or when the AMD implementation rejects the file.
    """
    if torch.version.hip is None:
        logger.warning(f"Ignoring tuned BLAS solutions {path}: not a ROCm build")
        return False
    from tokenspeed_kernel_amd.ops.gemm.hipblaslt import (
        load_hipblaslt_tuned_solutions as load_amd,
    )

    return load_amd(path)
