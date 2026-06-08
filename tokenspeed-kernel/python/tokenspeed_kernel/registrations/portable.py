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

import importlib
import logging

logger = logging.getLogger(__name__)

_MODULES = (
    # TODO: Replace these reference numerics registrations with Triton-backed
    # portable kernels so built-ins do not rely on numerics modules long-term.
    "tokenspeed_kernel.numerics.reference.gemm",
    "tokenspeed_kernel.numerics.reference.moe",
    "tokenspeed_kernel.numerics.reference.quantize",
    "tokenspeed_kernel.ops.attention.triton",
    "tokenspeed_kernel.ops.embedding.triton",
    "tokenspeed_kernel.ops.gemm.triton",
    "tokenspeed_kernel.ops.moe",
    "tokenspeed_kernel.ops.moe.triton",
    "tokenspeed_kernel.ops.moe.triton_kernels",
    "tokenspeed_kernel.ops.quantization.triton",
)


def load() -> None:
    for module_name in _MODULES:
        try:
            module = importlib.import_module(module_name)
            importlib.reload(module)
        except ImportError as exc:
            logger.debug(
                "Skipping built-in registration module %s: %s", module_name, exc
            )
