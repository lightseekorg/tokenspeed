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

"""TokenSpeed Intel XPU kernels.

Importing this package registers Intel XPU-specialized kernels into the shared
TokenSpeed kernel registry. The actual compute is provided by the third-party
``vllm_xpu_kernels`` package (SYCL/DPC++ + oneDNN). Importing ``vllm_xpu_kernels``
also registers its custom Torch ops into the PyTorch dispatcher.

The import is best-effort: on a machine without ``vllm_xpu_kernels`` installed
(or not on an Intel XPU), importing this package is a no-op so that the runtime
transparently falls back to the portable Triton kernels.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Registering vllm_xpu_kernels' Torch ops requires importing its compiled
# extension. Guard it so a missing/incompatible install does not break import.
try:
    import vllm_xpu_kernels._C  # noqa: F401  (side effect: registers torch ops)

    _VLLM_XPU_KERNELS_AVAILABLE = True
except Exception as exc:  # pragma: no cover - environment dependent
    _VLLM_XPU_KERNELS_AVAILABLE = False
    logger.warning(
        "tokenspeed_kernel_intel: vllm_xpu_kernels unavailable (%s); "
        "Intel XPU kernels will not be registered and the runtime will fall "
        "back to the portable Triton path.",
        exc,
    )

if _VLLM_XPU_KERNELS_AVAILABLE:
    # Side-effect imports: each submodule registers its kernels via
    # @register_kernel. Add new families here as they are implemented.
    from tokenspeed_kernel_intel.ops import attention  # noqa: F401

__all__ = ["_VLLM_XPU_KERNELS_AVAILABLE"]
