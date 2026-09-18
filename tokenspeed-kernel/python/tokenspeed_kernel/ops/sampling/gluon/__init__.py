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

"""Registration shims for AMD Gluon sampling kernels."""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

try:
    from tokenspeed_kernel_amd.ops.gfx950.sampling.argmax import (
        gluon_argmax_gfx950 as _argmax_gfx950_impl,
    )
    from tokenspeed_kernel_amd.ops.gfx1250.sampling.argmax import (
        gluon_argmax_gfx1250 as _argmax_gfx1250_impl,
    )
except ImportError as exc:
    # Keep the message only: an exception object carries its traceback, which
    # pins every frame that was importing at the time for the process lifetime.
    _IMPORT_ERROR_MESSAGE = str(exc)
    _argmax_gfx950_impl = None
    _argmax_gfx1250_impl = None
else:
    _IMPORT_ERROR_MESSAGE = None


if _IMPORT_ERROR_MESSAGE is None:

    @register_kernel(
        "sampling",
        "argmax",
        name="gluon_argmax_gfx950",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 5),
            max_arch_version=ArchVersion(9, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            "logits", "dense", {torch.float16, torch.bfloat16, torch.float32}
        ),
        priority=Priority.SPECIALIZED,
        tags={"latency", "throughput"},
    )
    def gluon_argmax_gfx950(
        logits: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _argmax_gfx950_impl(logits, out=out)

    @register_kernel(
        "sampling",
        "argmax",
        name="gluon_argmax_gfx1250",
        solution="gluon",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(12, 5),
            max_arch_version=ArchVersion(12, 5),
            vendors=frozenset({"amd"}),
        ),
        signatures=format_signatures(
            "logits", "dense", {torch.float16, torch.bfloat16, torch.float32}
        ),
        priority=Priority.SPECIALIZED,
        tags={"latency", "throughput"},
    )
    def gluon_argmax_gfx1250(
        logits: torch.Tensor,
        *,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        """Reduce ``(M, N)`` logits into optional int32/int64 ``(M,)`` out.

        Returns out when supplied, otherwise a newly allocated int64 tensor.
        """
        return _argmax_gfx1250_impl(logits, out=out)

else:

    def gluon_argmax_gfx950(
        logits: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise ImportError(
            f"gluon_argmax_gfx950 requires tokenspeed-kernel-amd: {_IMPORT_ERROR_MESSAGE}"
        )

    def gluon_argmax_gfx1250(
        logits: torch.Tensor,
        *,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        raise ImportError(
            f"gluon_argmax_gfx1250 requires tokenspeed-kernel-amd: {_IMPORT_ERROR_MESSAGE}"
        )


__all__ = ["gluon_argmax_gfx950", "gluon_argmax_gfx1250"]
