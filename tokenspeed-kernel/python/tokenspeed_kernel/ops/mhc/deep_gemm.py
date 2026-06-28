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

"""DeepGEMM kernels used by manifold-constrained hyper-connections."""

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, error_fn, register_kernel
from tokenspeed_kernel.signature import format_signatures

try:
    from tokenspeed_kernel.thirdparty.deep_gemm import tf32_hc_prenorm_gemm as _impl
except ImportError:
    _impl = None


def has_deep_gemm_mhc() -> bool:
    """Return whether DeepGEMM's mHC pre-normalization GEMM is installed."""

    return _impl is not None


deep_gemm_mhc_prenorm_gemm = error_fn

if _impl is not None:

    @register_kernel(
        "mhc",
        "prenorm_gemm",
        name="deep_gemm_mhc_prenorm_gemm",
        solution="deep_gemm",
        capability=CapabilityRequirement(
            min_arch_version=ArchVersion(9, 0),
            vendors=frozenset({"nvidia"}),
        ),
        signatures=format_signatures("x", "dense", {torch.bfloat16}),
        traits={},
        priority=Priority.SPECIALIZED,
        tags={"latency", "throughput"},
    )
    def deep_gemm_mhc_prenorm_gemm(
        x: torch.Tensor,
        weight: torch.Tensor,
        output: torch.Tensor,
        square_sum: torch.Tensor,
        num_splits: int,
    ) -> None:
        """Run TF32 mHC projection and residual square-sum accumulation.

        Args:
            x: Flattened BF16 hyper-connected residual input.
            weight: FP32 pre-mapping projection weight.
            output: FP32 projection accumulator output buffer.
            square_sum: FP32 residual square-sum output buffer.
            num_splits: Number of split-K accumulator partitions.

        Returns:
            None. ``output`` and ``square_sum`` are populated in place.
        """

        _impl(x, weight, output, square_sum, num_splits)


__all__ = ["deep_gemm_mhc_prenorm_gemm", "has_deep_gemm_mhc"]
