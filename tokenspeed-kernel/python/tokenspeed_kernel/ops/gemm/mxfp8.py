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

"""Preparation and execution boundary for weight-only MXFP8 dense linears."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Protocol

import torch
from tokenspeed_kernel.profiling import kernel_scope
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import ScaleFormat, format_signature, tensor_format


class _PreparedMxfp8Linear(Protocol):
    """Storage and operations supplied by a registered weight-only backend."""

    weight: torch.Tensor
    weight_scale: torch.Tensor
    size_n: int
    size_k: int

    def apply(
        self, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor: ...

    def dequantize_for_absorption(self) -> torch.Tensor: ...


@dataclass(frozen=True)
class Mxfp8LinearPlan:
    """Backend selected once before packing, shared by eager and graph execution.

    Selection is pinned because prepared weight layouts are backend-specific.
    Changing an override after preparation requires a fresh plan/model load.
    """

    kernel_name: str
    backend: _PreparedMxfp8Linear

    @property
    def weight(self) -> torch.Tensor:
        return self.backend.weight

    @property
    def weight_scale(self) -> torch.Tensor:
        return self.backend.weight_scale

    def apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """Compute the prepared linear, recording the selected backend in profiles."""
        with kernel_scope(
            "gemm",
            "mxfp8_w8a16",
            x.dtype,
            kernel_name=self.kernel_name,
            m=prod(x.shape[:-1]),
            n=self.backend.size_n,
            k=self.backend.size_k,
        ):
            return self.backend.apply(x, bias)

    def dequantize_for_absorption(self) -> torch.Tensor:
        """Materialize absorbed MLA weights; never substitute a dense GEMM."""
        return self.backend.dequantize_for_absorption()


def prepare_mxfp8_linear(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    input_dtype: torch.dtype,
    use_atomic_add: bool,
    use_fp32_reduce: bool,
    solution: str | None,
    override: str | None,
) -> Mxfp8LinearPlan:
    """Select and prepare a weight-only MXFP8 backend before graph capture.

    Args:
        weight: E4M3 checkpoint weight [N, K], with K divisible by 32.
        weight_scale: Colocated raw uint8/E8M0 scales [N, K/32].
        input_dtype: Activation/output dtype used to select the backend.
        use_atomic_add: Explicit permission for atomic split-K reduction.
        use_fp32_reduce: Explicit FP32 split-K reduction policy.
        solution: Registered solution restriction; None enables auto-selection.
        override: Registered name/solution override; None supplies no per-call
            override. Environment and scoped overrides retain normal precedence.

    Returns:
        A plan owning backend-specific packed storage and an apply operation.
        No BF16 dense-GEMM fallback or activation quantization is introduced.

    The registered gemm.mxfp8_w8a16 callable is a preparation factory, not a
    stateless GEMM. Selection uses the raw checkpoint format, before repacking.
    The selected backend checks its device and numerical contracts even when
    an explicit override bypasses the selector's applicability filtering.
    """
    if weight.dtype != torch.float8_e4m3fn:
        raise TypeError("MXFP8 dense weights must use E4M3 storage")
    if weight_scale.dtype not in (torch.uint8, torch.float8_e8m0fnu):
        raise TypeError("MXFP8 dense scales must use raw uint8/E8M0 storage")
    if weight.ndim != 2 or min(weight.shape) <= 0 or weight.shape[1] % 32:
        raise ValueError(
            "MXFP8 dense weights require positive [N, K], K divisible by 32"
        )
    n, k = weight.shape
    if weight_scale.shape != (n, k // 32) or weight_scale.device != weight.device:
        raise ValueError("MXFP8 scales must be colocated [N, K/32]")
    if not isinstance(use_atomic_add, bool) or not isinstance(use_fp32_reduce, bool):
        raise TypeError("MXFP8 reduction policies must be bool")
    raw_scale = weight_scale.view(torch.uint8)
    selected = select_kernel(
        "gemm",
        "mxfp8_w8a16",
        format_signature(
            a=tensor_format("dense", input_dtype),
            b=tensor_format(
                "mxfp8",
                weight.dtype,
                scale=ScaleFormat(
                    storage_dtype=raw_scale.dtype,
                    granularity="block",
                    block_shape=(1, 32),
                ),
            ),
        ),
        traits={
            "n": n,
            "k": k,
            "out_dtype": input_dtype,
            "use_atomic_add": use_atomic_add,
            "use_fp32_reduce": use_fp32_reduce,
        },
        solution=solution,
        override=override,
    )
    backend = selected(
        weight,
        raw_scale,
        input_dtype=input_dtype,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=use_fp32_reduce,
    )
    return Mxfp8LinearPlan(kernel_name=selected.name, backend=backend)
