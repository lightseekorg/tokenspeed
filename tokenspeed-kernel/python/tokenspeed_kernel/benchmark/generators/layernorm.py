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

"""Benchmark generators for RMSNorm operations."""

from __future__ import annotations

import math
from typing import Any

import torch
from tokenspeed_kernel.benchmark.graph import PreparedInvocation
from tokenspeed_kernel.benchmark.harness import (
    BenchmarkCaseError,
    BenchmarkRequest,
    BenchmarkStatus,
    PreparedBenchmark,
)
from tokenspeed_kernel.platform import PlatformInfo
from tokenspeed_kernel.registry import KernelRegistry, load_builtin_kernels
from tokenspeed_kernel.selection import NoKernelFoundError, select_kernel
from tokenspeed_kernel.signature import dense_format_signature

__all__ = ["prepare_weightless_rmsnorm_gated_sigmoid"]


def _invalid(message: str) -> BenchmarkCaseError:
    return BenchmarkCaseError(BenchmarkStatus.INVALID_CASE, message)


def _positive(parameters: dict[str, Any], name: str) -> int:
    value = parameters[name]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise _invalid(f"Layernorm {name} must be a positive integer")
    return value


def prepare_weightless_rmsnorm_gated_sigmoid(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare one per-head weightless RMSNorm with a sigmoid output gate.

    The attention output ``[tokens, num_heads * head_dim]`` is contiguous. The
    gate is the trailing ``num_heads * head_dim`` columns of a packed
    projection whose rows are ``gate_row_stride`` elements apart, as when the
    gate is split from a fused Q/K/V/gate projection.
    """

    if request.registration is not None or request.solution is not None:
        raise _invalid("weightless_rmsnorm_gated_sigmoid cases use normal selection")
    parameters = request.parameters
    if parameters.get("validation") is not None:
        raise _invalid(
            "weightless_rmsnorm_gated_sigmoid correctness validation is not implemented"
        )
    if parameters["dtype"] != "bfloat16":
        raise _invalid("weightless_rmsnorm_gated_sigmoid dtype must be bfloat16")
    tokens = _positive(parameters, "tokens")
    num_heads = _positive(parameters, "num_heads")
    head_dim = _positive(parameters, "head_dim")
    gate_row_stride = _positive(parameters, "gate_row_stride")
    eps = parameters["eps"]
    if not isinstance(eps, float) or not math.isfinite(eps) or eps <= 0.0:
        raise _invalid("weightless_rmsnorm_gated_sigmoid eps must be a positive float")
    width = num_heads * head_dim
    if gate_row_stride < width:
        raise _invalid("weightless_rmsnorm_gated_sigmoid gate_row_stride is too small")

    load_builtin_kernels()
    try:
        kernel = select_kernel(
            "layernorm",
            "weightless_rmsnorm_gated_sigmoid",
            dense_format_signature(("x", "gate"), (torch.bfloat16, torch.bfloat16)),
            platform=platform,
        )
    except NoKernelFoundError as error:
        raise BenchmarkCaseError(BenchmarkStatus.NOT_APPLICABLE, str(error)) from error
    spec = KernelRegistry.get().get_by_name(kernel.name)
    if spec is None:
        raise BenchmarkCaseError(
            BenchmarkStatus.REGISTRATION_MISSING,
            f"Selected registration {kernel.name!r} is not available",
        )

    generator = torch.Generator(device="cuda").manual_seed(request.seed)
    x = torch.randn(
        (tokens, width), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    packed = torch.randn(
        (tokens, gate_row_stride),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    gate = packed[:, gate_row_stride - width :]

    from tokenspeed_kernel.ops.layernorm import weightless_rmsnorm_gated_sigmoid

    def invoke() -> object:
        return weightless_rmsnorm_gated_sigmoid(
            x, gate, num_heads, head_dim, eps, False
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke),
        parameters={
            "tokens": tokens,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "gate_row_stride": gate_row_stride,
            "eps": eps,
            "dtype": "bfloat16",
        },
        validation=None,
    )
