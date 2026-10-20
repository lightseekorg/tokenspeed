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

"""Benchmark generators for activation quantization operations."""

from __future__ import annotations

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
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

__all__ = ["prepare_fp8_with_scale"]


_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}
_IMPLEMENTED_GRANULARITIES = frozenset({"token_group"})


def _invalid(message: str) -> BenchmarkCaseError:
    return BenchmarkCaseError(BenchmarkStatus.INVALID_CASE, message)


def _positive(parameters: dict[str, object], name: str) -> int:
    value = parameters.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise _invalid(f"FP8 quantization {name} must be a positive integer")
    return value


def prepare_fp8_with_scale(
    request: BenchmarkRequest,
    platform: PlatformInfo,
) -> PreparedBenchmark:
    """Prepare dynamic FP8 quantization of one ``[M, K]`` activation.

    Models quantize the input of an FP8 linear layer this way, computing one
    scale per ``group_size`` values of each row. As in the models, the call
    passes no PDL request. Cases select by traits rather than pinning a
    registration, so a faster registration with the same traits is compared
    against the current one.
    """

    parameters = request.parameters
    unknown = sorted(
        set(parameters)
        - {"M", "K", "granularity", "group_size", "scale_encoding", "dtype"}
    )
    if unknown:
        raise _invalid(f"Unknown FP8 quantization parameters: {', '.join(unknown)}")
    M = _positive(parameters, "M")
    K = _positive(parameters, "K")
    granularity = parameters.get("granularity")
    if granularity not in _IMPLEMENTED_GRANULARITIES:
        raise _invalid(
            f"Implemented FP8 quantization granularity values: "
            f"{', '.join(sorted(_IMPLEMENTED_GRANULARITIES))}"
        )
    group_size = _positive(parameters, "group_size")
    scale_encoding = parameters.get("scale_encoding")
    if not isinstance(scale_encoding, str):
        raise _invalid("FP8 quantization scale_encoding must be a string")
    dtype_name = parameters.get("dtype")
    if dtype_name not in _DTYPES:
        raise _invalid(
            f"FP8 quantization dtype must be one of: {', '.join(sorted(_DTYPES))}"
        )
    dtype = _DTYPES[dtype_name]

    from tokenspeed_kernel.ops.quantization import quantize_fp8

    load_builtin_kernels()
    try:
        selected = select_kernel(
            "quantization",
            "fp8_with_scale",
            format_signature(x=dense_tensor_format(dtype)),
            platform=platform,
            traits={
                "granularity": f"{granularity}_{group_size}",
                "scale_encoding": scale_encoding,
                "scale_layout": "token_major",
                # quantize_fp8 states the rows of a 2-D input.
                "m": M,
            },
            solution=request.solution,
            override=request.registration,
        )
    except NoKernelFoundError as error:
        raise BenchmarkCaseError(BenchmarkStatus.NOT_APPLICABLE, str(error)) from error
    spec = KernelRegistry.get().get_by_name(selected.name)
    if spec is None:
        raise BenchmarkCaseError(
            BenchmarkStatus.REGISTRATION_MISSING,
            f"Selected registration {selected.name!r} is not available",
        )

    generator = torch.Generator(device="cuda").manual_seed(request.seed)
    x = torch.randn((M, K), device="cuda", dtype=dtype, generator=generator)

    def invoke() -> object:
        return quantize_fp8(
            x,
            granularity=granularity,
            group_size=group_size,
            scale_encoding=scale_encoding,
            enable_pdl=False,
            override=request.registration,
            solution=request.solution,
        )

    return PreparedBenchmark(
        registration=spec,
        invocation=PreparedInvocation(invoke=invoke),
        parameters={
            "M": M,
            "K": K,
            "granularity": granularity,
            "group_size": group_size,
            "scale_encoding": scale_encoding,
            "dtype": dtype_name,
            "enable_pdl": False,
        },
        validation=None,
    )
