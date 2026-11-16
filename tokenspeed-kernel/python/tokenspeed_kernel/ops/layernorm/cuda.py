# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""CUDA layernorm kernels."""

from tokenspeed_kernel.op_profile import OpProfile, wrap_op_for_profiling
from tokenspeed_kernel.registry import KernelSpec, error_fn

try:
    from tokenspeed_kernel.thirdparty.cuda.rmsnorm import rmsnorm_fused_parallel

    rmsnorm_fused_parallel = wrap_op_for_profiling(
        rmsnorm_fused_parallel,
        spec=KernelSpec(
            name="cuda_rmsnorm_fused_parallel",
            family="layernorm",
            mode="rmsnorm_fused_parallel",
        ),
        profile=OpProfile(cost_rule="rmsnorm_fused_parallel.v1"),
    )
except ImportError:
    rmsnorm_fused_parallel = error_fn

__all__ = ["rmsnorm_fused_parallel"]
