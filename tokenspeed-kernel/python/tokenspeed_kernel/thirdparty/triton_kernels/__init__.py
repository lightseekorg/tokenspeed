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

from tokenspeed_kernel._triton import redirect_triton_to_tokenspeed_triton

# Different upstream triton_kernels versions ship different submodule names
# (older releases used ``matmul`` / ``matmul_details``; newer ones rename
# them to ``matmul_ogs`` / ``matmul_ogs_details``). Wrap each pre-import in
# try/except so an installed-version mismatch doesn't break the whole
# import chain — call sites in tokenspeed_kernel.ops already guard their
# actual usages with ImportError fallbacks.
with redirect_triton_to_tokenspeed_triton():
    import triton_kernels  # noqa: F401

    for _mod in (
        "triton_kernels.matmul",
        "triton_kernels.matmul_details",
        "triton_kernels.matmul_details.opt_flags",
        "triton_kernels.matmul_ogs",
        "triton_kernels.matmul_ogs_details",
        "triton_kernels.matmul_ogs_details.opt_flags",
        "triton_kernels.numerics",
        "triton_kernels.swiglu",
        "triton_kernels.tensor",
        "triton_kernels.tensor_details",
        "triton_kernels.tensor_details.layout",
        "triton_kernels.topk",
    ):
        try:
            __import__(_mod)
        except ImportError:
            pass
    del _mod
