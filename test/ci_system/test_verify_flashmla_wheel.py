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

import pytest
from verify_flashmla_wheel import verify_binary_reports, verify_torch_cuda


@pytest.mark.parametrize(("torch_cuda", "cuda_major"), [("12.6", 12), ("13.0", 13)])
def test_accepts_torch_with_the_selected_cuda_major(torch_cuda, cuda_major):
    verify_torch_cuda(torch_cuda, cuda_major)


@pytest.mark.parametrize("torch_cuda", [None, "12.6"])
def test_rejects_torch_incompatible_with_a_cuda_13_wheel(torch_cuda):
    with pytest.raises(ValueError, match="Expected Torch CUDA 13"):
        verify_torch_cuda(torch_cuda, 13)


@pytest.mark.parametrize(("cuda_major", "suffix"), [(12, "90a"), (13, "90")])
def test_accepts_hopper_kernels_with_matching_cuda_runtime(cuda_major, suffix):
    listing = "\n".join(
        f"ELF file {index}: v32_persistent_h{heads}.sm_{suffix}.cubin"
        for index, heads in enumerate((64, 128), start=1)
    )
    dynamic = f"0x1 (NEEDED) Shared library: [libcudart.so.{cuda_major}]"
    verify_binary_reports(listing, dynamic, cuda_major)


@pytest.mark.parametrize("present_heads", [(), (64,), (128,)])
def test_rejects_wheels_without_both_hopper_sparse_kernels(present_heads):
    listing = "\n".join(
        f"ELF file {heads}: v32_persistent_h{heads}.sm_90a.cubin"
        for heads in present_heads
    )
    listing += "\nELF file 999: v32_persistent_h64.sm_100a.cubin"
    with pytest.raises(ValueError, match="missing Hopper kernels"):
        verify_binary_reports(listing, "(NEEDED) [libcudart.so.12]", 12)


@pytest.mark.parametrize("runtime", ["13", "", "12]\n(NEEDED) [libcudart.so.13"])
def test_rejects_wrong_or_ambiguous_cuda_runtime(runtime):
    listing = "\n".join(f"v32_persistent_h{heads}.sm_90a.cubin" for heads in (64, 128))
    dynamic = f"(NEEDED) [libcudart.so.{runtime}]"
    with pytest.raises(ValueError, match="must link libcudart.so.12"):
        verify_binary_reports(listing, dynamic, 12)
