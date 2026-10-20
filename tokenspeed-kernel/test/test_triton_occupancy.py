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
import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import current_platform, max_active_ctas

pytestmark = pytest.mark.skipif(
    current_platform().vendor not in {"nvidia", "amd"},
    reason="Compiled-kernel occupancy requires a CUDA or HIP GPU",
)


@triton.jit
def _copy_for_occupancy(source, target, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    tl.store(target + offsets, tl.load(source + offsets))


@pytest.mark.parametrize("num_warps", [4, 8])
def test_occupancy_queries_loaded_kernel_resources(num_warps):
    source = torch.randn(256, device="cuda")
    target = torch.empty_like(source)
    kernel = _copy_for_occupancy.warmup(
        source, target, BLOCK=256, num_warps=num_warps, grid=(1,)
    )
    resident = max_active_ctas(kernel)
    assert resident > 0
    threads = num_warps * kernel.metadata.target.warp_size
    maximum_threads = current_platform().max_threads_per_sm
    if maximum_threads:
        assert resident * threads <= maximum_threads
    _copy_for_occupancy[(1,)](source, target, BLOCK=256, num_warps=num_warps)
    torch.testing.assert_close(target, source, rtol=0, atol=0)
