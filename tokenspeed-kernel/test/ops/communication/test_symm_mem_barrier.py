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

"""The symmetric-memory barrier signals and consumes its peers' flags."""

import pytest
import torch
from tokenspeed_kernel.ops.communication.triton import (
    _dp_sampling_swap_barrier_kernel,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an admitted GPU")
def test_a_single_rank_barrier_completes_and_clears_its_flag():
    # One rank: it signals itself, then consumes that signal.
    pad = torch.zeros(4, dtype=torch.int32, device="cuda")
    pad_ptrs = torch.tensor([pad.data_ptr()], dtype=torch.int64, device="cuda")
    for _ in range(3):
        _dp_sampling_swap_barrier_kernel[(1,)](pad_ptrs, RANK=0, WORLD_SIZE=1)
        torch.cuda.synchronize()
        assert pad.tolist() == [0, 0, 0, 0]
