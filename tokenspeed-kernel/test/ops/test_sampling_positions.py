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

"""Active sampling offsets preserve slots without per-batch recompilation."""

from test.utils import assert_no_triton_compile

import torch
from tokenspeed_kernel.ops.sampling import prepare_sampling_offsets
from tokenspeed_kernel.ops.sampling.triton.positions import _prepare_sampling_offsets


def test_active_counts_reuse_the_preparation_kernel():
    assert torch.cuda.is_available(), "Sampling offset acceptance requires CUDA"
    frontier = torch.arange(257, dtype=torch.int32, device="cuda")
    slots = torch.arange(1, 129, dtype=torch.int64, device="cuda")
    lengths = torch.full((128,), 37, dtype=torch.int32, device="cuda")
    out = torch.zeros(257, dtype=torch.int64, device="cuda")
    prepare_sampling_offsets(frontier, slots, lengths, out=out)
    with assert_no_triton_compile(_prepare_sampling_offsets):
        for rows in (1, 3, 16, 17, 65, 128):
            for extends in (0, 1, rows):
                before = out.clone()
                prepare_sampling_offsets(
                    frontier,
                    slots[:rows],
                    lengths[:extends] if extends else None,
                    out=out,
                )
                expected = frontier[slots[:rows]].long() + 1
                expected[:extends] += 36
                assert torch.equal(out[slots[:rows]], expected)
                assert out[0].item() == 0
                assert torch.equal(out[rows + 1 :], before[rows + 1 :])
