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

"""Bit-exact checks for the FP32 reductions used by QSA score kernels."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

if not current_platform().is_blackwell:
    pytest.skip("Packed FP32 scores require Blackwell", allow_module_level=True)

import cutlass
from cuda.bindings import driver as cuda
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
from tokenspeed_kernel.thirdparty.cute_dsl._qsa_stream_topk.score import (
    _add_head_pairs,
    _paired_head_sum,
    _positive_f32,
)


class _HeadSumCheck:
    def __init__(self, use_neighbor_heads: bool):
        self.use_neighbor_heads = use_neighbor_heads

    @cute.jit
    def __call__(
        self,
        values: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self._kernel(values, output).launch(
            grid=(cute.ceil_div(values.shape[0], 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def _kernel(self, values: cute.Tensor, output: cute.Tensor):
        block, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        row = block * 128 + thread
        lane = cute.arch.lane_idx()
        if row < values.shape[0]:
            if cutlass.const_expr(self.use_neighbor_heads):
                a = _positive_f32(values[row, 0, 0])
                b = _positive_f32(values[row, 0, 1])
                c = _positive_f32(values[row, 0, 2])
                d = _positive_f32(values[row, 0, 3])
                packed = _add_head_pairs(
                    a,
                    b,
                    cute.arch.shuffle_sync(a, lane ^ 1),
                    cute.arch.shuffle_sync(b, lane ^ 1),
                    c,
                    d,
                    cute.arch.shuffle_sync(c, lane ^ 1),
                    cute.arch.shuffle_sync(d, lane ^ 1),
                )
            else:
                packed = _paired_head_sum(
                    values[row, 0, 0],
                    values[row, 0, 1],
                    values[row, 0, 2],
                    values[row, 0, 3],
                    values[row, 1, 0],
                    values[row, 1, 1],
                    values[row, 1, 2],
                    values[row, 1, 3],
                )
            output[row, 0] = cutlass.Uint32(packed)
            output[row, 1] = cutlass.Uint32(packed >> cutlass.Uint64(32))


@pytest.mark.parametrize("use_neighbor_heads", [False, True])
def test_score_head_sums_preserve_fp32_bits(
    device: str, use_neighbor_heads: bool
) -> None:
    """Both score layouts match an independent CPU FP32 rounding oracle."""
    generator = torch.Generator().manual_seed(160)
    bits = torch.randint(
        -(2**31),
        2**31,
        (131072, 2, 4),
        generator=generator,
        dtype=torch.int32,
    )
    special = torch.tensor(
        [
            0,
            -(2**31),
            1,
            -(2**31) + 1,
            0x007FFFFF,
            0x00800000,
            0x3F800000,
            0x3F800001,
            0x33800000,
            0x33000000,
            0x7F7FFFFF,
            0x7F800000,
            -0x00800000,
            0x7FC00000,
            0x7F800001,
            -1,
        ],
        dtype=torch.int32,
    )
    for head in range(4):
        bits[:32, 0, head] = special.repeat(2).roll(head * 3)
        bits[:32, 1, head] = special.repeat(2).roll(head * 5)
    original = bits.view(torch.float32)
    values = original.to(device)
    output = torch.empty(values.shape[0], 2, device=device, dtype=torch.uint32)
    compiled = cute.compile(
        _HeadSumCheck(use_neighbor_heads),
        from_dlpack(values, assumed_align=16),
        from_dlpack(output, assumed_align=16),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --opt-level 3",
    )
    compiled(values, output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(values, output)
    neighbors = torch.arange(32) ^ 1
    for step in range(4):
        changed = original.roll(step * 3, dims=0).roll(step, dims=2)
        if step == 2:
            changed = (changed.view(torch.int32) ^ -(2**31)).view(torch.float32)
        positive = torch.where(changed > 0, changed, torch.zeros_like(changed))
        if use_neighbor_heads:
            heads = positive[:, 0].reshape(-1, 32, 4)
            peer = heads[:, neighbors]
            expected = (
                (heads[:, :, [0, 2]] + heads[:, :, [1, 3]]) + peer[:, :, [0, 2]]
            ) + peer[:, :, [1, 3]]
        else:
            expected = (
                (positive[:, :, 0] + positive[:, :, 1]) + positive[:, :, 2]
            ) + positive[:, :, 3]
        values.copy_(changed)
        graph.replay()
        torch.testing.assert_close(
            output.cpu().view(torch.int32),
            expected.reshape(-1, 2).view(torch.int32),
            rtol=0,
            atol=0,
        )
