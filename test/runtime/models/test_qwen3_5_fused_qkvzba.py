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

"""QKVZBA repacking preserves data and produces independent, aligned outputs."""

import os
import sys

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, suite="runtime-1gpu")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_repack_compiles_for_amd():
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from tokenspeed.runtime.models.qwen3_5 import (
        fused_qkvzba_split_reshape_cat_contiguous_kernel as kernel,
    )

    # Exercise AMD pointer canonicalization even when CI runs on NVIDIA GPUs.
    triton.compile(
        ASTSource(
            kernel,
            signature={name: "*bf16" for name in kernel.arg_names[:6]},
            constexprs=dict(
                stride_qkvz=4120,
                stride_ba=4120,
                NUM_HEADS_QK=4,
                NUM_HEADS_V=12,
                HEAD_QK=128,
                HEAD_V=128,
                BLOCK=2048,
                BLOCK_BA=16,
                ENABLE_PDL=False,
            ),
            attrs={
                (i,): [["tt.divisibility", 16], ["tt.pointer_range", 32]]
                for i in range(6)
            },
        ),
        target=GPUTarget("hip", "gfx950", 64),
        options={"num_warps": 4, "num_stages": 1},
    )


@pytest.mark.parametrize(
    "rows,nk,nv,dq,dv,dtype,gate_dtype,layout",
    [
        (0, 4, 12, 128, 128, torch.bfloat16, torch.bfloat16, "packed"),
        (1, 4, 12, 128, 128, torch.bfloat16, torch.bfloat16, "packed"),
        (3, 1, 3, 128, 128, torch.bfloat16, torch.bfloat16, "packed"),
        (24, 4, 12, 128, 128, torch.bfloat16, torch.bfloat16, "packed"),
        (65, 8, 24, 128, 128, torch.bfloat16, torch.bfloat16, "packed"),
        (257, 4, 12, 64, 128, torch.float16, torch.float16, "packed"),
        (64, 4, 4, 128, 128, torch.bfloat16, torch.bfloat16, "separate"),
        (64, 4, 8, 128, 128, torch.bfloat16, torch.bfloat16, "separate"),
        (3, 2, 6, 128, 128, torch.bfloat16, torch.float32, "unaligned"),
        (65, 4, 16, 128, 128, torch.bfloat16, torch.bfloat16, "unaligned"),
        (32, 4, 4, 128, 64, torch.float32, torch.float32, "unaligned"),
    ],
)
def test_repack_layouts(rows, nk, nv, dq, dv, dtype, gate_dtype, layout):
    from tokenspeed.runtime.models.qwen3_5 import (
        fused_qkvzba_split_reshape_cat_contiguous,
    )

    torch.manual_seed(4)
    width = 2 * nk * dq + 2 * nv * dv
    if layout == "packed":
        projection = torch.randn(rows, width + 2 * nv, dtype=dtype, device="cuda")
        qkvz, ba = projection.split([width, 2 * nv], dim=-1)
    elif layout == "unaligned":
        qkvz = torch.randn(rows, width + 3, dtype=dtype, device="cuda")[
            :, 1 : width + 1
        ]
        ba = torch.randn(rows, 2 * nv + 5, dtype=gate_dtype, device="cuda")[
            :, 1 : 2 * nv + 1
        ]
        assert qkvz.data_ptr() % 16 != 0
        assert ba.data_ptr() % 16 != 0
    else:
        qkvz = torch.randn(rows, width, dtype=dtype, device="cuda")
        ba = torch.randn(rows, 2 * nv, dtype=gate_dtype, device="cuda")
    qkv_width = 2 * nk * dq + nv * dv
    if rows:
        for tensor in (qkvz, ba):
            tensor[0, :5] = torch.tensor(
                [0.0, -0.0, float("inf"), -float("inf"), float("nan")],
                dtype=tensor.dtype,
                device="cuda",
            )
        qkvz[0, qkv_width : qkv_width + 5] = qkvz[0, :5]
    inputs = (qkvz.clone(), ba.clone())
    outputs = fused_qkvzba_split_reshape_cat_contiguous(qkvz, ba, nk, nv, dq, dv)
    expected = (
        inputs[0][:, :qkv_width],
        inputs[0][:, qkv_width:].reshape(rows, nv, dv),
        *inputs[1].split(nv, dim=-1),
    )
    for actual, reference in zip(outputs, expected, strict=True):
        assert actual.shape == reference.shape
        assert actual.dtype == reference.dtype
        assert actual.is_contiguous()
        assert actual.data_ptr() % 16 == 0
    # Compare bytes to check NaN payloads, signed zero and unchanged inputs.
    for actual, reference in zip(
        (*outputs, qkvz, ba), (*expected, *inputs), strict=True
    ):
        assert torch.equal(
            actual.contiguous().view(torch.uint8),
            reference.contiguous().view(torch.uint8),
        )
    if rows:
        input_storage = {tensor.untyped_storage().data_ptr() for tensor in (qkvz, ba)}
        output_storage = {tensor.untyped_storage().data_ptr() for tensor in outputs}
        assert input_storage.isdisjoint(output_storage)
        ranges = sorted(
            (tensor.data_ptr(), tensor.data_ptr() + tensor.nbytes) for tensor in outputs
        )
        assert all(end <= start for (_, end), (start, _) in zip(ranges, ranges[1:]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
