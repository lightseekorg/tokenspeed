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

"""CPU regression for numerical-reference isolation."""

import torch
from qsa_fp8_prefill_reference import reference


def test_fp8_reference_preserves_matmul_precision():
    previous = torch.backends.cuda.matmul.fp32_precision
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        q = torch.ones((1, 12, 256), dtype=torch.bfloat16)
        # Exactly representable FP8 values keep this reference-only check on CPU.
        k = torch.zeros((2, 1, 256))
        v = torch.ones_like(k)
        slots = torch.full((1, 2051), -1, dtype=torch.int32)
        slots[0, 2048] = 1
        reference(
            q,
            k,
            v,
            slots,
            torch.tensor([0]),
            torch.tensor([0]),
            attention_scale=1 / 16,
            k_scale=1.0,
            v_scale=1.0,
        )
        assert torch.backends.cuda.matmul.fp32_precision == "tf32"
    finally:
        torch.backends.cuda.matmul.fp32_precision = previous
