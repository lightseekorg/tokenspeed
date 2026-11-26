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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import torch

from tokenspeed.runtime.model_loader import weight_utils
from tokenspeed.runtime.model_loader.weight_utils import initialize_dummy_weights


class _MixedDtypeModule(torch.nn.Module):
    def __init__(self, fp8_shape: tuple[int, ...]) -> None:
        super().__init__()
        # Pre-fill with an out-of-range sentinel so missed writes are caught.
        self.fp8_weight = torch.nn.Parameter(
            torch.full(fp8_shape, 240.0, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        # Transposed view is non-contiguous; writes must still land in param.data.
        self.fp8_noncontig = torch.nn.Parameter(
            torch.full((64, 48), 240.0, dtype=torch.float8_e4m3fn).t(),
            requires_grad=False,
        )
        self.bf16_weight = torch.nn.Parameter(
            torch.empty((16, 32), dtype=torch.bfloat16), requires_grad=False
        )
        self.empty_fp8 = torch.nn.Parameter(
            torch.empty((0, 16), dtype=torch.float8_e4m3fn), requires_grad=False
        )


def test_initialize_dummy_weights_fp8_multi_chunk_and_noncontiguous(monkeypatch):
    # Use a small chunk size to exercise multiple iterations and a ragged tail cheaply.
    monkeypatch.setattr(weight_utils, "_DUMMY_SUB16_CHUNK_ELEMS", 1000)

    # 33 * 128 = 4224 elements -> 4 full chunks of 1000 + ragged tail of 224.
    mod_a = _MixedDtypeModule((33, 128))
    mod_b = _MixedDtypeModule((33, 128))
    assert not mod_a.fp8_noncontig.is_contiguous()

    initialize_dummy_weights(mod_a, low=-1e-2, high=1e-2, seed=42)
    initialize_dummy_weights(mod_b, low=-1e-2, high=1e-2, seed=42)

    assert torch.equal(
        mod_a.fp8_weight.view(torch.uint8), mod_b.fp8_weight.view(torch.uint8)
    )
    assert torch.equal(
        mod_a.fp8_noncontig.contiguous().view(torch.uint8),
        mod_b.fp8_noncontig.contiguous().view(torch.uint8),
    )
    assert torch.equal(mod_a.bf16_weight, mod_b.bf16_weight)

    for tensor in (mod_a.fp8_weight, mod_a.fp8_noncontig):
        as_f32 = tensor.to(torch.float32)
        assert torch.isfinite(as_f32).all()
        assert float(as_f32.min()) >= -1.2e-2
        assert float(as_f32.max()) <= 1.2e-2
        assert float(as_f32.abs().max()) > 0.0

    # Confirm values were actually written across both the first chunk and ragged tail.
    fp8_flat = mod_a.fp8_weight.to(torch.float32).view(-1)
    assert float(fp8_flat[:1000].abs().max()) > 0.0
    assert float(fp8_flat[-224:].abs().max()) > 0.0
