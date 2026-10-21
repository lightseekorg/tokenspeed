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

"""fused_decode_input_prep: positions, seq_lens and the clamped input ids."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from tokenspeed.runtime.execution.cache_loc_kernel import fused_decode_input_prep

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("clamp", [True, False])
@pytest.mark.parametrize("width", [1, 4])
def test_decode_prep_writes_positions_lengths_and_ids(width, clamp):
    dev = "cuda"
    vocab = 100
    pool = torch.tensor([3, 0, 5], dtype=torch.int64, device=dev)
    lengths = torch.tensor([7, 11, 0, 2, 9, 40], dtype=torch.int32, device=dev)
    # Out-of-vocab ids on both sides to exercise the clamp.
    future = torch.randint(-20, vocab + 20, (6, width), dtype=torch.int32, device=dev)
    bs = pool.numel()
    positions = torch.full((bs * width,), -1, dtype=torch.int64, device=dev)
    seq_lens = torch.zeros(bs, dtype=torch.int32, device=dev)
    ids = torch.full((bs * width,), -1, dtype=torch.int32, device=dev)

    fused_decode_input_prep(
        positions,
        seq_lens,
        pool,
        lengths,
        width,
        input_ids_out=ids,
        future_input_map=future,
        vocab_size=vocab if clamp else None,
    )

    starts = lengths[pool].long()
    want_positions = (starts[:, None] + torch.arange(width, device=dev)).flatten()
    assert torch.equal(positions, want_positions)
    assert torch.equal(seq_lens, (starts + width).int())
    want_ids = future[pool].flatten()
    assert torch.equal(ids, want_ids.clamp(0, vocab - 1) if clamp else want_ids)
