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

"""LongCat structural initial/local candidate selection."""

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsa._triton.topk import (
    mark_forced_initial_local_logits,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_forced_initial_and_local_candidates_survive_topk() -> None:
    topk = 2048
    causal_lens = torch.tensor([8, 2049, 4096], device="cuda", dtype=torch.int32)
    logits = torch.randn(3, 4096, device="cuda")
    columns = torch.arange(4096, device="cuda")
    logits.masked_fill_(columns.unsqueeze(0) >= causal_lens.unsqueeze(1), float("-inf"))

    mark_forced_initial_local_logits(
        logits,
        causal_lens,
        initial_tokens=16,
        local_tokens=1024,
    )

    selected = torch.topk(logits, topk).indices.cpu()
    for row, causal_len in enumerate(causal_lens.cpu().tolist()):
        selected_ids = set(selected[row, : min(causal_len, topk)].tolist())
        forced_ids = set(range(min(16, causal_len)))
        forced_ids.update(range(max(16, causal_len - 1024), causal_len))
        assert forced_ids <= selected_ids


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_combine_topk_weights_tolerates_padded_scales() -> None:
    from tokenspeed_kernel.ops.attention.dsa._triton.topk import combine_topk_weights
    from tokenspeed_kernel.ops.quantization import quantize_fp8_with_scale

    # Quantizers pad scale rows on some backends but not others. Add trailing
    # NaNs explicitly so only the 16 real rows may be read on every backend.
    q = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)
    _, scale = quantize_fp8_with_scale(
        q, granularity="token_group", group_size=128, scale_encoding="float32"
    )
    real_scale = scale.reshape(-1)[:16]
    padded_scale = torch.cat((real_scale, torch.full_like(real_scale, float("nan"))))
    weights = torch.randn(1, 16, device="cuda", dtype=torch.bfloat16)
    out = combine_topk_weights(weights, padded_scale, 0.25)
    expected = weights.float() * real_scale.view(1, 16) * 0.25
    torch.testing.assert_close(out, expected)
