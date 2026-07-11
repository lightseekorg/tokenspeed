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

from __future__ import annotations

import torch
from tokenspeed_kernel.thirdparty.deep_gemm import warmup


def test_prefill_warmup_keeps_deep_gemm_mhc_enabled_by_default(monkeypatch) -> None:
    calls: list[tuple[list[dict], int, torch.device]] = []

    def fake_prenorm_warmup(shapes, max_tokens, device):
        calls.append((shapes, max_tokens, device))

    monkeypatch.setattr(
        warmup,
        "_warmup_tf32_hc_prenorm_gemm",
        fake_prenorm_warmup,
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    device = torch.device("cuda", 0)

    warmup.warmup_prefill_jit(
        hidden_size=64,
        num_attention_heads=8,
        hc_mult=4,
        max_tokens=32,
        device=device,
    )

    assert calls == [
        (
            [{"hc_hidden_size": 256, "mix_hc": 24, "hc_dim": 256}],
            32,
            device,
        )
    ]
