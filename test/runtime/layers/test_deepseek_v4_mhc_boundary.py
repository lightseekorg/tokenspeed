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

import tokenspeed_kernel
import torch

from tokenspeed.runtime.layers import deepseek_v4_mhc


def test_runtime_mhc_calls_tokenspeed_kernel_boundary(monkeypatch) -> None:
    x_prev = torch.randn(2, 8, dtype=torch.bfloat16)
    residual_prev = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    post_prev = torch.randn(2, 4, 1, dtype=torch.float32)
    comb_prev = torch.randn(2, 4, 4, dtype=torch.float32)
    fn = torch.randn(24, 32, dtype=torch.float32)
    scale = torch.ones(3, dtype=torch.float32)
    base = torch.zeros(24, dtype=torch.float32)
    residual_cur = torch.empty_like(residual_prev)
    layer_input = torch.empty_like(x_prev)
    post_cur = torch.empty_like(post_prev)
    comb_cur = torch.empty_like(comb_prev)
    calls: list[tuple[str, tuple[object, ...]]] = []

    def fake_post(*args):
        calls.append(("post", args))
        return residual_cur

    def fake_pre(*args):
        calls.append(("pre", args))
        return layer_input, post_cur, comb_cur

    monkeypatch.setattr(tokenspeed_kernel, "mhc_post", fake_post)
    monkeypatch.setattr(tokenspeed_kernel, "mhc_pre", fake_pre)

    result = deepseek_v4_mhc.mhc_fused_hc(
        x_prev,
        residual_prev,
        post_prev,
        comb_prev,
        fn,
        scale,
        base,
        1e-6,
        2e-6,
        3,
    )

    assert result == (residual_cur, layer_input, post_cur, comb_cur)
    assert calls == [
        ("post", (x_prev, residual_prev, post_prev, comb_prev)),
        ("pre", (residual_cur, fn, scale, base, 1e-6, 2e-6, 3)),
    ]
