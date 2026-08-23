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

"""The runtime helpers must reach the fused kernels, not just agree with them.

The kernel suites call the helpers directly, so they stay green if the runtime
were wired back to its torch chains. These observe the dispatch instead: revert
the wiring and they fail.
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from tokenspeed.runtime.layers.attention import (  # noqa: E402
    page_table as page_table_mod,
)


def test_expansion_dispatches_to_the_fused_kernel(monkeypatch):
    calls = []
    real = page_table_mod.fused_expand_page_table

    def spy(*args, **kwargs):
        calls.append(kwargs.get("ratio"))
        return real(*args, **kwargs)

    monkeypatch.setattr(page_table_mod, "fused_expand_page_table", spy)
    table = torch.randint(1, 500, (3, 8), device="cuda", dtype=torch.int32)
    out = page_table_mod.expand_page_table(
        table, block_granularity=128, kernel_page_size=64, max_kernel_pages=16
    )
    assert calls == [2], "the CUDA expansion must go through the fused kernel"
    assert out.shape == (3, 16)


def test_the_fused_allocation_is_not_zero_filled_first(monkeypatch):
    """The kernel writes every column, so a zeroed allocation is a wasted fill.
    Production callers omit `out`, which is exactly this path."""
    seen = []
    real_zeros = torch.zeros

    def spy(*args, **kwargs):
        seen.append(args)
        return real_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", spy)
    table = torch.randint(1, 500, (3, 8), device="cuda", dtype=torch.int32)
    page_table_mod.expand_page_table(
        table, block_granularity=128, kernel_page_size=64, max_kernel_pages=16
    )
    assert seen == [], "the fused path must allocate without a zero fill"


def test_a_cpu_table_keeps_the_portable_path(monkeypatch):
    """The backends' unit tests build CPU tensors; they must not reach Triton."""
    calls = []
    monkeypatch.setattr(
        page_table_mod,
        "fused_expand_page_table",
        lambda *a, **k: calls.append(1) or pytest.fail("CPU reached the CUDA kernel"),
    )
    table = torch.randint(1, 500, (2, 4), dtype=torch.int32)
    out = page_table_mod.expand_page_table(
        table, block_granularity=128, kernel_page_size=64, max_kernel_pages=8
    )
    assert not calls
    assert out.shape == (2, 8)
