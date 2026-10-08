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

from types import SimpleNamespace

import pytest

from tokenspeed_kernel.ops.attention.mha._triton.prefill import (
    _has_large_shared_memory,
)
from tokenspeed_kernel.platform import _cuda_shared_memory_budget


@pytest.mark.parametrize(
    ("budget", "expected"),
    [(0, True), (101376, False), (128 * 1024, True)],
)
def test_mha_tile_memory_classification(budget: int, expected: bool) -> None:
    platform = SimpleNamespace(max_shared_memory_per_sm=budget)
    assert _has_large_shared_memory(platform) is expected


def test_cuda_shared_memory_budget_prefers_optin_limit() -> None:
    props = SimpleNamespace(
        shared_memory_per_block_optin=101376,
        shared_memory_per_block=49152,
    )
    assert _cuda_shared_memory_budget(props) == 101376


def test_cuda_shared_memory_budget_falls_back_to_default_limit() -> None:
    props = SimpleNamespace(shared_memory_per_block=49152)
    assert _cuda_shared_memory_budget(props) == 49152


def test_cuda_shared_memory_budget_warns_when_unavailable(caplog) -> None:
    with caplog.at_level("WARNING"):
        assert _cuda_shared_memory_budget(SimpleNamespace()) == 0
    assert "CUDA shared-memory limit is unavailable" in caplog.text
