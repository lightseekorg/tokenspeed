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

"""CPU-only checks for the full-path Petit benchmark adapter."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def benchmark() -> dict[str, object]:
    return runpy.run_path(str(Path(__file__).with_name("bench_megamoe.py")))


def test_import_aiter_topk_uses_upstream_function(benchmark, monkeypatch):
    aiter = ModuleType("aiter")
    aiter.__path__ = []
    fused_moe = ModuleType("aiter.fused_moe")

    def fused_topk():
        pass

    fused_moe.fused_topk = fused_topk
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    monkeypatch.setitem(sys.modules, "aiter.fused_moe", fused_moe)

    assert benchmark["import_aiter_topk"]() is fused_topk


def test_rank_m_rejects_workspace_overflow(benchmark, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bench_megamoe.py"])
    args = benchmark["parse_args"]()
    args.rank_m = [1025] + [1] * 7

    with pytest.raises(ValueError, match="at most 1024 tokens per rank"):
        benchmark["validate_args"](args)
