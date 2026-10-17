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
"""Shared-memory gate of the BF16 Triton MoE's NVIDIA-only deep pipeline."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.ops.moe._triton import bf16
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_nvidia, reason="the deep pipeline is NVIDIA-only"
)


def test_the_deep_pipeline_needs_the_shared_memory_it_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deep 16-row tiles stage 145 KiB and 64-row tiles 193 KiB, barriers included."""

    def fits(kib: int, block_m: int) -> bool:
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda device: SimpleNamespace(shared_memory_per_block_optin=kib * 1024),
        )
        return bf16._deep_pipeline(torch.device("cuda", 0), block_m)

    assert not fits(100, 16)
    assert fits(160, 16) and not fits(160, 64)
    assert not fits(192, 64) and fits(194, 64)
