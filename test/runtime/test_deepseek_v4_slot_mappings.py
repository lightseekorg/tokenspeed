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

"""The per-forward slot-mapping memo DeepSeek V4's layers share.

It replaced the ``dsa_swa_slot_mapping`` / ``dsa_compressor_slot_cache``
tensor fields on ``ForwardContext``: the mappings are backend scratch that
every metadata publish clears. CPU-only.
"""

from __future__ import annotations

from dataclasses import fields

import torch

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.attention.deepseek_v4.slot_mappings import (
    DeepseekV4ForwardSlotMappings,
)


def test_get_or_compute_computes_once_per_key_and_reuses():
    memo = DeepseekV4ForwardSlotMappings()
    calls: list[str] = []

    def build(tag: str):
        def compute():
            calls.append(tag)
            return torch.full((3,), len(calls), dtype=torch.int64)

        return compute

    swa = memo.get_or_compute("swa", build("swa"))
    state = memo.get_or_compute(("state", 4), build("state4"))

    assert memo.get_or_compute("swa", build("swa")) is swa
    assert memo.get_or_compute(("state", 4), build("state4")) is state
    # Different ratios are different mappings.
    assert memo.get_or_compute(("state", 128), build("state128")) is not state
    assert calls == ["swa", "state4", "state128"]


def test_clear_starts_a_fresh_forward():
    memo = DeepseekV4ForwardSlotMappings()
    first = memo.get_or_compute("swa", lambda: torch.zeros(2, dtype=torch.int64))
    memo.clear()
    second = memo.get_or_compute("swa", lambda: torch.ones(2, dtype=torch.int64))
    assert second is not first
    assert torch.equal(second, torch.ones(2, dtype=torch.int64))


def test_forward_context_carries_no_v4_memo_fields():
    names = {f.name for f in fields(ForwardContext)}
    assert not {"dsa_swa_slot_mapping", "dsa_compressor_slot_cache"} & names
