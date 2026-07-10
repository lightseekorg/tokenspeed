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

import importlib

import tokenspeed_kernel.ops.attention.trtllm.deepseek_v4 as trtllm_indexer
import torch
from tokenspeed_kernel.ops.transform import hadamard_transform
from tokenspeed_kernel.platform import ArchVersion, current_platform
from tokenspeed_kernel.registry import KernelRegistry, Priority
from tokenspeed_kernel.signature import format_signatures


def test_trtllm_indexer_q_uses_registered_hadamard_transform() -> None:
    assert trtllm_indexer.hadamard_transform is hadamard_transform


def test_trtllm_indexer_q_registration_matches_availability(fresh_registry) -> None:
    importlib.reload(trtllm_indexer)
    spec = KernelRegistry.get().get_by_name(
        "trtllm_deepseek_v4_indexer_q_prepare_mxfp4"
    )
    available = (
        trtllm_indexer.has_trtllm_deepseek_v4_indexer_q_prepare()
        and current_platform().is_nvidia
    )
    assert (spec is not None) is available
    if spec is None:
        return

    assert spec.family == "attention"
    assert spec.mode == "deepseek_v4_indexer_q_prepare_mxfp4"
    assert spec.solution == "trtllm"
    assert spec.priority == Priority.SPECIALIZED
    assert spec.format_signatures == format_signatures(
        "index_q", "dense", {torch.bfloat16}
    )
    assert spec.traits == {
        "num_heads": frozenset({64}),
        "head_dim": frozenset({128}),
        "rope_dim": frozenset({64}),
    }
    assert spec.capability.min_arch_version == ArchVersion(10, 0)
    assert spec.capability.max_arch_version == ArchVersion(10, 9)
