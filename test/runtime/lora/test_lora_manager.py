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

"""Tests for LoraManager.prepare_loras → persistent batch_info.

The captured CUDA graph references the manager's batch_info tensors, so
their pointers must be stable across ``prepare_loras`` calls and the
contents must reflect each step's per-request slot ids.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.lora.lora_manager import LoraManager


def _model_config():
    return SimpleNamespace(
        num_hidden_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=4,
    )


@pytest.fixture
def manager():
    if not torch.cuda.is_available():
        pytest.skip("LoraManager allocates GPU buffers")
    return LoraManager(
        model_config=_model_config(),
        max_loras=2,
        max_lora_rank=8,
        max_num_tokens=64,
        dtype=torch.float16,
        device=torch.device("cuda:0"),
    )


def test_batch_info_tensor_addresses_are_stable(manager):
    bi = manager.batch_info
    addrs_before = (
        bi.seg_lens.data_ptr(),
        bi.seg_indptr.data_ptr(),
        bi.weight_indices.data_ptr(),
        bi.lora_ranks.data_ptr(),
        bi.scalings.data_ptr(),
    )
    manager.prepare_loras([0, 0, 0], per_request_token_counts=1)
    manager.prepare_loras([0, 0], per_request_token_counts=4)
    addrs_after = (
        bi.seg_lens.data_ptr(),
        bi.seg_indptr.data_ptr(),
        bi.weight_indices.data_ptr(),
        bi.lora_ranks.data_ptr(),
        bi.scalings.data_ptr(),
    )
    assert addrs_before == addrs_after


def test_prepare_loras_uniform_decode(manager):
    n = manager.prepare_loras([0, 0, 0, 0], per_request_token_counts=1)
    assert n == 4
    bi = manager.batch_info
    assert bi.bs == 4
    assert bi.num_segments == 4
    assert bi.max_len == 1
    torch.cuda.synchronize()
    assert bi.seg_lens[:4].tolist() == [1, 1, 1, 1]
    assert bi.seg_indptr[:5].tolist() == [0, 1, 2, 3, 4]
    assert bi.weight_indices[:4].tolist() == [0, 0, 0, 0]


def test_prepare_loras_target_verify_repeats(manager):
    # Each request emits ``spec_num_tokens`` tokens; one segment per request.
    n = manager.prepare_loras([0, 0], per_request_token_counts=3)
    assert n == 6
    bi = manager.batch_info
    assert bi.bs == 2
    assert bi.max_len == 3
    torch.cuda.synchronize()
    assert bi.seg_lens[:2].tolist() == [3, 3]
    assert bi.seg_indptr[:3].tolist() == [0, 3, 6]


def test_prepare_loras_variable_segments(manager):
    n = manager.prepare_loras([0, 0, 0], per_request_token_counts=[5, 1, 2])
    assert n == 8
    bi = manager.batch_info
    assert bi.bs == 3
    assert bi.max_len == 5
    torch.cuda.synchronize()
    assert bi.seg_lens[:3].tolist() == [5, 1, 2]
    assert bi.seg_indptr[:4].tolist() == [0, 5, 6, 8]


def test_prepare_loras_unknown_id_falls_back_to_slot_zero(manager):
    n = manager.prepare_loras([99], per_request_token_counts=2)
    assert n == 2
    torch.cuda.synchronize()
    assert manager.batch_info.weight_indices[:1].tolist() == [0]


def test_prepare_loras_overflow_raises(manager):
    with pytest.raises(ValueError, match="overflow"):
        manager.prepare_loras([0] * 33, per_request_token_counts=2)


def test_prepare_loras_mismatched_lengths_raises(manager):
    with pytest.raises(ValueError, match="length"):
        manager.prepare_loras([0, 0], per_request_token_counts=[1, 2, 3])


def test_no_adapter_slot_has_zero_rank_and_scaling(manager):
    # Slot 0 stays at rank 0 / scaling 0 forever — it's the no-op sentinel
    # the Triton kernels short-circuit on.
    torch.cuda.synchronize()
    assert manager.batch_info.lora_ranks[0].item() == 0
    assert manager.batch_info.scalings[0].item() == 0.0
