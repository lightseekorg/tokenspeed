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


def test_has_active_lora_flag(manager):
    # All-base batch → flag is False.  CudaGraphWrapper uses this to pick
    # the no-LoRA captured graph variant (skip the per-step Triton kernels).
    manager.prepare_loras([0, 0, 0])
    assert manager.has_active_lora is False
    # Unknown id falls back to slot 0 → still no active adapter.
    manager.prepare_loras([99])
    assert manager.has_active_lora is False


# ──────────────────────────────────────────────────────────────────────────
# Tiered GPU↔CPU↔disk pool tests.  These don't actually do GEMMs, just
# verify the residence + eviction bookkeeping under various loads.
# ──────────────────────────────────────────────────────────────────────────


def _write_dummy_adapter(tmp_path, rank: int, hidden: int, n_layers: int) -> str:
    """Write a minimal PEFT-style adapter under tmp_path/adapter_X."""
    import json

    from safetensors.torch import save_file

    tensors = {}
    for layer in range(n_layers):
        for mod in ("q_proj", "k_proj", "v_proj", "o_proj"):
            base = f"base_model.model.model.layers.{layer}.self_attn.{mod}"
            tensors[f"{base}.lora_A.weight"] = torch.randn(
                rank, hidden, dtype=torch.float32
            )
            tensors[f"{base}.lora_B.weight"] = torch.randn(
                hidden, rank, dtype=torch.float32
            )
    save_file(tensors, str(tmp_path / "adapter_model.safetensors"))
    cfg = {
        "r": rank,
        "lora_alpha": rank,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }
    (tmp_path / "adapter_config.json").write_text(json.dumps(cfg))
    return str(tmp_path)


@pytest.fixture
def adapter_paths(tmp_path):
    """Create 4 dummy adapters on disk."""
    paths = {}
    for i in range(4):
        d = tmp_path / f"adapter_{i}"
        d.mkdir()
        paths[f"a{i}"] = _write_dummy_adapter(d, rank=8, hidden=32, n_layers=2)
    return paths


def _tiered_manager(max_loras_cpu: int) -> LoraManager:
    return LoraManager(
        model_config=_model_config(),
        max_loras=2,
        max_lora_rank=8,
        max_num_tokens=64,
        max_loras_cpu=max_loras_cpu,
        dtype=torch.float16,
        device=torch.device("cuda:0"),
    )


def test_max_loras_cpu_ge_max_loras(adapter_paths):
    if not torch.cuda.is_available():
        pytest.skip("LoraManager allocates GPU buffers")
    with pytest.raises(ValueError, match="max_loras_cpu"):
        _tiered_manager(max_loras_cpu=1)  # max_loras=2 in fixture


def test_load_adapter_warms_cpu_pool(adapter_paths):
    if not torch.cuda.is_available():
        pytest.skip("LoraManager allocates GPU buffers")
    m = _tiered_manager(max_loras_cpu=8)
    m.load_adapter("a0", adapter_paths["a0"])
    assert "a0" in m._cpu_cache
    assert "a0" not in m._name_to_slot  # not GPU-resident yet


def test_cpu_pool_lru_evicts_to_disk(adapter_paths):
    if not torch.cuda.is_available():
        pytest.skip("LoraManager allocates GPU buffers")
    # max_loras_cpu=2 → only 2 adapters fit in CPU at once.  Loading a
    # third evicts the LRU one back to disk.
    m = _tiered_manager(max_loras_cpu=2)
    for name in ("a0", "a1", "a2"):
        m.load_adapter(name, adapter_paths[name])
    # a0 was the LRU at the time a2 was loaded; should be evicted now.
    assert "a0" not in m._cpu_cache
    assert "a1" in m._cpu_cache
    assert "a2" in m._cpu_cache


def test_cpu_evicted_adapter_reloads_from_disk(adapter_paths):
    if not torch.cuda.is_available():
        pytest.skip("LoraManager allocates GPU buffers")
    m = _tiered_manager(max_loras_cpu=2)
    for name in ("a0", "a1", "a2"):
        m.load_adapter(name, adapter_paths[name])
    assert "a0" not in m._cpu_cache
    # Touching a0 again should reload it from disk into the CPU pool,
    # evicting whatever is now LRU.
    a0_id = m.get_id("a0")
    m.prepare_loras([a0_id])
    assert "a0" in m._cpu_cache
    assert "a0" in m._name_to_slot  # promoted to GPU too


def test_gpu_resident_evicted_only_when_no_alternative(adapter_paths):
    if not torch.cuda.is_available():
        pytest.skip("LoraManager allocates GPU buffers")
    # Prefer evicting non-GPU-resident entries first: they cost a disk
    # read to bring back, GPU-resident ones cost nothing until their
    # GPU slot is also evicted.
    m = _tiered_manager(max_loras_cpu=2)
    m.load_adapter("a0", adapter_paths["a0"])
    m.load_adapter("a1", adapter_paths["a1"])
    a0_id = m.get_id("a0")
    m.prepare_loras([a0_id])  # a0 → GPU; a1 stays CPU-only
    assert "a0" in m._name_to_slot
    # Loading a2: a1 (non-GPU) is evicted in preference to a0 (GPU).
    m.load_adapter("a2", adapter_paths["a2"])
    assert "a0" in m._cpu_cache
    assert "a1" not in m._cpu_cache
    assert "a2" in m._cpu_cache


def test_gpu_resident_can_be_cpu_evicted_when_pool_is_full(adapter_paths):
    if not torch.cuda.is_available():
        pytest.skip("LoraManager allocates GPU buffers")
    # max_loras=2 + max_loras_cpu=2 + two GPU-resident adapters: the
    # CPU pool MUST allow evicting GPU-resident entries to admit a
    # third adapter; otherwise the pool is permanently locked.
    m = _tiered_manager(max_loras_cpu=2)
    m.load_adapter("a0", adapter_paths["a0"])
    m.load_adapter("a1", adapter_paths["a1"])
    m.prepare_loras([m.get_id("a0"), m.get_id("a1")])  # both → GPU
    assert "a0" in m._name_to_slot
    assert "a1" in m._name_to_slot
    # Now register a2.  CPU pool is full and both entries are
    # GPU-resident — must evict one anyway (its GPU copy is still
    # valid; future reload costs a disk read).
    m.load_adapter("a2", adapter_paths["a2"])
    assert "a2" in m._cpu_cache
    # Exactly one of a0/a1 was kicked from the CPU pool.
    cpu_count = sum(name in m._cpu_cache for name in ("a0", "a1"))
    assert cpu_count == 1


def test_prefetch_warms_cpu_pool(adapter_paths):
    if not torch.cuda.is_available():
        pytest.skip("LoraManager allocates GPU buffers")
    m = _tiered_manager(max_loras_cpu=4)
    # Register two adapters but evict one.
    m.load_adapter("a0", adapter_paths["a0"])
    m.load_adapter("a1", adapter_paths["a1"])
    m._evict_from_cpu("a1")
    assert "a1" not in m._cpu_cache

    # prefetch kicks off async load; wait for it to finish.
    m.prefetch("a1")
    pending = m._pending_loads.get("a1")
    if pending is not None:
        pending.result()
    assert "a1" in m._cpu_cache


def test_prefetch_unknown_adapter_is_noop(adapter_paths):
    if not torch.cuda.is_available():
        pytest.skip("LoraManager allocates GPU buffers")
    m = _tiered_manager(max_loras_cpu=4)
    m.prefetch("never-registered")  # must not raise
    assert "never-registered" not in m._cpu_cache
    assert "never-registered" not in m._pending_loads


def test_unload_adapter_clears_both_tiers(adapter_paths):
    if not torch.cuda.is_available():
        pytest.skip("LoraManager allocates GPU buffers")
    m = _tiered_manager(max_loras_cpu=4)
    m.load_adapter("a0", adapter_paths["a0"])
    a0_id = m.get_id("a0")
    m.prepare_loras([a0_id])
    m.unload_adapter("a0")
    assert "a0" not in m._cpu_cache
    assert "a0" not in m._name_to_slot
    assert m.get_id("a0") is None
