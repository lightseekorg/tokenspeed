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

"""The PD contract must name the allocation actually read by attention."""

import ctypes
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=30, suite="runtime-1gpu")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
from tokenspeed.runtime.engine.scheduler_utils import _kv_pool_bytes
from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
from tokenspeed.runtime.layers.attention.kv_cache.offload_config import KVOffloadConfig
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    pack,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec
from tokenspeed.runtime.layers.attention.kv_cache.recipes.storage import (
    plan_cache_storage,
)
from tokenspeed.runtime.pd.cache_protocol import (
    CacheTransferContract,
    build_arena_cache_transfer_contract,
)


@pytest.mark.parametrize("prefix_granularity", [8, 16])
def test_segmented_host_device_contract_and_budget(prefix_granularity):
    group = CacheGroupSpec(
        group_id="full_attention",
        retention="full_history",
        rows_per_page=8,
        entry_stride_tokens=1,
        transfer_policy="full_suffix",
        replayable=False,
    )
    names = ("layer.0.latent_kv", "layer.1.latent_kv")
    layout = pack(
        ((group, tuple(CacheFieldSpec(n, n, (8, 1, 32), "bfloat16") for n in names)),),
        prefix_granularity=prefix_granularity,
        cache_blocks_per_lcm_block={"full_attention": prefix_granularity // 8},
        alignment=256,
        max_padding_fraction=0.25,
    )
    config = KVOffloadConfig(
        field_ids=(names[1],),
        hot_tokens=32,
        reserved_tokens=8,
        request_slots=3,
        topk=16,
        queries=1,
        host_budget_bytes=1 << 20,
        overlap=True,
        cyclic_tokens=0,
        selection_consumers=(),
        device_rows=512,
        max_extend_tokens=64,
    )
    arena = CacheArena(
        layout.bind(16),
        "cuda",
        cache_group_specs=(group,),
        storage_plan=plan_cache_storage(layout.bind(16), (group,), offload=config),
    )
    assert arena.field(names[0]).is_cuda
    assert arena.field(names[1]).is_pinned()
    contract, base = build_arena_cache_transfer_contract(arena)
    restored = CacheTransferContract.from_wire_bytes(contract.to_wire_bytes())
    assert restored == contract
    for name in names:
        assert restored.field_address(base, name) == arena.field(name).data_ptr()
    # A PD writer uses the advertised landing address, not the hot device view.
    source = torch.arange(32, dtype=torch.bfloat16)
    destination = restored.field_address(base, names[1]) + 8 * 32 * 2
    ctypes.memmove(destination, source.data_ptr(), source.numel() * 2)
    assert torch.equal(arena.field(names[1])[8, 0], source)
    # Sixteen LCM blocks plus the group's dummy page share the flat row view.
    assert arena.field(names[1]).shape == (16 * prefix_granularity + 8, 1, 32)
    assert arena.compute_field(names[1]).is_cuda
    assert (
        arena.compute_field(names[1]).shape[0]
        == arena.storage_plan.workspaces[0].device_rows
    )
    assert sum(n for _, n in arena.registration_regions()) == arena.plan.arena_bytes
    assert arena.allocated_device_bytes == (
        arena.storage_plan.device_history_bytes
        + arena.storage_plan.persistent_workspace_bytes
    )
    assert arena.allocated_host_bytes == arena.storage_plan.host_bytes
    # A compute view can omit layers and cannot count authoritative host
    # history as device memory. The arena counts its regions/workspace once.
    from types import SimpleNamespace

    target = SimpleNamespace(arena=arena)
    draft = SimpleNamespace(arena=arena)
    assert _kv_pool_bytes(target, draft) == arena.allocated_device_bytes
    # Region-bound zeroing must honor stream ordering for pinned host too.
    arena.zero_blocks({"full_attention": np.array([1], dtype=np.int32)})
    torch.cuda.synchronize()
    assert arena.field(names[1])[8:16].eq(0).all()
    device_pages = arena.field(names[0]).view(-1, 8, 1, 32)
    assert device_pages[1].eq(0).all()
    with pytest.raises(ValueError, match="complete plan"):
        replace(contract, field_addresses={names[1]: destination})
