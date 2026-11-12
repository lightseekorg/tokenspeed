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

"""Persistent cache storage remains registerable with expandable temporaries."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, suite="runtime-1gpu")

from test.runtime.cache_pool_test_utils import make_mha_memory_plan, plan_group_specs

from tokenspeed.runtime.layers.attention.kv_cache.arena import (
    CacheArena,
    _cache_allocation,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA allocator"
)


@pytest.fixture
def allocator():
    torch.empty(1, device="cuda")
    original_settings = settings()
    original = original_settings["PYTORCH_CUDA_ALLOC_CONF"]
    try:
        yield
    finally:
        enabled = original_settings["expandable_segments"]
        torch.cuda.memory._set_allocator_settings(f"expandable_segments:{enabled}")
        torch.cuda.memory._set_allocator_settings(original)


def settings():
    return torch.cuda.memory._snapshot()["allocator_settings"]


def expandable(tensor):
    return next(
        segment["is_expandable"]
        for segment in torch.cuda.memory_snapshot()
        if segment["address"]
        <= tensor.data_ptr()
        < segment["address"] + segment["total_size"]
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_arena_storage_and_following_allocations(allocator, enabled):
    torch.cuda.memory._set_allocator_settings(
        f"expandable_segments:{enabled},max_split_size_mb:128"
    )
    before = settings()
    plan = make_mha_memory_plan(
        size=65536,
        prefix_granularity=16,
        layer_num=1,
        kv_heads=1,
        head_dim=128,
        dtype=torch.bfloat16,
        layer_types=("full_attention",),
    )
    arena = CacheArena(plan, "cuda", cache_group_specs=plan_group_specs(plan))
    assert not expandable(arena.buffer)
    assert settings() == before
    assert torch.count_nonzero(arena.buffer).item() == 0
    # Hold the arena live; a fresh stream prevents reuse of legacy cached blocks.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        temporary = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    assert expandable(temporary) == enabled
    stream.synchronize()


def test_failed_and_nested_scope_restores_allocator(allocator):
    torch.cuda.memory._set_allocator_settings(
        "expandable_segments:True,max_split_size_mb:128"
    )
    before = settings()
    with pytest.raises(RuntimeError, match="allocation scope failed"):
        with _cache_allocation("cuda"):
            assert not settings()["expandable_segments"]
            with _cache_allocation("cuda"):
                storage = torch.empty(8 * 1024 * 1024, device="cuda", dtype=torch.uint8)
            assert not expandable(storage)
            raise RuntimeError("allocation scope failed")
    assert settings() == before


def test_sticky_expandable_setting_is_restored(allocator):
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.cuda.memory._set_allocator_settings("max_split_size_mb:128")
    before = settings()
    assert before["expandable_segments"]
    with _cache_allocation("cuda"):
        assert not settings()["expandable_segments"]
    assert settings() == before


def test_async_allocator_arena_in_fresh_process():
    env = dict(os.environ)
    for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        env[key] = "backend:cudaMallocAsync"
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import torch
from test.runtime.cache_pool_test_utils import make_mha_memory_plan, plan_group_specs
from tokenspeed.runtime.layers.attention.kv_cache.arena import CacheArena
assert torch.cuda.memory.get_allocator_backend() == "cudaMallocAsync"
plan = make_mha_memory_plan(size=32, prefix_granularity=16, layer_num=1,
    kv_heads=1, head_dim=128, dtype=torch.bfloat16, layer_types=("full_attention",))
arena = CacheArena(plan, "cuda", cache_group_specs=plan_group_specs(plan))
assert torch.count_nonzero(arena.buffer).item() == 0
arena.buffer.fill_(13)
torch.cuda.synchronize()
assert arena.buffer[0].item() == 13
""",
        ],
        env=env,
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        timeout=120,
    )
