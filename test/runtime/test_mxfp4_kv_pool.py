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


"""The MXFP4 MHA KV pool (``--kv-cache-dtype mxfp4``).

The pool sizes 34-byte rows per head (head_dim 64: 32 E2M1 bytes + 2 UE8M0
scales). Once the attention backend binds its kernel page size, each layer's
fields are MXFP4 kernel pages: the prologue's write target is their data
planes with the scale planes as its scales, and ``set_kv_buffer`` quantizes
into the same pages. Checks the sizing, the binding and the store against the
torch codec.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.layers.attention.configs.base import (
    mxfp4_row_bytes,
    resolve_dtype,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)

HEADS = 2
HEAD_DIM = 64
ROW = HEAD_DIM // 2 + HEAD_DIM // 32
LAYERS = 2
GRANULARITY = 128


def _codec():
    path = (
        Path(__file__).resolve().parents[2]
        / "tokenspeed-kernel"
        / "test"
        / "mxfp4_kv_reference.py"
    )
    spec = importlib.util.spec_from_file_location("mxfp4_kv_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_pool(size: int = 512):
    from test.runtime.cache_pool_test_utils import make_arena, make_mha_memory_plan

    from tokenspeed.runtime.layers.attention.kv_cache.mha import (
        MHATokenToKVPoolMXFP4,
    )

    arena = make_arena(
        make_mha_memory_plan(
            size=size,
            prefix_granularity=GRANULARITY,
            layer_num=LAYERS,
            kv_heads=HEADS,
            head_dim=HEAD_DIM,
            dtype=torch.bfloat16,
            mxfp4=True,
        )
    )
    return MHATokenToKVPoolMXFP4(
        arena,
        dtype=torch.bfloat16,
        head_num=HEADS,
        head_dim=HEAD_DIM,
        layer_num=LAYERS,
        rank=0,
    )


def _bind(pool, kernel_page: int) -> None:
    for group in set(pool.history_group_by_layer().values()):
        pool.atom_pages_per_block(group, kernel_page, write_slot_dtype=torch.int32)


def test_row_bytes_and_dtype():
    assert mxfp4_row_bytes(64) == 34
    assert mxfp4_row_bytes(128) == 68
    assert resolve_dtype("mxfp4") == torch.uint8


def test_size_accounting():
    pool = _make_pool()
    k_size, v_size = pool.get_kv_size_bytes()
    slots = pool.arena.size + pool.arena.prefix_granularity
    assert k_size == v_size == slots * HEADS * ROW * LAYERS


def test_planes_need_the_kernel_page_size():
    pool = _make_pool()
    slots = torch.arange(4, device="cuda")
    with pytest.raises(RuntimeError, match="page size"):
        pool.kv_write_target(0, slots, None)
    _bind(pool, 64)
    group = pool.history_group_by_layer()[0]
    with pytest.raises(ValueError, match="MXFP4 pages of 64 rows"):
        pool.atom_pages_per_block(group, 128, write_slot_dtype=torch.int32)


@pytest.mark.parametrize("kernel_page", [64, 128])
def test_write_target_is_the_mxfp4_planes(kernel_page):
    from tokenspeed_kernel.ops.attention.prologue import KVCacheFormat

    pool = _make_pool()
    _bind(pool, kernel_page)
    slots = torch.randperm(pool.arena.size, device="cuda")[:40]
    target = pool.kv_write_target(0, slots, None)
    assert target.format is KVCacheFormat.MXFP4
    assert target.geometry == (HEADS, HEAD_DIM)
    assert target.slots is slots
    data, scales = target.k_cache, target.scales.k
    assert data.shape[1:] == (kernel_page, HEADS, HEAD_DIM // 2)
    assert scales.shape[1:] == (kernel_page, HEADS, HEAD_DIM // 32)
    assert data.stride(0) == scales.stride(0) == kernel_page * HEADS * ROW
    assert data.data_ptr() == pool.get_key_buffer(0).data_ptr()


@pytest.mark.parametrize("kernel_page", [64, 128])
def test_set_kv_buffer_matches_the_codec(kernel_page):
    codec = _codec()
    torch.manual_seed(0)
    pool = _make_pool()
    _bind(pool, kernel_page)
    t = 96
    k = torch.randn(t, HEADS * HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 3
    v = torch.randn(t, HEADS * HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    loc = torch.randperm(pool.arena.size, device="cuda")[:t]
    pool.set_kv_buffer(SimpleNamespace(layer_id=1), loc, k, v)

    target = pool.kv_write_target(1, loc, None)
    page, row = loc // kernel_page, loc % kernel_page
    for x, data, scales in (
        (k, target.k_cache, target.scales.k),
        (v, target.v_cache, target.scales.v),
    ):
        want_data, want_scales = codec.mxfp4_quantize(x.view(t, HEADS, HEAD_DIM))
        assert torch.equal(data[page, row], want_data)
        assert torch.equal(scales[page, row], want_scales)
    # The other layer is untouched.
    assert not pool.get_key_buffer(0).any()
