"""FlatKV ``mla_fp8_commit_view`` + fused fp8 query/KV-commit parity.

The NoPE fp8 decode path folds the latent KV write into the query fp8
assembly launch (``mla_nope_query_kv_fp8``). Coverage:

- the pool hands out its flat fp8 view only where the fused write is legal
  (fp8 latent binding on CUDA; None on KDA layers so callers fall back);
- committing through the fused kernel writes byte-identical rows to the
  ``set_mla_kv_buffer`` path it replaces (including NaN/Inf sanitize).
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))
from test.runtime.conftest import MLA_KV_LORA_RANK as _KV_LORA_RANK
from test.runtime.conftest import MLA_LATENT_DIM as _LATENT_DIM
from test.runtime.conftest import kda_layer_id as _kda_layer_id
from test.runtime.conftest import make_kimi_pool as _make_pool
from test.runtime.conftest import mla_layer_id as _mla_layer_id
from test.runtime.conftest import requires_cuda

from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="runtime-1gpu")


class _FakeLayer:
    def __init__(self, layer_id: int) -> None:
        self.layer_id = layer_id


@requires_cuda
def test_commit_view_is_the_flat_fp8_view() -> None:
    pool = _make_pool("cuda", usable_pages=2)
    layer = _FakeLayer(_mla_layer_id(pool))
    loc = torch.tensor([pool.page_size + 3], device="cuda", dtype=torch.int64)

    view = pool.mla_fp8_commit_view(layer, loc)

    assert view is not None
    assert view.dtype == torch.float8_e4m3fn
    assert view.shape[-1] == _LATENT_DIM
    assert view.data_ptr() == pool.get_key_buffer(layer.layer_id).data_ptr()


@requires_cuda
def test_commit_view_none_for_non_mla_layer() -> None:
    pool = _make_pool("cuda", usable_pages=2)
    layer = _FakeLayer(_kda_layer_id(pool))
    loc = torch.tensor([pool.page_size], device="cuda", dtype=torch.int64)

    assert pool.mla_fp8_commit_view(layer, loc) is None


def test_commit_view_none_on_cpu_pool() -> None:
    pool = _make_pool("cpu", usable_pages=2)
    layer = _FakeLayer(_mla_layer_id(pool))
    loc = torch.tensor([pool.page_size], dtype=torch.int64)

    assert pool.mla_fp8_commit_view(layer, loc) is None


@requires_cuda
@pytest.mark.parametrize("poison", [False, True])
def test_fused_commit_matches_set_mla_kv_buffer(poison: bool) -> None:
    from tokenspeed_kernel.ops.attention.triton.mla_query_kv_fp8 import (
        mla_nope_query_kv_fp8,
    )

    torch.manual_seed(0)
    pool_ref = _make_pool("cuda", usable_pages=3)
    pool_new = _make_pool("cuda", usable_pages=3)
    layer = _FakeLayer(_mla_layer_id(pool_ref))
    page_size = pool_ref.page_size

    T, H = 2, 4
    locs = torch.tensor(
        [1 * page_size + (page_size - 1), 2 * page_size],
        device="cuda",
        dtype=torch.int64,
    )
    latent = torch.randn(T, 1, _LATENT_DIM, device="cuda", dtype=torch.bfloat16)
    if poison:
        latent[0, 0, 0] = float("nan")
        latent[1, 0, _KV_LORA_RANK + 1] = float("inf")
    q_nope = torch.randn(T, H, _KV_LORA_RANK, device="cuda", dtype=torch.bfloat16)
    q_pe = torch.randn(
        T, H, _LATENT_DIM - _KV_LORA_RANK, device="cuda", dtype=torch.bfloat16
    )

    # Reference: the write path the fused kernel replaces (sanitize defaults on).
    pool_ref.set_mla_kv_buffer(
        layer, locs, latent[..., :_KV_LORA_RANK], latent[..., _KV_LORA_RANK:]
    )

    view = pool_new.mla_fp8_commit_view(layer, locs)
    assert view is not None
    mla_nope_query_kv_fp8(q_nope, q_pe, latent, view, locs, sanitize=True)
    torch.cuda.synchronize()

    slot = pool_ref.physical_slot_for_layer(layer.layer_id)
    assert torch.equal(pool_ref.raw_slab(slot), pool_new.raw_slab(slot))
