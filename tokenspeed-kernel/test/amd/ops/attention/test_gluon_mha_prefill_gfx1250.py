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

import math

import pytest
import torch
from utils import is_cdna5

if not is_cdna5():
    pytest.skip(
        "AMD CDNA5 is required for gfx1250 Gluon MHA tests", allow_module_level=True
    )


from tokenspeed_kernel_amd.ops.gfx1250.attention.mha import prefill  # noqa: E402


def _inputs(seqlens, n_q_heads, n_kv_heads, head_dim, device, dtype):
    cu_cpu = [0]
    for s in seqlens:
        cu_cpu.append(cu_cpu[-1] + s)
    cu = torch.tensor(cu_cpu, device=device, dtype=torch.int32)
    total = cu_cpu[-1]
    q = torch.randn((total, n_q_heads, head_dim), device=device, dtype=dtype)
    k = torch.randn((total, n_kv_heads, head_dim), device=device, dtype=dtype)
    v = torch.randn((total, n_kv_heads, head_dim), device=device, dtype=dtype)
    return q, k, v, cu, cu_cpu, max(seqlens)


def _reference(q, k, v, cu_cpu, n_q_heads, n_kv_heads, head_dim, window_left=-1):
    sm_scale = 1.0 / math.sqrt(head_dim)
    group = n_q_heads // n_kv_heads
    outs = []
    for start, end in zip(cu_cpu[:-1], cu_cpu[1:]):
        q_i = q[start:end].float()
        k_exp = k[start:end].float().repeat_interleave(group, dim=1)
        v_exp = v[start:end].float().repeat_interleave(group, dim=1)
        n = end - start
        scores = torch.einsum("qhd,khd->hqk", q_i, k_exp) * sm_scale
        pos = torch.arange(n, device=q.device)
        mask = pos[:, None] >= pos[None, :]
        if window_left >= 0:
            mask &= (pos[:, None] - pos[None, :]) <= window_left
        scores = scores.masked_fill(~mask[None, :, :], float("-inf"))
        outs.append(torch.einsum("hqk,khd->qhd", torch.softmax(scores, dim=-1), v_exp))
    return torch.cat(outs, dim=0)


@pytest.mark.parametrize(
    "block_m,num_warps", [(128, 4), (256, 8)], ids=["narrow", "wide"]
)
@pytest.mark.parametrize("head_dim", [64, 128], ids=["d64", "d128"])
@pytest.mark.parametrize("window_left", [-1, 64], ids=["full", "sliding"])
def test_mha_prefill_tile_shapes(block_m, num_warps, head_dim, window_left):
    """Compile and check both prefill tile shapes.

    get_config() picks between these by shape, and the wide tile only triggers
    on a grid large enough to fill the device, which is more work than a compute
    simulator can run. So force each configuration on a small shape instead: the
    point is to exercise the distinct WMMA and store layouts and the launch
    arguments, which is where a layout regression would show up.
    """
    device, dtype = "cuda", torch.bfloat16
    n_q_heads, n_kv_heads = 4, 1
    # D=128 full attention crosses the max-ILP scheduler's minimum sequence
    # length, so this matrix compiles both scheduler policies as well.
    seqlen = 512 if head_dim == 128 and window_left < 0 else 320
    q, k, v, cu, cu_cpu, max_seqlen = _inputs(
        [seqlen], n_q_heads, n_kv_heads, head_dim, device, dtype
    )

    original = prefill.get_config
    used = []

    def forced(**kwargs):
        cfg = original(**kwargs)
        forced_cfg = cfg._replace(
            block_m=block_m,
            num_warps=num_warps,
            grid=(
                cfg.batch_size,
                cfg.n_heads,
                (cfg.max_seqlen + block_m - 1) // block_m,
            ),
        )
        used.append(forced_cfg)
        return forced_cfg

    prefill.get_config = forced
    try:
        out = prefill.launch_gluon_mha_prefill_gfx1250(
            q, k, v, cu, cu_cpu, max_seqlen, window_left=window_left
        )
    finally:
        prefill.get_config = original

    # Without this the test would still pass if the override silently missed and
    # the kernel ran the other tile, reporting coverage it does not have.
    assert used, "get_config was not called; the tile override did not take effect"
    assert (used[-1].block_m, used[-1].num_warps) == (block_m, num_warps)

    assert out.shape == q.shape
    assert not torch.isnan(out).any()
    expected = _reference(q, k, v, cu_cpu, n_q_heads, n_kv_heads, head_dim, window_left)
    torch.testing.assert_close(out.float(), expected, rtol=8e-2, atol=8e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_mha_prefill_selects_deep_pipeline(dtype, head_dim):
    """Check that a full, sufficiently occupied launch selects the deep path."""
    device = "cuda"
    seqlens = [2048] * 4 if head_dim == 64 else [1024] * 8
    n_q_heads, n_kv_heads = 8, 2
    q, k, v, cu, cu_cpu, max_seqlen = _inputs(
        seqlens, n_q_heads, n_kv_heads, head_dim, device, dtype
    )

    original_selector = prefill._select_deep_pipeline
    selected = []

    def capture_selection(**kwargs):
        result = original_selector(**kwargs)
        selected.append(result)
        return result

    prefill._select_deep_pipeline = capture_selection
    try:
        out = prefill.launch_gluon_mha_prefill_gfx1250(q, k, v, cu, cu_cpu, max_seqlen)
    finally:
        prefill._select_deep_pipeline = original_selector

    assert selected == [True]
    expected = _reference(q, k, v, cu_cpu, n_q_heads, n_kv_heads, head_dim)
    torch.testing.assert_close(out.float(), expected, rtol=8e-2, atol=8e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("group_size", [2, 4, 8])
def test_mha_prefill_packed_gqa(dtype, group_size):
    """Check packed (sequence, query-head) rows across GQA group sizes."""
    device = "cuda"
    seqlens = [257, 129]
    n_q_heads, n_kv_heads, head_dim = group_size, 1, 128
    q, k, v, cu, cu_cpu, max_seqlen = _inputs(
        seqlens, n_q_heads, n_kv_heads, head_dim, device, dtype
    )

    original_config = prefill.get_config

    def packed_config(**kwargs):
        cfg = original_config(**kwargs)
        block_m, num_warps = 128, 4
        return cfg._replace(
            block_m=block_m,
            num_warps=num_warps,
            packed_gqa=True,
            grid=(
                cfg.batch_size,
                cfg.n_kv_heads,
                prefill.triton_cdiv(cfg.max_seqlen * group_size, block_m),
            ),
        )

    prefill.get_config = packed_config
    try:
        out = prefill.launch_gluon_mha_prefill_gfx1250(q, k, v, cu, cu_cpu, max_seqlen)
    finally:
        prefill.get_config = original_config

    expected = _reference(q, k, v, cu_cpu, n_q_heads, n_kv_heads, head_dim)
    torch.testing.assert_close(out.float(), expected, rtol=8e-2, atol=8e-2)


def test_mha_prefill_addresses_past_four_gib():
    """Sequence and head bases must not wrap through 32-bit buffer offsets."""
    device, dtype = "cuda", torch.bfloat16
    tokens, n_heads, head_dim = 2, 17, 64
    head_stride = 2**27
    boundary = 2**31
    storage_elements = boundary + tokens * head_dim
    storage_bytes = storage_elements * torch.tensor([], dtype=dtype).element_size()
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if free_bytes < storage_bytes + 512 * 1024**2:
        pytest.skip("large-offset regression requires about 4.5 GiB free")

    storage = torch.empty(storage_elements, dtype=dtype, device=device)
    qkv = storage.as_strided(
        (tokens, n_heads, head_dim),
        (head_dim, head_stride, 1),
    )
    assert prefill._requires_wide_addressing(qkv)
    generator = torch.Generator(device=device).manual_seed(20261002)
    compact = torch.randn(qkv.shape, dtype=dtype, device=device, generator=generator)
    qkv.copy_(compact)
    cu_cpu = [0, tokens]
    cu = torch.tensor(cu_cpu, dtype=torch.int32, device=device)
    out = prefill.launch_gluon_mha_prefill_gfx1250(qkv, qkv, qkv, cu, cu_cpu, tokens)
    expected = _reference(compact, compact, compact, cu_cpu, n_heads, n_heads, head_dim)
    torch.testing.assert_close(out.float(), expected, rtol=8e-2, atol=8e-2)

    # Exercise a sequence/output base at exactly 4 GiB with one sparse allocation.
    base_token = boundary // head_dim
    sentinel = torch.full((head_dim,), -7.0, dtype=dtype, device=device)
    value = torch.randn((head_dim,), dtype=dtype, device=device, generator=generator)
    storage[:head_dim].copy_(sentinel)
    storage[boundary : boundary + head_dim].copy_(value)
    large_cu = torch.tensor(
        [base_token, base_token + 1], dtype=torch.int32, device=device
    )
    prefill.gluon_mha_prefill_gfx1250[(1, 1, 1)](
        storage,
        storage,
        storage,
        large_cu,
        storage,
        storage,
        storage,
        head_dim,
        head_dim,
        1,
        head_dim,
        head_dim,
        1,
        head_dim,
        head_dim,
        1,
        1,
        1,
        head_dim,
        (1.0 / math.sqrt(head_dim)) * prefill._INV_LN2_VALUE,
        128,
        64,
        False,
        False,
        False,
        -1,
        False,
        False,
        False,
        False,
        False,
        True,
        4,
        2,
        num_warps=4,
        waves_per_eu=1,
        llvm_fn_attrs="",
    )
    torch.cuda.synchronize()
    assert torch.equal(storage[:head_dim], sentinel)
    torch.testing.assert_close(
        storage[boundary : boundary + head_dim].float(),
        value.float(),
        rtol=8e-2,
        atol=8e-2,
    )


def test_select_llvm_fn_attrs():
    max_ilp = "amdgpu-sched-strategy=max-ilp"

    assert (
        prefill._select_llvm_fn_attrs(head_dim=128, max_seqlen=512, window_left=-1)
        == max_ilp
    )

    # Each guard has a measured regression when max-ILP is used.
    assert (
        prefill._select_llvm_fn_attrs(head_dim=64, max_seqlen=512, window_left=-1) == ""
    )
    assert (
        prefill._select_llvm_fn_attrs(head_dim=128, max_seqlen=256, window_left=-1)
        == ""
    )
    assert (
        prefill._select_llvm_fn_attrs(head_dim=128, max_seqlen=4096, window_left=512)
        == ""
    )


def test_select_deep_pipeline():
    kwargs = {
        "dtype": torch.bfloat16,
        "head_dim": 128,
        "block_m": 256,
        "block_n": 64,
        "num_warps": 8,
        "num_buffers": 2,
        "window_left": -1,
        "workgroups": 256,
        "min_positive_seqlen": 129,
        "seqlens": [4096] * 4,
        "max_seqlen": 4096,
    }
    assert prefill._select_deep_pipeline(**kwargs)
    assert prefill._select_deep_pipeline(
        **(kwargs | {"dtype": torch.float16, "head_dim": 64})
    )
    short_kwargs = kwargs | {
        "min_positive_seqlen": 1024,
        "seqlens": [1024] * 8,
        "max_seqlen": 1024,
    }
    assert prefill._select_deep_pipeline(**short_kwargs)
    assert not prefill._select_deep_pipeline(**(short_kwargs | {"head_dim": 64}))

    for override in (
        {"dtype": torch.float8_e4m3fn},
        {"block_m": 128},
        {"window_left": 64},
        {"workgroups": 255},
        {"seqlens": [4096, 3840]},
        {"seqlens": [4097] * 4, "max_seqlen": 4097},
    ):
        assert not prefill._select_deep_pipeline(**(kwargs | override))


def test_select_packed_gqa():
    kwargs = {
        "dtype": torch.bfloat16,
        "head_dim": 128,
        "n_heads": 8,
        "n_kv_heads": 1,
        "seqlens": [1024] * 8,
        "max_seqlen": 1024,
        "window_left": -1,
        "has_sink": False,
        "has_lse": False,
        "packed_q_block_bytes": 1024 * 1024,
    }
    assert prefill._select_packed_gqa(**kwargs)
    assert prefill._select_packed_gqa(
        **(
            kwargs
            | {
                "seqlens": [4096, 3584, 2305, 1024],
                "max_seqlen": 4096,
            }
        )
    )
    assert prefill._select_packed_gqa(
        **(
            kwargs
            | {
                "seqlens": [4096] * 4,
                "max_seqlen": 4096,
                "window_left": 512,
            }
        )
    )
    assert prefill._select_packed_gqa(
        **(kwargs | {"seqlens": [4096] * 4, "max_seqlen": 4096})
    )
    assert prefill._select_packed_gqa(
        **(kwargs | {"seqlens": [8192] * 2, "max_seqlen": 8192})
    )

    for override in (
        {"dtype": torch.float8_e4m3fn},
        {"head_dim": 64},
        {"n_heads": 32, "n_kv_heads": 8},
        {
            "dtype": torch.float16,
            "seqlens": [4096] * 4,
            "max_seqlen": 4096,
        },
        {"has_sink": True},
        {"packed_q_block_bytes": 2**32 + 1},
    ):
        assert not prefill._select_packed_gqa(**(kwargs | override))


def test_select_tdm_warp_hint():
    kwargs = {
        "block_m": 256,
        "block_n": 64,
        "num_warps": 8,
        "window_left": -1,
        "workgroups": 256,
    }
    assert prefill._select_tdm_warp_hint(**kwargs)

    for override in (
        {"block_m": 128},
        {"block_n": 128},
        {"num_warps": 4},
        {"window_left": 64},
        {"workgroups": 255},
    ):
        assert not prefill._select_tdm_warp_hint(**(kwargs | override))


def test_select_reverse_q_blocks():
    kwargs = {
        "block_m": 256,
        "max_seqlen": 4096,
        "window_left": -1,
        "workgroups": 256,
    }
    assert prefill._select_reverse_q_blocks(**kwargs)

    for override in (
        {"max_seqlen": 256},
        {"window_left": 64},
        {"workgroups": 255},
    ):
        assert not prefill._select_reverse_q_blocks(**(kwargs | override))


def test_mha_prefill_reverse_counts_live_ragged_workgroups():
    device, dtype = "cuda", torch.bfloat16
    seqlens = [4096] + [1] * 31
    n_q_heads, n_kv_heads, head_dim = 1, 1, 64
    q, k, v, cu, cu_cpu, max_seqlen = _inputs(
        seqlens, n_q_heads, n_kv_heads, head_dim, device, dtype
    )
    assert len(seqlens) * n_q_heads * prefill.triton_cdiv(max_seqlen, 256) == 512

    original_order = prefill._select_reverse_q_blocks
    observed_workgroups = []

    def capture_workgroups(**kwargs):
        observed_workgroups.append(kwargs["workgroups"])
        return False

    prefill._select_reverse_q_blocks = capture_workgroups
    try:
        out = prefill.launch_gluon_mha_prefill_gfx1250(q, k, v, cu, cu_cpu, max_seqlen)
    finally:
        prefill._select_reverse_q_blocks = original_order

    assert observed_workgroups == [47]
    assert out.shape == q.shape
    assert not torch.isnan(out).any()


def test_mha_prefill_tdm_warp_hint_remainder():
    device, dtype = "cuda", torch.bfloat16
    n_q_heads, n_kv_heads, head_dim = 8, 2, 128
    q, k, v, cu, cu_cpu, max_seqlen = _inputs(
        [300, 513], n_q_heads, n_kv_heads, head_dim, device, dtype
    )

    original_config = prefill.get_config
    original_hint = prefill._select_tdm_warp_hint

    def forced_config(**kwargs):
        cfg = original_config(**kwargs)
        return cfg._replace(
            block_m=256,
            num_warps=8,
            grid=(
                cfg.batch_size,
                cfg.n_heads,
                (cfg.max_seqlen + 255) // 256,
            ),
        )

    prefill.get_config = forced_config
    try:
        prefill._select_tdm_warp_hint = lambda **_kwargs: False
        control = prefill.launch_gluon_mha_prefill_gfx1250(
            q, k, v, cu, cu_cpu, max_seqlen
        )
        prefill._select_tdm_warp_hint = lambda **_kwargs: True
        out = prefill.launch_gluon_mha_prefill_gfx1250(q, k, v, cu, cu_cpu, max_seqlen)
    finally:
        prefill.get_config = original_config
        prefill._select_tdm_warp_hint = original_hint

    assert torch.equal(out, control)
    expected = _reference(q, k, v, cu_cpu, n_q_heads, n_kv_heads, head_dim)
    torch.testing.assert_close(out.float(), expected, rtol=8e-2, atol=8e-2)


def test_mha_prefill_reverse_q_blocks_ragged():
    device, dtype = "cuda", torch.bfloat16
    n_q_heads, n_kv_heads, head_dim = 8, 2, 128
    q, k, v, cu, cu_cpu, max_seqlen = _inputs(
        [300, 513], n_q_heads, n_kv_heads, head_dim, device, dtype
    )

    original_config = prefill.get_config
    original_order = prefill._select_reverse_q_blocks

    def forced_config(**kwargs):
        cfg = original_config(**kwargs)
        return cfg._replace(
            block_m=256,
            num_warps=8,
            grid=(
                cfg.batch_size,
                cfg.n_heads,
                (cfg.max_seqlen + 255) // 256,
            ),
        )

    prefill.get_config = forced_config
    try:
        prefill._select_reverse_q_blocks = lambda **_kwargs: False
        control = prefill.launch_gluon_mha_prefill_gfx1250(
            q, k, v, cu, cu_cpu, max_seqlen
        )
        prefill._select_reverse_q_blocks = lambda **_kwargs: True
        out = prefill.launch_gluon_mha_prefill_gfx1250(q, k, v, cu, cu_cpu, max_seqlen)
    finally:
        prefill.get_config = original_config
        prefill._select_reverse_q_blocks = original_order

    assert torch.equal(out, control)
    expected = _reference(q, k, v, cu_cpu, n_q_heads, n_kv_heads, head_dim)
    torch.testing.assert_close(out.float(), expected, rtol=8e-2, atol=8e-2)


def test_select_m_tile_gates():
    """Both gates on the wide tile are load-bearing.

    Measured on gfx1250, taking the 256-row tile when either gate fails costs up
    to 1.2x, so pin the behaviour at each boundary.
    """
    wide = (256, 8)
    narrow = (128, 4)

    # Sequence too short: the causally-masked half of the diagonal block is a
    # large fraction of the work, even though this grid fills the device.
    assert prefill._select_m_tile(batch_size=8, n_heads=32, max_seqlen=512) == narrow

    # Long enough, but 128 workgroups underfills the 256 CUs.
    assert prefill._select_m_tile(batch_size=1, n_heads=8, max_seqlen=4096) == narrow

    # Both satisfied.
    assert prefill._select_m_tile(batch_size=4, n_heads=32, max_seqlen=4096) == wide
