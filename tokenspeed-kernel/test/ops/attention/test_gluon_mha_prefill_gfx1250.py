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
    q, k, v, cu, cu_cpu, max_seqlen = _inputs(
        [320], n_q_heads, n_kv_heads, head_dim, device, dtype
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
        out = prefill.gluon_mha_prefill_gfx1250(
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


def test_select_m_tile_scales_with_device_cus():
    """The wide tile needs both a long sequence and a grid that fills the device,
    so the cutoff has to follow the CU count actually reported."""
    wide = (256, 8)
    narrow = (128, 4)

    # Short sequences pay too much of the causally-masked diagonal block.
    assert (
        prefill._select_m_tile(batch_size=8, n_heads=32, max_seqlen=512, num_cus=256)
        == narrow
    )

    # 128 workgroups underfills 256 CUs but fills a 128-CU partition.
    assert (
        prefill._select_m_tile(batch_size=1, n_heads=8, max_seqlen=4096, num_cus=256)
        == narrow
    )
    assert (
        prefill._select_m_tile(batch_size=1, n_heads=8, max_seqlen=4096, num_cus=128)
        == wide
    )

    assert (
        prefill._select_m_tile(batch_size=4, n_heads=32, max_seqlen=4096, num_cus=256)
        == wide
    )
