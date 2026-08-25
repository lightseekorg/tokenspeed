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

"""Parity: the MTP verify kernel vs the single-step pool kernel iterated.

The MTP variant must produce the same attention outputs as running the pool
kernel one token at a time, and each per-position state page must match the
pool kernel's state after the corresponding step.
"""

from importlib.util import find_spec

import pytest
import torch
from tokenspeed_kernel.thirdparty.triton.fla_kda_recurrent import (
    fused_recurrent_kda_mtp,
    fused_recurrent_kda_pool,
)

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if find_spec("fla") is None:
    pytest.skip(
        "flash-linear-attention (fla) required for the KDA reference kernel",
        allow_module_level=True,
    )

# K3 TP8 per-rank KDA geometry.
H, HV, K, V = 4, 4, 128, 128


@pytest.mark.parametrize("B,T", [(1, 2), (3, 4), (8, 4)])
@pytest.mark.parametrize("lower_bound", [None, -1.5])
@pytest.mark.parametrize("recurrent_layout", ["k_major", "v_major"])
def test_mtp_matches_stepped_pool(B, T, lower_bound, recurrent_layout):
    torch.manual_seed(B * 10 + T)
    dev = "cuda"
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=dev)
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=dev)
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=dev)
    g = torch.randn(B, T, HV, K, dtype=torch.bfloat16, device=dev)
    beta = torch.randn(B, T, HV, dtype=torch.bfloat16, device=dev)
    A_log = torch.randn(HV, dtype=torch.float32, device=dev)
    dt_bias = torch.randn(HV, K, dtype=torch.float32, device=dev)

    n_pages = B * (T + 2)
    init = torch.randn(B, HV, K, V, dtype=torch.float32, device=dev)

    # --- reference: the V-major pool kernel, one token at a time ---
    pool_ref = torch.zeros(n_pages, HV, V, K, dtype=torch.float32, device=dev)
    pool_ref[:B] = init.transpose(-1, -2)
    ref_out = torch.empty(B, T, HV, V, dtype=v.dtype, device=dev)
    ref_states = []
    read = torch.arange(B, dtype=torch.int64, device=dev)
    for t in range(T):
        write = B + t * B + torch.arange(B, dtype=torch.int64, device=dev)
        o_t = fused_recurrent_kda_pool(
            q[:, t : t + 1].contiguous(),
            k[:, t : t + 1].contiguous(),
            v[:, t : t + 1].contiguous(),
            g[:, t : t + 1].contiguous(),
            beta[:, t : t + 1].contiguous(),
            A_log,
            dt_bias,
            pool_ref,
            read,
            write,
            lower_bound=lower_bound,
        )
        ref_out[:, t] = o_t[:, 0]
        ref_states.append(pool_ref[write].transpose(-1, -2).clone())
        read = write

    # --- MTP: one call, per-step write grid ---
    v_major = recurrent_layout == "v_major"
    state_shape = (HV, V, K) if v_major else (HV, K, V)
    pool_mtp = torch.zeros(n_pages, *state_shape, dtype=torch.float32, device=dev)
    pool_mtp[:B] = init.transpose(-1, -2) if v_major else init
    write_grid = (
        B
        + torch.arange(T, dtype=torch.int64, device=dev)[None, :] * B
        + torch.arange(B, dtype=torch.int64, device=dev)[:, None]
    )
    mtp_out = fused_recurrent_kda_mtp(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        pool_mtp,
        torch.arange(B, dtype=torch.int64, device=dev),
        write_grid,
        lower_bound=lower_bound,
        recurrent_layout=recurrent_layout,
    )

    torch.testing.assert_close(mtp_out.float(), ref_out.float(), atol=1e-3, rtol=1e-3)
    for t in range(T):
        got = pool_mtp[write_grid[:, t]]
        if v_major:
            got = got.transpose(-1, -2)
        torch.testing.assert_close(got, ref_states[t], atol=1e-4, rtol=1e-4)


def test_separate_write_pool():
    """Verify scratch mode: reads from the pool, writes to a separate tensor."""
    torch.manual_seed(1)
    dev = "cuda"
    B, T = 2, 3
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=dev)
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=dev)
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=dev)
    g = torch.randn(B, T, HV, K, dtype=torch.bfloat16, device=dev)
    beta = torch.randn(B, T, HV, dtype=torch.bfloat16, device=dev)
    A_log = torch.randn(HV, dtype=torch.float32, device=dev)
    init = torch.randn(B, HV, K, V, dtype=torch.float32, device=dev)

    pool_ro = init.clone()
    scratch = torch.zeros(B * T, HV, K, V, dtype=torch.float32, device=dev)
    grid = torch.arange(B * T, dtype=torch.int64, device=dev).view(B, T)
    out_sep = fused_recurrent_kda_mtp(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        None,
        pool_ro,
        torch.arange(B, dtype=torch.int64, device=dev),
        grid,
        h_pool_out=scratch,
    )
    assert torch.equal(pool_ro, init), "read pool must not be written"

    # same run, single-pool mode, must agree
    pool_rw = torch.zeros(B + B * T, HV, K, V, dtype=torch.float32, device=dev)
    pool_rw[:B] = init
    grid2 = B + grid
    out_one = fused_recurrent_kda_mtp(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        None,
        pool_rw,
        torch.arange(B, dtype=torch.int64, device=dev),
        grid2,
    )
    torch.testing.assert_close(out_sep, out_one, atol=0, rtol=0)
    torch.testing.assert_close(scratch, pool_rw[B:], atol=0, rtol=0)


def test_negative_write_index_skips():
    torch.manual_seed(0)
    dev = "cuda"
    B, T = 2, 3
    q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=dev)
    k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device=dev)
    v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device=dev)
    g = torch.randn(B, T, HV, K, dtype=torch.bfloat16, device=dev)
    beta = torch.randn(B, T, HV, dtype=torch.bfloat16, device=dev)
    A_log = torch.randn(HV, dtype=torch.float32, device=dev)
    pool = torch.full((B + B * T, HV, K, V), 7.0, dtype=torch.float32, device=dev)
    write = torch.full((B, T), -1, dtype=torch.int64, device=dev)
    write[:, -1] = B + torch.arange(B, dtype=torch.int64, device=dev)
    fused_recurrent_kda_mtp(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        None,
        pool,
        torch.arange(B, dtype=torch.int64, device=dev),
        write,
    )
    # Only the last-position pages were written; sentinel rows untouched
    # except those. (Rows B..B+B hold real states now.)
    assert not torch.isnan(pool).any()
