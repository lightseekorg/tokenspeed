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

"""Requests shorter than BLOCK_N batched with other requests on gfx950.

The persistent prefill kernels run several (request, head, query block) items
per workgroup. A workgroup that processes a request shorter than BLOCK_N and
then another item must give that item the same result as a launch holding
only its request. Each batch below makes some workgroups do exactly that: the
short requests come first, and the later items land on the same workgroups.
"""

from __future__ import annotations

import pytest
import torch
from utils import is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel_amd.ops.gfx950.attention.mha.prefill import (  # noqa: E402
    launch_gluon_mha_prefill_gfx950,
)
from tokenspeed_kernel_amd.ops.gfx950.attention.rmha.prefill import (  # noqa: E402
    launch_gluon_rel_mha_prefill_gfx950,
)

_HEAD_DIM = 128
_REL_EXTENT = 512
_DTYPE = torch.bfloat16

# (n_heads, n_kv_heads, window_left, seqlens)
# The causal batches fill whole batch groups of the swizzled order (512 //
# n_heads requests each), so the second group runs on the workgroups that ran
# the short first group.
_MHA_CASES = [
    # EAGLE-4 drafter layer: 40 heads, sliding window 4096.
    (40, 40, 4095, [40, 40, 40, 392]),
    (40, 40, 100, [64, 40, 40, 362, 392]),
    (32, 8, -1, [30] * 16 + [200] * 16),
    (40, 40, -1, [30] * 12 + [200] * 12),
]
_RMHA_CASES = [
    (40, 40, 100, [40, 40, 40, 392]),
    (32, 8, 100, [40, 40, 40, 392]),
    (32, 8, -1, [30] * 16 + [200] * 16),
    (40, 40, -1, [30] * 12 + [200] * 12),
]


def _inputs(n_heads: int, n_kv_heads: int, seqlens: list[int], rel: bool):
    torch.manual_seed(0)
    total = sum(seqlens)
    q = torch.randn(total, n_heads, _HEAD_DIM, device="cuda", dtype=_DTYPE)
    k = torch.randn(total, n_kv_heads, _HEAD_DIM, device="cuda", dtype=_DTYPE)
    v = torch.randn(total, n_kv_heads, _HEAD_DIM, device="cuda", dtype=_DTYPE)
    rel_logits = (
        torch.randn(total, n_heads, _REL_EXTENT, device="cuda", dtype=_DTYPE) * 0.5
        if rel
        else None
    )
    return q, k, v, rel_logits


def _check_batch_matches_single_requests(run, inputs, seqlens: list[int]) -> None:
    cu = [0]
    for s in seqlens:
        cu.append(cu[-1] + s)
    out, lse = run(*inputs, cu)
    for i, s in enumerate(seqlens):
        rows = slice(cu[i], cu[i + 1])
        single = [None if t is None else t[rows] for t in inputs]
        ref_out, ref_lse = run(*single, [0, s])
        torch.testing.assert_close(
            out[rows], ref_out, rtol=0, atol=0, msg=f"request {i} (len {s})"
        )
        torch.testing.assert_close(
            lse[rows], ref_lse, rtol=0, atol=0, msg=f"request {i} (len {s}) lse"
        )


@pytest.mark.parametrize("n_heads,n_kv_heads,window_left,seqlens", _MHA_CASES)
def test_mha_prefill_short_requests(
    n_heads: int, n_kv_heads: int, window_left: int, seqlens: list[int]
) -> None:
    def run(q, k, v, _rel, cu):
        return launch_gluon_mha_prefill_gfx950(
            q,
            k,
            v,
            torch.tensor(cu, device="cuda", dtype=torch.int32),
            cu,
            max(b - a for a, b in zip(cu, cu[1:])),
            window_left=window_left,
            return_lse=True,
        )

    inputs = _inputs(n_heads, n_kv_heads, seqlens, rel=False)
    _check_batch_matches_single_requests(run, inputs, seqlens)


@pytest.mark.parametrize("n_heads,n_kv_heads,window_left,seqlens", _RMHA_CASES)
def test_rel_mha_prefill_short_requests(
    n_heads: int, n_kv_heads: int, window_left: int, seqlens: list[int]
) -> None:
    def run(q, k, v, rel, cu):
        return launch_gluon_rel_mha_prefill_gfx950(
            q,
            k,
            v,
            rel,
            torch.tensor(cu, device="cuda", dtype=torch.int32),
            cu,
            max(b - a for a, b in zip(cu, cu[1:])),
            window_left=window_left,
            return_lse=True,
        )

    inputs = _inputs(n_heads, n_kv_heads, seqlens, rel=True)
    _check_batch_matches_single_requests(run, inputs, seqlens)
