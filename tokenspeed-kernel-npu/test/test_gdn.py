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

"""Ascend scan versus an independent CPU FP64 recurrence and page lifecycle."""

import pytest
import torch

pytest.importorskip("torch_npu")

from tokenspeed_kernel.numerics.torch.attention.gdn import (
    gdn_gate,
    gdn_l2norm,
    torch_gdn_recurrence,
)
from tokenspeed_kernel.ops.attention.gdn import gdn_chunk_prefill, gdn_decode_step

pytestmark = pytest.mark.skipif(
    not torch.npu.is_available(), reason="GDN tests require an Ascend NPU"
)


def reference(q, k, v, g, beta, state):
    return torch_gdn_recurrence(
        gdn_l2norm(q.double(), 1e-6),
        gdn_l2norm(k.double(), 1e-6),
        v.double(),
        g.double(),
        beta.double(),
        state.double(),
        scale=q.shape[-1] ** -0.5,
        parents=None,
        compute_dtype=torch.float64,
    )


@pytest.mark.parametrize("length", [1, 3, 4, 5, 63, 64, 65, 129, 2048])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16])
def test_prefill_against_fp64(length, dtype):
    torch.manual_seed(17)
    # Include grouped heads and non-square state to expose layout mistakes.
    h, hv, dk, dv = (16, 16, 128, 128) if length == 65 else (2, 4, 32, 19)
    q, k = [torch.randn(1, length, h, dk).to(dtype) for _ in range(2)]
    v = torch.randn(1, length, hv, dv).to(dtype)
    g = -torch.rand(1, length, hv)
    beta = torch.rand(1, length, hv).to(dtype)
    state = torch.randn(1, hv, dv, dk) * 0.1
    expected, states = reference(q, k, v, g, beta, state)
    actual = gdn_chunk_prefill(
        *(x.npu() for x in (q, k, v, g, beta)),
        scale=dk**-0.5,
        initial_state=state.npu(),
        cu_seqlens=torch.tensor([0, length], device="npu", dtype=torch.int32),
        qk_l2norm=True,
        output_final_state=True,
        output_h=False,
        solution="ascend",
    )
    # BF16 output rounds once (unit roundoff 1/256); FP32 state does not.
    atol, rtol = {
        torch.bfloat16: (1e-3, 4e-3),
        torch.float16: (1e-4, 5e-4),
        torch.float32: (2e-6, 2e-5),
    }[dtype]
    torch.testing.assert_close(
        actual.out.cpu().double(), expected, atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        actual.final_state.cpu().double(),
        states[:, -1],
        atol=3e-6,
        rtol=3e-5,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_decode_paging_multistep_and_request_isolation(dtype):
    torch.manual_seed(29)
    batch, length, h, hv, dk, dv = 3, 9, 2, 4, 32, 19
    # Wider backing rows exercise projection views, including strided gates.
    backing = torch.randn(batch, length, h * dk * 2 + hv * dv + 11).to(dtype)
    q = backing[:, :, : h * dk].view(batch, length, h, dk)
    k = backing[:, :, h * dk : 2 * h * dk].view(batch, length, h, dk)
    v = backing[:, :, 2 * h * dk : 2 * h * dk + hv * dv].view(batch, length, hv, dv)
    a = torch.randn(batch, length, hv).to(dtype)
    b = torch.randn_like(a)
    alog, bias = torch.randn(hv), torch.randn(hv)
    start = torch.randn(batch, hv, dv, dk) * 0.1
    g = gdn_gate(a.float(), bias, alog)
    expected, states = reference(q, k, v, g, b.float().sigmoid(), start)
    pool = torch.randn(10, hv, dv, dk, device="npu")
    pool[1:4] = start.npu()
    untouched = pool.clone()
    read = torch.tensor([1, 2, 3], device="npu", dtype=torch.int32)
    write = torch.tensor([5, 6, 7], device="npu", dtype=torch.int32)
    qn, kn, vn, an, bn, ln, dn = [x.npu() for x in (q, k, v, a, b, alog, bias)]
    for t in range(length):
        out = gdn_decode_step(
            qn[:, t : t + 1],
            kn[:, t : t + 1],
            vn[:, t : t + 1],
            A_log=ln,
            a=an[:, t : t + 1],
            dt_bias=dn,
            b=bn[:, t : t + 1],
            initial_state=pool,
            initial_state_indices=read,
            output_state_indices=write,
            use_qk_l2norm=True,
            solution="ascend",
        )
        torch.testing.assert_close(
            out.cpu().double(),
            expected[:, t : t + 1],
            atol=1e-3,
            rtol=4e-3,
        )
        torch.testing.assert_close(
            pool[5:8].cpu().double(),
            states[:, t],
            atol=3e-6,
            rtol=3e-5,
        )
        read = write
    for idx in [0, 1, 2, 3, 4, 8, 9]:
        torch.testing.assert_close(pool[idx], untouched[idx], atol=0, rtol=0)
    # A completed request's old bytes cannot affect a new zero-state request.
    saved = pool.clone()
    gdn_decode_step(
        qn[:1, :1],
        kn[:1, :1],
        vn[:1, :1],
        A_log=ln,
        a=an[:1, :1],
        dt_bias=dn,
        b=bn[:1, :1],
        initial_state=pool,
        initial_state_indices=torch.tensor([-1], device="npu", dtype=torch.int32),
        output_state_indices=torch.tensor([-1], device="npu", dtype=torch.int32),
        use_qk_l2norm=True,
        solution="ascend",
    )
    torch.testing.assert_close(pool, saved, atol=0, rtol=0)


def test_ragged_prefill_and_split_equivalence():
    torch.manual_seed(31)
    lengths = [3, 0, 7]
    q, k = [torch.randn(1, 10, 2, 32, device="npu").bfloat16() for _ in range(2)]
    v = torch.randn(1, 10, 4, 19, device="npu").bfloat16()
    g = -torch.rand(1, 10, 4, device="npu")
    beta = torch.rand(1, 10, 4, device="npu").bfloat16()
    initial = torch.randn(3, 4, 19, 32, device="npu") * 0.1
    before = initial.clone()

    def scan(start, end, state, bounds):
        return gdn_chunk_prefill(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            g[:, start:end],
            beta[:, start:end],
            scale=32**-0.5,
            initial_state=state,
            cu_seqlens=torch.tensor(bounds, device="npu", dtype=torch.int32),
            qk_l2norm=True,
            output_final_state=True,
            output_h=False,
            solution="ascend",
        )

    packed = scan(0, 10, initial, [0, 3, 3, 10])
    offset = 0
    for i, length in enumerate(lengths):
        separate = scan(offset, offset + length, initial[i : i + 1], [0, length])
        torch.testing.assert_close(
            packed.out[:, offset : offset + length], separate.out, atol=0, rtol=0
        )
        torch.testing.assert_close(
            packed.final_state[i : i + 1], separate.final_state, atol=0, rtol=0
        )
        offset += length
    first = scan(3, 6, initial[2:3], [0, 3])
    second = scan(6, 10, first.final_state, [0, 4])
    torch.testing.assert_close(packed.out[:, 6:], second.out, atol=0, rtol=0)
    torch.testing.assert_close(
        packed.final_state[2:3], second.final_state, atol=0, rtol=0
    )
    torch.testing.assert_close(initial, before, atol=0, rtol=0)


@pytest.mark.parametrize("value_heads", [16, 32, 48])
def test_tp_head_partition_preserves_recurrence_exactly(value_heads):
    torch.manual_seed(53 + value_heads)
    length, key_heads, dim = 7, 16, 128
    q, k = [
        torch.randn(1, length, key_heads, dim, dtype=torch.float16).npu()
        for _ in range(2)
    ]
    v = torch.randn(1, length, value_heads, dim, dtype=torch.float16).npu()
    g = -torch.rand(1, length, value_heads).npu() / 8
    beta = torch.rand(1, length, value_heads, dtype=torch.float16).npu()
    state = torch.randn(1, value_heads, dim, dim).npu() * 0.01
    original_state = state.clone()
    bounds = torch.tensor([0, length], dtype=torch.int32, device="npu")

    def run(q, k, v, g, beta, state):
        return gdn_chunk_prefill(
            q,
            k,
            v,
            g,
            beta,
            scale=dim**-0.5,
            initial_state=state,
            cu_seqlens=bounds,
            qk_l2norm=True,
            output_final_state=True,
            output_h=False,
            solution="ascend",
        )

    full = run(q, k, v, g, beta, state)
    parts = []
    for rank in range(2):
        key = slice(rank * key_heads // 2, (rank + 1) * key_heads // 2)
        value = slice(rank * value_heads // 2, (rank + 1) * value_heads // 2)
        parts.append(
            run(
                q[:, :, key].contiguous(),
                k[:, :, key].contiguous(),
                v[:, :, value].contiguous(),
                g[:, :, value].contiguous(),
                beta[:, :, value].contiguous(),
                state[:, value].contiguous(),
            )
        )
    torch.testing.assert_close(
        torch.cat([p.out for p in parts], dim=2), full.out, atol=0, rtol=0
    )
    torch.testing.assert_close(
        torch.cat([p.final_state for p in parts], dim=1),
        full.final_state,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(state, original_state, atol=0, rtol=0)
