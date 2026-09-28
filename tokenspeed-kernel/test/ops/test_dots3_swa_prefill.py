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

from itertools import accumulate

import pytest
import torch
from tokenspeed_kernel.ops.attention.dots3_note import swa_prefill
from tokenspeed_kernel.ops.attention.dots3_note.triton import (
    _swa_prefill_kernel,
    triton_dots3_note_swa_prefill,
)
from tokenspeed_kernel.registry import KernelRegistry
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA or ROCm"
)
_SWA_SPEC = KernelRegistry.get().get_by_name("triton_dots3_note_swa_prefill")


@pytest.fixture(autouse=True)
def _register_swa(fresh_registry):
    KernelRegistry.get().register(_SWA_SPEC, triton_dots3_note_swa_prefill)


@pytest.mark.parametrize("window_left", [0, 1, 63, 64, 512])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_swa_prefill_window_ragged(cached, window_left, dtype):
    torch.manual_seed(19)
    q_lens = (0, 1, 31, 32, 33, 63, 64, 65, 511, 512, 513, 545)
    kv_lens = tuple(n + 512 if n and cached else n for n in q_lens)
    kv_heads = 2 if cached else 4
    cu_q = torch.tensor([0, *accumulate(q_lens)], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, *accumulate(kv_lens)], device="cuda", dtype=torch.int32)
    q = torch.randn((sum(q_lens), 8, 258), device="cuda", dtype=dtype)[:, ::2, 1:-1]
    k_storage = torch.full(
        (sum(kv_lens) + 64, 2 * kv_heads, 258), float("nan"), device="cuda", dtype=dtype
    )
    v_storage = torch.full(
        (sum(kv_lens) + 64, 2 * kv_heads, 130), float("nan"), device="cuda", dtype=dtype
    )
    k, v = k_storage[:-64, ::2, 1:-1], v_storage[:-64, ::2, 1:-1]
    k.normal_()
    v.normal_()

    def run():
        return swa_prefill(
            q,
            k,
            v,
            cu_q,
            cu_k,
            max(q_lens),
            max(kv_lens),
            1 / 16,
            window_left=window_left,
            solution="triton",
        )

    def verify(out):
        assert out.shape == (q.shape[0], 4, 128)
        assert out.dtype == dtype
        qo = ko = 0
        for nq, nk in zip(q_lens, kv_lens, strict=True):
            if nq:
                qi = q[qo : qo + nq].float()
                ki = k[ko : ko + nk].float().repeat_interleave(4 // kv_heads, dim=1)
                vi = v[ko : ko + nk].float().repeat_interleave(4 // kv_heads, dim=1)
                scores = torch.einsum("qhd,khd->hqk", qi, ki) / 16
                qpos = torch.arange(nq, device="cuda") + nk - nq
                kpos = torch.arange(nk, device="cuda")
                mask = (kpos[None, :] <= qpos[:, None]) & (
                    kpos[None, :] >= qpos[:, None] - window_left
                )
                scores.masked_fill_(~mask, -float("inf"))
                expected = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vi)
                torch.testing.assert_close(
                    out[qo : qo + nq].float(), expected, atol=1e-3, rtol=5e-3
                )
            qo += nq
            ko += nk

    verify(run())
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    q.mul_(0.5)
    v.neg_()
    q_lens, kv_lens = tuple(reversed(q_lens)), tuple(reversed(kv_lens))
    cu_q.copy_(torch.tensor([0, *accumulate(q_lens)], device="cuda", dtype=torch.int32))
    cu_k.copy_(
        torch.tensor([0, *accumulate(kv_lens)], device="cuda", dtype=torch.int32)
    )
    graph.replay()
    verify(result)


@pytest.mark.parametrize("nq,nk", [(5, 3), (69, 3), (33, 0), (0, 0)])
def test_swa_prefill_bottom_right_empty_rows(nq, nk):
    q = torch.zeros((nq, 2, 256), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros((nk, 1, 256), device="cuda", dtype=torch.bfloat16)
    v = torch.ones((nk, 1, 128), device="cuda", dtype=torch.bfloat16)
    cu_q = torch.tensor([0, nq], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, nk], device="cuda", dtype=torch.int32)
    out = swa_prefill(
        q, k, v, cu_q, cu_k, nq, nk, 1 / 16, window_left=512, solution="triton"
    )
    empty_rows = nq - nk
    assert torch.equal(out[:empty_rows], torch.zeros_like(out[:empty_rows]))
    assert torch.equal(out[empty_rows:], torch.ones_like(out[empty_rows:]))


def test_swa_prefill_ignores_expired_poison():
    q = torch.zeros((33, 2, 256), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros((1057, 1, 256), device="cuda", dtype=torch.bfloat16)
    v = torch.zeros((1057, 1, 128), device="cuda", dtype=torch.bfloat16)
    k[:512] = float("nan")
    v[:512] = float("nan")
    v[512:544] = 1
    cu_q = torch.tensor([0, 33], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 1057], device="cuda", dtype=torch.int32)
    out = swa_prefill(
        q, k, v, cu_q, cu_k, 33, 1057, 1 / 16, window_left=512, solution="triton"
    )
    expected = (32 - torch.arange(33, device="cuda")).float() / 513
    torch.testing.assert_close(
        out.float(), expected[:, None, None].expand_as(out), atol=1e-3, rtol=5e-3
    )
    assert (out[-1] == 0).all()


@pytest.mark.parametrize("length_dtype", [torch.int32, torch.int64])
def test_swa_prefill_batch_shapes_do_not_recompile(length_dtype):
    def run(q_lens, kv_lens):
        q = torch.zeros((sum(q_lens), 4, 256), device="cuda", dtype=torch.bfloat16)
        k = torch.zeros((sum(kv_lens), 2, 256), device="cuda", dtype=torch.bfloat16)
        v = torch.ones((sum(kv_lens), 2, 128), device="cuda", dtype=torch.bfloat16)
        cu_q = torch.tensor([0, *accumulate(q_lens)], device="cuda", dtype=length_dtype)
        cu_k = torch.tensor(
            [0, *accumulate(kv_lens)], device="cuda", dtype=length_dtype
        )
        out = swa_prefill(
            q,
            k,
            v,
            cu_q,
            cu_k,
            max(q_lens),
            max(kv_lens),
            1 / 16,
            window_left=512,
            solution="triton",
        )
        torch.testing.assert_close(out, torch.ones_like(out), atol=0, rtol=0)

    run((2,), (18,))
    with assert_no_triton_compile(_swa_prefill_kernel):
        for q_lens, kv_lens in (
            ((1,), (1,)),
            ((16,), (528,)),
            ((63, 64, 65), (575, 640, 1025)),
            ((0, 33, 511, 513), (0, 545, 1023, 2049)),
        ):
            run(q_lens, kv_lens)
