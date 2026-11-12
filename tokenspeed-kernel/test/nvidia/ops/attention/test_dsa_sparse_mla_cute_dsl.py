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

"""CuTe DSL sparse MLA decode against an FP64 reference and the trtllm-gen core.

Inputs are E4M3 queries and latent cache rows selected by global slot. The
kernel quantizes P to E4M3 at 448 of its running row max, as the FP8 trtllm-gen
core does; partials merge in FP32. Outputs must sit within the FP8 roundoff
bound of the FP64 reference that the Prims-TS adapter test uses.
"""

from __future__ import annotations

import math

import pytest
import torch
from utils import is_nvidia

if not is_nvidia():
    pytest.skip("requires an NVIDIA GPU", allow_module_level=True)
pytest.importorskip("cutlass")
from tokenspeed_kernel.platform import ArchVersion, current_platform

if not ArchVersion(10, 0) <= current_platform().arch_version <= ArchVersion(10, 3):
    pytest.skip("requires SM100 or SM103", allow_module_level=True)
import tokenspeed_kernel.ops.attention.dsa as dsa_ops  # noqa: E402
from tokenspeed_kernel.ops.attention.dsa import dsa_decode  # noqa: E402
from tokenspeed_kernel.ops.attention.dsa.cute_dsl import (  # noqa: E402
    SPARSE_CORE_SPLIT_BY_ROWS,
)
from tokenspeed_kernel.thirdparty.cute_dsl.k3_decode.dsa.sparse import (  # noqa: E402
    sparse_mla_decode,
)

ROW, LATENT = 576, 512
SCALE = 192**-0.5
POOL = 20480  # whole 64-row pages, as the trtllm-gen core views the cache


def _inputs(rows, heads, width, lengths, holes, seed, q_scale=2.0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    # Logits of about 3.5 nats spread: maxima differ across passes and CTAs.
    q = (torch.randn(rows, heads, ROW, device="cuda", generator=gen) * q_scale).to(
        torch.float8_e4m3fn
    )
    kv = torch.randn(POOL, 1, ROW, device="cuda", generator=gen)
    # A per-row offset of the values: a misweighted partial moves the output.
    kv[:, :, :LATENT] += torch.randn(POOL, 1, 1, device="cuda", generator=gen)
    kv = kv.to(q.dtype)
    slots = torch.full((rows, width), -1, dtype=torch.int32, device="cuda")
    for r, n in enumerate(lengths):
        slots[r, :n] = torch.randperm(POOL, device="cuda", generator=gen)[:n].int()
        if holes and n > holes:
            slots[r, torch.randperm(n, device="cuda", generator=gen)[:holes]] = -1
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    return q, kv, slots, lens


def _reference(q, kv, slots, lens):
    """FP64 attention and the FP8 roundoff bound of the E4M3 P x E4M3 V product."""
    kv = kv.view(-1, ROW).double()
    outputs, bounds = [], []
    for r in range(q.shape[0]):
        row = slots[r] if lens is None else slots[r, : int(lens[r])]
        row = row[row >= 0].long()
        if row.numel() == 0:
            outputs.append(torch.zeros(q.shape[1], LATENT, device="cuda").double())
            bounds.append(torch.zeros_like(outputs[-1]))
            continue
        keys = kv[row]
        p = (q[r].double() @ keys.T * SCALE).softmax(-1)
        values = keys[:, :LATENT].abs()
        outputs.append(p @ keys[:, :LATENT])
        bounds.append(
            (2**-4 + 3 * 2**-8 + 1e-5) * (p @ values)
            + (2**-10 / 448) * p.amax(-1, keepdim=True) * values.sum(0)
            + 1e-6
        )
    return torch.stack(outputs), torch.stack(bounds)


def _check(actual, q, kv, slots, lens):
    reference, bound = _reference(q, kv, slots, lens)
    error = (actual.double() - reference).abs()
    assert torch.isfinite(actual).all()
    assert (error <= bound).all(), (
        f"error exceeds the FP8 roundoff bound by "
        f"{(error / bound.clamp_min(1e-30)).max().item():.3f}x"
    )


def _run(q, kv, slots, lens, split):
    out = torch.full(
        (*q.shape[:2], LATENT), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    result = sparse_mla_decode(q, kv, slots, lens, out, SCALE, split=split, pdl=True)
    assert result is out
    return out


@pytest.mark.parametrize(
    ("lengths", "split"),
    [
        # Kimi-K3.1 verify rows: row j of a 600-token request sees 593 + j keys.
        ([593 + j for j in range(8)], 8),
        # Full top-k rows: one pass per CTA, then two, four and eight passes.
        ([2048] * 8, 8),
        ([2048] * 4, 4),
        ([2048] * 4, 2),
        ([2048] * 2, 1),
        # Empty, single-key and short rows; a row shorter than the cluster.
        ([0, 1, 5, 300], 8),
        # Pass boundaries: 128 and 129 keys per CTA, an odd pass count.
        ([1024, 1032, 1536, 1544], 8),
    ],
)
def test_matches_fp64_reference(lengths, split):
    q, kv, slots, lens = _inputs(len(lengths), 12, 2048, lengths, 0, seed=split)
    _check(_run(q, kv, slots, lens, split), q, kv, slots, lens)


@pytest.mark.parametrize("heads", [1, 12, 16])
def test_head_counts_and_masked_slots(heads):
    lengths = [700, 2048, 129, 257]
    q, kv, slots, lens = _inputs(4, heads, 2048, lengths, 17, seed=heads)
    _check(_run(q, kv, slots, lens, 4), q, kv, slots, lens)


def test_full_width_without_lengths():
    q, kv, slots, _ = _inputs(3, 12, 512, [300, 512, 5], 3, seed=5)
    _check(_run(q, kv, slots, None, 2), q, kv, slots, None)


def test_graph_replay_follows_new_inputs():
    lengths = [593 + j for j in range(8)]
    q, kv, slots, lens = _inputs(8, 12, 2048, lengths, 0, seed=11)
    out = torch.empty(8, 12, LATENT, device="cuda", dtype=torch.bfloat16)
    sparse_mla_decode(q, kv, slots, lens, out, SCALE, split=8, pdl=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sparse_mla_decode(q, kv, slots, lens, out, SCALE, split=8, pdl=True)
    q2, kv2, slots2, lens2 = _inputs(8, 12, 2048, [2048] * 4 + [1] * 4, 9, seed=12)
    q.copy_(q2)
    kv.copy_(kv2)
    slots.copy_(slots2)
    lens.copy_(lens2)
    graph.replay()
    torch.cuda.synchronize()
    _check(out, q, kv, slots, lens)


def _facade(q, kv, slots, lens, q_len, return_lse, solution):
    return dsa_decode(
        q=q,
        kv_cache=kv,
        sparse_kv_cache=None,
        topk_slots=slots,
        topk_lens=lens,
        max_seqlen_k=4096,
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        softmax_scale=SCALE,
        page_size=64,
        q_len_per_req=q_len,
        logit_cap=0.0,
        k_scale=1.0,
        return_lse=return_lse,
        solution=solution,
        slot_order="selection",
    )


def test_split_by_rows_is_pinned():
    # One or four requests of eight verify rows, where it beats trtllm-gen.
    assert SPARSE_CORE_SPLIT_BY_ROWS == {8: 8, 32: 4}


@pytest.mark.parametrize(
    ("requests", "q_len", "heads", "return_lse", "expected"),
    [
        (1, 8, 12, False, "cute_dsl_dsa_decode"),
        (4, 8, 12, False, "cute_dsl_dsa_decode"),
        (2, 8, 12, False, "flashinfer_trtllm_dsa_decode"),
        (3, 8, 12, False, "flashinfer_trtllm_dsa_decode"),
        (8, 8, 12, False, "flashinfer_trtllm_dsa_decode"),
        # 8 and 32 rows of one-row requests are not the measured verify shape.
        (8, 1, 12, False, "flashinfer_trtllm_dsa_decode"),
        (32, 1, 12, False, "flashinfer_trtllm_dsa_decode"),
        # Unmeasured head counts, gathered heads, and an LSE call.
        (1, 8, 8, False, "flashinfer_trtllm_dsa_decode"),
        (1, 8, 16, False, "flashinfer_trtllm_dsa_decode"),
        (1, 8, 96, False, "flashinfer_trtllm_dsa_decode"),
        (1, 8, 12, True, "flashinfer_trtllm_dsa_decode"),
    ],
)
def test_facade_selects_by_rows(
    monkeypatch, requests, q_len, heads, return_lse, expected
):
    selected = []
    select_kernel = dsa_ops.select_kernel

    def recording_select(*args, **kwargs):
        kernel = select_kernel(*args, **kwargs)
        selected.append(kernel.name)
        return kernel

    monkeypatch.setattr(dsa_ops, "select_kernel", recording_select)
    rows = requests * q_len
    lengths = [593 + j % 8 for j in range(rows)]
    q, kv, slots, lens = _inputs(rows, heads, 2048, lengths, 0, seed=rows + heads)
    result = _facade(q, kv, slots, lens, q_len, return_lse, None)
    assert selected == [expected]
    output = result[0] if return_lse else result
    _check(output.reshape(rows, heads, LATENT), q, kv, slots, lens)


def test_close_to_trtllm_core():
    """The K3.1 decode shape: within the FP8 bound of the trtllm-gen core."""
    lengths = [593 + j for j in range(8)] * 2
    q, kv, slots, lens = _inputs(16, 12, 2048, lengths, 0, seed=21)
    ours = _run(q, kv, slots, lens, 8)
    theirs = _facade(q, kv, slots, lens, 8, False, "flashinfer_trtllm")
    _, bound = _reference(q, kv, slots, lens)
    difference = (ours.double() - theirs.reshape(ours.shape).double()).abs()
    assert (difference <= 2 * bound).all()
    assert math.isfinite(difference.max().item())
