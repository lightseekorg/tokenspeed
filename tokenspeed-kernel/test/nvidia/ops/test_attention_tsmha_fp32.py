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

"""FP32-output relative attention on the FA4 (tokenspeed-mha) kernels.

Default selection must route FP32-output prefill, cached extend and
decode, full and sliding-window, to the ``*_fp32`` FA4 registrations. Each
route is pinned against an FP64
reference and the Triton FP32 solution: the output must be the unrounded FP32
accumulator (not a BF16 value cast up) and the natural-log LSE must match.
Graph replay and dynamic batch shapes must not recompile.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention.rmha import (
    rel_mha_decode_with_kvcache,
    rel_mha_extend_with_kvcache,
    rel_mha_gate_sink,
    rel_mha_prefill,
)
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

torch.manual_seed(7)

DTYPE = torch.bfloat16
NUM_Q_HEADS = 12
NUM_KV_HEADS = 2
HEAD_DIM = 128
EXTENT = 256
SCALE = HEAD_DIM**-0.5
# FP32 accumulation of BF16-rounded probabilities: both kernels sit this close to FP64
# (worst over 6 seeds: decode 1.3e-3, prefill 4.8e-3, extend 3.5e-3, LSE 1.6e-6).
DECODE_OUT_TOL = 2e-3
PREFILL_OUT_TOL = 7e-3
LSE_TOL = 1e-5
# Mean error is ~1.4e-4; a window off by one key raises it ~20x past this bound.
MEAN_OUT_TOL = 3e-4


def _require_fa4(require) -> None:
    require("attention", "rel_mha_decode_with_kvcache", "fa4", DTYPE, "q")


def _fp64_reference(q, k, v, rel, window_left):
    """Bottom-right-aligned causal relative attention of one sequence in FP64."""
    sq, sk = q.shape[0], k.shape[0]
    group = NUM_Q_HEADS // NUM_KV_HEADS
    kd = k.double().repeat_interleave(group, dim=1)
    vd = v.double().repeat_interleave(group, dim=1)
    logits = torch.einsum("qhd,khd->hqk", q.double(), kd) * SCALE
    dist = (torch.arange(sq, device=q.device) + sk - sq)[:, None] - torch.arange(
        sk, device=q.device
    )[None, :]
    idx = dist.clamp(0, EXTENT - 1)
    bias = rel.double().gather(-1, idx.unsqueeze(1).expand(sq, NUM_Q_HEADS, sk))
    bias = torch.where(((dist >= 0) & (dist < EXTENT)).unsqueeze(1), bias, 0.0)
    logits = logits + bias.permute(1, 0, 2)
    mask = dist < 0
    if window_left >= 0:
        mask |= dist > window_left
    logits = logits.masked_fill(mask.unsqueeze(0), float("-inf"))
    out = torch.einsum("hqk,khd->qhd", logits.softmax(-1), vd)
    return out, logits.logsumexp(-1).transpose(0, 1)


def _assert_fp32_close(out, lse, ref_out, ref_lse, out_tol):
    assert out.dtype == torch.float32 and lse.dtype == torch.float32
    # A BF16 store cast up would make every value BF16-representable.
    assert (out != out.to(DTYPE).float()).float().mean() > 0.9
    torch.testing.assert_close(out.double(), ref_out, atol=out_tol, rtol=0)
    assert (out.double() - ref_out).abs().mean() < MEAN_OUT_TOL
    torch.testing.assert_close(lse.double(), ref_lse, atol=LSE_TOL, rtol=0)


def _paged(kv_lens, page, device):
    pages = [(n + page - 1) // page for n in kv_lens]
    total = sum(pages) + 2
    k_cache = torch.zeros(
        total, page, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=DTYPE
    )
    v_cache = torch.zeros_like(k_cache)
    table = torch.zeros(len(kv_lens), max(pages), device=device, dtype=torch.int32)
    order = torch.randperm(total - 1, device=device) + 1
    ks, vs, nxt = [], [], 0
    for b, n in enumerate(kv_lens):
        k = torch.randn(n, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=DTYPE)
        v = torch.randn(n, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=DTYPE)
        ks.append(k)
        vs.append(v)
        for p in range(pages[b]):
            slot = int(order[nxt])
            nxt += 1
            rows = min(page, n - p * page)
            k_cache[slot, :rows] = k[p * page : p * page + rows]
            v_cache[slot, :rows] = v[p * page : p * page + rows]
            table[b, p] = slot
    return k_cache, v_cache, table, ks, vs


def _decode_inputs(kv_lens, rows, page, device):
    k_cache, v_cache, table, ks, vs = _paged(kv_lens, page, device)
    total = len(kv_lens) * rows
    q = torch.randn(total, NUM_Q_HEADS, HEAD_DIM, device=device, dtype=DTYPE)
    rel = (0.5 * torch.randn(total, NUM_Q_HEADS, EXTENT, device=device)).to(DTYPE)
    kwargs = dict(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=table,
        cache_seqlens=torch.tensor(kv_lens, device=device, dtype=torch.int32),
        max_seqlen_k=max(kv_lens),
        rel_logits=rel,
        cu_seqlens_q=torch.arange(len(kv_lens) + 1, device=device, dtype=torch.int32)
        * rows,
        max_seqlen_q=rows,
        window_left=-1,
        softmax_scale=SCALE,
        output_dtype=torch.float32,
        return_lse=True,
        gate=None,
        sink=None,
    )
    return kwargs, ks, vs


def _signature(*roles):
    return format_signature(**{role: dense_tensor_format(DTYPE) for role in roles})


@pytest.mark.parametrize(
    "op,roles,extra,window,expected",
    [
        ("rel_mha_decode_with_kvcache", ("q", "k_cache", "v_cache"), {"page_size": 128}, False, "fa4_rel_mha_decode_with_kvcache_fp32"),
        ("rel_mha_decode_with_kvcache", ("q", "k_cache", "v_cache"), {"page_size": 128}, True, "fa4_rel_mha_decode_with_kvcache_fp32"),
        ("rel_mha_decode_with_kvcache", ("q", "k_cache", "v_cache"), {"page_size": 128, "gate_sink": True}, True, "fa4_rel_mha_decode_with_kvcache_fp32"),
        ("rel_mha_extend_with_kvcache", ("q", "k_cache", "v_cache"), {"page_size": 128}, False, "fa4_rel_mha_extend_with_kvcache_fp32"),
        ("rel_mha_extend_with_kvcache", ("q", "k_cache", "v_cache"), {"page_size": 128}, True, "fa4_rel_mha_extend_with_kvcache_fp32"),
        ("rel_mha_prefill", ("q", "k", "v"), {}, False, "fa4_rel_mha_prefill_fp32"),
        ("rel_mha_prefill", ("q", "k", "v"), {}, True, "fa4_rel_mha_prefill_fp32"),
    ],
)  # fmt: skip
def test_fp32_default_selection(require, op, roles, extra, window, expected) -> None:
    """FP32 requests take FA4."""
    _require_fa4(require)
    traits = {
        "fp32_output": True,
        "head_dim": HEAD_DIM,
        "return_lse": True,
        "sliding_window": window,
        **extra,
    }
    kernel = select_kernel("attention", op, _signature(*roles), traits=traits)
    assert kernel.name == expected


@pytest.mark.parametrize("page", [64, 128, 256])
@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize("kv_lens", [[2600], [120, 1000, 2600]])
@pytest.mark.parametrize("window_left", [-1, EXTENT - 1])
def test_fp32_decode(device, require, page, rows, kv_lens, window_left) -> None:
    """Full and sliding-window decode (plain and spec-verify rows) vs FP64
    and Triton, for one request (more KV splits) and a batch."""
    _require_fa4(require)
    kv_lens = [n + rows for n in kv_lens]
    kwargs, ks, vs = _decode_inputs(kv_lens, rows, page, device)
    kwargs["window_left"] = window_left
    out, lse = rel_mha_decode_with_kvcache(**kwargs)
    tri_out, tri_lse = rel_mha_decode_with_kvcache(**kwargs, solution="triton")
    for b in range(len(kv_lens)):
        sl = slice(b * rows, (b + 1) * rows)
        ref_out, ref_lse = _fp64_reference(
            kwargs["q"][sl], ks[b], vs[b], kwargs["rel_logits"][sl], window_left
        )
        _assert_fp32_close(out[sl], lse[sl], ref_out, ref_lse, DECODE_OUT_TOL)
        _assert_fp32_close(tri_out[sl], tri_lse[sl], ref_out, ref_lse, DECODE_OUT_TOL)


@pytest.mark.parametrize("window_left", [-1, EXTENT - 1])
def test_fp32_prefill(device, require, window_left) -> None:
    """Packed varlen prefill, full and sliding window, vs FP64 and Triton."""
    _require_fa4(require)
    lens = [128, 300, 65]
    cpu = [0]
    for n in lens:
        cpu.append(cpu[-1] + n)
    total = cpu[-1]
    q = torch.randn(total, NUM_Q_HEADS, HEAD_DIM, device=device, dtype=DTYPE)
    k = torch.randn(total, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=DTYPE)
    v = torch.randn_like(k)
    rel = (0.5 * torch.randn(total, NUM_Q_HEADS, EXTENT, device=device)).to(DTYPE)
    kwargs = dict(
        q=q,
        k=k,
        v=v,
        rel_logits=rel,
        cu_seqlens=torch.tensor(cpu, device=device, dtype=torch.int32),
        cu_seqlens_cpu=cpu,
        max_seqlen=max(lens),
        window_left=window_left,
        softmax_scale=SCALE,
        output_dtype=torch.float32,
        return_lse=True,
    )
    out, lse = rel_mha_prefill(**kwargs)
    tri_out, tri_lse = rel_mha_prefill(**kwargs, solution="triton")
    for b in range(len(lens)):
        sl = slice(cpu[b], cpu[b + 1])
        ref_out, ref_lse = _fp64_reference(q[sl], k[sl], v[sl], rel[sl], window_left)
        _assert_fp32_close(out[sl], lse[sl], ref_out, ref_lse, PREFILL_OUT_TOL)
        _assert_fp32_close(tri_out[sl], tri_lse[sl], ref_out, ref_lse, PREFILL_OUT_TOL)


@pytest.mark.parametrize("page", [128, 256])
@pytest.mark.parametrize("window_left", [-1, EXTENT - 1])
def test_fp32_extend(device, require, page, window_left) -> None:
    """Cached extend with an unaligned prefix, full and sliding window."""
    _require_fa4(require)
    kv_lens, q_lens = [700, 1300], [100, 257]
    k_cache, v_cache, table, ks, vs = _paged(kv_lens, page, device)
    cpu = [0]
    for n in q_lens:
        cpu.append(cpu[-1] + n)
    q = torch.randn(cpu[-1], NUM_Q_HEADS, HEAD_DIM, device=device, dtype=DTYPE)
    rel = (0.5 * torch.randn(cpu[-1], NUM_Q_HEADS, EXTENT, device=device)).to(DTYPE)
    kwargs = dict(
        q=q,
        cu_seqlens_q=torch.tensor(cpu, device=device, dtype=torch.int32),
        cu_seqlens_kv=torch.tensor(
            [0, kv_lens[0], sum(kv_lens)], device=device, dtype=torch.int32
        ),
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=table,
        cache_seqlens=torch.tensor(kv_lens, device=device, dtype=torch.int32),
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(kv_lens),
        rel_logits=rel,
        window_left=window_left,
        softmax_scale=SCALE,
        output_dtype=torch.float32,
        return_lse=True,
    )
    out, lse = rel_mha_extend_with_kvcache(**kwargs)
    tri_out, tri_lse = rel_mha_extend_with_kvcache(**kwargs, solution="triton")
    for b in range(len(kv_lens)):
        sl = slice(cpu[b], cpu[b + 1])
        ref_out, ref_lse = _fp64_reference(q[sl], ks[b], vs[b], rel[sl], window_left)
        _assert_fp32_close(out[sl], lse[sl], ref_out, ref_lse, PREFILL_OUT_TOL)
        _assert_fp32_close(tri_out[sl], tri_lse[sl], ref_out, ref_lse, PREFILL_OUT_TOL)


@pytest.mark.parametrize("kv_lens", [[2600], [120, 1000, 2600]])
@pytest.mark.parametrize("window_left", [-1, EXTENT - 1])
def test_gated_decode_is_the_gate_sink_epilogue(
    device, require, kv_lens, window_left
) -> None:
    """FA4 applies the gate and sink in its reduction, and the facade applies
    them after the Triton solution: both equal ``rel_mha_gate_sink`` on the
    FP32 output and its LSE bit for bit."""
    _require_fa4(require)
    rows = 8
    kwargs, _, _ = _decode_inputs([n + rows for n in kv_lens], rows, 128, device)
    kwargs["window_left"] = window_left
    total = kwargs["q"].shape[0]
    width = NUM_Q_HEADS * HEAD_DIM
    # Rows of a wider projection, as the model slices them.
    projection = 3 * torch.randn(
        total, 8 + width + NUM_Q_HEADS, device=device, dtype=DTYPE
    )
    gate, sink = projection[:, 8 : 8 + width], projection[:, 8 + width :]
    for solution in (None, "triton"):
        out, lse = rel_mha_decode_with_kvcache(**kwargs, solution=solution)
        expected = rel_mha_gate_sink(
            out.reshape(total, -1), gate, lse, sink, head_dim=HEAD_DIM, out_dtype=DTYPE
        )
        gated_kwargs = dict(kwargs, output_dtype=DTYPE, return_lse=False)
        gated = rel_mha_decode_with_kvcache(
            **dict(gated_kwargs, gate=gate, sink=sink), solution=solution
        )
        assert gated.dtype == DTYPE
        assert torch.equal(
            gated.reshape(total, -1).view(torch.int16), expected.view(torch.int16)
        )


def test_fp32_decode_cuda_graph_replay(device, require) -> None:
    """A captured FP32 decode replays new contents identically to eager."""
    _require_fa4(require)
    rows = 8
    kwargs, _, _ = _decode_inputs([300, 2000], rows, 128, device)
    rel_mha_decode_with_kvcache(**kwargs)  # eager compile and workspaces
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, lse = rel_mha_decode_with_kvcache(**kwargs)
    kwargs["q"].copy_(torch.randn_like(kwargs["q"]))
    kwargs["rel_logits"].copy_(torch.randn_like(kwargs["rel_logits"]))
    kwargs["cache_seqlens"].copy_(
        torch.tensor([200, 1900], device=device, dtype=torch.int32)
    )
    graph.replay()
    eager_out, eager_lse = rel_mha_decode_with_kvcache(**kwargs)
    torch.testing.assert_close(out, eager_out, atol=0, rtol=0)
    torch.testing.assert_close(lse, eager_lse, atol=0, rtol=0)


def test_fp32_batch_shapes_do_not_recompile(device, require) -> None:
    """Batch, row, page and length changes reuse the compiled FP32 kernels."""
    _require_fa4(require)
    from tokenspeed_kernel.ops.attention.rmha._cute_dsl import rel_decode_v2, rel_extend

    def run(kv_lens, q_lens):
        kwargs, _, _ = _decode_inputs(kv_lens, 8, 128, device)
        rel_mha_decode_with_kvcache(**kwargs)
        k_cache, v_cache, table, _, _ = _paged(kv_lens, 128, device)
        cpu = [0]
        for n in q_lens:
            cpu.append(cpu[-1] + n)
        q = torch.randn(cpu[-1], NUM_Q_HEADS, HEAD_DIM, device=device, dtype=DTYPE)
        rel = torch.randn(cpu[-1], NUM_Q_HEADS, EXTENT, device=device, dtype=DTYPE)
        rel_mha_extend_with_kvcache(
            q=q,
            cu_seqlens_q=torch.tensor(cpu, device=device, dtype=torch.int32),
            cu_seqlens_kv=torch.tensor(
                [0] + list(torch.tensor(kv_lens).cumsum(0)),
                device=device,
                dtype=torch.int32,
            ),
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=table,
            cache_seqlens=torch.tensor(kv_lens, device=device, dtype=torch.int32),
            max_seqlen_q=max(q_lens),
            max_seqlen_k=max(kv_lens),
            rel_logits=rel,
            window_left=-1,
            softmax_scale=SCALE,
            output_dtype=torch.float32,
            return_lse=True,
        )

    run([400], [64])
    compiled = (
        len(rel_decode_v2._DECODE._cache),
        len(rel_extend._FWD_VARLEN),
        len(rel_extend._SHEAR),
    )
    run([300, 1700, 2900], [128, 77, 300])
    run([5000, 129], [1000, 9])
    assert (
        len(rel_decode_v2._DECODE._cache),
        len(rel_extend._FWD_VARLEN),
        len(rel_extend._SHEAR),
    ) == compiled
