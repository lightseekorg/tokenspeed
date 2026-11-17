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

"""The trtllm-gen sparse MLA leaf returns the natural-log LSE a DCP merge needs.

Under KVP every rank attends the gathered query heads over the slots it owns
(compacted to the front of each row, the row length the owned count) and the
partials merge by their LSE. The trtllm-gen leaf must agree with the Triton
leaf on both the output and the LSE, give a row it owns no slot the merge
identity's -inf LSE (the DCP combines take no value of such a row's output),
and merge back to the unsplit attention.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsa import dsa_decode, dsa_prefill

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="requires SM100-family Blackwell (trtllm-gen)",
)

TRT = "flashinfer_trtllm"
RANK, ROPE, NOPE, TOPK, SLOTS = 512, 64, 128, 2048, 8192
SCALE = (NOPE + ROPE) ** -0.5
COUNTS = [0, 3, 63, 64, 65, 127, 512, 1024, 2048, 2048]


def _inputs(heads, dtype, rows):
    q = (torch.randn((rows, heads, RANK + ROPE), device="cuda") * 0.5).to(dtype)
    kv = (torch.randn(SLOTS, RANK + ROPE, device="cuda") * 0.25).to(dtype)
    counts = torch.tensor(
        (COUNTS * (rows // len(COUNTS) + 1))[:rows], device="cuda", dtype=torch.int32
    )
    slots = torch.full((rows, TOPK), -1, device="cuda", dtype=torch.int32)
    for row, count in enumerate(counts.tolist()):
        slots[row, :count] = torch.randperm(SLOTS, device="cuda")[:count].to(
            torch.int32
        )
    return q, kv, slots, counts


def _run(mode, solution, q, kv, slots, counts, *, return_lse, width=1, out=None):
    call = dsa_prefill if mode == "prefill" else dsa_decode
    kwargs = {"q_len_per_req": width} if mode == "decode" else {}
    if mode == "decode":
        q = q.view(-1, width, q.shape[-2], q.shape[-1])
    return call(
        q=q,
        kv_cache=kv,
        sparse_kv_cache=None,
        topk_slots=slots,
        topk_lens=counts,
        max_seqlen_k=SLOTS,
        qk_nope_head_dim=NOPE,
        kv_lora_rank=RANK,
        qk_rope_head_dim=ROPE,
        kv_seq_lens=torch.full_like(counts, SLOTS),
        softmax_scale=SCALE,
        page_size=64,
        return_lse=return_lse,
        out=out,
        solution=solution,
        slot_order="selection",
        **kwargs,
    )


def _assert_out_close(actual, expected, dtype):
    """BF16 matches Triton elementwise. With FP8 inputs the trtllm-gen core and
    Triton's BF16-dequantizing core round the probabilities differently, so FP8
    is held to a relative L2 bound instead."""
    actual, expected = actual.float(), expected.float()
    if dtype == torch.bfloat16:
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-3)
        return
    error = ((actual - expected).norm() / expected.norm().clamp_min(1e-12)).item()
    assert error < 3e-2, f"FP8 relative L2 error {error:.4f}"


def _merge(parts):
    """FP32 natural-log merge of ``(out, lse)`` partials, as the DCP combine:
    a partial with a -inf LSE contributes zero, whatever its output holds."""
    lse = torch.stack([p[1] for p in parts])
    total = torch.logsumexp(lse, dim=0)
    weights = torch.exp(lse - total).nan_to_num(0.0)
    out = sum(
        w[..., None] * p[0].float().masked_fill(torch.isneginf(p[1])[..., None], 0)
        for w, p in zip(weights, parts)
    )
    return out, total


@pytest.mark.parametrize("heads", [16, 96])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    ("mode", "width"), [("prefill", 1), ("decode", 1), ("decode", 5)]
)
def test_trtllm_lse_matches_triton(heads, dtype, mode, width):
    torch.manual_seed(1203)
    q, kv, slots, counts = _inputs(heads, dtype, rows=10 * width)
    args = (q, kv, slots, counts)
    out, lse = _run(mode, TRT, *args, return_lse=True, width=width)
    ref_out, ref_lse = _run(mode, "triton", *args, return_lse=True, width=width)
    # The DCP combine takes an LSE shaped like the output without its last dim.
    assert lse.shape == out.shape[:-1] and lse.dtype == torch.float32
    out = out.reshape(-1, heads, RANK)
    ref_out = ref_out.reshape(-1, heads, RANK)
    lse = lse.reshape(-1, heads)
    empty = counts == 0
    _assert_out_close(out[~empty], ref_out[~empty], dtype)
    torch.testing.assert_close(lse, ref_lse.reshape(-1, heads), rtol=1e-3, atol=2e-3)
    assert torch.all(torch.isneginf(lse[empty]))
    # Requesting the LSE leaves the output unchanged; empty rows occur only under KVP.
    plain = _run(mode, TRT, *args, return_lse=False, width=width)
    torch.testing.assert_close(
        plain.reshape(-1, heads, RANK)[~empty], out[~empty], rtol=0, atol=0
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_trtllm_lse_partials_merge_to_unsplit_attention(dtype):
    """Eight owners of disjoint slot subsets merge back to the whole row."""
    torch.manual_seed(1204)
    heads, ranks = 96, 8
    q, kv, slots, counts = _inputs(heads, dtype, rows=10)
    whole = _run("prefill", TRT, q, kv, slots, counts, return_lse=False)
    parts = []
    for rank in range(ranks):
        owned = torch.full_like(slots, -1)
        owned_counts = torch.zeros_like(counts)
        for row, count in enumerate(counts.tolist()):
            row_slots = slots[row, :count]
            mine = row_slots[(row_slots // 64) % ranks == rank]
            owned[row, : mine.numel()] = mine
            owned_counts[row] = mine.numel()
        parts.append(_run("prefill", TRT, q, kv, owned, owned_counts, return_lse=True))
    merged, _ = _merge(parts)
    rows = counts > 0
    assert torch.all(merged[~rows] == 0)
    _assert_out_close(merged[rows], whole[rows], dtype)


def test_trtllm_lse_cuda_graph_replay():
    torch.manual_seed(1205)
    heads = 96
    q, kv, slots, counts = _inputs(heads, torch.float8_e4m3fn, rows=10)
    out = torch.empty((10, heads, RANK), device="cuda", dtype=torch.bfloat16)
    _run("prefill", TRT, q, kv, slots, counts, return_lse=True, out=out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out, captured_lse = _run(
            "prefill", TRT, q, kv, slots, counts, return_lse=True, out=out
        )
    q.copy_((torch.randn(q.shape, device="cuda") * 0.5).to(q.dtype))
    graph.replay()
    torch.cuda.synchronize()
    assert captured_out is out
    ref_out, ref_lse = _run("prefill", "triton", q, kv, slots, counts, return_lse=True)
    rows = counts > 0
    _assert_out_close(out[rows], ref_out[rows], torch.float8_e4m3fn)
    torch.testing.assert_close(captured_lse, ref_lse, rtol=1e-3, atol=2e-3)


@pytest.mark.parametrize("rows", [4096, 16384])
def test_trtllm_lse_full_prefill_chunk(rows):
    """A whole 16K KVP prefill chunk: the LSE path's query-token chunking keeps
    FlashInfer's softmax stats inside the workspace, and sampled rows (chunk
    boundaries included) match Triton."""
    torch.manual_seed(1206)
    heads = 96
    q = (torch.randn((rows, heads, RANK + ROPE), device="cuda") * 0.5).to(
        torch.float8_e4m3fn
    )
    kv = (torch.randn(SLOTS, RANK + ROPE, device="cuda") * 0.25).to(torch.float8_e4m3fn)
    # Owned counts as one KVP rank sees them: about an eighth of 2048, some 0.
    counts = torch.randint(0, 2 * TOPK // 8, (rows,), device="cuda", dtype=torch.int32)
    order = torch.argsort(torch.rand((rows, SLOTS), device="cuda"), dim=-1)
    slots = order[:, :TOPK].to(torch.int32)
    slots[torch.arange(TOPK, device="cuda")[None, :] >= counts[:, None]] = -1
    out, lse = _run("prefill", TRT, q, kv, slots, counts, return_lse=True)
    assert torch.isfinite(out[counts > 0]).all()
    # 4096-token LSE chunks at 128 kernel heads.
    edges = [edge for edge in (0, 4095, 4096, 4097, rows - 1) if edge < rows]
    sample = torch.cat(
        [torch.randperm(rows, device="cuda")[:64], torch.tensor(edges, device="cuda")]
    )
    sample = sample[counts[sample] > 0]
    ref_out, ref_lse = _run(
        "prefill",
        "triton",
        q[sample].contiguous(),
        kv,
        slots[sample].contiguous(),
        counts[sample].contiguous(),
        return_lse=True,
    )
    _assert_out_close(out[sample], ref_out, torch.float8_e4m3fn)
    torch.testing.assert_close(lse[sample], ref_lse, rtol=1e-3, atol=2e-3)


def test_trtllm_lse_prepadded_query_matches_internal_padding():
    """A caller-built 128-head query, zero after the 96 live heads (as
    gather_query_heads_padded lands it), gives the leaf's own padding's
    partials bit for bit; the padded call's partials come back as the view of
    the 128-head rows, uncopied."""
    torch.manual_seed(1208)
    heads = 96
    q, kv, slots, counts = _inputs(heads, torch.float8_e4m3fn, rows=10)
    padded = torch.zeros((10, 128, RANK + ROPE), device="cuda", dtype=q.dtype)
    padded[:, :heads].copy_(q)
    out, lse = _run("prefill", TRT, q, kv, slots, counts, return_lse=True)
    full_out, full_lse = _run(
        "prefill", TRT, padded, kv, slots, counts, return_lse=True
    )
    assert out.stride() == (128 * RANK, RANK, 1)
    rows = counts > 0
    assert torch.equal(full_out[:, :heads][rows], out[rows])
    torch.testing.assert_close(full_lse[:, :heads], lse, rtol=0, atol=0)


def test_reserved_lse_buffers_serve_every_later_call():
    """Reserved before any forward, the LSE workspace and counter are the ones
    every later LSE call runs on, a full 16K prefill chunk's four passes and a
    verify-width decode included: no call allocates, grows or retires either."""
    from tokenspeed_kernel.ops.attention.dsa import flashinfer as leaf

    leaf.reserve_flashinfer_trtllm_dsa_buffers("cuda", num_heads=96, return_lse=True)
    device = torch.device("cuda", torch.cuda.current_device())
    workspace = leaf._dsa_sparse_workspace_buffers[(device, True)]
    counter = leaf._dsa_sparse_counter_buffers[device]
    retired = len(leaf._retired_dsa_sparse_counter_buffers)
    leaf.reserve_flashinfer_trtllm_dsa_buffers(device, num_heads=96, return_lse=True)
    torch.manual_seed(1209)
    for rows in (16384, 10):
        q, kv, slots, counts = _inputs(96, torch.float8_e4m3fn, rows=rows)
        _run("prefill", TRT, q, kv, slots, counts, return_lse=True)
    q, kv, slots, counts = _inputs(96, torch.float8_e4m3fn, rows=10)
    _run("decode", TRT, q, kv, slots, counts, return_lse=True, width=5)
    assert leaf._dsa_sparse_workspace_buffers[(device, True)] is workspace
    assert leaf._dsa_sparse_counter_buffers[device] is counter
    assert len(leaf._retired_dsa_sparse_counter_buffers) == retired


def test_trtllm_lse_inside_startup_autotune():
    """The startup forward runs in tuning mode, where FlashInfer would profile
    MLA decode batches up to 8192 whatever the chunk: 2 GiB of LSE stats at
    128 heads. The LSE leaf must take the cached or heuristic tactic instead."""
    from tokenspeed_kernel.ops.tuning import autotune

    torch.manual_seed(1207)
    q, kv, slots, counts = _inputs(96, torch.float8_e4m3fn, rows=10)
    with autotune(tune_mode=True, tuning_buckets=None, round_up=None):
        out, lse = _run("prefill", TRT, q, kv, slots, counts, return_lse=True)
    ref_out, ref_lse = _run("prefill", "triton", q, kv, slots, counts, return_lse=True)
    rows = counts > 0
    _assert_out_close(out[rows], ref_out[rows], torch.float8_e4m3fn)
    torch.testing.assert_close(lse, ref_lse, rtol=1e-3, atol=2e-3)


def test_kvp_selection_prefers_trtllm_with_lse():
    """With an LSE requested (KVP), the SPECIALIZED trtllm leaf outranks Triton."""
    q, kv, slots, counts = _inputs(96, torch.float8_e4m3fn, rows=10)
    result = _run("prefill", None, q, kv, slots, counts, return_lse=True)
    ref_out, ref_lse = _run("prefill", TRT, q, kv, slots, counts, return_lse=True)
    # trtllm-gen leaves NaN in an empty row's output (its LSE is -inf).
    torch.testing.assert_close(result[0], ref_out, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(result[1], ref_lse, rtol=0, atol=0)
