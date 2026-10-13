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

"""Triton split decode over a dense BF16 KV cache with few heads per row.

HY V4 at TP32 decodes 2 heads per rank with a learnable sink. The split decode
attends every row's selected columns in slices that carry all of the row's
heads, then merges the slices by their LSE and applies the sink once. It must
match an FP32 reference and the FlashMLA path it replaces, and be the kernel
that dense-KV decode with a sink selects for such head counts on Hopper.
"""

import pytest
import torch
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

platform = current_platform()

pytestmark = pytest.mark.skipif(
    not (platform.is_nvidia and platform.is_hopper),
    reason="The Triton split DSA decode registers on Hopper NVIDIA GPUs only",
)

if platform.is_nvidia and platform.is_hopper:
    from tokenspeed_kernel.ops.attention.dsa import dsa_decode
    from tokenspeed_kernel.ops.attention.dsa import triton as dsa_triton

KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 192
QK_ROPE_HEAD_DIM = 64
ROW = KV_LORA_RANK + QK_ROPE_HEAD_DIM
TOPK = 2048
SLOTS = 8192
SCALE = ROW**-0.5

CASES = {
    "decode": dict(lens=[TOPK]),
    "mtp3": dict(lens=[TOPK - 3, TOPK - 2, TOPK - 1, TOPK], q_len=4),
    "short": dict(lens=[1, 31, 32, 33, 700, 1500]),
    "empty_rows": dict(lens=[0, 5, 0, TOPK]),
    "invalid_slots": dict(lens=[TOPK, 1200, 300, 64], invalid=0.4),
    "batch": dict(lens=[TOPK, 1, 900, 0, 2047, 64, 33, 1500] * 8),
}


def _inputs(lens, heads, invalid=0.0, topk=TOPK):
    generator = torch.Generator(device="cuda").manual_seed(len(lens) * 31 + heads)
    rows = len(lens)
    kv = (torch.randn(SLOTS, 1, ROW, device="cuda", generator=generator) * 0.5).to(
        torch.bfloat16
    )
    q = torch.randn(rows, heads, ROW, device="cuda", generator=generator).to(
        torch.bfloat16
    )
    slots = torch.stack(
        [
            torch.randperm(SLOTS, device="cuda", generator=generator)[:topk]
            for _ in range(rows)
        ]
    ).to(torch.int32)
    if invalid:
        slots[torch.rand(slots.shape, device="cuda", generator=generator) < invalid] = (
            -1
        )
    # Slots past a row's length stay valid: the kernel must not read them.
    lens = torch.tensor(lens, device="cuda", dtype=torch.int32)
    sink = torch.randn(heads, device="cuda", generator=generator)
    return q, kv, slots, lens, sink


def _reference(q, kv, slots, lens, sink):
    """FP32 attention over every row's live slots; the sink is one more logit."""
    if lens is not None:
        columns = torch.arange(slots.shape[1], device="cuda")
        slots = slots.masked_fill(columns >= lens[:, None], -1)
    keys_all = kv[:, 0].float()
    heads = q.shape[1]
    out = torch.zeros(q.shape[0], heads, KV_LORA_RANK, device="cuda")
    lse = torch.full((q.shape[0], heads), float("-inf"), device="cuda")
    for row in range(q.shape[0]):
        selected = slots[row][slots[row] >= 0].long()
        if selected.numel() == 0:
            continue
        keys = keys_all[selected]
        logits = q[row].float() @ keys.T * SCALE
        if sink is not None:
            logits = torch.cat([logits, sink[:, None]], dim=1)
        lse[row] = torch.logsumexp(logits, dim=1)
        probs = torch.softmax(logits, dim=1)[:, : selected.numel()]
        out[row] = probs @ keys[:, :KV_LORA_RANK]
    return out, lse


def _decode(q, kv, slots, lens, sink, *, q_len=1, **kwargs):
    return dsa_decode(
        q=q,
        kv_cache=kv,
        sparse_kv_cache=None,
        topk_slots=slots,
        topk_lens=lens,
        max_seqlen_k=SLOTS,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        softmax_scale=SCALE,
        page_size=64,
        q_len_per_req=q_len,
        slot_order="selection",
        attn_sink=sink,
        **kwargs,
    )


def _split_decode(q, kv, slots, lens, sink, **kwargs):
    return dsa_triton.triton_dsa_split_decode(
        q=q,
        kv_cache=kv,
        sparse_kv_cache=None,
        topk_slots=slots,
        topk_lens=lens,
        max_seqlen_k=SLOTS,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        softmax_scale=SCALE,
        page_size=64,
        attn_sink=sink,
        **kwargs,
    )


def _assert_matches_reference(out, lse, ref_out, ref_lse):
    # The probabilities and the per-split partials are rounded to BF16 before
    # the FP32 merge.
    torch.testing.assert_close(out.float(), ref_out, rtol=1e-2, atol=4e-3)
    finite = torch.isfinite(ref_lse)
    torch.testing.assert_close(lse[finite], ref_lse[finite], rtol=0.0, atol=1e-4)
    assert torch.isneginf(lse[~finite]).all()
    assert (out[~finite.all(dim=1)] == 0).all()


def _selected_name(heads, *, sinks, return_lse=True):
    return select_kernel(
        "attention",
        "dsa_decode",
        format_signature(q=dense_tensor_format(torch.bfloat16)),
        traits={
            "q_len": 1,
            "num_heads": heads,
            "qk_nope_head_dim": QK_NOPE_HEAD_DIM,
            "kv_lora_rank": KV_LORA_RANK,
            "qk_rope_head_dim": QK_ROPE_HEAD_DIM,
            "page_size": 64,
            "topk": TOPK,
            "has_kv_cache": True,
            "has_sparse_kv_cache": False,
            "logit_cap": False,
            "return_lse": return_lse,
            "sinks": sinks,
            "topk_layout": "global_slots",
        },
    ).name


@pytest.mark.parametrize("heads", [1, 2, 4, 8, 16])
def test_dense_decode_with_a_sink_selects_the_split_decode(heads):
    assert _selected_name(heads, sinks=True) == "triton_dsa_split_decode"
    assert _selected_name(heads, sinks=True, return_lse=False) == (
        "triton_dsa_split_decode"
    )


@pytest.mark.parametrize("heads", [32, 64])
def test_wide_rows_keep_the_flashmla_decode(heads):
    assert _selected_name(heads, sinks=True) == "flashmla_dense_dsa_decode"


def test_sinkless_partial_decode_keeps_its_kernel():
    assert _selected_name(2, sinks=False) == "flashmla_dense_dsa_partial_decode"


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("heads", [1, 2, 16])
def test_split_decode_matches_reference(case, heads):
    spec = CASES[case]
    q, kv, slots, lens, sink = _inputs(spec["lens"], heads, spec.get("invalid", 0.0))
    out, lse = _decode(
        q, kv, slots, lens, sink, q_len=spec.get("q_len", 1), return_lse=True
    )
    assert out.shape == (q.shape[0], heads, KV_LORA_RANK)
    assert out.dtype == torch.bfloat16
    _assert_matches_reference(out, lse, *_reference(q, kv, slots, lens, sink))


@pytest.mark.parametrize("case", ["decode", "mtp3", "invalid_slots", "batch"])
def test_split_decode_matches_flashmla(case):
    spec = CASES[case]
    q, kv, slots, lens, sink = _inputs(spec["lens"], 2, spec.get("invalid", 0.0))
    kwargs = dict(q_len=spec.get("q_len", 1), return_lse=True)
    out, lse = _decode(q, kv, slots, lens, sink, **kwargs)
    ref_out, ref_lse = _decode(q, kv, slots, lens, sink, solution="flashmla", **kwargs)
    # Both round their split partials to BF16, at different split counts.
    torch.testing.assert_close(out.float(), ref_out.float(), rtol=1e-2, atol=4e-3)
    finite = torch.isfinite(ref_lse)
    torch.testing.assert_close(lse[finite], ref_lse[finite], rtol=0.0, atol=1e-4)
    assert torch.isneginf(lse[~finite]).all()


def test_split_decode_without_a_sink_matches_reference():
    q, kv, slots, lens, _ = _inputs(CASES["batch"]["lens"], 2)
    out, lse = _split_decode(q, kv, slots, lens, None, return_lse=True)
    _assert_matches_reference(out, lse, *_reference(q, kv, slots, lens, None))


def test_split_decode_without_topk_lens_spans_the_padded_width():
    q, kv, slots, _, sink = _inputs([TOPK] * 4, 2, 0.4)
    slots[1] = -1
    out, lse = _split_decode(q, kv, slots, None, sink, return_lse=True)
    _assert_matches_reference(out, lse, *_reference(q, kv, slots, None, sink))
    assert torch.isneginf(lse[1]).all() and not out[1].any()


@pytest.mark.parametrize("topk", [512, 1024])
def test_split_decode_narrow_selections(topk):
    q, kv, slots, lens, sink = _inputs([topk, topk - 1, 17, 0], 2, 0.1, topk=topk)
    out, lse = _decode(q, kv, slots, lens, sink, return_lse=True)
    _assert_matches_reference(out, lse, *_reference(q, kv, slots, lens, sink))


def test_split_decode_writes_the_supplied_output_buffer():
    q, kv, slots, lens, sink = _inputs([TOPK, 900], 2)
    out = torch.full((2, 2 * KV_LORA_RANK), 7.0, device="cuda", dtype=q.dtype)
    result = _decode(q, kv, slots, lens, sink, out=out)
    assert result is out
    ref_out, _ = _reference(q, kv, slots, lens, sink)
    torch.testing.assert_close(
        out.view_as(ref_out).float(), ref_out, rtol=1e-2, atol=4e-3
    )


def test_split_decode_replays_in_a_cuda_graph():
    q, kv, slots, lens, sink = _inputs([TOPK, 900, 0, 64], 2, 0.2)
    _decode(q, kv, slots, lens, sink, return_lse=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out, graph_lse = _decode(q, kv, slots, lens, sink, return_lse=True)
    fresh_q, _, fresh_slots, fresh_lens, _ = _inputs([5, TOPK, 1200, 0], 2, 0.1)
    q.copy_(fresh_q)
    slots.copy_(fresh_slots)
    lens.copy_(fresh_lens)
    graph.replay()
    eager_out, eager_lse = _decode(q, kv, slots, lens, sink, return_lse=True)
    assert torch.equal(graph_out, eager_out)
    assert torch.equal(graph_lse, eager_lse)


@pytest.mark.parametrize("tokens", [1, 2, 4, 8, 31, 64, 78, 200, 1024])
@pytest.mark.parametrize("topk", [512, 1024, 2048])
def test_splits_fill_the_program_budget(tokens, topk):
    min_columns = dsa_triton._SPLIT_DECODE_MIN_SPLIT_COLUMNS
    budget = platform.sm_count * dsa_triton._SPLIT_DECODE_PROGRAMS_PER_SM
    splits = dsa_triton._split_decode_splits(tokens, topk)
    assert splits >= 1 and splits & (splits - 1) == 0
    assert splits * min_columns <= topk
    if splits > 1:
        assert tokens * splits <= budget
    # Doubling again would overflow the budget or narrow a split too far.
    assert tokens * splits * 2 > budget or splits * 2 * min_columns > topk


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


@pytest.mark.parametrize("splits", [2, 4, 32])
def test_splits_share_each_rows_live_columns(splits):
    lens = [TOPK, 1100, 600, 33, 1, 0]
    q, kv, slots, lens_t, _ = _inputs(lens, 2)
    _, _, split_lens = dsa_triton._split_decode_partials(
        q,
        kv[:, 0],
        slots,
        lens_t,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        softmax_scale=SCALE,
        splits=splits,
    )
    block = dsa_triton._SPLIT_DECODE_BLOCK_N
    for live, parts in zip(lens, split_lens.view(len(lens), splits).tolist()):
        assert sum(parts) == live
        # No split holds more than an even share rounded up to whole blocks, so
        # a row far shorter than its padded width still spreads over them.
        assert max(parts) <= _cdiv(_cdiv(live, splits), block) * block


def test_registration_is_gated_to_few_heads_with_a_sink():
    spec = KernelRegistry.get().get_by_name("triton_dsa_split_decode")
    assert spec is not None
    assert spec.traits["num_heads"] == frozenset({1, 2, 4, 8, 16})
    assert spec.traits["sinks"] == frozenset({True})
    assert spec.traits["has_sparse_kv_cache"] == frozenset({False})


@pytest.mark.parametrize("with_sink", [False, True])
def test_split_decode_preserves_nonfinite_live_partials(with_sink):
    """A poisoned live key must not become the empty-partial +inf sentinel."""
    q = torch.ones(3, 2, ROW, dtype=torch.bfloat16, device="cuda")
    kv = torch.ones(64, 1, ROW, dtype=torch.bfloat16, device="cuda")
    kv[0, 0, 0] = float("nan")
    slots = torch.full((3, TOPK), -1, dtype=torch.int32, device="cuda")
    slots[0, 0] = 0
    slots[1, 0] = 1
    lens = torch.tensor([1, 1, 0], dtype=torch.int32, device="cuda")
    sink = torch.zeros(2, device="cuda") if with_sink else None
    out, lse = _split_decode(q, kv, slots, lens, sink, return_lse=True)
    assert torch.isnan(lse[0]).all()
    assert torch.isnan(out[0]).any()
    assert torch.isfinite(lse[1]).all() and torch.isfinite(out[1]).all()
    assert torch.count_nonzero(out[1]) > 0
    assert torch.isneginf(lse[2]).all()
    assert torch.count_nonzero(out[2]) == 0
