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

"""Draft-tree attention on the trtllm, trtllm_mla and tokenspeed_mla leaves
(docs/design/tree-speculation.md): the leaf's causal prefix decode plus
``tree_window_attention``, reached through the verify and lane dispatch,
against an fp32 reference."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

from tokenspeed.runtime.layers.attention.backends.paged.tokenspeed_mla import (  # noqa: E402
    CuteDSLMLABackend,
)
from tokenspeed.runtime.layers.attention.backends.paged.tree_verify import (  # noqa: E402
    TreeDraftInputs,
    TreeVerifyInputs,
)
from tokenspeed.runtime.layers.attention.backends.paged.trtllm import (  # noqa: E402
    TRTLLMMHAAttnBackend,
)
from tokenspeed.runtime.layers.attention.backends.paged.trtllm_mla import (  # noqa: E402
    TRTLLMMLABackend,
)
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig  # noqa: E402
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig  # noqa: E402
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig  # noqa: E402

register_cuda_ci(
    est_time=30,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="TRT-LLM tree attention requires an NVIDIA GPU.",
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="needs an NVIDIA GPU",
)

PAGE = 64
HQ, HKV, D = 16, 4, 128


def _backend(
    num_draft_tokens: int, max_bs: int, is_draft: bool, kv_dtype: torch.dtype
) -> TRTLLMMHAAttnBackend:
    spec = MHAConfig(
        backend_name="trtllm",
        num_attention_heads=HQ,
        num_kv_heads=HKV,
        head_dim=D,
        attn_tp_size=1,
    )
    cfg = AttnConfig(
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_dtype=kv_dtype,
        prefix_granularity=PAGE,
        kernel_page_size=PAGE,
        context_len=2048,
        max_bs=max_bs,
        kv_cache_quant_method="none",
        speculative_num_steps=3,
        speculative_num_draft_tokens=num_draft_tokens,
        is_draft=is_draft,
        components=(spec,),
    )
    backend = TRTLLMMHAAttnBackend(cfg, spec, kernel_page_size=PAGE)
    backend.init_cuda_graph_state(max_bs)
    return backend


def _layer():
    return SimpleNamespace(
        layer_id=0,
        tp_q_head_num=HQ,
        tp_k_head_num=HKV,
        tp_v_head_num=HKV,
        head_dim=D,
        scaling=D**-0.5,
        sliding_window_size=-1,
    )


def _problem(seq_lens, rows, window, max_pages, gen, kv_dtype, poison=None):
    """Paged K/V rows for every request's keys, queries, window masks, and the
    fp32 reference: row r sees the prefix and window key j when bit j is set.
    Under an FP8 cache, K/V hold FP8 values and the keys trtllm-gen's causal
    prefix covers (row r: the first ``prefix - rows + 1 + r``) see the query
    cast to FP8, as that decode does; the window kernel sees it unquantized."""
    bs = len(seq_lens)
    pages = (max(seq_lens) + PAGE - 1) // PAGE
    table = torch.zeros(bs, max_pages, dtype=torch.int32)
    table[:, :pages] = (torch.randperm(bs * pages, generator=gen) + 1).view(bs, pages)
    rows_shape = ((bs * pages + 1) * PAGE, HKV, D)
    k_rows = torch.randn(rows_shape, generator=gen).to(kv_dtype).bfloat16()
    v_rows = torch.randn(rows_shape, generator=gen).to(kv_dtype).bfloat16()
    q = torch.randn(bs * rows, HQ, D, generator=gen).bfloat16()
    q_prefix = q.to(kv_dtype).float()
    bits = torch.randint(0, 1 << 62, (bs * rows,), generator=gen)
    # With a poison, rows only see step 1's slots (later lane steps unwritten yet).
    seen_width = window if poison is None else rows
    if seen_width < 62:
        bits &= (1 << seen_width) - 1
    ref = torch.empty(bs * rows, HQ, D)
    for b, length in enumerate(seq_lens):
        pos = torch.arange(length)
        slots = (table[b, pos // PAGE].long() * PAGE + pos % PAGE).tolist()
        kk = k_rows[slots].float().repeat_interleave(HQ // HKV, 1)
        vv = v_rows[slots].float().repeat_interleave(HQ // HKV, 1)
        prefix = length - window
        for i in range(rows):
            vis = torch.tensor(
                [
                    j < prefix or bool((int(bits[b * rows + i]) >> (j - prefix)) & 1)
                    for j in range(length)
                ]
            )
            covered = torch.arange(length) < prefix - rows + 1 + i
            s = torch.where(
                covered[None],
                torch.einsum("hd,xhd->hx", q_prefix[b * rows + i], kk),
                torch.einsum("hd,xhd->hx", q[b * rows + i].float(), kk),
            )
            s = (s * D**-0.5).masked_fill(~vis[None], float("-inf"))
            ref[b * rows + i] = torch.einsum("hx,xhd->hd", torch.softmax(s, -1), vv)
    if poison is not None:
        # Window slots no row of the request sees get a non-finite V after the reference.
        for b, length in enumerate(seq_lens):
            seen = 0
            for i in range(rows):
                seen |= int(bits[b * rows + i])
            # trtllm-gen loads the prefix's last page whole, so only later pages are poisoned here.
            prefix = length - window
            for j in range(rows, window):
                pos = prefix + j
                if not (seen >> j) & 1 and pos // PAGE > (prefix - 1) // PAGE:
                    v_rows[int(table[b, pos // PAGE]) * PAGE + pos % PAGE] = poison
    k_rows, v_rows = k_rows.cuda().to(kv_dtype), v_rows.cuda().to(kv_dtype)
    pool = SimpleNamespace(get_kv_buffer=lambda _: (k_rows, v_rows))
    return table.cuda(), q.cuda(), bits.cuda(), pool, ref


KV_DTYPES = [torch.bfloat16, torch.float8_e4m3fn]
# trtllm-gen's FP8 decode alone errs up to 0.07 over a few keys (bf16: within 0.02).
ATOL = {torch.bfloat16: 2e-2, torch.float8_e4m3fn: 8e-2}


@pytest.mark.parametrize("kv_dtype", KV_DTYPES)
@pytest.mark.parametrize("nodes", [4, 16, 64])
def test_tree_verify_matches_reference(nodes, kv_dtype):
    gen = torch.Generator().manual_seed(nodes)
    # Long and short committed prefixes, including ones shorter than the window.
    committed = [700, 1, nodes - 2, 130]
    seq_lens = [c + nodes for c in committed]
    backend = _backend(nodes, max_bs=8, is_draft=False, kv_dtype=kv_dtype)
    table, q, bits, pool, ref = _problem(
        seq_lens, nodes, nodes, backend.max_num_pages, gen, kv_dtype
    )
    mask = torch.zeros(8 * nodes, dtype=torch.int64, device="cuda")
    mask[: len(seq_lens) * nodes] = bits
    backend.bind_tree_verify(
        TreeVerifyInputs(mask, nodes, torch.zeros(8, nodes, dtype=torch.int32))
    )
    bs = len(seq_lens)
    backend.refresh_decode_metadata(
        bs, bs, torch.tensor(seq_lens, dtype=torch.int32, device="cuda"), table
    )
    out = backend.forward_decode(q, None, None, _layer(), None, pool, bs)
    torch.testing.assert_close(
        out.view(-1, HQ, D).float().cpu(), ref, atol=ATOL[kv_dtype], rtol=2e-2
    )


@pytest.mark.parametrize("kv_dtype", KV_DTYPES)
@pytest.mark.parametrize("poison", [None, float("nan")])
@pytest.mark.parametrize("topk,steps", [(2, 7), (4, 4), (8, 5)])
def test_tree_lanes_match_reference(topk, steps, poison, kv_dtype):
    gen = torch.Generator().manual_seed(topk * 10 + steps)
    window = (steps - 1) * topk
    frontier = [700, 1, 3, 130]
    seq_lens = [f + window for f in frontier]
    backend = _backend(window, max_bs=8, is_draft=True, kv_dtype=kv_dtype)
    table, q, bits, pool, ref = _problem(
        seq_lens, topk, window, backend.max_num_pages, gen, kv_dtype, poison
    )
    lanes = TreeDraftInputs(topk, steps, 8, torch.device("cuda"))
    backend.bind_tree_draft(lanes)
    bs = len(frontier)
    lanes.set_frontier(bs, torch.tensor(frontier, dtype=torch.int32, device="cuda"))
    lanes.lane_mask[: bs * topk] = bits
    backend.refresh_decode_metadata(
        bs, bs, torch.tensor(frontier, dtype=torch.int32, device="cuda"), table
    )
    lanes.active = True
    out = backend.forward_decode(q, None, None, _layer(), None, pool, bs)
    lanes.active = False
    torch.testing.assert_close(
        out.view(-1, HQ, D).float().cpu(), ref, atol=ATOL[kv_dtype], rtol=2e-2
    )


# MLA: Kimi K2.5 at TP4 (16 heads), one latent row of 512 + 64 rotary channels per token.
MLA_HEADS, LATENT, ROPE = 16, 512, 64
MLA_LEAVES = [TRTLLMMLABackend, CuteDSLMLABackend]


def _mla_backend(cls, num_draft_tokens, max_bs, is_draft, kv_dtype):
    spec = MLAConfig(
        backend_name="trtllm_mla" if cls is TRTLLMMLABackend else "tokenspeed_mla",
        num_attention_heads=MLA_HEADS,
        num_kv_heads=MLA_HEADS,
        head_dim=128 + ROPE,
        attn_tp_size=1,
        kv_lora_rank=LATENT,
        qk_nope_head_dim=128,
        qk_rope_head_dim=ROPE,
        v_head_dim=128,
        scaling=(128 + ROPE) ** -0.5,
        kv_cache_dim=LATENT + ROPE,
    )
    cfg = AttnConfig(
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_dtype=kv_dtype,
        prefix_granularity=PAGE,
        kernel_page_size=PAGE,
        context_len=2048,
        max_bs=max_bs,
        kv_cache_quant_method="none",
        speculative_num_steps=3,
        speculative_num_draft_tokens=num_draft_tokens,
        is_draft=is_draft,
        components=(spec,),
    )
    backend = cls(cfg, spec, kernel_page_size=PAGE)
    backend.init_cuda_graph_state(max_bs)
    return backend


def _mla_layer():
    return SimpleNamespace(
        layer_id=0,
        tp_q_head_num=MLA_HEADS,
        tp_k_head_num=1,
        head_dim=LATENT + ROPE,
        v_head_dim=LATENT,
        scaling=(128 + ROPE) ** -0.5,
        sliding_window_size=-1,
    )


def _mla_problem(seq_lens, rows, window, max_pages, gen, kv_dtype, query_dtype):
    """Paged latent rows, absorbed queries, window masks and the fp32 reference
    (the value is each key row's leading latent). Under an FP8 cache the keys
    the causal prefix covers see the query cast to FP8, as that decode does;
    an FP8 query reaches the backend already cast."""
    bs = len(seq_lens)
    pages = (max(seq_lens) + PAGE - 1) // PAGE
    table = torch.zeros(bs, max_pages, dtype=torch.int32)
    table[:, :pages] = (torch.randperm(bs * pages, generator=gen) + 1).view(bs, pages)
    latent = torch.randn(((bs * pages + 1) * PAGE, 1, LATENT + ROPE), generator=gen)
    latent = latent.to(kv_dtype).bfloat16()
    q = torch.randn(bs * rows, MLA_HEADS, LATENT + ROPE, generator=gen)
    q = q.to(query_dtype).bfloat16()
    q_prefix = q.to(kv_dtype).float()
    bits = torch.randint(0, 1 << 62, (bs * rows,), generator=gen)
    if window < 62:
        bits &= (1 << window) - 1
    scale = (128 + ROPE) ** -0.5
    ref = torch.empty(bs * rows, MLA_HEADS, LATENT)
    for b, length in enumerate(seq_lens):
        pos = torch.arange(length)
        keys = latent[(table[b, pos // PAGE].long() * PAGE + pos % PAGE), 0].float()
        prefix = length - window
        for i in range(rows):
            vis = torch.tensor(
                [
                    j < prefix or bool((int(bits[b * rows + i]) >> (j - prefix)) & 1)
                    for j in range(length)
                ]
            )
            covered = torch.arange(length) < prefix - rows + 1 + i
            s = torch.where(
                covered[None],
                q_prefix[b * rows + i] @ keys.T,
                q[b * rows + i].float() @ keys.T,
            )
            s = (s * scale).masked_fill(~vis[None], float("-inf"))
            ref[b * rows + i] = torch.softmax(s, -1) @ keys[:, :LATENT]
    rows_gpu = latent.cuda().to(kv_dtype)
    pool = SimpleNamespace(get_key_buffer=lambda _: rows_gpu)
    return table.cuda(), q.cuda().to(query_dtype), bits.cuda(), pool, ref


@pytest.mark.parametrize("cls", MLA_LEAVES)
@pytest.mark.parametrize(
    "kv_dtype,query_dtype",
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.float8_e4m3fn, torch.bfloat16),
        (torch.float8_e4m3fn, torch.float8_e4m3fn),
    ],
)
@pytest.mark.parametrize("nodes", [4, 16, 64])
def test_mla_tree_verify_matches_reference(cls, nodes, kv_dtype, query_dtype):
    gen = torch.Generator().manual_seed(nodes + 1)
    committed = [700, 1, nodes - 2, 130]
    seq_lens = [c + nodes for c in committed]
    backend = _mla_backend(cls, nodes, max_bs=8, is_draft=False, kv_dtype=kv_dtype)
    table, q, bits, pool, ref = _mla_problem(
        seq_lens, nodes, nodes, backend.max_num_pages, gen, kv_dtype, query_dtype
    )
    mask = torch.zeros(8 * nodes, dtype=torch.int64, device="cuda")
    mask[: len(seq_lens) * nodes] = bits
    backend.bind_tree_verify(
        TreeVerifyInputs(mask, nodes, torch.zeros(8, nodes, dtype=torch.int32))
    )
    bs = len(seq_lens)
    backend.refresh_decode_metadata(
        bs, bs, torch.tensor(seq_lens, dtype=torch.int32, device="cuda"), table
    )
    out = backend.forward_decode(q, None, None, _mla_layer(), None, pool, bs)
    torch.testing.assert_close(
        out.view(-1, MLA_HEADS, LATENT).float().cpu(),
        ref,
        atol=ATOL[kv_dtype],
        rtol=2e-2,
    )


@pytest.mark.parametrize("cls", MLA_LEAVES)
@pytest.mark.parametrize("kv_dtype", KV_DTYPES)
@pytest.mark.parametrize("topk,steps", [(2, 7), (4, 4), (8, 5)])
def test_mla_tree_lanes_match_reference(cls, topk, steps, kv_dtype):
    gen = torch.Generator().manual_seed(topk * 10 + steps + 1)
    window = (steps - 1) * topk
    frontier = [700, 1, 3, 130]
    seq_lens = [f + window for f in frontier]
    backend = _mla_backend(cls, window, max_bs=8, is_draft=True, kv_dtype=kv_dtype)
    table, q, bits, pool, ref = _mla_problem(
        seq_lens, topk, window, backend.max_num_pages, gen, kv_dtype, torch.bfloat16
    )
    lanes = TreeDraftInputs(topk, steps, 8, torch.device("cuda"))
    backend.bind_tree_draft(lanes)
    bs = len(frontier)
    lanes.set_frontier(bs, torch.tensor(frontier, dtype=torch.int32, device="cuda"))
    lanes.lane_mask[: bs * topk] = bits
    backend.refresh_decode_metadata(
        bs, bs, torch.tensor(frontier, dtype=torch.int32, device="cuda"), table
    )
    lanes.active = True
    out = backend.forward_decode(q, None, None, _mla_layer(), None, pool, bs)
    lanes.active = False
    torch.testing.assert_close(
        out.view(-1, MLA_HEADS, LATENT).float().cpu(),
        ref,
        atol=ATOL[kv_dtype],
        rtol=2e-2,
    )


@pytest.mark.parametrize("cls", [TRTLLMMHAAttnBackend, CuteDSLMLABackend])
def test_tree_verify_refuses_a_sliding_window_layer(cls):
    """Sliding windows are per layer, so the forward refuses them, not tree_support."""
    gen = torch.Generator().manual_seed(7)
    nodes, seq_lens = 4, [40, 50]
    if cls is TRTLLMMHAAttnBackend:
        backend = _backend(nodes, max_bs=8, is_draft=False, kv_dtype=torch.bfloat16)
        table, q, _, pool, _ = _problem(
            seq_lens, nodes, nodes, backend.max_num_pages, gen, torch.bfloat16
        )
        layer = _layer()
    else:
        backend = _mla_backend(cls, nodes, 8, False, torch.bfloat16)
        table, q, _, pool, _ = _mla_problem(
            seq_lens,
            nodes,
            nodes,
            backend.max_num_pages,
            gen,
            torch.bfloat16,
            torch.bfloat16,
        )
        layer = _mla_layer()
    layer.sliding_window_size = 16
    backend.bind_tree_verify(
        TreeVerifyInputs(
            torch.zeros(8 * nodes, dtype=torch.int64, device="cuda"),
            nodes,
            torch.zeros(8, nodes, dtype=torch.int32),
        )
    )
    bs = len(seq_lens)
    backend.refresh_decode_metadata(
        bs, bs, torch.tensor(seq_lens, dtype=torch.int32, device="cuda"), table
    )
    with pytest.raises(NotImplementedError, match="sliding-window"):
        backend.forward_decode(q, None, None, layer, None, pool, bs)
