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

"""A causal block draft (Kimi-K3.1 DSpark) on the MLA backend under KVP.

Without KVP each request expands to one decode entry per block position, entry
j backed up to ``block_end - (block - 1 - j)``. Under KVP the backend keeps one
block-end entry per request and the DCP decode derives the rows' causal
lengths at forward time. Every rank of the group is emulated on one GPU with
the real metadata and visible-length kernels; only the cross-rank exchange and
the attention kernel are stubbed, so the per-rank local lengths must sum to
exactly the expanded entries of the unsharded backend.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.layers.attention.backends.paged import mla as mla_module
from tokenspeed.runtime.layers.attention.backends.paged.mla import MLAAttnBackend
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the DCP metadata kernels need a GPU"
)

KV_LORA, ROPE = 512, 64
WIDTH = KV_LORA + ROPE
PAGE, GRANULARITY = 64, 128
PAGES_PER_BLOCK = GRANULARITY // PAGE
VIRTUAL_BLOCKS = 24
CONTEXT = 4096
HEADS = 12
BLOCK = 4
# Scheduler blocks per request (block 0 is the null block).
BLOCKS = ([5, 2, 8, 13, 21, 3, 17], [9, 4], [11])
# Live draft lengths after the block: one ending inside its last ownership
# block, a short one, and a padded graph row the fill must clamp to the block.
BLOCK_ENDS = (6 * GRANULARITY + 37, GRANULARITY + 12, 0)


def _kernel_page_table():
    # The router hands a leaf full-width [bs, max_num_pages] tables.
    table = torch.zeros(len(BLOCKS), CONTEXT // PAGE, dtype=torch.int32, device="cuda")
    for row, blocks in enumerate(BLOCKS):
        for i, block in enumerate(blocks):
            for sub in range(PAGES_PER_BLOCK):
                table[row, i * PAGES_PER_BLOCK + sub] = block * PAGES_PER_BLOCK + sub
    return table


def _backend(degree, rank):
    spec = MLAConfig(
        backend_name="mla",
        num_attention_heads=HEADS,
        num_kv_heads=1,
        head_dim=WIDTH,
        attn_tp_size=1,
        kv_lora_rank=KV_LORA,
        qk_nope_head_dim=128,
        qk_rope_head_dim=ROPE,
        v_head_dim=KV_LORA,
        kv_cache_dim=WIDTH,
        scaling=192**-0.5,
    )
    config = AttnConfig(
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        context_len=CONTEXT,
        max_bs=len(BLOCKS),
        components=(spec,),
        is_draft=True,
        speculative_num_draft_tokens=BLOCK,
        draft_block_decode=True,
        draft_block_causal=True,
        kvp_size=degree,
        kvp_rank=rank,
        kvp_group=tuple(range(degree)),
    )
    backend = MLAAttnBackend(config, spec, kernel_page_size=PAGE)
    if degree > 1:
        # configure_runtime's kernel-registry probe is platform specific;
        # the layout it publishes is all the decode metadata reads.
        backend.kvp_block_granularity = GRANULARITY
        backend.kvp_virtual_block_count = VIRTUAL_BLOCKS
    backend.init_cuda_graph_state(len(BLOCKS))
    return backend


def _publish(backend, table):
    bs = len(BLOCKS)
    backend.refresh_decode_metadata(
        bs, bs, torch.full((bs,), BLOCK, dtype=torch.int32, device="cuda"), table
    )
    # The drafter's in-graph writer; a replay sees only this.
    backend.fill_block_decode_seq_lens(
        bs, torch.tensor(BLOCK_ENDS, dtype=torch.int32, device="cuda")
    )


@pytest.mark.parametrize("degree", [2, 8])
def test_dcp_causal_block_rows_match_the_unsharded_entries(degree, monkeypatch):
    table = _kernel_page_table()
    reference = _backend(1, 0)
    _publish(reference, table)
    expected = reference.seq_lens_buf[: len(BLOCKS) * BLOCK].clone()
    ends = torch.tensor(BLOCK_ENDS, device="cuda").clamp_min(BLOCK)
    torch.testing.assert_close(
        expected.view(-1, BLOCK).long(),
        ends[:, None] - torch.arange(BLOCK - 1, -1, -1, device="cuda"),
        rtol=0,
        atol=0,
    )

    captured = {}

    def gather(q, group):
        return q.repeat(1, len(group), 1)

    def decode(*, q, cache_seqlens, **kwargs):
        captured["cache"] = cache_seqlens.clone()
        rows = q.shape[0]
        return (
            torch.zeros(rows, 1, q.shape[2], KV_LORA, device=q.device),
            torch.zeros(rows, q.shape[2], device=q.device),
        )

    def normalize(output, lse, local_lengths, sink):
        captured["local"] = local_lengths.clone()
        return output, lse

    def combine(output, lse, **kwargs):
        return output[:, :HEADS]

    monkeypatch.setattr(mla_module, "gather_query_heads_lent", gather)
    monkeypatch.setattr(mla_module, "mla_decode_with_kvcache", decode)
    monkeypatch.setattr(mla_module, "normalize_dcp_partials", normalize)
    monkeypatch.setattr(mla_module, "combine_attention_partials", combine)

    bs = len(BLOCKS)
    layer = SimpleNamespace(
        tp_q_head_num=HEADS,
        head_dim=WIDTH,
        v_head_dim=KV_LORA,
        layer_id=0,
        scaling=192**-0.5,
        logit_cap=0.0,
        sliding_window_size=-1,
    )
    pool = SimpleNamespace(
        get_key_buffer=lambda layer_id: torch.zeros(
            VIRTUAL_BLOCKS * GRANULARITY, WIDTH, dtype=torch.bfloat16, device="cuda"
        )
    )
    q = torch.zeros(bs * BLOCK, HEADS, WIDTH, dtype=torch.bfloat16, device="cuda")
    total = torch.zeros_like(expected)
    for rank in range(degree):
        backend = _backend(degree, rank)
        _publish(backend, table)
        backend.forward_decode(q, None, None, layer, None, pool, bs)
        local = captured["local"]
        assert local.shape == expected.shape
        groups = captured["cache"].numel() // local.numel()
        torch.testing.assert_close(
            captured["cache"],
            local.clamp_min(1).repeat_interleave(groups),
            rtol=0,
            atol=0,
        )
        total += local
    # Every visible token of every block row is owned by exactly one rank.
    torch.testing.assert_close(total, expected, rtol=0, atol=0)
