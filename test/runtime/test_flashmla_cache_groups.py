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

"""FlashMLABackend cache-group (LCM) decode metadata.

Validates that FlashMLA, when bound to a cache contract, resolves its
decode block table and latent write locations from the LCM full-history table
rather than the classic ``page_table`` path. Exercises the metadata math only
(no FlashMLA CUDA kernel), so it runs on any CUDA device.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))

from test.runtime.conftest import MLA_KV_LORA_RANK as _KV_LORA_RANK
from test.runtime.conftest import MLA_LATENT_DIM as _LATENT_DIM
from test.runtime.conftest import MLA_QK_ROPE_DIM as _QK_ROPE_DIM
from test.runtime.conftest import _poison
from test.runtime.conftest import make_kimi_pool as _make_pool
from test.runtime.conftest import mla_layer_id as _mla_layer_id
from test.runtime.conftest import requires_cuda

from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="runtime-1gpu")

_KERNEL_PAGE = 64


def _make_flashmla_backend(pool, speculative_num_draft_tokens: int = 1):
    from tokenspeed.runtime.layers.attention.backends.flashmla import FlashMLABackend
    from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig

    config = MLAConfig(
        device="cuda",
        backend_name="flashmla",
        num_attention_heads=16,
        num_kv_heads=1,
        head_dim=_LATENT_DIM,
        attn_tp_size=1,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        prefix_granularity=_KERNEL_PAGE,
        kernel_page_size=_KERNEL_PAGE,
        context_len=8 * pool.arena.prefix_granularity,
        max_bs=8,
        max_graph_bs=8,
        kv_cache_quant_method="",
        kv_lora_rank=_KV_LORA_RANK,
        qk_nope_head_dim=128,
        qk_rope_head_dim=_QK_ROPE_DIM,
        v_head_dim=128,
        scaling=192**-0.5,
        kv_cache_dim=_LATENT_DIM,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
    )
    backend = FlashMLABackend(config)
    # Learn the pool's history-group geometry (the wrapper does this at
    # startup through the registry).
    backend.set_cache_pool(pool)
    return backend


def _init_cache_decode(backend, pool, logical_rows, seq_lens_cpu, spec=1):
    # spec documents the backend's speculative_num_draft_tokens for the caller;
    # the write-window width is derived inside the backend from spec_num_tokens.
    del spec
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

    bs = len(logical_rows)
    table = torch.tensor(logical_rows, dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor(seq_lens_cpu, device="cuda", dtype=torch.int32)
    # Unified decode path: refresh writes the persistent buffers the wrapper
    # allocates at startup (init_cuda_graph_state runs unconditionally).
    if not hasattr(backend, "cuda_graph_kv_indices"):
        backend.mark_cache_contract()
        backend.init_cuda_graph_state(max_bs=max(bs, 4))
    backend.refresh_decode_metadata(
        bs,
        bs,
        # Poisoned: the grouped path must never consume page_table.
        _poison((bs,)).to(torch.int64),
        seq_lens,
        forward_mode=ForwardMode.DECODE,
        page_table=_poison((16, 256)),
        block_tables={"full_attention": table},
    )
    return table


@requires_cuda
def test_flashmla_grouped_decode_block_table_and_write_locs() -> None:
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

    pool = _make_pool("cuda", usable_pages=6)
    page_size = pool.arena.prefix_granularity  # logical (scheduler) page size
    ratio = page_size // _KERNEL_PAGE
    layer_id = _mla_layer_id(pool)
    layer = type("L", (), {"layer_id": layer_id})()

    # Two requests, each with two logical history pages.
    logical_rows = [[3, 5], [1, 4]]
    seq_lens_cpu = [page_size + 41, page_size + 7]

    backend = _make_flashmla_backend(pool)
    assert backend._cache_groups_bound is False
    _init_cache_decode(backend, pool, logical_rows, seq_lens_cpu)
    assert backend._cache_groups_bound is True

    meta = backend.forward_decode_metadata
    # block_table is the kernel-page expansion of the logical full-history
    # table: each logical page -> `ratio` consecutive kernel pages.
    expected_row0 = []
    for lpage in logical_rows[0]:
        expected_row0.extend(lpage * ratio + k for k in range(ratio))
    got_row0 = meta.page_table[0, : len(expected_row0)].tolist()
    assert got_row0 == expected_row0, (got_row0, expected_row0)

    # Write locations: position seq-1 -> logical page (from table) * page_size
    # + offset. Req 0: seq_len-1 = page_size+40 -> logical index 1 -> page 5,
    # offset 40. Req 1: page_size+6 -> logical index 1 -> page 4, offset 6.
    locs = backend.select_out_cache_loc(layer, None, ForwardMode.DECODE)
    assert locs.tolist() == [5 * page_size + 40, 4 * page_size + 6], locs.tolist()


@requires_cuda
def test_flashmla_grouped_prefill_index_math() -> None:
    """The two prefill index views built from the LCM full-history table:

    * per-token slot table (flashinfer paged prefill, plan page_size=1)
    * packed new-token write locations (_extend_out_cache_loc)

    Validates the metadata math directly through the mixin helpers. The
    flashinfer paged-prefill READ (wrapper.plan) needs a live serving wrapper
    state and is validated end-to-end on a real model, not here.
    """
    from tokenspeed.runtime.layers.attention.page_table import expand_page_table

    pool = _make_pool("cuda", usable_pages=6)
    page_size = pool.arena.prefix_granularity
    backend = _make_flashmla_backend(pool)

    # The mixin consumes the KERNEL-page table (upstream expansion); build it
    # the same way CacheBatchMetadata.kernel_table does.
    logical_table = torch.tensor([[3, 5]], device="cuda", dtype=torch.int32)
    table = expand_page_table(
        logical_table,
        block_granularity=page_size,
        kernel_page_size=_KERNEL_PAGE,
    )

    # Per-token slot table: token t -> table[0, t // p] * p + t % p, which by
    # the expansion invariant equals the logical-table slot. Position
    # page_size+2 -> logical index 1 -> page 5, offset 2.
    slots = backend._group_per_token_slot_table(
        table,
        batch_size=1,
        page_size=_KERNEL_PAGE,
        max_context_len=backend.max_context_len,
    )
    assert slots[0, 0].item() == 3 * page_size + 0
    assert slots[0, page_size - 1].item() == 3 * page_size + (page_size - 1)
    assert slots[0, page_size].item() == 5 * page_size + 0
    assert slots[0, page_size + 2].item() == 5 * page_size + 2

    # New-token write locations: prefix=page_size, extend=3 -> positions
    # [page_size, page_size+3) -> page 5, offsets 0/1/2, packed in query order.
    locs = backend._extend_out_cache_loc(
        table,
        torch.tensor([page_size], dtype=torch.int32),
        torch.tensor([3], dtype=torch.int32),
    )
    assert locs.tolist() == [5 * page_size + 0, 5 * page_size + 1, 5 * page_size + 2]


@requires_cuda
def test_flashmla_grouped_target_verify_writes_whole_window() -> None:
    """Target verify (spec_num_tokens>1, non-draft decode) writes the whole
    trailing window seq-N..seq-1 per request, request-major."""
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

    pool = _make_pool("cuda", usable_pages=6)
    page_size = pool.arena.prefix_granularity
    spec = 4
    backend = _make_flashmla_backend(pool, speculative_num_draft_tokens=spec)

    # Verify-window widths: target decode -> spec; graph -> spec; draft -> 1.
    assert backend._verify_q_len(ForwardMode.DECODE) == spec
    assert backend._graph_verify_q_len() == spec

    logical_rows = [[3, 5]]
    # seq_len = page_size + 10 -> window positions page_size+7 .. page_size+10,
    # all on logical index 1 -> page 5, offsets 7/8/9/10.
    seq_lens_cpu = [page_size + 11]
    _init_cache_decode(backend, pool, logical_rows, seq_lens_cpu, spec=spec)

    meta = backend.forward_decode_metadata
    assert meta.group_q_len_per_req == spec
    locs = backend.select_out_cache_loc(None, None, ForwardMode.DECODE)
    base = 5 * page_size
    assert locs.tolist() == [base + 7, base + 8, base + 9, base + 10], locs.tolist()


@requires_cuda
def test_flashmla_classic_path_uses_page_table() -> None:
    """Without cache metadata the backend keeps the classic page_table path."""
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

    pool = _make_pool("cuda", usable_pages=6)
    backend = _make_flashmla_backend(pool)

    bs = 2
    page_table = torch.arange(bs * 4, device="cuda", dtype=torch.int32).view(bs, 4)
    seq_lens = torch.tensor([10, 20], device="cuda", dtype=torch.int32)
    backend.init_cuda_graph_state(max_bs=4)
    backend.refresh_decode_metadata(
        bs,
        bs,
        torch.arange(bs, device="cuda", dtype=torch.int64),
        seq_lens,
        forward_mode=ForwardMode.DECODE,
        page_table=page_table,
    )
    assert backend._cache_groups_bound is False
    meta = backend.forward_decode_metadata
    assert meta.group_out_cache_loc is None
    assert meta.page_table[0, :4].tolist() == page_table[0].tolist()
    # Classic path: select_out_cache_loc is identity.
    caller = torch.tensor([1, 2], device="cuda", dtype=torch.int64)
    assert torch.equal(
        backend.select_out_cache_loc(None, caller, ForwardMode.DECODE), caller
    )


def _make_draft_flashmla_backend(pool):
    from tokenspeed.runtime.layers.attention.backends.flashmla import FlashMLABackend
    from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig

    config = MLAConfig(
        device="cuda",
        backend_name="flashmla",
        num_attention_heads=16,
        num_kv_heads=1,
        head_dim=_LATENT_DIM,
        attn_tp_size=1,
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        prefix_granularity=_KERNEL_PAGE,
        kernel_page_size=_KERNEL_PAGE,
        context_len=8 * pool.arena.prefix_granularity,
        max_bs=8,
        max_graph_bs=8,
        kv_cache_quant_method="",
        kv_lora_rank=_KV_LORA_RANK,
        qk_nope_head_dim=128,
        qk_rope_head_dim=_QK_ROPE_DIM,
        v_head_dim=128,
        scaling=192**-0.5,
        kv_cache_dim=_LATENT_DIM,
        is_draft=True,
    )
    backend = FlashMLABackend(config)
    backend.mark_cache_contract()
    return backend


@requires_cuda
def test_flashmla_draft_consumes_staged_page_table() -> None:
    """A contract-bound draft reads the batch-ordered draft page table
    (DraftPageStaging already expanded it into this backend's kernel pages)
    as-is; block_tables is ignored (the staging path replaced it)."""
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

    pool = _make_pool("cuda", usable_pages=6)
    page_size = pool.arena.prefix_granularity
    ratio = page_size // _KERNEL_PAGE
    backend = _make_draft_flashmla_backend(pool)

    # The staged table holds KERNEL page ids (already expanded at publish).
    logical_rows = [[3, 5], [1, 4]]
    staged_rows = [
        [lpage * ratio + k for lpage in row for k in range(ratio)]
        for row in logical_rows
    ]
    staged = torch.tensor(staged_rows, device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor(
        [page_size + 41, page_size + 7], device="cuda", dtype=torch.int32
    )
    backend.init_cuda_graph_state(max_bs=4)
    backend.refresh_decode_metadata(
        2,
        2,
        _poison((2,)).to(torch.int64),
        seq_lens,
        forward_mode=ForwardMode.DECODE,
        page_table=staged,
    )
    assert backend._cache_groups_bound is True
    meta = backend.forward_decode_metadata
    got_row0 = meta.page_table[0, : len(staged_rows[0])].tolist()
    assert got_row0 == staged_rows[0], (got_row0, staged_rows[0])
