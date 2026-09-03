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

"""FlashMLABackend decode metadata behind the cache-group router.

The router's ``GroupTableStacks`` is now the one place scheduler (logical)
blocks become kernel pages; the FlashMLA leaf consumes the pre-expanded
``[bs, max_num_pages]`` table and copies it into its persistent buffers.
Exercises the metadata math only (no FlashMLA CUDA kernel), so it runs on
any CUDA device.
"""

from __future__ import annotations

import os
import sys

import torch

_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))

from test.runtime.conftest import MLA_KV_LORA_RANK as _KV_LORA_RANK
from test.runtime.conftest import MLA_LATENT_DIM as _LATENT_DIM
from test.runtime.conftest import MLA_QK_ROPE_DIM as _QK_ROPE_DIM
from test.runtime.conftest import make_kimi_pool as _make_pool
from test.runtime.conftest import requires_cuda

from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="runtime-1gpu")

from tokenspeed.runtime.layers.attention.backends.group_tables import (
    GroupTableSpec,
    GroupTableStacks,
)

_KERNEL_PAGE = 64
_FULL = "full_attention"


def _flashmla_spec():
    from tokenspeed.runtime.layers.attention.configs.mla import MLAConfig

    return MLAConfig(
        backend_name="flashmla",
        num_attention_heads=16,
        num_kv_heads=1,
        head_dim=_LATENT_DIM,
        attn_tp_size=1,
        kv_lora_rank=_KV_LORA_RANK,
        qk_nope_head_dim=128,
        qk_rope_head_dim=_QK_ROPE_DIM,
        v_head_dim=128,
        scaling=192**-0.5,
        kv_cache_dim=_LATENT_DIM,
    )


def _make_flashmla_backend(
    pool, speculative_num_draft_tokens: int = 1, *, is_draft: bool = False
):
    from tokenspeed.runtime.layers.attention.backends.flashmla import FlashMLABackend
    from tokenspeed.runtime.layers.attention.configs.base import AttnConfig

    spec = _flashmla_spec()
    config = AttnConfig(
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        prefix_granularity=_KERNEL_PAGE,
        kernel_page_size=_KERNEL_PAGE,
        context_len=8 * pool.arena.prefix_granularity,
        max_bs=8,
        kv_cache_quant_method="",
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        is_draft=is_draft,
        components=(spec,),
    )
    backend = FlashMLABackend(config, spec, kernel_page_size=_KERNEL_PAGE)
    backend.set_cache_pool(pool)
    return backend


def _stacks_for(backend, pool, max_bs: int = 4) -> GroupTableStacks:
    """The router-side expansion stage: logical blocks -> this leaf's pages."""
    return GroupTableStacks(
        [
            GroupTableSpec(
                group_id=_FULL,
                block_granularity=pool.arena.prefix_granularity,
                kernel_page_size=backend.kernel_page_size,
                max_num_pages=backend.max_num_pages,
            )
        ],
        max_bs=max_bs,
        max_tokens_per_req=backend.spec_num_tokens,
        device="cuda",
    )


def _refresh_from_logical(backend, pool, logical_rows, seq_lens_cpu):
    """Expand the logical full-history table through the router's stacks and
    refresh the leaf from the resulting kernel-page view (the unified path)."""
    bs = len(logical_rows)
    stacks = _stacks_for(backend, pool, max_bs=max(bs, 4))
    raw = torch.tensor(logical_rows, dtype=torch.int32, device="cuda")
    stacks.fill(bs, bs, {_FULL: raw})
    seq_lens = torch.tensor(seq_lens_cpu, device="cuda", dtype=torch.int32)
    backend.init_cuda_graph_state(max_bs=max(bs, 4))
    backend.refresh_decode_metadata(bs, bs, seq_lens, stacks.table(_FULL, bs))
    return stacks


@requires_cuda
def test_flashmla_grouped_decode_block_table_follows_the_stack_expansion() -> None:
    pool = _make_pool("cuda", usable_pages=6)
    page_size = pool.arena.prefix_granularity  # logical (scheduler) page size
    ratio = page_size // _KERNEL_PAGE

    # Two requests, each with two logical history pages.
    logical_rows = [[3, 5], [1, 4]]
    seq_lens_cpu = [page_size + 41, page_size + 7]

    backend = _make_flashmla_backend(pool)
    _refresh_from_logical(backend, pool, logical_rows, seq_lens_cpu)

    meta = backend.forward_decode_metadata
    # The leaf's table is the kernel-page expansion of the logical
    # full-history table: each logical page -> `ratio` consecutive kernel pages.
    expected_row0 = []
    for lpage in logical_rows[0]:
        expected_row0.extend(lpage * ratio + k for k in range(ratio))
    got_row0 = meta.page_table[0, : len(expected_row0)].tolist()
    assert got_row0 == expected_row0, (got_row0, expected_row0)


@requires_cuda
def test_flashmla_grouped_prefill_index_math() -> None:
    """The per-token slot table built from the expanded kernel-page table
    (flashinfer paged prefill, plan page_size=1).

    Validates the metadata math directly through the module helper. The
    flashinfer paged-prefill READ (wrapper.plan) needs a live serving wrapper
    state and is validated end-to-end on a real model, not here.
    """
    from tokenspeed.runtime.layers.attention.backends.flashmla import (
        _per_token_slot_table,
    )

    pool = _make_pool("cuda", usable_pages=6)
    page_size = pool.arena.prefix_granularity
    backend = _make_flashmla_backend(pool)

    # The leaf consumes the KERNEL-page table the router's stacks expand.
    logical_table = torch.tensor([[3, 5]], device="cuda", dtype=torch.int32)
    stacks = _stacks_for(backend, pool)
    stacks.fill(1, 1, {_FULL: logical_table})
    table = stacks.table(_FULL, 1)

    # Per-token slot table: token t -> table[0, t // p] * p + t % p, which by
    # the expansion invariant equals the logical-table slot. Position
    # page_size+2 -> logical index 1 -> page 5, offset 2.
    slots = _per_token_slot_table(
        table,
        batch_size=1,
        page_size=_KERNEL_PAGE,
        max_context_len=backend.max_context_len,
    )
    assert slots[0, 0].item() == 3 * page_size + 0
    assert slots[0, page_size - 1].item() == 3 * page_size + (page_size - 1)
    assert slots[0, page_size].item() == 5 * page_size + 0
    assert slots[0, page_size + 2].item() == 5 * page_size + 2


@requires_cuda
def test_flashmla_grouped_target_verify_uses_the_whole_window() -> None:
    """Target verify (spec_num_tokens>1, non-draft decode) bakes the whole
    window width into the decode views and clamps short rows to it."""
    pool = _make_pool("cuda", usable_pages=6)
    page_size = pool.arena.prefix_granularity
    spec = 4
    backend = _make_flashmla_backend(pool, speculative_num_draft_tokens=spec)

    # Verify-window widths: target -> spec, draft -> 1 (the leaf property).
    assert backend.verify_floor == spec
    draft = _make_flashmla_backend(pool, spec, is_draft=True)
    assert draft.verify_floor == 1

    logical_rows = [[3, 5]]
    seq_lens_cpu = [page_size + 11]
    _refresh_from_logical(backend, pool, logical_rows, seq_lens_cpu)

    meta = backend.forward_decode_metadata
    assert meta.q_len_per_req == spec
    # A row shorter than the window clamps up to it.
    _refresh_from_logical(backend, pool, logical_rows, [1])
    assert backend.forward_decode_metadata.seq_lens_k.tolist()[0] == spec


@requires_cuda
def test_flashmla_draft_consumes_staged_page_table() -> None:
    """A draft leaf reads the batch-ordered draft page table (the staging
    already expanded it into this leaf's kernel pages) as-is."""
    pool = _make_pool("cuda", usable_pages=6)
    page_size = pool.arena.prefix_granularity
    ratio = page_size // _KERNEL_PAGE
    backend = _make_flashmla_backend(pool, is_draft=True)

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
    backend.refresh_decode_metadata(2, 2, seq_lens, staged)
    meta = backend.forward_decode_metadata
    got_row0 = meta.page_table[0, : len(staged_rows[0])].tolist()
    assert got_row0 == staged_rows[0], (got_row0, staged_rows[0])
