# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation

"""MiniMax MSA prefill block-score + top-k on the fmha_sm100 OnlyScore path.

Wraps the vendored ``thirdparty/msa`` nvcc-JIT dense FMHA in score-only mode
(``output_maxscore=True, output_o=False``) plus its ``sparse_topk_select``
kernel as a drop-in replacement for the Triton ``_prefill_block_score_kernel``
+ ``_topk_with_padding`` pair inside ``minimax_indexer``.  SM100 only.

The fmha ``max_score`` is the raw QK row max (``sm_scale`` only feeds the
softmax path), so scores here are unscaled relative to the Triton path;
rankings are identical and the buffer is consumed only by the top-k.  Forced
init/local blocks are applied by ``sparse_topk_select`` rather than written
into the score buffer.

Kernel variants JIT-compile with nvcc on first use (~45 s each, disk-cached
under ``~/.cache/minfer/fmha_sm100``).  Compilation runs on a background
thread; until it finishes ``prefill_score_supported`` returns False and
callers keep the Triton path, so serving never stalls on nvcc.
"""

from __future__ import annotations

import functools
import logging
import math
import threading
from collections.abc import Sequence

import torch

logger = logging.getLogger(__name__)

_PAGE_SIZE = 128
_TOPK = 16
# Below ~128 pages the upstream top-k kernel's fixed per-row cost outweighs
# the OnlyScore kernel's win over the Triton scorer (measured crossover).
_MIN_BLOCKS = 129
# Upstream sparse_topk_select only implements the insertion-sort path.
_MAX_K_TILES_LIMIT = 12288
# fmha_sm100_plan disables max_score output beyond 2^31 elements.
_MAX_SCORE_ELEMS = 1 << 31

_jit_state: dict = {"thread": None, "ready": False, "failed": False}
_jit_lock = threading.Lock()


@functools.lru_cache(maxsize=1)
def _load_api():
    from tokenspeed_kernel.thirdparty.msa.api import (
        _fmha_sm100,
        _fmha_sm100_plan,
        sparse_topk_select,
    )

    return _fmha_sm100, _fmha_sm100_plan, sparse_topk_select


def _compile_variants() -> None:
    try:
        from tokenspeed_kernel.thirdparty.msa import jit as msa_jit

        msa_jit.get_plan_fn()
        msa_jit.get_sparse_topk_module()
        bf16 = msa_jit._dlpack_dtype_code(torch.bfloat16)
        # All OnlyScore/paged/no-split/pack-1 variants reachable from this
        # wrapper: (qo_tile, single_wg) keyed off the batch's max query length.
        for qo_tile, single_wg in ((128, True), (128, False), (256, False)):
            msa_jit.get_fmha_variant(bf16, qo_tile, single_wg, 2, _PAGE_SIZE, False, 1)
        _load_api()
        _jit_state["ready"] = True
        logger.info("MSA fmha OnlyScore prefill scorer ready")
    except Exception:
        _jit_state["failed"] = True
        logger.exception(
            "MSA fmha OnlyScore JIT failed; prefill keeps the Triton scorer"
        )


def ensure_prefill_score_ready(timeout: float | None = 0.0) -> bool:
    """Start (and optionally wait for) the fmha OnlyScore JIT compilation.

    Args:
        timeout: Seconds to block waiting for compilation. ``0.0`` returns
            immediately after kicking off the background compile; ``None``
            blocks until it finishes.

    Returns:
        True once every kernel variant is compiled and loadable.
    """
    if _jit_state["ready"]:
        return True
    if _jit_state["failed"]:
        return False
    with _jit_lock:
        if _jit_state["thread"] is None:
            thread = threading.Thread(
                target=_compile_variants, name="msa-score-jit", daemon=True
            )
            _jit_state["thread"] = thread
            thread.start()
    if timeout != 0.0:
        _jit_state["thread"].join(timeout)
    return _jit_state["ready"]


def prefill_score_supported(
    index_q: torch.Tensor,
    cache_pages: torch.Tensor,
    topk: int,
    max_blocks: int,
    query_lens_cpu: Sequence[int] | None,
    seq_lens_cpu: Sequence[int] | None,
) -> bool:
    """Return True when the fmha OnlyScore prefill path applies.

    Args:
        index_q: Index queries shaped ``[tokens, heads, 128]``.
        cache_pages: Index-key cache pages shaped ``[pages, 128, 128]``.
        topk: Number of selected blocks (must be 16 upstream).
        max_blocks: Logical block columns scored for this batch.
        query_lens_cpu: Host-side per-request new-token counts.
        seq_lens_cpu: Host-side per-request total sequence lengths.

    Returns:
        Whether ``minimax_prefill_score_topk`` can serve this batch.
    """
    if query_lens_cpu is None or seq_lens_cpu is None:
        return False
    if topk != _TOPK:
        return False
    if index_q.dtype != torch.bfloat16 or cache_pages.dtype != torch.bfloat16:
        return False
    if index_q.shape[-1] != 128 or cache_pages.shape[-1] != 128:
        return False
    if cache_pages.shape[1] != _PAGE_SIZE or not cache_pages.is_contiguous():
        return False
    if max_blocks < _MIN_BLOCKS:
        return False
    max_k_tiles = math.ceil(max_blocks / 128) * 128
    if max_k_tiles >= _MAX_K_TILES_LIMIT:
        return False
    tokens, heads = index_q.shape[0], index_q.shape[1]
    if tokens * heads * max_k_tiles >= _MAX_SCORE_ELEMS:
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    return ensure_prefill_score_ready(0.0)


@functools.lru_cache(maxsize=8)
def _plan_for_batch(
    query_lens: tuple[int, ...],
    seq_lens: tuple[int, ...],
    num_heads: int,
    device: torch.device,
):
    _, _fmha_sm100_plan, _ = _load_api()
    qo_t = torch.tensor(query_lens, dtype=torch.int32)
    kv_t = torch.tensor(seq_lens, dtype=torch.int32)
    # num_kv_heads=-1 pins pack_factor to 1 so only the three precompiled
    # variants are ever dispatched (a packed variant would nvcc-JIT inline).
    plan = _fmha_sm100_plan(
        qo_t,
        kv_t,
        num_heads,
        num_kv_heads=-1,
        qo_offset=kv_t - qo_t,
        page_size=_PAGE_SIZE,
        output_maxscore=True,
        causal=True,
        num_kv_splits=1,
        device=device,
    )
    assert plan["max_k_tiles"] > 0, "planner disabled max_score output"
    positions = torch.cat(
        [
            torch.arange(k - q, k, dtype=torch.int32)
            for q, k in zip(query_lens, seq_lens)
        ]
    )
    num_valid_pages_tok = (positions // _PAGE_SIZE + 1).to(device)
    num_pages_req = torch.tensor(
        [(k + _PAGE_SIZE - 1) // _PAGE_SIZE for k in seq_lens],
        dtype=torch.int32,
        device=device,
    )
    return plan, num_valid_pages_tok, num_pages_req


def minimax_prefill_score_topk(
    index_q: torch.Tensor,
    cache_pages: torch.Tensor,
    block_table: torch.Tensor,
    *,
    scale: float,
    init_blocks: int,
    local_blocks: int,
    topk: int,
    query_lens_cpu: Sequence[int],
    seq_lens_cpu: Sequence[int],
) -> torch.Tensor:
    """Score visible blocks with fmha OnlyScore and select Top-K per token.

    Args:
        index_q: Index queries shaped ``[tokens, heads, 128]``, BF16.
        cache_pages: Index-key cache shaped ``[pages, 128, 128]``, BF16.
        block_table: Logical-to-physical page table ``[requests, pages]``.
        scale: Index score scale (forwarded as ``sm_scale``; the emitted
            scores stay in raw QK units — see the module docstring).
        init_blocks: Leading blocks forced into the selection.
        local_blocks: Most-recent blocks forced into the selection.
        topk: Number of selected blocks per token (16).
        query_lens_cpu: Host-side per-request new-token counts.
        seq_lens_cpu: Host-side per-request total sequence lengths.

    Returns:
        Selected logical block ids ``[tokens, heads, topk]`` int32, ascending
        per row with ``-1`` padding past each token's visible block count.
    """
    _fmha_sm100, _, sparse_topk_select = _load_api()
    tokens, num_heads, head_dim = index_q.shape
    device = index_q.device
    plan, num_valid_pages_tok, num_pages_req = _plan_for_batch(
        tuple(int(x) for x in query_lens_cpu),
        tuple(int(x) for x in seq_lens_cpu),
        num_heads,
        device,
    )
    max_k_tiles = plan["max_k_tiles"]

    cols = torch.arange(block_table.shape[1], device=device)
    flat_page_table = block_table[cols[None, :] < num_pages_req[:, None]]

    scores = torch.full(
        (tokens, num_heads, max_k_tiles),
        -float("inf"),
        dtype=torch.float32,
        device=device,
    )
    k_pages = cache_pages.view(cache_pages.shape[0], 1, _PAGE_SIZE, head_dim)
    _fmha_sm100(
        index_q,
        k_pages,
        k_pages,  # V placeholder; never read in OnlyScore mode
        plan,
        kv_indices=flat_page_table,
        output_o=False,
        output_maxscore=True,
        sm_scale=scale,
        max_score=scores,
    )

    selected = torch.empty((tokens, num_heads, topk), dtype=torch.int32, device=device)
    sparse_topk_select(
        scores,
        topk,
        num_valid_pages=num_valid_pages_tok,
        force_begin_blocks=init_blocks,
        force_end_blocks=local_blocks,
        output=selected,
        max_score_layout="THK",
    )
    return selected


__all__ = [
    "ensure_prefill_score_ready",
    "minimax_prefill_score_topk",
    "prefill_score_supported",
]
