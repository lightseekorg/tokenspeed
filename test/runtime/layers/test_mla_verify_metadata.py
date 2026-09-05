from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.layers.attention.backends.paged import mla as mla_backend


def _run_mla_decode(
    monkeypatch,
    *,
    is_draft: bool,
    bs: int = 2,
    q_len_per_req: int = 2,
    data_type: torch.dtype = torch.float32,
    page_size: int = 16,
    draft_block_decode: bool = False,
    sliding_window_size: int = -1,
    query_blocks: bool = False,
    block_size: int | None = None,
) -> dict[str, torch.Tensor]:
    captured = {}

    def fake_mla_decode_with_kvcache(**kwargs):
        captured.update(kwargs)
        return torch.zeros(*kwargs["q"].shape[:-1], 4)

    monkeypatch.setattr(
        mla_backend, "mla_decode_with_kvcache", fake_mla_decode_with_kvcache
    )
    monkeypatch.setattr(
        mla_backend,
        "supports_mla_decode_query_blocks",
        lambda **kwargs: query_blocks,
    )
    backend = object.__new__(mla_backend.MLAAttnBackend)
    spec = block_size or q_len_per_req
    metadata_rows = bs * spec if draft_block_decode else bs
    seq_lens = torch.tensor([64, 128], dtype=torch.int32)[:bs]
    if draft_block_decode:
        seq_lens = seq_lens.repeat_interleave(spec)
    backend.forward_decode_metadata = SimpleNamespace(
        num_extends=0,
        page_table=torch.zeros(metadata_rows, 1, dtype=torch.int32),
        seq_lens=seq_lens,
    )
    backend.is_draft = is_draft
    backend.draft_block_decode = draft_block_decode
    backend.spec_num_tokens = spec if draft_block_decode else 1
    backend.max_context_len = 256
    backend.kernel_page_size = page_size
    backend.kv_lora_rank = 2
    backend.qk_nope_head_dim = 2
    backend.qk_rope_head_dim = 2
    backend.kv_cache_dim = 4
    backend.data_type = data_type
    backend.kernel_solution = None
    backend._query_block_decode = {}

    layer = SimpleNamespace(
        tp_q_head_num=1,
        head_dim=4,
        v_head_dim=4,
        scaling=1.0,
        logit_cap=0.0,
        k_scale_float=None,
        layer_id=0,
        sliding_window_size=sliding_window_size,
    )
    token_to_kv_pool = SimpleNamespace(
        get_key_buffer=lambda layer_id: torch.zeros(page_size, 4).to(data_type)
    )

    backend.forward_decode(
        q=torch.zeros(bs * q_len_per_req, 4).to(data_type),
        k=None,
        v=None,
        layer=layer,
        out_cache_loc=torch.empty(0, dtype=torch.int32),
        token_to_kv_pool=token_to_kv_pool,
        bs=bs,
        save_kv_cache=False,
    )
    return captured


def test_target_verify_cache_seqlens_count_back_from_final_lengths(monkeypatch):
    cache_seqlens = _run_mla_decode(monkeypatch, is_draft=False)["cache_seqlens"]

    assert cache_seqlens.tolist() == [63, 64, 127, 128]


def test_draft_cache_seqlens_count_forward_from_base_lengths(monkeypatch):
    cache_seqlens = _run_mla_decode(monkeypatch, is_draft=True)["cache_seqlens"]

    assert cache_seqlens.tolist() == [64, 65, 128, 129]


def test_fp8_decode_dispatches_with_native_fp8_query(monkeypatch):
    captured = _run_mla_decode(
        monkeypatch,
        is_draft=False,
        bs=1,
        q_len_per_req=1,
        data_type=torch.float8_e4m3fn,
        page_size=64,
    )

    assert captured["q"].dtype == torch.float8_e4m3fn


def test_dflash2_block_decode_passes_exact_sliding_window(monkeypatch):
    captured = _run_mla_decode(
        monkeypatch,
        is_draft=True,
        bs=2,
        q_len_per_req=8,
        draft_block_decode=True,
        # PagedAttention stores HF's inclusive window as window_left.
        sliding_window_size=4095,
    )

    assert captured["window_left"] == 4095
    assert captured["noncausal_block_size"] == 8
    assert captured["q"].shape[0] == 16


def test_a_windowed_block_folds_onto_the_query_axis_when_a_kernel_takes_it(monkeypatch):
    """One row per request instead of one per block position, same mask.

    The flattened metadata repeats each request's page table and block-end
    length once per block position, so the first row of every group is the
    request. Folding is only correct because of that, and only allowed when a
    kernel says it reads the query-axis form.
    """
    captured = _run_mla_decode(
        monkeypatch,
        is_draft=True,
        bs=2,
        q_len_per_req=8,
        draft_block_decode=True,
        sliding_window_size=4095,
        query_blocks=True,
    )

    assert captured["q"].shape[:2] == (2, 8)
    assert captured["page_table"].shape[0] == 2
    assert captured["cache_seqlens"].tolist() == [64, 128]
    assert captured["noncausal_block_size"] == 8
    assert captured["window_left"] == 4095


def test_a_full_attention_block_layer_folds_on_the_same_terms(monkeypatch):
    """The layout follows the kernel, not the mask.

    A draft mixes windowed and full-attention layers over one metadata buffer,
    so both ask the same question and a windowless layer folds whenever a
    kernel reads that form.
    """
    captured = _run_mla_decode(
        monkeypatch,
        is_draft=True,
        bs=2,
        q_len_per_req=8,
        draft_block_decode=True,
        sliding_window_size=-1,
        query_blocks=True,
    )

    assert captured["q"].shape[:2] == (2, 8)
    assert captured["page_table"].shape[0] == 2
    assert captured["window_left"] == -1


def test_a_block_layer_keeps_the_flattened_rows_when_no_kernel_reads_the_fold(
    monkeypatch,
):
    """Declining leaves the contract every other configuration has always sent."""
    captured = _run_mla_decode(
        monkeypatch,
        is_draft=True,
        bs=2,
        q_len_per_req=8,
        draft_block_decode=True,
        sliding_window_size=4095,
        query_blocks=False,
    )

    assert captured["q"].shape[:2] == (16, 1)
    assert captured["page_table"].shape[0] == 16


def test_a_narrower_draft_forward_than_its_block_keeps_the_flattened_rows(monkeypatch):
    """Un-expanding is only valid on the stride the metadata was built with.

    ``resolve_speculative_num_tokens`` reconciles the draft's forward width
    with its block for every drafter in tree, so this is the arithmetic's
    guard rather than a live configuration: a forward narrower than its block
    would stride into the next request's rows.
    """
    captured = _run_mla_decode(
        monkeypatch,
        is_draft=True,
        bs=2,
        q_len_per_req=7,
        draft_block_decode=True,
        sliding_window_size=-1,
        query_blocks=True,
        block_size=8,
    )
    assert captured["q"].shape[:2] == (14, 1)
    assert captured["noncausal_block_size"] == 8


def test_the_cutedsl_drafter_backend_never_reaches_the_shared_dispatcher() -> None:
    """The K3 DSpark gate runs ``--drafter-attention-backend tokenspeed_mla``.

    That backend calls the CuteDSL decode kernel directly and never imports
    ``mla_decode_with_kvcache``, so which kernel the shared dispatcher selects
    is not a variable for that configuration -- registering a new candidate
    there cannot reach it.
    """
    from tokenspeed.runtime.layers.attention.backends import (
        tokenspeed_mla as cutedsl_backend,
    )

    assert not hasattr(cutedsl_backend, "mla_decode_with_kvcache")
    assert hasattr(cutedsl_backend, "tokenspeed_mla_decode")
