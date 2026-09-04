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
) -> dict[str, torch.Tensor]:
    captured = {}

    def fake_mla_decode_with_kvcache(**kwargs):
        captured.update(kwargs)
        return torch.zeros(*kwargs["q"].shape[:-1], 4)

    monkeypatch.setattr(
        mla_backend, "mla_decode_with_kvcache", fake_mla_decode_with_kvcache
    )
    backend = object.__new__(mla_backend.MLAAttnBackend)
    metadata_rows = bs * q_len_per_req if draft_block_decode else bs
    seq_lens = torch.tensor([64, 128], dtype=torch.int32)[:bs]
    if draft_block_decode:
        seq_lens = seq_lens.repeat_interleave(q_len_per_req)
    backend.forward_decode_metadata = SimpleNamespace(
        num_extends=0,
        page_table=torch.zeros(metadata_rows, 1, dtype=torch.int32),
        seq_lens=seq_lens,
    )
    backend.is_draft = is_draft
    backend.draft_block_decode = draft_block_decode
    backend.spec_num_tokens = q_len_per_req if draft_block_decode else 1
    backend.max_context_len = 256
    backend.kernel_page_size = page_size
    backend.kv_lora_rank = 2
    backend.qk_nope_head_dim = 2
    backend.qk_rope_head_dim = 2
    backend.kv_cache_dim = 4
    backend.data_type = data_type
    backend.kernel_solution = None

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
