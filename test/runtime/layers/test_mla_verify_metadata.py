from __future__ import annotations

from types import SimpleNamespace

import torch

from tokenspeed.runtime.layers.attention.backends import mla as mla_backend
from tokenspeed.runtime.layers.attention.backends import (
    trtllm_mla as trtllm_mla_backend,
)


def _run_mla_decode(
    monkeypatch,
    *,
    is_draft: bool,
    bs: int = 2,
    q_len_per_req: int = 2,
    data_type: torch.dtype = torch.float32,
    page_size: int = 16,
) -> dict[str, torch.Tensor]:
    captured = {}

    def fake_mla_decode_with_kvcache(**kwargs):
        captured["q"] = kwargs["q"]
        captured["cache_seqlens"] = kwargs["cache_seqlens"]
        return torch.zeros(*kwargs["q"].shape[:-1], 4)

    monkeypatch.setattr(
        mla_backend, "mla_decode_with_kvcache", fake_mla_decode_with_kvcache
    )
    backend = object.__new__(mla_backend.MLAAttnBackend)
    backend.forward_decode_metadata = SimpleNamespace(
        num_extends=0,
        page_table=torch.zeros(bs, 1, dtype=torch.int32),
        seq_lens=torch.tensor([64, 128], dtype=torch.int32)[:bs],
    )
    backend.is_draft = is_draft
    backend.draft_block_decode = False
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


def _run_trtllm_mla_decode(
    monkeypatch,
    *,
    is_draft: bool,
    bs: int = 2,
    q_len_per_req: int = 2,
) -> dict[str, torch.Tensor | int]:
    captured = {}

    def fake_trtllm_decode(**kwargs):
        captured.update(kwargs)
        query = kwargs["query"]
        return torch.zeros(*query.shape[:-1], 4)

    monkeypatch.setattr(
        trtllm_mla_backend,
        "trtllm_batch_decode_with_kv_cache_mla",
        fake_trtllm_decode,
    )
    backend = object.__new__(trtllm_mla_backend.TRTLLMMLABackend)
    backend.forward_decode_metadata = SimpleNamespace(
        num_extends=0,
        block_kv_indices=torch.tensor([[3], [5]], dtype=torch.int32)[:bs],
        seq_lens_k=torch.tensor([64, 128], dtype=torch.int32)[:bs],
        max_seq_len_k=256,
    )
    backend.is_draft = is_draft
    backend.draft_block_decode = False
    backend.kernel_page_size = 16
    backend.kv_lora_rank = 2
    backend.qk_nope_head_dim = 2
    backend.qk_rope_head_dim = 2
    backend.kv_cache_dim = 4
    backend.data_type = torch.float32
    backend.trtllm_workspace = torch.empty(0)

    layer = SimpleNamespace(
        tp_q_head_num=1,
        head_dim=4,
        v_head_dim=4,
        scaling=1.0,
        k_scale_float=None,
        layer_id=0,
    )
    token_to_kv_pool = SimpleNamespace(
        get_key_buffer=lambda layer_id: torch.zeros(32, 4)
    )

    backend.forward_decode(
        q=torch.zeros(bs * q_len_per_req, 4),
        k=None,
        v=None,
        layer=layer,
        out_cache_loc=torch.empty(0, dtype=torch.int32),
        token_to_kv_pool=token_to_kv_pool,
        bs=bs,
        save_kv_cache=False,
    )
    return captured


def test_trtllm_target_verify_uses_per_token_causal_lengths(monkeypatch):
    captured = _run_trtllm_mla_decode(monkeypatch, is_draft=False)

    assert captured["seq_lens"].tolist() == [63, 64, 127, 128]
    assert captured["block_tables"].tolist() == [[3], [3], [5], [5]]
    assert captured["max_seq_len"] == 256


def test_trtllm_draft_catchup_counts_forward_from_base_lengths(monkeypatch):
    captured = _run_trtllm_mla_decode(monkeypatch, is_draft=True)

    assert captured["seq_lens"].tolist() == [64, 65, 128, 129]
    assert captured["max_seq_len"] == 258
