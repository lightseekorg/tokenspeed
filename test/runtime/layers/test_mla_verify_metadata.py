from __future__ import annotations

from types import SimpleNamespace

import torch

from tokenspeed.runtime.layers.attention.backends import mla as mla_backend


def _run_mla_decode(monkeypatch, *, is_draft: bool) -> torch.Tensor:
    captured = {}

    def fake_mla_decode_with_kvcache(**kwargs):
        captured["cache_seqlens"] = kwargs["cache_seqlens"]
        return torch.zeros(4, 1, 1, 4)

    monkeypatch.setattr(
        mla_backend, "mla_decode_with_kvcache", fake_mla_decode_with_kvcache
    )

    backend = object.__new__(mla_backend.MLAAttnBackend)
    backend.forward_decode_metadata = SimpleNamespace(
        num_extends=0,
        page_table=torch.zeros(2, 1, dtype=torch.int32),
        seq_lens=torch.tensor([64, 128], dtype=torch.int32),
    )
    backend.is_draft = is_draft
    backend.max_context_len = 256
    backend.page_size = 16
    backend.kv_lora_rank = 2
    backend.qk_nope_head_dim = 2
    backend.qk_rope_head_dim = 2
    backend.kv_cache_dim = 4
    backend.data_type = torch.float32
    backend.kernel_solution = "default"

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
        get_key_buffer=lambda layer_id: torch.zeros(16, 4)
    )

    backend.forward_decode(
        q=torch.zeros(4, 4),
        k=None,
        v=None,
        layer=layer,
        out_cache_loc=torch.empty(0, dtype=torch.int32),
        token_to_kv_pool=token_to_kv_pool,
        bs=2,
        save_kv_cache=False,
    )
    return captured["cache_seqlens"]


def test_target_verify_cache_seqlens_count_back_from_final_lengths(monkeypatch):
    cache_seqlens = _run_mla_decode(monkeypatch, is_draft=False)

    assert cache_seqlens.tolist() == [63, 64, 127, 128]


def test_draft_cache_seqlens_count_forward_from_base_lengths(monkeypatch):
    cache_seqlens = _run_mla_decode(monkeypatch, is_draft=True)

    assert cache_seqlens.tolist() == [64, 65, 128, 129]
