from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.layers.attention.configs import dsa as dsa_config_mod
from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
from tokenspeed.runtime.layers.attention.backends import dsa as dsa_mod
from tokenspeed.runtime.layers.attention.backends.dsa import DSABackend


class _TokenToKVPool:
    def __init__(self, key_buffer: torch.Tensor) -> None:
        self._key_buffer = key_buffer

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        assert layer_id == 3
        return self._key_buffer


def _server_args(kv_cache_dtype: str = "fp8") -> SimpleNamespace:
    return SimpleNamespace(
        device="cuda",
        attention_backend="dsa",
        drafter_attention_backend=None,
        attn_tp_size=None,
        mapping=SimpleNamespace(attn=SimpleNamespace(tp_size=1, dp_size=1)),
        kv_cache_dtype=kv_cache_dtype,
        max_num_seqs=8,
        data_parallel_size=None,
        block_size=64,
        max_cudagraph_capture_size=4,
        kv_cache_quant_method="none",
        speculative_algorithm=None,
    )


def _model_config() -> SimpleNamespace:
    return SimpleNamespace(
        context_len=4096,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=576,
        dtype=torch.bfloat16,
        kv_lora_rank=512,
        qk_nope_head_dim=512,
        qk_rope_head_dim=64,
        v_head_dim=512,
        scaling=0.5,
        index_topk=4,
        index_head_dim=128,
        index_n_heads=2,
    )


def test_blackwell_dsa_prefill_defaults_to_trtllm(monkeypatch) -> None:
    monkeypatch.setattr(
        dsa_mod,
        "current_platform",
        lambda: SimpleNamespace(is_nvidia=True, is_blackwell=True),
    )

    assert dsa_mod._default_dsa_sparse_prefill_impl() == "trtllm"


def test_non_blackwell_dsa_prefill_keeps_flashmla(monkeypatch) -> None:
    monkeypatch.setattr(
        dsa_mod,
        "current_platform",
        lambda: SimpleNamespace(is_nvidia=True, is_blackwell=False),
    )

    assert dsa_mod._default_dsa_sparse_prefill_impl() == "flashmla"


def test_glm_dsa_sparse_prefill_dispatch_uses_selected_impl(monkeypatch) -> None:
    backend = object.__new__(DSABackend)
    out = torch.empty(1, 1)
    seen = {}

    def fake_prefill(**kwargs):
        seen.update(kwargs)
        return out

    monkeypatch.setattr(backend, "_forward_sparse_prefill_trtllm", fake_prefill)
    backend._set_sparse_prefill_impl("trtllm")

    q = torch.empty(1, 2, dtype=torch.bfloat16)
    block_tables = torch.zeros(1, 1, dtype=torch.int32)
    seq_lens = torch.ones(1, dtype=torch.int32)
    workspace_indices = torch.zeros(1, 4, dtype=torch.int32)
    topk_lens = torch.ones(1, dtype=torch.int32)
    kv_workspace_slots = torch.arange(4, dtype=torch.int64)

    result = backend.forward_sparse_prefill(
        q=q,
        layer=SimpleNamespace(),
        token_to_kv_pool=SimpleNamespace(),
        block_tables=block_tables,
        seq_lens=seq_lens,
        workspace_indices=workspace_indices,
        topk_lens=topk_lens,
        kv_workspace_slots=kv_workspace_slots,
        max_seq_len=8,
    )

    assert result is out
    assert seen["q"] is q
    assert seen["block_tables"] is block_tables
    assert seen["seq_lens"] is seq_lens
    assert seen["workspace_indices"] is workspace_indices
    assert seen["topk_lens"] is topk_lens
    assert seen["kv_workspace_slots"] is kv_workspace_slots
    assert seen["max_seq_len"] == 8


def test_glm_dsa_sparse_dispatch_rejects_unknown_impl() -> None:
    backend = object.__new__(DSABackend)

    with pytest.raises(ValueError, match="sparse decode implementation"):
        backend._set_sparse_decode_impl("unknown")


def test_glm_dsa_fp8_kv_allowed_on_blackwell(monkeypatch) -> None:
    monkeypatch.setattr(
        dsa_config_mod,
        "_is_blackwell_device",
        lambda device: True,
    )

    config = DSAConfig.generate(_server_args("fp8"), _model_config())

    assert config.kv_cache_dtype == torch.float8_e4m3fn


def test_glm_dsa_fp8_kv_requires_blackwell(monkeypatch) -> None:
    monkeypatch.setattr(
        dsa_config_mod,
        "_is_blackwell_device",
        lambda device: False,
    )

    with pytest.raises(ValueError, match="requires the Blackwell TRTLLM"):
        DSAConfig.generate(_server_args("fp8"), _model_config())


def test_glm_dsa_plain_sparse_decode_uses_trtllm_path(monkeypatch) -> None:
    backend = object.__new__(DSABackend)
    backend._dense_backend = SimpleNamespace(
        trtllm_workspace=torch.empty(16, dtype=torch.uint8),
        forward_decode_metadata=SimpleNamespace(
            num_extends=0,
            seq_lens_k=torch.tensor([2305, 2306], dtype=torch.int32),
            max_seq_len_k=4096,
        ),
    )
    backend.data_type = torch.bfloat16
    backend.page_size = 64
    backend.kv_cache_dim = 576
    backend.qk_nope_head_dim = 512
    backend.kv_lora_rank = 512
    backend.qk_rope_head_dim = 64
    backend.index_topk = 4

    layer = SimpleNamespace(
        layer_id=3,
        tp_q_head_num=2,
        head_dim=576,
        v_head_dim=512,
        k_scale_float=1.25,
        scaling=0.5,
    )
    q = torch.arange(2 * 2 * 576, dtype=torch.float32).to(torch.bfloat16)
    q = q.view(2, 2 * 576)
    key_buffer = torch.zeros(2 * 64, 576, dtype=torch.bfloat16)
    topk_indices = torch.tensor(
        [[9, 7, 5, -1], [14, 12, 10, 8]],
        dtype=torch.int32,
    )
    seen = {}

    def fake_trtllm_batch_decode_with_kv_cache_mla(**kwargs):
        seen.update(kwargs)
        return torch.ones(2, 1, 2, 512, dtype=torch.bfloat16)

    monkeypatch.setattr(
        dsa_mod,
        "trtllm_batch_decode_with_kv_cache_mla",
        fake_trtllm_batch_decode_with_kv_cache_mla,
    )

    out = backend._forward_sparse_decode_trtllm(
        q=q,
        layer=layer,
        token_to_kv_pool=_TokenToKVPool(key_buffer),
        num_reqs=2,
        topk_indices=topk_indices,
    )

    assert out is not None
    assert out.shape == (2, 2 * 512)
    assert seen["query"].shape == (2, 1, 2, 576)
    assert seen["kv_cache"].shape == (2, 1, 64, 576)
    assert seen["block_tables"].shape == (2, 1, 4)
    assert torch.equal(seen["block_tables"].view(2, 4), topk_indices)
    assert torch.equal(seen["seq_lens"], torch.tensor([2305, 2306], dtype=torch.int32))
    assert seen["max_seq_len"] == 4096
    assert seen["sparse_mla_top_k"] == 4
    assert seen["bmm1_scale"] == 0.625
    assert seen["backend"] == "trtllm-gen"


def test_glm_dsa_sparse_prefill_uses_trtllm_path(monkeypatch) -> None:
    backend = object.__new__(DSABackend)
    backend._dense_backend = SimpleNamespace(
        trtllm_workspace=torch.empty(16, dtype=torch.uint8),
    )
    backend.data_type = torch.float8_e4m3fn
    backend.page_size = 64
    backend.kv_cache_dim = 576
    backend.qk_nope_head_dim = 512
    backend.kv_lora_rank = 512
    backend.qk_rope_head_dim = 64
    backend.index_topk = 4

    layer = SimpleNamespace(
        layer_id=3,
        tp_q_head_num=2,
        head_dim=576,
        v_head_dim=512,
        k_scale_float=1.25,
        scaling=0.5,
    )
    q = torch.arange(3 * 2 * 576, dtype=torch.float32).to(torch.bfloat16)
    q = q.view(3, 2 * 576)
    key_buffer = torch.zeros(2 * 64, 576, dtype=torch.float8_e4m3fn)
    workspace_indices = torch.tensor(
        [[0, 2, -1, -1], [3, 4, 5, 6], [1, -1, -1, -1]],
        dtype=torch.int32,
    )
    kv_workspace_slots = torch.tensor([10, 11, 12, 70, 71, 72, 73], dtype=torch.int64)
    topk_lens = torch.tensor([2, 4, 1], dtype=torch.int32)
    seen = {}

    def fake_trtllm_batch_decode_with_kv_cache_mla(**kwargs):
        seen.update(kwargs)
        return torch.ones(3, 1, 2, 512, dtype=torch.bfloat16)

    monkeypatch.setattr(
        dsa_mod,
        "trtllm_batch_decode_with_kv_cache_mla",
        fake_trtllm_batch_decode_with_kv_cache_mla,
    )

    out = backend._forward_sparse_prefill_trtllm(
        q=q,
        layer=layer,
        token_to_kv_pool=_TokenToKVPool(key_buffer),
        workspace_indices=workspace_indices,
        topk_lens=topk_lens,
        kv_workspace_slots=kv_workspace_slots,
        max_seq_len=2307,
    )

    expected_block_tables = torch.tensor(
        [[10, 12, -1, -1], [70, 71, 72, 73], [11, -1, -1, -1]],
        dtype=torch.int32,
    )
    assert out.shape == (3, 2 * 512)
    assert seen["query"].dtype == torch.float8_e4m3fn
    assert seen["query"].shape == (3, 1, 2, 576)
    assert seen["kv_cache"].shape == (2, 1, 64, 576)
    assert torch.equal(seen["block_tables"].view(3, 4), expected_block_tables)
    assert torch.equal(seen["seq_lens"], topk_lens)
    assert seen["max_seq_len"] == 2307
    assert seen["sparse_mla_top_k"] == 4
    assert seen["bmm1_scale"] == 0.625
    assert seen["backend"] == "trtllm-gen"


def test_glm_dsa_multi_token_sparse_decode_uses_trtllm(monkeypatch) -> None:
    backend = object.__new__(DSABackend)
    backend._dense_backend = SimpleNamespace(
        trtllm_workspace=torch.empty(16, dtype=torch.uint8),
        forward_decode_metadata=SimpleNamespace(
            num_extends=0,
            seq_lens_k=torch.tensor([2306, 3002], dtype=torch.int32),
            max_seq_len_k=4096,
        ),
    )
    backend.data_type = torch.bfloat16
    backend.page_size = 64
    backend.kv_cache_dim = 576
    backend.qk_nope_head_dim = 512
    backend.kv_lora_rank = 512
    backend.qk_rope_head_dim = 64
    backend.index_topk = 4
    backend._sparse_decode_impl = "trtllm"

    layer = SimpleNamespace(
        layer_id=3,
        tp_q_head_num=2,
        head_dim=576,
        v_head_dim=512,
        k_scale_float=None,
        scaling=0.5,
    )
    q = torch.arange(4 * 2 * 576, dtype=torch.float32).to(torch.bfloat16)
    q = q.view(4, 2 * 576)
    key_buffer = torch.zeros(4 * 64, 576, dtype=torch.bfloat16)
    topk_indices = torch.tensor(
        [
            [9, 7, 5, -1],
            [10, 8, 6, 4],
            [14, 12, 10, 8],
            [15, 13, 11, 9],
        ],
        dtype=torch.int32,
    )
    seen = {}

    def fake_trtllm_batch_decode_with_kv_cache_mla(**kwargs):
        seen.update(kwargs)
        return torch.ones(4, 1, 2, 512, dtype=torch.bfloat16)

    monkeypatch.setattr(
        dsa_mod,
        "trtllm_batch_decode_with_kv_cache_mla",
        fake_trtllm_batch_decode_with_kv_cache_mla,
    )

    out = backend._forward_sparse_decode(
        q=q,
        k=torch.empty(0),
        v=torch.empty(0),
        layer=layer,
        out_cache_loc=torch.empty(0, dtype=torch.int64),
        token_to_kv_pool=_TokenToKVPool(key_buffer),
        bs=2,
        save_kv_cache=False,
        topk_indices=topk_indices,
        topk_lens=None,
    )

    assert out.shape == (4, 2 * 512)
    assert seen["query"].shape == (4, 1, 2, 576)
    assert torch.equal(seen["block_tables"].view(4, 4), topk_indices)
    assert torch.equal(
        seen["seq_lens"],
        torch.tensor([2305, 2306, 3001, 3002], dtype=torch.int32),
    )
    assert seen["max_seq_len"] == 4096
    assert seen["sparse_mla_top_k"] == 4
    assert seen["bmm1_scale"] == 0.5


def test_glm_dsa_plain_sparse_decode_requires_trtllm(monkeypatch) -> None:
    backend = object.__new__(DSABackend)
    backend.page_size = 64
    backend.index_topk = 4
    backend.kv_lora_rank = 512

    monkeypatch.setattr(
        dsa_mod,
        "trtllm_batch_decode_with_kv_cache_mla",
        dsa_mod.error_fn,
    )

    with pytest.raises(RuntimeError, match="requires TRTLLM sparse MLA"):
        backend._forward_sparse_decode(
            q=torch.zeros(2, 2 * 576, dtype=torch.bfloat16),
            k=torch.empty(0),
            v=torch.empty(0),
            layer=SimpleNamespace(),
            out_cache_loc=torch.empty(0, dtype=torch.int64),
            token_to_kv_pool=SimpleNamespace(),
            bs=2,
            save_kv_cache=False,
            topk_indices=torch.zeros(2, 4, dtype=torch.int32),
            topk_lens=None,
        )
