from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends import dsa as dsa_mod
from tokenspeed.runtime.layers.attention.backends.dsa import DSABackend
from tokenspeed.runtime.layers.attention.backends.trtllm_mla import TRTLLMMLABackend
from tokenspeed.runtime.layers.attention.configs import dsa as dsa_config_mod
from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig
from tokenspeed.runtime.models import glm5 as glm5_mod
from tokenspeed.runtime.models.glm5 import GlmMoeDsaAttention


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


def test_blackwell_fp8_dsa_prefill_defaults_to_trtllm(monkeypatch) -> None:
    monkeypatch.setattr(
        dsa_mod,
        "current_platform",
        lambda: SimpleNamespace(is_nvidia=True, is_blackwell=True),
    )

    assert dsa_mod._default_dsa_sparse_prefill_impl(torch.float8_e4m3fn) == "trtllm"


def test_blackwell_bf16_dsa_prefill_defaults_to_flashmla(monkeypatch) -> None:
    monkeypatch.setattr(
        dsa_mod,
        "current_platform",
        lambda: SimpleNamespace(is_nvidia=True, is_blackwell=True),
    )

    assert dsa_mod._default_dsa_sparse_prefill_impl(torch.bfloat16) == "flashmla"


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


def test_glm_dsa_backend_allows_cuda_graph_capture(monkeypatch) -> None:
    monkeypatch.setattr(
        dsa_mod,
        "TRTLLMMLABackend",
        lambda config: SimpleNamespace(),
    )

    config = SimpleNamespace(
        device="cuda",
        num_attention_heads=2,
        attn_tp_size=1,
        num_kv_heads=1,
        dtype=torch.bfloat16,
        head_dim=576,
        is_draft=False,
        speculative_num_draft_tokens=6,
        context_len=4096,
        page_size=64,
        kv_lora_rank=512,
        qk_nope_head_dim=512,
        qk_rope_head_dim=64,
        v_head_dim=512,
        kv_cache_dim=576,
        scaling=0.5,
        kv_cache_dtype=torch.float8_e4m3fn,
        index_topk=16,
        index_head_dim=128,
        index_n_heads=2,
    )

    backend = DSABackend(config)

    assert getattr(backend, "max_cuda_graph_batch_size", None) is None


def test_glm_dsa_seq_lens_fit_topk_cache_tracks_in_place_updates() -> None:
    metadata = SimpleNamespace()
    seq_lens_buf = torch.tensor([4, 4, 0, 0], dtype=torch.int32)
    seq_lens = seq_lens_buf[:2]

    assert GlmMoeDsaAttention._decode_seq_lens_fit_topk(
        metadata,
        seq_lens=seq_lens,
        topk=4,
        num_extends=0,
        num_decode_reqs=2,
        capturing=False,
    )

    seq_lens_buf[:2].copy_(torch.tensor([4, 5], dtype=torch.int32))

    assert not GlmMoeDsaAttention._decode_seq_lens_fit_topk(
        metadata,
        seq_lens=seq_lens,
        topk=4,
        num_extends=0,
        num_decode_reqs=2,
        capturing=False,
    )


def test_glm_dsa_seq_lens_fit_topk_handles_inference_tensors() -> None:
    metadata = SimpleNamespace()
    with torch.inference_mode():
        seq_lens = torch.tensor([4, 5], dtype=torch.int32)

    assert GlmMoeDsaAttention._tensor_version_or_none(seq_lens) is None
    assert not GlmMoeDsaAttention._decode_seq_lens_fit_topk(
        metadata,
        seq_lens=seq_lens,
        topk=4,
        num_extends=0,
        num_decode_reqs=2,
        capturing=False,
    )
    assert not hasattr(metadata, "_glm_dsa_seq_lens_fit_topk_cache")


def test_glm_dsa_full_context_topk_cache_tracks_in_place_updates(
    monkeypatch,
) -> None:
    attention = object.__new__(GlmMoeDsaAttention)
    attention.index_topk = 4
    calls = []

    def fake_compute_decode_full_context_topk_indices(**kwargs):
        calls.append(kwargs["seq_lens"].clone())
        idx = len(calls)
        return (
            torch.full((2, 4), idx, dtype=torch.int32),
            torch.full((2,), idx, dtype=torch.int32),
        )

    monkeypatch.setattr(
        attention,
        "_compute_decode_full_context_topk_indices",
        fake_compute_decode_full_context_topk_indices,
    )

    seq_lens_buf = torch.tensor([3, 3, 0, 0], dtype=torch.int32)
    block_tables = torch.tensor([[0], [1]], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_extends=0,
        seq_lens_k=seq_lens_buf,
        block_kv_indices=block_tables,
    )
    ctx = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        bs=2,
        num_extends=0,
        attn_backend=SimpleNamespace(
            forward_decode_metadata=metadata,
            spec_num_tokens=1,
        ),
        token_to_kv_pool=SimpleNamespace(
            page_size=1,
            get_index_k_buffer=lambda layer_id: None,
        ),
    )

    first = attention._try_compute_decode_full_context_topk_indices(
        ctx,
        num_tokens=2,
        device=torch.device("cpu"),
    )
    assert first is not None
    assert torch.equal(first.topk_lens, torch.tensor([1, 1], dtype=torch.int32))

    seq_lens_buf[:2].copy_(torch.tensor([3, 4], dtype=torch.int32))

    second = attention._try_compute_decode_full_context_topk_indices(
        ctx,
        num_tokens=2,
        device=torch.device("cpu"),
    )
    assert second is not None
    assert torch.equal(second.topk_lens, torch.tensor([2, 2], dtype=torch.int32))
    assert len(calls) == 2
    assert torch.equal(calls[0], torch.tensor([3, 3], dtype=torch.int32))
    assert torch.equal(calls[1], torch.tensor([3, 4], dtype=torch.int32))


def test_glm_dsa_mtp_draft_full_context_topk_clamps_to_metadata_rows(
    monkeypatch,
) -> None:
    attention = object.__new__(GlmMoeDsaAttention)
    attention.index_topk = 16
    calls = []

    def fake_compute_decode_full_context_topk_indices(**kwargs):
        calls.append(kwargs)
        num_decode_tokens = kwargs["num_decode_tokens"]
        lens = torch.zeros((kwargs["num_tokens"],), dtype=torch.int32)
        decode_start = kwargs["decode_start"]
        lens[decode_start : decode_start + num_decode_tokens] = 7
        return (
            torch.full((kwargs["num_tokens"], 16), 9, dtype=torch.int32),
            lens,
        )

    monkeypatch.setattr(
        attention,
        "_compute_decode_full_context_topk_indices",
        fake_compute_decode_full_context_topk_indices,
    )

    metadata = SimpleNamespace(
        num_extends=0,
        seq_lens_k=torch.tensor([3], dtype=torch.int32),
        block_kv_indices=torch.tensor([[0]], dtype=torch.int32),
    )
    ctx = SimpleNamespace(
        forward_mode=ForwardMode.DRAFT_EXTEND,
        bs=2,
        num_extends=0,
        attn_backend=SimpleNamespace(
            forward_decode_metadata=metadata,
            spec_num_tokens=6,
            is_draft=True,
        ),
        token_to_kv_pool=SimpleNamespace(
            page_size=1,
            get_index_k_buffer=lambda layer_id: None,
        ),
    )

    full_topk = attention._try_compute_decode_full_context_topk_indices(
        ctx,
        num_tokens=12,
        device=torch.device("cpu"),
    )

    assert full_topk is not None
    assert len(calls) == 1
    assert calls[0]["num_decode_tokens"] == 6
    assert calls[0]["decode_start"] == 6
    assert torch.equal(
        calls[0]["seq_lens"],
        torch.tensor([3, 4, 5, 6, 7, 8], dtype=torch.int32),
    )
    assert calls[0]["block_tables"].shape == (6, 1)
    assert torch.equal(
        full_topk.topk_lens,
        torch.tensor([0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7], dtype=torch.int32),
    )


def test_glm_dsa_deepgemm_prefill_passes_chunk_max_seqlen(monkeypatch) -> None:
    attention = object.__new__(GlmMoeDsaAttention)
    attention.indexer = SimpleNamespace(
        index_head_dim=128,
        index_n_heads=2,
        softmax_scale=0.5,
    )
    attention.attn_mqa = SimpleNamespace(layer_id=3)
    seen = {}

    def fake_quantize_fp8_with_scale(x, **kwargs):
        return (
            x.to(torch.float8_e4m3fn),
            torch.ones((x.shape[0], 1), dtype=torch.float32),
        )

    def fake_fp8_mqa_logits(
        q,
        kv,
        weights,
        row_starts,
        row_ends,
        *,
        clean_logits,
        max_seqlen_k,
    ):
        seen["max_seqlen_k"] = max_seqlen_k
        seen["row_ends"] = row_ends.clone()
        return torch.zeros((q.shape[0], 13), dtype=torch.float32)

    def fake_indexer_topk_prefill(logits, row_starts, row_ends, out, topk):
        out[:, :2] = torch.tensor([0, 1], dtype=torch.int32)

    class _TokenToKVPool:
        page_size = 64

        def has_index_k_with_scale_buffer(self) -> bool:
            return True

        def gather_index_k_with_scale(self, layer_id, slots):
            assert layer_id == 3
            return (
                torch.zeros((slots.numel(), 128), dtype=torch.uint8),
                torch.ones((slots.numel(), 1), dtype=torch.float32),
            )

    monkeypatch.setattr(glm5_mod, "fast_topk_v2", object())
    monkeypatch.setattr(
        glm5_mod,
        "deep_gemm",
        SimpleNamespace(fp8_mqa_logits=fake_fp8_mqa_logits),
    )
    monkeypatch.setattr(
        glm5_mod,
        "quantize_fp8_with_scale",
        fake_quantize_fp8_with_scale,
    )
    monkeypatch.setattr(
        glm5_mod.torch.ops,
        "trtllm",
        SimpleNamespace(indexer_topk_prefill=fake_indexer_topk_prefill),
        raising=False,
    )
    monkeypatch.setitem(
        glm5_mod.global_server_args_dict,
        "deepseek_v4_indexer_prefill_max_logits_mb",
        1,
    )

    result = attention._compute_prefill_topk_indices_deepgemm(
        indexer_output=glm5_mod.GlmDsaIndexerOutput(
            query=torch.zeros((3, 2, 128), dtype=torch.float32),
            key=torch.empty(0),
            weights=torch.ones((3, 2), dtype=torch.float32),
        ),
        ctx=SimpleNamespace(token_to_kv_pool=_TokenToKVPool()),
        prefix_lens=torch.tensor([10], dtype=torch.int32),
        extend_lens=torch.tensor([3], dtype=torch.int32),
        seq_lens=torch.tensor([13], dtype=torch.int32),
        block_tables=torch.tensor([[0]], dtype=torch.int32),
        kv_workspace_slots=torch.arange(13, dtype=torch.int64),
        kv_workspace_bases=torch.tensor([100], dtype=torch.int32),
        max_seq_len=13,
        num_prefill_tokens=3,
        topk=512,
    )

    assert result is not None
    assert seen["max_seqlen_k"] == 13
    assert torch.equal(seen["row_ends"], torch.tensor([11, 12, 13], dtype=torch.int32))
    assert torch.equal(result.topk_lens, torch.tensor([11, 12, 13], dtype=torch.int32))


def test_glm_dsa_decode_window_uses_metadata_rows_for_mtp_mismatch() -> None:
    metadata = SimpleNamespace(
        num_extends=0,
        seq_lens_k=torch.tensor([3], dtype=torch.int32),
        block_kv_indices=torch.tensor([[0]], dtype=torch.int32),
    )
    ctx = SimpleNamespace(
        bs=2,
        num_extends=0,
        attn_backend=SimpleNamespace(
            forward_decode_metadata=metadata,
            spec_num_tokens=6,
        ),
    )

    window = GlmMoeDsaAttention._resolve_decode_window(
        ctx,
        metadata,
        total_tokens=12,
    )

    assert window.start == 6
    assert window.end == 12
    assert window.num_tokens == 6
    assert window.num_reqs == 1
    assert window.q_len_per_req == 6


def test_glm_dsa_decode_topk_window_slices_indices_and_lens() -> None:
    decode_topk = glm5_mod.GlmDsaDecodeTopK(
        topk_indices=torch.arange(12 * 4, dtype=torch.int32).view(12, 4),
        topk_lens=torch.arange(12, dtype=torch.int32),
    )

    assert GlmMoeDsaAttention._decode_topk_covers_window(decode_topk, 6, 12)
    assert not GlmMoeDsaAttention._decode_topk_covers_window(decode_topk, 12, 18)

    topk_indices, topk_lens = GlmMoeDsaAttention._slice_decode_topk(
        decode_topk,
        6,
        12,
    )

    assert torch.equal(topk_indices, decode_topk.topk_indices[6:12])
    assert torch.equal(topk_lens, decode_topk.topk_lens[6:12])


def test_glm_dsa_decode_topk_prefers_deterministic_kernel_in_capture(
    monkeypatch,
) -> None:
    attention = object.__new__(GlmMoeDsaAttention)
    calls = []

    def fake_deterministic(logits, out, topk):
        calls.append(("deterministic", logits, topk))
        out.fill_(7)

    def fake_fast_topk(*args, **kwargs):
        calls.append(("fast", args, kwargs))

    monkeypatch.setattr(
        glm5_mod.torch.cuda,
        "is_current_stream_capturing",
        lambda: True,
    )
    monkeypatch.setattr(glm5_mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(glm5_mod, "has_deterministic_decode_topk", lambda: True)
    monkeypatch.setattr(
        glm5_mod,
        "glm_dsa_decode_topk_deterministic",
        fake_deterministic,
    )
    monkeypatch.setattr(glm5_mod, "fast_topk_v2", fake_fast_topk)

    logits = torch.arange(8, dtype=torch.float32).view(2, 4)
    seq_lens = torch.tensor([[4], [3]], dtype=torch.int32)
    out = torch.empty((2, 2), dtype=torch.int32)

    attention._write_decode_topk_offsets(
        logits=logits,
        seq_lens_2d=seq_lens,
        local_topk_offsets=out,
        topk=2,
    )

    assert calls == [("deterministic", logits, 2)]
    assert torch.equal(out, torch.full((2, 2), 7, dtype=torch.int32))


def test_glm_dsa_fp8_kv_allowed_on_blackwell(monkeypatch) -> None:
    monkeypatch.setattr(
        dsa_config_mod,
        "_is_blackwell_device",
        lambda device: True,
    )

    config = DSAConfig.generate(_server_args("fp8"), _model_config())

    assert config.kv_cache_dtype == torch.float8_e4m3fn


def test_glm_dsa_keeps_cuda_graph_replay_enabled_for_sparse_decode(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        dsa_mod,
        "TRTLLMMLABackend",
        lambda config: SimpleNamespace(_block_table_aliased=False),
    )

    backend = DSABackend(
        SimpleNamespace(
            device=torch.device("cpu"),
            num_attention_heads=2,
            attn_tp_size=1,
            num_kv_heads=1,
            dtype=torch.bfloat16,
            head_dim=576,
            is_draft=False,
            speculative_num_draft_tokens=1,
            index_topk=4,
            context_len=4096,
            page_size=64,
            kv_lora_rank=512,
            qk_nope_head_dim=512,
            qk_rope_head_dim=64,
            v_head_dim=512,
            kv_cache_dim=576,
            scaling=0.5,
            kv_cache_dtype=torch.float8_e4m3fn,
        )
    )

    assert getattr(backend, "max_cuda_graph_batch_size", None) is None


def test_trtllm_mla_target_verify_initializes_decode_metadata(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "tokenspeed.runtime.layers.attention.backends.trtllm_mla.get_trtllm_workspace_buffer",
        lambda device: torch.empty(16, dtype=torch.uint8),
    )
    backend = TRTLLMMLABackend(
        SimpleNamespace(
            device=torch.device("cpu"),
            context_len=4096,
            page_size=64,
            kv_lora_rank=512,
            qk_nope_head_dim=512,
            qk_rope_head_dim=64,
            v_head_dim=512,
            kv_cache_dim=576,
            scaling=0.5,
            kv_cache_dtype=torch.float8_e4m3fn,
            dtype=torch.bfloat16,
            num_attention_heads=2,
            num_kv_heads=1,
            head_dim=576,
            attn_tp_size=1,
            is_draft=False,
            speculative_num_draft_tokens=6,
        )
    )
    req_pool_indices = torch.tensor([2, 0], dtype=torch.int64)
    seq_lens = torch.tensor([128, 256], dtype=torch.int32)
    req_to_page = torch.arange(4 * 64, dtype=torch.int32).view(4, 64)

    backend.init_forward_metadata(
        bs=2,
        num_extends=0,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        forward_mode=ForwardMode.TARGET_VERIFY,
        req_to_page=req_to_page,
    )

    metadata = backend.forward_decode_metadata
    assert metadata is not None
    assert metadata.num_extends == 0
    assert metadata.seq_lens_k is seq_lens
    assert metadata.max_seq_len_k == 4096
    assert torch.equal(metadata.block_kv_indices[0, :4], req_to_page[2, :4])
    assert torch.equal(metadata.block_kv_indices[1, :4], req_to_page[0, :4])


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
    topk_lens = torch.tensor([3, 4], dtype=torch.int32)
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
        topk_lens=topk_lens,
    )

    assert out is not None
    assert out.shape == (2, 2 * 512)
    assert seen["query"].shape == (2, 1, 2, 576)
    assert seen["kv_cache"].shape == (2, 1, 64, 576)
    assert seen["block_tables"].shape == (2, 1, 4)
    assert torch.equal(seen["block_tables"].view(2, 4), topk_indices)
    assert torch.equal(seen["seq_lens"], topk_lens)
    assert seen["max_seq_len"] == 4096
    assert seen["sparse_mla_top_k"] == 4
    assert seen["bmm1_scale"] == 0.625
    assert seen["backend"] == "trtllm-gen"


def test_glm_dsa_mtp_draft_sparse_decode_clamps_to_metadata_rows(
    monkeypatch,
) -> None:
    backend = object.__new__(DSABackend)
    backend._dense_backend = SimpleNamespace(
        trtllm_workspace=torch.empty(16, dtype=torch.uint8),
        forward_decode_metadata=SimpleNamespace(
            num_extends=0,
            seq_lens_k=torch.tensor([2305], dtype=torch.int32),
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
    backend.is_draft = True

    layer = SimpleNamespace(
        layer_id=3,
        tp_q_head_num=2,
        head_dim=576,
        v_head_dim=512,
        k_scale_float=1.25,
        scaling=0.5,
    )
    q = torch.arange(6 * 2 * 576, dtype=torch.float32).to(torch.bfloat16)
    q = q.view(6, 2 * 576)
    key_buffer = torch.zeros(2 * 64, 576, dtype=torch.bfloat16)
    topk_indices = torch.arange(6 * 4, dtype=torch.int32).view(6, 4)
    topk_lens = torch.tensor([1, 2, 3, 4, 4, 4], dtype=torch.int32)
    seen = {}

    def fake_trtllm_batch_decode_with_kv_cache_mla(**kwargs):
        seen.update(kwargs)
        return torch.ones(6, 1, 2, 512, dtype=torch.bfloat16)

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
        q_len_per_req=3,
        topk_indices=topk_indices,
        topk_lens=topk_lens,
    )

    assert out.shape == (6, 2 * 512)
    assert seen["query"].shape == (6, 1, 2, 576)
    assert torch.equal(seen["block_tables"].view(6, 4), topk_indices)
    assert torch.equal(seen["seq_lens"], topk_lens)


def test_glm_dsa_flashmla_sparse_prefill_omits_topk_length(monkeypatch) -> None:
    backend = object.__new__(DSABackend)
    backend.data_type = torch.bfloat16
    backend.page_size = 64
    backend.kv_cache_dim = 576
    backend.kv_lora_rank = 512
    backend.index_topk = 4
    backend._prefill_workspace_buffer = None
    backend._prefill_workspace_rows = 0
    backend._prefill_workspace_dim = 0
    backend._prefill_query_workspace = None
    backend._prefill_query_workspace_num_heads = None

    layer = SimpleNamespace(
        layer_id=3,
        logit_cap=0.0,
        tp_q_head_num=2,
        head_dim=576,
        v_head_dim=512,
        scaling=0.5,
    )
    q = torch.arange(3 * 2 * 576, dtype=torch.float32).to(torch.bfloat16)
    q = q.view(3, 2 * 576)
    key_buffer = torch.zeros(2 * 64, 576, dtype=torch.bfloat16)
    workspace_indices = torch.tensor(
        [[0, 2, -1, -1], [3, 4, 5, 6], [1, -1, -1, -1]],
        dtype=torch.int32,
    )
    kv_workspace_slots = torch.tensor([10, 11, 12, 70, 71, 72, 73], dtype=torch.int64)
    topk_lens = torch.tensor([2, 4, 1], dtype=torch.int32)
    seen = {}

    def fake_flash_mla_sparse_fwd(**kwargs):
        seen.update(kwargs)
        return (
            torch.ones(
                kwargs["q"].shape[0],
                kwargs["q"].shape[1],
                layer.v_head_dim,
                dtype=torch.bfloat16,
            ),
            None,
            None,
        )

    monkeypatch.setattr(dsa_mod, "flash_mla_sparse_fwd", fake_flash_mla_sparse_fwd)

    out = backend._forward_sparse_prefill_flashmla(
        q=q,
        layer=layer,
        token_to_kv_pool=_TokenToKVPool(key_buffer),
        block_tables=torch.zeros(1, 1, dtype=torch.int32),
        seq_lens=torch.tensor([7], dtype=torch.int32),
        workspace_indices=workspace_indices,
        topk_lens=topk_lens,
        kv_workspace_slots=kv_workspace_slots,
        max_seq_len=7,
    )

    assert out.shape == (3, 2 * 512)
    assert "topk_length" not in seen
    assert seen["q"].shape[0] == 3
    assert seen["kv"].shape == (7, 1, 576)
    assert torch.equal(seen["indices"].squeeze(1), workspace_indices)
    assert seen["sm_scale"] == 0.5
    assert seen["d_v"] == 512


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


def test_glm_dsa_fp8_sparse_prefill_accepts_fp8_query_for_trtllm(
    monkeypatch,
) -> None:
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
        k_scale_float=None,
        scaling=0.5,
    )
    q = torch.arange(2 * 2 * 576, dtype=torch.float32).to(torch.float8_e4m3fn)
    q = q.view(2, 2 * 576)
    key_buffer = torch.zeros(2 * 64, 576, dtype=torch.float8_e4m3fn)
    workspace_indices = torch.tensor(
        [[0, 2, -1, -1], [3, 4, 5, 6]],
        dtype=torch.int32,
    )
    kv_workspace_slots = torch.tensor([10, 11, 12, 70, 71, 72, 73], dtype=torch.int64)
    topk_lens = torch.tensor([2, 4], dtype=torch.int32)
    seen = {}

    def fake_trtllm_batch_decode_with_kv_cache_mla(**kwargs):
        seen.update(kwargs)
        return torch.ones(2, 1, 2, 512, dtype=torch.bfloat16)

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
        [[10, 12, -1, -1], [70, 71, 72, 73]],
        dtype=torch.int32,
    )
    assert out.shape == (2, 2 * 512)
    assert seen["query"].dtype == torch.float8_e4m3fn
    assert seen["query"].shape == (2, 1, 2, 576)
    assert seen["kv_cache"].dtype == torch.float8_e4m3fn
    assert torch.equal(seen["block_tables"].view(2, 4), expected_block_tables)
    assert torch.equal(seen["seq_lens"], topk_lens)
    assert seen["max_seq_len"] == 2307
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
    topk_lens = torch.tensor([3, 4, 4, 4], dtype=torch.int32)
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
        topk_lens=topk_lens,
    )

    assert out.shape == (4, 2 * 512)
    assert seen["query"].shape == (4, 1, 2, 576)
    assert torch.equal(seen["block_tables"].view(4, 4), topk_indices)
    assert torch.equal(seen["seq_lens"], topk_lens)
    assert seen["max_seq_len"] == 4096
    assert seen["sparse_mla_top_k"] == 4
    assert seen["bmm1_scale"] == 0.5


def test_glm_dsa_fp8_sparse_decode_accepts_fp8_query_for_trtllm(
    monkeypatch,
) -> None:
    backend = object.__new__(DSABackend)
    backend._dense_backend = SimpleNamespace(
        trtllm_workspace=torch.empty(16, dtype=torch.uint8),
        forward_decode_metadata=SimpleNamespace(
            num_extends=0,
            seq_lens_k=torch.tensor([2306, 3002], dtype=torch.int32),
            max_seq_len_k=4096,
        ),
    )
    backend.data_type = torch.float8_e4m3fn
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
        k_scale_float=1.25,
        scaling=0.5,
    )
    q = torch.arange(4 * 2 * 576, dtype=torch.float32).to(torch.float8_e4m3fn)
    q = q.view(4, 2 * 576)
    key_buffer = torch.zeros(4 * 64, 576, dtype=torch.float8_e4m3fn)
    topk_indices = torch.tensor(
        [
            [9, 7, 5, -1],
            [10, 8, 6, 4],
            [14, 12, 10, 8],
            [15, 13, 11, 9],
        ],
        dtype=torch.int32,
    )
    topk_lens = torch.tensor([3, 4, 4, 4], dtype=torch.int32)
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
        topk_lens=topk_lens,
    )

    assert out.shape == (4, 2 * 512)
    assert seen["query"].dtype == torch.float8_e4m3fn
    assert seen["query"].shape == (4, 1, 2, 576)
    assert seen["kv_cache"].dtype == torch.float8_e4m3fn
    assert torch.equal(seen["block_tables"].view(4, 4), topk_indices)
    assert torch.equal(seen["seq_lens"], topk_lens)
    assert seen["backend"] == "trtllm-gen"


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
