"""GLM-5.3-Flash model smoke tests using dummy configs and weights."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from tokenspeed.runtime.configs.glm53_flash_config import (
    Glm53FlashConfig,
    Glm53FlashTextConfig,
    Glm53FlashVisionConfig,
)
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention import kpool as kpool_runtime
from tokenspeed.runtime.layers.attention.backends import dsa as dsa_backend_module
from tokenspeed.runtime.layers.attention.backends.dsa import (
    DSABackend,
    DSAForwardMetadata,
)
from tokenspeed.runtime.layers.attention.kpool import (
    KPoolRuntime,
    build_kpool_prefill_plan,
)
from tokenspeed.runtime.layers.dense.fp8 import Fp8LinearMethod
from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
from tokenspeed.runtime.models import deepseek_v3, glm53_flash, glm53_flash_nextn
from tokenspeed.runtime.models.glm5 import GlmDsaDecodeTopK, GlmDsaPrefillTopK
from tokenspeed.runtime.models.glm53_flash import (
    Glm53FlashAttention,
    Glm53FlashForCausalLM,
    Glm53FlashForConditionalGeneration,
    Glm53FlashIndexerOutput,
    Glm53FlashKDA,
    Glm53FlashMoE,
)
from tokenspeed.runtime.models.glm53_flash_nextn import (
    Glm53FlashForConditionalGenerationNextN,
)
from tokenspeed.runtime.multimodal.inputs import Modality
from tokenspeed.runtime.utils.env import global_server_args_dict


class _FakeBackbone(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(32, hidden_size)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens


class _FakeLanguageModel(nn.Module):
    def __init__(self, config, **_kwargs) -> None:
        super().__init__()
        self.model = _FakeBackbone(config.hidden_size)
        self.lm_head = nn.Identity()
        self.logits_processor = object()
        self.forward_kwargs = None
        self.loaded_weights = None

    def forward(self, _ctx, _input_ids, _positions, _out_cache_loc, **kwargs):
        self.forward_kwargs = kwargs
        return kwargs.get("input_embeds")

    def load_weights(self, weights) -> None:
        self.loaded_weights = list(weights)


def test_dsa_draft_capture_keeps_target_cache_groups_out_of_dense_mla() -> None:
    captured = {}
    backend = DSABackend.__new__(DSABackend)
    backend.is_draft = True
    backend.spec_num_tokens = 1
    backend.kernel_page_size = 64
    backend._reset_forward_metadata = lambda _indices: None
    backend._dense_backend = SimpleNamespace(
        init_forward_metadata_capture_cuda_graph=lambda **kwargs: captured.update(
            kwargs
        ),
        forward_decode_metadata=SimpleNamespace(),
    )

    with mock.patch.object(dsa_backend_module, "dsa_plan", return_value=object()):
        backend.init_forward_metadata_capture_cuda_graph(
            bs=1,
            req_pool_indices=torch.zeros(1, dtype=torch.int64),
            seq_lens=torch.ones(1, dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
            cache_group_ids=("full_attention", "glm53_flash.dsa_index"),
        )

    assert captured["cache_group_ids"] == ()


def _tiny_text_config() -> Glm53FlashTextConfig:
    return Glm53FlashTextConfig(
        vocab_size=32,
        pad_token_id=0,
        hidden_size=16,
        intermediate_size=32,
        # Registered portable and gfx950 MoE kernels require a tile-aligned
        # intermediate dimension; the hierarchy test does not depend on size.
        moe_intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        kv_lora_rank=4,
        q_lora_rank=8,
        qk_head_dim=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=0,
        v_head_dim=8,
        index_topk=8,
        index_head_dim=8,
        index_n_heads=2,
        index_kpool=4,
        linear_attn_config={
            "num_heads": 2,
            "head_dim": 8,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "kda_layers": [0, 1, 2],
            "full_attn_layers": [3],
        },
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "deepseek_sparse_attention",
        ],
        mlp_layer_types=["dense", "dense", "dense", "sparse"],
        indexer_types=["full", "full", "full", "full"],
    )


def _tiny_config() -> Glm53FlashConfig:
    return Glm53FlashConfig(
        text_config=_tiny_text_config(),
        vision_config=Glm53FlashVisionConfig(
            depth=1,
            hidden_size=16,
            num_heads=2,
            patch_size=2,
            temporal_patch_size=1,
            spatial_merge_size=2,
            out_hidden_size=16,
            intermediate_size=32,
            projection_intermediate_size=32,
        ),
    )


def test_kpool_prefill_plan_maps_unaligned_writes_and_ragged_rows() -> None:
    plan = build_kpool_prefill_plan(
        prefix_lens_cpu=torch.tensor([3, 8], dtype=torch.int32),
        extend_lens_cpu=torch.tensor([6, 5], dtype=torch.int32),
        index_block_table=torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int32),
        request_slots=torch.tensor([30, 40], dtype=torch.int32),
        kpool=4,
        index_rows_per_page=2,
    )

    assert plan.num_prefill_tokens == 11
    assert plan.max_num_pools == 3
    assert plan.query_start_loc.tolist() == [0, 6, 11]
    assert plan.positions.tolist() == [3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12]
    assert plan.req_ids.tolist() == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    assert plan.causal_lens.tolist() == [4, 5, 6, 7, 8, 9, 9, 10, 11, 12, 13]
    assert plan.pool_workspace_slots.tolist() == [20, 21, 40, 41, 42]
    assert plan.row_starts.tolist() == [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2]
    assert plan.row_ends.tolist() == [1, 1, 1, 1, 2, 2, 4, 4, 4, 5, 5]

    write = plan.write
    assert write.pool_req_ids.tolist() == [0, 0, 1]
    assert write.pool_n_from_tail.tolist() == [3, 0, 0]
    assert write.pool_chunk_src.tolist() == [0, 1, 6]
    assert write.pool_tail_logical_base.tolist() == [0, 4, 8]
    assert write.pool_write_slots.tolist() == [20, 21, 42]
    assert write.tail_req_ids.tolist() == [0, 1]
    assert write.tail_chunk_src.tolist() == [5, 10]
    assert write.tail_dst_positions.tolist() == [8, 12]
    assert write.tail_write_counts.tolist() == [1, 1]
    assert write.request_slots.tolist() == [30, 40]


def test_kpool_prefill_plan_marks_bucket_tail_inactive() -> None:
    plan = build_kpool_prefill_plan(
        prefix_lens_cpu=torch.tensor([3, 8], dtype=torch.int32),
        extend_lens_cpu=torch.tensor([6, 5], dtype=torch.int32),
        index_block_table=torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int32),
        request_slots=torch.tensor([30, 40], dtype=torch.int32),
        kpool=4,
        index_rows_per_page=2,
        token_capacity=15,
    )

    assert plan.num_prefill_tokens == 11
    assert plan.positions[:11].tolist() == [3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12]
    assert plan.positions[11:].tolist() == [0, 0, 0, 0]
    assert plan.req_ids[11:].tolist() == [0, 0, 0, 0]
    assert plan.causal_lens[11:].tolist() == [0, 0, 0, 0]
    assert plan.row_starts[11:].tolist() == [0, 0, 0, 0]
    assert plan.row_ends[11:].tolist() == [0, 0, 0, 0]


def test_kpool_prefill_plan_rejects_capacity_below_live_rows() -> None:
    with pytest.raises(ValueError, match="capacity=10, tokens=11"):
        build_kpool_prefill_plan(
            prefix_lens_cpu=torch.tensor([3, 8], dtype=torch.int32),
            extend_lens_cpu=torch.tensor([6, 5], dtype=torch.int32),
            index_block_table=torch.tensor(
                [[10, 11, 12], [20, 21, 22]], dtype=torch.int32
            ),
            request_slots=torch.tensor([30, 40], dtype=torch.int32),
            kpool=4,
            index_rows_per_page=2,
            token_capacity=10,
        )


def test_kpool_prefill_tail_write_uses_fused_kernel(monkeypatch) -> None:
    plan = build_kpool_prefill_plan(
        prefix_lens_cpu=torch.tensor([0], dtype=torch.int32),
        extend_lens_cpu=torch.tensor([1], dtype=torch.int32),
        index_block_table=torch.tensor([[1]], dtype=torch.int32),
        request_slots=torch.tensor([30], dtype=torch.int32),
        kpool=4,
        index_rows_per_page=16,
    )
    captured = {}

    def fake_tail_write(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(kpool_runtime, "kpool_prefill_tail_write", fake_tail_write)
    index_cache = torch.empty((2, 16, 132), dtype=torch.uint8)
    tail_k = torch.empty((32, 7, 128), dtype=torch.bfloat16)
    tail_gate = torch.empty_like(tail_k)
    pool = SimpleNamespace(
        get_kpool_buffers=lambda _layer_id: (index_cache, tail_k, tail_gate),
    )
    backend = SimpleNamespace(
        forward_metadata=DSAForwardMetadata(kpool_prefill_plan=plan)
    )
    ctx = SimpleNamespace(token_to_kv_pool=pool)
    key = torch.randn((1, 128), dtype=torch.bfloat16)
    gate = torch.randn_like(key)

    KPoolRuntime(pool_size=4, index_topk=8).write_prefill(
        key=key,
        gate=gate,
        compress_ape=torch.empty((4, 128)),
        ctx=ctx,
        backend=backend,
        layer_id=3,
    )

    args = captured["args"]
    assert args[0] is key
    assert args[1] is gate
    assert args[2] is tail_k
    assert args[3] is tail_gate
    assert args[4] is plan.write.tail_chunk_src
    assert args[5].tolist() == [30]
    assert args[6] is plan.write.tail_dst_positions
    assert args[7] is plan.write.tail_write_counts
    assert captured["kwargs"] == {"pool_size": 4}


def test_kpool_decode_forwards_configured_context_bound(monkeypatch) -> None:
    captured = {}

    def fake_kpool_decode_topk(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return kwargs["out"], kwargs["lens_out"]

    monkeypatch.setattr(kpool_runtime, "kpool_decode_topk", fake_kpool_decode_topk)
    index_cache = torch.empty((4, 16, 132), dtype=torch.uint8)
    ctx = SimpleNamespace(
        attn_backend=SimpleNamespace(max_context_len=131072),
        token_to_kv_pool=SimpleNamespace(
            arena=SimpleNamespace(kv_page_size=64),
            get_kpool_buffers=lambda _layer_id: (index_cache, None, None),
        ),
    )
    query = torch.zeros((5, 2, 128), dtype=torch.bfloat16)
    weights = torch.zeros((5, 2), dtype=torch.float32)
    seq_lens = torch.tensor([1024, 2048], dtype=torch.int32)
    page_table = torch.zeros((2, 4), dtype=torch.int32)
    out = torch.empty((5, 2051), dtype=torch.int32)
    lens_out = torch.empty(5, dtype=torch.int32)

    KPoolRuntime(pool_size=4, index_topk=2048).select_decode(
        query=query,
        weights=weights,
        softmax_scale=0.125,
        ctx=ctx,
        layer_id=3,
        seq_lens=seq_lens,
        page_table=page_table,
        q_len_per_req=2,
        decode_start=1,
        num_decode_tokens=4,
        out=out,
        lens_out=lens_out,
    )

    assert captured["args"][0].shape == (4, 2, 128)
    assert captured["kwargs"]["max_seq_len"] == 131072
    assert captured["kwargs"]["page_size"] == 16
    assert captured["kwargs"]["kv_page_size"] == 64
    assert captured["kwargs"]["out"].shape == (4, 2051)
    assert captured["kwargs"]["lens_out"].shape == (4,)


def test_glm53_flash_prefill_preserves_expanded_tail_slots(monkeypatch) -> None:
    attention = Glm53FlashAttention.__new__(Glm53FlashAttention)
    attention.index_kpool = 4
    attention.index_topk = 12
    attention.indexer = SimpleNamespace(weights_softmax_scale=0.25)
    attention.attn_mqa = SimpleNamespace(layer_id=3)
    runtime = KPoolRuntime(pool_size=4, index_topk=12)
    metadata = SimpleNamespace(
        extend_prefix_lens=torch.tensor([4], dtype=torch.int32),
        extend_seq_lens=torch.tensor([1], dtype=torch.int32),
        extend_prefix_lens_cpu=torch.tensor([4], dtype=torch.int32),
        extend_seq_lens_cpu=torch.tensor([1], dtype=torch.int32),
    )
    backend = SimpleNamespace(
        chunked_prefill_metadata=metadata,
        forward_metadata=DSAForwardMetadata(
            kpool_prefill_plan=SimpleNamespace(
                num_prefill_tokens=1,
                query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
                positions=torch.tensor([4, 0, 0], dtype=torch.int32),
                req_ids=torch.tensor([0, 0, 0], dtype=torch.int32),
                causal_lens=torch.tensor([5, 0, 0], dtype=torch.int32),
                pool_workspace_slots=torch.empty(0, dtype=torch.int64),
                row_starts=torch.tensor([0, 0, 0], dtype=torch.int32),
                row_ends=torch.tensor([0, 0, 0], dtype=torch.int32),
                max_num_pools=1,
            )
        ),
        kpool_prefill_page_table=lambda num_requests: torch.zeros(
            (1, 1), dtype=torch.int32
        )[:num_requests],
        require_kpool_runtime=lambda: runtime,
    )
    index_cache = torch.zeros((1, 64), dtype=torch.uint8)
    ctx = SimpleNamespace(
        num_extends=1,
        attn_backend=backend,
        token_to_kv_pool=SimpleNamespace(
            arena=SimpleNamespace(kv_page_size=64),
            get_kpool_buffers=lambda _layer_id: (index_cache, None, None),
        ),
    )
    selected = torch.full((3, 15), -1, dtype=torch.int32)
    selected[0, :5] = torch.tensor([8, 9, 10, 11, 64], dtype=torch.int32)

    def fake_kpool_prefill_topk(*args, **kwargs):
        assert "output_layout" not in kwargs
        assert args[5].data_ptr() == args[6].data_ptr()
        assert args[0].shape[0] == 3
        assert kwargs["causal_lens"].tolist() == [5, 0, 0]
        return selected, torch.tensor([5, 0, 0], dtype=torch.int32)

    monkeypatch.setattr(kpool_runtime, "kpool_prefill_topk", fake_kpool_prefill_topk)
    result = attention._compute_prefill_topk_indices(
        Glm53FlashIndexerOutput(
            query=torch.zeros((3, 1, 128), dtype=torch.bfloat16),
            key=torch.empty(0),
            weights=torch.zeros((3, 1)),
            gate=torch.empty(0),
        ),
        ctx,
        num_prefill_tokens=1,
    )

    assert result is not None
    torch.testing.assert_close(
        result.workspace_indices[0],
        torch.tensor(
            [0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            dtype=torch.int32,
        ),
    )
    assert torch.equal(
        result.workspace_indices[1:],
        torch.full((2, 15), -1, dtype=torch.int32),
    )
    torch.testing.assert_close(result.kv_workspace_slots, selected.flatten())
    torch.testing.assert_close(
        result.kv_seq_lens,
        torch.tensor([5, 0, 0], dtype=torch.int32),
    )


def test_glm53_flash_decode_topk_uses_live_rows_before_padding() -> None:
    attention = Glm53FlashAttention.__new__(Glm53FlashAttention)
    attention.index_topk = 12
    captured = {}

    def fake_portable(**kwargs):
        captured.update(kwargs)
        return GlmDsaDecodeTopK(
            topk_indices=torch.empty((8, 12), dtype=torch.int32),
            topk_lens=torch.empty(8, dtype=torch.int32),
        )

    attention._compute_decode_topk_indices_portable = fake_portable
    metadata = SimpleNamespace(
        num_extends=2,
        seq_lens_k=torch.tensor([6, 5, 9], dtype=torch.int32),
        block_kv_indices=torch.zeros((3, 2), dtype=torch.int32),
    )
    backend = SimpleNamespace(
        forward_decode_metadata=metadata,
        spec_num_tokens=1,
    )
    ctx = SimpleNamespace(bs=3, num_extends=2, attn_backend=backend)
    indexer_output = Glm53FlashIndexerOutput(
        query=torch.zeros((8, 1, 128), dtype=torch.bfloat16),
        key=torch.empty(0),
        weights=torch.zeros((8, 1)),
        gate=torch.empty(0),
    )

    result = attention._compute_decode_topk_indices(
        indexer_output,
        ctx,
        logical_num_tokens=4,
    )

    assert result is not None
    assert captured["decode_start"] == 3
    assert captured["num_decode_tokens"] == 1
    assert captured["num_tokens"] == 8


def test_sparse_prefill_projects_bucket_but_writes_only_live_kv(monkeypatch) -> None:
    attention = Glm53FlashAttention.__new__(Glm53FlashAttention)
    attention.num_local_heads = 1
    attention.qk_head_dim = 2
    attention.qk_nope_head_dim = 2
    attention.qk_rope_head_dim = 0
    attention.kv_lora_rank = 3
    attention.attention_backend = "dsa"
    attention.rotary_emb = None
    attention.attn_mqa = SimpleNamespace(k_scale_float=1.0, layer_id=7)
    attention.w_kc = torch.zeros((1, 3, 2), dtype=torch.bfloat16)

    def fake_bmm(_lhs, _rhs, *, out):
        out.fill_(7)
        return out

    monkeypatch.setattr(deepseek_v3, "bmm", fake_bmm)
    captured = {}

    def select_out_cache_loc(_layer, locations, _mode):
        captured["input_locations"] = locations.clone()
        return locations + 100

    def set_mla_kv_buffer(_layer, locations, cache_k_nope, cache_k_rope):
        captured["write_locations"] = locations.clone()
        captured["cache_k_nope"] = cache_k_nope.clone()
        captured["cache_k_rope"] = cache_k_rope.clone()

    ctx = SimpleNamespace(
        attn_backend=SimpleNamespace(
            data_type=torch.bfloat16,
            select_out_cache_loc=select_out_cache_loc,
        ),
        token_to_kv_pool=SimpleNamespace(set_mla_kv_buffer=set_mla_kv_buffer),
        forward_mode=ForwardMode.EXTEND,
    )
    q = torch.zeros((5, 2), dtype=torch.bfloat16)
    latent = torch.arange(15, dtype=torch.bfloat16).view(5, 3)
    locations = torch.arange(5, dtype=torch.int64)

    projected_q, projected_k = attention.forward_absorb_qkv_proj(
        q,
        latent,
        torch.zeros(5, dtype=torch.int64),
        ctx,
        locations,
        cache_num_tokens=2,
    )

    assert projected_q.shape == (5, 1, 3)
    assert projected_k is not None and projected_k.shape == (5, 1, 3)
    assert captured["input_locations"].tolist() == [0, 1]
    assert captured["write_locations"].tolist() == [100, 101]
    torch.testing.assert_close(captured["cache_k_nope"].squeeze(1), latent[:2])
    assert captured["cache_k_rope"].shape == (2, 1, 0)


def test_mixed_prefill_decode_keeps_live_rows_before_padding(monkeypatch) -> None:
    events = {}
    prefill_plan = SimpleNamespace(num_prefill_tokens=3)

    class _FakeKPoolRuntime:
        def ensure_prefill_plan(
            self, _ctx, _backend, _layer_id, *, token_capacity
        ) -> None:
            events["capacity"] = token_capacity

        def write_prefill(self, *, key, gate, **_kwargs) -> None:
            events["prefill_write_rows"] = (key.shape[0], gate.shape[0])

        def write_decode(self, *, key, gate, **kwargs) -> None:
            events["decode_write_rows"] = (key.shape[0], gate.shape[0])
            events["decode_write_key"] = key.clone()
            events["decode_shape"] = (
                kwargs["num_reqs"],
                kwargs["q_len_per_req"],
            )

    class _FakeDSABackend:
        spec_num_tokens = 1

        def __init__(self) -> None:
            self.forward_metadata = SimpleNamespace(kpool_prefill_plan=prefill_plan)
            self.forward_decode_metadata = SimpleNamespace(
                num_extends=2,
                seq_lens_k=torch.tensor([2, 1, 9], dtype=torch.int32),
                block_kv_indices=torch.zeros((3, 1), dtype=torch.int32),
            )
            self.kpool_runtime = _FakeKPoolRuntime()

        def require_kpool_runtime(self):
            return self.kpool_runtime

    monkeypatch.setattr(glm53_flash, "DSABackend", _FakeDSABackend)
    backend = _FakeDSABackend()
    attention = Glm53FlashAttention.__new__(Glm53FlashAttention)
    attention.attn_mqa = SimpleNamespace(layer_id=7)
    attention.q_lora_rank = 2
    attention.kv_lora_rank = 3
    attention.qk_rope_head_dim = 0
    attention.num_local_heads = 1
    attention.v_head_dim = 3
    attention.skip_indexer_topk = False
    attention.is_nextn = False

    def fused_qkv(hidden_states, _block_scale, _dtype):
        rows = hidden_states.shape[0]
        q = torch.zeros((rows, 2), dtype=torch.bfloat16)
        latent = torch.arange(rows, dtype=torch.bfloat16).unsqueeze(1).expand(-1, 3)
        return torch.cat((q, latent), dim=-1)

    def fused_norm(*, input_q_a, input_kv_a, output_q_a):
        del input_kv_a
        output_q_a.copy_(input_q_a)

    class _FakeIndexer:
        index_kpool_compress_ape = torch.empty(0)

        def __call__(self, hidden_states, _q_norm, _positions):
            rows = hidden_states.shape[0]
            row_values = torch.arange(rows, dtype=torch.bfloat16).unsqueeze(1)
            return Glm53FlashIndexerOutput(
                query=torch.zeros((rows, 1, 128), dtype=torch.bfloat16),
                key=row_values.expand(-1, 128).clone(),
                weights=torch.zeros((rows, 1), dtype=torch.float32),
                gate=row_values.expand(-1, 128).clone(),
            )

    def prefill_topk(_indexer_output, _ctx, num_tokens):
        events["prefill_topk_tokens"] = num_tokens
        return object()

    def decode_topk(_indexer_output, _ctx, *, logical_num_tokens):
        events["decode_topk_logical_rows"] = logical_num_tokens
        return GlmDsaDecodeTopK(
            topk_indices=torch.zeros((8, 4), dtype=torch.int32),
            topk_lens=torch.tensor([0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.int32),
        )

    def sparse_prefill(
        positions,
        q,
        latent_cache,
        prefill_ctx,
        out_cache_loc,
        output,
        *,
        prefill_topk,
        cache_num_tokens,
    ):
        del prefill_topk
        events["prefill_shapes"] = tuple(
            tensor.shape[0]
            for tensor in (positions, q, latent_cache, out_cache_loc, output)
        )
        events["prefill_context"] = (
            prefill_ctx.bs,
            prefill_ctx.input_num_tokens,
            cache_num_tokens,
        )
        output.zero_()
        output[:cache_num_tokens].fill_(1)

    def sparse_decode(
        positions,
        q,
        latent_cache,
        decode_ctx,
        out_cache_loc,
        output,
        *,
        topk_indices,
        topk_lens,
    ):
        events["decode_shapes"] = tuple(
            tensor.shape[0]
            for tensor in (positions, q, latent_cache, out_cache_loc, output)
        )
        events["decode_context"] = (decode_ctx.bs, decode_ctx.input_num_tokens)
        events["decode_topk"] = (topk_indices.shape[0], topk_lens.tolist())
        output.fill_(2)

    attention.fused_qkv_a_proj_with_mqa = fused_qkv
    attention.fused_qk_layernorm = fused_norm
    attention.indexer = _FakeIndexer()
    attention._compute_prefill_topk_indices = prefill_topk
    attention._compute_decode_topk_indices = decode_topk
    attention.q_b_proj = lambda q_norm: (
        torch.zeros((q_norm.shape[0], 1, 3), dtype=torch.bfloat16),
        None,
    )
    attention.forward_dsa_sparse_prefill = sparse_prefill
    attention.forward_absorb = sparse_decode
    attention.o_proj = lambda output: (output, None)

    ctx = ForwardContext(
        attn_backend=backend,
        token_to_kv_pool=SimpleNamespace(),
        bs=3,
        num_extends=2,
        input_num_tokens=8,
        real_input_num_tokens=4,
        forward_mode=ForwardMode.MIXED,
    )
    comm_manager = SimpleNamespace(pre_attn_comm=lambda tensor, _ctx: tensor)
    result = Glm53FlashAttention.forward.__wrapped__(
        attention,
        torch.arange(8),
        torch.zeros((8, 4), dtype=torch.bfloat16),
        ctx,
        torch.arange(8),
        comm_manager,
    )

    torch.testing.assert_close(
        events.pop("decode_write_key"),
        torch.full((1, 128), 3, dtype=torch.bfloat16),
    )
    assert events == {
        "capacity": 8,
        "prefill_write_rows": (8, 8),
        "decode_write_rows": (1, 1),
        "decode_shape": (1, 1),
        "prefill_topk_tokens": 3,
        "decode_topk_logical_rows": 4,
        "prefill_shapes": (8, 8, 8, 8, 8),
        "prefill_context": (2, 8, 3),
        "decode_shapes": (1, 1, 1, 1, 1),
        "decode_context": (1, 1),
        "decode_topk": (1, [1]),
    }
    torch.testing.assert_close(result[:3], torch.ones((3, 3), dtype=torch.bfloat16))
    torch.testing.assert_close(result[3], torch.full((3,), 2, dtype=torch.bfloat16))
    assert torch.count_nonzero(result[4:]) == 0


def _build_model(monkeypatch) -> Glm53FlashForConditionalGeneration:
    monkeypatch.setattr(glm53_flash, "Glm53FlashForCausalLM", _FakeLanguageModel)
    return Glm53FlashForConditionalGeneration(
        _tiny_config(),
        mapping=Mapping(rank=0, world_size=1),
    )


def test_glm53_flash_vision_applies_clamped_swiglu() -> None:
    config = Glm53FlashVisionConfig(
        hidden_size=2,
        intermediate_size=2,
        out_hidden_size=2,
        projection_intermediate_size=2,
        swiglu_limit=1.0,
    )
    mlp = glm53_flash.Glm53FlashVisionMLP(config)
    mlp.gate_proj = nn.Identity()
    mlp.up_proj = nn.Identity()
    mlp.down_proj = nn.Identity()

    hidden_states = torch.tensor([[2.0, -2.0]])
    expected_gate = torch.tensor([[1.0, -2.0]])
    expected_up = torch.tensor([[1.0, -1.0]])

    torch.testing.assert_close(
        mlp(hidden_states),
        torch.nn.functional.silu(expected_gate) * expected_up,
    )

    merger = glm53_flash.Glm53FlashVisionPatchMerger(config)
    merger.proj = nn.Identity()
    merger.post_projection_norm = nn.Identity()
    merger.gate_proj = nn.Identity()
    merger.up_proj = nn.Identity()
    merger.down_proj = nn.Identity()
    projected = torch.nn.functional.gelu(hidden_states)
    expected_gate = projected.clamp(max=1.0)
    expected_up = projected.clamp(min=-1.0, max=1.0)

    torch.testing.assert_close(
        merger(hidden_states),
        torch.nn.functional.silu(expected_gate) * expected_up,
    )


def test_glm53_flash_forward_splices_prefill_and_skips_decode(monkeypatch) -> None:
    model = _build_model(monkeypatch)
    merged = torch.randn(3, 16)
    apply_calls = []

    class FakeEmbedder:
        def apply(self, **kwargs):
            apply_calls.append(kwargs)
            return merged, {}

    model.vision_embedder = FakeEmbedder()
    multimodal_context = SimpleNamespace(has_extend_inputs=lambda: True)
    prefill_ctx = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode_or_idle=lambda: False)
    )
    args = (
        torch.tensor([1, 2, 3]),
        torch.arange(3),
        torch.arange(3),
    )

    output = model.forward(
        prefill_ctx,
        *args,
        multimodal_context=multimodal_context,
    )

    assert output is merged
    assert model.language_model.forward_kwargs["input_embeds"] is merged
    assert len(apply_calls) == 1
    assert apply_calls[0]["encoders"][Modality.IMAGE].fn is model.image_encoder
    assert apply_calls[0]["encoders"][Modality.VIDEO].fn is model.video_encoder

    decode_ctx = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode_or_idle=lambda: True)
    )
    output = model.forward(
        decode_ctx,
        *args,
        multimodal_context=multimodal_context,
    )

    assert output is None
    assert len(apply_calls) == 1
    assert "input_embeds" not in model.language_model.forward_kwargs


def test_glm53_flash_vision_loader_routes_checkpoint_weights(monkeypatch) -> None:
    model = _build_model(monkeypatch)
    checkpoint = []
    expected = {}
    for index, (name, param) in enumerate(
        model.vision.named_parameters(remove_duplicate=False), start=1
    ):
        value = torch.full_like(param, index / 100)
        checkpoint_name = name.replace(".attn.qkv_proj.", ".attn.qkv.")
        checkpoint.append((f"model.visual.{checkpoint_name}", value))
        expected[name] = value
    checkpoint.append(("language_model.probe", torch.tensor([1.0])))

    model.load_weights(iter(checkpoint))

    for name, param in model.vision.named_parameters(remove_duplicate=False):
        torch.testing.assert_close(param, expected[name])
    assert len(model.language_model.loaded_weights) == 1
    assert model.language_model.loaded_weights[0][0] == "probe"


def test_kda_checkpoint_input_weights_load_into_one_projection(monkeypatch) -> None:
    config = _tiny_text_config()
    mapping = Mapping(
        rank=1,
        world_size=2,
        attn_tp_size=2,
        dense_tp_size=2,
        moe_tp_size=2,
        vision_tp_size=2,
    )
    attention = Glm53FlashKDA(config, mapping, layer_id=0)

    model = Glm53FlashForCausalLM.__new__(Glm53FlashForCausalLM)
    nn.Module.__init__(model)
    model.config = config
    model.mapping = mapping
    model.quant_config = None
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module()])
    model.model.layers[0].self_attn = attention

    class _NoMoeWeights:
        @staticmethod
        def matches(_name) -> bool:
            return False

    monkeypatch.setattr(
        glm53_flash,
        "build_moe_checkpoint_loader",
        lambda **_kwargs: _NoMoeWeights(),
    )
    monkeypatch.setattr(Glm53FlashForCausalLM, "post_load_weights", lambda _self: None)

    projection_size = (
        config.linear_attn_config["num_heads"] * config.linear_attn_config["head_dim"]
    )
    shard_sizes = (
        projection_size,
        projection_size,
        projection_size,
        config.linear_attn_config["num_heads"],
        config.linear_attn_config["head_dim"],
        config.linear_attn_config["head_dim"],
    )
    shards = [
        torch.arange(size * config.hidden_size, dtype=torch.float32).view(
            size, config.hidden_size
        )
        + index * 1000
        for index, size in enumerate(shard_sizes)
    ]
    names = ("q", "k", "v", "b", "f_a", "g_a")
    model.load_weights(
        iter(
            (
                f"model.layers.0.self_attn.{name}_proj.weight",
                shard,
            )
            for name, shard in zip(names, shards, strict=True)
        )
    )

    expected = torch.cat(
        [
            *[shard[projection_size // 2 :] for shard in shards[:3]],
            shards[3][1:],
            shards[4],
            shards[5],
        ],
        dim=0,
    )
    torch.testing.assert_close(attention.fused_qkvbfg_a_proj.weight, expected)


def test_text_and_nextn_model_hierarchies(monkeypatch) -> None:
    mapping = Mapping(rank=0, world_size=1)
    monkeypatch.setitem(global_server_args_dict, "mapping", mapping)
    monkeypatch.setitem(global_server_args_dict, "attention_backend", "dsa")
    monkeypatch.setitem(global_server_args_dict, "ep_num_redundant_experts", 0)

    causal = Glm53FlashForCausalLM(_tiny_text_config(), mapping)
    nextn = Glm53FlashForConditionalGenerationNextN(_tiny_config(), mapping)

    assert causal.model.layers[0].is_kda_layer
    assert not causal.model.layers[3].is_kda_layer
    assert isinstance(nextn.model.decoder.self_attn, Glm53FlashAttention)
    assert isinstance(nextn.model.decoder.mlp, Glm53FlashMoE)
    assert not nextn.model.decoder.mhc
    assert (
        nextn.model.decoder.self_attn.kv_b_proj.prefix
        == "model.layers.4.self_attn.kv_b_proj"
    )


def test_nextn_checkpoint_prefix_preserves_bf16_kv_b_projection() -> None:
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        ignored_layers=["model.layers.4.self_attn.kv_b_proj"],
        weight_block_size=[128, 128],
    )
    with torch.device("meta"):
        attention = Glm53FlashAttention(
            _tiny_text_config(),
            Mapping(rank=0, world_size=1),
            layer_id=0,
            quant_config=quant_config,
            prefix="model.layers.4.self_attn",
        )

    assert isinstance(attention.kv_b_proj.quant_method, UnquantizedLinearMethod)
    assert isinstance(attention.q_b_proj.quant_method, Fp8LinearMethod)


def test_nextn_keeps_flat_kv_prefill_slots_when_compacting_topk() -> None:
    prefill = GlmDsaPrefillTopK(
        workspace_indices=torch.tensor(
            [[0, 1, -1], [3, 4, 5], [6, 7, 8]], dtype=torch.int32
        ),
        topk_lens=torch.tensor([2, 3, 3], dtype=torch.int32),
        page_table=torch.empty((0, 0), dtype=torch.int32),
        seq_lens=torch.empty(0, dtype=torch.int32),
        kv_seq_lens=torch.tensor([1, 2, 3], dtype=torch.int32),
        max_seq_len=0,
        kv_workspace_slots=torch.tensor(
            [10, 11, -1, 20, 21, 22, 30, 31, 32], dtype=torch.int32
        ),
    )
    decode = GlmDsaDecodeTopK(
        topk_indices=torch.tensor(
            [[40, 41, 42], [50, 51, 52], [60, 61, 62]], dtype=torch.int32
        ),
        topk_lens=torch.tensor([3, 3, 3], dtype=torch.int32),
    )

    _, selected = (
        Glm53FlashForConditionalGenerationNextN.prepare_dsa_topk_for_mtp_decode(
            (prefill, decode),
            torch.tensor([2, 0, 1], dtype=torch.int64),
            num_prefill_rows=2,
        )
    )

    torch.testing.assert_close(
        selected.topk_indices,
        torch.tensor([[30, 31, 32], [10, 11, -1], [50, 51, 52]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        selected.topk_lens,
        torch.tensor([3, 2, 3], dtype=torch.int32),
    )


def test_nextn_fused_qkv_fp8_scale_uses_block_row_offset() -> None:
    model = Glm53FlashForConditionalGenerationNextN.__new__(
        Glm53FlashForConditionalGenerationNextN
    )
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_hidden_layers=45,
        num_nextn_predict_layers=1,
        n_routed_experts=1,
        q_lora_rank=1536,
    )
    model.quant_config = SimpleNamespace(weight_block_size=[128, 128])
    model.mapping = SimpleNamespace(moe=SimpleNamespace(ep_rank=0, ep_size=1))

    scale = nn.Parameter(torch.zeros(16, 4), requires_grad=False)
    calls = []

    def weight_loader(param, loaded_weight, *, begin_size):
        calls.append((param, loaded_weight, begin_size))

    scale.weight_loader = weight_loader
    scale_name = "model.decoder.self_attn." "fused_qkv_a_proj_with_mqa.weight_scale_inv"
    model.named_parameters = lambda: [(scale_name, scale)]
    model.named_modules = lambda: []
    model.post_load_weights = lambda: None

    loaded_scale = torch.ones(4, 4)
    model.load_weights(
        [
            (
                "model.language_model.layers.45.self_attn."
                "kv_a_proj_with_mqa.weight_scale_inv",
                loaded_scale,
            )
        ]
    )

    assert calls == [(scale, loaded_scale, 12)]


def test_dsa_mtp_draft_keeps_caller_owned_cache_locations() -> None:
    backend = DSABackend.__new__(DSABackend)
    backend.is_draft = True
    caller_locs = torch.tensor([10, 11, 12, 13], dtype=torch.int32)

    selected = backend.select_out_cache_loc(
        SimpleNamespace(layer_id=0), caller_locs, ForwardMode.DECODE
    )

    assert selected is caller_locs


def test_dsa_mtp_draft_does_not_bind_dense_mla_cache_contract() -> None:
    calls = []
    backend = DSABackend.__new__(DSABackend)
    backend.is_draft = True
    backend._dense_backend = SimpleNamespace(
        mark_cache_contract=lambda: calls.append("marked")
    )

    backend.mark_cache_contract()

    assert calls == []


def test_dsa_target_binds_dense_mla_cache_contract() -> None:
    calls = []
    backend = DSABackend.__new__(DSABackend)
    backend.is_draft = False
    backend._dense_backend = SimpleNamespace(
        mark_cache_contract=lambda: calls.append("marked")
    )

    backend.mark_cache_contract()

    assert calls == ["marked"]


def test_nextn_catchup_keeps_full_window_until_model_forward(monkeypatch) -> None:
    events = []
    ctx = SimpleNamespace(
        draft_seq_lens_buf=torch.tensor([258], dtype=torch.int32),
        accept_lengths=torch.tensor([2], dtype=torch.int32),
        num_extends=0,
        bs=1,
        global_bs=None,
        attn_backend=SimpleNamespace(
            spec_num_tokens=4,
            advance_draft_forward_metadata=lambda seq_lens: events.append(
                ("publish", seq_lens.clone())
            ),
        ),
    )

    class _CatchupModel(nn.Module):
        def forward(
            self,
            input_ids,
            positions,
            forward_ctx,
            out_cache_loc,
            captured_hidden_states=None,
        ):
            del input_ids, positions, out_cache_loc
            events.append(("model", forward_ctx.draft_seq_lens_buf.clone()))
            return captured_hidden_states, None

    class _LogitsProcessor(nn.Module):
        def forward(self, input_ids, hidden_states, lm_head, metadata):
            del input_ids, lm_head, metadata
            events.append(("logits", ctx.draft_seq_lens_buf.clone()))
            return hidden_states

    model = Glm53FlashForConditionalGenerationNextN.__new__(
        Glm53FlashForConditionalGenerationNextN
    )
    nn.Module.__init__(model)
    model.model = _CatchupModel()
    model.lm_head = nn.Identity()
    model.logits_processor = _LogitsProcessor()
    monkeypatch.setattr(
        glm53_flash_nextn,
        "report_collective_sizing",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        glm53_flash_nextn.LogitsMetadata,
        "from_forward_context",
        lambda _ctx: None,
    )

    hidden_states = torch.randn(4, 8)
    output = model.forward(
        ctx=ctx,
        input_ids=torch.arange(4),
        positions=torch.arange(4),
        out_cache_loc=torch.arange(4),
        captured_hidden_states=hidden_states,
        spec_step_idx=0,
    )

    assert output is hidden_states
    assert [name for name, _ in events] == ["model", "publish", "logits"]
    torch.testing.assert_close(events[0][1], torch.tensor([258], dtype=torch.int32))
    torch.testing.assert_close(events[1][1], torch.tensor([256], dtype=torch.int32))
    torch.testing.assert_close(events[2][1], torch.tensor([256], dtype=torch.int32))
