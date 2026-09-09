"""GLM-5.3-Flash model smoke tests using dummy configs and weights."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tokenspeed.runtime.configs.glm53_flash_config import (
    Glm53FlashConfig,
    Glm53FlashTextConfig,
    Glm53FlashVisionConfig,
)
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.layers.attention import kpool as kpool_runtime
from tokenspeed.runtime.layers.attention.kpool import KPoolRuntime
from tokenspeed.runtime.layers.dense.fp8 import Fp8LinearMethod
from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
from tokenspeed.runtime.models import glm53_flash
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


@pytest.mark.parametrize(
    ("decode_start", "num_decode_tokens", "expected_fill_value", "expected_fill"),
    [
        pytest.param(0, 4, None, False, id="full-decode"),
        pytest.param(1, 2, -1, True, id="mixed-prefill"),
    ],
)
def test_glm53_flash_decode_topk_skips_only_overwritten_workspace_fills(
    decode_start: int,
    num_decode_tokens: int,
    expected_fill_value: int | None,
    expected_fill: bool,
) -> None:
    attention = Glm53FlashAttention.__new__(Glm53FlashAttention)
    attention.index_topk = 12
    attention.index_kpool = 4
    attention.indexer = SimpleNamespace(weights_softmax_scale=0.25)
    attention.attn_mqa = SimpleNamespace(layer_id=3)
    captured = {}
    topk_indices = torch.empty((4, 15), dtype=torch.int32)
    topk_lens = torch.empty(4, dtype=torch.int32)

    def get_indices(_name, _rows, _width, _device, *, fill_value=-1):
        captured["fill_value"] = fill_value
        return topk_indices

    def get_lens(_rows, _device, *, fill=True):
        captured["fill"] = fill
        return topk_lens

    def select_decode(**kwargs):
        captured["out"] = kwargs["out"]
        captured["lens_out"] = kwargs["lens_out"]

    attention._get_decode_topk_workspace = get_indices
    attention._get_decode_topk_lens_workspace = get_lens
    ctx = SimpleNamespace(
        attn_backend=SimpleNamespace(
            require_kpool_runtime=lambda: SimpleNamespace(
                select_decode=select_decode,
            )
        )
    )
    indexer_output = Glm53FlashIndexerOutput(
        query=torch.zeros((4, 1, 128), dtype=torch.bfloat16),
        key=torch.empty(0),
        weights=torch.zeros((4, 1), dtype=torch.bfloat16),
        gate=torch.empty(0),
    )

    result = attention._compute_decode_topk_indices_portable(
        indexer_output=indexer_output,
        ctx=ctx,
        seq_lens=torch.tensor([8], dtype=torch.int32),
        page_table=torch.zeros((1, 1), dtype=torch.int32),
        q_len_per_req=1,
        decode_start=decode_start,
        num_tokens=4,
        num_decode_tokens=num_decode_tokens,
        topk=12,
    )

    assert captured["fill_value"] == expected_fill_value
    assert captured["fill"] is expected_fill
    assert captured["out"] is topk_indices
    assert captured["lens_out"] is topk_lens
    assert result.topk_indices is topk_indices
    assert result.topk_lens is topk_lens


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
        attn_tp_size=1,
        attn_dp_size=2,
        linear_attn_tp_size=2,
        dense_tp_size=2,
        moe_tp_size=2,
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
