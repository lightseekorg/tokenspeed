"""Kimi-K3 multimodal wrapper coverage without constructing the 93-layer LM."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tokenspeed.runtime.configs.kimi_k3_config import KimiK3Config
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.models import kimi_k3
from tokenspeed.runtime.models.kimi_k3 import (
    KimiK3ForConditionalGeneration,
    KimiK3Vision,
)
from tokenspeed.runtime.models.moonvit import MoonViTVisionPath
from tokenspeed.runtime.multimodal.inputs import Modality


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


def _tiny_config() -> KimiK3Config:
    return KimiK3Config(
        text_config={"hidden_size": 24},
        vision_config={
            "patch_size": 2,
            "init_pos_emb_height": 4,
            "init_pos_emb_width": 4,
            "init_pos_emb_time": 2,
            "vt_num_attention_heads": 3,
            "vt_num_hidden_layers": 1,
            "vt_hidden_size": 16,
            "vt_intermediate_size": 32,
            "qkv_hidden_size": 24,
            "merge_kernel_size": (2, 2),
            "text_hidden_size": 24,
        },
    )


def _item_dp_mapping(rank: int = 5) -> Mapping:
    return Mapping(
        rank=rank,
        world_size=8,
        attn_tp_size=8,
        vision_tp_size=1,
        vision_dp_size=8,
    )


def _build_model(monkeypatch, mapping: Mapping | None = None):
    monkeypatch.setattr(kimi_k3, "KimiLinearForCausalLM", _FakeLanguageModel)
    return KimiK3ForConditionalGeneration(
        _tiny_config(),
        mapping=mapping or Mapping(rank=0, world_size=1),
        mm_attention_backend="triton_attn",
    )


def _vision_checkpoint_weights(model):
    loaded = []
    expected = {}
    for index, (name, param) in enumerate(
        model.vision.named_parameters(remove_duplicate=False), start=1
    ):
        checkpoint_name = name.replace("attn.qkv_proj.", "wqkv.")
        checkpoint_name = checkpoint_name.replace("attn.proj.", "wo.")
        checkpoint_name = checkpoint_name.replace(
            "mm_projector.linear_1", "mm_projector.proj.0"
        )
        checkpoint_name = checkpoint_name.replace(
            "mm_projector.linear_2", "mm_projector.proj.2"
        )
        value = torch.full_like(param, index / 100)
        loaded.append((checkpoint_name, value))
        expected[name] = value
    return loaded, expected


def test_kimi_k3_factory_wires_real_vision_item_dp(monkeypatch):
    mapping = _item_dp_mapping()
    model = _build_model(monkeypatch, mapping)

    assert isinstance(model.vision, KimiK3Vision)
    assert isinstance(model.vision, MoonViTVisionPath)
    assert model.vision.vision_spec.hidden_size == 16
    assert not hasattr(model.config.vision_config, "hidden_size")
    assert model.vision.vision_tower.encoder.blocks[0].attn.tp_size == 1
    assert model.vision_embedder._encoder_dp_rank == 5
    assert model.vision_embedder._encoder_dp_group == tuple(range(8))
    assert model.image_encoder == model.vision.embed_media


def test_kimi_k3_forward_splices_prefill_and_skips_decode(monkeypatch):
    model = _build_model(monkeypatch)
    merged = torch.randn(3, 24)
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


def test_kimi_k3_vision_loader_maps_checkpoint_names(monkeypatch):
    model = _build_model(monkeypatch)
    loaded, expected = _vision_checkpoint_weights(model)
    original_named_parameters = model.vision.named_parameters
    loaded.append(("language_model.probe", torch.tensor([1.0])))

    named_parameter_calls = 0

    def counted_named_parameters(*args, **kwargs):
        nonlocal named_parameter_calls
        named_parameter_calls += 1
        return original_named_parameters(*args, **kwargs)

    monkeypatch.setattr(model.vision, "named_parameters", counted_named_parameters)
    model.load_weights(loaded)

    assert named_parameter_calls == 1
    for name, param in original_named_parameters(remove_duplicate=False):
        torch.testing.assert_close(param, expected[name])
    assert len(model.language_model.loaded_weights) == 1
    assert model.language_model.loaded_weights[0][0] == "probe"


def test_kimi_k3_encoder_only_skips_language_model(monkeypatch):
    config = _tiny_config()
    config.encoder_only = True

    def fail_language_model_construction(*_args, **_kwargs):
        raise AssertionError("encoder-only K3 must not construct the language model")

    monkeypatch.setattr(
        kimi_k3, "KimiLinearForCausalLM", fail_language_model_construction
    )
    model = KimiK3ForConditionalGeneration(
        config,
        mapping=Mapping(rank=0, world_size=1),
        mm_attention_backend="triton_attn",
    )

    assert model.language_model is None
    assert model.image_encoder == model.vision.embed_media
    assert model.vision_tower is model.vision.vision_tower
    with pytest.raises(AttributeError, match="encoder-only"):
        model.get_input_embeddings()
    with pytest.raises(RuntimeError, match="encoder-only"):
        model.forward(None, None, None, None)


def test_kimi_k3_encoder_only_drains_stream_and_loads_all_vision_weights(
    monkeypatch,
):
    config = _tiny_config()
    config.encoder_only = True
    monkeypatch.setattr(
        kimi_k3,
        "KimiLinearForCausalLM",
        lambda *_args, **_kwargs: pytest.fail(
            "encoder-only K3 must not construct the language model"
        ),
    )
    model = KimiK3ForConditionalGeneration(
        config,
        mapping=Mapping(rank=0, world_size=1),
        mm_attention_backend="triton_attn",
    )
    vision_weights, expected = _vision_checkpoint_weights(model)
    midpoint = len(vision_weights) // 2
    checkpoint = [
        ("language_model.before", torch.tensor([1.0])),
        *vision_weights[:midpoint],
        ("language_model.middle", torch.tensor([2.0])),
        *vision_weights[midpoint:],
        ("language_model.after", torch.tensor([3.0])),
    ]

    model.load_weights(iter(checkpoint))

    for name, param in model.vision.named_parameters(remove_duplicate=False):
        torch.testing.assert_close(param, expected[name])


def test_kimi_k3_encoder_graph_targets_top_level_image_encoder(monkeypatch):
    mapping = _item_dp_mapping()
    model = _build_model(monkeypatch, mapping)

    singular = model.make_encoder_cudagraph_wrapper(mapping)
    wrappers = model.make_encoder_cudagraph_wrappers(mapping)

    assert set(wrappers) == {"image_encoder"}
    for wrapper in (singular, wrappers["image_encoder"]):
        assert wrapper.adapter.tower is model.vision.vision_tower.encoder
        assert wrapper.capture_tp_size == 1
        assert wrapper.capture_tp_group == (5,)
