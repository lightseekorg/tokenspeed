"""Unit tests for the shared Qwen3-ASR/Qwen3-Omni audio tower."""

import json
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.models.qwen3_audio import (
    Qwen3AudioEncoder,
    pack_qwen3_audio_features,
    qwen3_audio_output_lengths,
)
from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem


def _upstream_output_lengths(lengths: torch.Tensor) -> torch.Tensor:
    remainder = lengths % 100
    after_first_conv = (remainder - 1) // 2 + 1
    tail = ((after_first_conv - 1) // 2 + 1 - 1) // 2 + 1
    return tail + (lengths // 100) * 13


def _audio_item(feature, **model_specific_data):
    return MultimodalDataItem(
        modality=Modality.AUDIO,
        feature=feature,
        offsets=[(0, 0)],
        model_specific_data=model_specific_data,
    )


def _tiny_config():
    return SimpleNamespace(
        activation_function="gelu",
        conv_chunksize=2,
        d_model=32,
        downsample_hidden_size=4,
        encoder_attention_heads=4,
        encoder_ffn_dim=64,
        encoder_layers=1,
        max_source_positions=16,
        n_window=4,
        n_window_infer=16,
        num_mel_bins=8,
        output_dim=32,
    )


def _tiny_asr_config():
    return SimpleNamespace(
        architectures=["Qwen3ASRForConditionalGeneration"],
        thinker_config=SimpleNamespace(
            audio_config=_tiny_config(),
            text_config=SimpleNamespace(hidden_size=32, num_hidden_layers=2),
        ),
    )


def test_output_lengths_match_reference_and_custom_chunking():
    lengths = torch.tensor([1, 2, 7, 8, 99, 100, 101, 199, 200, 3000])
    torch.testing.assert_close(
        qwen3_audio_output_lengths(lengths), _upstream_output_lengths(lengths)
    )
    custom = qwen3_audio_output_lengths([8, 9, 16, 17], n_window=4)
    assert custom.tolist() == [1, 2, 2, 3]


def test_audio_tower_uses_platform_default_for_vision_only_cudnn_backend():
    normalize = Qwen3AudioEncoder._normalize_attention_backend
    assert normalize("flashinfer_cudnn") is None
    assert normalize("fa4") == "fa4"


def test_asr_config_parses_official_nested_shape(tmp_path):
    from tokenspeed.runtime.configs.qwen3_asr_config import Qwen3ASRConfig

    raw_config = {
        "architectures": ["Qwen3ASRForConditionalGeneration"],
        "model_type": "qwen3_asr",
        "support_languages": ["Chinese", "English"],
        "thinker_config": {
            "audio_config": {
                "model_type": "qwen3_asr_audio_encoder",
                "d_model": 1024,
                "encoder_layers": 24,
                "n_window": 50,
                "output_dim": 2048,
            },
            "audio_start_token_id": 151669,
            "audio_end_token_id": 151670,
            "audio_token_id": 151676,
            "dtype": "bfloat16",
            "text_config": {
                "model_type": "qwen3",
                "hidden_size": 2048,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "intermediate_size": 6144,
                "vocab_size": 151936,
            },
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(raw_config))
    config = Qwen3ASRConfig.from_pretrained(tmp_path)

    assert config.architectures == ["Qwen3ASRForConditionalGeneration"]
    assert config.thinker_config.text_config.hidden_size == 2048
    assert config.thinker_config.audio_config.encoder_layers == 24
    assert config.audio_start_token_id == 151669
    assert config.audio_end_token_id == 151670
    assert config.audio_token_id == 151676


def test_pack_features_uses_explicit_lengths_and_masks():
    first = torch.arange(8 * 6, dtype=torch.float32).reshape(1, 8, 6)
    second = torch.arange(8 * 5, dtype=torch.float32).reshape(8, 5) + 100
    packed, lengths = pack_qwen3_audio_features(
        [
            _audio_item(first, audio_feature_lengths=torch.tensor([4])),
            _audio_item(
                second,
                feature_attention_mask=torch.tensor([[1, 1, 1, 0, 0]]),
            ),
        ],
        num_mel_bins=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert lengths.tolist() == [4, 3]
    assert packed.shape == (8, 7)
    torch.testing.assert_close(packed[:, :4], first[0, :, :4])
    torch.testing.assert_close(packed[:, 4:], second[:, :3])


def test_pack_features_rejects_sparse_mask():
    item = _audio_item(
        torch.zeros(8, 4),
        feature_attention_mask=torch.tensor([1, 0, 1, 0]),
    )
    with pytest.raises(ValueError, match="right-padded"):
        pack_qwen3_audio_features(
            [item],
            num_mel_bins=8,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_asr_wrapper_maps_official_thinker_weight_names():
    from tokenspeed.runtime.models import qwen3_asr

    assert Qwen3AudioEncoder.map_weight_name(
        "thinker.audio_tower.layers.0.self_attn.q_proj.weight"
    ) == ("layers.0.self_attn.qkv_proj.weight", "q")
    assert qwen3_asr.Qwen3ASRForConditionalGeneration.map_language_weight_name(
        "thinker.model.layers.0.self_attn.k_proj.weight"
    ) == ("language_model.model.layers.0.self_attn.qkv_proj.weight", "k")
    assert qwen3_asr.Qwen3ASRForConditionalGeneration.map_language_weight_name(
        "thinker.model.layers.0.mlp.gate_proj.weight"
    ) == ("language_model.model.layers.0.mlp.gate_up_proj.weight", 0)
    assert qwen3_asr.Qwen3ASRForConditionalGeneration.map_language_weight_name(
        "thinker.lm_head.weight"
    ) == ("language_model.lm_head.weight", None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="model loader needs CUDA")
def test_tiny_asr_wrapper_loads_audio_and_language_roots():
    from tokenspeed.runtime.configs.qwen3_asr_config import (
        Qwen3ASRAudioEncoderConfig,
        Qwen3ASRConfig,
        Qwen3ASRThinkerConfig,
    )
    from tokenspeed.runtime.configs.qwen3_config import Qwen3Config
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.models.qwen3_asr import Qwen3ASRForConditionalGeneration

    audio_config = Qwen3ASRAudioEncoderConfig(**vars(_tiny_config()))
    text_config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    config = Qwen3ASRConfig(
        architectures=["Qwen3ASRForConditionalGeneration"],
        thinker_config=Qwen3ASRThinkerConfig(
            audio_config=audio_config,
            text_config=text_config,
        ),
    )
    with torch.device("cuda"):
        model = Qwen3ASRForConditionalGeneration(
            config,
            Mapping(rank=0, world_size=1),
            mm_attention_backend="triton_attn",
        ).eval()

    qkv = model.language_model.model.layers[0].self_attn.qkv_proj.weight
    q_weight = torch.randn(32, 32, device="cuda", dtype=qkv.dtype)
    audio_bias = torch.randn_like(model.audio_tower.conv2d1.bias)
    lm_head = torch.randn_like(model.language_model.lm_head.weight)
    loaded = model.load_weights(
        [
            ("thinker.audio_tower.conv2d1.bias", audio_bias),
            ("thinker.model.layers.0.self_attn.q_proj.weight", q_weight),
            ("thinker.lm_head.weight", lm_head),
        ]
    )

    torch.testing.assert_close(model.audio_tower.conv2d1.bias, audio_bias)
    torch.testing.assert_close(qkv[:32], q_weight)
    torch.testing.assert_close(model.language_model.lm_head.weight, lm_head)
    assert loaded == {
        "audio_tower.conv2d1.bias",
        "language_model.model.layers.0.self_attn.qkv_proj.weight",
        "language_model.lm_head.weight",
    }


def test_asr_wrapper_rejects_audio_text_dimension_mismatch():
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.models.qwen3_asr import Qwen3ASRForConditionalGeneration

    config = _tiny_asr_config()
    config.thinker_config.audio_config.output_dim = 8
    with pytest.raises(ValueError, match="output_dim"):
        Qwen3ASRForConditionalGeneration(config, Mapping(rank=0, world_size=1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="audio tower needs CUDA")
def test_tiny_audio_tower_matches_transformers_and_loads_fused_qkv():
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoderConfig,
    )
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoder as HFAudioEncoder,
    )

    from tokenspeed.runtime.distributed.mapping import Mapping

    config = Qwen3OmniMoeAudioEncoderConfig(**vars(_tiny_config()))
    mapping = Mapping(rank=0, world_size=1)
    torch.manual_seed(4)
    reference = HFAudioEncoder(config).cuda().eval()
    with torch.device("cuda"):
        tower = Qwen3AudioEncoder(
            config,
            mapping,
            mm_attention_backend="triton_attn",
        ).eval()
    loaded = tower.load_weights(reference.state_dict().items())
    assert loaded == set(dict(tower.named_parameters()))

    items = [
        _audio_item(torch.randn(8, 8)),
        _audio_item(torch.randn(1, 8, 10), audio_feature_lengths=9),
    ]
    with torch.no_grad():
        output = tower.encode(items)
        packed, feature_lengths = pack_qwen3_audio_features(
            items,
            num_mel_bins=config.num_mel_bins,
            device=tower.device,
            dtype=tower.dtype,
        )
        expected = reference(
            input_features=packed,
            feature_lens=feature_lengths,
        ).last_hidden_state
    expected_tokens = qwen3_audio_output_lengths([8, 9], n_window=4).sum().item()
    assert output.shape == (expected_tokens, config.output_dim)
    torch.testing.assert_close(output, expected, atol=2e-4, rtol=2e-4)

    qkv = tower.layers[0].self_attn.qkv_proj.weight
    q = torch.randn(config.d_model, config.d_model, device="cuda", dtype=qkv.dtype)
    loaded_name = tower.load_weight("layers.0.self_attn.q_proj.weight", q)
    assert loaded_name == "layers.0.self_attn.qkv_proj.weight"
    torch.testing.assert_close(qkv[: config.d_model], q)
