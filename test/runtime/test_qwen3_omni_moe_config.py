# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Unit tests for Qwen3-Omni MoE configuration classes."""

import json

from tokenspeed.runtime.configs import get_config_class
from tokenspeed.runtime.configs.qwen3_omni_moe_config import (
    Qwen3OmniMoeAudioEncoderConfig,
    Qwen3OmniMoeCode2WavConfig,
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeTalkerCodePredictorConfig,
    Qwen3OmniMoeTalkerConfig,
    Qwen3OmniMoeTalkerTextConfig,
    Qwen3OmniMoeTextConfig,
    Qwen3OmniMoeThinkerConfig,
    Qwen3OmniMoeVisionEncoderConfig,
)
from tokenspeed.runtime.configs.utils import get_config

# ── Top-level Qwen3OmniMoeConfig ───────────────────────────────────────


def test_qwen3_omni_moe_config_is_registered() -> None:
    assert get_config_class("qwen3_omni_moe") is Qwen3OmniMoeConfig


def test_qwen3_omni_moe_config_default_construction() -> None:
    config = Qwen3OmniMoeConfig()

    assert config.model_type == "qwen3_omni_moe"
    assert isinstance(config.thinker_config, Qwen3OmniMoeThinkerConfig)
    assert isinstance(config.talker_config, Qwen3OmniMoeTalkerConfig)
    assert isinstance(config.code2wav_config, Qwen3OmniMoeCode2WavConfig)
    assert config.enable_audio_output is True
    assert config.im_start_token_id == 151644
    assert config.im_end_token_id == 151645
    assert config.tts_pad_token_id == 151671
    assert config.tts_bos_token_id == 151672
    assert config.tts_eos_token_id == 151673
    assert config.system_token_id == 8948
    assert config.user_token_id == 872
    assert config.assistant_token_id == 77091


def test_qwen3_omni_moe_config_sub_configs_from_dict() -> None:
    config = Qwen3OmniMoeConfig(
        thinker_config={
            "text_config": {"hidden_size": 512, "num_attention_heads": 4},
        },
        talker_config={
            "text_config": {"hidden_size": 256, "num_attention_heads": 8},
            "code_predictor_config": {"num_code_groups": 64},
        },
        code2wav_config={"codebook_size": 1024},
    )

    assert isinstance(config.thinker_config, Qwen3OmniMoeThinkerConfig)
    assert isinstance(config.talker_config, Qwen3OmniMoeTalkerConfig)
    assert isinstance(config.code2wav_config, Qwen3OmniMoeCode2WavConfig)
    assert config.thinker_config.text_config.hidden_size == 512
    assert config.talker_config.text_config.hidden_size == 256
    assert config.talker_config.code_predictor_config.num_code_groups == 64
    assert config.code2wav_config.codebook_size == 1024


def test_qwen3_omni_moe_config_sub_configs_from_constructed_objects() -> None:
    thinker = Qwen3OmniMoeThinkerConfig()
    talker = Qwen3OmniMoeTalkerConfig()
    code2wav = Qwen3OmniMoeCode2WavConfig()

    config = Qwen3OmniMoeConfig(
        thinker_config=thinker,
        talker_config=talker,
        code2wav_config=code2wav,
    )

    assert config.thinker_config is thinker
    assert config.talker_config is talker
    assert config.code2wav_config is code2wav


def test_qwen3_omni_moe_config_initializer_range_falls_back_to_thinker() -> None:
    config = Qwen3OmniMoeConfig()
    assert config.initializer_range == config.thinker_config.initializer_range


def test_qwen3_omni_moe_config_explicit_initializer_range_overrides_thinker() -> None:
    config = Qwen3OmniMoeConfig(initializer_range=0.05)
    assert config.initializer_range == 0.05


def test_qwen3_omni_moe_config_get_text_config() -> None:
    config = Qwen3OmniMoeConfig()
    text = config.get_text_config()
    assert isinstance(text, Qwen3OmniMoeTextConfig)
    assert text is config.thinker_config.text_config


# ── Qwen3OmniMoeThinkerConfig ──────────────────────────────────────────


def test_thinker_config_default_construction() -> None:
    config = Qwen3OmniMoeThinkerConfig()

    assert isinstance(config.audio_config, Qwen3OmniMoeAudioEncoderConfig)
    assert isinstance(config.vision_config, Qwen3OmniMoeVisionEncoderConfig)
    assert isinstance(config.text_config, Qwen3OmniMoeTextConfig)
    assert config.position_id_per_seconds == 25
    assert config.audio_start_token_id == 151647
    assert config.user_token_id == 872
    assert config.initializer_range == 0.02


def test_thinker_config_sub_configs_from_dict() -> None:
    config = Qwen3OmniMoeThinkerConfig(
        audio_config={"num_mel_bins": 64},
        vision_config={"depth": 12},
        text_config={"hidden_size": 1024},
    )

    assert isinstance(config.audio_config, Qwen3OmniMoeAudioEncoderConfig)
    assert isinstance(config.vision_config, Qwen3OmniMoeVisionEncoderConfig)
    assert isinstance(config.text_config, Qwen3OmniMoeTextConfig)
    assert config.audio_config.num_mel_bins == 64
    assert config.vision_config.depth == 12
    assert config.text_config.hidden_size == 1024


def test_thinker_config_sub_configs_from_constructed_objects() -> None:
    audio = Qwen3OmniMoeAudioEncoderConfig(num_mel_bins=32)
    vision = Qwen3OmniMoeVisionEncoderConfig(depth=8)
    text = Qwen3OmniMoeTextConfig(hidden_size=512)

    config = Qwen3OmniMoeThinkerConfig(
        audio_config=audio,
        vision_config=vision,
        text_config=text,
    )

    assert config.audio_config is audio
    assert config.vision_config is vision
    assert config.text_config is text


# ── Qwen3OmniMoeAudioEncoderConfig ─────────────────────────────────────


def test_audio_encoder_config_defaults() -> None:
    config = Qwen3OmniMoeAudioEncoderConfig()

    assert config.model_type == "qwen3_omni_moe_audio_encoder"
    assert config.num_mel_bins == 128
    assert config.encoder_layers == 32
    assert config.encoder_attention_heads == 20
    assert config.encoder_ffn_dim == 5120
    assert config.d_model == 1280
    assert config.dropout == 0.0
    assert config.attention_dropout == 0.0
    assert config.activation_function == "gelu"
    assert config.max_source_positions == 1500
    assert config.n_window == 50
    assert config.output_dim == 3584
    assert config.n_window_infer == 800
    assert config.conv_chunksize == 500
    assert config.downsample_hidden_size == 480


def test_audio_encoder_config_attribute_map() -> None:
    config = Qwen3OmniMoeAudioEncoderConfig.from_dict(
        {
            "num_hidden_layers": 24,
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
        }
    )
    assert config.encoder_layers == 24
    assert config.d_model == 1024
    assert config.encoder_attention_heads == 16
    assert config.encoder_ffn_dim == 4096

    d = config.to_dict()
    assert d["encoder_layers"] == 24
    assert d["d_model"] == 1024
    assert d["encoder_attention_heads"] == 16
    assert d["encoder_ffn_dim"] == 4096


# ── Qwen3OmniMoeVisionEncoderConfig ────────────────────────────────────


def test_vision_encoder_config_defaults() -> None:
    config = Qwen3OmniMoeVisionEncoderConfig()

    assert config.model_type == "qwen3_omni_moe_vision_encoder"
    assert config.base_config_key == "vision_config"
    assert config.depth == 27
    assert config.hidden_size == 1152
    assert config.hidden_act == "gelu_pytorch_tanh"
    assert config.intermediate_size == 4304
    assert config.num_heads == 16
    assert config.in_channels == 3
    assert config.patch_size == 16
    assert config.spatial_merge_size == 2
    assert config.temporal_patch_size == 2
    assert config.out_hidden_size == 3584
    assert config.num_position_embeddings == 2304
    assert config.deepstack_visual_indexes == (8, 16, 24)


# ── Qwen3OmniMoeTextConfig ─────────────────────────────────────────────


def test_text_config_defaults() -> None:
    config = Qwen3OmniMoeTextConfig()

    assert config.model_type == "qwen3_omni_moe_text"
    assert config.vocab_size == 3584
    assert config.hidden_size == 2048
    assert config.intermediate_size == 18944
    assert config.num_hidden_layers == 28
    assert config.num_attention_heads == 28
    assert config.num_key_value_heads == 4
    assert config.hidden_act == "silu"
    assert config.max_position_embeddings == 32768
    assert config.rms_norm_eps == 1e-6
    assert config.default_theta == 1000000.0
    assert config.decoder_sparse_step == 1
    assert config.moe_intermediate_size == 768
    assert config.num_experts_per_tok == 8
    assert config.num_experts == 128
    assert config.norm_topk_prob is True
    assert config.output_router_logits is False
    assert config.router_aux_loss_coef == 0.001


def test_text_config_mlp_only_layers_defaults_to_empty_list() -> None:
    config = Qwen3OmniMoeTextConfig()
    assert config.mlp_only_layers == []


def test_text_config_mlp_only_layers_explicit() -> None:
    config = Qwen3OmniMoeTextConfig(mlp_only_layers=[0, 2, 4])
    assert config.mlp_only_layers == [0, 2, 4]


def test_text_config_ignore_keys_at_rope_validation() -> None:
    """MRoPE extension keys are ignored during rope validation."""
    config = Qwen3OmniMoeTextConfig()
    assert "mrope_section" in config.ignore_keys_at_rope_validation
    assert "interleaved" in config.ignore_keys_at_rope_validation
    assert "mrope_interleaved" in config.ignore_keys_at_rope_validation


# ── Qwen3OmniMoeTalkerConfig ───────────────────────────────────────────


def test_talker_config_default_construction() -> None:
    config = Qwen3OmniMoeTalkerConfig()

    assert isinstance(
        config.code_predictor_config, Qwen3OmniMoeTalkerCodePredictorConfig
    )
    assert isinstance(config.text_config, Qwen3OmniMoeTalkerTextConfig)
    assert config.num_code_groups == 32
    assert config.thinker_hidden_size == 2048
    assert config.codec_eos_token_id == 4198
    assert config.accept_hidden_layer == 18
    assert config.codec_nothink_id == 4203
    assert config.codec_think_bos_id == 4204
    assert config.codec_think_eos_id == 4205
    assert config.codec_pad_id == 4196
    assert config.codec_bos_id == 4197


def test_talker_config_sub_configs_from_dict() -> None:
    config = Qwen3OmniMoeTalkerConfig(
        code_predictor_config={"num_code_groups": 16},
        text_config={"hidden_size": 512},
    )

    assert isinstance(
        config.code_predictor_config, Qwen3OmniMoeTalkerCodePredictorConfig
    )
    assert isinstance(config.text_config, Qwen3OmniMoeTalkerTextConfig)
    assert config.code_predictor_config.num_code_groups == 16
    assert config.text_config.hidden_size == 512


def test_talker_config_sub_configs_from_constructed_objects() -> None:
    cp = Qwen3OmniMoeTalkerCodePredictorConfig(num_code_groups=8)
    text = Qwen3OmniMoeTalkerTextConfig(hidden_size=256)

    config = Qwen3OmniMoeTalkerConfig(
        code_predictor_config=cp,
        text_config=text,
    )

    assert config.code_predictor_config is cp
    assert config.text_config is text


# ── Qwen3OmniMoeTalkerCodePredictorConfig ──────────────────────────────


def test_talker_code_predictor_config_defaults() -> None:
    config = Qwen3OmniMoeTalkerCodePredictorConfig()

    assert config.model_type == "qwen3_omni_moe_talker_code_predictor"
    assert config.vocab_size == 2048
    assert config.hidden_size == 1024
    assert config.intermediate_size == 3072
    assert config.num_hidden_layers == 5
    assert config.num_attention_heads == 16
    assert config.num_key_value_heads == 8
    assert config.head_dim == 128
    assert config.num_code_groups == 32


def test_talker_code_predictor_num_key_value_heads_defaults_to_num_attention_heads() -> (
    None
):
    config = Qwen3OmniMoeTalkerCodePredictorConfig(
        num_key_value_heads=None, num_attention_heads=32
    )
    assert config.num_key_value_heads == 32


def test_talker_code_predictor_layer_types_default_all_full_attention() -> None:
    """Without ``sliding_window``, all layers are ``full_attention``."""
    config = Qwen3OmniMoeTalkerCodePredictorConfig(
        num_hidden_layers=4, sliding_window=None
    )
    assert config.layer_types == [
        "full_attention",
        "full_attention",
        "full_attention",
        "full_attention",
    ]


def test_talker_code_predictor_layer_types_with_sliding_window() -> None:
    """Layers at or past ``max_window_layers`` use ``sliding_attention``."""
    config = Qwen3OmniMoeTalkerCodePredictorConfig(
        num_hidden_layers=6,
        sliding_window=32,
        max_window_layers=4,
    )
    assert config.layer_types == [
        "full_attention",  # layer 0
        "full_attention",  # layer 1
        "full_attention",  # layer 2
        "full_attention",  # layer 3
        "sliding_attention",  # layer 4 >= max_window_layers
        "sliding_attention",  # layer 5
    ]


def test_talker_code_predictor_layer_types_explicit_override() -> None:
    explicit = ["sliding_attention"] * 3
    config = Qwen3OmniMoeTalkerCodePredictorConfig(
        num_hidden_layers=3, layer_types=explicit
    )
    assert config.layer_types is explicit


# ── Qwen3OmniMoeTalkerTextConfig ───────────────────────────────────────


def test_talker_text_config_defaults() -> None:
    config = Qwen3OmniMoeTalkerTextConfig()

    assert config.model_type == "qwen3_omni_moe_talker_text"
    assert config.vocab_size == 3072
    assert config.hidden_size == 1024
    assert config.intermediate_size == 2048
    assert config.num_hidden_layers == 20
    assert config.num_attention_heads == 16
    assert config.num_key_value_heads == 2
    assert config.hidden_act == "silu"
    assert config.decoder_sparse_step == 1
    assert config.moe_intermediate_size == 384
    assert config.num_experts_per_tok == 8
    assert config.num_local_experts == 128
    assert config.norm_topk_prob is False


def test_talker_text_config_attribute_map() -> None:
    config = Qwen3OmniMoeTalkerTextConfig.from_dict({"num_experts": 64})
    assert config.num_local_experts == 64

    d = config.to_dict()
    assert d["num_local_experts"] == 64


def test_talker_text_config_mlp_only_layers_defaults_to_empty_list() -> None:
    config = Qwen3OmniMoeTalkerTextConfig()
    assert config.mlp_only_layers == []


def test_talker_text_config_mlp_only_layers_explicit() -> None:
    config = Qwen3OmniMoeTalkerTextConfig(mlp_only_layers=[1, 3, 5])
    assert config.mlp_only_layers == [1, 3, 5]


# ── Qwen3OmniMoeCode2WavConfig ─────────────────────────────────────────


def test_code2wav_config_defaults() -> None:
    config = Qwen3OmniMoeCode2WavConfig()

    assert config.codebook_size == 2048
    assert config.hidden_size == 1024
    assert config.max_position_embeddings == 8000
    assert config.num_attention_heads == 16
    assert config.num_key_value_heads == 16
    assert config.attention_bias is False
    assert config.sliding_window == 72
    assert config.intermediate_size == 3072
    assert config.hidden_act == "silu"
    assert config.layer_scale_initial_scale == 0.01
    assert config.rms_norm_eps == 1e-5
    assert config.num_hidden_layers == 8
    assert config.num_quantizers == 16
    assert config.upsample_rates == (8, 5, 4, 3)
    assert config.upsampling_ratios == (2, 2)
    assert config.decoder_dim == 1536
    assert config.attention_dropout == 0.0
    assert config.initializer_range == 0.02


def test_code2wav_config_layer_types_property_all_sliding() -> None:
    config = Qwen3OmniMoeCode2WavConfig(num_hidden_layers=8)
    assert config.layer_types == ["sliding_attention"] * 8


def test_code2wav_config_layer_types_scales_with_num_hidden_layers() -> None:
    config = Qwen3OmniMoeCode2WavConfig(num_hidden_layers=3)
    assert config.layer_types == [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
    ]


# ── get_config integration ─────────────────────────────────────────────


def test_registered_qwen3_omni_moe_config_loads_from_raw_json(tmp_path) -> None:
    raw_config = {
        "model_type": "qwen3_omni_moe",
        "architectures": ["Qwen3OmniMoeForConditionalGeneration"],
        "thinker_config": {
            "text_config": {
                "hidden_size": 512,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
            },
        },
        "talker_config": {
            "text_config": {
                "hidden_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
            },
        },
        "code2wav_config": {"num_hidden_layers": 4},
        "checkpoint_extension": "preserved",
    }
    (tmp_path / "config.json").write_text(json.dumps(raw_config), encoding="utf-8")

    config = get_config(str(tmp_path))

    assert isinstance(config, Qwen3OmniMoeConfig)
    assert isinstance(config.thinker_config, Qwen3OmniMoeThinkerConfig)
    assert isinstance(config.talker_config, Qwen3OmniMoeTalkerConfig)
    assert isinstance(config.code2wav_config, Qwen3OmniMoeCode2WavConfig)
    assert config.thinker_config.text_config.hidden_size == 512
    assert config.talker_config.text_config.hidden_size == 256
    assert config.code2wav_config.num_hidden_layers == 4
    assert config.checkpoint_extension == "preserved"
    assert config.name_or_path == str(tmp_path)


def test_qwen3_omni_moe_num_experts_override_targets_thinker_text(tmp_path) -> None:
    """``num_experts`` must reach the thinker text config, not be rewritten to
    the talker's ``num_local_experts`` alias."""
    raw_config = {
        "model_type": "qwen3_omni_moe",
        "architectures": ["Qwen3OmniMoeForConditionalGeneration"],
        "thinker_config": {
            "text_config": {
                "hidden_size": 512,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "num_experts": 128,
            },
        },
        "talker_config": {
            "text_config": {
                "hidden_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
            },
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(raw_config), encoding="utf-8")

    config = get_config(str(tmp_path), model_override_args={"num_experts": 256})

    assert config.thinker_config.text_config.num_experts == 256
    assert not hasattr(config.thinker_config.text_config, "num_local_experts")


def test_qwen3_omni_moe_num_local_experts_override_preserves_thinker_num_experts(
    tmp_path,
) -> None:
    """A talker-only ``num_local_experts`` override must not delete the
    checkpoint's thinker ``num_experts``."""
    raw_config = {
        "model_type": "qwen3_omni_moe",
        "architectures": ["Qwen3OmniMoeForConditionalGeneration"],
        "thinker_config": {
            "text_config": {
                "hidden_size": 512,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "num_experts": 64,
            },
        },
        "talker_config": {
            "text_config": {
                "hidden_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
            },
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(raw_config), encoding="utf-8")

    config = get_config(str(tmp_path), model_override_args={"num_local_experts": 256})

    assert config.thinker_config.text_config.num_experts == 64
