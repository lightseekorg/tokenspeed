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

"""CPU tests for checkpoint values, independent draft settings, and loading."""

from pathlib import Path

import pytest
import torch
from transformers import AutoConfig

from tokenspeed.runtime.configs.inkling2 import (
    Inkling2MMConfig,
    Inkling2ModelConfig,
    Inkling2MTPConfig,
)
from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.execution.input_buffer import InputBuffers
from tokenspeed.runtime.execution.multimodal_runtime import MultimodalRuntime
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalDataItem,
    substitute_mm_pad_,
)
from tokenspeed.runtime.utils.hf_transformers_utils import get_config
from tokenspeed.runtime.utils.server_args import ServerArgs
from tokenspeed.runtime.utils.spec_block_geometry import (
    read_checkpoint_block_size,
    resolve_block_widths,
)


def _checkpoint() -> dict:
    return dict(
        model_type="inkling2",
        architectures=["Inkling2ForConditionalGeneration"],
        eos_token_id=200006,
        text_config=dict(
            model_max_length=1048576,
            torch_dtype="bfloat16",
            hidden_size=4608,
            num_hidden_layers=48,
            vocab_size=201024,
            num_attention_heads=48,
            num_key_value_heads=8,
            head_dim=128,
            d_rel=16,
            rel_extent=1024,
            q_bias=False,
            o_bias=False,
            log_scaling_n_floor=128000,
            log_scaling_alpha=0.1,
            rms_norm_eps=1e-6,
            use_value_norm_no_gamma=True,
            qk_l2_clip_threshold=22.627416997969522,
            rel_l2_clip_threshold=8.0,
            local_layer_ids=[i for i in range(48) if i % 6 != 5],
            sconv_kernel_size=4,
            unpadded_vocab_size=200058,
            final_logit_softcapping=30.0,
            swa_head_dim=128,
            swa_num_attention_heads=48,
            swa_num_key_value_heads=8,
            sliding_window_size=1024,
            n_layers_per_attn_res_block=6,
            attn_res_normalize_values=True,
            n_routed_experts=384,
            num_experts_per_tok=8,
            n_shared_experts=0,
            intermediate_size=2304,
            mlp_gate_softcap=4.0,
            mlp_up_softcap=25.0,
            route_scale=8.0,
            use_gate_bias=True,
        ),
        vision_config=dict(
            temporal_patch_size=2,
            native_patch_size=20,
            adapter_merge_size=2,
            projection_hidden_dim=4096,
            projector_bias=False,
            hidden_size=1024,
            num_hidden_layers=23,
            num_attention_heads=16,
            mlp_ratio=4.0,
            posemb_grid_size=32,
            layer_norm_eps=1e-5,
        ),
        mtp_config=dict(
            hidden_size=4608,
            num_hidden_layers=5,
            num_attention_heads=48,
            num_key_value_heads=8,
            head_dim=128,
            intermediate_size=18432,
            max_position_embeddings=262144,
            torch_dtype="bfloat16",
            q_bias=False,
            o_bias=False,
            rms_norm_eps=1e-6,
            rope_theta=1000000.0,
            local_layer_ids=[0, 1, 3, 4],
            sliding_window_size=1024,
            block_size=8,
            markov_rank=256,
            markov_head_type="rnn_hidden",
            target_layer_ids=[5, 11, 17, 23, 29, 35, 41, 48],
        ),
    )


def test_checkpoint_values_and_independent_draft() -> None:
    source = _checkpoint()
    config = Inkling2MMConfig(**source)
    text = config.get_text_config()
    draft = Inkling2MTPConfig.from_target(config)
    assert text.n_routed_experts == 384
    assert text.num_experts_per_tok == 8
    assert text.n_shared_experts == 0
    assert text.final_logit_softcapping == 30.0
    assert text.use_value_norm_no_gamma
    assert text.attn_res_normalize_values
    assert (text.mlp_gate_softcap, text.mlp_up_softcap) == (4.0, 25.0)
    assert text.max_position_embeddings == 1048576
    assert text.vocab_size == 200058
    assert text.padded_vocab_size == 201024
    assert text.eos_token_id == config.eos_token_id == 200006
    assert text.num_nextn_predict_layers == 0
    assert text.num_hidden_layers == 48
    assert draft.num_hidden_layers == 5
    assert draft.intermediate_size == 18432
    assert draft.max_position_embeddings == 262144
    assert draft.local_layer_ids == [0, 1, 3, 4]
    assert draft.dspark_block_size == 7
    assert draft.dspark_target_layer_ids[-1] == 48
    assert draft.dflash_config["target_layer_ids"] == [5, 11, 17, 23, 29, 35, 41]
    assert config.mtp_config == source["mtp_config"]
    assert config.vision_config.native_patch_size == 20
    assert config.vision_config.posemb_grid_size == 32
    assert text.cache_layer_types.count("full_attention") == 8
    assert len(text.cache_layer_types) == text.num_hidden_layers
    assert text.sliding_window == 1024


@pytest.mark.parametrize("block_size", [2, 8, 16])
def test_draft_config_roundtrip(tmp_path: Path, block_size: int) -> None:
    source = _checkpoint()
    source["mtp_config"]["block_size"] = block_size
    draft = Inkling2MTPConfig.from_target(Inkling2MMConfig(**source))
    draft.save_pretrained(tmp_path)
    restored = AutoConfig.from_pretrained(tmp_path, trust_remote_code=False)
    assert restored.target_layer_ids == draft.target_layer_ids
    assert resolve_block_widths("DSPARK", read_checkpoint_block_size(restored)) == (
        block_size - 1,
        block_size,
    )
    assert restored.architectures == ["Inkling2ForConditionalGenerationMTP"]


def test_config_roundtrip_and_native_registration(tmp_path: Path) -> None:
    config = Inkling2MMConfig(**_checkpoint())
    config.save_pretrained(tmp_path)
    restored = AutoConfig.from_pretrained(tmp_path, trust_remote_code=False)
    restored_values = restored.to_dict()
    original_values = config.to_dict()
    restored_values.pop("_name_or_path")
    original_values.pop("_name_or_path")
    assert restored_values == original_values
    loaded = get_config(str(tmp_path), trust_remote_code=False, revision=None)
    assert loaded.architectures == ["Inkling2ForConditionalGeneration"]
    assert loaded.text_config.use_value_norm_no_gamma is True
    assert loaded.mtp_config["num_hidden_layers"] == 5


def test_draft_rejects_quantization_metadata() -> None:
    values = _checkpoint()
    values["mtp_config"]["quantization_config"] = {"quant_method": "nvfp4"}
    with pytest.raises(ValueError, match="unquantized BF16"):
        Inkling2MTPConfig.from_target(Inkling2MMConfig(**values))


def test_multimodal_draft_restores_native_image_payload_token(tmp_path: Path) -> None:
    Inkling2MMConfig(**_checkpoint()).save_pretrained(tmp_path)
    args = ServerArgs(model=str(tmp_path), device="cpu", max_model_len=2048)
    metadata = ModelConfig(
        str(tmp_path),
        trust_remote_code=False,
        model_override_args="{}",
        dtype="bfloat16",
        server_args=args,
        context_length=2048,
    )
    buffers = InputBuffers(1, 16, 2, device="cpu")
    MultimodalRuntime.wire_drafter(buffers, metadata)
    image = MultimodalDataItem(modality=Modality.IMAGE, hash=123)
    image.set_pad_value()
    tokens = torch.tensor([200005, image.pad_value, 200054, 7], dtype=torch.int32)
    substitute_mm_pad_(tokens, buffers.mm_pad_substitute_ids)
    assert tokens.tolist() == [200005, 200005, 200054, 7]


@pytest.mark.parametrize("value_norm", [False, True])
def test_optional_norm_settings(value_norm: bool) -> None:
    values = _checkpoint()["text_config"]
    values.update(use_value_norm_no_gamma=value_norm)
    text = Inkling2ModelConfig(**values)
    assert text.use_value_norm_no_gamma == value_norm


def test_draft_loading_selects_native_config_without_mutating_target(
    tmp_path: Path,
) -> None:
    Inkling2MMConfig(**_checkpoint()).save_pretrained(tmp_path)
    before = (tmp_path / "config.json").read_bytes()
    draft = get_config(
        str(tmp_path),
        trust_remote_code=False,
        revision=None,
        is_draft_worker=True,
        speculative_algorithm="DSPARK",
    )
    assert isinstance(draft, Inkling2MTPConfig)
    assert draft.architectures == ["Inkling2ForConditionalGenerationMTP"]
    assert draft.num_hidden_layers == 5
    assert draft.max_position_embeddings == 262144
    assert draft.target_layer_ids == [5, 11, 17, 23, 29, 35, 41, 48]
    assert draft._name_or_path == str(tmp_path)
    assert (tmp_path / "config.json").read_bytes() == before
    target = get_config(str(tmp_path), trust_remote_code=False, revision=None)
    assert target.text_config.num_hidden_layers == 48
    assert target.text_config.max_position_embeddings == 1048576


@pytest.mark.parametrize("algorithm", [None, "EAGLE3", "MTP", "DFLASH"])
@pytest.mark.parametrize("standalone", [False, True])
def test_draft_loading_rejects_old_nextn_path(
    tmp_path: Path, algorithm: str | None, standalone: bool
) -> None:
    config = Inkling2MMConfig(**_checkpoint())
    if standalone:
        config = Inkling2MTPConfig.from_target(config)
    config.save_pretrained(tmp_path)
    with pytest.raises(NotImplementedError, match="separate DSpark MTP"):
        get_config(
            str(tmp_path),
            trust_remote_code=False,
            revision=None,
            is_draft_worker=True,
            speculative_algorithm=algorithm,
        )


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"num_experts_per_tok": 385}, "routed expert count"),
        ({"local_layer_ids": [0, 0]}, "duplicates"),
        ({"local_layer_ids": [48]}, "out-of-range"),
        ({"n_layers_per_attn_res_block": 0}, "block size"),
        ({"mlp_gate_softcap": -1.0}, "nonnegative"),
        ({"qk_l2_clip_threshold": float("nan")}, "finite and positive"),
        ({"gate_activation": "softmax"}, "requires gate_activation"),
        ({"num_nextn_predict_layers": 5}, "requires num_nextn_predict_layers"),
        ({"max_position_embeddings": 4096}, "must match model_max_length"),
        ({"v_head_dim": 64}, "v_head_dim must match head_dim"),
        ({"swa_v_head_dim": 64}, "swa_v_head_dim must match swa_head_dim"),
    ],
)
def test_invalid_decoder_settings(changes: dict, message: str) -> None:
    values = _checkpoint()["text_config"]
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        Inkling2ModelConfig(**values)


def test_missing_routing_field_has_no_reference_fallback() -> None:
    values = _checkpoint()["text_config"]
    del values["n_routed_experts"]
    with pytest.raises(ValueError, match="missing.*n_routed_experts"):
        Inkling2ModelConfig(**values)


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"hidden_size": 4096}, "hidden sizes must match"),
        ({"markov_head_type": "vanilla"}, "rnn_hidden"),
        ({"torch_dtype": "float16"}, "bfloat16"),
        ({"target_layer_ids": [5, 11, 47]}, "final-norm sentinel"),
        ({"target_layer_ids": [11, 5, 48]}, "sorted unique"),
        ({"local_layer_ids": [0, 5]}, "valid layer indices"),
        ({"block_size": 1}, "block_size >= 2"),
    ],
)
def test_invalid_draft_settings(changes: dict, message: str) -> None:
    values = _checkpoint()
    values["mtp_config"].update(changes)
    with pytest.raises(ValueError, match=message):
        Inkling2MTPConfig.from_target(Inkling2MMConfig(**values))


def test_absent_draft_does_not_enable_nextn() -> None:
    values = _checkpoint()
    values["mtp_config"] = None
    config = Inkling2MMConfig(**values)
    assert config.text_config.num_nextn_predict_layers == 0
    with pytest.raises(ValueError, match="requires mtp_config"):
        Inkling2MTPConfig.from_target(config)


def test_draft_precision_must_be_explicit() -> None:
    values = _checkpoint()["mtp_config"]
    del values["torch_dtype"]
    with pytest.raises(ValueError, match="explicit dtype or torch_dtype"):
        Inkling2MTPConfig(**values)


def test_draft_precision_accepts_dtype_metadata() -> None:
    values = _checkpoint()["mtp_config"]
    values["dtype"] = values.pop("torch_dtype")
    draft = Inkling2MTPConfig(**values)
    assert str(draft.dtype) == "torch.bfloat16"
