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

"""``--decode-monokernel`` at launch: a model without decode monokernels fails
startup instead of serving without them, and a model with them reads the
switch from its config. CPU only."""

from __future__ import annotations

import json
import os
import sys

import pytest

# The CI runner executes each registered test file as a standalone script.
_TEST_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))
from test.runtime.models.conftest import model_config, server_args  # noqa: E402

from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")


def test_the_switch_fails_a_model_without_monokernels(checkpoints) -> None:
    with pytest.raises(ValueError, match="no decode monokernel for.*Qwen3ForCausalLM"):
        server_args(checkpoints["other"], decode_monokernel=True)


def test_without_the_switch_the_model_launches(checkpoints) -> None:
    assert not server_args(checkpoints["other"]).decode_monokernel


@pytest.mark.parametrize("switch", [False, True])
def test_the_switch_reaches_the_model_config(tmp_path, switch) -> None:
    """A Qwen4-Exp model builds its monokernel layers when its text config
    says so: ModelConfig sets it from the switch."""
    config = {
        "architectures": ["Qwen4ExpForCausalLM"],
        "model_type": "qwen4_exp_text",
        "vocab_size": 128,
        "hidden_size": 128,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "max_position_embeddings": 2048,
        "torch_dtype": "bfloat16",
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    args = server_args(tmp_path, decode_monokernel=switch)
    assert model_config(args).hf_text_config.decode_monokernel is switch


@pytest.mark.parametrize("switch", [False, True])
def test_the_switch_reaches_the_muse_spark_config(tmp_path, switch) -> None:
    """Muse Spark's config is its own text config (no nested text config
    object), and ModelConfig sets the switch on it."""
    layer_types = ["sliding_attention", "full_attention"]
    config = {
        "architectures": ["MuseSparkForConditionalGeneration"],
        "model_type": "muse_spark",
        "torch_dtype": "bfloat16",
        "text_config": {
            "vocab_size": 128,
            "hidden_size": 128,
            "head_dim": 64,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_hidden_layers": len(layer_types),
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-5,
            "post_norm_eps": 1e-8,
            "final_logit_softcapping": 20.0,
            "qk_scale_factor": 1.0,
            "output_multiplier": 1.0,
            "residual_gate_temperature": 0.3,
            "sliding_window": 64,
            "layer_types": layer_types,
            "layer_rope_theta": [500000.0, 0.0],
            "mlp_layer_types": ["sparse", "sparse"],
            "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
            "moe_hidden_size": 128,
            "moe_intermediate_size": 128,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "num_shared_experts": 0,
            "routed_scaling_factor": 1.0,
        },
        "vision_config": {
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "intermediate_size": 32,
            "layer_types": ["full_attention"],
            "layer_norm_eps": 1e-5,
            "merge_size": 2,
            "patch_size": 2,
            "patch_temporal": 2,
            "pos_emb_height": 2,
            "pos_emb_width": 2,
            "max_position_embeddings": 4,
            "rope_parameters": {"rope_theta": 10000, "rope_type": "axial"},
            "hidden_act": "gelu",
        },
        "audio_config": {},
        "image_token_id": 120,
        "video_token_id": 121,
        "audio_token_id": 122,
        "out_hidden_size": 128,
        "projector_hidden_size": 64,
        "projector_hidden_act": "gelu",
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    args = server_args(tmp_path, decode_monokernel=switch)
    assert model_config(args).hf_text_config.decode_monokernel is switch
