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

"""Replicated DSpark Markov heads load and score distinct token domains."""

import os
import sys
from test.runtime.conftest import model_init_config

import pytest
import torch
from transformers import PretrainedConfig

from tokenspeed.runtime.configs.kimi_k3_dspark_config import KimiK3DSparkConfig
from tokenspeed.runtime.distributed.mapping import Mapping
from tokenspeed.runtime.execution.drafter.base import BaseDrafter
from tokenspeed.runtime.execution.drafter.dspark import DSpark
from tokenspeed.runtime.models.dspark import DSparkDraftModel
from tokenspeed.runtime.models.kimi_k3_dspark import K3DSparkModel
from tokenspeed.runtime.models.minimax_m3_dspark import DSparkMiniMaxDraftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=15, suite="runtime-1gpu")


@pytest.mark.parametrize("output_size", [64, 128])
@pytest.mark.parametrize("family", ["generic", "minimax", "k3"])
def test_replicated_markov_checkpoint_and_bias_use_separate_domains(
    family, output_size
):
    if family == "k3":
        config = KimiK3DSparkConfig(
            vocab_size=128,
            output_vocab_size=output_size,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            q_lora_rank=8,
            kv_lora_rank=8,
            qk_nope_head_dim=16,
            qk_rope_head_dim=16,
            v_head_dim=16,
            max_position_embeddings=64,
            num_target_layers=1,
            target_hidden_size=32,
            target_num_hidden_layers=2,
            target_layer_ids=[0],
            mask_token_id=127,
            markov_rank=4,
            enable_confidence_head=False,
        )
        model_class = K3DSparkModel
    else:
        config = PretrainedConfig(
            vocab_size=128,
            output_vocab_size=output_size,
            hidden_size=8,
            num_hidden_layers=0,
            rms_norm_eps=1e-5,
            dspark_config={"markov_rank": 4},
            dflash_config={"target_layer_ids": [0]},
        )
        model_class = (
            DSparkDraftModel if family == "generic" else DSparkMiniMaxDraftModel
        )
    model = model_class(
        model_init_config(config, is_draft=True),
        Mapping(rank=0, world_size=1, attn_tp_size=1),
    )
    embedding = (torch.arange(128 * 4).reshape(128, 4) % 11).float()
    projection = (torch.arange(output_size * 4).reshape(output_size, 4) % 7).float()
    checkpoint = {
        name: torch.zeros_like(param) for name, param in model.named_parameters()
    }
    checkpoint["markov_head.markov_w1.weight"] = embedding
    checkpoint["markov_head.markov_w2.weight"] = projection
    model.load_weights(checkpoint.items())

    drafter = DSpark.__new__(DSpark)
    BaseDrafter.__init__(drafter, spec_num_tokens=4)
    drafter.markov_head = model.markov_head
    # Rows 64 and 127 are valid inputs even when they are not output tokens;
    # only graph-padding IDs outside the input domain clamp to its boundaries.
    previous = torch.tensor([0, 64, 127, -1, 200])
    expected = embedding[torch.tensor([0, 64, 127, 0, 127])] @ projection.T
    bias = drafter._make_step_bias_fn(previous)
    torch.testing.assert_close(bias(0, output_size), expected)
    torch.testing.assert_close(bias(5, 17), expected[:, 5:22])
    torch.testing.assert_close(
        bias(output_size - 2, 5),
        torch.cat((expected[:, -2:], torch.zeros(5, 3)), dim=-1),
    )
    torch.testing.assert_close(
        model.markov_head.compute_step_bias(torch.tensor([64, 127])), expected[1:3]
    )
