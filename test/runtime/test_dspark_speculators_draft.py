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

"""Speculators-format DSpark drafts: config conversion and causal block rows.
CPU only; the GPU end-to-end check is the K3 DSpark serving run.
"""

import json
import os
import sys

import pytest
import torch

from tokenspeed.runtime.layers.attention.backends.paged import mha
from tokenspeed.runtime.layers.attention.configs.base import (
    AttnConfig,
    resolve_draft_block_causal,
)
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig
from tokenspeed.runtime.utils import hf_transformers_utils

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")


def _speculators_config(**overrides) -> dict:
    """A trimmed RedHatAI/Kimi-K3-speculator.dspark config.json."""
    config = {
        "architectures": ["DSparkDraftModel"],
        "speculators_model_type": "dspark",
        "aux_hidden_state_layer_ids": [24, 48, 72, 88, 92],
        "block_size": 8,
        "mask_token_id": 163837,
        "markov_rank": 256,
        "markov_head_type": "vanilla",
        "sample_from_anchor": True,
        "sliding_window_non_causal": False,
        "target_hidden_size": 7168,
        "transformer_layer_config": {
            "model_type": "qwen3",
            "hidden_size": 7168,
            "intermediate_size": 14336,
            "num_hidden_layers": 5,
            "num_attention_heads": 96,
            "num_key_value_heads": 16,
            "head_dim": 64,
            "layer_types": ["sliding_attention"] * 5,
            "sliding_window": 2048,
            "use_sliding_window": True,
            "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
            "vocab_size": 163840,
        },
    }
    config.update(overrides)
    return config


def test_speculators_dspark_config_maps_onto_dflash(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_speculators_config()))
    config = hf_transformers_utils.get_config(str(tmp_path), trust_remote_code=False)
    assert config.architectures == ["DSparkDraftModel"]
    assert config.num_attention_heads == 96 and config.num_key_value_heads == 16
    assert config.sliding_window == 2048
    assert config.block_size == 8 and config.markov_rank == 256
    # Hidden-states indices name the tapped layer plus one.
    assert config.dflash_config == {
        "mask_token_id": 163837,
        "target_layer_ids": [23, 47, 71, 87, 91],
        "causal": True,
    }
    assert resolve_draft_block_causal(config)


@pytest.mark.parametrize(
    "overrides",
    [
        {"sample_from_anchor": False},
        {
            "transformer_layer_config": {
                **_speculators_config()["transformer_layer_config"],
                "layer_types": ["sliding_attention", "full_attention"] * 2
                + ["full_attention"],
            }
        },
    ],
)
def test_speculators_dspark_config_rejects_what_the_drafter_cannot_serve(overrides):
    with pytest.raises(ValueError):
        hf_transformers_utils._convert_speculators_dspark_config(
            _speculators_config(**overrides)
        )


def _block_draft_leaf(*, causal: bool) -> mha.MHAAttnBackend:
    spec = MHAConfig(
        backend_name="mha",
        num_attention_heads=12,
        num_kv_heads=2,
        head_dim=64,
        attn_tp_size=1,
    )
    config = AttnConfig(
        device="cpu",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        prefix_granularity=16,
        context_len=1024,
        max_bs=4,
        speculative_num_draft_tokens=4,
        is_draft=True,
        draft_block_decode=True,
        draft_block_causal=causal,
        components=(spec,),
    )
    leaf = mha.MHAAttnBackend(config, spec, kernel_page_size=16)
    leaf.init_cuda_graph_state(4)
    return leaf


@pytest.mark.parametrize(
    ("causal", "expected"),
    [
        # Every row of a non-causal block reads up to the block end.
        (False, [10, 10, 10, 10, 7, 7, 7, 7]),
        # Row i of a causal block reads up to its own position.
        (True, [7, 8, 9, 10, 4, 5, 6, 7]),
    ],
)
def test_block_rows_seq_lens(causal, expected):
    leaf = _block_draft_leaf(causal=causal)
    leaf.fill_block_decode_seq_lens(2, torch.tensor([10, 7], dtype=torch.int32))
    assert leaf.seq_lens_buf[:8].tolist() == expected
