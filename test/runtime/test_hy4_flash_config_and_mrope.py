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

"""Unit tests for Hy4-flash config normalization, ModelConfig hooks, and 3D M-RoPE."""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.configs.hy4_flash_config import (  # noqa: E402
    DEEPSEEK_SPARSE_ATTENTION,
    INDEXED_ATTENTION,
    SLIDING_ATTENTION,
    HYV4FlashConfig,
    HYV4FlashDFlyConfig,
    HYV4FlashTextConfig,
    HYV4FlashVisionConfig,
)
from tokenspeed.runtime.engine.scheduler_utils import (  # noqa: E402
    engram_context_len,
)
from tokenspeed.runtime.execution.drafter import (  # noqa: E402
    get_drafter_impl,
    registered_drafter_algorithms,
)
from tokenspeed.runtime.execution.drafter.dfly import DFly  # noqa: E402
from tokenspeed.runtime.multimodal.inputs import (  # noqa: E402
    Modality,
    MultimodalDataItem,
)
from tokenspeed.runtime.multimodal.mrope import (  # noqa: E402
    compute_mrope_positions,
)
from tokenspeed.runtime.utils.spec_block_geometry import (  # noqa: E402
    BLOCK_SPEC_ALGORITHMS,
    read_checkpoint_block_size,
    validate_block_widths,
)


def test_text_config_normalizes_checkpoint_aliases_and_per_layer_rope():
    cfg = HYV4FlashTextConfig(
        num_hidden_layers=6,
        num_attention_heads=32,
        swa_num_attention_heads=64,
        layer_types=[
            DEEPSEEK_SPARSE_ATTENTION,
            DEEPSEEK_SPARSE_ATTENTION,
            SLIDING_ATTENTION,
            SLIDING_ATTENTION,
            SLIDING_ATTENTION,
            DEEPSEEK_SPARSE_ATTENTION,
        ],
        compressed_indexer_compress_ratio=4,
        compressed_indexer_num_attention_heads=16,
        compressed_indexer_head_dim=128,
        compressed_indexer_candidate_topk=2048,
        rope_parameters={
            DEEPSEEK_SPARSE_ATTENTION: {
                "rope_type": "default",
                "rope_theta": 10000000.0,
                "mrope_section": [12, 10, 10],
                "mrope_interleaved": True,
            },
            SLIDING_ATTENTION: {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "mrope_section": [12, 10, 10],
                "mrope_interleaved": True,
            },
        },
    )
    assert cfg.layer_types == [
        INDEXED_ATTENTION,
        INDEXED_ATTENTION,
        SLIDING_ATTENTION,
        SLIDING_ATTENTION,
        SLIDING_ATTENTION,
        INDEXED_ATTENTION,
    ]
    assert cfg.layer_num_attention_heads(0) == 32
    assert cfg.layer_num_attention_heads(2) == 64
    assert cfg.index_kpool == 4
    assert cfg.index_n_heads == 16
    assert cfg.index_head_dim == 128
    assert cfg.index_topk == 2048
    assert cfg.rope_parameters[INDEXED_ATTENTION]["rope_theta"] == 10000000.0
    assert cfg.rope_parameters[SLIDING_ATTENTION]["rope_theta"] == 10000.0
    assert cfg.rope_parameters[INDEXED_ATTENTION]["mrope_section"] == [12, 10, 10]


def test_full_config_and_dfly_from_target_preserve_quant_and_dfly_schedule():
    quant_cfg = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": ["language_model.model.lm_head"],
    }
    text = HYV4FlashTextConfig(
        num_hidden_layers=42,
        quantization_config=quant_cfg,
    )
    vision = HYV4FlashVisionConfig(
        hidden_size=1152,
        out_hidden_size=3072,
    )
    full = HYV4FlashConfig(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        dfly_num_hidden_layers=5,
        dfly_block_size=8,
        dfly_mask_token_id=120009,
        dfly_target_kv_layers=[25, 29, 33, 37, 41],
        dfly_target_index_layer=[25, 29, 33, 37, 41],
        dfly_target_index_topk=2048,
        dfly_mtp_fusion="slot0",
        dfly_enable_hidden_correction=True,
        dfly_enable_confidence_head=True,
    )
    assert isinstance(full.text_config, HYV4FlashTextConfig)
    assert isinstance(full.vision_config, HYV4FlashVisionConfig)
    assert full.model_type == "hy_v4_flash"
    assert full.text_config.model_type == "hy_v4_flash_text"
    assert full.vision_config.model_type == "hy_v4_vl_vision"

    dfly = HYV4FlashDFlyConfig.from_target(full)
    assert dfly.architectures == ["HYV4FlashForCausalLMDFly"]
    assert dfly.num_hidden_layers == 5
    assert dfly.target_kv_layers == [25, 29, 33, 37, 41]
    assert dfly.target_index_layer == [25, 29, 33, 37, 41]
    assert dfly.target_index_topk == 2048
    assert dfly.rope_theta == 10000000.0
    assert dfly.mtp_fusion == "slot0"
    assert dfly.quantization_config["quant_method"] == "fp8"
    assert read_checkpoint_block_size(dfly) == 8

    # Block geometry validation for DFLY (sample_from_anchor=False -> steps = verify - 1)
    assert "DFLY" in BLOCK_SPEC_ALGORITHMS
    validate_block_widths("DFLY", 8, 7, 8, allow_shorter=True)
    validate_block_widths("DFLY", 8, 5, 6, allow_shorter=True)
    with pytest.raises(ValueError):
        validate_block_widths("DFLY", 8, 8, 9, allow_shorter=True)


def test_published_checkpoint_model_type_does_not_override_text_subconfig():
    text_dict = {
        "model_type": "hy_v4_flash",
        "num_hidden_layers": 4,
        "engram_layer_ids": [2],
        "engram_max_ngram_size": 3,
        "engram_n_head_per_ngram": 2,
        "engram_layer_head_vocab_sizes": {"2": [[17, 19], [23, 29]]},
    }
    vision_dict = {"model_type": "hy_v4_flash"}
    full = HYV4FlashConfig(text_config=text_dict, vision_config=vision_dict)
    assert full.model_type == "hy_v4_flash"
    assert full.text_config.model_type == "hy_v4_flash_text"
    assert full.vision_config.model_type == "hy_v4_vl_vision"
    assert engram_context_len(full.text_config) == 0


def test_dfly_registered_in_drafter_registry():
    assert "DFLY" in registered_drafter_algorithms()
    assert get_drafter_impl("DFLY", None) is DFly


def test_hy4_flash_3d_mrope_positions_image_and_text():
    # Sequence: 2 text tokens, image_start (120118), 2x(2+1) merged image tokens
    # (120120) with row-end newline markers (120121), image_end (120119), 2 text tokens.
    img_start, img_end, img_tok, img_nl = 120118, 120119, 120120, 120121
    cfg = HYV4FlashConfig(
        architectures=["HYV4FlashForConditionalGeneration"],
        image_start_token_id=img_start,
        image_end_token_id=img_end,
        image_token_id=img_tok,
        image_newline_token_id=img_nl,
        spatial_merge_size=2,
    )
    input_ids = [
        10,
        11,
        img_start,
        img_tok,
        img_tok,
        img_nl,
        img_tok,
        img_tok,
        img_nl,
        img_end,
        12,
        13,
    ]
    mm_item = MultimodalDataItem(
        modality=Modality.IMAGE,
        offsets=[(3, 8)],
        feature=torch.zeros((16, 1)),
        model_specific_data={"image_grid_thw": [[1, 4, 4]]},
    )
    positions, delta = compute_mrope_positions(cfg, input_ids, [mm_item])
    assert positions is not None
    assert positions.shape == (3, len(input_ids))
    # Prefix text tokens have identical 3D coordinates (0,0,0), (1,1,1)
    assert positions[:, 0].tolist() == [0, 0, 0]
    assert positions[:, 1].tolist() == [1, 1, 1]
    # Trailing text tokens advance strictly past the image grid along all 3 axes equally
    assert positions[0, -1].item() == positions[1, -1].item() == positions[2, -1].item()
    assert positions[0, -1].item() - positions[0, -2].item() == 1
    assert delta is not None
