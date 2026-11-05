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

"""rope_scaling parsing/validation and the get_rope dispatch table."""

from __future__ import annotations

import math

import pytest
import torch
from tokenspeed_kernel.platform import current_platform

import tokenspeed.runtime.layers.rotary_embedding as rotary_module
from tokenspeed.runtime.layers.rotary_embedding import (
    DeepseekScalingRotaryEmbedding,
    DynamicNTKAlphaRotaryEmbedding,
    DynamicNTKScalingRotaryEmbedding,
    LinearScalingRotaryEmbedding,
    Llama3RotaryEmbedding,
    MRotaryEmbedding,
    Phi3LongRoPEScaledRotaryEmbedding,
    RopeScaling,
    RotaryEmbedding,
    YaRNScalingRotaryEmbedding,
    get_rope,
)
from tokenspeed.runtime.utils.env import global_server_args_dict

HEAD_SIZE = 128
ROTARY_DIM = 64
MAX_POSITION = 8192
ORIGINAL_MAX_POSITION = 4096
BASE = 10000


def _get_rope(rope_scaling, **overrides):
    kwargs = dict(
        head_size=HEAD_SIZE,
        rotary_dim=ROTARY_DIM,
        max_position=MAX_POSITION,
        base=BASE,
        is_neox_style=True,
        rope_scaling=rope_scaling,
        dtype=torch.float32,
    )
    kwargs.update(overrides)
    return get_rope(**kwargs)


@pytest.fixture
def fresh_rope_cache(monkeypatch):
    monkeypatch.setattr(rotary_module, "_ROPE_DICT", {})


class TestRopeScalingParsing:
    def test_missing_type_tag_rejected(self):
        with pytest.raises(ValueError, match="Unknown RoPE scaling type"):
            RopeScaling.from_dict({"factor": 1.0})

    def test_legacy_type_key_accepted(self):
        scaling = RopeScaling.from_dict({"type": "linear", "factor": 2.0})
        assert scaling.rope_type == "linear"
        assert scaling.factor == 2.0

    def test_missing_required_keys_are_all_named(self):
        with pytest.raises(ValueError) as exc_info:
            RopeScaling.from_dict({"rope_type": "llama3"})
        message = str(exc_info.value)
        for key in (
            "factor",
            "low_freq_factor",
            "high_freq_factor",
            "original_max_position_embeddings",
        ):
            assert key in message

    def test_dynamic_alpha_needs_no_factor(self):
        scaling = RopeScaling.from_dict({"rope_type": "dynamic", "alpha": 2.0})
        assert scaling.alpha == 2.0

    def test_dynamic_without_alpha_needs_factor(self):
        with pytest.raises(ValueError, match="factor"):
            RopeScaling.from_dict({"rope_type": "dynamic"})

    def test_unrelated_keys_stay_ignored(self):
        scaling = RopeScaling.from_dict(
            {"rope_type": "linear", "factor": 2.0, "some_future_key": 1}
        )
        assert scaling.factor == 2.0


@pytest.mark.usefixtures("fresh_rope_cache")
class TestGetRopeDispatch:
    def test_plain_rotary(self):
        rope = _get_rope(None)
        assert type(rope) is RotaryEmbedding
        assert rope.cos_sin_cache.shape == (MAX_POSITION, ROTARY_DIM)

    def test_llama3(self):
        rope = _get_rope(
            {
                "rope_type": "llama3",
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": ORIGINAL_MAX_POSITION,
            }
        )
        direct = Llama3RotaryEmbedding(
            HEAD_SIZE,
            ROTARY_DIM,
            MAX_POSITION,
            BASE,
            True,
            torch.float32,
            8.0,
            1.0,
            4.0,
            ORIGINAL_MAX_POSITION,
        )
        assert type(rope) is Llama3RotaryEmbedding
        assert torch.equal(rope.cos_sin_cache, direct.cos_sin_cache)

    def test_linear(self):
        rope = _get_rope({"rope_type": "linear", "factor": 2.0})
        direct = LinearScalingRotaryEmbedding(
            HEAD_SIZE, ROTARY_DIM, MAX_POSITION, BASE, True, 2.0, torch.float32
        )
        assert type(rope) is LinearScalingRotaryEmbedding
        assert torch.equal(rope.cos_sin_cache, direct.cos_sin_cache)

    def test_dynamic(self):
        rope = _get_rope({"rope_type": "dynamic", "factor": 4.0})
        direct = DynamicNTKScalingRotaryEmbedding(
            HEAD_SIZE, ROTARY_DIM, MAX_POSITION, BASE, True, 4.0, torch.float32
        )
        assert type(rope) is DynamicNTKScalingRotaryEmbedding
        assert torch.equal(rope.cos_sin_cache, direct.cos_sin_cache)

    def test_dynamic_alpha(self):
        rope = _get_rope({"rope_type": "dynamic", "alpha": 8.0})
        direct = DynamicNTKAlphaRotaryEmbedding(
            HEAD_SIZE, ROTARY_DIM, MAX_POSITION, BASE, True, 8.0, torch.float32
        )
        assert type(rope) is DynamicNTKAlphaRotaryEmbedding
        assert torch.equal(rope.cos_sin_cache, direct.cos_sin_cache)

    def test_yarn(self):
        rope = _get_rope(
            {
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": ORIGINAL_MAX_POSITION,
                "attn_factor": 0.9,
            }
        )
        direct = YaRNScalingRotaryEmbedding(
            HEAD_SIZE,
            ROTARY_DIM,
            ORIGINAL_MAX_POSITION,
            BASE,
            True,
            4.0,
            torch.float32,
            attn_factor=0.9,
        )
        assert type(rope) is YaRNScalingRotaryEmbedding
        assert torch.equal(rope.cos_sin_cache, direct.cos_sin_cache)
        assert rope.mscale == pytest.approx((0.1 * math.log(4.0) + 1.0) * 0.9)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_deepseek_yarn(self, monkeypatch):
        # The cache builds on the platform default device (CUDA here), so
        # build the reference the same way; the point is that mscale and
        # mscale_all_dim reach the constructor. Real DeepSeek checkpoints set
        # mscale_all_dim: 1.0, far from the RopeScaling default of 0, so a
        # dropped kwarg changes the cache numerically.
        monkeypatch.setitem(global_server_args_dict, "yarn_ramp_mask_device", "cpu")
        monkeypatch.setitem(global_server_args_dict, "yarn_frequency_device", "cpu")
        rope = _get_rope(
            {
                "rope_type": "deepseek_yarn",
                "factor": 40.0,
                "original_max_position_embeddings": ORIGINAL_MAX_POSITION,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
            }
        )
        direct = DeepseekScalingRotaryEmbedding(
            HEAD_SIZE,
            ROTARY_DIM,
            ORIGINAL_MAX_POSITION,
            BASE,
            True,
            40.0,
            torch.float32,
            mscale=1.0,
            mscale_all_dim=1.0,
            ramp_device="cpu",
            frequency_device="cpu",
        )
        assert type(rope) is DeepseekScalingRotaryEmbedding
        assert torch.equal(rope.cos_sin_cache, direct.cos_sin_cache)

    def test_longrope(self):
        factors = {
            "rope_type": "longrope",
            "short_factor": [1.0] * (ROTARY_DIM // 2),
            "long_factor": [1.1] * (ROTARY_DIM // 2),
            "original_max_position_embeddings": ORIGINAL_MAX_POSITION,
        }
        rope = _get_rope(factors)
        direct = Phi3LongRoPEScaledRotaryEmbedding(
            HEAD_SIZE,
            ROTARY_DIM,
            MAX_POSITION,
            ORIGINAL_MAX_POSITION,
            BASE,
            True,
            factors["short_factor"],
            factors["long_factor"],
        )
        assert type(rope) is Phi3LongRoPEScaledRotaryEmbedding
        assert torch.equal(rope.cos_sin_cache, direct.cos_sin_cache)

    def test_mrope_section_selects_mrotary(self):
        rope = _get_rope({"rope_type": "default", "mrope_section": [8, 12, 12]})
        assert type(rope) is MRotaryEmbedding
        assert rope.mrope_section == [8, 12, 12]
        assert rope.mrope_interleaved is False

    def test_default_without_mrope_section_is_plain(self):
        rope = _get_rope({"rope_type": "default"})
        assert type(rope) is RotaryEmbedding

    def test_unknown_scaling_type_rejected(self):
        with pytest.raises(ValueError, match="Unknown RoPE scaling type"):
            _get_rope({"rope_type": "su", "factor": 1.0})

    def test_cache_returns_same_object(self):
        scaling = {"rope_type": "linear", "factor": 2.0}
        assert _get_rope(scaling) is _get_rope(scaling)


class TestDeepseekYarnDeviceDefault:
    def test_explicit_device_is_preserved(self):
        rope = DeepseekScalingRotaryEmbedding(
            HEAD_SIZE,
            ROTARY_DIM,
            ORIGINAL_MAX_POSITION,
            BASE,
            False,
            40.0,
            torch.bfloat16,
            device="cpu",
            ramp_device="cpu",
            frequency_device="cpu",
        )
        assert rope.device == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_omitted_device_resolves_the_platform_default(self, monkeypatch):
        monkeypatch.setattr(rotary_module, "_ROPE_DICT", {})
        monkeypatch.setitem(global_server_args_dict, "yarn_ramp_mask_device", "cpu")
        monkeypatch.setitem(global_server_args_dict, "yarn_frequency_device", "cpu")
        rope = _get_rope(
            {
                "rope_type": "deepseek_yarn",
                "factor": 40.0,
                "original_max_position_embeddings": ORIGINAL_MAX_POSITION,
            },
            dtype=torch.bfloat16,
        )
        expected = "npu" if current_platform().is_npu else "cuda"
        assert rope.device == expected
        assert rope.cos_sin_cache.device.type == expected
