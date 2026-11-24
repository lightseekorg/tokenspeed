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

"""Kimi-K3 configuration, checkpoint loading, and cache-layout contracts."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.configs.kimi_k3_config import (  # noqa: E402
    KimiK3Config,
    KimiK3VisionConfig,
    KimiK31Config,
    KimiLinearConfig,
)
from tokenspeed.runtime.distributed.mapping import Mapping  # noqa: E402
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (  # noqa: E402
    FULL_ATTENTION,
    LINEAR_ATTENTION,
)


class TestKimiK3Config:
    @pytest.mark.parametrize("config_class", [KimiK3Config, KimiK31Config])
    def test_materializes_subconfigs_and_aligns_vision_output(self, config_class):
        config = config_class.from_dict(
            {
                "text_config": {"hidden_size": 4096, "vocab_size": 8192},
                "vision_config": {"vt_hidden_size": 512, "text_hidden_size": 1234},
            }
        )

        assert isinstance(config.text_config, KimiLinearConfig)
        assert isinstance(config.vision_config, KimiK3VisionConfig)
        assert config.hidden_size == config.text_config.hidden_size == 4096
        assert config.vocab_size == config.text_config.vocab_size == 8192
        assert config.vision_config.vt_hidden_size == 512
        assert config.vision_config.text_hidden_size == 4096

    def test_default_layer_partition(self):
        config = KimiLinearConfig()
        kda = set(config.linear_attn_config["kda_layers"])
        full = set(config.linear_attn_config["full_attn_layers"])

        assert len(kda) == 69
        assert len(full) == 24
        assert not kda & full
        assert kda | full == set(range(1, 94))

    def test_custom_layer_partition_uses_zero_based_runtime_indices(self):
        linear = dict(KimiLinearConfig().linear_attn_config)
        linear.update(kda_layers=[2, 4], full_attn_layers=[1, 3])
        config = KimiLinearConfig(num_hidden_layers=4, linear_attn_config=linear)

        assert config.linear_layer_ids == [1, 3]
        assert config.full_attention_layer_ids == [0, 2]
        assert [config.is_kda_layer(i) for i in range(4)] == [False, True, False, True]
        assert config.layers_block_type == [
            "attention",
            "linear_attention",
            "attention",
            "linear_attention",
        ]
        assert config.layer_types == [
            FULL_ATTENTION,
            LINEAR_ATTENTION,
            FULL_ATTENTION,
            LINEAR_ATTENTION,
        ]

    @pytest.mark.parametrize(
        "frequency,expected", [(1, [30, 31, 31]), (2, [15, 15, 16])]
    )
    def test_local_moe_blocks_count_the_pipeline_stage(self, frequency, expected):
        from tokenspeed.runtime.models.kimi_k3 import _k3_local_moe_blocks

        config = KimiLinearConfig(moe_layer_freq=frequency)
        assert _k3_local_moe_blocks(config, Mapping(rank=0, world_size=1)) == sum(
            expected
        )
        assert [
            _k3_local_moe_blocks(config, Mapping(rank=rank, world_size=3, pp_size=3))
            for rank in range(3)
        ] == expected


class TestKimiK3WeightLoading:
    def test_kda_loads_projection_shards_in_checkpoint_order(self):
        from tokenspeed.runtime.models.kimi_k3 import KimiLinearKDA

        config = KimiLinearConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            linear_attn_config={
                "kda_layers": [1],
                "full_attn_layers": [],
                "num_heads": 4,
                "head_dim": 16,
                "short_conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
                "use_full_rank_gate": True,
            },
        )
        layer = KimiLinearKDA(
            config,
            Mapping(rank=0, world_size=1),
            layer_id=0,
            qkv_parallel=None,
            output_parallel=None,
        )
        weight = layer.qkvgb_proj.weight
        shards = []
        for value, (shard_id, rows) in enumerate(
            (("q", 64), ("k", 64), ("v", 64), ("g", 64), ("f_a", 16), ("b", 4)),
            start=1,
        ):
            loaded = torch.full((rows, 64), float(value), dtype=weight.dtype)
            weight.weight_loader(weight, loaded, shard_id)
            shards.append(loaded)

        expected = torch.cat(shards)
        torch.testing.assert_close(
            weight[: expected.shape[0]], expected, rtol=0, atol=0
        )

    def test_text_only_wrapper_filters_vision_checkpoint_weights(self, monkeypatch):
        from tokenspeed.runtime.models import kimi_k3

        class RecordingLanguageModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weights = []

            def load_weights(self, weights):
                self.weights = list(weights)

        language_model = RecordingLanguageModel()
        monkeypatch.setattr(
            kimi_k3, "KimiLinearForCausalLM", lambda *args, **kwargs: language_model
        )
        wrapper = kimi_k3.KimiK3ForConditionalGeneration(
            KimiK3Config(),
            Mapping(rank=0, world_size=1),
            is_multimodal_active=False,
        )
        embedding, head = torch.ones(1), torch.full((1,), 2.0)
        wrapper.load_weights(
            iter(
                (
                    ("language_model.model.embed_tokens.weight", embedding),
                    ("vision_tower.blocks.0.weight", torch.ones(1)),
                    ("mm_projector.weight", torch.ones(1)),
                    ("language_model.lm_head.weight", head),
                )
            )
        )

        assert [name for name, _ in language_model.weights] == [
            "model.embed_tokens.weight",
            "lm_head.weight",
        ]
        assert language_model.weights[0][1] is embedding
        assert language_model.weights[1][1] is head


class TestKimiK3CacheConfig:
    @staticmethod
    def _plan(config):
        from test.runtime.conftest import kimi_tp8_layout

        return kimi_tp8_layout(text_config=config, tp_size=8)[2].bind(64)

    @staticmethod
    def _reduced_config(num_layers):
        linear = dict(KimiLinearConfig().linear_attn_config)
        for name in ("kda_layers", "full_attn_layers"):
            linear[name] = [layer for layer in linear[name] if layer <= num_layers]
        return KimiLinearConfig(num_hidden_layers=num_layers, linear_attn_config=linear)

    def test_reduced_layer_variant_plans(self):
        config = self._reduced_config(24)
        plan = self._plan(config)
        assert len(plan.planes) == 6

    def test_full_size_split_is_enforced(self):
        linear = dict(KimiLinearConfig().linear_attn_config)
        linear["kda_layers"] = linear["kda_layers"][:-3]
        linear["full_attn_layers"] = sorted(
            set(range(1, 94)) - set(linear["kda_layers"])
        )
        config = KimiLinearConfig(linear_attn_config=linear)
        with pytest.raises(ValueError, match="69 KDA and 24 MLA"):
            self._plan(config)

    def test_reduced_layer_states_must_fit_the_mla_planes(self):
        with pytest.raises(ValueError, match="requires 5 target planes.*got 6"):
            self._plan(self._reduced_config(23))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
