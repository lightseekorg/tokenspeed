import sys
from unittest import mock

import pytest
from tokenspeed_kernel.ops.attention import attn_merge_state

from tokenspeed.runtime.models import deepseek_v3
from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3ForCausalLM


class TestDeepseekV3Loader:
    def test_cached_prefix_merge_uses_attention_dispatcher(self):
        assert (deepseek_v3.attn_merge_state) is (attn_merge_state)
        assert not (hasattr(deepseek_v3, "merge_state"))

    def test_missing_checkpoint_scale_params_are_silent(self):
        model = object.__new__(DeepseekV3ForCausalLM)

        with mock.patch("tokenspeed.runtime.models.deepseek_v3.logger") as logger:
            assert (
                model.get_param(
                    {},
                    "model.layers.2.self_attn.k_proj.k_scale",
                )
            ) is None
            assert (
                model.get_param(
                    {},
                    "model.layers.2.self_attn.v_proj.v_scale",
                )
            ) is None

        logger.warning.assert_not_called()

    def test_missing_regular_params_still_warn(self):
        model = object.__new__(DeepseekV3ForCausalLM)

        with mock.patch("tokenspeed.runtime.models.deepseek_v3.logger") as logger:
            assert (
                model.get_param({}, "model.layers.2.self_attn.q_proj.weight")
            ) is None

        logger.warning.assert_called_once()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
