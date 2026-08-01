import unittest
from unittest import mock

from tokenspeed_kernel.ops.attention import attn_merge_state

from tokenspeed.runtime.models import deepseek_v3
from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3ForCausalLM


class TestDeepseekV3Loader(unittest.TestCase):
    def test_cached_prefix_merge_uses_attention_dispatcher(self):
        self.assertIs(deepseek_v3.attn_merge_state, attn_merge_state)
        self.assertFalse(hasattr(deepseek_v3, "merge_state"))

    def test_missing_checkpoint_scale_params_are_silent(self):
        model = object.__new__(DeepseekV3ForCausalLM)

        with mock.patch("tokenspeed.runtime.models.deepseek_v3.logger") as logger:
            self.assertIsNone(
                model.get_param(
                    {},
                    "model.layers.2.self_attn.k_proj.k_scale",
                )
            )
            self.assertIsNone(
                model.get_param(
                    {},
                    "model.layers.2.self_attn.v_proj.v_scale",
                )
            )

        logger.warning.assert_not_called()

    def test_missing_regular_params_still_warn(self):
        model = object.__new__(DeepseekV3ForCausalLM)

        with mock.patch("tokenspeed.runtime.models.deepseek_v3.logger") as logger:
            self.assertIsNone(
                model.get_param({}, "model.layers.2.self_attn.q_proj.weight")
            )

        logger.warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
