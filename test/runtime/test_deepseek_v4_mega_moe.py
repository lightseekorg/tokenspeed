# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

import unittest
from unittest.mock import patch

import torch

from tokenspeed.runtime.models import deepseek_v4 as deepseek_v4_model
from tokenspeed.runtime.models.deepseek_v4 import DeepseekV4MegaMoEExperts


class TestDeepseekV4MegaMoE(unittest.TestCase):
    def test_weight_loader_places_expert_shards(self):
        with patch.object(
            deepseek_v4_model, "dsv4_mega_moe_plan", return_value=object()
        ):
            experts = DeepseekV4MegaMoEExperts(
                num_experts=4,
                num_local_experts=2,
                top_k=2,
                hidden_size=128,
                intermediate_size=128,
                mapping=None,
                prefix="layers.0.ffn.experts",
                swiglu_limit=None,
            )

        w1 = torch.full((128, 64), 1, dtype=torch.uint8)
        w3 = torch.full((128, 64), 3, dtype=torch.uint8)
        w2 = torch.full((128, 64), 2, dtype=torch.uint8)
        s1 = torch.full((128, 4), 11, dtype=torch.uint8)
        s3 = torch.full((128, 4), 13, dtype=torch.uint8)
        s2 = torch.full((128, 4), 12, dtype=torch.uint8)

        experts.weight_loader(experts.w13_weight, w1, "w1", local_expert_id=1)
        experts.weight_loader(experts.w13_weight, w3, "w3", local_expert_id=1)
        experts.weight_loader(experts.w2_weight, w2, "w2", local_expert_id=1)
        experts.weight_loader(experts.w13_weight_scale, s1, "w1", local_expert_id=1)
        experts.weight_loader(experts.w13_weight_scale, s3, "w3", local_expert_id=1)
        experts.weight_loader(experts.w2_weight_scale, s2, "w2", local_expert_id=1)

        torch.testing.assert_close(experts.w13_weight[1, :128], w1)
        torch.testing.assert_close(experts.w13_weight[1, 128:], w3)
        torch.testing.assert_close(experts.w2_weight[1], w2)
        torch.testing.assert_close(experts.w13_weight_scale[1, :128], s1)
        torch.testing.assert_close(experts.w13_weight_scale[1, 128:], s3)
        torch.testing.assert_close(experts.w2_weight_scale[1], s2)

    def test_init_passes_swiglu_limit_to_kernel_plan(self):
        with patch.object(
            deepseek_v4_model, "dsv4_mega_moe_plan", return_value=object()
        ) as plan:
            DeepseekV4MegaMoEExperts(
                num_experts=4,
                num_local_experts=2,
                top_k=2,
                hidden_size=128,
                intermediate_size=128,
                mapping=None,
                prefix="layers.0.ffn.experts",
                swiglu_limit=10.0,
            )
        self.assertEqual(plan.call_args.kwargs["activation_clamp"], 10.0)

    def test_forward_preserves_mega_moe_kernel_defaults(self):
        plan = object()
        with patch.object(deepseek_v4_model, "dsv4_mega_moe_plan", return_value=plan):
            experts = DeepseekV4MegaMoEExperts(
                num_experts=4,
                num_local_experts=2,
                top_k=2,
                hidden_size=128,
                intermediate_size=128,
                mapping=None,
                prefix="layers.0.ffn.experts",
                swiglu_limit=None,
            )
        state = object()
        experts._processed_state = state
        hidden_states = torch.zeros((1, 128), dtype=torch.bfloat16)
        topk_weights = torch.full((1, 2), 0.5, dtype=torch.float32)
        topk_ids = torch.tensor([[0, 1]], dtype=torch.int64)
        expected = torch.ones_like(hidden_states)

        with patch.object(
            deepseek_v4_model, "dsv4_mega_moe_apply", return_value=expected
        ) as apply:
            actual = experts(hidden_states, topk_weights, topk_ids)

        self.assertIs(actual, expected)
        apply.assert_called_once_with(
            plan,
            state,
            hidden_states,
            topk_weights,
            topk_ids,
        )


if __name__ == "__main__":
    unittest.main()
