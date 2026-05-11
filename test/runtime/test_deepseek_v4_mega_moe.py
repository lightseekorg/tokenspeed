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

import torch

from tokenspeed.runtime.models.deepseek_v4 import DeepseekV4MegaMoEExperts


class TestDeepseekV4MegaMoE(unittest.TestCase):
    def test_weight_loader_places_expert_shards(self):
        experts = DeepseekV4MegaMoEExperts(
            num_experts=4,
            num_local_experts=2,
            top_k=2,
            hidden_size=128,
            intermediate_size=128,
            mapping=None,
            prefix="layers.0.ffn.experts",
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


if __name__ == "__main__":
    unittest.main()
