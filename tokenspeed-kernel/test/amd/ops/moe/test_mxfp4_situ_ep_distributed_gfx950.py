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

"""RCCL smoke test for Kimi 3 MXFP4 expert parallelism.

Normal one-GPU pytest runs skip this file.  Exercise it with, for example:

``torchrun --standalone --nproc-per-node=2 -m pytest -q <this file>``.
"""

from __future__ import annotations

import os

import pytest
import tokenspeed_kernel
import torch
import torch.distributed as dist
from kimi3_reference import (
    mxfp4_moe_reference,
)
from utils import make_mxfp4_moe_weights


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


@pytest.mark.skipif(
    _world_size() not in {2, 4, 8},
    reason="launch with torchrun world size 2, 4, or 8",
)
def test_distributed_ep_partial_sum_matches_global_reference() -> None:
    world_size = _world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    assert dist.get_world_size() == world_size

    generator = torch.Generator(device="cuda").manual_seed(20260718)
    num_experts = world_size
    num_tokens = 8
    latent_size = intermediate_size = 512
    top_k = world_size
    raw = make_mxfp4_moe_weights(
        num_experts,
        latent_size,
        intermediate_size,
        generator,
    )
    w13, s13 = raw["w13_weight"], raw["w13_scale"]
    w2, s2 = raw["w2_weight"], raw["w2_scale"]
    hidden_states = (
        torch.randn(
            (num_tokens, latent_size),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.1
    )
    topk_ids = torch.arange(num_experts, dtype=torch.int32, device="cuda").expand(
        num_tokens, -1
    )
    topk_weights = torch.full(
        (num_tokens, top_k),
        1.0 / top_k,
        dtype=torch.float32,
        device="cuda",
    )

    module = torch.nn.Module()
    module.w13_weight = torch.nn.Parameter(w13[rank : rank + 1].clone(), False)
    module.w13_weight_scale = torch.nn.Parameter(s13[rank : rank + 1].clone(), False)
    module.w2_weight = torch.nn.Parameter(w2[rank : rank + 1].clone(), False)
    module.w2_weight_scale = torch.nn.Parameter(s2[rank : rank + 1].clone(), False)
    module.top_k = top_k
    module.num_experts = num_experts
    module.num_local_experts = 1
    module.ep_rank = rank
    module.ep_size = world_size
    module.activation_situ_beta = 4.0
    module.activation_situ_linear_beta = 25.0

    plan = tokenspeed_kernel.moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation="situ",
        routing_mode="precomputed_topk",
        ep_size=world_size,
        ispp=intermediate_size,
        internal_activation_dtype="input",
        solution="gluon",
    )
    tokenspeed_kernel.moe_process_weights(plan, module)
    partial = tokenspeed_kernel.moe_apply(
        plan,
        hidden_states,
        module,
        torch.zeros((num_tokens, num_experts), dtype=torch.float32, device="cuda"),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
    dist.all_reduce(partial, op=dist.ReduceOp.SUM)

    expected = mxfp4_moe_reference(
        hidden_states,
        w13,
        s13,
        w2,
        s2,
        topk_ids,
        topk_weights,
        activation_dtype=torch.bfloat16,
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )
    torch.testing.assert_close(partial, expected, atol=3e-2, rtol=3e-2)
    dist.barrier()
    dist.destroy_process_group()
